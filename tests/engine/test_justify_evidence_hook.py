"""Focused test: the Justify grade point writes a cognitive-evidence record."""

import json
from unittest.mock import MagicMock

from engine import main as m
from engine.schemas import (
    JustifyQuestion,
    LLMEvaluationResponse,
    UserProgress,
    ModuleMetadata,
    CurriculumManifest,
)


def test_submit_justify_writes_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTERY_EVIDENCE_PATH", str(tmp_path / "ev.jsonl"))

    question = JustifyQuestion(
        id="q1",
        question="Why subtract the max in softmax?",
        model_answer="prevents overflow via shift invariance",
        failure_modes=[],
        required_concepts=["overflow"],
    )
    module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
    manifest = CurriculumManifest(curriculum_name="cs336_a1", author="x", version="1", modules=[module])
    progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0, current_stage="justify")
    state_mgr = MagicMock()
    curr_mgr = MagicMock()

    # JustifyRunner: one question, fast filter passes.
    jr = MagicMock()
    jr.load_questions.return_value = [question]
    jr.check_fast_filter.return_value = (False, None)
    monkeypatch.setattr(m, "JustifyRunner", MagicMock(return_value=jr))

    # Grader: live (not mock), returns a correct evaluation.
    svc = MagicMock()
    svc.use_mock = False
    svc.evaluate_justification.return_value = LLMEvaluationResponse(
        is_correct=True, feedback="Correct — overflow prevention via shift invariance."
    )
    monkeypatch.setattr(m, "LLMService", MagicMock(return_value=svc))

    # Editor: write the answer into the temp file the stage opens.
    def fake_editor(args, *a, **k):
        path = args[1]
        with open(path, "w") as f:
            f.write("# Justify Question\n\n# Your Answer\nbecause it prevents overflow\n")
        return MagicMock(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", fake_editor)

    ok = m._submit_justify_stage(state_mgr, curr_mgr, progress, manifest)
    assert ok is True

    lines = (tmp_path / "ev.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["curriculum_id"] == "cs336_a1"
    assert rec["module_id"] == "softmax"
    assert rec["question_id"] == "q1"
    assert rec["is_correct"] is True
    assert rec["answer"].strip() == "because it prevents overflow"
    assert rec["outcome"] == "graded"
    assert rec["ts"]


def test_submit_justify_records_fast_filter_reject(tmp_path, monkeypatch):
    """A fast-filter rejection is recorded (outcome=fast_filter_reject) and skips the grader."""
    monkeypatch.setenv("MASTERY_EVIDENCE_PATH", str(tmp_path / "ev.jsonl"))

    question = JustifyQuestion(
        id="q1", question="why?", model_answer="m", failure_modes=[], required_concepts=["c"]
    )
    module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
    manifest = CurriculumManifest(curriculum_name="cs336_a1", author="x", version="1", modules=[module])
    progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0, current_stage="justify")
    state_mgr = MagicMock()
    curr_mgr = MagicMock()

    jr = MagicMock()
    jr.load_questions.return_value = [question]
    jr.check_fast_filter.return_value = (True, "too vague")  # fast filter rejects
    monkeypatch.setattr(m, "JustifyRunner", MagicMock(return_value=jr))

    svc = MagicMock()
    svc.use_mock = False
    monkeypatch.setattr(m, "LLMService", MagicMock(return_value=svc))

    def fake_editor(args, *a, **k):
        with open(args[1], "w") as f:
            f.write("# Justify Question\n\n# Your Answer\nvague stuff\n")
        return MagicMock(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", fake_editor)

    ok = m._submit_justify_stage(state_mgr, curr_mgr, progress, manifest)
    assert ok is False
    svc.evaluate_justification.assert_not_called()  # grader skipped by the fast filter

    rec = json.loads((tmp_path / "ev.jsonl").read_text().strip())
    assert rec["outcome"] == "fast_filter_reject"
    assert rec["is_correct"] is False
    assert rec["answer"].strip() == "vague stuff"
