"""Unit tests for engine.services.evidence_sink.EvidenceSink."""

import json

from engine.services.evidence_sink import EvidenceSink


def _sink(tmp_path):
    return EvidenceSink(path=str(tmp_path / "evidence.jsonl"))


def test_record_appends_jsonl(tmp_path):
    sink = _sink(tmp_path)
    rec = sink.record(
        curriculum_id="cs336_a1",
        module_id="softmax",
        question_id="q1",
        answer="because it prevents overflow",
        is_correct=True,
        feedback="correct",
    )
    lines = (tmp_path / "evidence.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk == rec
    assert on_disk["curriculum_id"] == "cs336_a1"
    assert on_disk["module_id"] == "softmax"
    assert on_disk["question_id"] == "q1"
    assert on_disk["is_correct"] is True
    assert on_disk["stage"] == "justify"
    assert on_disk["outcome"] == "graded"
    assert "ts" in on_disk and on_disk["ts"]


def test_record_is_append_only(tmp_path):
    sink = _sink(tmp_path)
    sink.record(curriculum_id="c", module_id="m1", question_id="q", answer="a", is_correct=True, feedback="ok")
    sink.record(curriculum_id="c", module_id="m2", question_id="q", answer="a", is_correct=False, feedback="hint")
    lines = (tmp_path / "evidence.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["module_id"] == "m1"
    assert json.loads(lines[1])["module_id"] == "m2"


def test_count_graded_only(tmp_path):
    sink = _sink(tmp_path)
    sink.record(curriculum_id="c", module_id="m", question_id="q", answer="a", is_correct=True, feedback="ok")  # graded
    sink.record(
        curriculum_id="c",
        module_id="m",
        question_id="q",
        answer="a",
        is_correct=True,
        feedback="ok",
        outcome="mock_autopass",
    )
    sink.record(
        curriculum_id="c",
        module_id="m",
        question_id="q",
        answer="",
        is_correct=False,
        feedback="too vague",
        outcome="fast_filter_reject",
    )
    assert sink.count() == 1  # graded-only by default
    assert sink.count(only_graded=False) == 3


def test_count_missing_file_is_zero(tmp_path):
    assert _sink(tmp_path).count() == 0


def test_env_override(tmp_path, monkeypatch):
    target = tmp_path / "via_env.jsonl"
    monkeypatch.setenv("MASTERY_EVIDENCE_PATH", str(target))
    sink = EvidenceSink()
    sink.record(curriculum_id="c", module_id="m", question_id="q", answer="a", is_correct=True, feedback="ok")
    assert target.exists()
