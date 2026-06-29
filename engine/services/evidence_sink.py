"""
Cognitive-evidence sink for the Mastery Engine.

The Justify stage produces graded comprehension (is_correct + feedback), but the
engine historically persisted only a completion bit (current_stage / completed_modules).
This module records every graded justification as an append-only, timestamped event so
the comprehension is COUNTABLE (the Flywheel currency), the way SVP verdicts are.

Storage: append-only JSONL at ~/.mastery_evidence.jsonl (override via MASTERY_EVIDENCE_PATH).

MIGRATION NOTE (Cycle-2 decision): JSONL is the interim store to avoid touching the live
cultivation-os flash.db while this is proven. Once stable, migrate to the flashcore/console
evidence path so graded-comprehension lands in the SAME ledger as SVP verdicts
(one currency, one ledger). See references/2026-06-26_cycle2_consolidation.md §4.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EVIDENCE_PATH_ENV = "MASTERY_EVIDENCE_PATH"
DEFAULT_EVIDENCE_PATH = "~/.mastery_evidence.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceSink:
    """Append-only writer for graded-justification evidence records."""

    def __init__(self, path: Optional[str] = None):
        raw = path or os.getenv(EVIDENCE_PATH_ENV) or DEFAULT_EVIDENCE_PATH
        self.path = Path(os.path.expanduser(raw))

    def record(
        self,
        *,
        curriculum_id: str,
        module_id: str,
        question_id: str,
        answer: str,
        is_correct: bool,
        feedback: str,
        stage: str = "justify",
        outcome: str = "graded",
    ) -> dict:
        """Append one evidence event and return the record written.

        outcome distinguishes a real LLM grade ("graded") from non-graded transitions
        (e.g. "fast_filter_reject", "mock_autopass") so the ledger stays honest about
        what is and isn't genuine cognitive evidence.
        """
        record = {
            "ts": _utc_now_iso(),
            "stage": stage,
            "outcome": outcome,
            "curriculum_id": curriculum_id,
            "module_id": module_id,
            "question_id": question_id,
            "is_correct": is_correct,
            "answer": answer,
            "feedback": feedback,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def count(self, *, only_graded: bool = True) -> int:
        """Count recorded events (graded-only by default) — the cognitive-evidence tally."""
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if only_graded and rec.get("outcome") != "graded":
                    continue
                n += 1
        return n
