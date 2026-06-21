"""
Unit tests for engine.schemas.UserProgress.mark_stage_complete — CORR-001.

Bug: mark_stage_complete("harden") appends a synthetic f"module_{index}"
string to completed_modules instead of the caller-supplied actual module ID.
This breaks every downstream caller that queries `module.id in
progress.completed_modules` (engine/main.py:2196 curriculum-list,
engine/main.py:2335 progress-reset).

Catalog: tests/engine/schemas.bug-catalog.md
Audit:   https://github.com/ImmortalDemonGod/mastery-engine/blob/
         7f6610a902befcb84fc47e5c82a161e3d3184ce4/audit/02-static-audit.md#L17
"""

import pytest

from engine.schemas import UserProgress


class TestMarkStageCompleteHardenRecordsRealId:
    """CORR-001: harden branch must store the caller-supplied module id."""

    def test_harden_records_caller_supplied_module_id_not_synthetic_index(self):
        """CORR-001 BUG-A: mark_stage_complete('harden', module_id='softmax') must add
        'softmax' to completed_modules — guards against synthetic 'module_0' alias
        making curriculum-list completion check at main.py:2196 always miss."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("harden", module_id="softmax")
        assert "softmax" in progress.completed_modules

    def test_harden_does_not_append_synthetic_index(self):
        """CORR-001 BUG-A (contract pin): after harden, the synthetic 'module_0' string
        must NOT appear in completed_modules — its presence makes module.id lookup
        always return False."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("harden", module_id="softmax")
        assert "module_0" not in progress.completed_modules

    def test_curriculum_list_lookup_matches_after_mark_complete(self):
        """CORR-001 BUG-A (integration contract): id stored by mark_stage_complete must
        satisfy the 'module.id in progress.completed_modules' check used by the
        curriculum-list command at engine/main.py:2196."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("harden", module_id="softmax")
        # Simulate the main.py:2196 read path
        checked_module_id = "softmax"
        assert checked_module_id in progress.completed_modules

    def test_harden_subsequent_module_records_its_own_id(self):
        """CORR-001 BUG-D: each module's real id must be stored independent of its
        position index — 'rmsnorm' at index 1 must not collide with 'softmax' at
        index 0; both must appear in completed_modules."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("harden", module_id="softmax")
        progress.mark_stage_complete("harden", module_id="rmsnorm")
        assert "softmax" in progress.completed_modules
        assert "rmsnorm" in progress.completed_modules

    def test_harden_idempotent_real_module_id(self):
        """CORR-001 BUG-C: calling mark_stage_complete('harden') twice with the same
        module_id must not duplicate the entry — deduplication guard must operate on
        the real id, not the synthetic one."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("harden", module_id="softmax")
        # Simulate a retry: same module, same id
        progress.current_module_index = 0
        progress.mark_stage_complete("harden", module_id="softmax")
        assert progress.completed_modules.count("softmax") == 1


class TestMarkStageCompleteHardenValidation:
    """CORR-001 BUG-B: None module_id in harden branch must be rejected explicitly."""

    def test_harden_raises_if_module_id_is_none(self):
        """CORR-001 BUG-B: passing module_id=None to harden must raise ValueError or
        TypeError — silent acceptance would leave completed_modules with a None
        entry that never matches any module.id string."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        with pytest.raises((ValueError, TypeError)):
            progress.mark_stage_complete("harden", module_id=None)


class TestMarkStageCompleteNonHardenRegressions:
    """CORR-001 BUG-E: build and justify transitions must still work without module_id
    and must NOT modify completed_modules."""

    def test_build_advances_to_justify_without_module_id(self):
        """Non-regression: mark_stage_complete('build') must advance stage to 'justify'
        and must not require module_id."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("build")
        assert progress.current_stage == "justify"

    def test_justify_advances_to_harden_without_module_id(self):
        """Non-regression: mark_stage_complete('justify') must advance stage to 'harden'
        and must not require module_id."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0,
                                current_stage="justify")
        progress.mark_stage_complete("justify")
        assert progress.current_stage == "harden"

    def test_build_does_not_record_completed_module(self):
        """CORR-001 BUG-E: build transition must not touch completed_modules."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0)
        progress.mark_stage_complete("build")
        assert progress.completed_modules == []

    def test_justify_does_not_record_completed_module(self):
        """CORR-001 BUG-E: justify transition must not touch completed_modules."""
        progress = UserProgress(curriculum_id="cs336_a1", current_module_index=0,
                                current_stage="justify")
        progress.mark_stage_complete("justify")
        assert progress.completed_modules == []
