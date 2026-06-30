"""Shared fixtures for the end-to-end suite.

These E2E tests drive the real engine (in-process via CliRunner and out-of-process
via subprocess). Two cross-cutting concerns are handled here for every E2E test:

1. HOME isolation — the engine persists progress to `~/.mastery_progress.json` and
   the cognitive-evidence ledger to `~/.mastery_evidence.jsonl` (both via Path.home()).
   Without isolation, running the suite would read/overwrite the developer's REAL
   progress file. We redirect HOME (honored by Path.home() on POSIX) to a per-test
   temp dir so the suite is non-destructive by construction.

2. Wide console — the engine renders Rich panels whose width follows the COLUMNS env
   var when stdout is captured (not a TTY). At the default 80 cols, long temp paths
   wrap across panel borders and break substring assertions on stdout. We force a wide
   console so paths/messages stay on one line.
"""
import pytest


@pytest.fixture(autouse=True)
def isolated_home_and_wide_console(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "200")
    yield
