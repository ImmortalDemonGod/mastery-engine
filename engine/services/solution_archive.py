"""Best-effort archival of the learner's own implementations to a side branch.

On module completion the engine snapshots the just-passed source file to a
long-lived, never-merged branch (default ``my-solutions``) so the learner keeps
a durable, versioned record of their work — without that record living in the
stub-validated student tree (where it would trip the pre-commit hook) or being
clobbered by a mode-switch.

Mechanism: low-level git plumbing (``hash-object`` → ``commit-tree`` →
``update-ref``). This never checks out, never touches the working tree or the
currently checked-out branch, and never runs hooks — so it cannot disrupt an
in-progress session. The whole operation is best-effort: any failure is logged
and swallowed so a git hiccup can never block learning.

Opt-in by design: it only acts if the solutions branch already exists. Creating
``my-solutions`` arms it; deleting the branch disarms it. No config surface.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SOLUTIONS_BRANCH = "my-solutions"


def _git(repo: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        input=stdin,
    )


def _branch_exists(repo: Path, branch: str) -> bool:
    return _git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0


def archive_solution(
    repo_root: Path,
    source_content: str,
    repo_relpath: str,
    commit_message: str,
    branch: str = SOLUTIONS_BRANCH,
) -> bool:
    """Commit ``source_content`` at ``repo_relpath`` onto ``branch`` via plumbing.

    Returns ``True`` when a new commit was written, ``False`` when the operation
    was skipped (branch absent, content unchanged, or any error). Never raises.
    """
    try:
        repo = Path(repo_root)

        # Opt-in: only archive if the learner has created the side branch.
        if not _branch_exists(repo, branch):
            return False

        parent = _git(repo, "rev-parse", branch).stdout.strip()
        parent_tree = _git(repo, "rev-parse", f"{branch}^{{tree}}").stdout.strip()
        if not parent or not parent_tree:
            logger.warning("archive: could not resolve branch '%s' tip/tree", branch)
            return False

        # 1. Store the file content as a loose blob.
        h = _git(repo, "hash-object", "-w", "--stdin", stdin=source_content)
        if h.returncode != 0:
            logger.warning("archive: hash-object failed: %s", h.stderr.strip())
            return False
        blob = h.stdout.strip()

        # 2. Build a tree = branch tip's tree with this one path updated, using a
        #    throwaway index so the real index/working tree are never touched.
        with tempfile.TemporaryDirectory() as td:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(td) / "index")}

            def _plumb(*args: str) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["git", *args], cwd=str(repo), env=env,
                    capture_output=True, text=True,
                )

            r = _plumb("read-tree", branch)
            if r.returncode != 0:
                logger.warning("archive: read-tree failed: %s", r.stderr.strip())
                return False
            r = _plumb("update-index", "--add", "--cacheinfo", f"100644,{blob},{repo_relpath}")
            if r.returncode != 0:
                logger.warning("archive: update-index failed: %s", r.stderr.strip())
                return False
            r = _plumb("write-tree")
            if r.returncode != 0:
                logger.warning("archive: write-tree failed: %s", r.stderr.strip())
                return False
            tree = r.stdout.strip()

        # 3. Nothing to record if the tree is identical to the branch tip.
        if tree == parent_tree:
            logger.info("archive: %s unchanged on '%s'; skipping", repo_relpath, branch)
            return False

        # 4. Commit the tree onto the branch (no checkout, no hooks) and move the ref.
        c = _git(repo, "commit-tree", tree, "-p", parent, "-m", commit_message)
        if c.returncode != 0:
            logger.warning("archive: commit-tree failed: %s", c.stderr.strip())
            return False
        commit = c.stdout.strip()

        u = _git(repo, "update-ref", f"refs/heads/{branch}", commit, parent)
        if u.returncode != 0:
            logger.warning("archive: update-ref failed: %s", u.stderr.strip())
            return False

        logger.info("archive: recorded %s to '%s' (%s)", repo_relpath, branch, commit[:8])
        return True
    except Exception as exc:  # never let archival break a session
        logger.warning("archive: unexpected error: %s", exc)
        return False
