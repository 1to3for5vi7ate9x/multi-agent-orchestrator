"""Git versioning & rollback module.

Responsibilities:
- Verify (or optionally initialize) the repository before an experiment.
- Snapshot dirty state so every trial starts from a committed baseline.
- Commit improvements with structured messages + lightweight tags:
      experiment(trial-3): val_loss=0.3120
- Discard failed/regressed edits (checkout of tracked files).
- Hard-rollback to the best-known commit when required.
- Keep orchestrator artifacts (metrics.json, experiments_history.json)
  out of the experiment history via .gitignore management.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


class GitError(RuntimeError):
    """Raised when a git operation fails in a way the loop cannot ignore."""


DEFAULT_IGNORES = [
    "metrics.json",
    "experiments_history.json",
    "experiment_report.md",
    "__pycache__/",
    "*.pyc",
    ".DS_Store",
]


class GitManager:
    def __init__(self, repo_path: Path, author: str = "ml-agent-orchestrator") -> None:
        self.repo_path = Path(repo_path).resolve()
        self.author = author

    # -- low-level ----------------------------------------------------------

    def _git(
        self,
        *args: str,
        check: bool = True,
        timeout: float = 60.0,
    ) -> Tuple[int, str, str]:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise GitError("The 'git' executable was not found on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {' '.join(args)} timed out after {timeout}s.") from exc

        if check and proc.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed (exit {proc.returncode}):\n"
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    # -- repository state ----------------------------------------------------

    def is_repo(self) -> bool:
        rc, out, _ = self._git("rev-parse", "--is-inside-work-tree", check=False)
        return rc == 0 and out == "true"

    def verify_or_init(self, auto_init: bool = False) -> None:
        """Ensure we are inside a usable git repository.

        With ``auto_init=True`` a missing repo is initialized and an initial
        commit is created; otherwise a GitError explains what to do.
        """
        if self.is_repo():
            self._ensure_identity()
            return
        if not auto_init:
            raise GitError(
                f"{self.repo_path} is not a git repository. "
                "Run 'git init && git add -A && git commit -m init' there, "
                "or pass --init-git to let the orchestrator do it."
            )
        self._git("init")
        self._ensure_identity()
        self.ensure_ignored(DEFAULT_IGNORES)
        self._git("add", "-A")
        rc, _, _ = self._git(
            "commit", "-m", "chore: initial experiment baseline", check=False
        )
        if rc != 0 and not self.head_commit():
            # Empty directory: create an empty initial commit so HEAD exists.
            self._git("commit", "--allow-empty", "-m", "chore: empty baseline")

    def _ensure_identity(self) -> None:
        """Guarantee commits won't fail on missing user.name/user.email."""
        for key, value in (
            ("user.name", self.author),
            ("user.email", f"{self.author}@localhost"),
        ):
            rc, out, _ = self._git("config", "--get", key, check=False)
            if rc != 0 or not out:
                self._git("config", key, value)

    def ensure_ignored(self, patterns: List[str]) -> None:
        """Append missing *patterns* to .gitignore (created if absent)."""
        gitignore = self.repo_path / ".gitignore"
        existing = set()
        if gitignore.exists():
            existing = {
                line.strip()
                for line in gitignore.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        missing = [p for p in patterns if p not in existing]
        if not missing:
            return
        with gitignore.open("a", encoding="utf-8") as fh:
            if existing:
                fh.write("\n")
            fh.write("# ml-agent-orchestrator artifacts\n")
            fh.write("\n".join(missing) + "\n")

    def head_commit(self) -> Optional[str]:
        rc, out, _ = self._git("rev-parse", "HEAD", check=False)
        return out if rc == 0 else None

    def is_dirty(self) -> bool:
        _, out, _ = self._git("status", "--porcelain")
        return bool(out.strip())

    def changed_files(self) -> List[str]:
        # diff vs HEAD covers staged + unstaged edits; ls-files adds new
        # untracked files. Avoids fragile porcelain column parsing.
        rc, tracked, _ = self._git("diff", "--name-only", "HEAD", check=False)
        if rc != 0:  # e.g. unborn HEAD
            tracked = ""
        _, untracked, _ = self._git(
            "ls-files", "--others", "--exclude-standard"
        )
        files = {
            f.strip()
            for f in tracked.splitlines() + untracked.splitlines()
            if f.strip()
        }
        return sorted(files)

    # -- experiment lifecycle --------------------------------------------------

    def snapshot_dirty_state(self) -> Optional[str]:
        """Commit any pre-existing uncommitted work so trials start clean.

        Returns the new commit hash, or None if the tree was already clean.
        """
        if not self.is_dirty():
            return None
        self._git("add", "-A")
        self._git("commit", "-m", "chore(orchestrator): pre-experiment snapshot")
        return self.head_commit()

    def commit_trial(
        self,
        trial: int,
        val_loss: Optional[float],
        note: str = "",
    ) -> Optional[str]:
        """Commit all current changes with the structured trial message.

        Returns the commit hash, or None when there was nothing to commit.
        """
        if not self.is_dirty():
            return self.head_commit()
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        message = f"experiment(trial-{trial}): val_loss={loss_str}"
        if note:
            message += f"\n\n{note}"
        self._git("add", "-A")
        rc, out, err = self._git("commit", "-m", message, check=False)
        if rc != 0:
            if "nothing to commit" in (out + err).lower():
                return self.head_commit()
            raise GitError(f"Commit for trial {trial} failed: {err or out}")
        commit = self.head_commit()
        # Lightweight tag; -f so re-running a session doesn't hard-fail.
        self._git("tag", "-f", f"exp-trial-{trial}", check=False)
        return commit

    def discard_changes(self, clean_untracked: bool = False) -> None:
        """Revert all tracked-file modifications back to HEAD.

        With ``clean_untracked=True`` also removes newly created untracked
        files (respecting .gitignore, so metrics/history artifacts survive).
        """
        if self.head_commit() is None:
            return  # no baseline to restore
        # Unstage anything the editor may have staged, then restore.
        self._git("reset", "--mixed", "HEAD", check=False)
        self._git("checkout", "--", ".")
        if clean_untracked:
            self._git("clean", "-fd", check=False)

    def rollback_to(self, commit: str) -> None:
        """Hard reset the working tree to a specific known-good commit."""
        if not commit:
            raise GitError("rollback_to() called with an empty commit hash.")
        rc, _, err = self._git("reset", "--hard", commit, check=False)
        if rc != 0:
            raise GitError(f"Rollback to {commit[:10]} failed: {err}")

    def short(self, commit: Optional[str]) -> str:
        return commit[:8] if commit else "n/a"
