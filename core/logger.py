"""Trial history & artifact manager.

Maintains ``experiments_history.json`` — an append-only record of every
session and trial: trial index, git commit hash, metrics/loss curves,
status, and agent feedback. Also renders the final experiment summary
report (markdown) at the end of a session.

The JSON file is written atomically (tmp file + rename) so a killed run
never corrupts the history.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentLogger:
    def __init__(self, history_path: Path) -> None:
        self.history_path = Path(history_path).resolve()
        self._data = self._load()
        self._session: Optional[Dict[str, Any]] = None

    # -- persistence ---------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if not self.history_path.exists():
            return {"sessions": []}
        try:
            with self.history_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "sessions" not in data:
                raise ValueError("unexpected schema")
            return data
        except (json.JSONDecodeError, ValueError, OSError):
            # Never lose a corrupt file silently — park it aside.
            backup = self.history_path.with_suffix(".corrupt.json")
            try:
                self.history_path.replace(backup)
            except OSError:
                pass
            return {"sessions": []}

    def _save(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.history_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp, self.history_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- session lifecycle -----------------------------------------------------

    def start_session(self, goal: str, config: Dict[str, Any]) -> str:
        session_id = uuid.uuid4().hex[:12]
        self._session = {
            "session_id": session_id,
            "started_at": _now(),
            "finished_at": None,
            "goal": goal,
            "config": config,
            "result": "IN_PROGRESS",
            "best_trial": None,
            "trials": [],
            "events": [],
        }
        self._data["sessions"].append(self._session)
        self._save()
        return session_id

    def record_trial(
        self,
        trial: int,
        commit: Optional[str],
        status: str,
        action: str,
        metrics: Optional[Dict[str, Any]],
        evaluation: Optional[Dict[str, str]],
        editor_summary: str = "",
        error_type: Optional[str] = None,
        runtime_seconds: Optional[float] = None,
    ) -> None:
        if self._session is None:
            raise RuntimeError("record_trial() called before start_session().")
        entry = {
            "trial": trial,
            "timestamp": _now(),
            "commit": commit,
            "status": status,
            "action": action,  # COMMITTED | REVERTED | GOAL_COMMIT | SKIPPED
            "train_loss": (metrics or {}).get("train_loss"),
            "val_loss": (metrics or {}).get("val_loss"),
            "loss_curve": (metrics or {}).get("history"),
            "metrics": metrics,
            "evaluation": evaluation,
            "editor_summary": editor_summary,
            "error_type": error_type,
            "runtime_seconds": runtime_seconds,
        }
        self._session["trials"].append(entry)
        self._save()

    def record_event(self, kind: str, detail: Any) -> None:
        """Log a non-trial event (e.g. an agent session rotation)."""
        if self._session is None:
            raise RuntimeError("record_event() called before start_session().")
        self._session.setdefault("events", []).append({
            "timestamp": _now(),
            "kind": kind,
            "detail": detail,
        })
        self._save()

    def finish_session(self, result: str, best_trial: Optional[int]) -> None:
        if self._session is None:
            return
        self._session["result"] = result
        self._session["best_trial"] = best_trial
        self._session["finished_at"] = _now()
        self._save()

    # -- queries -----------------------------------------------------------------

    @property
    def trials(self) -> List[Dict[str, Any]]:
        return list(self._session["trials"]) if self._session else []

    def best_trial(self) -> Optional[Dict[str, Any]]:
        candidates = [
            t for t in self.trials
            if t.get("val_loss") is not None
            and t.get("action") in ("COMMITTED", "GOAL_COMMIT", "BASELINE")
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t["val_loss"])

    def history_summary(self, max_trials: int = 10) -> str:
        """Compact per-trial digest injected into agent prompts."""
        lines = []
        for t in self.trials[-max_trials:]:
            val = t.get("val_loss")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
            note = ""
            ev = t.get("evaluation") or {}
            if ev.get("reasoning"):
                note = " | " + ev["reasoning"][:160].replace("\n", " ")
            lines.append(
                f"- trial {t['trial']}: status={t['status']}, "
                f"val_loss={val_str}, action={t['action']}{note}"
            )
        return "\n".join(lines)

    # -- reporting ------------------------------------------------------------------

    def render_report(self) -> str:
        if self._session is None:
            return "# Experiment Report\n\nNo session data."
        s = self._session
        best = self.best_trial()
        lines = [
            "# Experiment Summary Report",
            "",
            f"- **Session**   : `{s['session_id']}`",
            f"- **Goal**      : {s['goal']}",
            f"- **Started**   : {s['started_at']}",
            f"- **Finished**  : {s.get('finished_at') or 'n/a'}",
            f"- **Result**    : **{s['result']}**",
            f"- **Trials run**: {len(s['trials'])}",
        ]
        if best:
            lines += [
                f"- **Best trial**: {best['trial']} "
                f"(val_loss={best['val_loss']:.4f}, "
                f"commit `{(best.get('commit') or 'n/a')[:8]}`)",
            ]
        lines += [
            "",
            "## Trial log",
            "",
            "| Trial | Status | val_loss | train_loss | Action | Commit |",
            "|------:|--------|---------:|-----------:|--------|--------|",
        ]
        for t in s["trials"]:
            def fmt(x: Any) -> str:
                return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
            lines.append(
                f"| {t['trial']} | {t['status']} | {fmt(t.get('val_loss'))} "
                f"| {fmt(t.get('train_loss'))} | {t['action']} "
                f"| `{(t.get('commit') or 'n/a')[:8]}` |"
            )
        lines.append("")
        lines.append("## Evaluator feedback per trial")
        for t in s["trials"]:
            ev = t.get("evaluation") or {}
            lines += [
                "",
                f"### Trial {t['trial']} — {t['status']}",
                f"**Reasoning:** {ev.get('reasoning', 'n/a')}",
                "",
                f"**Recommendations:** {ev.get('recommendations', 'n/a')}",
            ]
            if t.get("editor_summary"):
                lines += ["", f"**Editor change summary:** {t['editor_summary']}"]
        events = s.get("events") or []
        if events:
            lines += ["", "## Session events"]
            for e in events:
                lines.append(
                    f"- `{e['timestamp']}` **{e['kind']}** — "
                    f"{json.dumps(e['detail']) if not isinstance(e['detail'], str) else e['detail']}"
                )
        return "\n".join(lines) + "\n"

    def write_report(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(self.render_report(), encoding="utf-8")
        return path
