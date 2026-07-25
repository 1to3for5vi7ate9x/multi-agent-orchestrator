"""Referee — deterministic flow watchdog.

The judge decides who drives; the referee makes sure nobody cheats or
spins in circles. It is intentionally NOT an LLM: every check is a pure
rule over observable facts (git diffs, AST diffs, fitness numbers, the
knowledge graph), so its flags are trustworthy inputs both for the
loop's own decisions and for the judge panel's next tournament.

Severities:
    CRITICAL — the trial result cannot be trusted; the loop force-reverts
               regardless of the evaluator's verdict (e.g. the editor
               modified test files in coding mode = reward hacking).
    WARN     — suspicious; recorded, surfaced to evaluator + judges.
    INFO     — notable flow signal for the record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import kernel

CRITICAL = "CRITICAL"
WARN = "WARN"
INFO = "INFO"

# What counts as a test file (coding preset reward-hacking guard).
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?/|conftest\.py$|pytest\.ini$|[^/]*\btest[s]?_[^/]*\.py$|"
    r"[^/]*_tests?\.py$|[^/]*\.(test|spec)\.[jt]sx?$)"
)


@dataclass
class Flag:
    code: str
    severity: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"code": self.code, "severity": self.severity,
                "detail": self.detail}


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH_RE.search(path.strip()))


class Referee:
    def __init__(self, mode: str, metrics_filename: str,
                 direction: str = "minimize") -> None:
        self.mode = mode
        self.metrics_filename = metrics_filename
        self.direction = direction

    # -- individual rules ----------------------------------------------------

    def check_edit_phase(
        self,
        changed_files: List[str],
        metrics_file_written_during_edit: bool,
    ) -> List[Flag]:
        # Canonical path: the Haskell decision kernel (fail-open).
        resp = kernel.call({
            "op": "review_edit",
            "mode": self.mode,
            "metrics_file": self.metrics_filename,
            "changed_files": changed_files,
            "metrics_written": bool(metrics_file_written_during_edit),
        })
        if resp and isinstance(resp.get("flags"), list):
            return [Flag(f["code"], f["severity"], f["detail"])
                    for f in resp["flags"]]
        flags: List[Flag] = []
        if self.mode == "coding":
            touched_tests = [f for f in changed_files if is_test_path(f)]
            if touched_tests:
                flags.append(Flag(
                    "TEST_TAMPERING", CRITICAL,
                    "Editor modified test files in coding mode: "
                    + ", ".join(touched_tests[:5])
                    + ". Fitness cannot be trusted; forcing revert.",
                ))
        if metrics_file_written_during_edit:
            flags.append(Flag(
                "METRICS_TAMPERING", WARN,
                f"Editor wrote '{self.metrics_filename}' directly during "
                "the edit phase (it was deleted before the run, but this "
                "is a fabrication attempt signal).",
            ))
        if not changed_files:
            flags.append(Flag(
                "NO_CHANGES", INFO,
                "Editor reported completion without modifying any file.",
            ))
        return flags

    def check_result_phase(
        self,
        evaluation_status: str,
        measured: bool,
        score: Optional[float],
        best_score: Optional[float],
        goal_met: bool,
        is_better: bool,
        repeated_dead_end: Optional[str] = None,
    ) -> List[Flag]:
        # Canonical path: the Haskell decision kernel (fail-open).
        resp = kernel.call({
            "op": "review_result",
            "direction": self.direction,
            "status": evaluation_status,
            "measured": bool(measured),
            "score": score,
            "best_score": best_score,
            "goal_met": bool(goal_met),
            "is_better": bool(is_better),
            "repeated_dead_end": repeated_dead_end,
        })
        if resp and isinstance(resp.get("flags"), list):
            return [Flag(f["code"], f["severity"], f["detail"])
                    for f in resp["flags"]]
        flags: List[Flag] = []
        if evaluation_status in ("IMPROVED", "GOAL_REACHED") and not measured:
            flags.append(Flag(
                "VERDICT_ON_CRASH", CRITICAL,
                f"Evaluator returned {evaluation_status} but the run "
                "produced no measurable objective.",
            ))
        if (evaluation_status == "IMPROVED" and measured
                and best_score is not None and not is_better):
            flags.append(Flag(
                "VERDICT_CONTRADICTION", WARN,
                f"Evaluator said IMPROVED but score={score} does not beat "
                f"best={best_score}; numeric truth wins.",
            ))
        if (evaluation_status == "GOAL_REACHED" and measured
                and not goal_met):
            flags.append(Flag(
                "PREMATURE_GOAL", WARN,
                f"Evaluator declared GOAL_REACHED but score={score} does "
                "not satisfy the numeric goal condition.",
            ))
        if repeated_dead_end:
            flags.append(Flag(
                "REPEATED_DEAD_END", WARN,
                "This trial repeats a change already recorded as a dead "
                f"end: {repeated_dead_end}",
            ))
        if (measured and score is not None and best_score is not None
                and best_score != 0):
            jump = (best_score - score) / abs(best_score) \
                if self.direction == "minimize" \
                else (score - best_score) / abs(best_score)
            if jump > 0.9:
                flags.append(Flag(
                    "SUSPICIOUS_JUMP", INFO,
                    f"Objective improved by {jump:.0%} in a single trial "
                    "— verify it is not an evaluation artifact.",
                ))
        return flags

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def find_repeated_dead_end(
        kg: Any,
        const_changes: List[Tuple[str, str, Optional[str], Optional[str]]],
    ) -> Optional[str]:
        """Return a description if any current change matches a dead-end
        fact already in the knowledge graph."""
        try:
            dead = kg.dead_ends()
        except Exception:
            return None
        dead_keys = {
            (f.get("subject"), f.get("object"))
            for f in dead if f.get("predicate") == "CHANGED"
        }
        for module, name, old, new in const_changes or []:
            subject = f"{module}:{name}"
            obj = f"{old} -> {new}" if old is not None and new is not None \
                else (f"added={new}" if old is None else f"removed (was {old})")
            if (subject, obj) in dead_keys:
                return f"{subject} {obj}"
        return None

    @staticmethod
    def worst_severity(flags: List[Flag]) -> Optional[str]:
        order = {CRITICAL: 3, WARN: 2, INFO: 1}
        if not flags:
            return None
        return max(flags, key=lambda f: order.get(f.severity, 0)).severity

    @staticmethod
    def render(flags: List[Flag]) -> str:
        if not flags:
            return ""
        return "\n".join(
            f"- [{f.severity}] {f.code}: {f.detail}" for f in flags
        )
