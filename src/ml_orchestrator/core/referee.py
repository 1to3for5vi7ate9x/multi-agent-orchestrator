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

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import kernel

CRITICAL = "CRITICAL"
WARN = "WARN"
INFO = "INFO"

# What counts as a test file (coding preset reward-hacking guard).
#
# THIS REGEX IS THE SPEC. The `isTestPath` function in the Haskell
# kernel must agree with it on every path, and tests/test_paths.json
# is the shared corpus both are checked against. Note the \b before
# test/tests: 'latest_thing.py' and 'contest_form.py' are NOT test
# files, and force-reverting them would be a critical false positive.
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?/|conftest\.py$|pytest\.ini$|[^/]*\btest[s]?_[^/]*\.py$|"
    r"[^/]*_tests?\.py$|[^/]*\.(test|spec)\.[jt]sx?$)"
)

# Default objective keys, mirroring FitnessExtractor's read order.
DEFAULT_METRIC_KEYS: Tuple[str, ...] = ("score", "val_loss")

# A run this much faster than the baseline while also improving is
# fabrication-shaped (the work was removed, not optimized). Legitimate
# causes exist — early stopping, caching — hence WARN, never CRITICAL.
RUNTIME_COLLAPSE_RATIO = 10.0


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


# ---------------------------------------------------------------------------
# ML-preset fabrication detector (AST)
# ---------------------------------------------------------------------------
#
# The coding preset's objective is guarded by TEST_TAMPERING: you cannot
# edit the thing that measures you. The ml preset had no equivalent —
# nothing stopped an editor from gutting the training loop and writing a
# hardcoded metrics.json, which the loop would have committed as an
# improvement. These two rules close the obvious shapes of that.
#
# Scoped deliberately narrow to keep false positives near zero: only a
# literal objective value that flows into a metrics-file *write* counts.
# `val_loss = 0.0` as an accumulator initializer — ubiquitous in real
# training loops — must never fire.

def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_number(node.operand)
    return False


def _literal_metric_in_dict(node: ast.AST,
                            metric_keys: Sequence[str]) -> Optional[str]:
    """Return 'key=value' when a dict literal maps a metric key to a
    numeric constant, else None."""
    if not isinstance(node, ast.Dict):
        return None
    for key, value in zip(node.keys, node.values):
        if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                and key.value in metric_keys and _is_number(value)):
            return f"{key.value}={ast.unparse(value)}"
    return None


def _is_json_write(func: ast.AST) -> bool:
    return (isinstance(func, ast.Attribute)
            and func.attr in ("dump", "dumps")
            and isinstance(func.value, ast.Name)
            and func.value.id == "json")


def find_literal_metric_writes(
    workdir: Path,
    changed_files: Iterable[str],
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> List[str]:
    """Detect hardcoded objective values being written as metrics.

    Catches three shapes:
      json.dump({"val_loss": 0.001}, fh)
      metrics = {"val_loss": 0.001}; json.dump(metrics, fh)
      path.write_text('{"val_loss": 0.001}')
    """
    workdir = Path(workdir)
    key_re = re.compile(
        r"[\"']("
        + "|".join(re.escape(k) for k in metric_keys)
        + r")[\"']\s*:\s*-?[0-9]"
    )
    findings: List[str] = []
    for rel in changed_files:
        if not str(rel).endswith(".py"):
            continue
        path = workdir / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue

        # Names bound to a dict literal that hardcodes an objective value.
        tainted: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                detail = _literal_metric_in_dict(node.value, metric_keys)
                if detail:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            tainted[target.id] = detail

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_json_write(node.func):
                arg = node.args[0] if node.args else None
                detail = _literal_metric_in_dict(arg, metric_keys) \
                    if arg is not None else None
                if detail is None and isinstance(arg, ast.Name):
                    detail = tainted.get(arg.id)
                if detail:
                    findings.append(
                        f"{rel}: json.{node.func.attr}() of a hardcoded "
                        f"{detail}"
                    )
            elif (isinstance(node, ast.Constant)
                  and isinstance(node.value, str)
                  and key_re.search(node.value)):
                findings.append(
                    f"{rel}: string literal embedding a hardcoded objective "
                    f"value ({node.value.strip()[:60]})"
                )
    return sorted(set(findings))


class Referee:
    def __init__(self, mode: str, metrics_filename: str,
                 direction: str = "minimize",
                 metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS) -> None:
        self.mode = mode
        self.metrics_filename = metrics_filename
        self.direction = direction
        self.metric_keys = tuple(metric_keys)

    # -- individual rules ----------------------------------------------------

    def check_edit_phase(
        self,
        changed_files: List[str],
        metrics_file_written_during_edit: bool,
        literal_metric_writes: Optional[List[str]] = None,
    ) -> List[Flag]:
        literal_metric_writes = list(literal_metric_writes or [])
        # Canonical path: the Haskell decision kernel (fail-open).
        resp = kernel.call({
            "op": "review_edit",
            "mode": self.mode,
            "metrics_file": self.metrics_filename,
            "changed_files": changed_files,
            "metrics_written": bool(metrics_file_written_during_edit),
            "literal_metric_writes": literal_metric_writes,
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
        if literal_metric_writes:
            flags.append(Flag(
                "METRIC_FABRICATION", WARN,
                "The edit writes a hardcoded objective value instead of "
                "measuring one: " + "; ".join(literal_metric_writes[:3])
                + ". Verify the objective is still computed from a real "
                "run.",
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
        duration_seconds: Optional[float] = None,
        baseline_duration_seconds: Optional[float] = None,
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
            "duration_seconds": duration_seconds,
            "baseline_duration_seconds": baseline_duration_seconds,
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
        # Suppressed when the goal condition is met: reaching the target
        # IS the terminal event (10 failing tests -> 0 is a 100% "jump"),
        # and flagging it would feed the judge panel a referee concern
        # about the exact outcome the loop exists to produce.
        if (measured and not goal_met and score is not None
                and best_score is not None and best_score != 0):
            jump = (best_score - score) / abs(best_score) \
                if self.direction == "minimize" \
                else (score - best_score) / abs(best_score)
            if jump > 0.9:
                flags.append(Flag(
                    "SUSPICIOUS_JUMP", INFO,
                    f"Objective improved by {jump:.0%} in a single trial "
                    "— verify it is not an evaluation artifact.",
                ))
        if (is_better and duration_seconds is not None
                and baseline_duration_seconds is not None
                and duration_seconds > 0 and baseline_duration_seconds > 0
                and baseline_duration_seconds / duration_seconds
                >= RUNTIME_COLLAPSE_RATIO):
            flags.append(Flag(
                "RUNTIME_COLLAPSE", WARN,
                f"Run improved the objective while finishing "
                f"{baseline_duration_seconds / duration_seconds:.0f}x "
                f"faster than the baseline "
                f"({duration_seconds:.1f}s vs "
                f"{baseline_duration_seconds:.1f}s) — confirm the work is "
                "still being done and not skipped.",
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
