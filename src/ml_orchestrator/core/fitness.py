"""Generalized fitness contract — what the loop optimizes.

v0.3 hard-coded the objective as ``val_loss`` read from ``metrics.json``.
This module generalizes that into pluggable *fitness extraction* so the
same closed loop drives any goal with an objective signal:

Score sources, tried in order per run:
1. **Metrics file** — a JSON dict containing ``score`` (generic) or
   ``val_loss`` (ML legacy). Always wins when present and numeric.
2. **Test-runner output** — pytest / jest / vitest / go-test summaries
   parsed from stdout+stderr. Score = number of failing tests
   (minimize; goal target 0 == green suite). A test command exiting 1
   because tests failed is a *measurement*, not a crash.
3. **Exit code** — optionally (coding preset), a clean exit with no
   metrics and no recognizable test output scores 0.0, a dirty exit
   scores None (crash).

A run whose score is None is treated as CRASHED by the loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

MINIMIZE = "minimize"
MAXIMIZE = "maximize"


@dataclass
class FitnessResult:
    score: Optional[float]           # objective value; None => not measured
    metrics: Optional[Dict[str, Any]]  # structured metrics when available
    source: str                      # metrics-file | test-parse | exit-code | none
    detail: str                      # human/agent readable one-liner

    @property
    def measured(self) -> bool:
        return self.score is not None


# ---------------------------------------------------------------------------
# Test-output parsing
# ---------------------------------------------------------------------------

def parse_test_results(output: str) -> Optional[Dict[str, int]]:
    """Extract pass/fail counts from common test-runner output.

    Returns {"passed": N, "failed": M} or None when nothing matches.
    ``failed`` includes errored tests.
    """
    if not output:
        return None

    # jest / vitest:  "Tests:  2 failed, 1 skipped, 17 passed, 20 total"
    m = re.search(
        r"Tests:\s+(?:(\d+)\s+failed,\s*)?(?:\d+\s+skipped,\s*)?"
        r"(\d+)\s+passed,\s*\d+\s+total",
        output,
    )
    if m:
        return {"failed": int(m.group(1) or 0), "passed": int(m.group(2))}

    # pytest summary line: "2 failed, 10 passed, 1 error in 0.5s" (any order)
    tail = output[-4000:]
    passed = failed = errors = None
    pm = re.search(r"(\d+) passed", tail)
    fm = re.search(r"(\d+) failed", tail)
    em = re.search(r"(\d+) error(?:s)?\b", tail)
    if pm or fm or em:
        passed = int(pm.group(1)) if pm else 0
        failed = (int(fm.group(1)) if fm else 0) + (int(em.group(1)) if em else 0)
        return {"failed": failed, "passed": passed}

    # mocha:  "  12 passing (340ms)" / "  3 failing"
    mp = re.search(r"^\s*(\d+)\s+passing\b", output, re.MULTILINE)
    mf = re.search(r"^\s*(\d+)\s+failing\b", output, re.MULTILINE)
    if mp or mf:
        return {
            "failed": int(mf.group(1)) if mf else 0,
            "passed": int(mp.group(1)) if mp else 0,
        }

    # go test: count FAIL lines vs the presence of "ok"
    go_fails = re.findall(r"^(?:---\s+)?FAIL(?::|\s)", output, re.MULTILINE)
    go_ok = re.search(r"^ok\s+\S+", output, re.MULTILINE)
    if go_fails or go_ok:
        return {"failed": len(go_fails), "passed": 1 if go_ok else 0}

    return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FitnessExtractor:
    def __init__(
        self,
        metrics_path: Path,
        direction: str = MINIMIZE,
        goal_op: str = "<",
        allow_exit_code_score: bool = False,
    ) -> None:
        if direction not in (MINIMIZE, MAXIMIZE):
            raise ValueError(f"direction must be minimize|maximize, got {direction!r}")
        if goal_op not in ("<", "<=", ">", ">="):
            raise ValueError(f"goal_op must be one of < <= > >=, got {goal_op!r}")
        self.metrics_path = Path(metrics_path)
        self.direction = direction
        self.goal_op = goal_op
        self.allow_exit_code_score = allow_exit_code_score

    # -- score extraction ------------------------------------------------------

    def _read_metrics_file(self) -> Optional[Dict[str, Any]]:
        if not self.metrics_path.exists():
            return None
        try:
            data = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def extract(self, exec_result: Any) -> FitnessResult:
        """Derive the fitness of one run (see module docstring for order)."""
        # 1. Metrics file
        metrics = self._read_metrics_file()
        if metrics is not None:
            raw = metrics.get("score", metrics.get("val_loss"))
            try:
                score = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                score = None
            if score is not None:
                key = "score" if "score" in metrics else "val_loss"
                return FitnessResult(
                    score=score, metrics=metrics, source="metrics-file",
                    detail=f"{key}={score:.6g} from {self.metrics_path.name}",
                )

        # 2. Test-runner output
        output = ""
        if exec_result is not None:
            output = getattr(exec_result, "stdout", "") + "\n" + \
                     getattr(exec_result, "stderr", "")
        tests = parse_test_results(output)
        if tests is not None:
            score = float(tests["failed"])
            detail = f"{tests['passed']} passed, {tests['failed']} failed"
            return FitnessResult(
                score=score,
                metrics={"passed": tests["passed"], "failed": tests["failed"],
                         "score": score},
                source="test-parse",
                detail=detail,
            )

        # 3. Exit code (opt-in)
        returncode = getattr(exec_result, "returncode", None)
        succeeded = bool(getattr(exec_result, "succeeded", False))
        if self.allow_exit_code_score and succeeded:
            return FitnessResult(
                score=0.0, metrics={"score": 0.0}, source="exit-code",
                detail="command succeeded (no metrics/tests parsed)",
            )

        return FitnessResult(
            score=None, metrics=metrics, source="none",
            detail=f"no usable objective signal (exit code {returncode})",
        )

    # -- comparisons ---------------------------------------------------------------

    def is_better(self, new: Optional[float], old: Optional[float]) -> bool:
        if new is None:
            return False
        if old is None:
            return True
        return new < old if self.direction == MINIMIZE else new > old

    def goal_met(self, score: Optional[float], target: Optional[float]) -> bool:
        if score is None or target is None:
            return False
        return {
            "<": score < target,
            "<=": score <= target,
            ">": score > target,
            ">=": score >= target,
        }[self.goal_op]

    def describe(self) -> str:
        return (f"{self.direction} the score; goal condition: "
                f"score {self.goal_op} target")
