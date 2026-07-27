"""Tests for ml_orchestrator/core/fitness.py — the generalized objective."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ml_orchestrator.core.fitness import (
    FitnessExtractor, parse_test_results,
)
from ml_orchestrator.core.agents import (
    heuristic_evaluation, parse_goal_condition, parse_goal_target,
)
from ml_orchestrator.main import PRESETS, resolve_goal_condition

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")

def fake_exec(returncode=0, stdout="", stderr="", succeeded=None):
    return SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr,
        succeeded=(returncode == 0) if succeeded is None else succeeded,
    )

# ---- test-output parsing ------------------------------------------------------
print("== parse_test_results ==")
cases = [
    ("== 2 failed, 10 passed in 1.2s ==", {"failed": 2, "passed": 10}),
    ("===== 15 passed in 0.4s =====", {"failed": 0, "passed": 15}),
    ("1 failed, 3 passed, 2 errors in 9s", {"failed": 3, "passed": 3}),
    ("Tests:  2 failed, 17 passed, 19 total", {"failed": 2, "passed": 17}),
    ("Tests:  20 passed, 20 total", {"failed": 0, "passed": 20}),
    ("--- FAIL: TestFoo (0.00s)\nFAIL\nexit status 1", None_go := None),
    ("compiling...\nno tests here", None),
]
for text, expected in cases[:5]:
    got = parse_test_results(text)
    check(f"parse {expected}", got == expected, got)
go = parse_test_results("--- FAIL: TestFoo (0.00s)\nFAIL\nexit status 1")
check("go fail lines", go is not None and go["failed"] >= 1, go)
check("no signal", parse_test_results("compiling...\nnothing") is None)
check("empty", parse_test_results("") is None)

# ---- extractor: metrics file wins --------------------------------------------
print("== extractor ==")
with tempfile.TemporaryDirectory() as td:
    mp = Path(td) / "metrics.json"

    fx = FitnessExtractor(mp, direction="minimize", goal_op="<")
    mp.write_text(json.dumps({"val_loss": 0.25, "train_loss": 0.2}))
    r = fx.extract(fake_exec(0))
    check("val_loss file", r.score == 0.25 and r.source == "metrics-file", r)

    mp.write_text(json.dumps({"score": 0.9}))
    r = fx.extract(fake_exec(0))
    check("generic score file", r.score == 0.9 and r.source == "metrics-file")

    mp.write_text(json.dumps({"score": "not-a-number"}))
    r = fx.extract(fake_exec(1, stdout="3 failed, 5 passed"))
    check("bad score falls to tests", r.source == "test-parse"
          and r.score == 3.0, r)

    mp.unlink()
    r = fx.extract(fake_exec(1, stdout="== 4 failed, 6 passed in 2s =="))
    check("tests parsed on nonzero exit", r.score == 4.0 and r.measured)

    # ml mode: clean exit with no signal is NOT a score
    r = fx.extract(fake_exec(0, stdout="done"))
    check("ml no-signal -> crash", not r.measured and r.source == "none")

    # coding mode: clean exit counts as pass
    fx2 = FitnessExtractor(mp, direction="minimize", goal_op="<=",
                           allow_exit_code_score=True)
    r = fx2.extract(fake_exec(0, stdout="built ok"))
    check("coding exit-code score", r.score == 0.0 and r.source == "exit-code")
    r = fx2.extract(fake_exec(2, stdout="boom"))
    check("coding dirty exit no score", not r.measured)

    # comparisons
    check("min is_better", fx.is_better(0.2, 0.3) and not fx.is_better(0.4, 0.3))
    check("first always better", fx.is_better(9.9, None))
    fxmax = FitnessExtractor(mp, direction="maximize", goal_op=">=")
    check("max is_better", fxmax.is_better(0.9, 0.5) and not fxmax.is_better(0.4, 0.5))
    check("goal <", fx.goal_met(0.24, 0.25) and not fx.goal_met(0.25, 0.25))
    check("goal <=", fx2.goal_met(0.0, 0.0))
    check("goal >=", fxmax.goal_met(0.95, 0.9))
    check("goal none", not fx.goal_met(None, 0.25) and not fx.goal_met(0.1, None))

    try:
        FitnessExtractor(mp, direction="sideways")
        check("bad direction raises", False)
    except ValueError:
        check("bad direction raises", True)

# ---- goal parsing extensions ----------------------------------------------------
print("== goal parsing ==")
check("score under", parse_goal_target("get the score under 0.1") == 0.1)
check("score >=", parse_goal_target("achieve score >= 0.95") == 0.95)
check("plain loss", parse_goal_target("reduce loss below 1.5") == 1.5)
check("val loss still works",
      parse_goal_target("Achieve validation loss < 0.25") == 0.25)

# The operator is as load-bearing as the number: dropping it let a
# maximize goal be tested with the preset's '<'.
check("condition keeps >=",
      parse_goal_condition("achieve score >= 0.95") == (">=", 0.95))
check("condition keeps <",
      parse_goal_condition("validation loss < 0.25") == ("<", 0.25))
check("condition word: at least",
      parse_goal_condition("score at least 0.9") == (">=", 0.9))
check("condition word: above",
      parse_goal_condition("get score above 0.8") == (">", 0.8))
check("condition word: under",
      parse_goal_condition("val loss under .15") == ("<", 0.15))
check("condition none", parse_goal_condition("make the tests pass") is None)

# ---- goal condition resolution (preset < goal text < flags) ----------------------
print("== goal condition resolution ==")

class _Args:
    def __init__(self, goal, direction=None, goal_target=None, goal_op=None):
        self.goal = goal
        self.direction = direction
        self.goal_target = goal_target
        self.goal_op = goal_op

ML = PRESETS["ml"]
CODING = PRESETS["coding"]

c = resolve_goal_condition(_Args("Achieve validation loss < 0.25"), ML)
check("ml default minimize",
      (c["direction"], c["goal_op"], c["goal_target"]) == ("minimize", "<", 0.25), c)

# 2b: a maximize goal written in plain English, no flags at all.
c = resolve_goal_condition(_Args("Achieve score >= 0.9"), ML)
check("goal text flips to maximize",
      (c["direction"], c["goal_op"], c["goal_target"]) == ("maximize", ">=", 0.9), c)

# 2: --direction maximize must flip the comparison, not just the ordering.
c = resolve_goal_condition(_Args("hit the target", direction="maximize",
                                 goal_target=0.9), ML)
check("--direction maximize flips op",
      (c["direction"], c["goal_op"], c["goal_target"]) == ("maximize", ">", 0.9), c)

c = resolve_goal_condition(_Args("make the suite pass", direction="maximize"),
                           CODING)
check("coding <= flips to >=",
      (c["direction"], c["goal_op"]) == ("maximize", ">="), c)

c = resolve_goal_condition(_Args("make the whole test suite pass"), CODING)
check("coding default",
      (c["direction"], c["goal_op"], c["goal_target"]) == ("minimize", "<=", 0.0), c)

c = resolve_goal_condition(_Args("score >= 0.9", goal_op="<", goal_target=0.2),
                           ML)
check("explicit flags win",
      (c["goal_op"], c["goal_target"]) == ("<", 0.2), c)

# The resolved condition must be what actually judges the run.
c = resolve_goal_condition(_Args("Achieve score >= 0.9"), ML)
fx = FitnessExtractor(Path("/nonexistent/metrics.json"),
                      direction=c["direction"], goal_op=c["goal_op"])
check("resolved condition drives goal_met",
      fx.goal_met(0.95, c["goal_target"])
      and not fx.goal_met(0.85, c["goal_target"]))

# ---- heuristic with direction/op --------------------------------------------------
print("== heuristic direction-aware ==")
r = heuristic_evaluation(0.0, 0.0, 3.0, False, goal_op="<=")
check("coding goal at 0", r.status == "GOAL_REACHED", r.status)
r = heuristic_evaluation(0.95, 0.97, 0.9, False, direction="maximize", goal_op=">=")
check("maximize goal", r.status == "GOAL_REACHED")
r = heuristic_evaluation(None, 0.7, 0.9, False, direction="maximize")
check("maximize regressed", r.status == "REGRESSED")
r = heuristic_evaluation(None, 2.0, 5.0, False, goal_op="<=")
check("fewer failures improved", r.status == "IMPROVED")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
