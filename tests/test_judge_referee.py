"""Tests for the tournament (judge), referee, roster, and kernel parity.

The golden vectors in kernel_vectors.json define the canonical behavior
of the decision kernel. The Python fallback is checked against them
always; the compiled Haskell kernel (MAO_KERNEL / mao-kernel on PATH)
is checked against the SAME vectors when present.
"""
import json
import os
import random
import sys
from pathlib import Path

from ml_orchestrator.core import kernel
from ml_orchestrator.core.referee import Referee, Flag, is_test_path
from ml_orchestrator.core.roster import default_roster, resolve_roster
from ml_orchestrator.core.tournament import (
    parse_scores, run_tournament, scrub_identity,
)

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")

# Force the Python fallback for the deterministic unit tests: point the
# kernel env at a non-existent path so kernel.call() returns None.
os.environ[kernel.KERNEL_ENV] = "/nonexistent/mao-kernel"

# ---- scrubbing & score parsing ------------------------------------------------
print("== anonymization & parsing ==")
s = scrub_identity("As Claude Code (an Anthropic model) I'd beat GPT-5 and Gemini.")
check("scrub", "Claude" not in s and "Anthropic" not in s
      and "GPT" not in s and "Gemini" not in s, s)
check("scrub keeps text", "[model]" in s and "I'd beat" in s)

check("scores basic", parse_scores("SCORES: A=7 B=5 C=9", "ABC")
      == {"A": 7.0, "B": 5.0, "C": 9.0})
check("scores decimals/markdown",
      parse_scores("**SCORES:** A=7.5, B: 6", "AB")
      == {"A": 7.5, "B": 6.0})
check("scores clamp", parse_scores("A=15 B=-2", "AB") == {"A": 10.0, "B": 0.0})
check("scores none", parse_scores("no scores here", "AB") is None)

# ---- tournament with scripted fake agents ---------------------------------------
print("== tournament ==")

class FakeAgent:
    def __init__(self, name, proposal, judge_reply=None, fail=False):
        self.name = name
        self.proposal = proposal
        self.judge_reply = judge_reply
        self.fail = fail
    def ask(self, prompt):
        if self.fail:
            raise RuntimeError("CLI dead")
        if "impartial judge" in prompt:
            if self.judge_reply:
                return self.judge_reply(prompt)
            # score candidates by quality keyword
            import re as _re
            out = []
            for label, block in _re.findall(
                    r"### Candidate (\w)\n(.*?)(?=### Candidate |\Z)",
                    prompt, _re.DOTALL):
                score = 9 if "weight decay" in block else \
                        7 if "learning rate" in block else 4
                out.append(f"{label}={score}")
            return "SCORES: " + " ".join(out) + "\nREASON: keyword judge."
        return self.proposal

agents = {
    "claude": FakeAgent("claude", "CHANGE: lower the learning rate to 0.005"),
    "antigravity": FakeAgent("antigravity", "CHANGE: refactor the dataloader"),
    "codex": FakeAgent("codex", "CHANGE: add weight decay 1e-3"),
}
res = run_tournament(agents=agents, goal="val loss < 0.25", trial=1,
                     knowledge_context="", history_summary="",
                     last_feedback="", mode="ml",
                     rng=random.Random(7), incumbent="claude")
check("winner by merit", res is not None and res.winner == "codex",
      res and res.winner)
check("evaluator is runner-up", res.evaluator == "claude", res.evaluator)
check("ranking order", [a for a, _ in res.ranking]
      == ["codex", "claude", "antigravity"], res.ranking)
check("blind labels", set(res.label_map.values())
      == {"claude", "antigravity", "codex"})
check("winning proposal", "weight decay" in res.winning_proposal)

# dead agent excluded, tournament still works with 2
agents2 = dict(agents)
agents2["antigravity"] = FakeAgent("antigravity", "", fail=True)
res2 = run_tournament(agents=agents2, goal="g", trial=2,
                      knowledge_context="", history_summary="",
                      last_feedback="", mode="ml",
                      rng=random.Random(7), incumbent=None)
check("dead agent excluded", res2 is not None
      and "antigravity" not in dict(res2.ranking)
      and any("antigravity" in n for n in res2.notes))

# fewer than two proposals -> abort
res3 = run_tournament(agents={"claude": agents["claude"],
                              "codex": FakeAgent("codex", "", fail=True)},
                      goal="g", trial=3, knowledge_context="",
                      history_summary="", last_feedback="", mode="ml",
                      rng=random.Random(7))
check("aborts under 2 proposals", res3 is None)

# tie prefers incumbent
tie_judge = lambda prompt: "SCORES: " + " ".join(
    f"{l}=8" for l in ["A", "B"]) + "\nREASON: tie."
agents4 = {
    "claude": FakeAgent("claude", "CHANGE: x", judge_reply=tie_judge),
    "codex": FakeAgent("codex", "CHANGE: y", judge_reply=tie_judge),
}
res4 = run_tournament(agents=agents4, goal="g", trial=4,
                      knowledge_context="", history_summary="",
                      last_feedback="", mode="ml",
                      rng=random.Random(1), incumbent="codex")
check("tie keeps incumbent", res4 is not None and res4.winner == "codex",
      res4 and res4.winner)

# ---- referee -------------------------------------------------------------------
print("== referee ==")
check("test path detection",
      all(is_test_path(p) for p in
          ["tests/test_calc.py", "test_app.py", "src/foo_test.py",
           "conftest.py", "e2e/button.spec.ts", "pkg/api.test.js"]))
check("non-test paths",
      not any(is_test_path(p) for p in
              ["calc.py", "src/main.py", "contest.py", "latest_news.py"]))

ref_c = Referee(mode="coding", metrics_filename="metrics.json")
flags = ref_c.check_edit_phase(["calc.py", "tests/test_calc.py"], False)
check("tamper critical", any(
    f.code == "TEST_TAMPERING" and f.severity == "CRITICAL" for f in flags))

ref_ml = Referee(mode="ml", metrics_filename="metrics.json")
flags = ref_ml.check_edit_phase(["tests/test_calc.py"], True)
check("ml ignores tests, flags metrics",
      not any(f.code == "TEST_TAMPERING" for f in flags)
      and any(f.code == "METRICS_TAMPERING" and f.severity == "WARN"
              for f in flags))

flags = ref_ml.check_result_phase("IMPROVED", measured=False, score=None,
                                  best_score=0.4, goal_met=False,
                                  is_better=False)
check("verdict on crash", any(f.code == "VERDICT_ON_CRASH"
                              and f.severity == "CRITICAL" for f in flags))
flags = ref_ml.check_result_phase("IMPROVED", True, 0.5, 0.4, False, False)
check("contradiction", any(f.code == "VERDICT_CONTRADICTION" for f in flags))
flags = ref_ml.check_result_phase("GOAL_REACHED", True, 0.3, 0.4, False, True)
check("premature goal", any(f.code == "PREMATURE_GOAL" for f in flags))
flags = ref_ml.check_result_phase("IMPROVED", True, 0.01, 0.4, False, True)
check("suspicious jump", any(f.code == "SUSPICIOUS_JUMP" for f in flags))
flags = ref_ml.check_result_phase("IMPROVED", True, 0.35, 0.4, False, True)
check("clean trial no flags", flags == [])

class FakeKG:
    def dead_ends(self):
        return [{"subject": "train.py:LR", "predicate": "CHANGED",
                 "object": "0.01 -> 0.5", "outcome": "CRASHED"}]
check("repeated dead end",
      Referee.find_repeated_dead_end(
          FakeKG(), [("train.py", "LR", "0.01", "0.5")]) is not None)
check("new change not dead end",
      Referee.find_repeated_dead_end(
          FakeKG(), [("train.py", "LR", "0.01", "0.002")]) is None)
check("worst severity",
      Referee.worst_severity([Flag("A", "INFO", ""),
                              Flag("B", "CRITICAL", "")]) == "CRITICAL")

# ---- roster --------------------------------------------------------------------
print("== roster ==")
r = default_roster()
check("roster vendors", set(r) == {"claude", "antigravity", "codex"})
check("codex capture flag", r["codex"].last_message_flag == "-o"
      and r["codex"].session_adapter is None)
try:
    resolve_roster(["claude", "nonsense"])
    check("unknown agent raises", False)
except ValueError:
    check("unknown agent raises", True)

# ---- kernel: fail-open + golden vectors -------------------------------------------
print("== kernel ==")
check("kernel fail-open", kernel.call({"op": "aggregate"}) is None)

vectors = json.loads(
    (Path(__file__).parent / "kernel_vectors.json").read_text())["vectors"]

def python_kernel(request):
    """Run a vector through the Python fallback implementations."""
    op = request["op"]
    if op == "aggregate":
        labels = request["labels"]
        label_map = request["label_map"]
        judges = request["judge_scores"]
        means = {}
        for label in labels:
            vals = [s[label] for s in judges.values() if label in s]
            if vals:
                means[label_map[label]] = sum(vals) / len(vals)
        ranking = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        top = ranking[0][1]
        tied = [a for a, s in ranking if abs(s - top) < 1e-9]
        incumbent = request.get("incumbent")
        winner = incumbent if incumbent in tied else tied[0]
        others = [a for a, _ in ranking if a != winner]
        return {"winner": winner,
                "evaluator": others[0] if others else winner,
                "ranking": [[a, s] for a, s in ranking]}
    ref = Referee(mode=request.get("mode", "ml"),
                  metrics_filename=request.get("metrics_file", "metrics.json"),
                  direction=request.get("direction", "minimize"))
    if op == "review_edit":
        flags = ref.check_edit_phase(request["changed_files"],
                                     request["metrics_written"])
    else:
        flags = ref.check_result_phase(
            request["status"], request["measured"], request["score"],
            request["best_score"], request["goal_met"],
            request["is_better"], request.get("repeated_dead_end"))
    return {"flags": [f.to_dict() for f in flags]}

def matches(expect, got, name):
    if "winner" in expect:
        if got.get("winner") != expect["winner"]:
            return f"winner {got.get('winner')} != {expect['winner']}"
        if got.get("evaluator") != expect["evaluator"]:
            return f"evaluator {got.get('evaluator')}"
        if "ranking" in expect:
            g = [(a, round(float(s), 6)) for a, s in got["ranking"]]
            e = [(a, round(float(s), 6)) for a, s in expect["ranking"]]
            if g != e:
                return f"ranking {g} != {e}"
    if "flags" in expect:
        g = [(f["code"], f["severity"]) for f in got.get("flags", [])]
        e = [(f["code"], f["severity"]) for f in expect["flags"]]
        if sorted(g) != sorted(e):
            return f"flags {g} != {e}"
    return None

for vec in vectors:
    got = python_kernel(vec["request"])
    err = matches(vec["expect"], got, vec["name"])
    check(f"python vector: {vec['name']}", err is None, err or "")

# Haskell kernel parity (only when a real binary is available).
os.environ.pop(kernel.KERNEL_ENV, None)
kpath = kernel.kernel_path()
if kpath:
    print(f"== haskell kernel parity ({kpath}) ==")
    for vec in vectors:
        got = kernel.call(vec["request"])
        err = matches(vec["expect"], got or {}, vec["name"]) \
            if got else "kernel returned None"
        check(f"haskell vector: {vec['name']}", err is None, err or "")
else:
    print("== haskell kernel not installed; parity vectors skipped "
          "(python fallback verified above) ==")
os.environ[kernel.KERNEL_ENV] = "/nonexistent/mao-kernel"

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
