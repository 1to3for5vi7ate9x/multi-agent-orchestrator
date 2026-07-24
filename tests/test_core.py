"""Smoke tests for ml_orchestrator core modules."""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path



from ml_orchestrator.core.agents import (
    GeminiEvaluator, heuristic_evaluation, parse_goal_target, AgentError,
)
from ml_orchestrator.core.runner import ExecutionHarness
from ml_orchestrator.core.git_manager import GitManager, DEFAULT_IGNORES
from ml_orchestrator.core.logger import ExperimentLogger

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")

# ---- agents: schema parsing -------------------------------------------------
print("== agents.parse_response ==")
r = GeminiEvaluator.parse_response(
    "STATUS: [IMPROVED]\nREASONING: Val loss fell from 0.4 to 0.3, "
    "healthy convergence.\nRECOMMENDATIONS: 1. Lower LR. 2. Add dropout."
)
check("status", r.status == "IMPROVED", r.status)
check("reasoning", "0.4 to 0.3" in r.reasoning)
check("recommendations", "Lower LR" in r.recommendations)

r = GeminiEvaluator.parse_response(
    "**STATUS:** GOAL_REACHED\n**REASONING:** target met.\n"
    "**RECOMMENDATIONS:** none."
)
check("markdown-tolerant", r.status == "GOAL_REACHED", r.status)

r = GeminiEvaluator.parse_response(
    "Based on my analysis the run CRASHED due to an OOM."
)
check("bare-token fallback", r.status == "CRASHED", r.status)

try:
    GeminiEvaluator.parse_response("I cannot determine anything.")
    check("unparseable raises", False)
except AgentError:
    check("unparseable raises", True)

# ---- agents: goal parsing / heuristic ----------------------------------------
print("== goal parsing / heuristic ==")
check("goal <", parse_goal_target("Achieve validation loss < 0.25 on data") == 0.25)
check("goal below", parse_goal_target("get val_loss below 0.3") == 0.3)
check("goal under", parse_goal_target("val loss under .15") == 0.15)
check("goal none", parse_goal_target("maximize accuracy") is None)

check("heur crash", heuristic_evaluation(0.25, None, 0.5, True).status == "CRASHED")
check("heur goal", heuristic_evaluation(0.25, 0.2, 0.5, False).status == "GOAL_REACHED")
check("heur improved", heuristic_evaluation(0.1, 0.3, 0.5, False).status == "IMPROVED")
check("heur regressed", heuristic_evaluation(0.1, 0.6, 0.5, False).status == "REGRESSED")
check("heur first run", heuristic_evaluation(None, 0.6, None, False).status == "IMPROVED")

# ---- runner: classification --------------------------------------------------
print("== runner.classify_error ==")
cases = [
    ("RuntimeError: CUDA out of memory. Tried to allocate 2GiB", "CUDA_OOM"),
    ("RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x10 and 20x5)", "SHAPE_MISMATCH"),
    ("  File train.py, line 3\n    def f(:\nSyntaxError: invalid syntax", "SYNTAX_ERROR"),
    ("ModuleNotFoundError: No module named 'torchvision'", "IMPORT_ERROR"),
    ("epoch 5: loss = nan, stopping", "NAN_LOSS"),
    ("Traceback (most recent call last):\n  ValueError: weird", "GENERIC_EXCEPTION"),
    ("all good", None),
]
for text, expected in cases:
    etype, hint = ExecutionHarness.classify_error(text)
    check(f"classify {expected}", etype == expected, f"got {etype}")
    if expected:
        check(f"hint {expected}", bool(hint))

# ---- runner: execution + timeout -------------------------------------------------
print("== runner.run ==")
with tempfile.TemporaryDirectory() as td:
    h = ExecutionHarness(cwd=Path(td), echo=False)
    res = h.run([sys.executable, "-c", "print('hello'); import sys; sys.exit(0)"])
    check("run ok", res.succeeded and "hello" in res.stdout)

    res = h.run([sys.executable, "-c",
                 "raise RuntimeError('CUDA out of memory. boom')"])
    check("run crash detected", res.crashed and res.error_type == "CUDA_OOM",
          f"{res.error_type}")
    diag = ExecutionHarness.build_diagnostic_prompt(res)
    check("diag has hint", "batch size" in diag and "Log tail" in diag)

    t0 = time.monotonic()
    res = h.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=2)
    check("timeout kills", res.timed_out and res.error_type == "TIMEOUT"
          and time.monotonic() - t0 < 15, f"{time.monotonic()-t0:.1f}s")

    res = h.run(["definitely-not-a-binary-xyz"])
    check("launch failure", res.error_type == "LAUNCH_FAILURE" and res.crashed)

# ---- git manager --------------------------------------------------------------
print("== git_manager ==")
with tempfile.TemporaryDirectory() as td:
    repo = Path(td)
    (repo / "train.py").write_text("print('v1')\n")
    g = GitManager(repo)
    check("not repo yet", not g.is_repo())
    g.verify_or_init(auto_init=True)
    check("repo initialized", g.is_repo() and g.head_commit())
    check("gitignore written",
          "metrics.json" in (repo / ".gitignore").read_text())

    # dirty snapshot
    (repo / "train.py").write_text("print('v2')\n")
    snap = g.snapshot_dirty_state()
    check("snapshot commit", snap is not None and not g.is_dirty())

    # trial commit
    (repo / "train.py").write_text("print('v3')\n")
    c = g.commit_trial(3, 0.312)
    log = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=td,
                         capture_output=True, text=True).stdout.strip()
    check("trial commit msg", log == "experiment(trial-3): val_loss=0.3120", log)
    tags = subprocess.run(["git", "tag"], cwd=td, capture_output=True,
                          text=True).stdout
    check("trial tag", "exp-trial-3" in tags)

    # discard
    (repo / "train.py").write_text("print('bad edit')\n")
    check("dirty before discard", g.is_dirty())
    g.discard_changes()
    check("discard restores", (repo / "train.py").read_text() == "print('v3')\n")

    # ignored artifacts survive discard
    (repo / "metrics.json").write_text("{}")
    g.discard_changes()
    check("artifacts survive", (repo / "metrics.json").exists())

    # rollback
    (repo / "train.py").write_text("print('v4')\n")
    g.commit_trial(4, 0.5)
    g.rollback_to(c)
    check("rollback", (repo / "train.py").read_text() == "print('v3')\n")

    # commit with nothing to commit
    check("noop commit ok", g.commit_trial(5, 0.1) == g.head_commit())

# ---- logger -------------------------------------------------------------------
print("== logger ==")
with tempfile.TemporaryDirectory() as td:
    hp = Path(td) / "experiments_history.json"
    lg = ExperimentLogger(hp)
    sid = lg.start_session("val loss < 0.25", {"max_trials": 5})
    lg.record_trial(trial=1, commit="abc123", status="IMPROVED",
                    action="COMMITTED",
                    metrics={"val_loss": 0.4, "train_loss": 0.3,
                             "history": [1, 2]},
                    evaluation={"status": "IMPROVED", "reasoning": "better",
                                "recommendations": "more"})
    lg.record_trial(trial=2, commit=None, status="CRASHED", action="REVERTED",
                    metrics=None, evaluation={"status": "CRASHED",
                                              "reasoning": "oom",
                                              "recommendations": "smaller"})
    lg.record_trial(trial=3, commit="def456", status="GOAL_REACHED",
                    action="GOAL_COMMIT",
                    metrics={"val_loss": 0.2, "train_loss": 0.18},
                    evaluation={"status": "GOAL_REACHED", "reasoning": "done",
                                "recommendations": "none"})
    best = lg.best_trial()
    check("best trial", best and best["trial"] == 3, best)
    lg.finish_session("GOAL_REACHED", 3)

    data = json.loads(hp.read_text())
    check("persisted", data["sessions"][0]["result"] == "GOAL_REACHED"
          and len(data["sessions"][0]["trials"]) == 3)
    summary = lg.history_summary()
    check("summary lines", summary.count("- trial") == 3, summary)
    report = lg.render_report()
    check("report", "GOAL_REACHED" in report and "| 3 |" in report)

    # corrupt-file recovery
    hp.write_text("{{{not json")
    lg2 = ExperimentLogger(hp)
    check("corrupt recovery", lg2._data == {"sessions": []}
          and hp.with_suffix(".corrupt.json").exists())

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
