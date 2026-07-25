#!/usr/bin/env python3
"""ml-agent-orchestrator — closed-loop automated ML experiment engine.

Coordinates:
  Claude Code      (Editor)    -> edits model/training source files
  Training run     (Harness)   -> sandboxed execution + metrics.json capture
  Antigravity CLI  (Evaluator) -> scientific verdict in a strict schema
  Git              (State)     -> commit improvements, revert regressions

Usage:
  python main.py --goal "Achieve validation loss < 0.25 without overfitting" \\
                 --max-trials 5 --train-script train.py --timeout 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import (
    AgentError,
    AntigravityEvaluator,
    ClaudeEditor,
    ExecutionHarness,
    ExperimentLogger,
    FitnessExtractor,
    FitnessResult,
    GitError,
    GitManager,
    STATUS_CRASHED,
    STATUS_GOAL_REACHED,
    STATUS_IMPROVED,
    STATUS_REGRESSED,
)
from .core import CodeGraph, ExperimentKnowledgeGraph, format_changes
from .core import AskAgent, Referee, resolve_roster, run_tournament
from .core.agents import heuristic_evaluation, parse_goal_target
from .core.referee import CRITICAL as FLAG_CRITICAL
from .core.git_manager import DEFAULT_IGNORES
from .core.runner import ExecutionHarness as Harness  # alias for clarity
from .core.session import (
    AntigravitySessionAdapter,
    ClaudeSessionAdapter,
    ManagedSession,
    MemoryStore,
)

# ---------------------------------------------------------------------------
# Presets: how the objective is defined and judged
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    # Classic ML experimentation: minimize val_loss from metrics.json.
    "ml": {
        "direction": "minimize",
        "goal_op": "<",
        "metric_name": "val_loss",
        "default_goal_target": None,       # parsed from the goal text
        "allow_exit_code_score": False,    # missing metrics = crash
    },
    # General/vibe coding: minimize failing tests; 0 failures = goal.
    "coding": {
        "direction": "minimize",
        "goal_op": "<=",
        "metric_name": "failing_tests",
        "default_goal_target": 0.0,
        "allow_exit_code_score": True,     # clean exit w/o tests counts as pass
    },
}

# --------------------------------------------------------------------------
# Terminal UI — uses `rich` when available, plain ANSI otherwise
# --------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    _console = Console()

    def banner(text: str) -> None:
        _console.print(Panel.fit(text, border_style="cyan"))

    def phase(text: str) -> None:
        _console.print(Rule(f"[bold cyan]{text}[/bold cyan]"))

    def info(text: str) -> None:
        _console.print(f"[white]•[/white] {text}")

    def success(text: str) -> None:
        _console.print(f"[bold green]✔[/bold green] {text}")

    def warn(text: str) -> None:
        _console.print(f"[bold yellow]⚠[/bold yellow] {text}")

    def error(text: str) -> None:
        _console.print(f"[bold red]✘[/bold red] {text}")

    def print_trial_table(trials: List[Dict[str, Any]]) -> None:
        table = Table(title="Session results", header_style="bold cyan")
        for col, justify in (
            ("Trial", "right"), ("Status", "left"), ("val_loss", "right"),
            ("Action", "left"), ("Commit", "left"),
        ):
            table.add_column(col, justify=justify)
        for t in trials:
            val = t.get("score") if t.get("score") is not None \
                else t.get("val_loss")
            table.add_row(
                str(t["trial"]),
                t["status"],
                f"{val:.4f}" if isinstance(val, (int, float)) else "n/a",
                t["action"],
                (t.get("commit") or "n/a")[:8],
            )
        _console.print(table)

except ImportError:  # graceful ANSI fallback
    _C = {"cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
          "red": "\033[91m", "bold": "\033[1m", "end": "\033[0m"}

    def banner(text: str) -> None:
        bar = "=" * 64
        print(f"{_C['cyan']}{bar}\n{text}\n{bar}{_C['end']}")

    def phase(text: str) -> None:
        print(f"\n{_C['bold']}{_C['cyan']}── {text} " + "─" * 40 + _C["end"])

    def info(text: str) -> None:
        print(f"  • {text}")

    def success(text: str) -> None:
        print(f"{_C['green']}  ✔ {text}{_C['end']}")

    def warn(text: str) -> None:
        print(f"{_C['yellow']}  ⚠ {text}{_C['end']}")

    def error(text: str) -> None:
        print(f"{_C['red']}  ✘ {text}{_C['end']}")

    def print_trial_table(trials: List[Dict[str, Any]]) -> None:
        print(f"\n{'Trial':>5} {'Status':<14} {'score':>10} "
              f"{'Action':<12} Commit")
        for t in trials:
            val = t.get("score") if t.get("score") is not None \
                else t.get("val_loss")
            val_s = f"{val:.4f}" if isinstance(val, (int, float)) else "n/a"
            print(f"{t['trial']:>5} {t['status']:<14} {val_s:>10} "
                  f"{t['action']:<12} {(t.get('commit') or 'n/a')[:8]}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml-agent-orchestrator",
        description="Closed-loop ML experiment engine driven by Claude Code "
                    "(editor) and Google Antigravity CLI (evaluator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--goal", required=True,
                   help="Target objective, e.g. 'Achieve validation loss "
                        "< 0.25 without overfitting' or 'Make the whole "
                        "test suite pass'.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="ml",
                   help="Objective preset: 'ml' minimizes val_loss/score "
                        "from metrics.json; 'coding' minimizes failing "
                        "tests from --eval-command output (goal: 0).")
    p.add_argument("--eval-command", default=None,
                   help="Shell command that measures the objective each "
                        "trial (e.g. 'pytest -q', 'npm test'). Default: "
                        "'<python> <train-script>' for the ml preset; "
                        "auto-detected test runner for the coding preset.")
    p.add_argument("--direction", choices=["minimize", "maximize"],
                   default=None,
                   help="Whether a lower or higher score is better "
                        "(default: from --preset).")
    p.add_argument("--goal-target", type=float, default=None,
                   help="Explicit numeric goal target for the score "
                        "(default: parsed from --goal, or the preset's "
                        "default).")
    p.add_argument("--scaffold-demo", action="store_true",
                   help="Copy the bundled demo train.py/model_example.py "
                        "into --workdir before starting (self-contained "
                        "try-out, no repo clone needed).")
    p.add_argument("--max-trials", type=int, default=5,
                   help="Maximum number of edit->train->evaluate iterations.")
    p.add_argument("--train-script", default="train.py",
                   help="Training script to execute each trial.")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="Per-run training time limit in seconds.")
    p.add_argument("--workdir", default=".",
                   help="Experiment workspace (must be / become a git repo).")
    p.add_argument("--metrics-file", default="metrics.json",
                   help="Structured metrics file the training script writes.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter used to run the training script.")
    p.add_argument("--editable-files", nargs="*", default=None,
                   help="Files the Editor may modify (default: the training "
                        "script plus model.py/model_example.py if present).")
    p.add_argument("--init-git", action="store_true",
                   help="Initialize a git repo in --workdir if none exists.")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip the initial baseline training run (trial 0).")
    p.add_argument("--claude-cmd", default=None,
                   help="Override the Claude CLI command template, e.g. "
                        "'claude --model claude-fable-5 -p'. The prompt is "
                        "appended as the final argument.")
    p.add_argument("--evaluator-cmd", "--gemini-cmd", dest="evaluator_cmd",
                   default=None,
                   help="Override the Evaluator command template (default: "
                        "'agy -p', the Antigravity CLI in print mode). "
                        "--gemini-cmd is kept as a deprecated alias.")
    p.add_argument("--agent-timeout", type=float, default=900.0,
                   help="Time limit for each agent CLI invocation (seconds).")
    p.add_argument("--no-echo", action="store_true",
                   help="Do not stream training output live to the terminal.")
    p.add_argument("--stateless", action="store_true",
                   help="Disable persistent agent sessions: every call "
                        "spawns a fresh, memoryless CLI process (the old "
                        "behavior).")
    p.add_argument("--context-limit", type=int, default=1_000_000,
                   help="Model context window size in tokens (Claude Code "
                        "on the 1M-context model).")
    p.add_argument("--rotate-at", type=float, default=0.5,
                   help="Fraction of --context-limit at which a session is "
                        "closed and reopened with a distilled memory "
                        "snapshot (quality degrades past ~50%% on 1M "
                        "contexts).")
    p.add_argument("--memory-dir", default=".agent_memory",
                   help="Directory (inside --workdir) holding memory "
                        "snapshots, archives, and live session state.")
    p.add_argument("--agents", nargs="*", default=None,
                   help="Agent pool for seat rotation (subset of: claude "
                        "antigravity codex). Default: every installed one.")
    p.add_argument("--no-rotate", action="store_true",
                   help="Disable blind-tournament seat rotation; pin the "
                        "v0.4 static seats (claude edits, antigravity "
                        "evaluates).")
    p.add_argument("--tournament-every", type=int, default=0,
                   help="Re-run the blind tournament every N trials "
                        "(0 = only at start and after 2 consecutive "
                        "non-improving trials).")
    return p


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def scaffold_demo(workdir: Path) -> List[str]:
    """Copy the bundled demo templates into *workdir*. Returns filenames."""
    workdir.mkdir(parents=True, exist_ok=True)
    pkg = resources.files("ml_orchestrator") / "templates"
    written = []
    for src_name, dst_name in (("train_example.py", "train.py"),
                               ("model_example.py", "model_example.py")):
        dst = workdir / dst_name
        if not dst.exists():
            dst.write_text((pkg / src_name).read_text(encoding="utf-8"),
                           encoding="utf-8")
            written.append(dst_name)
    return written


def build_eval_command(args, preset: Dict[str, Any],
                       workdir: Path) -> Optional[List[str]]:
    """Resolve the command that measures the objective each trial.

    Returns an argv list, or None when nothing sensible can be built.
    """
    if args.eval_command:
        shell = ["cmd", "/c"] if os.name == "nt" else ["/bin/sh", "-c"]
        return shell + [args.eval_command]
    if args.preset == "coding":
        # Auto-detect a test runner in the workspace.
        if any((workdir / f).exists() for f in
               ("pytest.ini", "pyproject.toml", "setup.cfg", "tests",
                "conftest.py")):
            return [args.python, "-m", "pytest", "-q"]
        if (workdir / "package.json").exists():
            shell = ["cmd", "/c"] if os.name == "nt" else ["/bin/sh", "-c"]
            return shell + ["npm test --silent"]
        return None
    return [args.python, args.train_script]


def read_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load and minimally validate metrics.json. Returns None if unusable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    val = data.get("val_loss")
    if val is not None:
        try:
            data["val_loss"] = float(val)
        except (TypeError, ValueError):
            data["val_loss"] = None
    train = data.get("train_loss")
    if train is not None:
        try:
            data["train_loss"] = float(train)
        except (TypeError, ValueError):
            data["train_loss"] = None
    return data


def default_editable_files(workdir: Path, train_script: str) -> List[str]:
    files = [train_script]
    for candidate in ("model.py", "model_example.py", "data.py",
                      "preprocess.py"):
        if (workdir / candidate).exists() and candidate not in files:
            files.append(candidate)
    return files


def run_evaluation(
    harness: ExecutionHarness,
    command: List[str],
    metrics_path: Path,
    timeout: float,
    fitness: FitnessExtractor,
):
    """Delete stale metrics, run the eval command, extract fitness.

    Returns (ExecutionResult, FitnessResult).
    """
    try:
        metrics_path.unlink(missing_ok=True)
    except OSError as exc:
        warn(f"Could not remove stale metrics file: {exc}")
    result = harness.run(command, timeout=timeout)
    fit = fitness.extract(result)
    return result, fit


def evaluate_run(
    evaluator: Optional[AntigravityEvaluator],
    goal: str,
    goal_target: Optional[float],
    trial: int,
    metrics: Optional[Dict[str, Any]],
    best_metrics: Optional[Dict[str, Any]],
    history_summary: str,
    log_tail: str,
    crash_diagnostic: str,
    crashed: bool,
    error_type: Optional[str],
    change_summary: str = "",
    knowledge_context: str = "",
    mode: str = "ml",
    eval_command_display: str = "",
    score: Optional[float] = None,
    best_score: Optional[float] = None,
    direction: str = "minimize",
    goal_op: str = "<",
    evaluator_name: str = "",
):
    """Evaluator-seat verdict with a numeric-heuristic safety net."""
    if evaluator is not None:
        try:
            evaluation = evaluator.evaluate(
                goal=goal,
                trial=trial,
                metrics=metrics,
                best_metrics=best_metrics,
                history_summary=history_summary,
                log_tail=log_tail,
                crash_diagnostic=crash_diagnostic,
                change_summary=change_summary,
                knowledge_context=knowledge_context,
                mode=mode,
                eval_command=eval_command_display,
            )
            # Sanity-guard: never let a hallucinated verdict contradict a
            # hard crash — a run without metrics cannot have improved.
            if crashed and evaluation.status in (STATUS_GOAL_REACHED,
                                                 STATUS_IMPROVED):
                warn("Evaluator claimed success on a crashed run; "
                     "overriding to CRASHED.")
                evaluation.status = STATUS_CRASHED
            if evaluator_name:
                evaluation.source = evaluator_name
            return evaluation
        except AgentError as exc:
            warn(f"Antigravity evaluation failed ({exc}); "
                 "using numeric fallback.")
    return heuristic_evaluation(goal_target, score, best_score, crashed,
                                error_type, direction=direction,
                                goal_op=goal_op)


def format_feedback(evaluation) -> str:
    return (
        f"STATUS: {evaluation.status}\n"
        f"REASONING: {evaluation.reasoning}\n"
        f"RECOMMENDATIONS: {evaluation.recommendations}"
    )


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()
    preset = PRESETS[args.preset]
    direction = args.direction or preset["direction"]
    metric_name = preset["metric_name"]

    banner("ml-agent-orchestrator\n"
           f"Goal: {args.goal}\n"
           f"Preset: {args.preset} ({direction} {metric_name})\n"
           f"Workspace: {workdir}")

    # ---- Phase 0: environment & workspace checks --------------------------
    phase("Phase 0 — Initialization")

    if args.scaffold_demo:
        written = scaffold_demo(workdir)
        if written:
            success(f"Scaffolded demo files into {workdir}: "
                    f"{', '.join(written)}")
        else:
            info("Demo files already present; scaffold skipped.")
    if not workdir.exists():
        error(f"Workspace {workdir} does not exist.")
        return 2

    eval_command = build_eval_command(args, preset, workdir)
    if eval_command is None:
        error("Could not determine an evaluation command for the coding "
              "preset. Pass one explicitly, e.g. "
              "--eval-command 'pytest -q' or --eval-command 'npm test'.")
        return 2
    eval_command_display = args.eval_command or " ".join(
        eval_command[-1:] if eval_command[0] in ("/bin/sh", "cmd")
        else eval_command)

    if args.preset == "ml" and not args.eval_command:
        train_path = workdir / args.train_script
        if not train_path.exists():
            error(f"Training script not found: {train_path}")
            info("Tip: run with --scaffold-demo to drop a working demo "
                 "train.py/model_example.py into the workspace.")
            return 2
    info(f"Evaluation command: {eval_command_display}")

    git = GitManager(workdir)
    try:
        git.verify_or_init(auto_init=args.init_git)
        # All orchestrator artifacts (incl. the agent memory dir) must be
        # ignored BEFORE the pre-experiment snapshot, so the .gitignore
        # change is committed and can never be lost to a trial revert.
        # The memory dir also hosts the code/knowledge graphs, which are
        # maintained even in --stateless mode.
        ignores = list(DEFAULT_IGNORES) + [f"{args.memory_dir.rstrip('/')}/"]
        git.ensure_ignored(ignores)
        snapshot = git.snapshot_dirty_state()
        if snapshot:
            info(f"Committed pre-existing changes as snapshot "
                 f"{git.short(snapshot)}.")
    except GitError as exc:
        error(str(exc))
        return 2
    success(f"Git repository ready at HEAD {git.short(git.head_commit())}.")

    editable = args.editable_files or default_editable_files(
        workdir, args.train_script
    )
    info(f"Editable files: {', '.join(editable)}")

    # ---- Agent roster (claude / antigravity / codex) ----------------------
    try:
        roster = resolve_roster(args.agents)
    except ValueError as exc:
        error(str(exc))
        return 2
    if args.claude_cmd and "claude" in roster:
        roster["claude"].edit_cmd = args.claude_cmd.split()
    if args.evaluator_cmd and "antigravity" in roster:
        roster["antigravity"].ask_cmd = args.evaluator_cmd.split()
    if not roster:
        error("No agent CLIs found on PATH. Install at least one of: "
              "claude (Claude Code), agy (Antigravity), codex (Codex CLI) "
              "and log in.")
        return 2
    info(f"Agent roster: {', '.join(roster)}")
    rotate = (not args.no_rotate) and len(roster) >= 2
    if rotate:
        info("Seat rotation ON — blind tournament (anonymous proposals, "
             "panel judging) assigns the Editor/Evaluator seats; the "
             "referee watches the flow.")
    else:
        info("Seat rotation OFF — static seats.")

    ref = Referee(mode=args.preset, metrics_filename=args.metrics_file,
                  direction=direction)
    recent_flags: List[Dict[str, str]] = []

    # ---- Knowledge layer: code graph + temporal experiment graph -----------
    memory_dir = workdir / args.memory_dir
    memory_dir.mkdir(parents=True, exist_ok=True)
    code_graph_path = memory_dir / "code_graph.json"
    kg = ExperimentKnowledgeGraph(memory_dir / "knowledge_graph.json")
    graph_state = {"current": CodeGraph.build(workdir)}
    graph_state["current"].save(code_graph_path)
    info(f"Code graph: {len(graph_state['current'].modules)} module(s) "
         "mapped from AST.")
    if kg.facts:
        info(f"Knowledge graph: {len(kg.facts)} fact(s) restored from "
             "previous runs.")

    def knowledge_context() -> str:
        """Live machine-derived context: experiment facts + repo map."""
        parts = []
        kg_text = kg.render_context()
        if kg_text:
            parts.append(kg_text)
        parts.append(graph_state["current"].render_map())
        return "\n\n".join(p for p in parts if p.strip())

    memory: Optional[MemoryStore] = None
    if not args.stateless:
        memory = MemoryStore(memory_dir)
        memory.context_provider = knowledge_context
        info("Persistent agent sessions ON — conversations resume across "
             f"trials; rotation at {args.rotate_at:.0%} of "
             f"{args.context_limit:,} tokens with memory handoff.")
    else:
        info("Persistent agent sessions OFF (--stateless): agents forget "
             "everything between calls.")

    # Seat state + lazily-built per-agent instances. Editors get
    # persistent sessions where the CLI supports resume (claude, agy);
    # codex runs stateless and relies on the injected knowledge context.
    live_sessions: Dict[str, ManagedSession] = {}
    editor_cache: Dict[str, ClaudeEditor] = {}
    evaluator_cache: Dict[str, AntigravityEvaluator] = {}
    ask_cache: Dict[str, AskAgent] = {}

    def _session_for(role: str, name: str, base_cmd: List[str]):
        spec = roster[name]
        if memory is None or spec.session_adapter is None:
            return None
        key = f"{role}-{name}"
        if key not in live_sessions:
            live_sessions[key] = ManagedSession(
                agent_name=key,
                role_description=(
                    "the Code & Experiment Editor" if role == "editor"
                    else "the Research & Evaluation Specialist"),
                adapter=spec.session_adapter(base_cmd),
                memory=memory,
                cwd=workdir,
                context_limit=args.context_limit,
                rotate_at=args.rotate_at,
                timeout=args.agent_timeout,
            )
        return live_sessions[key]

    def editor_for(name: str) -> ClaudeEditor:
        if name not in editor_cache:
            spec = roster[name]
            editor_cache[name] = ClaudeEditor(
                cwd=workdir,
                command=spec.edit_cmd,
                timeout=args.agent_timeout,
                session=_session_for("editor", name, spec.edit_cmd),
                last_message_flag=spec.last_message_flag,
            )
        return editor_cache[name]

    def evaluator_for(name: Optional[str]) -> Optional[AntigravityEvaluator]:
        if name is None:
            return None
        if name not in evaluator_cache:
            spec = roster[name]
            evaluator_cache[name] = AntigravityEvaluator(
                cwd=workdir,
                command=spec.ask_cmd,
                timeout=args.agent_timeout,
                session=_session_for("evaluator", name, spec.ask_cmd),
                last_message_flag=spec.last_message_flag,
            )
        return evaluator_cache[name]

    def ask_agents() -> Dict[str, AskAgent]:
        for name, spec in roster.items():
            if name not in ask_cache:
                ask_cache[name] = AskAgent(spec, workdir, args.agent_timeout)
        return ask_cache

    # Initial seats (tournament reassigns them when rotation is on).
    seat_editor = "claude" if "claude" in roster else next(iter(roster))
    _others = [n for n in roster if n != seat_editor]
    seat_evaluator: Optional[str] = (
        "antigravity" if "antigravity" in _others
        else (_others[0] if _others else None))
    if seat_evaluator is None:
        warn("Only one agent available — evaluator falls back to the "
             "built-in numeric heuristic.")

    harness = ExecutionHarness(cwd=workdir, echo=not args.no_echo)
    metrics_path = workdir / args.metrics_file
    fitness = FitnessExtractor(
        metrics_path=metrics_path,
        direction=direction,
        goal_op=preset["goal_op"],
        allow_exit_code_score=preset["allow_exit_code_score"],
    )
    logger = ExperimentLogger(workdir / "experiments_history.json")
    session_id = logger.start_session(args.goal, {
        "max_trials": args.max_trials,
        "preset": args.preset,
        "direction": direction,
        "metric_name": metric_name,
        "eval_command": eval_command_display,
        "train_script": args.train_script,
        "timeout": args.timeout,
        "metrics_file": args.metrics_file,
        "editable_files": editable,
        "workdir": str(workdir),
    })
    info(f"Session {session_id} started; history -> experiments_history.json")

    goal_target = args.goal_target
    if goal_target is None:
        goal_target = parse_goal_target(args.goal)
    if goal_target is None:
        goal_target = preset["default_goal_target"]
    if goal_target is not None:
        info(f"Numeric goal condition: {metric_name} "
             f"{preset['goal_op']} {goal_target}")

    best_metrics: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_commit: Optional[str] = git.head_commit()
    stagnation = 0  # consecutive trials without a commit (rotation trigger)
    pending_feedback = ""
    pending_diagnostic = ""
    final_result = "MAX_TRIALS_EXHAUSTED"

    # ---- Phase 0.5: baseline run ------------------------------------------
    if not args.skip_baseline:
        phase("Trial 0 — Baseline evaluation run")
        result, fit = run_evaluation(
            harness, eval_command, metrics_path, args.timeout, fitness,
        )
        if fit.measured:
            best_metrics = fit.metrics
            best_score = fit.score
            success(f"Baseline established: {metric_name}={fit.score:.4f} "
                    f"[{fit.detail}] ({result.duration_seconds:.1f}s)")
            logger.record_trial(
                trial=0, commit=git.head_commit(), status="BASELINE",
                action="BASELINE", metrics=fit.metrics, evaluation=None,
                runtime_seconds=result.duration_seconds, score=fit.score,
            )
            kg.record_trial(
                trial=0, outcome="BASELINE", action="BASELINE",
                val_loss=fit.score, commit=git.head_commit(),
            )
            if fitness.goal_met(fit.score, goal_target):
                success("Baseline already satisfies the goal — nothing to do.")
                logger.finish_session("GOAL_REACHED_AT_BASELINE", 0)
                report = logger.write_report(workdir / "experiment_report.md")
                info(f"Report written to {report}")
                return 0
        else:
            diag = Harness.build_diagnostic_prompt(result)
            pending_diagnostic = diag
            warn(f"Baseline run produced no measurable objective "
                 f"(error class: {result.error_type or 'none'}); the Editor "
                 "will be asked to repair it in trial 1.")
            logger.record_trial(
                trial=0, commit=git.head_commit(), status="CRASHED",
                action="BASELINE", metrics=fit.metrics, evaluation=None,
                error_type=result.error_type,
                runtime_seconds=result.duration_seconds,
            )

    # ---- Main closed loop ---------------------------------------------------
    for trial in range(1, args.max_trials + 1):
        phase(f"Trial {trial}/{args.max_trials}")

        # -- -1. Blind tournament: reassign seats when due -------------------
        if rotate and (
            trial == 1
            or stagnation >= 2
            or (args.tournament_every > 0 and trial > 1
                and (trial - 1) % args.tournament_every == 0)
        ):
            info("Tournament: collecting anonymous proposals from "
                 f"{', '.join(roster)}; panel judging follows...")
            t_result = run_tournament(
                agents=ask_agents(),
                goal=args.goal,
                trial=trial,
                knowledge_context=knowledge_context(),
                history_summary=logger.history_summary(),
                last_feedback=pending_feedback,
                mode=args.preset,
                referee_notes="\n".join(
                    f"- [{f['severity']}] {f['code']}: {f['detail']}"
                    for f in recent_flags[-8:]),
                incumbent=seat_editor,
                log=info,
            )
            if t_result is None:
                warn("Tournament inconclusive (not enough proposals or "
                     "scores); keeping current seats.")
                logger.record_event("TOURNAMENT_ABORTED", {"trial": trial})
            else:
                seat_editor = t_result.winner
                seat_evaluator = (t_result.evaluator
                                  if t_result.evaluator != t_result.winner
                                  else seat_evaluator)
                for agent_name, mean_score in t_result.ranking:
                    info(f"  panel score — {agent_name}: {mean_score:.2f}")
                success(f"Seats assigned — editor: {seat_editor}, "
                        f"evaluator: {seat_evaluator or 'heuristic'}")
                logger.record_event("TOURNAMENT",
                                    {"trial": trial, **t_result.to_dict()})
                pending_feedback = (
                    "## Your winning tournament proposal — implement "
                    "exactly this\n" + t_result.winning_proposal
                    + (("\n\n" + pending_feedback) if pending_feedback
                       else "")
                )
                stagnation = 0

        # -- 0. Context-budget checkpoint: rotate degraded sessions ----------
        for name, sess in list(live_sessions.items()):
            if sess is None:
                continue
            if sess.should_rotate():
                warn(f"{name} session at {sess.context_fraction():.0%} of "
                     "context — closing it with a memory handoff...")
                try:
                    snapshot = sess.rotate(
                        reason=f"context threshold before trial {trial}"
                    )
                    logger.record_event("SESSION_ROTATED", {
                        "agent": name,
                        "trial": trial,
                        "snapshot_saved": bool(snapshot),
                    })
                    success(f"{name} session rotated; the next call opens a "
                            "fresh session seeded with the memory snapshot.")
                except AgentError as exc:
                    warn(f"Rotation of {name} session failed ({exc}); "
                         "continuing with a fresh session anyway.")
            info(f"Context [{sess.status_line()}]")

        # -- 1. Editor modifies the code ------------------------------------
        info(f"Editor ({seat_editor}): requesting code modification...")
        editor = editor_for(seat_editor)

        def _metrics_sig() -> Optional[str]:
            try:
                return hashlib.sha1(metrics_path.read_bytes()).hexdigest()
            except OSError:
                return None
        pre_edit_metrics_sig = _metrics_sig()

        try:
            editor_summary = editor.request_edit(
                goal=args.goal,
                trial=trial,
                train_script=args.train_script,
                editable_files=editable,
                feedback=pending_feedback,
                diagnostic=pending_diagnostic,
                history_summary=logger.history_summary(),
                knowledge_context=knowledge_context(),
                mode=args.preset,
                eval_command=eval_command_display,
            )
            info(f"Editor summary: {editor_summary[:400]}")
        except AgentError as exc:
            error(f"Editor invocation failed: {exc}")
            logger.record_trial(
                trial=trial, commit=git.head_commit(),
                status="EDITOR_FAILED", action="SKIPPED", metrics=None,
                evaluation=None, error_type="EDITOR_FAILURE",
                editor_agent=seat_editor, evaluator_agent=seat_evaluator,
            )
            pending_diagnostic = ""
            pending_feedback = (
                "The previous editor invocation failed at the CLI level; "
                "no code was changed."
            )
            stagnation += 1
            continue

        if not git.is_dirty():
            warn("Editor reported completion but made no file changes; "
                 "recording and moving on.")
            logger.record_trial(
                trial=trial, commit=git.head_commit(), status="NO_CHANGES",
                action="SKIPPED", metrics=None, evaluation=None,
                editor_summary=editor_summary,
            )
            kg.record_trial(
                trial=trial, outcome="NO_CHANGES", action="SKIPPED",
                val_loss=None, commit=None, editor_summary=editor_summary,
            )
            stagnation += 1
            pending_feedback = (
                "Your previous turn produced NO file modifications. You must "
                "actually edit the source files, not just describe changes."
            )
            pending_diagnostic = ""
            continue
        changed_files = git.changed_files()
        info(f"Modified files: {', '.join(changed_files)}")

        # -- 1.5 AST diff: what exactly changed this trial --------------------
        new_graph = CodeGraph.build(workdir)
        const_changes = graph_state["current"].constants_diff(new_graph)
        func_changes = graph_state["current"].changed_functions(new_graph)
        change_text = format_changes(const_changes, func_changes)
        if change_text:
            for line in change_text.splitlines():
                info(line)
        graph_state["current"] = new_graph
        new_graph.save(code_graph_path)

        # -- 1.6 Referee: edit-phase rules ------------------------------------
        edit_flags = ref.check_edit_phase(
            changed_files,
            metrics_file_written_during_edit=(
                _metrics_sig() != pre_edit_metrics_sig),
        )
        for f in edit_flags:
            (error if f.severity == FLAG_CRITICAL else warn)(
                f"REFEREE [{f.severity}] {f.code}: {f.detail}")
        recent_flags = (recent_flags
                        + [f.to_dict() for f in edit_flags])[-12:]
        if any(f.severity == FLAG_CRITICAL for f in edit_flags):
            git.discard_changes()
            warn("Referee force-reverted the edit; no run performed.")
            logger.record_trial(
                trial=trial, commit=None, status="REFEREE_BLOCKED",
                action="REVERTED", metrics=None, evaluation=None,
                editor_summary=editor_summary,
                referee_flags=[f.to_dict() for f in edit_flags],
                editor_agent=seat_editor, evaluator_agent=seat_evaluator,
            )
            kg.record_trial(
                trial=trial, outcome="REFEREE_BLOCKED", action="REVERTED",
                val_loss=None, commit=None, const_changes=const_changes,
                func_changes=func_changes, editor_summary=editor_summary,
            )
            logger.record_event("REFEREE_BLOCK", {
                "trial": trial, "editor": seat_editor,
                "flags": [f.to_dict() for f in edit_flags],
            })
            graph_state["current"] = CodeGraph.build(workdir)
            graph_state["current"].save(code_graph_path)
            pending_feedback = (
                "THE REFEREE BLOCKED your previous edit and it was "
                "reverted:\n" + Referee.render(edit_flags)
                + "\nFix the SOURCE code — never the tests or the "
                "measurement."
            )
            pending_diagnostic = ""
            stagnation += 1
            continue

        # -- 2. Sandboxed evaluation run ---------------------------------------
        info(f"Runner: executing '{eval_command_display}' "
             f"(timeout {args.timeout:.0f}s)...")
        result, fit = run_evaluation(
            harness, eval_command, metrics_path, args.timeout, fitness,
        )
        metrics = fit.metrics
        crashed = not fit.measured
        crash_diagnostic = ""
        if crashed:
            if result.crashed:
                crash_diagnostic = Harness.build_diagnostic_prompt(result)
                error(f"Run failed after {result.duration_seconds:.1f}s "
                      f"(error class: {result.error_type}).")
            else:
                crash_diagnostic = (
                    "## Objective contract violation\nThe process exited 0 "
                    "but produced no measurable objective: no "
                    f"'{args.metrics_file}' with a numeric score/val_loss, "
                    "and no recognizable test-runner summary in the output."
                )
                warn("Run exited cleanly but produced no measurable "
                     "objective.")
        else:
            note = "" if result.succeeded else \
                " (non-zero exit treated as a measurement, not a crash)"
            success(f"Run completed in {result.duration_seconds:.1f}s: "
                    f"{metric_name}={fit.score:.4f} [{fit.detail}]{note}")

        # -- 3. Evaluator verdict ---------------------------------------------
        info(f"Evaluator ({seat_evaluator or 'numeric heuristic'}): "
             "analyzing metrics and dynamics...")
        try:
            active_evaluator = evaluator_for(seat_evaluator)
        except Exception:
            active_evaluator = None
        evaluation = evaluate_run(
            evaluator=active_evaluator,
            goal=args.goal,
            goal_target=goal_target,
            trial=trial,
            metrics=metrics,
            best_metrics=best_metrics,
            history_summary=logger.history_summary(),
            log_tail=result.tail(3000),
            crash_diagnostic=crash_diagnostic,
            crashed=crashed,
            error_type=result.error_type,
            change_summary=change_text,
            knowledge_context=kg.render_context(),
            mode=args.preset,
            eval_command_display=eval_command_display,
            score=fit.score,
            best_score=best_score,
            direction=direction,
            goal_op=preset["goal_op"],
            evaluator_name=seat_evaluator or "",
        )
        info(f"Verdict: {evaluation.status} (via {evaluation.source})")
        info(f"Reasoning: {evaluation.reasoning[:300]}")

        # -- 3.5 Referee: result-phase rules + verdict arbitration -------------
        result_flags = ref.check_result_phase(
            evaluation_status=evaluation.status,
            measured=fit.measured,
            score=fit.score,
            best_score=best_score,
            goal_met=fitness.goal_met(fit.score, goal_target),
            is_better=fitness.is_better(fit.score, best_score),
            repeated_dead_end=Referee.find_repeated_dead_end(
                kg, const_changes),
        )
        for f in result_flags:
            (error if f.severity == FLAG_CRITICAL else warn)(
                f"REFEREE [{f.severity}] {f.code}: {f.detail}")
        codes = {f.code for f in result_flags}
        if "VERDICT_ON_CRASH" in codes:
            evaluation.status = STATUS_CRASHED
        elif "VERDICT_CONTRADICTION" in codes:
            warn("Referee override: verdict downgraded to REGRESSED "
                 "(numeric ground truth wins).")
            evaluation.status = STATUS_REGRESSED
        elif "PREMATURE_GOAL" in codes:
            downgraded = STATUS_IMPROVED if fitness.is_better(
                fit.score, best_score) else STATUS_REGRESSED
            warn(f"Referee override: GOAL_REACHED downgraded to "
                 f"{downgraded} (goal condition not met numerically).")
            evaluation.status = downgraded
        all_flags = edit_flags + result_flags
        trial_flag_dicts = [f.to_dict() for f in all_flags]
        recent_flags = (recent_flags
                        + [f.to_dict() for f in result_flags])[-12:]
        if result_flags:
            logger.record_event("REFEREE_FLAGS", {
                "trial": trial,
                "flags": [f.to_dict() for f in result_flags],
            })

        # -- 4. Git state decision ---------------------------------------------
        if evaluation.status == STATUS_GOAL_REACHED:
            commit = git.commit_trial(
                trial, fit.score,
                note=f"GOAL REACHED — {args.goal}",
                metric_name=metric_name,
            )
            success(f"GOAL REACHED — committed {git.short(commit)}.")
            logger.record_trial(
                trial=trial, commit=commit, status=evaluation.status,
                action="GOAL_COMMIT", metrics=metrics,
                evaluation=evaluation.to_dict(),
                editor_summary=editor_summary,
                runtime_seconds=result.duration_seconds, score=fit.score,
                referee_flags=trial_flag_dicts,
                editor_agent=seat_editor, evaluator_agent=seat_evaluator,
            )
            kg.record_trial(
                trial=trial, outcome=evaluation.status, action="GOAL_COMMIT",
                val_loss=fit.score, commit=commit,
                const_changes=const_changes, func_changes=func_changes,
                editor_summary=editor_summary,
                evaluator_reasoning=evaluation.reasoning,
            )
            final_result = "GOAL_REACHED"
            break

        if evaluation.status == STATUS_IMPROVED and not crashed:
            commit = git.commit_trial(trial, fit.score,
                                      metric_name=metric_name)
            best_metrics = metrics
            best_score = fit.score
            best_commit = commit
            success(f"Improvement committed as {git.short(commit)} "
                    f"(experiment(trial-{trial}): "
                    f"{metric_name}={fit.score:.4f}).")
            logger.record_trial(
                trial=trial, commit=commit, status=evaluation.status,
                action="COMMITTED", metrics=metrics,
                evaluation=evaluation.to_dict(),
                editor_summary=editor_summary,
                runtime_seconds=result.duration_seconds, score=fit.score,
                referee_flags=trial_flag_dicts,
                editor_agent=seat_editor, evaluator_agent=seat_evaluator,
            )
            kg.record_trial(
                trial=trial, outcome=evaluation.status, action="COMMITTED",
                val_loss=fit.score, commit=commit,
                const_changes=const_changes, func_changes=func_changes,
                editor_summary=editor_summary,
                evaluator_reasoning=evaluation.reasoning,
            )
            pending_feedback = format_feedback(evaluation)
            pending_diagnostic = ""
            stagnation = 0
            continue

        # REGRESSED / CRASHED (or IMPROVED claimed on a crash) -> revert
        try:
            git.discard_changes()
            warn(f"Changes reverted; workspace restored to "
                 f"{git.short(git.head_commit())}.")
        except GitError as exc:
            error(f"Revert failed: {exc}. Attempting hard rollback to best "
                  f"commit {git.short(best_commit)}.")
            if best_commit:
                git.rollback_to(best_commit)
        logger.record_trial(
            trial=trial, commit=None,
            status=STATUS_CRASHED if crashed else evaluation.status,
            action="REVERTED", metrics=metrics,
            evaluation=evaluation.to_dict(),
            editor_summary=editor_summary,
            error_type=result.error_type,
            runtime_seconds=result.duration_seconds, score=fit.score,
            referee_flags=trial_flag_dicts,
            editor_agent=seat_editor, evaluator_agent=seat_evaluator,
        )
        # The failed change is reverted on disk, but it lives on in the
        # knowledge graph as a dead-end fact so no future session (even a
        # post-rotation one) repeats it.
        kg.record_trial(
            trial=trial,
            outcome=STATUS_CRASHED if crashed else evaluation.status,
            action="REVERTED", val_loss=fit.score,
            commit=None, const_changes=const_changes,
            func_changes=func_changes, editor_summary=editor_summary,
            evaluator_reasoning=evaluation.reasoning,
        )
        graph_state["current"] = CodeGraph.build(workdir)  # reverted state
        graph_state["current"].save(code_graph_path)
        pending_feedback = format_feedback(evaluation)
        if all_flags:
            pending_feedback += ("\n\n## Referee flags on your trial\n"
                                 + Referee.render(all_flags))
        pending_diagnostic = crash_diagnostic
        stagnation += 1
        if crashed:
            pending_feedback += (
                "\n\nNOTE: your previous edit was REVERTED because the run "
                "crashed — the workspace is back at the last good state. "
                "Try a DIFFERENT approach; do not repeat the same edit."
            )
        else:
            pending_feedback += (
                "\n\nNOTE: your previous edit was REVERTED because it "
                "regressed performance. Try an orthogonal approach."
            )

    # ---- Wrap-up -----------------------------------------------------------
    phase("Session complete")
    best = logger.best_trial()
    logger.finish_session(final_result, best["trial"] if best else None)
    print_trial_table(logger.trials)
    seats_seen = {t["editor_agent"] for t in logger.trials
                  if t.get("editor_agent")}
    if seats_seen:
        counts = Counter(
            (t["editor_agent"], t.get("action")) for t in logger.trials
            if t.get("editor_agent"))
        info("Editor-seat record:")
        for a in sorted(seats_seen):
            commits = sum(v for (ag, act), v in counts.items()
                          if ag == a and act in ("COMMITTED", "GOAL_COMMIT"))
            reverts = sum(v for (ag, act), v in counts.items()
                          if ag == a and act in ("REVERTED",))
            info(f"  {a}: {commits} committed, {reverts} reverted")
    if best:
        best_obj = best.get("score") if best.get("score") is not None \
            else best.get("val_loss")
        success(f"Best result: trial {best['trial']} with "
                f"{metric_name}={best_obj:.4f} "
                f"(commit {(best.get('commit') or 'n/a')[:8]}).")
    else:
        warn("No successful trial produced usable metrics.")
    report = logger.write_report(workdir / "experiment_report.md")
    info(f"Summary report: {report}")
    info("Full history: experiments_history.json")

    if final_result == "GOAL_REACHED":
        success("Experiment goal reached. 🎯")
        return 0
    warn(f"Loop ended: {final_result}.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
