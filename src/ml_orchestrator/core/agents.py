"""CLI bridges to the two agents.

- ClaudeEditor         : wraps ``claude -p "<prompt>"`` — modifies source
                         files (model architecture, training loop,
                         hyperparameters).
- AntigravityEvaluator : wraps ``agy -p "<prompt>"`` (Google Antigravity
                         CLI) — analyzes metrics.json, logs and history,
                         and must answer in a strict schema:

      STATUS: [GOAL_REACHED | IMPROVED | REGRESSED | CRASHED]
      REASONING: <analysis of loss and dynamics>
      RECOMMENDATIONS: <actionable suggestions for next edits>

Both bridges are plain subprocess wrappers so they work with whatever
authenticated CLI binaries are on PATH; command templates are overridable
for custom flags or model selection.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

STATUS_GOAL_REACHED = "GOAL_REACHED"
STATUS_IMPROVED = "IMPROVED"
STATUS_REGRESSED = "REGRESSED"
STATUS_CRASHED = "CRASHED"
STATUS_UNKNOWN = "UNKNOWN"

VALID_STATUSES = (
    STATUS_GOAL_REACHED,
    STATUS_IMPROVED,
    STATUS_REGRESSED,
    STATUS_CRASHED,
)


class AgentError(RuntimeError):
    """Raised when an agent CLI cannot be executed or returns garbage."""


@dataclass
class EvaluationResult:
    status: str
    reasoning: str
    recommendations: str
    raw: str = ""
    source: str = "antigravity"  # "antigravity" or "heuristic-fallback"

    def to_dict(self) -> Dict[str, str]:
        return {
            "status": self.status,
            "reasoning": self.reasoning,
            "recommendations": self.recommendations,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# Shared CLI invocation
# --------------------------------------------------------------------------

class _CLIAgent:
    def __init__(
        self,
        base_command: Sequence[str],
        cwd: Optional[Path] = None,
        timeout: float = 600.0,
        session: Optional[Any] = None,
        last_message_flag: Optional[str] = None,
    ) -> None:
        self.base_command = list(base_command)
        self.cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self.timeout = timeout
        # Optional core.session.ManagedSession: when set, calls run inside
        # a persistent, resumable conversation with context tracking and
        # automatic memory handoff instead of stateless one-shot processes.
        self.session = session
        # CLIs whose stdout mixes event noise with the answer (codex)
        # write the final message to a temp file via this flag instead.
        self.last_message_flag = last_message_flag

    def _invoke(self, prompt: str) -> str:
        if self.session is not None:
            return self.session.send(prompt)
        return self._invoke_stateless(prompt)

    def _invoke_stateless(self, prompt: str) -> str:
        capture_path: Optional[str] = None
        command = list(self.base_command)
        if self.last_message_flag:
            fd, capture_path = tempfile.mkstemp(suffix=".lastmsg.txt")
            os.close(fd)
            command += [self.last_message_flag, capture_path]
        command.append(prompt)
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise AgentError(
                f"Agent CLI {self.base_command[0]!r} not found on PATH. "
                "Install it and make sure you are logged in."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentError(
                f"Agent CLI {self.base_command[0]!r} timed out after "
                f"{self.timeout:.0f}s."
            ) from exc
        finally:
            captured = ""
            if capture_path:
                try:
                    captured = Path(capture_path).read_text(
                        encoding="utf-8").strip()
                    Path(capture_path).unlink(missing_ok=True)
                except OSError:
                    captured = ""

        if proc.returncode != 0:
            raise AgentError(
                f"Agent CLI {self.base_command[0]!r} exited with code "
                f"{proc.returncode}:\n{(proc.stderr or proc.stdout).strip()[:2000]}"
            )
        text = captured or proc.stdout.strip()
        if not text and proc.stderr.strip():
            # e.g. agy headless: tool permission auto-denied -> empty
            # stdout, explanation on stderr. Surface it, don't swallow it.
            raise AgentError(
                f"Agent CLI {self.base_command[0]!r} produced no output; "
                f"stderr: {proc.stderr.strip()[:600]}"
            )
        return text


# --------------------------------------------------------------------------
# Claude — Model & Experiment Editor
# --------------------------------------------------------------------------

class ClaudeEditor(_CLIAgent):
    """Drives Claude Code in non-interactive mode to edit local source files."""

    DEFAULT_COMMAND = ["claude", "--permission-mode", "acceptEdits", "-p"]

    def __init__(
        self,
        cwd: Path,
        command: Optional[Sequence[str]] = None,
        timeout: float = 900.0,
        session: Optional[Any] = None,
        last_message_flag: Optional[str] = None,
    ) -> None:
        super().__init__(command or self.DEFAULT_COMMAND, cwd=cwd,
                         timeout=timeout, session=session,
                         last_message_flag=last_message_flag)

    def build_prompt(
        self,
        goal: str,
        trial: int,
        train_script: str,
        editable_files: Sequence[str],
        feedback: str = "",
        diagnostic: str = "",
        history_summary: str = "",
        knowledge_context: str = "",
        mode: str = "ml",
        eval_command: str = "",
    ) -> str:
        if mode == "coding":
            role = (
                "You are the Code Editor in an automated development loop. "
                "Your ONLY job in this turn is to edit the local source "
                "files so the evaluation command gets closer to fully "
                "passing."
            )
            constraints = [
                "\n## Hard constraints",
                f"- The objective is measured by running: `{eval_command}`. "
                "Your edits must reduce its failures / make it pass.",
                "- Do NOT modify test files, test fixtures, or the "
                "evaluation command itself unless the goal explicitly asks "
                "for it — fixing the code, not the test, is the job.",
                "- Edit files in place. Do NOT create new entrypoints or "
                "rename files.",
                "- Do not install packages or touch git; the orchestrator "
                "handles versioning.",
                "- Make ONE focused, hypothesis-driven change set per trial "
                "rather than many unrelated edits.",
            ]
        else:
            role = (
                "You are the Model & Experiment Editor in an automated ML "
                "experimentation loop. Your ONLY job in this turn is to "
                "edit the local source files to improve the next training "
                "run."
            )
            constraints = [
                "\n## Hard constraints",
                "- Edit files in place. Do NOT create new entrypoints or "
                "rename files.",
                f"- `{train_script}` must remain directly runnable via "
                f"`python {train_script}` with no new CLI arguments "
                "required.",
                "- The script MUST keep writing progress to `metrics.json` "
                "in the working directory with at least: "
                '{"epoch": N, "train_loss": X, "val_loss": Y, "status": '
                '"COMPLETED"}.',
                "- Do not install packages or touch git; the orchestrator "
                "handles versioning.",
                "- Make ONE focused, hypothesis-driven change set per "
                "trial (e.g. adjust LR schedule, change capacity, add "
                "regularization) rather than many unrelated edits.",
            ]
        if editable_files:
            files_section = ("\n## Files you may edit\n"
                             + "\n".join(f"- {f}" for f in editable_files))
        else:
            files_section = (
                "\n## Files you may edit\nAny source file in this "
                "workspace — EXCEPT test files, test fixtures, and the "
                "evaluation harness itself (a referee inspects every diff "
                "and reverts violations)."
            )
        sections = [
            role,
            "",
            f"## Target goal\n{goal}",
            f"\n## Current trial\nTrial {trial}.",
            files_section,
            *constraints,
        ]
        if knowledge_context:
            sections.append(
                "\n" + knowledge_context.strip() + "\n"
                "\nUse the knowledge above instead of re-reading files to "
                "answer structural questions; never repeat a change listed "
                "as a dead end."
            )
        if history_summary:
            sections.append(f"\n## Experiment history so far\n{history_summary}")
        if feedback:
            sections.append(
                f"\n## Evaluator feedback from the previous trial\n{feedback}"
            )
        if diagnostic:
            sections.append(
                "\n## The previous run FAILED — fix this first\n" + diagnostic
            )
        sections.append(
            "\n## Output\nAfter editing, reply with a short summary (max 5 "
            "lines) of exactly what you changed and the hypothesis behind it."
        )
        return "\n".join(sections)

    def request_edit(self, **prompt_kwargs: Any) -> str:
        """Ask Claude to modify the code. Returns its change summary text."""
        prompt = self.build_prompt(**prompt_kwargs)
        return self._invoke(prompt)


# --------------------------------------------------------------------------
# Antigravity — Research & Evaluation Specialist
# --------------------------------------------------------------------------

class AntigravityEvaluator(_CLIAgent):
    """Drives the Google Antigravity CLI (``agy``) in non-interactive
    print mode to produce structured scientific feedback."""

    DEFAULT_COMMAND = ["agy", "-p"]

    def __init__(
        self,
        cwd: Path,
        command: Optional[Sequence[str]] = None,
        timeout: float = 300.0,
        session: Optional[Any] = None,
        last_message_flag: Optional[str] = None,
    ) -> None:
        super().__init__(command or self.DEFAULT_COMMAND, cwd=cwd,
                         timeout=timeout, session=session,
                         last_message_flag=last_message_flag)

    def build_prompt(
        self,
        goal: str,
        trial: int,
        metrics: Optional[Dict[str, Any]],
        best_metrics: Optional[Dict[str, Any]],
        history_summary: str,
        log_tail: str,
        crash_diagnostic: str = "",
        change_summary: str = "",
        knowledge_context: str = "",
        mode: str = "ml",
        eval_command: str = "",
    ) -> str:
        metrics_json = json.dumps(metrics, indent=2) if metrics else "null"
        best_json = json.dumps(best_metrics, indent=2) if best_metrics else "null"
        crashed = bool(crash_diagnostic)
        if mode == "coding":
            analysis_focus = (
                "Analyze the latest evaluation run: which tests/checks "
                f"pass or fail (command: `{eval_command}`), what the "
                "failure messages imply about root causes, and whether "
                "the trend across trials is converging toward a fully "
                "passing suite."
            )
            goal_rule = ("the evaluation command fully passes (0 failures) "
                         "or the stated objective is otherwise objectively "
                         "satisfied")
        else:
            analysis_focus = (
                "Analyze the latest training run scientifically: loss "
                "curves, convergence rate, train/val gap (overfitting), "
                "variance/instability, and GPU memory behavior if reported."
            )
            goal_rule = ("the stated target goal is objectively satisfied "
                         "by the latest metrics (and not by an obviously "
                         "divergent/overfit fluke)")
        changes_block = (
            f"\n## Exact code changes made this trial (AST diff)\n"
            f"{change_summary.strip()}\n" if change_summary.strip() else ""
        )
        knowledge_block = (
            f"\n{knowledge_context.strip()}\n" if knowledge_context.strip() else ""
        )

        return f"""You are the Research & Evaluation Specialist in an automated experimentation loop.
{analysis_focus}

## Target goal
{goal}

## Trial
{trial}

## Latest run metrics (metrics.json)
```json
{metrics_json}
```

## Best metrics achieved so far (for comparison)
```json
{best_json}
```
{changes_block}{knowledge_block}
## Experiment history
{history_summary or "No previous trials."}

## Run crashed?
{"YES — diagnostic below." if crashed else "No, the run completed."}
{crash_diagnostic}

## Terminal log tail
```
{log_tail.strip()[:4000] if log_tail else "(empty)"}
```

## Decision rules
- GOAL_REACHED : {goal_rule}.
- IMPROVED     : the latest objective score is better than the best so far (or clearly better dynamics), but the goal is not yet met.
- REGRESSED    : run completed but performance is worse than / not better than the best so far.
- CRASHED      : the run failed to complete / produced no measurable result.

## Response format — respond with EXACTLY these three lines and nothing else
STATUS: [GOAL_REACHED | IMPROVED | REGRESSED | CRASHED]
REASONING: <one dense paragraph analyzing loss levels, convergence, train/val gap, stability, and memory>
RECOMMENDATIONS: <numbered, concrete, code-level suggestions for the next edit>
"""

    def evaluate(self, **prompt_kwargs: Any) -> EvaluationResult:
        prompt = self.build_prompt(**prompt_kwargs)
        raw = self._invoke(prompt)
        return self.parse_response(raw)

    # -- parsing ------------------------------------------------------------

    @staticmethod
    def parse_response(raw: str) -> EvaluationResult:
        """Parse the strict STATUS/REASONING/RECOMMENDATIONS schema.

        Tolerates markdown bolding, brackets, and stray whitespace. Raises
        AgentError only if no STATUS token can be found at all.
        """
        text = raw.replace("**", "").replace("__", "")

        status_match = re.search(
            r"STATUS\s*:\s*\[?\s*(GOAL_REACHED|IMPROVED|REGRESSED|CRASHED)\s*\]?",
            text,
            re.IGNORECASE,
        )
        if not status_match:
            # Last resort: a bare status token anywhere in the reply.
            bare = re.search(
                r"\b(GOAL_REACHED|IMPROVED|REGRESSED|CRASHED)\b", text
            )
            if not bare:
                raise AgentError(
                    "Evaluator reply did not contain a parseable STATUS "
                    f"line. Raw reply (truncated):\n{raw[:1000]}"
                )
            status_match = bare
        status = status_match.group(1).upper()

        def _section(name: str, stop_names: List[str]) -> str:
            stop = "|".join(stop_names + ["$"])
            m = re.search(
                rf"{name}\s*:\s*(.*?)(?=(?:{stop})\s*:|\Z)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            return m.group(1).strip() if m else ""

        reasoning = _section("REASONING", ["RECOMMENDATIONS"])
        recommendations = _section("RECOMMENDATIONS", ["STATUS", "REASONING"])

        return EvaluationResult(
            status=status,
            reasoning=reasoning or "(no reasoning provided)",
            recommendations=recommendations or "(no recommendations provided)",
            raw=raw,
        )


# Backward-compat alias for code written against the original Gemini bridge.
GeminiEvaluator = AntigravityEvaluator


# --------------------------------------------------------------------------
# Heuristic fallback evaluator
# --------------------------------------------------------------------------

def heuristic_evaluation(
    goal_target: Optional[float],
    val_loss: Optional[float],
    best_val_loss: Optional[float],
    crashed: bool,
    error_type: Optional[str] = None,
    direction: str = "minimize",
    goal_op: str = "<",
) -> EvaluationResult:
    """Purely numeric verdict used when the Antigravity CLI is unavailable
    or returns an unparseable reply. Keeps the loop functional.

    ``val_loss``/``best_val_loss`` are the generic objective scores
    (names kept for backward compatibility); comparisons honor
    *direction* and *goal_op*.
    """
    score, best = val_loss, best_val_loss

    def better(a: float, b: float) -> bool:
        return a < b if direction == "minimize" else a > b

    def met(s: float, t: float) -> bool:
        return {"<": s < t, "<=": s <= t, ">": s > t, ">=": s >= t}[goal_op]

    if crashed or score is None:
        return EvaluationResult(
            status=STATUS_CRASHED,
            reasoning=(
                f"Run failed before producing a measurable objective "
                f"(error class: {error_type or 'unknown'})."
            ),
            recommendations=(
                "Fix the runtime failure described in the diagnostic before "
                "attempting further optimization."
            ),
            source="heuristic-fallback",
        )
    if goal_target is not None and met(score, goal_target):
        return EvaluationResult(
            status=STATUS_GOAL_REACHED,
            reasoning=(
                f"score={score:.4f} satisfies the numeric goal condition "
                f"(score {goal_op} {goal_target:.4f})."
            ),
            recommendations="Goal satisfied; no further edits required.",
            source="heuristic-fallback",
        )
    if best is None or better(score, best):
        return EvaluationResult(
            status=STATUS_IMPROVED,
            reasoning=(
                f"Objective improved to {score:.4f} "
                f"(previous best: "
                f"{'n/a' if best is None else f'{best:.4f}'})."
            ),
            recommendations=(
                "Continue in the same direction with a finer, focused "
                "adjustment next."
            ),
            source="heuristic-fallback",
        )
    return EvaluationResult(
        status=STATUS_REGRESSED,
        reasoning=(
            f"score={score:.4f} did not beat the best so far "
            f"({best:.4f})."
        ),
        recommendations=(
            "Revert conceptually and try an orthogonal change (different "
            "approach or subsystem)."
        ),
        source="heuristic-fallback",
    )


# Comparison words -> the operator they mean. Order matters in the regex
# alternation below: multi-character operators must precede their
# prefixes ('<=' before '<') or the longer form never matches.
_OP_WORDS: Dict[str, str] = {
    "<": "<", "<=": "<=", "under": "<", "below": "<", "less than": "<",
    ">": ">", ">=": ">=", "above": ">", "over": ">", "at least": ">=",
}

_MIN_OPS = r"<=|<|under|below|less than"
_ANY_OPS = r"<=|<|>=|>|under|below|above|over|at least|less than"
_NUM = r"([0-9]*\.?[0-9]+)"

_GOAL_PATTERNS = [
    rf"val(?:idation)?[\s_]*loss\s*({_MIN_OPS})\s*{_NUM}",
    rf"\bscore\s*({_ANY_OPS})\s*{_NUM}",
    rf"\bloss\s*({_MIN_OPS})\s*{_NUM}",
]


def parse_goal_condition(goal: str) -> Optional[Tuple[str, float]]:
    """Extract ``(operator, target)`` from a natural-language goal.

    Understands 'validation loss < 0.25', 'val_loss below 0.3', 'score
    >= 0.95', 'score at least 0.9'.

    The operator matters as much as the number: 'score >= 0.9' is a
    MAXIMIZE goal, and returning only 0.9 (as this function's
    predecessor did) left the caller comparing with the preset's '<',
    silently declaring success at the worst possible score. Callers
    should derive both the direction and the comparison from this pair.
    """
    for pattern in _GOAL_PATTERNS:
        m = re.search(pattern, goal, re.IGNORECASE)
        if not m:
            continue
        op = _OP_WORDS.get(m.group(1).lower().strip())
        if op is None:
            continue
        try:
            return op, float(m.group(2))
        except ValueError:
            continue
    return None


def parse_goal_target(goal: str) -> Optional[float]:
    """Numeric target only — kept for callers that don't need the operator."""
    parsed = parse_goal_condition(goal)
    return parsed[1] if parsed else None
