"""Isolated command execution harness.

Runs training scripts (or any subprocess) with:
- hard timeout enforcement (kills the whole process group),
- live streaming of stdout/stderr to the terminal while capturing both,
- automatic classification of common ML runtime failures (CUDA OOM,
  tensor shape mismatches, syntax errors, missing modules, NaN losses),
- generation of actionable diagnostic prompts for the Editor agent.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

# (error_type, [regex patterns], actionable hint for the Editor agent)
ERROR_PATTERNS: List[Tuple[str, List[str], str]] = [
    (
        "CUDA_OOM",
        [
            r"CUDA out of memory",
            r"torch\.cuda\.OutOfMemoryError",
            r"OutOfMemoryError",
            r"CUBLAS_STATUS_ALLOC_FAILED",
            r"CUDA error: out of memory",
            r"RESOURCE_EXHAUSTED",
            r"cudaErrorMemoryAllocation",
        ],
        "GPU ran out of memory. Reduce the batch size, shrink hidden layer "
        "sizes, enable gradient accumulation, use mixed precision "
        "(torch.cuda.amp / autocast), or move part of the pipeline to CPU.",
    ),
    (
        "SHAPE_MISMATCH",
        [
            r"mat1 and mat2 shapes cannot be multiplied",
            r"size mismatch",
            r"shapes .* not aligned",
            r"Expected input .* to have",
            r"The size of tensor a \(\d+\) must match the size of tensor b",
            r"stack expects each tensor to be equal size",
            r"Dimension out of range",
            r"expected .*\d+ channels.* but got",
            r"ValueError: operands could not be broadcast",
        ],
        "Tensor shape mismatch. Trace the tensor dimensions layer by layer: "
        "verify the model's input dimension matches the dataset feature "
        "count, that Linear/Conv layer in/out sizes chain correctly, and "
        "that target tensors match the output shape expected by the loss.",
    ),
    (
        "SYNTAX_ERROR",
        [
            r"SyntaxError",
            r"IndentationError",
            r"TabError",
        ],
        "The last code edit introduced invalid Python syntax. Re-open the "
        "file at the line indicated in the traceback and fix the syntax "
        "before making any further functional changes.",
    ),
    (
        "IMPORT_ERROR",
        [
            r"ModuleNotFoundError",
            r"ImportError",
        ],
        "A module failed to import. Either the dependency is not installed "
        "in this environment or the import path is wrong. Prefer rewriting "
        "the code to use already-available libraries instead of adding new "
        "dependencies.",
    ),
    (
        "NAN_LOSS",
        [
            r"(?i)loss[^\n]{0,40}\bnan\b",
            r"(?i)\bnan\b[^\n]{0,40}loss",
            r"Function '.*' returned nan",
            r"(?i)loss (?:is|became|=) ?inf",
        ],
        "Training produced NaN/Inf loss values (numerical divergence). "
        "Lower the learning rate, add gradient clipping "
        "(torch.nn.utils.clip_grad_norm_), check for log/div-by-zero in "
        "the loss, and verify input normalization.",
    ),
    (
        "FILE_NOT_FOUND",
        [
            r"FileNotFoundError",
            r"No such file or directory",
        ],
        "A required file or path is missing. Verify dataset paths and "
        "output directories exist (create them with os.makedirs(..., "
        "exist_ok=True)) and that relative paths resolve from the "
        "working directory.",
    ),
    (
        "GENERIC_EXCEPTION",
        [
            r"Traceback \(most recent call last\)",
        ],
        "An unhandled Python exception occurred. Read the traceback below, "
        "identify the failing line, and fix the root cause with a minimal, "
        "targeted change.",
    ),
]


@dataclass
class ExecutionResult:
    """Outcome of a single sandboxed subprocess run."""

    command: List[str]
    returncode: Optional[int]
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    error_type: Optional[str] = None
    error_hint: Optional[str] = None
    launch_error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and self.launch_error is None
        )

    @property
    def crashed(self) -> bool:
        return not self.succeeded

    def combined_output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout)
        if self.stderr.strip():
            parts.append(self.stderr)
        return "\n".join(parts)

    def tail(self, max_chars: int = 4000) -> str:
        """Last portion of combined output — the part with the traceback."""
        out = self.combined_output()
        return out[-max_chars:] if len(out) > max_chars else out


class ExecutionHarness:
    """Runs subprocesses with timeout, live log capture and error triage."""

    def __init__(
        self,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        echo: bool = True,
        echo_prefix: str = "  │ ",
        max_captured_chars: int = 400_000,
    ) -> None:
        self.cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self.env = {**os.environ, **(env or {})}
        self.echo = echo
        self.echo_prefix = echo_prefix
        self.max_captured_chars = max_captured_chars

    # -- public API --------------------------------------------------------

    def run(
        self,
        command: Sequence[str],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute *command*, streaming output live, enforcing *timeout*."""
        command = list(command)
        start = time.monotonic()

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(self.cwd),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                errors="replace",
                # New session so a timeout kill reaps the whole tree
                # (DataLoader workers, etc.) on POSIX.
                start_new_session=(os.name == "posix"),
            )
        except (OSError, ValueError) as exc:
            return ExecutionResult(
                command=command,
                returncode=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                launch_error=f"Failed to launch {command[0]!r}: {exc}",
                error_type="LAUNCH_FAILURE",
                error_hint=(
                    f"The command {command[0]!r} could not be started. "
                    "Verify it exists, is executable, and is on PATH."
                ),
            )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        readers = [
            threading.Thread(
                target=self._pump,
                args=(proc.stdout, stdout_lines, sys.stdout),
                daemon=True,
            ),
            threading.Thread(
                target=self._pump,
                args=(proc.stderr, stderr_lines, sys.stderr),
                daemon=True,
            ),
        ]
        for r in readers:
            r.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(proc)
        finally:
            for r in readers:
                r.join(timeout=5)

        duration = time.monotonic() - start
        stdout = self._clip("".join(stdout_lines))
        stderr = self._clip("".join(stderr_lines))

        result = ExecutionResult(
            command=command,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=timed_out,
        )

        if timed_out:
            result.error_type = "TIMEOUT"
            result.error_hint = (
                f"Training exceeded the {timeout:.0f}s time limit and was "
                "killed. Reduce epochs, dataset size, or model size so a "
                "full run fits inside the limit — or make the script "
                "checkpoint and emit metrics incrementally."
            )
        elif result.returncode != 0:
            etype, hint = self.classify_error(stdout + "\n" + stderr)
            result.error_type = etype or "NONZERO_EXIT"
            result.error_hint = hint or (
                f"Process exited with code {result.returncode} but no known "
                "error signature was found. Inspect the log tail for clues."
            )
        return result

    @staticmethod
    def classify_error(output: str) -> Tuple[Optional[str], Optional[str]]:
        """Match *output* against known ML failure signatures.

        Returns (error_type, actionable_hint) or (None, None).
        Ordered: the first (most specific) matching category wins.
        """
        for error_type, patterns, hint in ERROR_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, output):
                    return error_type, hint
        return None, None

    @staticmethod
    def build_diagnostic_prompt(result: ExecutionResult) -> str:
        """Format a failed run into an actionable diagnostic for the Editor."""
        if result.succeeded:
            return ""
        lines = [
            "## Runtime Failure Diagnostic",
            f"- Command       : {' '.join(result.command)}",
            f"- Exit code     : {result.returncode}",
            f"- Duration      : {result.duration_seconds:.1f}s",
            f"- Timed out     : {result.timed_out}",
            f"- Error class   : {result.error_type or 'UNKNOWN'}",
        ]
        if result.launch_error:
            lines.append(f"- Launch error  : {result.launch_error}")
        if result.error_hint:
            lines += ["", "### Suggested fix direction", result.error_hint]
        tail = result.tail()
        if tail.strip():
            lines += [
                "",
                "### Log tail (stdout + stderr, most recent last)",
                "```",
                tail.strip(),
                "```",
            ]
        return "\n".join(lines)

    # -- internals ----------------------------------------------------------

    def _pump(self, pipe, sink: List[str], echo_stream) -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                sink.append(line)
                if self.echo:
                    try:
                        echo_stream.write(self.echo_prefix + line)
                        echo_stream.flush()
                    except (OSError, ValueError):
                        pass  # terminal went away; keep capturing
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Terminate the process and all of its children."""
        try:
            if os.name == "posix":
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(2)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
            else:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_captured_chars:
            return text
        half = self.max_captured_chars // 2
        return (
            text[:half]
            + f"\n... [{len(text) - self.max_captured_chars} chars truncated] ...\n"
            + text[-half:]
        )
