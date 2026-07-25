"""Live per-agent status board for parallel tournament stages.

While proposals/judging run concurrently, the terminal shows one line
per agent with a spinner, state, a ticking elapsed clock, and a result
detail — instead of a silent multi-minute gap:

    Tournament — proposals
      ⠹ claude       running   0:47
      ✔ antigravity  done      0:31  1,480 chars
      ✘ codex        failed    0:02  exited 1: Error loading config...

Implementation notes:
- Rich `Live` is used only on a real terminal; on pipes/CI (and when
  rich is unavailable) it degrades to plain state-transition prints, so
  logs stay readable and e2e output stays greppable.
- The board is thread-safe: worker threads call `update()`, a small
  refresher thread re-renders 4×/s so the clocks tick.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

WAITING = "waiting"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

_ICONS = {WAITING: "…", DONE: "✔", FAILED: "✘"}
_STYLES = {WAITING: "dim", RUNNING: "yellow", DONE: "green", FAILED: "red"}


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


class LiveStatus:
    """Context manager showing live parallel-agent progress."""

    def __init__(self, title: str, agents: List[str]) -> None:
        self.title = title
        self.agents = list(agents)
        self._lock = threading.Lock()
        now = time.monotonic()
        # agent -> (state, detail, started_at, finished_at)
        self._state: Dict[str, Tuple[str, str, float, Optional[float]]] = {
            a: (WAITING, "", now, None) for a in self.agents
        }
        self._live = None
        self._ticker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._console = None
        self._plain = True

    # -- worker-facing API ---------------------------------------------------

    def update(self, agent: str, state: str, detail: str = "") -> None:
        now = time.monotonic()
        with self._lock:
            prev = self._state.get(agent)
            started = prev[2] if prev else now
            if state == RUNNING and (prev is None or prev[0] == WAITING):
                started = now
            finished = now if state in (DONE, FAILED) else None
            self._state[agent] = (state, detail[:80], started, finished)
        if self._plain and state in (DONE, FAILED):
            elapsed = _fmt_elapsed(now - started)
            icon = _ICONS.get(state, "•")
            print(f"    {icon} {agent}: {state} ({elapsed})"
                  + (f" — {detail[:100]}" if detail else ""))

    # -- rendering ---------------------------------------------------------------

    def _render(self):
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(padding=(0, 2))
        table.add_column()
        table.add_column()
        table.add_column(justify="right")
        table.add_column()
        now = time.monotonic()
        frame = _SPINNER[int(now * 8) % len(_SPINNER)]
        with self._lock:
            rows = [(a, *self._state[a]) for a in self.agents]
        out = Table.grid()
        out.add_row(Text(f"  {self.title}", style="bold cyan"))
        for agent, state, detail, started, finished in rows:
            icon = frame if state == RUNNING else _ICONS.get(state, "•")
            elapsed = _fmt_elapsed((finished or now) - started)
            table_row = Text.assemble(
                (f"  {icon} ", _STYLES.get(state, "")),
                (f"{agent:<14}", "bold"),
                (f"{state:<9}", _STYLES.get(state, "")),
                (f"{elapsed:>6}  ", "dim"),
                (detail, "dim"),
            )
            out.add_row(table_row)
        return out

    # -- lifecycle -----------------------------------------------------------------

    def __enter__(self) -> "LiveStatus":
        try:
            from rich.console import Console
            from rich.live import Live

            console = Console()
            if console.is_terminal:
                self._console = console
                self._live = Live(self._render(), console=console,
                                  refresh_per_second=8, transient=False)
                self._live.__enter__()
                self._plain = False

                def _tick() -> None:
                    while not self._stop.wait(0.25):
                        try:
                            self._live.update(self._render())
                        except Exception:
                            return

                self._ticker = threading.Thread(target=_tick, daemon=True)
                self._ticker.start()
        except ImportError:
            pass
        if self._plain:
            print(f"    {self.title}:")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._stop.set()
            if self._ticker:
                self._ticker.join(timeout=1)
            try:
                self._live.update(self._render())
                self._live.__exit__(exc_type, exc, tb)
            except Exception:
                pass


class NullStatus:
    """No-op stand-in (tests, embedding)."""

    def __init__(self, title: str, agents: List[str]) -> None:
        pass

    def update(self, agent: str, state: str, detail: str = "") -> None:
        pass

    def __enter__(self) -> "NullStatus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass
