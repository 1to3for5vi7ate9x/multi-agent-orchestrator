"""Agent roster — the pool of CLI agents that can play any seat.

v0.4 hard-wired Claude as the Editor and Antigravity as the Evaluator.
v0.5 makes seats dynamic: every registered agent can play *editor*
(file-modifying invocation), *evaluator*, *proposer* or *judge*
(text-only invocation), and the tournament (see tournament.py) decides
who sits where.

Built-in vendors:
    claude       — Claude Code       edit: --permission-mode acceptEdits -p
    antigravity  — Google agy        edit: --mode accept-edits -p
    codex        — OpenAI Codex CLI  edit: exec -s workspace-write (stateless;
                                     final message captured via -o <file>)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .agents import _CLIAgent
from .session import AntigravitySessionAdapter, ClaudeSessionAdapter


class AskAgent:
    """Text-only Q&A wrapper for tournament proposals and judging.

    Deliberately stateless: proposals and judging must not leak between
    rounds through a session, and judges must see only what the prompt
    shows them.
    """

    def __init__(self, spec: "AgentSpec", cwd: Path, timeout: float) -> None:
        self.name = spec.name
        self._cli = _CLIAgent(
            spec.ask_cmd, cwd=cwd, timeout=timeout,
            last_message_flag=spec.last_message_flag,
        )

    def ask(self, prompt: str) -> str:
        return self._cli._invoke(prompt)


@dataclass
class AgentSpec:
    name: str
    binary: str
    edit_cmd: List[str]              # prompt appended last
    ask_cmd: List[str]               # text-only Q&A; prompt appended last
    # Flag used to capture the agent's final message into a temp file
    # (for CLIs whose stdout mixes event noise with the answer).
    last_message_flag: Optional[str] = None
    # Factory building a persistent-session adapter for a given base
    # command, or None when the CLI has no usable resume mechanism.
    session_adapter: Optional[Callable[[List[str]], object]] = None

    def available(self) -> bool:
        return shutil.which(self.binary) is not None


def default_roster() -> Dict[str, AgentSpec]:
    return {
        "claude": AgentSpec(
            name="claude",
            binary="claude",
            edit_cmd=["claude", "--permission-mode", "acceptEdits", "-p"],
            ask_cmd=["claude", "-p"],
            session_adapter=lambda base: ClaudeSessionAdapter(base),
        ),
        "antigravity": AgentSpec(
            name="antigravity",
            binary="agy",
            edit_cmd=["agy", "--mode", "accept-edits", "-p"],
            ask_cmd=["agy", "-p"],
            session_adapter=lambda base: AntigravitySessionAdapter(base),
        ),
        # --ignore-user-config: the Codex desktop app writes non-boolean
        # keys into [features] in ~/.codex/config.toml, which hard-fails
        # `codex exec` at config load. Auth still comes from CODEX_HOME.
        "codex": AgentSpec(
            name="codex",
            binary="codex",
            edit_cmd=["codex", "exec", "--ignore-user-config",
                      "-s", "workspace-write", "--skip-git-repo-check"],
            ask_cmd=["codex", "exec", "--ignore-user-config",
                     "-s", "read-only", "--skip-git-repo-check"],
            last_message_flag="-o",
            session_adapter=None,  # stateless in v1; resume support later
        ),
    }


def resolve_roster(
    requested: Optional[List[str]] = None,
    overrides: Optional[Dict[str, AgentSpec]] = None,
) -> Dict[str, AgentSpec]:
    """Return the available subset of the requested agents, in order.

    ``requested`` of None means "everything installed". Unknown names
    raise ValueError; unavailable ones are silently dropped (callers
    report what they got).
    """
    registry = default_roster()
    if overrides:
        registry.update(overrides)
    names = requested if requested is not None else list(registry)
    out: Dict[str, AgentSpec] = {}
    for name in names:
        if name not in registry:
            raise ValueError(
                f"Unknown agent {name!r}; known: {', '.join(sorted(registry))}"
            )
        spec = registry[name]
        if spec.available():
            out[name] = spec
    return out
