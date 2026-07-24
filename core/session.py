"""Persistent agent sessions, context-budget tracking, and memory handoff.

Problem this solves
-------------------
Spawning a fresh ``claude -p`` / ``agy -p`` process per trial gives the
agents amnesia — every trial they re-discover the codebase from scratch.
But simply keeping one session forever is also wrong: model quality
degrades once the context window fills past a point (for Claude Code on
a 1M-context model, roughly the 50% mark).

So each agent gets a *managed session*:

1. **Continuity** — calls resume the same CLI conversation
   (``claude --resume <session-id>`` / ``agy -c``), so the agent keeps
   its working memory of the code, past edits and reasoning.
2. **Context ledger** — real token usage is read from Claude's
   ``--output-format json`` responses (input + cache + output tokens);
   for CLIs that report nothing (agy), a chars/4 estimator is used.
3. **Rotation with smart memory** — when the ledger crosses the
   configured threshold (default 50% of the context limit), the session
   is asked to distill everything it knows into a structured MEMORY
   SNAPSHOT (goal state, codebase map, experiment ledger, failure
   modes, ranked next hypotheses). The snapshot is persisted to disk,
   the session is closed, and the next call opens a fresh session
   seeded with that snapshot.
4. **Cross-run persistence** — session ids, ledgers and snapshots are
   stored in the memory directory, so even restarting the orchestrator
   resumes where the agents left off.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents import AgentError

# Sentinel session id for CLIs (agy) that can only "continue latest"
# rather than address a conversation by id.
LATEST = "@latest"


def estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Memory store — the durable "brain" that survives session rotations
# ---------------------------------------------------------------------------

class MemoryStore:
    """File-backed memory: one current snapshot per agent + full archive.

    Layout inside the memory directory:
        <agent>_memory.md            current distilled snapshot (markdown)
        <agent>_memory_archive.jsonl every snapshot ever taken, with metadata
        sessions.json                live session ids + token ledgers
    """

    MAX_SNAPSHOT_CHARS = 24_000

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.dir / "sessions.json"
        # Optional zero-arg callable returning live, machine-derived
        # context (code map + experiment knowledge graph). Called at
        # send-time so fresh sessions always see the CURRENT state, not
        # a stale copy baked into a snapshot.
        self.context_provider = None

    # -- snapshots ---------------------------------------------------------

    def snapshot_path(self, agent: str) -> Path:
        return self.dir / f"{agent}_memory.md"

    def save_snapshot(
        self, agent: str, text: str, reason: str, context_tokens: int
    ) -> None:
        text = text.strip()[: self.MAX_SNAPSHOT_CHARS]
        if not text:
            return
        self.snapshot_path(agent).write_text(text + "\n", encoding="utf-8")
        archive = self.dir / f"{agent}_memory_archive.jsonl"
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": agent,
            "reason": reason,
            "context_tokens_at_rotation": context_tokens,
            "snapshot": text,
        }
        with archive.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def load_snapshot(self, agent: str) -> Optional[str]:
        path = self.snapshot_path(agent)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8").strip()
        return content or None

    def render_preamble(self, agent: str) -> str:
        """Context block injected into the first message of a new session.

        Two layers: the predecessor's distilled snapshot (lossy, rich in
        intent) and the live machine-derived knowledge (lossless facts:
        code map + experiment graph) from ``context_provider``.
        """
        sections: List[str] = []
        snapshot = self.load_snapshot(agent)
        if snapshot:
            sections.append(
                "## RESTORED MEMORY (from your previous session)\n"
                "You are the continuation of an earlier agent session that "
                "was closed to keep the context window healthy. Everything "
                "below was distilled by your predecessor — treat it as "
                "ground truth you already know. Do NOT re-explore or "
                "re-derive what is already mapped here; build on it.\n\n"
                "<memory>\n"
                f"{snapshot}\n"
                "</memory>"
            )
        if self.context_provider is not None:
            try:
                live = self.context_provider()
            except Exception:  # a broken provider must never block the loop
                live = ""
            if live and live.strip():
                sections.append(
                    "## LIVE PROJECT KNOWLEDGE (auto-generated, current "
                    "as of this message)\n" + live.strip()
                )
        if not sections:
            return ""
        return "\n\n".join(sections) + "\n\n---\n\n"

    # -- live session state -------------------------------------------------

    def load_state(self, agent: str) -> Dict[str, Any]:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return data.get(agent, {})
        except (OSError, json.JSONDecodeError, ValueError):
            return {}

    def save_state(self, agent: str, state: Dict[str, Any]) -> None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError, ValueError):
            data = {}
        data[agent] = state
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._state_path)


# ---------------------------------------------------------------------------
# CLI adapters — how each tool starts/resumes a conversation
# ---------------------------------------------------------------------------

@dataclass
class ParsedReply:
    text: str
    session_id: Optional[str]
    context_tokens: Optional[int]  # None -> caller must estimate


class ClaudeSessionAdapter:
    """claude: '--output-format json' yields session_id + real token usage;
    '--resume <id>' continues that conversation in print mode."""

    def __init__(self, base_command: Sequence[str]) -> None:
        # base command conventionally ends with -p; the prompt goes last.
        self.base = list(base_command)

    def command(self, prompt: str, session_id: Optional[str]) -> List[str]:
        base = list(self.base)
        idx = base.index("-p") if "-p" in base else len(base)
        pre, post = base[:idx], base[idx:] or ["-p"]
        extra: List[str] = []
        if "--output-format" not in pre:
            extra += ["--output-format", "json"]
        if session_id and session_id != LATEST:
            extra += ["--resume", session_id]
        return pre + extra + post + [prompt]

    @staticmethod
    def parse(stdout: str) -> ParsedReply:
        try:
            obj = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return ParsedReply(text=stdout.strip(), session_id=None,
                               context_tokens=None)
        if not isinstance(obj, dict):
            return ParsedReply(text=stdout.strip(), session_id=None,
                               context_tokens=None)
        text = obj.get("result") or obj.get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text)
        usage = obj.get("usage") or {}
        tokens = None
        if isinstance(usage, dict):
            total = 0
            for key in ("input_tokens", "cache_creation_input_tokens",
                        "cache_read_input_tokens", "output_tokens"):
                v = usage.get(key)
                if isinstance(v, (int, float)):
                    total += int(v)
            tokens = total or None
        return ParsedReply(
            text=text.strip(),
            session_id=obj.get("session_id"),
            context_tokens=tokens,
        )


class AntigravitySessionAdapter:
    """agy: '-c' continues the most recent conversation in this workspace;
    '--conversation <id>' addresses a specific one when an id is known.
    Print mode reports no token usage, so the ledger falls back to
    estimation."""

    def __init__(self, base_command: Sequence[str]) -> None:
        self.base = list(base_command)

    def command(self, prompt: str, session_id: Optional[str]) -> List[str]:
        base = list(self.base)
        idx = base.index("-p") if "-p" in base else len(base)
        pre, post = base[:idx], base[idx:] or ["-p"]
        extra: List[str] = []
        if session_id == LATEST:
            extra = ["-c"]
        elif session_id:
            extra = ["--conversation", session_id]
        return pre + extra + post + [prompt]

    @staticmethod
    def parse(stdout: str) -> ParsedReply:
        # No structured metadata in plain print mode; the conversation is
        # addressed as "most recent" on subsequent calls.
        return ParsedReply(text=stdout.strip(), session_id=LATEST,
                           context_tokens=None)


# ---------------------------------------------------------------------------
# Managed session
# ---------------------------------------------------------------------------

def build_handoff_prompt(role_description: str) -> str:
    return f"""IMPORTANT — SESSION HANDOFF. This session is being closed because its context window is nearing the degradation threshold. A fresh session of you ({role_description}) will take over and will receive ONLY what you write now.

Write a MEMORY SNAPSHOT in markdown, maximum ~700 words, with exactly these sections:

# MEMORY SNAPSHOT
## Goal & current status
(the target objective, best metric achieved, how far from the goal)
## Codebase map
(each relevant file: its role, key functions/classes, tensor shapes / data dims / important constants)
## Experiment ledger
(compact table: trial | change made | outcome | val_loss | verdict)
## What works / what fails
(confirmed-good techniques vs. dead ends, each with the WHY in one clause)
## Constraints & gotchas
(hard rules: metrics.json contract, files you may edit, environment limits, anything that burned you)
## Ranked next hypotheses
(numbered, most promising first, one line each)

Rules: be dense and factual — no prose padding, no apologies, no code blocks longer than 5 lines. Do NOT edit any files in this turn. Output ONLY the snapshot."""


class ManagedSession:
    """A resumable CLI conversation with a context ledger and auto-handoff."""

    def __init__(
        self,
        agent_name: str,
        role_description: str,
        adapter: Any,
        memory: MemoryStore,
        cwd: Path,
        context_limit: int = 1_000_000,
        rotate_at: float = 0.5,
        timeout: float = 900.0,
    ) -> None:
        self.agent_name = agent_name
        self.role_description = role_description
        self.adapter = adapter
        self.memory = memory
        self.cwd = Path(cwd).resolve()
        self.context_limit = max(1, int(context_limit))
        self.rotate_at = min(max(rotate_at, 0.05), 0.95)
        self.timeout = timeout
        self.rotations = 0

        state = memory.load_state(agent_name)
        self.session_id: Optional[str] = state.get("session_id")
        self.context_tokens: int = int(state.get("context_tokens") or 0)

    # -- introspection ---------------------------------------------------------

    def context_fraction(self) -> float:
        return self.context_tokens / self.context_limit

    def should_rotate(self) -> bool:
        return (
            self.session_id is not None
            and self.context_fraction() >= self.rotate_at
        )

    def status_line(self) -> str:
        sid = self.session_id or "none"
        if sid not in (None, LATEST) and len(sid) > 12:
            sid = sid[:12]
        return (
            f"{self.agent_name}: {self.context_tokens:,} tok "
            f"({self.context_fraction():.0%} of {self.context_limit:,}), "
            f"session={sid}, rotations={self.rotations}"
        )

    # -- core I/O -----------------------------------------------------------------

    def send(self, prompt: str, inject_memory: bool = True) -> str:
        """Send *prompt* within the persistent conversation.

        A brand-new session (first ever call, or first call after a
        rotation/restart) gets the restored-memory preamble prepended.
        """
        full_prompt = prompt
        if self.session_id is None and inject_memory:
            preamble = self.memory.render_preamble(self.agent_name)
            if preamble:
                full_prompt = preamble + prompt

        try:
            reply = self._invoke(full_prompt, self.session_id)
        except AgentError:
            if self.session_id is None:
                raise
            # The stored conversation may have expired or been deleted —
            # fall back to a fresh session seeded with memory.
            self.session_id = None
            self.context_tokens = 0
            preamble = self.memory.render_preamble(self.agent_name)
            full_prompt = (preamble + prompt) if (preamble and inject_memory) \
                else prompt
            reply = self._invoke(full_prompt, None)

        if reply.session_id:
            self.session_id = reply.session_id
        elif self.session_id is None:
            self.session_id = LATEST

        if reply.context_tokens is not None:
            # Authoritative report of the conversation's current size.
            self.context_tokens = max(self.context_tokens,
                                      reply.context_tokens)
        else:
            self.context_tokens += (
                estimate_tokens(full_prompt) + estimate_tokens(reply.text)
            )
        self._persist()
        return reply.text

    def _invoke(self, prompt: str, session_id: Optional[str]) -> ParsedReply:
        command = self.adapter.command(prompt, session_id)
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
                f"Agent CLI {command[0]!r} not found on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentError(
                f"Agent CLI {command[0]!r} timed out after "
                f"{self.timeout:.0f}s."
            ) from exc
        if proc.returncode != 0:
            raise AgentError(
                f"Agent CLI {command[0]!r} exited with code "
                f"{proc.returncode}:\n"
                f"{(proc.stderr or proc.stdout).strip()[:2000]}"
            )
        return self.adapter.parse(proc.stdout)

    # -- rotation --------------------------------------------------------------------

    def rotate(self, reason: str = "") -> Optional[str]:
        """Distill the session into a memory snapshot, then close it.

        Returns the snapshot text (None if there was no session to close).
        The next send() starts fresh, seeded with the snapshot.
        """
        if self.session_id is None:
            return None
        tokens_at_rotation = self.context_tokens
        try:
            snapshot = self.send(
                build_handoff_prompt(self.role_description),
                inject_memory=False,
            )
        except AgentError:
            # Session too degraded/broken to summarize — keep the previous
            # snapshot (if any) rather than losing everything.
            snapshot = None
        if snapshot:
            self.memory.save_snapshot(
                self.agent_name, snapshot,
                reason or "context threshold reached",
                tokens_at_rotation,
            )
        self.session_id = None
        self.context_tokens = 0
        self.rotations += 1
        self._persist()
        return snapshot

    def maybe_rotate(self, reason: str = "") -> Optional[str]:
        if self.should_rotate():
            return self.rotate(reason)
        return None

    # -- persistence -------------------------------------------------------------------

    def _persist(self) -> None:
        self.memory.save_state(self.agent_name, {
            "session_id": self.session_id,
            "context_tokens": self.context_tokens,
            "context_limit": self.context_limit,
            "rotations": self.rotations,
        })
