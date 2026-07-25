"""Core package for ml-agent-orchestrator.

Exposes the primary building blocks of the closed-loop experiment engine:

- ExecutionHarness / ExecutionResult  : sandboxed subprocess execution
- GitManager / GitError               : versioning, commits, rollback
- ClaudeEditor / AntigravityEvaluator : CLI bridges to the agent tools
- ExperimentLogger                    : persistent trial history + reports
"""

from .runner import ExecutionHarness, ExecutionResult
from .git_manager import GitManager, GitError
from .agents import (
    AgentError,
    ClaudeEditor,
    AntigravityEvaluator,
    GeminiEvaluator,  # backward-compat alias
    EvaluationResult,
    STATUS_GOAL_REACHED,
    STATUS_IMPROVED,
    STATUS_REGRESSED,
    STATUS_CRASHED,
    STATUS_UNKNOWN,
)
from .logger import ExperimentLogger
from .fitness import FitnessExtractor, FitnessResult, parse_test_results
from .roster import AgentSpec, AskAgent, default_roster, resolve_roster
from .tournament import TournamentResult, run_tournament
from .referee import Flag, Referee
from .code_graph import CodeGraph, format_changes
from .knowledge_graph import ExperimentKnowledgeGraph
from .session import (
    ManagedSession,
    MemoryStore,
    ClaudeSessionAdapter,
    AntigravitySessionAdapter,
)

__all__ = [
    "ExecutionHarness",
    "ExecutionResult",
    "GitManager",
    "GitError",
    "AgentError",
    "ClaudeEditor",
    "AntigravityEvaluator",
    "GeminiEvaluator",
    "EvaluationResult",
    "ExperimentLogger",
    "FitnessExtractor",
    "FitnessResult",
    "parse_test_results",
    "AgentSpec",
    "AskAgent",
    "default_roster",
    "resolve_roster",
    "TournamentResult",
    "run_tournament",
    "Flag",
    "Referee",
    "CodeGraph",
    "format_changes",
    "ExperimentKnowledgeGraph",
    "ManagedSession",
    "MemoryStore",
    "ClaudeSessionAdapter",
    "AntigravitySessionAdapter",
    "STATUS_GOAL_REACHED",
    "STATUS_IMPROVED",
    "STATUS_REGRESSED",
    "STATUS_CRASHED",
    "STATUS_UNKNOWN",
]
