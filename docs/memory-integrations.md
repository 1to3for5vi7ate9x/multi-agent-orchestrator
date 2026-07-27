# Going further: external graph-memory projects

The orchestrator ships its own code graph and temporal experiment
graph (see [`../README.md`](../README.md)), which need no external
services. This page is for going beyond them — semantic search,
multi-project graphs, cross-repo recall.

For even richer memory (semantic search, multi-project graphs), these
open-source projects slot in cleanly — most are MCP servers, which
Claude Code can use even in print mode after a one-time
`claude mcp add <name> ...`:

| Project | What it gives you |
|---------|-------------------|
| [`shaneholloman/mcp-knowledge-graph`](https://github.com/shaneholloman/mcp-knowledge-graph) | Local knowledge-graph memory MCP (entities/relations/observations) — the classic "remember across sessions" server. |
| [`DeusData/codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp) | Indexes the repo into a persistent code knowledge graph (tree-sitter, 158 languages) so structural questions cost ~99% fewer tokens than re-reading files. |
| [`getzep/graphiti`](https://github.com/getzep/graphiti) | Temporal knowledge graphs for agents — facts carry validity intervals, ideal for "what worked, then stopped working" experiment history. |
| [`topoteretes/cognee`](https://github.com/topoteretes/cognee) | Pipeline that turns documents/history into a queryable semantic graph ("memory engine") with a few lines of Python. |
| [`mem0ai/mem0`](https://github.com/mem0ai/mem0) | Lightweight self-improving memory layer with an MCP server; good for preference/feedback-style memories. |
| [`doobidoo/mcp-memory-service`](https://github.com/doobidoo/mcp-memory-service) | Semantic memory MCP with time-based recall and tagging. |

Integration points: (1) register a memory MCP server with Claude Code
and tell the Editor (via `--claude-cmd` extra flags or CLAUDE.md) to
store/query it; (2) replace `core/session.py::MemoryStore` with an
adapter that writes snapshots into one of these graphs instead of
markdown — the `ManagedSession` API (`save_snapshot` /
`render_preamble` / `load_state` / `save_state`) is the only contract.

