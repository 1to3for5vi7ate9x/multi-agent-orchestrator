# ml-agent-orchestrator

A closed-loop, CLI-driven automated ML experiment engine.

- **Claude Code** acts as the **Model & Experiment Editor** — it edits your
  `model.py` / `train.py` (architecture, training loop, hyperparameters,
  preprocessing) directly on disk.
- **Google Antigravity CLI** (`agy`) acts as the **Research & Evaluation
  Specialist** — it reads `metrics.json`, terminal logs and the trial
  history, and returns a strict, parseable scientific verdict.
- **Git** is the state machine — improvements are committed automatically,
  regressions and crashes are reverted automatically.

```
        ┌────────────────────────────────────────────────────┐
        │                     Orchestrator                   │
        └────────────────────────────────────────────────────┘
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐
   │  Claude  │───▶│  Sandboxed   │───▶│ Antigravity  │───▶│   Git   │
   │  Editor  │    │ training run │    │  Evaluator   │    │ decide  │
   └──────────┘    │ + metrics.json│   └──────────────┘    └─────────┘
        ▲          └──────────────┘                             │
        │            feedback / crash diagnostics / history     │
        └───────────────────────────────────────────────────────┘
```

Per trial:
1. **Edit** — Claude modifies the editable source files toward the goal.
2. **Run** — the training script executes in a monitored subprocess with a
   hard timeout; CUDA OOM, shape mismatches, syntax errors, NaN losses etc.
   are auto-classified into actionable diagnostics.
3. **Evaluate** — Antigravity analyzes loss curves, convergence, train/val
   gap, stability and GPU memory, replying in a strict schema.
4. **Decide** —
   - `GOAL_REACHED` → commit, write the summary report, exit 0.
   - `IMPROVED` → `git commit` tagged `experiment(trial-N): val_loss=...`,
     continue.
   - `REGRESSED` / `CRASHED` → `git checkout` reverts the edit; the failure
     reasoning + traceback is fed back to the Editor for a different attempt.

Every trial is recorded in `experiments_history.json`; a markdown
`experiment_report.md` is generated when the session ends.

---

## Prerequisites

1. **[uv](https://docs.astral.sh/uv/)** and **git** on PATH. uv manages
   the Python version (pinned in `.python-version`), the virtualenv, and
   all dependencies from `pyproject.toml`/`uv.lock` — no manual venv or
   pip juggling:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # if not installed
   uv sync                                           # creates .venv + installs everything
   ```
2. **Claude Code CLI**, installed and logged in:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude          # complete login once, then exit
   ```
3. **Google Antigravity CLI** (`agy`), installed and logged in with your
   Google account (chosen over Gemini CLI, which doesn't support account
   login in this setup). Follow Google's official install guide
   (https://codelabs.developers.google.com/antigravity-cli-hands-on), then
   run `agy` once interactively to complete the account login.
   Verify both work non-interactively:
   ```bash
   claude -p "say ok"
   agy -p "say ok"
   ```
   > If your install exposes a different binary name, pass it via
   > `--evaluator-cmd "<binary> -p"`. If no evaluator CLI is found at all,
   > the orchestrator still runs using a built-in numeric fallback
   > evaluator (compares `val_loss` against the best so far) — you lose
   > the scientific reasoning, not the loop.
4. Optional: PyTorch for the example templates (they fall back to
   NumPy, then pure-stdlib, automatically):
   ```bash
   uv sync --extra torch
   ```
   (`requirements.txt` is kept only as a legacy pip fallback.)

## The metrics contract

Your training script must write a `metrics.json` in its working directory:

```json
{"epoch": 30, "train_loss": 0.041, "val_loss": 0.118, "status": "COMPLETED"}
```

Optional extra keys the evaluator will use if present: `history` (per-epoch
loss curve) and `gpu_mem_mb`. Write it every epoch (like the template does)
so partial progress survives timeouts.

## Quick start (self-contained demo)

```bash
# 1. Create an experiment workspace from the templates
mkdir -p ~/experiments/demo
cp templates/train_example.py ~/experiments/demo/train.py
cp templates/model_example.py ~/experiments/demo/model_example.py

# 2. Run the closed loop (--init-git creates the repo + baseline commit)
uv run ml-orchestrator \
  --goal "Achieve validation loss < 0.05 on the synthetic dataset without overfitting" \
  --workdir ~/experiments/demo \
  --train-script train.py \
  --max-trials 5 \
  --timeout 300 \
  --init-git
```

`uv run ml-orchestrator` is the installed console entry point;
`uv run main.py` works identically. The training subprocess uses the
uv-managed interpreter by default (override with `--python` to point at
a different env, e.g. a CUDA conda env).

## Using it on your own project

```bash
cd your-ml-project        # must be a git repo (or pass --init-git)
uv run --project /path/to/ml_orchestrator ml-orchestrator \
  --goal "Achieve validation loss < 0.25 without overfitting" \
  --train-script train.py \
  --max-trials 8 \
  --timeout 1800 \
  --editable-files train.py model.py data.py \
  --python "$(command -v python)"   # interpreter that has YOUR training deps
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | *(required)* | Natural-language target objective. A numeric target (e.g. "val loss < 0.25") is also parsed for the fallback evaluator. |
| `--max-trials` | `5` | Maximum edit→train→evaluate iterations. |
| `--train-script` | `train.py` | Script executed each trial via `--python`. |
| `--timeout` | `900` | Per-run training limit (seconds); the whole process tree is killed on expiry. |
| `--workdir` | `.` | Experiment workspace / git repo. |
| `--metrics-file` | `metrics.json` | Structured metrics file the script writes. |
| `--python` | current interpreter | Interpreter for the training script (point at your venv/conda env). |
| `--editable-files` | auto-detected | Whitelist of files the Editor may modify. |
| `--init-git` | off | Initialize a git repo + baseline commit if missing. |
| `--skip-baseline` | off | Skip the trial-0 baseline training run. |
| `--claude-cmd` | `claude --permission-mode acceptEdits -p` | Override the Editor command template (prompt appended last). |
| `--evaluator-cmd` | `agy -p` | Override the Evaluator command template (`--gemini-cmd` is kept as a deprecated alias). |
| `--agent-timeout` | `900` | Time limit per agent CLI call (seconds). |
| `--no-echo` | off | Don't stream training logs live to the terminal. |
| `--stateless` | off | Disable persistent sessions — every agent call becomes a fresh, memoryless process (pre-v2 behavior). |
| `--context-limit` | `1000000` | Model context window in tokens (Claude Code 1M-context model). |
| `--rotate-at` | `0.5` | Fraction of the context limit at which a session is closed & reborn with a memory snapshot. |
| `--memory-dir` | `.agent_memory` | Directory (inside the workdir) for snapshots, archives and live session state. |

## Evaluator response schema

The Antigravity evaluator is forced to answer in exactly this shape
(parsed tolerantly):

```
STATUS: [GOAL_REACHED | IMPROVED | REGRESSED | CRASHED]
REASONING: <analysis of loss levels, convergence rate, train/val gap, variance, GPU memory>
RECOMMENDATIONS: <numbered, concrete code-level suggestions for the next edit>
```

A hard safety rule overrides hallucinations: a run that crashed or produced
no usable `val_loss` can never be scored `IMPROVED`/`GOAL_REACHED`.

## Persistent sessions, context budget & memory handoff

By default (v2) the agents are **not** amnesiac one-shot processes:

- **Continuity.** The Editor's calls resume one Claude Code conversation
  (`claude --output-format json -p` on the first call captures the
  `session_id`; later calls pass `--resume <session-id>`). The Evaluator
  continues its latest Antigravity conversation via `agy -c -p`. Agents
  therefore remember the codebase, their past edits, and why previous
  attempts failed — instead of re-discovering everything each trial.
- **Context ledger.** Claude's JSON output reports real token usage
  (input + cache-read + cache-creation + output), which the orchestrator
  treats as the authoritative conversation size. CLIs that report nothing
  (agy print mode) fall back to a conservative chars/4 estimator. The
  live percentage is printed at the top of every trial:
  `Context [editor: 512,340 tok (51% of 1,000,000), session=1a2b3c, rotations=0]`
- **Rotation at the degradation threshold.** Claude Code's 1M-context
  model starts degrading noticeably past ~50% fill. When a session's
  ledger crosses `--rotate-at × --context-limit` (default 500k tokens),
  the orchestrator asks that session for a structured **MEMORY
  SNAPSHOT** — goal state, codebase map (files/shapes/constants),
  experiment ledger, confirmed-working vs. dead-end techniques,
  constraints, and ranked next hypotheses — then closes the session.
  The very next call opens a fresh session whose first message is
  seeded with the snapshot inside a `<memory>` block.
- **Cross-run persistence.** Session ids, ledgers and snapshots live in
  `--memory-dir`, so killing and restarting the orchestrator resumes
  the same conversations and memory. Rotations are logged as
  `SESSION_ROTATED` events in `experiments_history.json`.

Memory directory layout:

```
.agent_memory/
├── editor_memory.md              # current distilled snapshot (Editor)
├── editor_memory_archive.jsonl   # every snapshot ever taken + metadata
├── evaluator_memory.md           # same for the Evaluator
├── sessions.json                 # live session ids + token ledgers
├── code_graph.json               # AST code map of the workspace
└── knowledge_graph.json          # temporal experiment facts (see below)
```

The directory is gitignored *before* the pre-experiment snapshot commit,
so memory artifacts never pollute experiment diffs.

## Built-in knowledge graphs (v0.3)

The two capabilities the graph-memory ecosystem provides are implemented
**natively** (stdlib-only, zero extra dependencies), living in
`.agent_memory/` alongside the session memory:

### Code knowledge graph (`core/code_graph.py`)

An AST-derived map of the workspace, in the spirit of
codebase-memory-mcp / aider's repo map:

- Every module's **hyperparameter constants** (top-level `UPPER_SNAKE`
  literals), **functions with call edges**, **classes with methods**,
  and imports — rendered as a dense, token-budgeted digest injected into
  Editor prompts, so structural questions are answered from the map
  instead of burning context re-reading files.
- Snapshots taken before/after each Editor turn are **diffed**: the loop
  knows *exactly* what changed each trial
  (`train.py:LEARNING_RATE: 0.01 -> 0.005`, `train.py::MLP.forward
  modified`) and prints it, feeds it to the Evaluator, and stores it as
  facts. Persisted at `.agent_memory/code_graph.json`.

### Temporal experiment knowledge graph (`core/knowledge_graph.py`)

Graphiti-style temporal facts — every trial emits
`(subject, predicate, object, trial, outcome)` triples:

```
t1: train.py:HIDDEN_DIM CHANGED 32 -> 4096        => CRASHED
t2: train.py:LEARNING_RATE CHANGED 0.01 -> 0.005  => IMPROVED
t2: technique:learning-rate adjustment APPLIED     => IMPROVED
```

Unlike the LLM memory snapshot (lossy by design), these facts are exact
and survive every rotation. Each trial the agents receive the rendered
digest — best config so far, **changes that HELPED**, **DEAD ENDS (do
NOT repeat)**, techniques tried with their outcome history — and the
Editor is explicitly instructed never to repeat a listed dead end.
Persisted at `.agent_memory/knowledge_graph.json`; restored on restart.

Both graphs feed three places: (1) every Editor prompt, (2) every
Evaluator prompt (including the exact AST diff of the trial under
review), and (3) the `LIVE PROJECT KNOWLEDGE` section of the preamble
that seeds every fresh session after a rotation — so a newborn session
knows the codebase and the full experiment history from message one.

### Going further: external graph-memory projects

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

## Artifacts

| File | Purpose |
|------|---------|
| `experiments_history.json` | Full session/trial record: commit hashes, loss curves, agent feedback. Written atomically. |
| `experiment_report.md` | Human-readable summary report generated at session end. |
| git history | One commit per improvement: `experiment(trial-3): val_loss=0.3120`, plus `exp-trial-N` tags. |

All three artifact files are auto-added to `.gitignore` so they never
pollute the experiment diffs.

## Safety & failure handling

- **Timeouts** kill the entire process group (DataLoader workers included).
- **Error triage**: CUDA OOM, shape mismatches, syntax/import errors,
  NaN-loss divergence and missing files are detected from logs and turned
  into targeted fix instructions for the Editor.
- **Dirty repos** are snapshotted (`chore(orchestrator): pre-experiment
  snapshot`) before the loop starts, so nothing of yours is ever lost.
- **No-op edits** (Editor claims success but changed nothing) are detected
  via `git status` and penalized in the next prompt.
- **Claude permissions**: the default uses `--permission-mode acceptEdits`,
  which auto-approves file edits only. Widen at your own risk via
  `--claude-cmd`.

## Project layout

```
ml_orchestrator/
├── pyproject.toml         # uv project: deps, entry point, build config
├── uv.lock                # locked dependency graph (commit this)
├── .python-version        # interpreter pin used by uv
├── main.py                # CLI entrypoint + closed-loop state machine
├── core/
│   ├── agents.py          # Claude/Antigravity CLI bridges, schema parser, fallback evaluator
│   ├── runner.py          # Sandboxed subprocess harness, timeout, error triage
│   ├── git_manager.py     # Commit/revert/rollback + .gitignore management
│   ├── session.py         # Persistent sessions, context ledger, memory rotation
│   ├── code_graph.py      # AST code map: constants, call edges, trial diffs
│   ├── knowledge_graph.py # Temporal experiment facts: wins, dead ends, techniques
│   └── logger.py          # experiments_history.json + markdown report
├── templates/
│   ├── train_example.py   # PyTorch (NumPy/stdlib-fallback) demo trainer → metrics.json
│   └── model_example.py   # Baseline MLP with tunable hyperparameters
├── requirements.txt       # legacy pip fallback only
└── README.md
```
