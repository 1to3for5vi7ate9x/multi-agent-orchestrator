# ml-agent-orchestrator

A closed-loop, CLI-driven engine for **objective-driven agentic work** —
ML experimentation and general ("vibe") coding alike — where **three
agents (Claude Code, Google Antigravity, OpenAI Codex) compete for the
working seats**: a blind judge panel rates anonymous proposals and the
winner drives, a deterministic referee polices the flow (test
tampering, fabricated verdicts, repeated dead ends), and git + an
objective fitness signal remain the final arbiter of every change.
See [`docs/judge-referee.md`](docs/judge-referee.md) for the full
harness design (including the Haskell decision kernel).

Two presets share one loop:

- `--preset ml` (default): minimize `val_loss`/`score` from
  `metrics.json` produced by your training script.
- `--preset coding`: minimize failing tests from any
  `--eval-command` (`pytest -q`, `npm test`, ...) — goal reached when
  the suite is green.

- The **Editor seat** edits your source on disk — `model.py` / `train.py`
  (architecture, training loop, hyperparameters) in the ml preset, any
  non-test source in the coding preset.
- The **Evaluator seat** reads `metrics.json`, terminal logs and the trial
  history, and returns a strict, parseable verdict.
- **Seats are won, not assigned**: a blind tournament ranks anonymous
  proposals and the winner drives (`--no-rotate` pins the old static
  seats). Any installed agent can play either seat.
- **Git** is the state machine — improvements are committed automatically,
  regressions and crashes are reverted automatically.

```
   ┌───────────────────────────────────────────────────────────────────┐
   │   Blind tournament assigns the seats · referee polices the flow   │
   └───────────────────────────────────────────────────────────────────┘

   ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐
   │   Editor   │──▶│  Sandboxed   │──▶│  Evaluator   │──▶│    Git     │
   │    seat    │   │   eval run   │   │     seat     │   │  decides   │
   └────────────┘   └──────────────┘   └──────────────┘   └────────────┘
          ▲                                                     │
          │   feedback / diagnostics / knowledge-graph facts    │
          └─────────────────────────────────────────────────────┘
```

Per trial:
1. **Edit** — the Editor seat modifies the editable source toward the goal.
2. **Run** — the evaluation command executes in a monitored subprocess with
   a hard timeout; CUDA OOM, shape mismatches, syntax errors, NaN losses
   etc. are auto-classified into actionable diagnostics.
3. **Evaluate** — the Evaluator seat analyzes the objective and its
   dynamics (loss curves and train/val gap, or which tests still fail),
   replying in a strict schema. With no second agent installed, a numeric
   heuristic stands in.
4. **Decide** — the referee arbitrates first (a verdict that contradicts
   the numbers is downgraded; a tampered trial is force-reverted), then:
   - `GOAL_REACHED` → commit, write the summary report, exit 0.
   - `IMPROVED` → `git commit` tagged `experiment(trial-N): val_loss=...`,
     continue.
   - `REGRESSED` / `CRASHED` → `git checkout` reverts the edit; the failure
     reasoning + traceback is fed back to the Editor for a different attempt.

Every trial is recorded in `experiments_history.json`; a markdown
`experiment_report.md` is generated when the session ends.

---

## What it's for (and what it isn't)

The loop is driven by an **objective fitness signal**, so it only works
where one already exists. That makes it a *refinement* engine, not a
scaffolding engine: it is good at "here is a precise, measurable
definition of done — get there", and has nothing to optimize against
when asked to "build me a thing".

For code, writing the fitness function means **writing the tests first**.

### Works well

| Use case | Fitness signal | Shape |
|----------|----------------|-------|
| **Test-first feature work** | failing tests → 0 | write the spec as failing tests, let the loop implement |
| **Bug fixing** | a reproducing test → green | usually converges in 1–2 trials |
| **ML experimentation** | `val_loss` from `metrics.json` | the original preset |
| **Any numeric objective** | `score` from `metrics.json` | latency, bundle size, memory, Lighthouse, accuracy |
| **Migrations / ports** | the existing suite stays green | tests are the behavioral oracle |

Test-first feature work is the canonical shape:

```bash
git checkout -b agent/pagination
# write tests/test_pagination.py describing the API you want — it must FAIL
uv run ml-orchestrator --preset coding \
  --goal "Implement cursor pagination so tests/test_pagination.py passes" \
  --editable-files api/pagination.py api/views.py \
  --max-trials 8
```

The referee guarantees it fixes the code rather than the test;
`--editable-files` keeps it from wandering across a large repo.

Any number you can emit is optimizable, which is the most underused mode:

```bash
# bench.py writes {"score": <p95 latency in ms>}
uv run ml-orchestrator --preset ml \
  --goal "Get p95 latency under 120ms" \
  --eval-command "python bench.py" --max-trials 10
```

Use `--direction maximize` where higher is better.

### Doesn't work

- **Greenfield project creation** — there is no signal before tests exist.
- **Subjective goals** — UI design, API ergonomics, prose quality. Nothing
  to minimize.
- **Broad multi-file refactors** — a green suite cannot tell "refactored
  well" from "barely touched".

> [!WARNING]
> **The empty-project trap.** In the coding preset a command that exits 0
> with no parseable test output scores `0.0`, which satisfies
> `failing_tests <= 0`. Point the loop at a project with no tests and it
> reports `Baseline already satisfies the goal — nothing to do` and exits
> 0 having changed nothing. **Always read the baseline line**: if it does
> not show a real failing count, the run is meaningless.

### Starting a new project

Bootstrap by hand; hand over once there is a signal.

1. **Phase 0 — you.** Scaffold, dependencies, a runnable test command, and
   the first spec-as-tests. The orchestrator is the wrong tool here; use
   an interactive coding agent.
2. **Phase 1 — one feature per run.** Branch, failing tests, run. Keep
   `--max-trials` at 6–10: it either converges in 1–3 trials or it is
   stuck on something no edit can fix (the loop gives up with `STALLED`
   after 3 consecutive no-op trials).
3. **Phase 2 — non-functional goals.** Once it works, switch to numeric
   objectives for performance.

Practical notes:

- **Cost scales at 2N agent calls per tournament** (6 with three agents).
  For a small, well-scoped task, `--no-rotate` skips the tournament and
  lets one agent drive — often the better trade.
- **Tournament proposals are written blind**: agents may not read files
  during the proposal stage, so on trial 1 in an unfamiliar repo the panel
  is ranking speculation. It pays off over longer runs, once the knowledge
  graph carries real outcome facts.
- **Scope the eval command.** Auto-detection runs your whole suite; on a
  big repo pass `--eval-command "pytest -q tests/test_feature.py"` so
  every trial isn't paying for the full run.
- **Memory persists across runs** in `--memory-dir`, so dead ends carry
  forward between sessions on the same workspace.

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
4. Optional but recommended: **OpenAI Codex CLI** (`codex`), the third
   competitor for the tournament (`npm install -g @openai/codex`, then
   run `codex` once to log in). Any missing agent is simply dropped
   from the roster.
5. Optional: the compiled **Haskell decision kernel** (`mao-kernel`) —
   see [`docs/judge-referee.md`](docs/judge-referee.md); without it a
   verified Python fallback is used.
6. Optional: PyTorch for the example templates (they fall back to
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

## Install

```bash
# One-shot from PyPI, no clone, no venv:
uvx ml-agent-orchestrator --help          # `ml-orchestrator` also works

# Or straight from GitHub:
uvx --from git+https://github.com/1to3for5vi7ate9x/multi-agent-orchestrator \
    ml-orchestrator --help

# Or as a developer:
git clone git@github.com:1to3for5vi7ate9x/multi-agent-orchestrator.git
cd multi-agent-orchestrator
uv sync --dev
uv run python tests/run_all.py            # all suites should pass
```

## Quick start (self-contained ML demo)

```bash
# --scaffold-demo drops a working train.py/model_example.py in the
# workspace; --init-git creates the repo + baseline commit.
uv run ml-orchestrator \
  --goal "Achieve validation loss < 0.05 on the synthetic dataset without overfitting" \
  --workdir ~/experiments/demo \
  --scaffold-demo \
  --max-trials 5 \
  --timeout 300 \
  --init-git
```

The evaluation subprocess uses the uv-managed interpreter by default
(override with `--python` to point at e.g. a CUDA conda env).

## Vibe coding / general development

```bash
cd your-app        # git repo with a test suite (or pass --init-git)
uv run ml-orchestrator \
  --preset coding \
  --goal "Implement the pagination feature and make the whole test suite pass" \
  --eval-command "pytest -q" \
  --editable-files api/pagination.py api/views.py \
  --max-trials 10
```

How the coding preset differs:

- **Fitness = failing tests.** The score is parsed from pytest / jest /
  vitest / go-test output (a test command exiting 1 because tests fail
  is a *measurement*, not a crash). Goal condition: `failing_tests <= 0`.
- `--eval-command` is auto-detected when omitted (pytest config or
  `tests/` → `pytest -q`; `package.json` → `npm test`).
- The Editor is instructed to fix the code, **never the tests**, and
  commits land as `experiment(trial-3): failing_tests=2.0000`.
- No `metrics.json` needed — but if your eval command writes one with a
  `score` field, it takes precedence (custom fitness functions:
  benchmark latency, lighthouse score, anything numeric; combine with
  `--direction maximize` and `--goal-target`).

Everything else — git commit/revert rails, persistent sessions,
context rotation, code graph, knowledge graph — works identically in
both presets.

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
| `--goal` | *(required)* | Natural-language target objective. A numeric target (e.g. "val loss < 0.25", "score >= 0.9") is also parsed for the fallback evaluator. |
| `--preset` | `ml` | Objective preset: `ml` (minimize val_loss from metrics.json) or `coding` (minimize failing tests). |
| `--eval-command` | preset-dependent | Shell command measuring the objective each trial (`pytest -q`, `npm test`, `python train.py`...). |
| `--direction` | from preset | `minimize` or `maximize` the score. |
| `--goal-target` | parsed from goal | Explicit numeric goal for the score. |
| `--goal-op` | derived | Comparison used to test the goal (`<`, `<=`, `>`, `>=`). Rarely needed — see below. |
| `--scaffold-demo` | off | Drop the bundled demo train.py/model into the workspace first. |
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
| `--agents` | all installed | Competing agent pool: any of `claude` `antigravity` `codex`. |
| `--no-rotate` | off | Disable blind-tournament seat rotation (static v0.4 seats). |
| `--tournament-every` | `0` | Extra fixed-cadence tournaments every N trials (0 = start + stagnation only). |
| `--stateless` | off | Disable persistent sessions — every agent call becomes a fresh, memoryless process (pre-v2 behavior). |
| `--context-limit` | `1000000` | Model context window in tokens (Claude Code 1M-context model). |
| `--rotate-at` | `0.5` | Fraction of the context limit at which a session is closed & reborn with a memory snapshot. |
| `--memory-dir` | `.agent_memory` | Directory (inside the workdir) for snapshots, archives and live session state. |

### How the goal condition is decided

One place owns the comparison; three inputs feed it, in increasing
precedence:

1. **The preset** — `ml` is `val_loss < target`, `coding` is
   `failing_tests <= 0`.
2. **The goal text** — a numeric condition in `--goal` sets the target
   *and* the operator *and* the direction. `--goal "Achieve score >= 0.9"`
   is a maximize run; `--goal "validation loss < 0.25"` is a minimize one.
3. **Flags** — `--goal-target`, then `--direction` (which flips the
   operator to match, preserving strictness), then `--goal-op` for full
   manual control.

The resolved condition is printed at startup, e.g.
`Numeric goal condition: val_loss < 0.25 (minimize; from goal text)`.
Check that line before a long run — it is the exact test used to declare
`GOAL_REACHED`.

## Interrupting a run

`Ctrl-C` finishes the session record rather than aborting mid-write: the
history is closed as `INTERRUPTED`, the markdown report is still
generated, and the working tree is **left exactly as it is**. Any
uncommitted changes from the trial in flight are listed on exit, along
with the command to discard them. Nothing is auto-reverted — interrupting
is usually deliberate. Exit code is `130`.

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

## The judge, the referee, and the Haskell kernel (v0.5)

- **Blind tournament (judge).** Each roster agent writes an anonymous
  proposal for the next change; identities are scrubbed and labels
  shuffled; every agent then scores all candidates 1-10 against a fixed
  rubric. Highest mean takes the **Editor** seat and implements its own
  proposal; the runner-up takes the **Evaluator** seat. The judge
  decides *who drives* — commits/reverts are still decided only by the
  objective fitness signal.
  - **Self-scores are dropped** once at least 3 judges respond: a judge
    rating its own anonymized candidate is the one bias blind labelling
    cannot remove. Below 3 judges the rule is off, so a two-agent panel
    never collapses to a single voter.
  - **Re-runs** happen at start, on stagnation, and optionally every N
    trials (`--tournament-every`). The stagnation trigger requires the
    knowledge graph to have gained facts since the last tournament, and
    its threshold backs off (2 → 4 → 8): rotating the driver when stuck
    is the point, but re-ranking the same unchanged context at 2N CLI
    calls a round is not.
- **Referee (deterministic watchdog).** Pure rules, not an LLM.
  - `TEST_TAMPERING` (CRITICAL) — modifying test files in coding mode is
    force-reverted before any run is wasted.
  - `VERDICT_ON_CRASH` / `VERDICT_CONTRADICTION` / `PREMATURE_GOAL` —
    verdicts that contradict the numbers are downgraded; numeric truth
    wins.
  - `METRIC_FABRICATION` (WARN) — the ml-preset counterpart of test
    tampering: an edit that writes a *hardcoded* objective value
    (`json.dump({"val_loss": 0.001}, ...)`) instead of measuring one.
  - `RUNTIME_COLLAPSE` (WARN) — the objective improved while the run got
    ≥10× shorter than the baseline, which is fabrication-shaped. Early
    stopping and caching do this legitimately, hence WARN.
  - `METRICS_TAMPERING`, `REPEATED_DEAD_END`, `SUSPICIOUS_JUMP` — flagged
    into the history, the editor's feedback and the next tournament's
    judging context. `SUSPICIOUS_JUMP` is suppressed once the goal
    condition is met, since reaching the target *is* the terminal event.
- **Haskell decision kernel (`haskell/`).** The judge aggregation and
  referee rules — the trust-critical decision core — are canonically
  implemented as a pure Haskell binary (`mao-kernel`, JSON in/out).
  The orchestrator uses it when found (`$MAO_KERNEL` or PATH) and
  falls back to a behavior-identical Python implementation otherwise,
  so `uvx ml-agent-orchestrator` works with zero extra toolchain.
  Parity is *tested*, not asserted: the shared path corpus
  (`tests/test_paths.json`) generates one parity vector per case, and
  the vector runner exercises the production Python functions rather
  than a copy, so neither implementation can drift alone. Build
  instructions: [`docs/judge-referee.md`](docs/judge-referee.md).

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

The built-in graphs need no external services. For richer memory
(semantic search, multi-project graphs) several MCP servers slot in
cleanly — see [`docs/memory-integrations.md`](docs/memory-integrations.md)
for the options and the two integration points.

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
multi-agent-orchestrator/
├── pyproject.toml             # uv project: deps, entry point, build config
├── uv.lock                    # locked dependency graph (committed)
├── .python-version            # interpreter pin used by uv
├── .github/workflows/
│   ├── ci.yml                 # tests + build on every push/PR
│   └── release.yml            # tag v* → PyPI (trusted publishing) + GitHub Release
├── src/ml_orchestrator/
│   ├── main.py                # CLI entrypoint + closed-loop state machine
│   ├── core/
│   │   ├── agents.py          # Role prompts + CLI invocation, schema parser, fallback evaluator
│   │   ├── roster.py          # Agent pool: claude / antigravity / codex specs + AskAgent
│   │   ├── tournament.py      # Blind proposals, anonymization, panel judging, seats
│   │   ├── referee.py         # Deterministic watchdog rules (Python fallback of the kernel)
│   │   ├── kernel.py          # Bridge to the Haskell decision kernel (fail-open)
│   │   ├── runner.py          # Sandboxed subprocess harness, timeout, error triage
│   │   ├── fitness.py         # Generalized objective: metrics file / test parsing / exit code
│   │   ├── git_manager.py     # Commit/revert/rollback + .gitignore management
│   │   ├── session.py         # Persistent sessions, context ledger, memory rotation
│   │   ├── code_graph.py      # AST code map: constants, call edges, trial diffs
│   │   ├── knowledge_graph.py # Temporal experiment facts: wins, dead ends, techniques
│   │   └── logger.py          # experiments_history.json + markdown report
│   └── templates/             # bundled demo (used by --scaffold-demo)
├── haskell/                   # mao-kernel: canonical judge/referee decision kernel
├── docs/
│   ├── judge-referee.md       # harness design: tournament, referee, kernel protocol
│   └── memory-integrations.md # optional external graph-memory servers
├── tests/                     # runnable suites: uv run python tests/run_all.py
│   ├── kernel_vectors.json    # golden vectors: Haskell kernel <-> Python parity
│   └── test_paths.json        # shared test-path corpus (generates parity vectors)
├── requirements.txt           # legacy pip fallback only
└── README.md
```

## Releasing (maintainers)

Tag and push — CI does the rest:

```bash
git tag v0.4.0 && git push origin v0.4.0
```

`release.yml` runs the tests, builds sdist+wheel with `uv build`,
publishes to PyPI via OIDC **trusted publishing** (one-time setup: add
this repo as a Trusted Publisher on pypi.org with environment `pypi`,
and create that environment in the GitHub repo settings — no API tokens
ever), and attaches the artifacts to a GitHub Release.
