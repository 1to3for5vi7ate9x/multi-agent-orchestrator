# The Judge & Referee Harness

v0.5 turns the orchestrator from a static two-agent loop into a
self-improving, merit-rotating tri-agent system. Three CLI agents —
**Claude Code** (`claude`), **Google Antigravity** (`agy`), and
**OpenAI Codex** (`codex`) — compete for the working seats; a blind
**judge panel** decides who drives, and a deterministic **referee**
polices the flow.

```
            ┌─────────────────────────────────────────────┐
            │           BLIND TOURNAMENT (judge)          │
            │  proposals → anonymize → panel scores 1-10  │
            └──────────────┬──────────────────────────────┘
                   winner  │  runner-up
                 ┌─────────▼────────┐   ┌──────────────────┐
                 │   EDITOR seat    │   │  EVALUATOR seat  │
                 └─────────┬────────┘   └────────┬─────────┘
        edit → run → fitness → verdict ──────────┘
                           │
            ┌──────────────▼──────────────────────────────┐
            │      REFEREE (deterministic watchdog)       │
            │  test tampering · metrics fabrication ·     │
            │  verdict-vs-numbers arbitration · dead-end  │
            │  repeats · suspicious jumps                 │
            └─────────────────────────────────────────────┘
```

## The judge: blind tournament

1. **Proposals.** Every roster agent answers the same prompt: hypothesis
   → precise change → expected effect, ≤250 words, text only. Nothing
   is applied to the workspace.
2. **Anonymization.** Proposals get shuffled labels (Candidate A/B/C)
   and are scrubbed of vendor/model self-references
   (`tournament.scrub_identity`), so no judge knows who wrote what.
3. **Panel judging.** Every roster agent scores ALL candidates 1-10
   against a fixed rubric (grounding 40%, specificity 30%, expected
   impact 20%, risk awareness 10%), replying in the strict
   `SCORES: A=n B=n C=n` schema. Recent referee flags are provided as
   judging context.
4. **Aggregation & seats.** Mean score per candidate across the panel,
   **excluding each judge's score for its own candidate** once at least
   3 judges responded; ties prefer the incumbent (stability). The winner
   takes the **Editor** seat and is instructed to implement its own
   winning proposal; the best-ranked other agent takes the **Evaluator**
   seat. The panel is always everyone — judging is a role all agents
   keep.

   The canonical statement of this is the SPEC comment above
   `aggregate_scores` in `core/tournament.py`; `haskell/src/Main.hs`
   ports it and `tests/kernel_vectors.json` pins both.

Tournaments run at trial 1, on stagnation, and optionally every N trials
(`--tournament-every N`). The stagnation trigger fires when the
incumbent editor goes **N consecutive trials without a commit** *and*
the knowledge graph has gained facts since the last tournament — with no
new facts the panel can only re-rank the same speculation. The threshold
backs off after each tournament (2 → 4 → 8), so a stuck loop doesn't
spend its whole budget re-proposing from unchanged context.

Both stages run **all agents in parallel** (one thread per CLI), with a
live status board on the terminal — per agent: spinner, state, ticking
elapsed clock, and the result or failure reason:

```
  Tournament — anonymous proposals (parallel)
  ⠹ claude        running    0:47
  ✔ antigravity   done       0:31  1,480 chars
  ✘ codex         failed     0:02  exited 1: Error loading config...
```

On pipes/CI the board degrades to plain state-transition lines so logs
stay greppable.

**Ground truth stays king.** The judge only decides *who drives*.
Commit/revert decisions are still made by the objective fitness signal
(val_loss / failing tests), the evaluator verdict, and the referee —
never by judge opinion.

### Design honesty

Blind judging reduces — but cannot eliminate — self-preference bias: a
model may recognize its own style even unbranded. The mitigations are
(a) the panel averages N judges so one biased voice is diluted, and
(b) seat performance is tracked empirically (`editor_agent` per trial
in `experiments_history.json`), so a chronically over-rated agent is
exposed by its commit/revert record and dethroned by the stagnation
trigger.

## The referee: deterministic watchdog

Deliberately **not** an LLM. Every rule is a pure function over
observable facts, so its flags are trustworthy inputs for both the loop
and the judges:

| Flag | Severity | Trigger | Consequence |
|------|----------|---------|-------------|
| `TEST_TAMPERING` | CRITICAL | Editor modified test files in coding mode (reward hacking) | Edit force-reverted; **no run wasted**; editor scolded |
| `VERDICT_ON_CRASH` | CRITICAL | Evaluator claimed IMPROVED/GOAL on an unmeasured run | Verdict overridden to CRASHED |
| `METRICS_TAMPERING` | WARN | Editor wrote the metrics file directly during the edit phase | Recorded (the file is deleted before every run anyway) |
| `VERDICT_CONTRADICTION` | WARN | Evaluator said IMPROVED but the score didn't beat the best | Verdict downgraded to REGRESSED — numeric truth wins |
| `PREMATURE_GOAL` | WARN | GOAL_REACHED claimed but the numeric goal condition isn't met | Downgraded to IMPROVED/REGRESSED |
| `METRIC_FABRICATION` | WARN | The edit writes a **hardcoded** objective value instead of measuring one | Surfaced to evaluator + next judges |
| `RUNTIME_COLLAPSE` | WARN | Objective improved while the run got ≥10× shorter than the baseline | Surfaced to evaluator + next judges |
| `REPEATED_DEAD_END` | WARN | The trial repeats a change recorded as a dead end in the knowledge graph | Surfaced to evaluator + next judges |
| `SUSPICIOUS_JUMP` | INFO | Objective improved >90% in one trial, goal **not** yet met | Recorded for scrutiny |
| `NO_CHANGES` | INFO | Editor claimed completion without touching a file | Recorded |

### Guarding the ml preset

`TEST_TAMPERING` protects the coding preset's arbiter: you may not edit
the thing that measures you. `METRIC_FABRICATION` and `RUNTIME_COLLAPSE`
are its ml-preset counterparts — without them, nothing stopped an editor
from gutting `train.py` and writing a hardcoded `metrics.json`, which
the loop would have committed as an improvement.

Both are deliberately **WARN, not CRITICAL**, and deliberately narrow.
`find_literal_metric_writes` only fires when a literal objective value
flows into a metrics-file *write* (a dict literal passed to
`json.dump`, directly or via a variable, or a JSON string literal) —
never on `val_loss = 0.0` as an accumulator initializer, which is
ubiquitous in real training loops. Early stopping and caching produce
legitimate runtime collapses. These flags are evidence for the
evaluator and the next judge panel, not an automatic revert.

`SUSPICIOUS_JUMP` is suppressed once the goal condition holds: N failing
tests → 0 is a 100% jump *and* the exact outcome the loop exists to
produce, so flagging it would feed the next panel a referee concern
about success.

All flags land in `experiments_history.json` (per trial +
`REFEREE_FLAGS`/`REFEREE_BLOCK` events), in the next editor feedback,
and in the next tournament's judging context.

## The Haskell decision kernel (`haskell/`)

The judge aggregation and referee rules are the **trust-critical
decision core** of the loop, so their canonical implementation is a
small pure Haskell program, `mao-kernel`: total functions over typed
inputs, one JSON request on stdin, one JSON response on stdout, no
other IO.

Protocol (`op` field): `aggregate` (panel means, ranking,
incumbent-stable tie-break, seat assignment), `review_edit`,
`review_result` (the referee rule tables above).

**Resolution & fallback.** The Python orchestrator invokes the kernel
when it finds one — `$MAO_KERNEL` (explicit path) or `mao-kernel` on
PATH — and every call is *fail-open*: a missing or crashing kernel
falls back to the built-in Python implementation, which is kept
behavior-identical. Both implementations are held to the same golden
vectors (`tests/kernel_vectors.json`); the parity suite runs the
vectors against the Python fallback always, and against the compiled
kernel whenever one is present. Flag comparisons pin code+severity;
aggregate comparisons pin ranking, scores, winner and evaluator.
This is how the Haskell harness stays load-bearing without breaking
the zero-toolchain `uvx ml-agent-orchestrator` install.

### Keeping parity honest

Golden vectors only prove what they probe, and a hand-maintained vector
list drifts toward the cases where the implementations already agree —
which is exactly the failure they exist to prevent. Two structural rules
close that gap:

1. **One spec, N consumers.** Each rule has exactly one Python
   implementation, and the vector runner calls *that* function. It must
   never inline its own copy — an earlier version reimplemented the
   aggregation inside the test, so the vectors bound the kernel to the
   test's arithmetic while `core/tournament.py` went unverified.
2. **Generated path vectors.** `tests/test_paths.json` is the shared
   corpus for the test-file predicate. The suite asserts the Python spec
   against every entry *and* generates one `review_edit` parity vector
   per entry, so a case asserted anywhere is checked against both
   implementations by construction. Add cases there, never inline.

This matters because the predicate is force-reverting. A port that used
a plain infix search for `test_` classified `latest_thing.py`,
`fastest_run.py` and `contest_form.py` as test files — a CRITICAL flag
and a destroyed edit, but only for users who had the compiled kernel
installed. The regex in `core/referee.py` is the spec; `\b` is
load-bearing.

Build it:

```bash
# toolchain (once): https://www.haskell.org/ghcup/
ghcup install ghc cabal
cd haskell
cabal build exe:mao-kernel
# put the binary on PATH (or export MAO_KERNEL=<path to it>)
cp "$(cabal list-bin mao-kernel)" ~/.local/bin/mao-kernel
# verify parity:
cd .. && uv run python tests/test_judge_referee.py
```

CI builds the kernel and runs the same parity vectors on every push
(`kernel` job in `.github/workflows/ci.yml`).

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--agents claude antigravity codex` | all installed | The competing pool. |
| `--no-rotate` | off | Pin the v0.4 static seats (claude edits, antigravity evaluates). |
| `--tournament-every N` | 0 | Extra fixed-cadence tournaments (0 = start + stagnation only). |

Cost note: a tournament adds one proposal call + one judge call per
roster agent. With the default triggers (start + stagnation) that
overhead is a few calls per session, not per trial.
