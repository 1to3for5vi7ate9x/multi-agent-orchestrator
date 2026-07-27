"""Blind tournament — merit-based seat rotation across the agent roster.

Instead of a static Claude-edits / Antigravity-evaluates loop, every
agent in the roster periodically competes for the Editor seat:

1. **Propose** — each agent independently writes a short, concrete
   proposal for the next change (hypothesis + planned edit + expected
   effect). Text only; nothing is applied.
2. **Anonymize** — proposals get shuffled single-letter labels and are
   scrubbed of vendor/model self-references, so no judge knows whose
   work it is rating (arena-style blind judging).
3. **Panel judge** — every roster agent scores ALL candidates 1-10
   against a fixed rubric (referee flags included as context). Scores
   are averaged across the panel; a judge's self-preference bias is
   diluted because it is one voice among N and formally blind.
4. **Seat assignment** — highest mean score takes the Editor seat and
   implements its own winning proposal; the best-ranked *other* agent
   takes the Evaluator seat. The judge panel is always everyone.

Ground truth stays king: the tournament only decides WHO DRIVES.
Whether a change is committed or reverted is still decided by the
objective fitness signal + evaluator verdict + referee, never by
judge opinion.
"""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import kernel
from .live_status import DONE, FAILED, RUNNING, NullStatus

# Vendor/self-identifying strings scrubbed before judging.
_IDENTITY_RE = re.compile(
    r"(?i)\b(claude(?:\s+code)?|anthropic|gemini|antigravity|agy|codex|"
    r"openai|gpt-?[0-9a-z.]*|chatgpt|google|bard)\b"
)

LABELS = ["A", "B", "C", "D", "E", "F"]


def scrub_identity(text: str) -> str:
    """Replace vendor/model self-references so judges rate blind."""
    return _IDENTITY_RE.sub("[model]", text)


@dataclass
class Proposal:
    agent: str
    text: str


@dataclass
class TournamentResult:
    winner: str                                  # editor seat
    evaluator: str                               # evaluator seat
    ranking: List[Tuple[str, float]]             # (agent, mean score) desc
    proposals: Dict[str, str]                    # agent -> proposal text
    label_map: Dict[str, str]                    # label -> agent (unblinded)
    judge_scores: Dict[str, Dict[str, float]]    # judge -> {label: score}
    winning_proposal: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "evaluator": self.evaluator,
            "ranking": [[a, round(s, 3)] for a, s in self.ranking],
            "label_map": self.label_map,
            "judge_scores": self.judge_scores,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_proposal_prompt(
    goal: str,
    trial: int,
    knowledge_context: str,
    history_summary: str,
    last_feedback: str,
    mode: str,
) -> str:
    domain = ("failing tests / the evaluation command" if mode == "coding"
              else "the training/validation metrics")
    return f"""You are competing (anonymously) for the Editor seat in an automated improvement loop. Write a PROPOSAL for the single next change — do NOT edit any files in this turn.

## Target goal
{goal}

## Upcoming trial
{trial}

{knowledge_context or ""}

## Recent history
{history_summary or "No previous trials."}

## Latest feedback
{last_feedback or "(none)"}

## Required output — exactly these three sections, total under 250 words
HYPOTHESIS: <why the current bottleneck in {domain} exists — one dense paragraph grounded in the facts above>
CHANGE: <the precise edit you would make: files, functions/constants, old->new values where applicable>
EXPECTED: <the measurable effect on the objective and the risk if wrong>

Rules: answer DIRECTLY from the context given above — do NOT use tools, do NOT read files, do NOT run commands (tool use is auto-denied in this mode and produces an empty answer). Be specific, never propose anything listed as a dead end, and do not mention what model or product you are."""


JUDGE_RUBRIC = """Score each candidate proposal from 1-10 using this rubric (weights in brackets):
- [40%] Grounding: does it correctly diagnose the bottleneck using the experiment facts/knowledge given, and avoid known dead ends?
- [30%] Specificity: is the change precise enough to implement without guessing (files, values, mechanism)?
- [20%] Expected impact: is the predicted effect on the objective plausible and material?
- [10%] Risk awareness: does it state a falsifiable expectation and the failure mode?"""


def build_judge_prompt(
    goal: str,
    labeled: List[Tuple[str, str]],
    knowledge_context: str,
    referee_notes: str,
) -> str:
    blocks = "\n\n".join(
        f"### Candidate {label}\n{text}" for label, text in labeled
    )
    return f"""You are an impartial judge in a blind review. Several anonymous candidates propose the next change toward this goal:

{goal}

{knowledge_context or ""}

{("## Referee flags on recent flow (context for judging)" + chr(10) + referee_notes) if referee_notes else ""}

{JUDGE_RUBRIC}

## Candidates
{blocks}

## Response format — respond with EXACTLY these two lines and nothing else (answer directly; do NOT use tools, read files, or run commands)
SCORES: {" ".join(f"{label}=<1-10>" for label, _ in labeled)}
REASON: <one paragraph justifying the ranking>"""


# ---------------------------------------------------------------------------
# Aggregation — the canonical Python implementation
# ---------------------------------------------------------------------------
#
# SPEC (mirrored by the `aggregate` op in haskell/src/Main.hs; both are
# pinned to tests/kernel_vectors.json — change one, change all three):
#
# 1. exclude_self := (number of judges that returned scores) >= 3.
# 2. For each label L in `labels` with an owner A = label_map[L]:
#      vals := scores for L from every judge except (when exclude_self) A
#      if vals is empty: vals := scores for L from every judge  [fallback]
#      if vals is still empty: skip L entirely
#      mean[A] := sum(vals) / len(vals)
# 3. ranking := means sorted by score descending, STABLY, with the
#    underlying order being `labels` order.
# 4. winner := the incumbent if it is tied at the top, else the first
#    agent in the tied-at-top group.
# 5. evaluator := the highest-ranked agent that is not the winner (the
#    winner itself when it is the only one). When several are tied at
#    that score, prefer `incumbent_evaluator`.
#
# Rule 1 exists because a judge scoring its own anonymized candidate is
# the one systematic bias blind labelling cannot remove. It is gated at
# 3 judges so that a two-agent panel does not collapse to a single
# voter. The rule-2 fallback guarantees exclusion can never make a
# candidate vanish from the ranking when only its owner scored it.
#
# Rule 5's incumbent preference mirrors rule 4's and exists for the same
# reason plus a concrete one: the evaluator seat owns a persistent,
# resumable session, so churning it on a numeric tie throws away a warm
# conversation and starts cold. Note what is deliberately NOT here: a
# lexicographic final fallback. When neither incumbent is tied, order
# comes from the shuffled labels, which is arbitrary per run but
# unbiased across runs — sorting by agent name instead would hand one
# vendor a permanent structural advantage on every tie.

def aggregate_scores(
    labels: Sequence[str],
    label_map: Dict[str, str],
    judge_scores: Dict[str, Dict[str, float]],
    incumbent: Optional[str] = None,
    incumbent_evaluator: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Aggregate a judge panel into a ranking and seat assignment.

    Returns None when no label received a usable score. This is the
    implementation the orchestrator, the test harness and the kernel
    golden vectors all exercise — do not inline a second copy.
    """
    exclude_self = len(judge_scores) >= 3
    means: Dict[str, float] = {}
    for label in labels:
        owner = label_map.get(label)
        if owner is None:
            continue
        scored = [(judge, s[label]) for judge, s in judge_scores.items()
                  if label in s]
        vals = [v for judge, v in scored
                if not (exclude_self and judge == owner)]
        if not vals:  # only the owner scored it — keep the candidate alive
            vals = [v for _, v in scored]
        if not vals:
            continue
        means[owner] = sum(vals) / len(vals)
    if not means:
        return None
    ranking = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranking[0][1]
    tied = [a for a, s in ranking if abs(s - top_score) < 1e-9]
    winner = incumbent if incumbent in tied else tied[0]
    other_pairs = [(a, s) for a, s in ranking if a != winner]
    if not other_pairs:
        evaluator = winner
    else:
        top_other = other_pairs[0][1]
        tied_others = [a for a, s in other_pairs
                       if abs(s - top_other) < 1e-9]
        evaluator = (incumbent_evaluator
                     if incumbent_evaluator in tied_others
                     else other_pairs[0][0])
    return {
        "ranking": [[a, s] for a, s in ranking],
        "winner": winner,
        "evaluator": evaluator,
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_scores(raw: str, labels: Sequence[str]) -> Optional[Dict[str, float]]:
    """Parse 'SCORES: A=7 B=5.5 C=9' tolerantly. None if nothing usable."""
    text = raw.replace("**", "")
    scores: Dict[str, float] = {}
    for label in labels:
        m = re.search(rf"\b{label}\s*[=:]\s*(-?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            scores[label] = min(max(val, 0.0), 10.0)
    return scores if scores else None


# ---------------------------------------------------------------------------
# Parallel gathering
# ---------------------------------------------------------------------------

def _gather_parallel(
    agents: Dict[str, Any],
    prompt: str,
    status: Any,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Ask every agent concurrently. Returns name -> (text, error);
    exactly one of the pair is None. The status board is updated live."""
    results: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        futures = {}
        for name, agent in agents.items():
            status.update(name, RUNNING)
            futures[pool.submit(agent.ask, prompt)] = name
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                text = (fut.result() or "").strip()
            except Exception as exc:  # dead CLI must not kill the loop
                results[name] = (None, str(exc))
                status.update(name, FAILED, str(exc).splitlines()[0][:80])
                continue
            if text:
                results[name] = (text, None)
                status.update(name, DONE, f"{len(text):,} chars")
            else:
                results[name] = (None, "empty reply")
                status.update(name, FAILED, "empty reply")
    return results


# ---------------------------------------------------------------------------
# The tournament
# ---------------------------------------------------------------------------

def run_tournament(
    agents: Dict[str, Any],       # name -> object with .ask(prompt) -> str
    goal: str,
    trial: int,
    knowledge_context: str,
    history_summary: str,
    last_feedback: str,
    mode: str,
    referee_notes: str = "",
    rng: Optional[random.Random] = None,
    incumbent: Optional[str] = None,
    incumbent_evaluator: Optional[str] = None,
    log=lambda msg: None,
    status_factory: Optional[Callable[[str, List[str]], Any]] = None,
) -> Optional[TournamentResult]:
    """Run one blind tournament. Returns None when fewer than two agents
    produced proposals (in which case the caller keeps the incumbent).

    Both stages (proposals, panel judging) run all agents IN PARALLEL;
    ``status_factory(title, agent_names)`` supplies a live status board
    (see live_status.LiveStatus) so the user watches each CLI's state
    and elapsed time instead of a silent gap."""
    rng = rng or random.Random()
    status_factory = status_factory or NullStatus
    notes: List[str] = []

    # 1. Collect proposals — all agents concurrently.
    proposals: List[Proposal] = []
    prompt = build_proposal_prompt(goal, trial, knowledge_context,
                                   history_summary, last_feedback, mode)
    with status_factory("Tournament — anonymous proposals (parallel)",
                        list(agents)) as status:
        gathered = _gather_parallel(agents, prompt, status)
    for name in agents:
        text, err = gathered.get(name, (None, "no result"))
        if text:
            proposals.append(Proposal(name, text))
            log(f"proposal received from {name} "
                f"({len(text.split())} words)")
        elif err == "empty reply":
            notes.append(f"{name}: empty proposal — excluded")
            log(f"proposal from {name} was EMPTY — excluded")
        else:
            notes.append(f"{name}: proposal failed ({err}) — excluded")
            log(f"proposal from {name} failed: {err}")
    if len(proposals) < 2:
        notes.append("fewer than 2 proposals; tournament aborted")
        return None

    # 2. Anonymize: shuffle labels, scrub identity strings.
    rng.shuffle(proposals)
    labels = LABELS[: len(proposals)]
    label_map = {label: p.agent for label, p in zip(labels, proposals)}
    labeled = [(label, scrub_identity(p.text))
               for label, p in zip(labels, proposals)]

    # 3. Panel judging — every agent scores all candidates, concurrently.
    judge_prompt = build_judge_prompt(goal, labeled, knowledge_context,
                                      referee_notes)
    judge_scores: Dict[str, Dict[str, float]] = {}
    with status_factory("Tournament — blind panel judging (parallel)",
                        list(agents)) as status:
        gathered = _gather_parallel(agents, judge_prompt, status)
    for name in agents:
        raw, err = gathered.get(name, (None, "no result"))
        if raw is None:
            notes.append(f"judge {name}: failed ({err}) — ignored")
            log(f"judge {name} failed: {err}")
            continue
        parsed = parse_scores(raw, labels)
        if parsed:
            judge_scores[name] = parsed
            log(f"judge {name}: " + " ".join(
                f"{k}={v:g}" for k, v in sorted(parsed.items())))
        else:
            notes.append(f"judge {name}: unparseable scores — ignored")
            log(f"judge {name}: unparseable scores — ignored")
    if not judge_scores:
        notes.append("no judge produced scores; tournament aborted")
        return None

    # 4. Aggregate and assign seats.
    # Canonical path: the Haskell decision kernel (fail-open to Python).
    ranking: List[Tuple[str, float]] = []
    winner = ""
    evaluator = ""
    resp = kernel.call({
        "op": "aggregate",
        "labels": list(labels),
        "label_map": label_map,
        "judge_scores": judge_scores,
        "incumbent": incumbent,
        "incumbent_evaluator": incumbent_evaluator,
    })
    if resp and resp.get("ranking") and resp.get("winner"):
        ranking = [(str(a), float(s)) for a, s in resp["ranking"]]
        winner = str(resp["winner"])
        evaluator = str(resp.get("evaluator") or "")
        notes.append("aggregation: haskell kernel")
    else:
        agg = aggregate_scores(labels, label_map, judge_scores, incumbent,
                               incumbent_evaluator)
        if agg is None:
            return None
        ranking = [(str(a), float(s)) for a, s in agg["ranking"]]
        winner = str(agg["winner"])
        evaluator = str(agg["evaluator"])
        notes.append("aggregation: python fallback")
    if not ranking or not winner:
        return None
    if len(judge_scores) >= 3:
        notes.append("self-scores excluded (>=3 judges)")

    # The seat assignment comes FROM the aggregation, kernel or Python.
    # It used to be recomputed here, which silently discarded the
    # `evaluator` both implementations return and the vectors pin — so
    # the tie-break rule would have had no effect on the live loop.
    # The recompute survives only as a guard against a kernel that
    # omits the field entirely.
    if not evaluator:
        others = [a for a, _ in ranking if a != winner]
        evaluator = others[0] if others else winner
    winning_proposal = next(
        p.text for p in proposals if p.agent == winner)

    return TournamentResult(
        winner=winner,
        evaluator=evaluator,
        ranking=ranking,
        proposals={p.agent: p.text for p in proposals},
        label_map=label_map,
        judge_scores=judge_scores,
        winning_proposal=winning_proposal,
        notes=notes,
    )
