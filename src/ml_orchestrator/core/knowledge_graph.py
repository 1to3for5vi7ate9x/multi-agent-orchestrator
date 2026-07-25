"""Temporal experiment knowledge graph.

Inspired by Graphiti-style temporal graphs: every trial produces
*facts* — (subject, predicate, object) triples stamped with the trial
they occurred in and the outcome they led to. Unlike the freeform LLM
memory snapshot (which is lossy by design), these facts are exact and
survive any number of session rotations.

Fact examples the loop generates automatically:
    (train.py:LEARNING_RATE, CHANGED, 0.01 -> 0.005)   trial=2  outcome=IMPROVED
    (train.py::train_torch,  MODIFIED, -)              trial=3  outcome=CRASHED
    (technique:weight decay, APPLIED, -)               trial=4  outcome=GOAL_REACHED

Queries the orchestrator uses every trial:
    wins()        -> changes that improved or reached the goal
    dead_ends()   -> changes that regressed or crashed (do-not-repeat list)
    param_history -> full value trajectory of one hyperparameter
    render_context() -> dense markdown block injected into agent prompts
                        and into new-session memory preambles
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GOOD_OUTCOMES = {"IMPROVED", "GOAL_REACHED", "BASELINE"}
BAD_OUTCOMES = {"REGRESSED", "CRASHED", "REFEREE_BLOCKED"}

# Heuristic technique taxonomy extracted from editor change summaries.
TECHNIQUE_PATTERNS: List[Tuple[str, str]] = [
    (r"learning[\s_-]*rate|\blr\b", "learning-rate adjustment"),
    (r"weight[\s_-]*decay|l2\b", "weight decay / L2"),
    (r"dropout", "dropout"),
    (r"batch[\s_-]*size", "batch-size change"),
    (r"schedul", "LR schedule"),
    (r"warm[\s_-]*up", "LR warmup"),
    (r"clip", "gradient clipping"),
    (r"normali[sz]", "normalization"),
    (r"augment", "data augmentation"),
    (r"hidden|layer|width|depth|capacit|architect", "architecture change"),
    (r"optimi[sz]er|adam|sgd|adamw", "optimizer change"),
    (r"epoch", "epoch-count change"),
    (r"early[\s_-]*stop", "early stopping"),
    (r"mixed[\s_-]*precision|amp\b|autocast", "mixed precision"),
    (r"accumulat", "gradient accumulation"),
    (r"refactor", "refactor"),
]


def extract_techniques(summary: str) -> List[str]:
    found: List[str] = []
    lowered = (summary or "").lower()
    for pattern, label in TECHNIQUE_PATTERNS:
        if re.search(pattern, lowered) and label not in found:
            found.append(label)
    return found


class ExperimentKnowledgeGraph:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = {"facts": [], "trials": {}}
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "facts" in data:
                self._data = data
        except (json.JSONDecodeError, ValueError, OSError):
            pass  # start fresh; the archive of record is experiments_history

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- writing ------------------------------------------------------------------

    def record_trial(
        self,
        trial: int,
        outcome: str,
        action: str,
        val_loss: Optional[float],
        commit: Optional[str],
        const_changes: Optional[List[Tuple[str, str, Optional[str], Optional[str]]]] = None,
        func_changes: Optional[List[str]] = None,
        editor_summary: str = "",
        evaluator_reasoning: str = "",
    ) -> None:
        self._data["trials"][str(trial)] = {
            "outcome": outcome,
            "action": action,
            "val_loss": val_loss,
            "commit": commit,
            "editor_summary": (editor_summary or "")[:400],
            "evaluator_reasoning": (evaluator_reasoning or "")[:400],
        }
        facts = self._data["facts"]
        for module, name, old, new in (const_changes or []):
            facts.append({
                "trial": trial,
                "subject": f"{module}:{name}",
                "predicate": "CHANGED",
                "object": f"{old} -> {new}" if old is not None and new is not None
                          else (f"added={new}" if old is None else f"removed (was {old})"),
                "outcome": outcome,
            })
        for key in (func_changes or [])[:20]:
            facts.append({
                "trial": trial,
                "subject": key,
                "predicate": "MODIFIED",
                "object": "-",
                "outcome": outcome,
            })
        for technique in extract_techniques(editor_summary):
            facts.append({
                "trial": trial,
                "subject": f"technique:{technique}",
                "predicate": "APPLIED",
                "object": "-",
                "outcome": outcome,
            })
        self.save()

    # -- queries ---------------------------------------------------------------------

    @property
    def facts(self) -> List[Dict[str, Any]]:
        return list(self._data["facts"])

    def trials(self) -> Dict[int, Dict[str, Any]]:
        return {int(k): v for k, v in self._data["trials"].items()}

    def best_trial(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        candidates = [
            (t, info) for t, info in self.trials().items()
            if info.get("val_loss") is not None
            and info.get("outcome") in GOOD_OUTCOMES
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1]["val_loss"])

    def wins(self) -> List[Dict[str, Any]]:
        return [f for f in self._data["facts"]
                if f.get("outcome") in GOOD_OUTCOMES]

    def dead_ends(self) -> List[Dict[str, Any]]:
        return [f for f in self._data["facts"]
                if f.get("outcome") in BAD_OUTCOMES]

    def param_history(self, subject_contains: str) -> List[Dict[str, Any]]:
        return [
            f for f in self._data["facts"]
            if f["predicate"] == "CHANGED"
            and subject_contains.lower() in f["subject"].lower()
        ]

    def techniques_tried(self) -> Dict[str, List[str]]:
        """technique -> list of outcomes, in trial order."""
        out: Dict[str, List[str]] = {}
        for f in self._data["facts"]:
            if f["predicate"] == "APPLIED":
                name = f["subject"].replace("technique:", "")
                out.setdefault(name, []).append(f["outcome"])
        return out

    # -- rendering ---------------------------------------------------------------------

    @staticmethod
    def _fact_line(f: Dict[str, Any]) -> str:
        obj = f" {f['object']}" if f.get("object") and f["object"] != "-" else ""
        return f"  - t{f['trial']}: {f['subject']}{obj} => {f['outcome']}"

    def render_context(self, max_chars: int = 2000) -> str:
        """Dense markdown digest for prompt injection."""
        if not self._data["facts"] and not self._data["trials"]:
            return ""
        lines: List[str] = ["## Experiment knowledge graph (exact facts, auto-generated)"]

        best = self.best_trial()
        if best:
            t, info = best
            lines.append(
                f"- Best so far: trial {t}, val_loss={info['val_loss']:.4f}"
                + (f", commit {str(info.get('commit'))[:8]}" if info.get("commit") else "")
            )

        wins = [f for f in self.wins() if f["predicate"] == "CHANGED"]
        if wins:
            lines.append("- Changes that HELPED (keep building on these):")
            lines += [self._fact_line(f) for f in wins[-8:]]

        dead = self.dead_ends()
        if dead:
            lines.append("- DEAD ENDS (do NOT repeat these):")
            lines += [self._fact_line(f) for f in dead[-10:]]

        techniques = self.techniques_tried()
        if techniques:
            summary = ", ".join(
                f"{name} [{','.join(outs[-3:])}]"
                for name, outs in list(techniques.items())[:10]
            )
            lines.append(f"- Techniques tried: {summary}")

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 22] + "\n... [graph truncated]"
        return text
