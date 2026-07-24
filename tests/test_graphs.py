"""Tests for core/code_graph.py and core/knowledge_graph.py."""
import json
import sys
import tempfile
from pathlib import Path



from ml_orchestrator.core.code_graph import CodeGraph, format_changes
from ml_orchestrator.core.knowledge_graph import ExperimentKnowledgeGraph, extract_techniques

passed = failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")

SRC_V1 = '''
import json
import math

LEARNING_RATE = 0.01
HIDDEN_DIM = 32
EPOCHS = 30

def make_data(n):
    return [math.sin(i) for i in range(n)]

def train():
    data = make_data(100)
    write_metrics(data)

class MLP:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x):
        return x
'''

SRC_V2 = '''
import json
import math

LEARNING_RATE = 0.005
HIDDEN_DIM = 32
EPOCHS = 30
WEIGHT_DECAY = 0.001

def make_data(n):
    return [math.sin(i) for i in range(n)]

def train():
    data = make_data(200)
    write_metrics(data)

class MLP:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x):
        return x * 2
'''

# ---- code graph -------------------------------------------------------------
print("== code graph ==")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "train.py").write_text(SRC_V1)
    (td / ".venv").mkdir()
    (td / ".venv" / "junk.py").write_text("SHOULD_NOT_APPEAR = 1")
    (td / "broken.py").write_text("def broken(:\n")

    g1 = CodeGraph.build(td)
    check("modules found", set(g1.modules) == {"train.py", "broken.py"},
          set(g1.modules))
    m = g1.modules["train.py"]
    check("constants", m.constants == {"LEARNING_RATE": "0.01",
                                       "HIDDEN_DIM": "32", "EPOCHS": "30"},
          m.constants)
    check("functions", [f.name for f in m.functions] == ["make_data", "train"])
    check("call edges", "make_data" in
          [c for f in m.functions if f.name == "train" for c in f.calls])
    check("classes", g1.modules["train.py"].classes[0].name == "MLP"
          and [mm.name for mm in m.classes[0].methods] == ["__init__", "forward"])
    check("parse error captured",
          g1.modules["broken.py"].parse_error is not None)

    rmap = g1.render_map()
    check("map has consts", "LEARNING_RATE=0.01" in rmap, rmap)
    check("map has call edge", "def train() -> make_data" in rmap, rmap)
    check("map skips venv", "SHOULD_NOT_APPEAR" not in rmap)
    check("map budget", len(g1.render_map(max_chars=100)) <= 100)

    # persistence roundtrip
    g1.save(td / "cg.json")
    g1b = CodeGraph.load(td / "cg.json")
    check("roundtrip", g1b.render_map() == g1.render_map())
    check("load missing", CodeGraph.load(td / "nope.json") is None)

    # diffing v1 -> v2
    (td / "train.py").write_text(SRC_V2)
    g2 = CodeGraph.build(td)
    consts = g1.constants_diff(g2)
    check("const change", ("train.py", "LEARNING_RATE", "0.01", "0.005") in consts,
          consts)
    check("const added", ("train.py", "WEIGHT_DECAY", None, "0.001") in consts)
    check("unchanged not reported",
          not any(c[1] == "HIDDEN_DIM" for c in consts))
    funcs = g1.changed_functions(g2)
    check("func change", "train.py::train" in funcs and
          "train.py::MLP.forward" in funcs, funcs)
    check("unchanged func not reported", "train.py::make_data" not in funcs)

    text = format_changes(consts, funcs)
    check("format_changes", "LEARNING_RATE: 0.01 -> 0.005" in text
          and "train.py::MLP.forward" in text, text)
    check("format empty", format_changes([], []) == "")

# ---- technique extraction -----------------------------------------------------
print("== knowledge graph ==")
t = extract_techniques("Lowered the learning rate and added dropout + weight decay")
check("techniques", t == ["learning-rate adjustment", "weight decay / L2",
                          "dropout"], t)
check("techniques empty", extract_techniques("") == [])

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    kg = ExperimentKnowledgeGraph(td / "kg.json")
    kg.record_trial(0, "BASELINE", "BASELINE", 0.40, "c0")
    kg.record_trial(
        1, "CRASHED", "REVERTED", None, None,
        const_changes=[("train.py", "HIDDEN_DIM", "32", "4096")],
        func_changes=["train.py::train"],
        editor_summary="Massively increased hidden width",
    )
    kg.record_trial(
        2, "IMPROVED", "COMMITTED", 0.30, "c2",
        const_changes=[("train.py", "LEARNING_RATE", "0.01", "0.005")],
        editor_summary="Lowered learning rate",
    )
    kg.record_trial(
        3, "GOAL_REACHED", "GOAL_COMMIT", 0.20, "c3",
        const_changes=[("train.py", "WEIGHT_DECAY", None, "0.001")],
        editor_summary="Added weight decay",
    )

    best = kg.best_trial()
    check("best trial", best[0] == 3 and best[1]["val_loss"] == 0.20, best)
    dead = kg.dead_ends()
    check("dead ends", any("HIDDEN_DIM" in f["subject"] for f in dead), dead)
    wins = kg.wins()
    check("wins", any("LEARNING_RATE" in f["subject"] for f in wins))
    hist = kg.param_history("LEARNING_RATE")
    check("param history", len(hist) == 1 and hist[0]["trial"] == 2)
    techs = kg.techniques_tried()
    check("technique outcomes",
          techs.get("learning-rate adjustment") == ["IMPROVED"]
          and techs.get("architecture change") == ["CRASHED"], techs)

    ctx = kg.render_context()
    check("context best", "Best so far: trial 3" in ctx, ctx)
    check("context dead ends", "DEAD ENDS" in ctx and "HIDDEN_DIM" in ctx)
    check("context wins", "HELPED" in ctx and "LEARNING_RATE" in ctx)
    check("context budget", len(kg.render_context(max_chars=150)) <= 150)

    # persistence across instances
    kg2 = ExperimentKnowledgeGraph(td / "kg.json")
    check("kg persistence", len(kg2.facts) == len(kg.facts)
          and kg2.best_trial()[0] == 3)

    # corrupt file tolerated
    (td / "kg.json").write_text("{{{")
    kg3 = ExperimentKnowledgeGraph(td / "kg.json")
    check("kg corrupt recovery", kg3.facts == [])

# ---- memory preamble integration ------------------------------------------------
print("== preamble with context provider ==")
from ml_orchestrator.core.session import MemoryStore
with tempfile.TemporaryDirectory() as td:
    ms = MemoryStore(Path(td) / "mem")
    ms.context_provider = lambda: "## Experiment knowledge graph\n- fact A"
    pre = ms.render_preamble("editor")
    check("provider-only preamble", "LIVE PROJECT KNOWLEDGE" in pre
          and "fact A" in pre and "RESTORED MEMORY" not in pre)
    ms.save_snapshot("editor", "# MEMORY SNAPSHOT\nold wisdom", "t", 1)
    pre = ms.render_preamble("editor")
    check("both layers", "RESTORED MEMORY" in pre and "old wisdom" in pre
          and "fact A" in pre)
    ms.context_provider = lambda: 1 / 0  # broken provider must not raise
    pre = ms.render_preamble("editor")
    check("broken provider safe", "old wisdom" in pre)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
