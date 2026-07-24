"""Code knowledge graph — an AST-derived map of the experiment workspace.

Inspired by code-graph projects (codebase-memory-mcp, CodeGraph, aider's
repo map), implemented natively on the stdlib ``ast`` module so it adds
zero dependencies and rebuilds in milliseconds for experiment-sized
repos.

What it provides to the orchestration loop:

- **Repo map** (`render_map`): a dense, token-budgeted digest of every
  module — constants/hyperparameters, functions with their call edges,
  classes with methods — injected into agent prompts so agents answer
  structural questions from the map instead of re-reading files.
- **Trial diffing** (`constants_diff`, `changed_functions`): snapshots
  taken before/after an Editor turn yield the exact hyperparameter
  changes (``LEARNING_RATE: 0.01 -> 0.005``) and touched functions.
  These feed the Evaluator (it sees *what* changed, not just the
  metrics) and the experiment knowledge graph (change -> outcome facts).
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".agent_memory",
             "node_modules", ".mypy_cache", ".pytest_cache"}
MAX_FILES = 200


@dataclass
class FunctionInfo:
    name: str
    lineno: int
    args: List[str]
    calls: List[str]
    source_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lineno": self.lineno, "args": self.args,
                "calls": self.calls, "source_hash": self.source_hash}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FunctionInfo":
        return cls(d["name"], d["lineno"], d.get("args", []),
                   d.get("calls", []), d.get("source_hash", ""))


@dataclass
class ClassInfo:
    name: str
    lineno: int
    bases: List[str]
    methods: List[FunctionInfo]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lineno": self.lineno, "bases": self.bases,
                "methods": [m.to_dict() for m in self.methods]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassInfo":
        return cls(d["name"], d["lineno"], d.get("bases", []),
                   [FunctionInfo.from_dict(m) for m in d.get("methods", [])])


@dataclass
class ModuleInfo:
    path: str  # relative to workspace root
    constants: Dict[str, str] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "constants": self.constants,
            "imports": self.imports,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleInfo":
        return cls(
            path=d["path"],
            constants=d.get("constants", {}),
            imports=d.get("imports", []),
            functions=[FunctionInfo.from_dict(f) for f in d.get("functions", [])],
            classes=[ClassInfo.from_dict(c) for c in d.get("classes", [])],
            parse_error=d.get("parse_error"),
        )


# ---------------------------------------------------------------------------
# AST extraction helpers
# ---------------------------------------------------------------------------

def _call_names(node: ast.AST) -> List[str]:
    """Names of everything called inside *node* (function-level edges)."""
    names: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                names.append(f.id)
            elif isinstance(f, ast.Attribute):
                names.append(f.attr)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:20]


def _function_info(node: ast.AST, source: str) -> FunctionInfo:
    args = [a.arg for a in node.args.args]
    segment = ast.get_source_segment(source, node) or ""
    return FunctionInfo(
        name=node.name,
        lineno=node.lineno,
        args=args,
        calls=_call_names(node),
        source_hash=hashlib.md5(segment.encode("utf-8")).hexdigest()[:12],
    )


def _literal_repr(node: ast.AST) -> Optional[str]:
    """Short repr for constant-ish values (numbers, strings, bools, small
    containers). Returns None for anything non-literal."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    r = repr(value)
    return r if len(r) <= 60 else r[:57] + "..."


def _parse_module(path: Path, root: Path) -> ModuleInfo:
    rel = str(path.relative_to(root))
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ModuleInfo(path=rel, parse_error=f"SyntaxError: {exc}")
    except OSError as exc:
        return ModuleInfo(path=rel, parse_error=str(exc))

    info = ModuleInfo(path=rel)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                info.imports += [a.name for a in node.names]
            else:
                info.imports.append(node.module or ".")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(_function_info(node, source))
        elif isinstance(node, ast.ClassDef):
            methods = [
                _function_info(n, source) for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            info.classes.append(ClassInfo(node.name, node.lineno, bases, methods))
        elif isinstance(node, ast.Assign):
            # UPPER_SNAKE top-level literals = tunable hyperparameters.
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    lit = _literal_repr(node.value)
                    if lit is not None:
                        info.constants[target.id] = lit
    info.imports = sorted(set(info.imports))[:25]
    return info


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

class CodeGraph:
    def __init__(self, modules: Optional[Dict[str, ModuleInfo]] = None) -> None:
        self.modules: Dict[str, ModuleInfo] = modules or {}

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, root: Path, files: Optional[List[str]] = None) -> "CodeGraph":
        """Parse the workspace. With *files*, only those paths are parsed
        (plus nothing else); otherwise all .py files under *root* are
        walked, skipping vendored/artifact directories."""
        root = Path(root).resolve()
        paths: List[Path] = []
        if files:
            for f in files:
                p = (root / f).resolve()
                if p.exists() and p.suffix == ".py":
                    paths.append(p)
        else:
            for p in sorted(root.rglob("*.py")):
                if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
                    continue
                paths.append(p)
                if len(paths) >= MAX_FILES:
                    break
        modules = {}
        for p in paths:
            mi = _parse_module(p, root)
            modules[mi.path] = mi
        return cls(modules)

    # -- persistence ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"modules": {k: v.to_dict() for k, v in self.modules.items()}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CodeGraph":
        return cls({k: ModuleInfo.from_dict(v)
                    for k, v in d.get("modules", {}).items()})

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["CodeGraph"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    # -- rendering ---------------------------------------------------------------

    def render_map(self, max_chars: int = 2400) -> str:
        """Dense repo digest for prompt injection."""
        lines: List[str] = ["## Code map (auto-generated from AST)"]
        for path in sorted(self.modules):
            m = self.modules[path]
            if m.parse_error:
                lines.append(f"- {path}: UNPARSEABLE ({m.parse_error})")
                continue
            parts: List[str] = []
            if m.constants:
                consts = ", ".join(f"{k}={v}" for k, v in m.constants.items())
                parts.append(f"consts[{consts}]")
            for fn in m.functions:
                calls = f" -> {','.join(fn.calls[:6])}" if fn.calls else ""
                parts.append(f"def {fn.name}({','.join(fn.args)}){calls}")
            for c in m.classes:
                bases = f"({','.join(c.bases)})" if c.bases else ""
                methods = ",".join(mm.name for mm in c.methods)
                parts.append(f"class {c.name}{bases}[{methods}]")
            if m.imports:
                parts.append(f"imports[{','.join(m.imports[:10])}]")
            lines.append(f"- {path}: " + "; ".join(parts))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 22] + "\n... [map truncated]"
        return text

    # -- diffing -----------------------------------------------------------------

    def constants_diff(self, other: "CodeGraph") -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """Hyperparameter changes from self (old) to other (new).

        Returns [(module, NAME, old_value, new_value)]; old/new is None
        when the constant was added/removed.
        """
        changes: List[Tuple[str, str, Optional[str], Optional[str]]] = []
        paths = set(self.modules) | set(other.modules)
        for path in sorted(paths):
            old_c = self.modules.get(path).constants if path in self.modules else {}
            new_c = other.modules.get(path).constants if path in other.modules else {}
            for name in sorted(set(old_c) | set(new_c)):
                ov, nv = old_c.get(name), new_c.get(name)
                if ov != nv:
                    changes.append((path, name, ov, nv))
        return changes

    def changed_functions(self, other: "CodeGraph") -> List[str]:
        """Functions/methods whose source changed between snapshots."""
        changed: List[str] = []

        def index(graph: "CodeGraph") -> Dict[str, str]:
            idx: Dict[str, str] = {}
            for path, m in graph.modules.items():
                if m.parse_error:
                    continue
                for fn in m.functions:
                    idx[f"{path}::{fn.name}"] = fn.source_hash
                for c in m.classes:
                    for mm in c.methods:
                        idx[f"{path}::{c.name}.{mm.name}"] = mm.source_hash
            return idx

        old_idx, new_idx = index(self), index(other)
        for key in sorted(set(old_idx) | set(new_idx)):
            if old_idx.get(key) != new_idx.get(key):
                changed.append(key)
        return changed


def format_changes(
    const_changes: List[Tuple[str, str, Optional[str], Optional[str]]],
    func_changes: List[str],
) -> str:
    """Human/agent-readable digest of a trial's code changes."""
    if not const_changes and not func_changes:
        return ""
    lines = []
    if const_changes:
        lines.append("Hyperparameter/constant changes:")
        for module, name, old, new in const_changes:
            if old is None:
                lines.append(f"  + {module}:{name} = {new} (added)")
            elif new is None:
                lines.append(f"  - {module}:{name} (removed, was {old})")
            else:
                lines.append(f"  * {module}:{name}: {old} -> {new}")
    if func_changes:
        lines.append("Functions with modified implementations:")
        for key in func_changes[:15]:
            lines.append(f"  * {key}")
        if len(func_changes) > 15:
            lines.append(f"  ... and {len(func_changes) - 15} more")
    return "\n".join(lines)
