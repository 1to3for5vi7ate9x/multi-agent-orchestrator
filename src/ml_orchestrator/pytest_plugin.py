"""Pytest bridge for script-style test suites.

Some repos (including this one) ship suites as standalone scripts: checks
run at module level and the file ends with sys.exit(...). Importing such a
file aborts pytest collection with INTERNALERROR (SystemExit), so plain
`pytest -q` can never report pass/fail counts for them.

This plugin detects script-style suites statically — module-level
sys.exit()/exit() present and no top-level test functions — and collects
each as a single item that runs the script in a subprocess, the same
execution model as tests/run_all.py. The item passes iff the script exits
0, so failures propagate honestly. Ordinary pytest-style modules are left
to native collection untouched.

Loaded two ways: the `pytest11` entry point (effective once the
distribution's metadata is rebuilt on install) and `-p
ml_orchestrator.pytest_plugin` via addopts in pyproject.toml, which works
immediately in an editable checkout. The entry point is named with the
full import path so pytest recognizes both registrations as the same
plugin and does not register it twice.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def _is_script_style(path: Path) -> bool:
    """True for modules that sys.exit() at import and define no tests."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False

    has_test_defs = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
        for node in tree.body
    )
    if has_test_defs:
        return False

    calls_exit = False

    class _Finder(ast.NodeVisitor):
        # Do not descend into defs/classes: only module-level exits count.
        def visit_FunctionDef(self, node):
            pass

        def visit_AsyncFunctionDef(self, node):
            pass

        def visit_ClassDef(self, node):
            pass

        def visit_Call(self, node):
            nonlocal calls_exit
            func = node.func
            if isinstance(func, ast.Attribute):
                if (func.attr == "exit"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "sys"):
                    calls_exit = True
            elif isinstance(func, ast.Name) and func.id == "exit":
                calls_exit = True
            self.generic_visit(node)

    _Finder().visit(tree)
    return calls_exit


class ScriptSuiteFailure(Exception):
    """Carries the captured output of a failed suite subprocess."""


class ScriptSuiteItem(pytest.Item):
    def runtest(self):
        proc = subprocess.run(
            [sys.executable, str(self.path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(self.config.rootpath),
        )
        if proc.returncode != 0:
            raise ScriptSuiteFailure(
                f"{self.path.name} exited {proc.returncode}\n"
                f"{proc.stdout}\n{proc.stderr}"
            )

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, ScriptSuiteFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, 0, self.name


class ScriptSuiteFile(pytest.Module):
    def collect(self):
        # Deliberately no import: the script runs in a subprocess instead.
        yield ScriptSuiteItem.from_parent(
            self, name=self.path.stem, path=self.path
        )


def pytest_pycollect_makemodule(module_path, parent):
    if _is_script_style(module_path):
        return ScriptSuiteFile.from_parent(parent, path=module_path)
    return None
