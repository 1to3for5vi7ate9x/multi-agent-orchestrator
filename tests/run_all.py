"""Run every test suite; exit nonzero if any fails.

Usage: uv run python tests/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SUITES = sorted(p for p in HERE.glob("test_*.py"))

failures = []
for suite in SUITES:
    print(f"\n=== {suite.name} ===")
    proc = subprocess.run([sys.executable, str(suite)])
    if proc.returncode != 0:
        failures.append(suite.name)

print("\n" + "=" * 40)
if failures:
    print(f"FAILED suites: {', '.join(failures)}")
    sys.exit(1)
print(f"All {len(SUITES)} suites passed.")
