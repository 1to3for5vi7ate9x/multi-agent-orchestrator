"""Bridge to the Haskell judge/referee decision kernel (``mao-kernel``).

The judge aggregation and referee rules are the trust-critical decision
core of the loop, so their canonical implementation is a small, pure
Haskell program (see ``haskell/`` at the repo root): total functions
over typed inputs, no IO beyond one JSON request on stdin and one JSON
response on stdout, property-testable, and independent of the Python
process that hosts the agents.

Resolution order:
1. ``$MAO_KERNEL`` — explicit path to the compiled kernel binary.
2. ``mao-kernel`` on PATH.
3. None — callers fall back to the built-in Python implementation,
   which is kept behavior-identical and is verified against the same
   golden vectors (``tests/kernel_vectors.json``) as the Haskell one.

Every call is fail-open: a missing, crashing, or garbled kernel yields
None and the Python fallback takes over — the loop never dies because
of the kernel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, Optional

KERNEL_ENV = "MAO_KERNEL"
KERNEL_BINARY = "mao-kernel"
KERNEL_TIMEOUT = 30.0


def kernel_path() -> Optional[str]:
    explicit = os.environ.get(KERNEL_ENV)
    if explicit:
        return explicit if os.path.isfile(explicit) and os.access(
            explicit, os.X_OK) else None
    return shutil.which(KERNEL_BINARY)


def call(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Send one request to the kernel. None on any failure (fail-open)."""
    path = kernel_path()
    if not path:
        return None
    try:
        proc = subprocess.run(
            [path],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=KERNEL_TIMEOUT,
        )
        if proc.returncode != 0:
            return None
        out = json.loads(proc.stdout)
        return out if isinstance(out, dict) else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError,
            ValueError):
        return None
