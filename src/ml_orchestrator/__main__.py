"""Allow ``python -m ml_orchestrator`` as an alias for the CLI."""

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main())
