"""Pytest configuration.

Adds the repository root to ``sys.path`` so tests can import modules without
installing the package (matching the in-repo import layout the server uses).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENV_STRUCTURED_LOGGING", "false")
