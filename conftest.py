"""Root-level pytest configuration.

Ensures the project root is importable as top-level packages (config,
core, acquisition, ...) regardless of the working directory pytest is
invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
