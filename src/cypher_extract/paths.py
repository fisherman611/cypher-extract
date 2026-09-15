"""Shared dataset locations for the project."""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "CYPHER_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def get_data_root() -> Path:
    """Return the repository ``data/`` directory, allowing an explicit override."""

    configured = os.environ.get(DATA_ROOT_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_DATA_ROOT
