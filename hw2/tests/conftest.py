"""Shared pytest fixtures.

By default the tests run against ``src.predictor`` (the module students
complete). A different module can be selected via the ``PREDICTOR_MODULE``
environment variable, e.g. for the staff reference solution::

    python -m pytest tests                                # student module
    PREDICTOR_MODULE=src.predictor_solved python -m pytest tests
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


# Make sure the project root is importable so that ``import src.xxx`` works
# regardless of where pytest was launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def predictor_module():
    """Import and return the predictor module under test."""
    name = os.environ.get("PREDICTOR_MODULE", "src.predictor")
    return importlib.import_module(name)


@pytest.fixture(scope="session")
def traces_dir() -> Path:
    return PROJECT_ROOT / "traces"
