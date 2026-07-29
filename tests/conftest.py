"""Pytest config — shared fixtures, sys.path bootstrap."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return ROOT / "tests" / "data" / "images"


@pytest.fixture(scope="session")
def ensure_dataset(sample_dir: Path) -> Path:
    if not any(sample_dir.glob("*.png")):
        from scripts import make_synthetic_dataset as gen
        gen.main(sample_dir)
    return sample_dir
