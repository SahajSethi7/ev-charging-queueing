"""Canonical repository paths and serialization helpers.

All analysis scripts should resolve inputs and outputs from this module rather
than from the caller's current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
REPORTS_DIR = PROJECT_ROOT / "Reports"
SIM_ENGINE_PATH = RESULTS_DIR / "week5_results" / "sim_engine.py"


def to_builtin(value: Any) -> Any:
    """Recursively convert NumPy/Pandas-compatible scalars for strict JSON."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
