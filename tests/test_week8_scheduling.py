"""Regression checks for Week 8 arrival and queueing constraints."""

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Code'))
pulp = pytest.importorskip('pulp')
import week8_analysis as week8


def test_lp_never_starts_before_arrival_and_allows_queueing():
    sessions = pd.DataFrame({
        'arrival_time_min': [10 * 60 + 29, 10 * 60 + 29],
        'duration_slots': [2, 2],
        'service_duration_min': [60.0, 60.0],
        'max_shift_slots': [0, 0],
        'energy_kwh': [10.0, 10.0],
        'flexibility_tier': ['inflexible', 'inflexible'],
        'is_carry_in': [False, False],
    })
    result = week8.solve_lp_day(
        sessions, s=1, tou_lookup={hour: 1.0 for hour in range(24)})
    assert result['is_feasible']
    assert min(result['assigned_slots']) * week8.SLOT_MINUTES >= 10 * 60 + 29
