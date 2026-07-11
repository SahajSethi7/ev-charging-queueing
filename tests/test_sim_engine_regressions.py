"""Focused regressions for the simulation defects found in the audit."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / 'Results' / 'week5_results' / 'sim_engine.py'
SPEC = importlib.util.spec_from_file_location('audited_sim_engine', ENGINE_PATH)
ENGINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENGINE
SPEC.loader.exec_module(ENGINE)


def base_config(n_chargers: int):
    return ENGINE.StationConfig(
        station_name='test',
        n_chargers=n_chargers,
        arrival_mode='poisson',
        constant_lambda=0.5,
        service_params={
            'DC_Fast': {
                'best_distribution': 'exponential',
                'mean_min': 20.0,
                'fit_params': {'scale': 20.0},
            },
        },
        charger_type_mix={'DC_Fast': 1.0},
        faults_enabled=False,
        sim_days=1,
        warmup_days=0,
        random_seed=7001,
    )


def test_exogenous_customer_count_is_fleet_independent():
    small = ENGINE.simulate_station(base_config(2), n_replications=3)
    large = ENGINE.simulate_station(base_config(6), n_replications=3)
    assert [row['n_sessions'] for row in small] == [
        row['n_sessions'] for row in large]


def test_fault_retries_preserve_customer_identity():
    config = base_config(4)
    config.faults_enabled = True
    config.fault_rates = {'DC_Fast': 1.0}
    config.max_fault_retries = 2
    results = ENGINE.simulate_station(config, n_replications=1)
    result = results[0]
    assert result['n_attempts'] == 3 * result['n_sessions']
    assert result['n_failed_after_retries'] == result['n_sessions']
    assert result['throughput_per_hour'] == 0.0
