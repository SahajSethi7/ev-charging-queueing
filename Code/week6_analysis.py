"""
Week 6: SimPy Validation + Baseline FCFS Pareto Frontiers + Fault Tax
======================================================================
Prerequisites:
    - sim_engine.py (from Week 5)
    - parameter_summary.json
    - nhpp_rate_functions.csv
    - erlang_c_results_4rep.csv
    - mgs_comparison_4rep.csv

Usage:
    python week6_analysis.py [--output-dir OUTPUT_DIR] [--reps 50] [--sim-days 30]
                              [--skip-validation] [--skip-sweeps] [--skip-fault-tax]

Outputs:
    - Figures: pareto_*.png, fault_tax_*.png, validation_*.png
    - Data: pareto_sweep_results.csv, fault_tax_results.csv,
            fault_sensitivity.csv, week6_metadata.json
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import sem
from project_paths import CODE_DIR, RESULTS_DIR, to_builtin

# ── Ensure sim_engine is importable ──────────────────────────────────
# Search the project root and the Week 5 output directory, since the
# generated engine lives in week5_results by default.
SCRIPT_DIR = CODE_DIR

# sim_engine is loaded after CLI parsing so --week5-dir controls the actual
# imported module, not only an incidental sys.path search order.
StationConfig = None
simulate_station = None
erlang_c_pwait = None
DEFAULT_FAULT_RATES = None
validate_mms = None
HAS_SIMPY = False
SIM_ENGINE_INFO = {}


# =====================================================================
# CONSTANTS
# =====================================================================

# The 4 representative stations from Week 4 fleet sizing
REPRESENTATIVE_STATIONS = [
    'Xiuzhou_Expressway Service District A',  # expressway, DC_Fast heavy
    'Nanhu_Technology Park',                   # mixed, DC_Fast dominant
    'Xiuzhou_Government Agency',               # institutional, L2 heavy
    'Tongxiang_Bus Station',                   # high-volume, L2 heavy
]

STATION_LABELS = {
    'Xiuzhou_Expressway Service District A': 'Expressway A (DC Fast)',
    'Nanhu_Technology Park': 'Technology Park (Mixed)',
    'Xiuzhou_Government Agency': 'Gov Agency (L2)',
    'Tongxiang_Bus Station': 'Bus Station (L2, High-Vol)',
}

# Fleet size sweep range
S_VALUES = list(range(2, 32, 2))  # 2, 4, 6, ..., 30

# Fault sensitivity sweep
FAULT_SENSITIVITY_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FAULT_SENSITIVITY_STATION = 'Xiuzhou_Government Agency'  # L2-heavy, most affected

# Plot styling
COLORS = {
    'sim_faults_on': '#2166ac',
    'sim_faults_off': '#b2182b',
    'erlang_c': '#4daf4a',
    'mgs': '#ff7f00',
}


# =====================================================================
# HELPERS
# =====================================================================

def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_signature(path: Path) -> dict:
    p = Path(path)
    return {
        'path': str(p.resolve()),
        'sha256': sha256_file(p),
        'mtime': os.path.getmtime(p),
        'size_bytes': os.path.getsize(p),
    }


def load_sim_engine_from_week5(week5_dir: Path) -> dict:
    """Import the exact Week 5 engine selected by --week5-dir."""
    global StationConfig, simulate_station, erlang_c_pwait
    global DEFAULT_FAULT_RATES, validate_mms, HAS_SIMPY, SIM_ENGINE_INFO

    engine_path = (week5_dir / 'sim_engine.py').resolve()
    if not engine_path.exists():
        raise FileNotFoundError(
            f"Cannot find sim_engine.py in requested Week 5 dir: {week5_dir}"
        )

    spec = importlib.util.spec_from_file_location('week6_sim_engine', engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {engine_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    StationConfig = module.StationConfig
    simulate_station = module.simulate_station
    erlang_c_pwait = module.erlang_c_pwait
    DEFAULT_FAULT_RATES = module.DEFAULT_FAULT_RATES
    validate_mms = module.validate_mms
    HAS_SIMPY = bool(module.HAS_SIMPY)
    SIM_ENGINE_INFO = file_signature(engine_path)

    required = ['fault_repair_median_min', 'activation_schedule',
                'capacity_event_log', 'fault_repair_duration']
    source = engine_path.read_text(encoding='utf-8')
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError(
            "Selected sim_engine.py is stale; missing patched markers: "
            + ', '.join(missing)
        )

    return SIM_ENGINE_INFO


def provenance_signature(paths: Dict[str, str],
                         n_reps: int,
                         sim_days: int,
                         sim_engine_info: dict) -> dict:
    return {
        'inputs': {
            label: file_signature(Path(path))
            for label, path in paths.items()
        },
        'sim_engine': sim_engine_info,
        'week6_analysis': file_signature(Path(__file__).resolve()),
        'run_parameters': {
            'n_replications': n_reps,
            'sim_days': sim_days,
            's_values': S_VALUES,
        },
    }


def warn_if_cached_outputs_stale(output_dir: Path, current_provenance: dict):
    meta_path = output_dir / 'week6_metadata.json'
    if not meta_path.exists():
        print("  [WARN] No week6_metadata.json found; cannot verify cached sweeps.")
        return

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            previous = json.load(f)
    except Exception as exc:
        print(f"  [WARN] Could not read {meta_path}: {exc}")
        return

    old = previous.get('provenance')
    if not old:
        print("  [WARN] Cached Week 6 outputs have no provenance hash block. "
              "Treat them as pre-guard outputs unless rerun.")
        return

    mismatches = []
    for label, sig in current_provenance.get('inputs', {}).items():
        old_sig = old.get('inputs', {}).get(label, {})
        if old_sig.get('sha256') != sig.get('sha256'):
            mismatches.append(label)

    old_engine = old.get('sim_engine', {})
    if old_engine.get('sha256') != current_provenance.get('sim_engine', {}).get('sha256'):
        mismatches.append('sim_engine.py')

    old_params = old.get('run_parameters', {})
    if old_params.get('n_replications') != current_provenance['run_parameters']['n_replications']:
        mismatches.append('n_replications')
    if old_params.get('sim_days') != current_provenance['run_parameters']['sim_days']:
        mismatches.append('sim_days')

    if mismatches:
        print("  " + "!" * 60)
        print("  WARNING: Cached Week 6 sweep CSVs may be stale.")
        print("  Mismatched provenance fields: " + ', '.join(mismatches))
        print("  Re-run without --skip-sweeps before using final conclusions.")
        print("  " + "!" * 60)
    else:
        print("  Cached Week 6 outputs match current provenance.")


def load_station_config_accurate(
    param_path: str,
    nhpp_path: str,
    mgs_path: str,
    station: str,
    n_chargers: int,
    faults_enabled: bool = True,
    fault_rate_override: Optional[float] = None,
    fault_repair_median_min: float = 30.0,
    sim_days: int = 30,
    seed: int = 42,
) -> StationConfig:
    """
    Build a StationConfig using the EMPIRICAL per-station charger type mix
    from mgs_comparison_4rep.csv instead of the coarse heuristic in
    sim_engine.load_config.

    If fault_rate_override is set, ALL charger types use that single rate
    (for the sensitivity sweep). Otherwise, per-type defaults apply.

    fault_repair_median_min: median charger downtime after a fault (minutes).
    Default 30 min. This is a modeling assumption: the dataset does not
    record charger recovery time, only the failed session duration (~1 min).
    """
    with open(param_path, 'r') as f:
        params = json.load(f)

    nhpp_df = pd.read_csv(nhpp_path)
    station_rates = nhpp_df[nhpp_df['station'] == station].sort_values('hour')
    if len(station_rates) == 0:
        raise ValueError(f"Station '{station}' not in NHPP rates")
    nhpp_24 = station_rates['lambda_mean'].values

    # Service time params from parameter_summary
    service_params = {}
    for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
        if ctype in params.get('service_time', {}):
            service_params[ctype] = params['service_time'][ctype]

    # Empirical charger type mix from mgs_comparison
    mgs_df = pd.read_csv(mgs_path)
    mgs_row = mgs_df[
        (mgs_df['station'] == station) &
        (mgs_df['sizing_scenario'] == 'historical_avg')
    ]
    if len(mgs_row) > 0:
        mix = json.loads(mgs_row.iloc[0]['station_mix_json'])
    else:
        # Fallback
        mix = {'DC_Fast': 0.4, 'Level_2': 0.4, 'Mixed': 0.2}
        print(f"  [WARN] No empirical mix for {station}, using fallback")

    # Fault rates
    fault_rates = None
    if fault_rate_override is not None:
        fault_rates = {
            'DC_Fast': fault_rate_override,
            'Level_2': fault_rate_override,
            'Mixed': fault_rate_override,
        }

    return StationConfig(
        station_name=station,
        n_chargers=n_chargers,
        arrival_mode='nhpp',
        nhpp_rates=nhpp_24,
        service_params=service_params,
        charger_type_mix=mix,
        faults_enabled=faults_enabled,
        fault_rates=fault_rates,
        fault_repair_median_min=fault_repair_median_min,
        sim_days=sim_days,
        random_seed=seed,
    )


def aggregate_sweep_results(results_list: List[dict]) -> dict:
    """
    Aggregate N replications into mean ± CI for each metric.
    """
    if not results_list:
        return {}
    keys = ['mean_utilization', 'mean_wait', 'median_wait', 'p95_wait',
            'p_wait_gt_10min', 'p_wait_gt_15min', 'throughput_per_hour',
            'n_sessions', 'fault_fraction']
    agg = {}
    for k in keys:
        vals = [r[k] for r in results_list if k in r]
        if vals:
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
            agg[f'{k}_ci95'] = float(1.96 * sem(vals)) if len(vals) > 1 else 0.0
    if 'p_wait_gt_15min_mean' in agg:
        agg['p_wait_gt_15min_upper95'] = (
            agg['p_wait_gt_15min_mean'] + agg.get('p_wait_gt_15min_ci95', 0.0)
        )
    return agg


def save_replication_level(
    rep_results: List[dict],
    station: str,
    n_chargers: int,
    faults: str,
    collector: List[dict],
):
    """Append replication-level metrics for later diagnostics."""
    for r in rep_results:
        collector.append({
            'station': station,
            'n_chargers': n_chargers,
            'faults': faults,
            'replication': r.get('replication', None),
            'seed': r.get('seed', None),
            'mean_utilization': r.get('mean_utilization', None),
            'mean_wait': r.get('mean_wait', None),
            'p_wait_gt_10min': r.get('p_wait_gt_10min', None),
            'p_wait_gt_15min': r.get('p_wait_gt_15min', None),
            'throughput_per_hour': r.get('throughput_per_hour', None),
            'n_sessions': r.get('n_sessions', None),
            'fault_fraction': r.get('fault_fraction', None),
            'drain_cap_hit': r.get('drain_cap_hit', None),
        })


def get_analytical_curves(erlang_path: str, station: str, scenario: str = 'historical_avg'):
    """
    Load Erlang-C (M/M/s) and M/G/s P(wait) from Week 4 results.
    Returns DataFrame with columns: s, p_wait_mms, p_wait_mgs, rho_total.
    """
    ec = pd.read_csv(erlang_path)
    sub = ec[(ec['station'] == station) & (ec['sizing_scenario'] == scenario)]
    if len(sub) == 0:
        print(f"  [WARN] No analytical results for {station}/{scenario}")
        return pd.DataFrame()
    return sub[['s', 'p_wait_mms', 'p_wait_mgs', 'rho_total',
                'peak_lambda_per_hour', 'mu_per_min']].copy()


# =====================================================================
# STEP 1: SimPy M/M/s VALIDATION
# =====================================================================

def run_validation(output_dir: Path, n_reps: int = 200) -> dict:
    """
    Validate SimPy against Erlang-C at two load levels.
    Returns validation results dict.
    """
    print("\n" + "=" * 70)
    print("STEP 1: M/M/s VALIDATION")
    print("=" * 70)

    if not HAS_SIMPY:
        print("  [ERROR] SimPy is not available in the current Python environment.")
        print("  Run Week 6 with the same interpreter used for Week 5, or install simpy.")
        return {
            'all_pass': False,
            'blocked_reason': 'simpy_not_installed',
        }

    results = {}
    test_cases = [
        {'s': 5, 'lam': 3.0, 'mu_min': 60.0, 'label': 'moderate_load'},   # ρ = 0.60
        {'s': 5, 'lam': 4.0, 'mu_min': 60.0, 'label': 'high_load'},       # ρ = 0.80
        {'s': 8, 'lam': 6.0, 'mu_min': 60.0, 'label': 'large_system'},    # ρ = 0.75
    ]

    all_pass = True
    for tc in test_cases:
        print(f"\n  Test: {tc['label']} (s={tc['s']}, λ={tc['lam']}/hr, "
              f"ρ={tc['lam']/(tc['s'] * 60/tc['mu_min']):.3f})")

        vr = validate_mms(
            s=tc['s'],
            lam_per_hour=tc['lam'],
            mean_service_min=tc['mu_min'],
            n_reps=n_reps,
            sim_days=30,
            verbose=True,
        )
        vr['label'] = tc['label']
        vr['s'] = tc['s']
        vr['lambda'] = tc['lam']

        wait_err = abs(vr['simulated_mean_wait'] - vr['analytical_mean_wait']) / \
                   max(vr['analytical_mean_wait'], 0.01)
        util_err = abs(vr['simulated_mean_util'] - vr['analytical_rho'])
        vr['wait_error_pct'] = wait_err * 100
        vr['util_error'] = util_err
        vr['pass'] = wait_err < 0.10 and util_err < 0.02

        if not vr['pass']:
            all_pass = False
            print(f"  ⚠ FAIL: wait_err={wait_err*100:.1f}%, util_err={util_err:.4f}")

        results[tc['label']] = vr

    results['all_pass'] = all_pass

    # Save
    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(to_builtin(results), f, indent=2)

    if not all_pass:
        print("\n  ⚠ WARNING: Not all validation tests passed.")
        print("    Inspect results before proceeding to sweeps.")
    else:
        print("\n  ✓ All validation tests passed.")

    return results


# =====================================================================
# STEP 2: BASELINE FCFS PARETO FRONTIER SWEEPS
# =====================================================================

def run_pareto_sweeps(
    param_path: str,
    nhpp_path: str,
    mgs_path: str,
    erlang_path: str,
    output_dir: Path,
    n_reps: int = 50,
    sim_days: int = 30,
) -> pd.DataFrame:
    """
    For each representative station, sweep fleet size and compute
    utilization–wait tradeoff with faults ON (FCFS baseline).
    """
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE FCFS PARETO FRONTIER SWEEPS")
    print("=" * 70)

    all_rows = []
    rep_level_collector = []

    for station in REPRESENTATIVE_STATIONS:
        print(f"\n  Station: {station}")
        print(f"  Sweep: s = {S_VALUES[0]}..{S_VALUES[-1]}, "
              f"{n_reps} reps × {sim_days} days each")

        t0 = time.time()

        for s in S_VALUES:
            # Fixed seed schedule means replication k uses seed 42+k for every
            # configuration. This provides partial common-random-number
            # variance reduction across ON/OFF and neighboring fleet sizes.
            config = load_station_config_accurate(
                param_path, nhpp_path, mgs_path,
                station=station,
                n_chargers=s,
                faults_enabled=True,
                sim_days=sim_days,
                seed=42,
            )

            rep_results = simulate_station(config, n_replications=n_reps, verbose=False)
            agg = aggregate_sweep_results(rep_results)
            save_replication_level(rep_results, station, s, 'on', rep_level_collector)

            row = {
                'station': station,
                'n_chargers': s,
                'faults': 'on',
                **agg,
            }
            all_rows.append(row)

            p15 = agg.get('p_wait_gt_15min_mean', 0)
            util = agg.get('mean_utilization_mean', 0)
            print(f"    s={s:3d}: util={util:.3f}, P(W>15)={p15:.4f}, "
                  f"E[W]={agg.get('mean_wait_mean', 0):.2f} min")

        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.0f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / 'pareto_sweep_results.csv', index=False)
    print(f"\n  Saved: pareto_sweep_results.csv ({len(df)} rows)")
    rep_df = pd.DataFrame(rep_level_collector)
    rep_df.to_csv(output_dir / 'pareto_sweep_replications.csv', index=False)
    print(f"  Saved: pareto_sweep_replications.csv ({len(rep_df)} rows)")
    return df


def plot_pareto_frontiers(
    sweep_df: pd.DataFrame,
    erlang_path: str,
    output_dir: Path,
):
    """
    Plot Pareto frontiers (utilization vs P(W>15 min)) with analytical overlays.
    One figure per station + one combined 2×2 panel figure.
    """
    print("\n  Generating Pareto frontier plots...")

    fig_combined, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    for idx, station in enumerate(REPRESENTATIVE_STATIONS):
        ax = axes[idx]
        sub = sweep_df[(sweep_df['station'] == station) & (sweep_df['faults'] == 'on')]
        if len(sub) == 0:
            continue

        # Simulated frontier
        ax.errorbar(
            sub['mean_utilization_mean'], sub['p_wait_gt_15min_mean'],
            xerr=sub['mean_utilization_ci95'],
            yerr=sub['p_wait_gt_15min_ci95'],
            fmt='o-', color=COLORS['sim_faults_on'], markersize=5, linewidth=1.5,
            label='Simulated (NHPP + faults)',
            capsize=2, capthick=0.8,
        )

        # Annotate fleet sizes at selected points
        for _, row in sub.iterrows():
            s_val = int(row['n_chargers'])
            if s_val in [4, 8, 12, 16, 20, 24, 28]:
                ax.annotate(
                    f's={s_val}',
                    (row['mean_utilization_mean'], row['p_wait_gt_15min_mean']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=7, color='#333333',
                )

        ax.set_xlabel('Mean Charger Utilization', fontsize=10)
        ax.set_ylabel('P(Wait > 15 min)', fontsize=10)
        ax.set_title(STATION_LABELS.get(station, station), fontsize=11, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(0.02, 0.07, '5% target', fontsize=7, color='gray')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig_combined.suptitle(
        'Baseline FCFS Simulation Frontiers: Utilization vs P(Wait > 15 min)\n'
        'Full-day NHPP simulation; analytical P(W>0) references are reported separately',
        fontsize=12, fontweight='bold',
    )
    fig_combined.tight_layout(rect=[0, 0, 1, 0.94])
    fig_combined.savefig(output_dir / 'pareto_baseline_combined.png', dpi=150)
    plt.close(fig_combined)
    print(f"  Saved: pareto_baseline_combined.png")


# =====================================================================
# STEP 3: FAULT TAX
# =====================================================================

def run_fault_off_sweeps(
    param_path: str,
    nhpp_path: str,
    mgs_path: str,
    output_dir: Path,
    n_reps: int = 50,
    sim_days: int = 30,
) -> pd.DataFrame:
    """
    Rerun the identical sweep with faults OFF for fault tax calculation.
    """
    print("\n" + "=" * 70)
    print("STEP 3a: FAULT-OFF SWEEPS (for fault tax)")
    print("=" * 70)

    all_rows = []
    rep_level_collector = []

    for station in REPRESENTATIVE_STATIONS:
        print(f"\n  Station: {station}")
        t0 = time.time()

        for s in S_VALUES:
            # Same seed schedule as faults ON. Independent component streams
            # and arrival-time customer tapes provide stable CRN pairing.
            config = load_station_config_accurate(
                param_path, nhpp_path, mgs_path,
                station=station,
                n_chargers=s,
                faults_enabled=False,
                sim_days=sim_days,
                seed=42,
            )

            rep_results = simulate_station(config, n_replications=n_reps, verbose=False)
            agg = aggregate_sweep_results(rep_results)
            save_replication_level(rep_results, station, s, 'off', rep_level_collector)

            row = {
                'station': station,
                'n_chargers': s,
                'faults': 'off',
                **agg,
            }
            all_rows.append(row)

            p15 = agg.get('p_wait_gt_15min_mean', 0)
            util = agg.get('mean_utilization_mean', 0)
            print(f"    s={s:3d}: util={util:.3f}, P(W>15)={p15:.4f}")

        print(f"  Elapsed: {time.time()-t0:.0f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / 'fault_off_sweep_results.csv', index=False)
    print(f"\n  Saved: fault_off_sweep_results.csv ({len(df)} rows)")
    rep_df = pd.DataFrame(rep_level_collector)
    rep_df.to_csv(output_dir / 'fault_off_replications.csv', index=False)
    print(f"  Saved: fault_off_replications.csv ({len(rep_df)} rows)")
    return df


def find_s_star(df: pd.DataFrame, target: float = 0.05) -> dict:
    """
    Return both point-estimate and conservative s* values.

    Point estimate: smallest s with mean P(wait>15) < target.
    Conservative: smallest s with upper 95% CI bound < target.
    """
    s_point = None
    s_conservative = None

    for _, row in df.sort_values('n_chargers').iterrows():
        point = row.get('p_wait_gt_15min_mean')
        upper = row.get('p_wait_gt_15min_upper95')
        if s_point is None and pd.notna(point) and point < target:
            s_point = int(row['n_chargers'])
        if s_conservative is None and pd.notna(upper) and upper < target:
            s_conservative = int(row['n_chargers'])

    return {
        's_star_point': s_point,
        's_star_conservative': s_conservative,
        'disagree': (s_point != s_conservative)
                    if (s_point is not None and s_conservative is not None)
                    else None,
    }


def compute_fault_tax(
    sweep_on: pd.DataFrame,
    sweep_off: pd.DataFrame,
    output_dir: Path,
    target_p_wait: float = 0.05,
) -> pd.DataFrame:
    """
    For each station, find s* at P(W>15) < target with faults ON vs OFF.
    Reports both point-estimate and conservative (upper-CI) s* values.
    Fault tax uses the conservative estimate when available.
    """
    print("\n" + "=" * 70)
    print("STEP 3b: FAULT TAX CALCULATION")
    print("=" * 70)

    rows = []
    for station in REPRESENTATIVE_STATIONS:
        on = sweep_on[sweep_on['station'] == station].sort_values('n_chargers')
        off = sweep_off[sweep_off['station'] == station].sort_values('n_chargers')
        ss_on = find_s_star(on, target_p_wait)
        ss_off = find_s_star(off, target_p_wait)
        s_star_on = ss_on['s_star_conservative'] or ss_on['s_star_point']
        s_star_off = ss_off['s_star_conservative'] or ss_off['s_star_point']

        fault_frac = on['fault_fraction_mean'].mean() if 'fault_fraction_mean' in on.columns else None

        row = {
            'station': station,
            'label': STATION_LABELS.get(station, station),
            'target_p_wait_15': target_p_wait,
            's_star_on_point': ss_on['s_star_point'],
            's_star_on_conservative': ss_on['s_star_conservative'],
            's_star_off_point': ss_off['s_star_point'],
            's_star_off_conservative': ss_off['s_star_conservative'],
            's_star_faults_on': s_star_on,
            's_star_faults_off': s_star_off,
            'fault_tax': (s_star_on - s_star_off)
                         if (s_star_on is not None and s_star_off is not None) else None,
            'fault_tax_point': (
                ss_on['s_star_point'] - ss_off['s_star_point']
            ) if (ss_on['s_star_point'] is not None and ss_off['s_star_point'] is not None)
              else None,
            'on_point_vs_conservative_disagree': ss_on['disagree'],
            'off_point_vs_conservative_disagree': ss_off['disagree'],
            'empirical_fault_fraction': fault_frac,
        }
        rows.append(row)

        tax_str = f"{row['fault_tax']}" if row['fault_tax'] is not None else 'N/A (target not reached)'
        point_tax = row['fault_tax_point']
        print(f"  {station}:")
        print(f"    s* ON  (point/conservative): {ss_on['s_star_point']} / {ss_on['s_star_conservative']}")
        print(f"    s* OFF (point/conservative): {ss_off['s_star_point']} / {ss_off['s_star_conservative']}")
        print(f"    Fault tax (conservative):    {tax_str} chargers")
        if point_tax is not None and point_tax != row['fault_tax']:
            print(f"    Fault tax (point est.):      {point_tax} chargers  [differs from conservative]")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'fault_tax_results.csv', index=False)
    print(f"\n  Saved: fault_tax_results.csv")
    return df


def plot_fault_tax(
    sweep_on: pd.DataFrame,
    sweep_off: pd.DataFrame,
    output_dir: Path,
):
    """
    Plot fault-ON vs fault-OFF Pareto frontiers on the same axes.
    """
    print("\n  Generating fault tax comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    for idx, station in enumerate(REPRESENTATIVE_STATIONS):
        ax = axes[idx]
        on = sweep_on[sweep_on['station'] == station].sort_values('n_chargers')
        off = sweep_off[sweep_off['station'] == station].sort_values('n_chargers')

        ax.plot(on['mean_utilization_mean'], on['p_wait_gt_15min_mean'],
                'o-', color=COLORS['sim_faults_on'], markersize=5, linewidth=1.5,
                label='Faults ON (empirical rates)')
        ax.plot(off['mean_utilization_mean'], off['p_wait_gt_15min_mean'],
                's-', color=COLORS['sim_faults_off'], markersize=5, linewidth=1.5,
                label='Faults OFF')

        ax.axhline(0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(0.02, 0.07, '5% target', fontsize=7, color='gray')
        ax.set_xlabel('Mean Charger Utilization', fontsize=10)
        ax.set_ylabel('P(Wait > 15 min)', fontsize=10)
        ax.set_title(STATION_LABELS.get(station, station), fontsize=11, fontweight='bold')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Fault Tax: Impact of Charger Faults on Pareto Frontier\n'
        'Horizontal shift at 5% target = additional chargers needed',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / 'fault_tax_comparison.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: fault_tax_comparison.png")


def run_fault_sensitivity(
    param_path: str,
    nhpp_path: str,
    mgs_path: str,
    output_dir: Path,
    n_reps: int = 50,
    sim_days: int = 30,
) -> pd.DataFrame:
    """
    Vary fault rate from 0% to 30% for one representative station.
    Find s* at P(W>15) < 5% for each rate.
    """
    print("\n" + "=" * 70)
    print("STEP 3c: FAULT SENSITIVITY CURVE")
    print(f"  Station: {FAULT_SENSITIVITY_STATION}")
    print("=" * 70)

    station = FAULT_SENSITIVITY_STATION
    rows = []

    for fault_rate in FAULT_SENSITIVITY_RATES:
        print(f"\n  Fault rate = {fault_rate*100:.0f}%")
        t0 = time.time()

        s_star = None
        for s in S_VALUES:
            # Hold everything fixed except the fault rate:
            # station, NHPP rates, service sampler, charger mix,
            # and seed schedule remain unchanged.
            config = load_station_config_accurate(
                param_path, nhpp_path, mgs_path,
                station=station,
                n_chargers=s,
                faults_enabled=(fault_rate > 0),
                fault_rate_override=fault_rate if fault_rate > 0 else None,
                sim_days=sim_days,
                seed=42,
            )

            rep_results = simulate_station(config, n_replications=n_reps, verbose=False)
            agg = aggregate_sweep_results(rep_results)
            p15 = agg.get('p_wait_gt_15min_mean', 1.0)
            util = agg.get('mean_utilization_mean', 0)

            print(f"    s={s:3d}: util={util:.3f}, P(W>15)={p15:.4f}")

            if p15 < 0.05 and s_star is None:
                s_star = s

            rows.append({
                'station': station,
                'fault_rate': fault_rate,
                'n_chargers': s,
                'mean_utilization': util,
                'p_wait_gt_15min': p15,
                'mean_wait': agg.get('mean_wait_mean', 0),
            })

        print(f"    s* at 5% target: {s_star}")
        print(f"    Elapsed: {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'fault_sensitivity.csv', index=False)
    print(f"\n  Saved: fault_sensitivity.csv ({len(df)} rows)")

    # Plot: s* vs fault rate
    summary = []
    for fr in FAULT_SENSITIVITY_RATES:
        sub = df[df['fault_rate'] == fr].sort_values('n_chargers')
        s_star = None
        for _, r in sub.iterrows():
            if r['p_wait_gt_15min'] < 0.05:
                s_star = int(r['n_chargers'])
                break
        summary.append({'fault_rate': fr, 's_star': s_star})

    sdf = pd.DataFrame(summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid = sdf.dropna(subset=['s_star'])
    ax.plot(valid['fault_rate'] * 100, valid['s_star'], 'o-',
            color=COLORS['sim_faults_on'], markersize=8, linewidth=2)
    ax.set_xlabel('Fault Rate (%)', fontsize=11)
    ax.set_ylabel('s* (chargers needed for P(W>15) < 5%)', fontsize=11)
    ax.set_title(
        f'Fault Sensitivity Curve: {STATION_LABELS.get(station, station)}\n'
        'Additional chargers needed per percentage point of faults',
        fontsize=12, fontweight='bold',
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'fault_sensitivity_curve.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: fault_sensitivity_curve.png")

    return df


# =====================================================================
# ANALYTICAL GAP QUANTIFICATION
# =====================================================================

def quantify_analytical_gap(
    sweep_df: pd.DataFrame,
    erlang_path: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    For each station, compare simulated P(wait>0) at each s against
    Erlang-C and M/G/s analytical predictions. Compute the gap at
    the utilization where the 5% target is crossed.
    """
    print("\n" + "=" * 70)
    print("ANALYTICAL GAP QUANTIFICATION")
    print("=" * 70)

    rows = []
    on = sweep_df[sweep_df['faults'] == 'on']

    for station in REPRESENTATIVE_STATIONS:
        sub_sim = on[on['station'] == station].sort_values('n_chargers')
        analytical = get_analytical_curves(erlang_path, station, 'historical_avg')

        if len(analytical) == 0 or len(sub_sim) == 0:
            continue

        peak_lam = analytical['peak_lambda_per_hour'].iloc[0]
        mu_per_min = analytical['mu_per_min'].iloc[0]
        mu_per_hour = mu_per_min * 60

        for _, sim_row in sub_sim.iterrows():
            s = int(sim_row['n_chargers'])
            an_row = analytical[analytical['s'] == s]
            if len(an_row) == 0:
                continue
            an_row = an_row.iloc[0]

            rho = peak_lam / (s * mu_per_hour)
            if rho >= 1:
                continue

            sim_p15 = sim_row['p_wait_gt_15min_mean']
            mms_p_wait = an_row['p_wait_mms']  # P(wait>0) from Erlang-C
            mgs_p_wait = an_row['p_wait_mgs']

            rows.append({
                'station': station,
                'n_chargers': s,
                'rho_analytical': rho,
                'util_simulated': sim_row['mean_utilization_mean'],
                'sim_p_wait_15': sim_p15,
                'sim_mean_wait': sim_row.get('mean_wait_mean', None),
                'mms_p_wait_0': mms_p_wait,
                'mgs_p_wait_0': mgs_p_wait,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'analytical_gap.csv', index=False)
    print(f"  Saved: analytical_gap.csv ({len(df)} rows)")

    # Print summary per station
    for station in REPRESENTATIVE_STATIONS:
        sub = df[df['station'] == station]
        if len(sub) == 0:
            continue
        # Find row closest to 5% simulated P(wait>15)
        sub = sub.copy()
        sub['dist_to_5pct'] = abs(sub['sim_p_wait_15'] - 0.05)
        nearest = sub.loc[sub['dist_to_5pct'].idxmin()]
        print(f"\n  {STATION_LABELS.get(station, station)} "
              f"(near 5% target, s={int(nearest['n_chargers'])}):")
        print(f"    Sim util: {nearest['util_simulated']:.3f} "
              f"(analytical ρ: {nearest['rho_analytical']:.3f})")
        print(f"    Sim P(W>15): {nearest['sim_p_wait_15']:.4f}")
        print(f"    Erlang-C P(W>0): {nearest['mms_p_wait_0']:.4f}")
        print(f"    M/G/s P(W>0): {nearest['mgs_p_wait_0']:.4f}")

    return df


def write_metric_safe_references(
    sweep_df: pd.DataFrame,
    erlang_path: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Write metric-safe simulation and analytical reference tables.

    The primary Week 6 frontier is simulated full-day NHPP P(W>15).
    Week 4 analytical curves are peak-hour P(W>0), so they are kept in
    separate outputs and not overlaid on P(W>15) plots.
    """
    print("\n" + "=" * 70)
    print("METRIC-SAFE SIMULATION AND ANALYTICAL REFERENCES")
    print("=" * 70)

    sim_rows = []
    summary_rows = []
    analytical_rows = []
    tail_rows = []
    on = sweep_df[sweep_df['faults'] == 'on']

    for station in REPRESENTATIVE_STATIONS:
        sub_sim = on[on['station'] == station].sort_values('n_chargers')
        analytical = get_analytical_curves(erlang_path, station, 'historical_avg')

        if len(sub_sim) == 0:
            continue

        ss = find_s_star(sub_sim, target=0.05)
        summary_rows.append({
            'station': station,
            'metric': 'full_day_nhpp_sim_p_wait_gt_15min',
            'target_p_wait_15': 0.05,
            's_star_point': ss['s_star_point'],
            's_star_conservative': ss['s_star_conservative'],
            'comparison_status': 'simulation_only_same_metric',
            'note': (
                'Analytical P(W>0) is a peak-hour reference and is not '
                'subtracted from simulated full-day P(W>15).'
            ),
        })

        for _, sim_row in sub_sim.iterrows():
            sim_rows.append({
                'station': station,
                'n_chargers': int(sim_row['n_chargers']),
                'metric_family': 'simulation_full_day_nhpp',
                'metric_name': 'p_wait_gt_15min',
                'sim_utilization': sim_row['mean_utilization_mean'],
                'sim_p_wait_gt_15min': sim_row['p_wait_gt_15min_mean'],
                'sim_p_wait_gt_15min_ci95': sim_row.get(
                    'p_wait_gt_15min_ci95', np.nan),
                'sim_p_wait_gt_15min_upper95': sim_row.get(
                    'p_wait_gt_15min_upper95', np.nan),
                'sim_mean_wait_min': sim_row.get('mean_wait_mean', np.nan),
                'same_metric_basis': (
                    'fleet-frontier rows are comparable to each other'
                ),
            })

        if len(analytical) == 0:
            continue

        peak_lam = analytical['peak_lambda_per_hour'].iloc[0]
        mu_per_min = analytical['mu_per_min'].iloc[0]
        mu_per_hour = mu_per_min * 60

        for _, an_row in analytical[analytical['s'].isin(S_VALUES)].iterrows():
            s = int(an_row['s'])
            rho = peak_lam / (s * mu_per_hour)
            if rho >= 1:
                continue

            p_wait_mms = an_row['p_wait_mms']
            p_wait_mgs = an_row['p_wait_mgs']
            analytical_rows.append({
                'station': station,
                'n_chargers': s,
                'metric_family': 'analytical_peak_hour',
                'metric_name': 'p_wait_gt_0',
                'rho_peak_hour': rho,
                'mms_p_wait_gt_0': p_wait_mms,
                'mgs_p_wait_gt_0': p_wait_mgs,
                'note': 'Reference only; not same metric as full-day P(W>15).',
            })

            tail_rows.append({
                'station': station,
                'n_chargers': s,
                'metric_family': 'analytical_peak_hour_mms',
                'metric_name': 'p_wait_gt_15min',
                'rho_peak_hour': rho,
                'mms_p_wait_gt_15_peak_hour': float(
                    p_wait_mms * np.exp(
                        -(s * mu_per_hour - peak_lam) * (15.0 / 60.0)
                    )
                ),
                'note': (
                    'Exact for M/M/s at the peak-hour arrival rate; still not '
                    'a full-day NHPP simulation estimate.'
                ),
            })

    sim_frontier = pd.DataFrame(sim_rows)
    analytical_ref = pd.DataFrame(analytical_rows)
    peak_tail_ref = pd.DataFrame(tail_rows)
    summary_df = pd.DataFrame(summary_rows)

    sim_frontier.to_csv(output_dir / 'simulation_frontier_summary.csv',
                        index=False)
    analytical_ref.to_csv(output_dir / 'analytical_pwait0_reference.csv',
                          index=False)
    peak_tail_ref.to_csv(output_dir / 'analytical_peak_mms_tail_reference.csv',
                         index=False)
    summary_df.to_csv(output_dir / 'analytical_gap.csv', index=False)

    print(f"  Saved: simulation_frontier_summary.csv ({len(sim_frontier)} rows)")
    print(f"  Saved: analytical_pwait0_reference.csv ({len(analytical_ref)} rows)")
    print(f"  Saved: analytical_peak_mms_tail_reference.csv ({len(peak_tail_ref)} rows)")
    print("  Saved: analytical_gap.csv "
          "(metric-safe summary; no mismatched analytical gap computed)")

    if len(analytical_ref) > 0:
        print("  Generating analytical P(W>0) reference plot...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.ravel()
        for idx, station in enumerate(REPRESENTATIVE_STATIONS):
            ax = axes[idx]
            sub = analytical_ref[analytical_ref['station'] == station]
            if len(sub) == 0:
                continue
            sub = sub.sort_values('rho_peak_hour')
            ax.plot(sub['rho_peak_hour'], sub['mms_p_wait_gt_0'], 's--',
                    color=COLORS['erlang_c'], markersize=4,
                    linewidth=1.2, label='Erlang-C P(W>0)')
            ax.plot(sub['rho_peak_hour'], sub['mgs_p_wait_gt_0'], '^--',
                    color=COLORS['mgs'], markersize=4,
                    linewidth=1.2, label='M/G/s P(W>0)')
            ax.set_title(STATION_LABELS.get(station, station),
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Peak-hour utilization rho', fontsize=10)
            ax.set_ylabel('Analytical P(W > 0)', fontsize=10)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper left')

        fig.suptitle(
            'Analytical Peak-Hour Reference Only: P(W > 0)\n'
            'Not overlaid on simulated full-day P(W > 15 min)',
            fontsize=12, fontweight='bold',
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(output_dir / 'analytical_pwait0_reference.png', dpi=150)
        plt.close(fig)
        print("  Saved: analytical_pwait0_reference.png")

    for _, row in summary_df.iterrows():
        print(f"\n  {STATION_LABELS.get(row['station'], row['station'])}:")
        print(f"    Sim s* point/conservative for P(W>15)<5%: "
              f"{row['s_star_point']} / {row['s_star_conservative']}")

    return summary_df


# =====================================================================
# METADATA
# =====================================================================

def save_metadata(
    output_dir: Path,
    validation: dict,
    sweep_df: pd.DataFrame,
    fault_tax_df: pd.DataFrame,
    n_reps: int,
    sim_days: int,
    provenance: dict,
):
    """Save week6_metadata.json."""
    meta = {
        'week': 6,
        'components': [
            'SimPy M/M/s Validation',
            'Baseline FCFS Pareto Frontier Sweeps',
            'Erlang-C / M/G/s Analytical Overlay',
            'Fault Tax Calculation',
            'Fault Sensitivity Curve',
        ],
        'stations_simulated': REPRESENTATIVE_STATIONS,
        'sweep_parameters': {
            's_values': S_VALUES,
            'n_replications': n_reps,
            'sim_days': sim_days,
            'arrival_mode': 'nhpp',
            'scheduling_policy': 'fcfs',
            'common_random_numbers': True,
            'crn_note': ('SeedSequence creates independent arrival, type, service, '
                         'fault, and repair streams. Customer attributes are '
                         'generated at arrival, so a given replication replays '
                         'the same exogenous demand across fleet/policy variants.'),
            'warmup_days': 7,
        },
        'compute_estimate': {
            'pareto_on': f'{len(REPRESENTATIVE_STATIONS)} stations x '
                         f'{len(S_VALUES)} fleet sizes x {n_reps} reps = '
                         f'{len(REPRESENTATIVE_STATIONS) * len(S_VALUES) * n_reps} runs',
            'pareto_off': f'{len(REPRESENTATIVE_STATIONS)} stations x '
                          f'{len(S_VALUES)} fleet sizes x {n_reps} reps = '
                          f'{len(REPRESENTATIVE_STATIONS) * len(S_VALUES) * n_reps} runs',
            'sensitivity': f'{len(FAULT_SENSITIVITY_RATES)} fault rates x '
                           f'{len(S_VALUES)} fleet sizes x {n_reps} reps = '
                           f'{len(FAULT_SENSITIVITY_RATES) * len(S_VALUES) * n_reps} runs',
            'total_simulation_runs': (
                2 * len(REPRESENTATIVE_STATIONS) * len(S_VALUES) * n_reps +
                len(FAULT_SENSITIVITY_RATES) * len(S_VALUES) * n_reps
            ),
            'per_run': f'{sim_days} simulated days',
        },
        'validation': {
            'all_pass': validation.get('all_pass', False),
            'test_cases': [k for k in validation if k != 'all_pass'],
        },
        'fault_model': {
            'type': 'charger_downtime_with_requeue',
            'description': 'A faulted session blocks the charger for a repair '
                           'duration drawn from lognormal(median, sigma=0.8). '
                            'The same customer retries after repair without '
                            'changing identity or original arrival time. This '
                            'ensures faults are unambiguously a '
                           'capacity penalty: same demand must be served with '
                           'reduced effective capacity during repair periods.',
            'fault_repair_median_min': 30.0,
            'fault_repair_sigma': 0.8,
            'fault_repair_floor_min': 5.0,
            'customer_requeue': True,
            'assumption_note': 'Repair duration is a modeling assumption. The dataset '
                               'does not record charger recovery time. Sensitivity '
                               'analysis should vary the median (15, 30, 60 min).',
        },
        'metric_definitions': {
            'primary_simulation_frontier': (
                'Full-day NHPP SimPy estimate of P(W>15 min), wait metrics, '
                'and utilization across the fleet-size sweep.'
            ),
            'analytical_pwait0_reference': (
                'Week 4 Erlang-C/M/G/s peak-hour P(W>0). Reported as a '
                'reference only and not overlaid on P(W>15) simulation plots.'
            ),
            'analytical_peak_mms_tail_reference': (
                'Exact M/M/s peak-hour P(W>15) tail using Erlang-C and the '
                'exponential wait-tail identity; not a full-day NHPP result.'
            ),
        },
        'provenance': provenance,
        'fault_tax_summary': fault_tax_df.to_dict('records') if len(fault_tax_df) > 0 else [],
        'fault_sensitivity_station': FAULT_SENSITIVITY_STATION,
        'fault_sensitivity_rates': FAULT_SENSITIVITY_RATES,
        'files_produced': [
            'validation_results.json',
            'pareto_sweep_results.csv',
            'pareto_sweep_replications.csv',
            'fault_off_sweep_results.csv',
            'fault_off_replications.csv',
            'fault_tax_results.csv',
            'fault_sensitivity.csv',
            'analytical_gap.csv',
            'simulation_frontier_summary.csv',
            'analytical_pwait0_reference.csv',
            'analytical_peak_mms_tail_reference.csv',
            'pareto_sweep_all.csv',
            'pareto_baseline_combined.png',
            'analytical_pwait0_reference.png',
            'fault_tax_comparison.png',
            'fault_sensitivity_curve.png',
            'week6_metadata.json',
        ],
    }
    with open(output_dir / 'week6_metadata.json', 'w') as f:
        json.dump(to_builtin(meta), f, indent=2)
    print(f"\n  Saved: week6_metadata.json")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Week 6: SimPy Validation + Pareto Frontiers + Fault Tax')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'week6_results'),
                        help='Directory for outputs (default: ./week6_results)')
    parser.add_argument('--week3-dir', type=str,
                        default=str(RESULTS_DIR / 'week3_results'),
                        help='Directory with nhpp_rate_functions.csv')
    parser.add_argument('--week4-dir', type=str,
                        default=str(RESULTS_DIR / 'week4_results'),
                        help='Directory with Week 4 analytical outputs')
    parser.add_argument('--week5-dir', type=str,
                        default=str(RESULTS_DIR / 'week5_results'),
                        help='Directory with sim_engine.py from Week 5')
    parser.add_argument('--reps', type=int, default=50,
                        help='Replications per configuration (default: 50)')
    parser.add_argument('--sim-days', type=int, default=30,
                        help='Simulated days per replication (default: 30)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip M/M/s validation (if already passed)')
    parser.add_argument('--skip-sweeps', action='store_true',
                        help='Skip Pareto sweeps (load from CSV)')
    parser.add_argument('--skip-fault-tax', action='store_true',
                        help='Skip fault tax computation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    week3_dir = resolve_repo_path(args.week3_dir)
    week4_dir = resolve_repo_path(args.week4_dir)
    week5_dir = resolve_repo_path(args.week5_dir)
    sim_engine_info = load_sim_engine_from_week5(week5_dir)

    # Resolve input file paths (project root + week-specific result dirs)
    search_roots = [
        Path('.'),
        SCRIPT_DIR,
        week3_dir,
        week4_dir,
        week5_dir,
    ]

    def find_file(name, extra_dirs=None):
        dirs = []
        for d in (extra_dirs or []) + search_roots:
            p = Path(d)
            if p not in dirs:
                dirs.append(p)
        for d in dirs:
            p = d / name
            if p.exists():
                return str(p)
        searched = ', '.join(str(d) for d in dirs)
        raise FileNotFoundError(f"Cannot find {name} in: {searched}")

    param_path = find_file('parameter_summary.json', [week4_dir])
    nhpp_path = find_file('nhpp_rate_functions.csv', [week3_dir])
    erlang_path = find_file('erlang_c_results_4rep.csv', [week4_dir])
    mgs_path = find_file('mgs_comparison_4rep.csv', [week4_dir])

    input_paths = {
        'parameter_summary': param_path,
        'nhpp_rate_functions': nhpp_path,
        'erlang_c_results_4rep': erlang_path,
        'mgs_comparison_4rep': mgs_path,
    }
    provenance = provenance_signature(
        input_paths, args.reps, args.sim_days, sim_engine_info)

    print("=" * 70)
    print("WEEK 6: SimPy Validation + Pareto Frontiers + Fault Tax")
    print("=" * 70)
    print(f"  Output dir:   {output_dir.resolve()}")
    print(f"  Replications: {args.reps}")
    print(f"  Sim days:     {args.sim_days}")
    print(f"  Stations:     {len(REPRESENTATIVE_STATIONS)}")
    print(f"  Fleet sizes:  {S_VALUES}")
    print(f"  Week 3 dir:   {week3_dir}")
    print(f"  Week 4 dir:   {week4_dir}")
    print(f"  Week 5 dir:   {week5_dir}")
    print(f"  sim_engine:   {sim_engine_info['path']}")
    print(f"  engine hash:  {sim_engine_info['sha256'][:16]}...")

    if not HAS_SIMPY:
        print("\n[ERROR] SimPy is not installed in this Python environment.")
        print("Use the Week 5 interpreter or install simpy, for example:")
        print("  .venv\\Scripts\\python.exe week6_analysis.py --reps 10")
        sys.exit(1)

    # ── Step 1: Validation ──
    if not args.skip_validation:
        validation = run_validation(output_dir, n_reps=200)
        if not validation.get('all_pass', False):
            print("\n  ⚠ Validation failed. Fix sim_engine.py before proceeding.")
            print("    Use --skip-validation to bypass (at your own risk).")
            sys.exit(1)
    else:
        print("\n  [SKIP] Validation (--skip-validation)")
        validation = {'all_pass': None, 'status': 'skipped'}

    # ── Step 2: Pareto sweeps (faults ON) ──
    if not args.skip_sweeps:
        sweep_on = run_pareto_sweeps(
            param_path, nhpp_path, mgs_path, erlang_path,
            output_dir, n_reps=args.reps, sim_days=args.sim_days,
        )
    else:
        print("\n  [SKIP] Pareto sweeps (loading from CSV)")
        warn_if_cached_outputs_stale(output_dir, provenance)
        sweep_on = pd.read_csv(output_dir / 'pareto_sweep_results.csv')

    # Plot simulation-only Pareto frontiers.
    plot_pareto_frontiers(sweep_on, erlang_path, output_dir)

    # Write metric-safe analytical references.
    gap_df = write_metric_safe_references(sweep_on, erlang_path, output_dir)

    # ── Step 3: Fault tax ──
    fault_tax_df = pd.DataFrame()
    if not args.skip_fault_tax:
        sweep_off = run_fault_off_sweeps(
            param_path, nhpp_path, mgs_path,
            output_dir, n_reps=args.reps, sim_days=args.sim_days,
        )

        # Merge ON + OFF for combined analysis
        sweep_all = pd.concat([sweep_on, sweep_off], ignore_index=True)
        sweep_all.to_csv(output_dir / 'pareto_sweep_all.csv', index=False)

        fault_tax_df = compute_fault_tax(sweep_on, sweep_off, output_dir)
        plot_fault_tax(sweep_on, sweep_off, output_dir)

        # Fault sensitivity
        run_fault_sensitivity(
            param_path, nhpp_path, mgs_path,
            output_dir, n_reps=args.reps, sim_days=args.sim_days,
        )
    else:
        print("\n  [SKIP] Fault tax (--skip-fault-tax)")
        existing_tax = output_dir / 'fault_tax_results.csv'
        if existing_tax.exists():
            fault_tax_df = pd.read_csv(existing_tax)
            print(f"  Loaded existing fault_tax_results.csv "
                  f"({len(fault_tax_df)} rows) for metadata.")

    # ── Metadata ──
    save_metadata(output_dir, validation, sweep_on, fault_tax_df,
                  args.reps, args.sim_days, provenance)

    print("\n" + "=" * 70)
    print("WEEK 6 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
