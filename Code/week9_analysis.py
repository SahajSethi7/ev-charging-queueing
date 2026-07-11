"""
Week 9: Phase 3b — Demand-Responsive Activation + Greedy Scheduler
====================================================================
Components:
  9A. Validation of patched sim_engine.py (activation + realized capacity)
  9B. Heuristic activation schedule computation (mix-weighted μ_eff,
      Erlang-C / Allen-Cunneen initializer, two demand scenarios)
  9C. Greedy historical replay on Week 8 station-days (3-way comparison)
  9D. Activation frontier sweeps (Full + Heuristic-Historical +
      Heuristic-RecentQ, faults ON; Heuristic-Historical faults OFF)
  9D+. Simulation-tuned activation schedules (uniform-margin tuning)
  9E. Figures: capstone (active charger-hours vs P(Wq>15min)),
      secondary (utilization vs wait), master Pareto overlay
  9F. Sensitivity analysis (fault rate, wait threshold τ, repair median)
  9G. Metadata assembly

Locked design decisions:
  - The hourly Erlang-C / Allen-Cunneen computation is an analytically
    motivated initializer, not a derived optimum. The simulator is
    the binding evaluation.
  - Utilization denominator: realized active-server-minutes (event-driven
    integral), not scheduled. Both are reported for transparency.
  - Greedy runs as historical replay on Week 8 sessions only. No
    synthetic tier assignment. No fault buffer (simulator already
    models fault-induced capacity loss).
  - "Recent quarter" = most recent non-degenerate quarter per station.
    Tech Park uses 2021Q3; others use 2021Q4.
  - The primary result figure uses active charger-hours vs P(Wq>15min).
    The utilization-vs-wait figure is secondary and explicitly labeled
    as using time-varying realized active capacity in the denominator.

Input files:
  - jiaxing_clean.parquet          (441k sessions, local)
  - parameter_summary.json         (service time fits, fleet sizing)
  - nhpp_rate_functions.csv        (historical-average NHPP rates)
  - nhpp_rate_functions_quarterly.csv (quarterly NHPP rates)
  - mgs_comparison_4rep.csv        (station mixes, CVs)
  - fault_tax_results.csv          (s* values from Week 6)
  - pareto_sweep_results.csv       (Week 6 baseline frontiers)
  - flexibility_summary.json       (from Week 7)
  - representative_days.csv        (from Week 8)
  - lp_results.csv                 (from Week 8)
  - fcfs_replay_results.csv        (from Week 8)
  - sim_engine.py                  (patched, with activation support)

Output files:
  - activation_schedules.csv
  - greedy_results.csv
  - three_way_comparison.csv
  - pareto_activation_results.csv
  - pareto_activation_replications.csv
  - sim_tuned_schedules.csv
  - sensitivity_results.csv
  - week9_metadata.json
  - figures/activation_bars_*.png
  - figures/capstone_frontier_*.png
  - figures/secondary_frontier_*.png
  - figures/master_overlay_*.png

Usage:
  python week9_analysis.py --data-dir ./data --output-dir ./week9_results
                           [--week6-dir ./week6_results]
                           [--week8-dir ./week8_results]
                           [--skip-sweeps] [--skip-sensitivity] [--verbose]

  To run 9D+ only from existing artifacts:
    python week9_analysis.py --skip-validation --skip-sweeps --skip-sensitivity

Date: Week 9, Apr 2026
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_paths import (CODE_DIR, DATA_DIR, PROJECT_ROOT, RESULTS_DIR,
                           to_builtin)

warnings.filterwarnings("ignore", category=FutureWarning)

SIM_ENGINE_DIR = RESULTS_DIR / 'week5_results'
if str(SIM_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_ENGINE_DIR))

try:
    import simpy
    HAS_SIMPY = True
except ImportError:
    HAS_SIMPY = False
    print("[WARN] SimPy not installed. Simulation components will be skipped.")

# Import patched sim_engine
try:
    from sim_engine import (
        StationConfig, simulate_station, load_config,
        erlang_c_pwait, sample_service_time, choose_charger_type,
        DEFAULT_FAULT_RATES, MetricsCollector, SessionRecord
    )
    HAS_ENGINE = True
except ImportError as exc:
    HAS_ENGINE = False
    print(f"[WARN] sim_engine not importable from {SIM_ENGINE_DIR}: {exc}")

# =====================================================================
# CONSTANTS (display labels only — all numeric values loaded from files)
# =====================================================================

REPRESENTATIVE_STATIONS = [
    'Xiuzhou_Expressway Service District A',
    'Nanhu_Technology Park',
    'Xiuzhou_Government Agency',
    'Tongxiang_Bus Station',
]

STATION_LABELS = {
    'Xiuzhou_Expressway Service District A': 'Expressway A (DC Fast)',
    'Nanhu_Technology Park': 'Technology Park (Mixed)',
    'Xiuzhou_Government Agency': 'Gov Agency (L2)',
    'Tongxiang_Bus Station': 'Bus Station (L2, High-Vol)',
}

STATION_SHORT = {
    'Xiuzhou_Expressway Service District A': 'XZ_ExpA',
    'Nanhu_Technology Park': 'NH_Tech',
    'Xiuzhou_Government Agency': 'XZ_Gov',
    'Tongxiang_Bus Station': 'TX_Bus',
}

# Fleet size sweep grid (same as Week 6)
FLEET_SIZES = list(range(2, 32, 2))


def resolve_input_path(label: str, *candidates: Path) -> str:
    """Return the first existing candidate path, else the first candidate.

    The repo stores upstream artifacts across multiple week result
    folders, so Week 9 should resolve them explicitly instead of
    assuming they all live under ./data.
    """
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    first = candidates[0]
    print(f"  [WARN] Could not resolve {label}; expected one of:")
    for candidate in candidates:
        print(f"    - {candidate}")
    return str(first)


def load_s_star_from_file(fault_tax_path: str) -> Dict[str, int]:
    """Load s* faults-ON from fault_tax_results.csv (Week 6 output).

    Falls back to hard-coded values if file is missing, with a warning.
    """
    _FALLBACK = {
        'Xiuzhou_Expressway Service District A': 4,
        'Nanhu_Technology Park': 6,
        'Xiuzhou_Government Agency': 8,
        'Tongxiang_Bus Station': 10,
    }
    if not Path(fault_tax_path).exists():
        print(f"  [WARN] {fault_tax_path} not found, using hard-coded s*")
        return _FALLBACK

    df = pd.read_csv(fault_tax_path)
    result = {}
    for station in REPRESENTATIVE_STATIONS:
        row = df[df['station'] == station]
        if len(row) > 0:
            result[station] = int(row.iloc[0]['s_star_faults_on'])
        elif station in _FALLBACK:
            result[station] = _FALLBACK[station]
    return result


def load_service_means_from_file(param_path: str) -> Dict[str, float]:
    """Load service time means from parameter_summary.json.

    Falls back to hard-coded values if file is missing, with a warning.
    """
    _FALLBACK = {'DC_Fast': 34.916, 'Level_2': 96.481, 'Mixed': 44.096}
    if not Path(param_path).exists():
        print(f"  [WARN] {param_path} not found, using hard-coded service means")
        return _FALLBACK

    with open(param_path, 'r') as f:
        params = json.load(f)
    result = {}
    for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
        st = params.get('service_time', {}).get(ctype, {})
        result[ctype] = st.get('mean_min', _FALLBACK.get(ctype, 50.0))
    return result


def detect_recent_quarters(quarterly_path: str) -> Dict[str, str]:
    """Detect most recent non-degenerate quarter per station.

    A quarter is degenerate if peak lambda is 0 (no arrivals).
    Falls back to hard-coded values if file is missing.
    """
    _FALLBACK = {
        'Xiuzhou_Expressway Service District A': '2021Q4',
        'Nanhu_Technology Park': '2021Q3',
        'Xiuzhou_Government Agency': '2021Q4',
        'Tongxiang_Bus Station': '2021Q4',
    }
    if not Path(quarterly_path).exists():
        print(f"  [WARN] {quarterly_path} not found, using hard-coded quarters")
        return _FALLBACK

    df = pd.read_csv(quarterly_path)
    result = {}
    for station in REPRESENTATIVE_STATIONS:
        sdata = df[df['station'] == station]
        quarters = sorted(sdata['quarter'].unique(), reverse=True)
        found = False
        for q in quarters:
            qdata = sdata[sdata['quarter'] == q]
            if qdata['lambda_mean'].max() > 0:
                result[station] = q
                found = True
                break
        if not found:
            result[station] = _FALLBACK.get(station, quarters[0] if quarters else '2021Q4')
    return result

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


# =====================================================================
# EMPIRICAL-MIX CONFIG WRAPPER
# =====================================================================

def load_config_empirical(param_path: str, nhpp_path: str,
                          mgs_path: str, station: str,
                          **kwargs) -> 'StationConfig':
    """Wrap load_config() and patch in the empirical charger-type mix.

    load_config() uses a coarse dominant-type heuristic for the charger
    mix. This wrapper replaces it with the empirical per-station mix
    from mgs_comparison_4rep.csv, matching the approach used in Week 6's
    load_station_config_accurate().

    This eliminates the systematic service-time discrepancy between
    Week 6 (Pareto baseline) and Week 9 (activation sweeps).

    Raises ValueError if the empirical mix cannot be found for the
    given station, rather than silently falling back to the heuristic.
    """
    config = load_config(param_path, nhpp_path, station, **kwargs)
    mixes = load_station_mixes(mgs_path)
    if station not in mixes:
        raise ValueError(
            f"load_config_empirical: no empirical charger-type mix found "
            f"for '{station}' in {mgs_path}. Cannot fall back to heuristic "
            f"mix — this would re-introduce the Week 6/9 service-time "
            f"discrepancy. Check that mgs_comparison_4rep.csv contains a "
            f"'historical_avg' row for this station."
        )
    config.charger_type_mix = mixes[station]
    return config


# =====================================================================
# 9B: HEURISTIC ACTIVATION SCHEDULE COMPUTATION
# =====================================================================

def load_station_mixes(mgs_path: str) -> Dict[str, Dict[str, float]]:
    """Load empirical charger-type mixes from mgs_comparison_4rep.csv."""
    df = pd.read_csv(mgs_path)
    mixes = {}
    for station in REPRESENTATIVE_STATIONS:
        row = df[(df['station'] == station) &
                 (df['sizing_scenario'] == 'historical_avg')]
        if len(row) == 0:
            continue
        mix_json = json.loads(row.iloc[0]['station_mix_json'])
        mixes[station] = mix_json
    return mixes


def compute_effective_mu(station_mix: Dict[str, float]) -> Tuple[float, float]:
    """Compute mix-weighted effective service rate and mean service time.

    Returns (mu_eff_per_hour, mean_service_min).
    mu_eff = 1 / E[S] where E[S] = sum(mix_fraction * mean_min per type).
    """
    e_service = sum(
        station_mix.get(t, 0) * SERVICE_MEANS.get(t, 50)
        for t in ['DC_Fast', 'Level_2', 'Mixed']
    )
    mu_eff_per_min = 1.0 / e_service if e_service > 0 else 0.01
    mu_eff_per_hour = mu_eff_per_min * 60
    return mu_eff_per_hour, e_service


def erlang_c_tail_probability(s: int, lam_per_hour: float,
                              mu_per_hour: float, cv: float,
                              tau_min: float = 15.0) -> float:
    """Compute P(Wq > tau) for M/G/s using Erlang-C with Allen-Cunneen correction.

    This is a heuristic approximation, not exact for the actual NHPP +
    mixed-service + faults system. The simulator is the binding evaluation.

    P(Wq > tau) ≈ P(wait > 0) * exp(-s*mu*(1-rho)*tau) * (1+CV²)/2

    where the (1+CV²)/2 correction is the Allen-Cunneen/Kingman
    approximation for non-exponential service times.
    """
    if lam_per_hour <= 0:
        return 0.0

    rho = lam_per_hour / (s * mu_per_hour)
    if rho >= 1.0:
        return 1.0

    # Erlang-C: P(wait > 0)
    p_wait_0 = erlang_c_pwait(s, lam_per_hour, mu_per_hour)

    # Tail probability: P(Wq > tau | waited) for M/M/s
    mu_per_min = mu_per_hour / 60
    lam_per_min = lam_per_hour / 60
    clearance_rate = s * mu_per_min - lam_per_min

    # Allen-Cunneen M/G/s correction: multiply by (1+CV²)/2
    ac_factor = (1 + cv**2) / 2
    mean_wait_mms = p_wait_0 / clearance_rate
    mean_wait_mgs = mean_wait_mms * ac_factor
    conditional_mean_mgs = mean_wait_mgs / p_wait_0
    exponent = tau_min / conditional_mean_mgs
    p_wq_gt_tau = p_wait_0 * (np.exp(-exponent)
                              if exponent < 500 else 0.0)
    return min(p_wq_gt_tau, 1.0)


def compute_heuristic_schedule(nhpp_rates_24: np.ndarray,
                                mu_eff_per_hour: float,
                                cv: float,
                                n_chargers_max: int,
                                target_p: float = 0.05,
                                tau_min: float = 15.0) -> np.ndarray:
    """Compute heuristic activation schedule for one station.

    For each hour, find min s >= 1 such that:
        P(Wq > tau_min | s, lambda_h, mu_eff) < target_p

    This is an analytically-motivated initializer. It treats each hour
    as an isolated M/G/s queue (no carryover, no session spillover).
    This makes it systematically optimistic about how few chargers
    each hour needs. The simulator is the binding evaluation.

    s_heuristic(h) is NOT capped at the historical s*. If recent demand
    requires more, that is a finding about regime shift, not an error.
    However, it IS capped at n_chargers_max (installed capacity).

    Returns: (24,) array of per-hour active charger counts.
    """
    schedule = np.ones(24, dtype=int)
    for h in range(24):
        lam_h = nhpp_rates_24[h]
        if lam_h <= 0:
            schedule[h] = 1
            continue

        for s in range(1, n_chargers_max + 1):
            p = erlang_c_tail_probability(s, lam_h, mu_eff_per_hour, cv, tau_min)
            if p < target_p:
                schedule[h] = s
                break
        else:
            # Even max chargers doesn't meet target — use max
            schedule[h] = n_chargers_max

    return schedule


def build_activation_schedules(nhpp_path: str, quarterly_path: str,
                               mgs_path: str, output_dir: Path) -> pd.DataFrame:
    """Build heuristic activation schedules for all representative stations.

    Two demand scenarios:
      - Historical average: full-dataset NHPP rates
      - Recent quarter: most recent non-degenerate quarter per station

    Returns DataFrame with columns:
      station, demand_scenario, hour, lambda_h, mu_eff_per_hr, cv,
      s_heuristic, n_chargers_installed, s_star_week6
    """
    print("\n" + "=" * 70)
    print("COMPONENT 9B: HEURISTIC ACTIVATION SCHEDULES")
    print("=" * 70)

    nhpp_df = pd.read_csv(nhpp_path)
    quarterly_df = pd.read_csv(quarterly_path)
    mixes = load_station_mixes(mgs_path)

    # Load CVs from mgs file
    mgs_df = pd.read_csv(mgs_path)
    station_cvs = {}
    for station in REPRESENTATIVE_STATIONS:
        row = mgs_df[(mgs_df['station'] == station) &
                     (mgs_df['sizing_scenario'] == 'historical_avg')]
        if len(row) > 0:
            station_cvs[station] = row.iloc[0]['cv']

    rows = []

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        s_star = S_STAR_FAULTS_ON[station]
        mix = mixes.get(station, {'DC_Fast': 0.33, 'Level_2': 0.34, 'Mixed': 0.33})
        mu_eff, mean_svc = compute_effective_mu(mix)
        cv = station_cvs.get(station, 1.4)

        print(f"\n  {label}:")
        print(f"    Mix: {mix}")
        print(f"    E[S] = {mean_svc:.1f} min, mu_eff = {mu_eff:.3f}/hr, CV = {cv:.3f}")
        print(f"    s* (Week 6) = {s_star}")

        # Use a generous installed capacity for the heuristic (allow findings
        # where recent demand exceeds historical s*)
        n_installed = max(s_star + 6, 30)

        # Scenario A: Historical average
        hist_rates = nhpp_df[nhpp_df['station'] == station].sort_values('hour')
        if len(hist_rates) == 24:
            lam_hist = hist_rates['lambda_mean'].values
            sched_hist = compute_heuristic_schedule(
                lam_hist, mu_eff, cv, n_installed, target_p=0.05, tau_min=15.0)

            print(f"    Historical: peak_lam={lam_hist.max():.2f}, "
                  f"peak_s_heuristic={sched_hist.max()}, "
                  f"total_scheduled_ch={sched_hist.sum()}/day vs "
                  f"{s_star*24}/day (full)")

            for h in range(24):
                rows.append({
                    'station': station,
                    'demand_scenario': 'historical_avg',
                    'hour': h,
                    'lambda_h': round(lam_hist[h], 4),
                    'mu_eff_per_hr': round(mu_eff, 4),
                    'cv': round(cv, 4),
                    's_heuristic': int(sched_hist[h]),
                    'n_chargers_installed': n_installed,
                    's_star_week6': s_star,
                })

        # Scenario B: Recent quarter
        recent_q = RECENT_QUARTER[station]
        recent_rates = quarterly_df[
            (quarterly_df['station'] == station) &
            (quarterly_df['quarter'] == recent_q)
        ].sort_values('hour')

        if len(recent_rates) == 24:
            lam_recent = recent_rates['lambda_mean'].values
            sched_recent = compute_heuristic_schedule(
                lam_recent, mu_eff, cv, n_installed, target_p=0.05, tau_min=15.0)

            print(f"    Recent ({recent_q}): peak_lam={lam_recent.max():.2f}, "
                  f"peak_s_heuristic={sched_recent.max()}, "
                  f"total_scheduled_ch={sched_recent.sum()}/day vs "
                  f"{s_star*24}/day (full)")

            if sched_recent.max() > s_star:
                print(f"    *** Peak s_heuristic ({sched_recent.max()}) > "
                      f"historical s* ({s_star}): demand growth finding ***")

            for h in range(24):
                rows.append({
                    'station': station,
                    'demand_scenario': f'recent_quarter_{recent_q}',
                    'hour': h,
                    'lambda_h': round(lam_recent[h], 4),
                    'mu_eff_per_hr': round(mu_eff, 4),
                    'cv': round(cv, 4),
                    's_heuristic': int(sched_recent[h]),
                    'n_chargers_installed': n_installed,
                    's_star_week6': s_star,
                })
        else:
            print(f"    Recent ({recent_q}): INSUFFICIENT DATA "
                  f"({len(recent_rates)} hours found)")

    sched_df = pd.DataFrame(rows)
    sched_df.to_csv(output_dir / 'activation_schedules.csv', index=False)
    print(f"\n  Saved: activation_schedules.csv ({len(sched_df)} rows)")

    # Plot activation bar charts
    plot_activation_bars(sched_df, output_dir)

    return sched_df


def plot_activation_bars(sched_df: pd.DataFrame, output_dir: Path):
    """Plot heuristic activation schedules per station."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        short = STATION_SHORT.get(station, station[:10])
        s_star = S_STAR_FAULTS_ON[station]

        station_data = sched_df[sched_df['station'] == station]
        scenarios = station_data['demand_scenario'].unique()

        fig, axes = plt.subplots(1, len(scenarios), figsize=(6*len(scenarios), 4),
                                 squeeze=False, sharey=True)

        for idx, scenario in enumerate(scenarios):
            ax = axes[0, idx]
            sdata = station_data[station_data['demand_scenario'] == scenario]
            hours = sdata['hour'].values
            s_heur = sdata['s_heuristic'].values
            lam_h = sdata['lambda_h'].values

            ax.bar(hours, s_heur, color='steelblue', alpha=0.7, label='s_heuristic')
            ax.axhline(s_star, color='red', linestyle='--', alpha=0.6,
                       label=f's* = {s_star}')

            ax2 = ax.twinx()
            ax2.plot(hours, lam_h, color='orange', linewidth=1.5,
                     marker='.', markersize=4, label='λ(h)')
            ax2.set_ylabel('λ (arrivals/hr)', color='orange')

            ax.set_xlabel('Hour of day')
            ax.set_ylabel('Active chargers (heuristic)')
            ax.set_title(f'{scenario}')
            ax.set_xticks(range(0, 24, 3))
            ax.legend(loc='upper left', fontsize=7)
            ax2.legend(loc='upper right', fontsize=7)

        fig.suptitle(f'{label}: Heuristic Activation Schedule', fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / f'activation_bars_{short}.png')
        plt.close(fig)
        print(f"  Saved: activation_bars_{short}.png")


# =====================================================================
# 9A: VALIDATION
# =====================================================================

def run_validation_tests(param_path: str, nhpp_path: str,
                         mgs_path: str,
                         output_dir: Path,
                         sim_days: int = 30) -> dict:
    """Run validation tests A/B/C for the patched sim_engine.

    Uses the same sim_days as the main sweep so that validation covers
    the actual run regime, not a hard-coded default.

    Test A (regression): Full activation = no activation. Results must
           match Week 6 baseline within tolerance.
    Test B (directional): Reduce capacity at low-demand hour. Confirm
           utilization increases, P(Wq>15min) barely changes.
    Test C (edge case): Verify realized_capacity >= scheduled under
           full activation (no distortion from logging).
    """
    print("\n" + "=" * 70)
    print("COMPONENT 9A: SIM ENGINE VALIDATION")
    print("=" * 70)
    print(f"  (sim_days={sim_days}, matching sweep settings)")

    if not HAS_ENGINE or not HAS_SIMPY:
        print("  [SKIP] SimPy or sim_engine not available.")
        return {'status': 'skipped'}

    results = {}

    def _clean_validation_reps(reps, label: str):
        """Apply the same drain-cap filtering discipline as 9D and 9F."""
        clean = [r for r in reps if not r.get('drain_cap_hit', False)]
        n_hits = len(reps) - len(clean)
        if n_hits > 0:
            print(f"    [WARN] {label}: {n_hits}/{len(reps)} reps hit drain cap "
                  f"and were excluded")
        if len(clean) == 0:
            print(f"    [WARN] {label}: ALL reps hit drain cap")
        return clean, n_hits

    # Test A: Full activation = no activation
    print("\n  Test A: Full activation vs no activation (regression)")
    station = 'Tongxiang_Bus Station'
    s = S_STAR_FAULTS_ON.get(station, 10)
    n_reps = 20  # Enough for regression, not 50

    config_none = load_config_empirical(param_path, nhpp_path, mgs_path, station,
                              n_chargers=s, faults_enabled=True,
                              sim_days=sim_days, random_seed=42)
    config_none.activation_schedule = None

    config_full = load_config_empirical(param_path, nhpp_path, mgs_path, station,
                              n_chargers=s, faults_enabled=True,
                              sim_days=sim_days, random_seed=42)
    config_full.activation_schedule = np.full(24, s)

    res_none_all = simulate_station(config_none, n_replications=n_reps, verbose=False)
    res_full_all = simulate_station(config_full, n_replications=n_reps, verbose=False)
    res_none, n_hits_none = _clean_validation_reps(
        res_none_all, 'validation test A / no activation')
    res_full, n_hits_full = _clean_validation_reps(
        res_full_all, 'validation test A / full activation')

    if len(res_none) == 0 or len(res_full) == 0:
        util_none = util_full = pw15_none = pw15_full = 0.0
        rc_ratio = 0.0
    else:
        util_none = np.mean([r['mean_utilization'] for r in res_none])
        util_full = np.mean([r['mean_utilization'] for r in res_full])
        pw15_none = np.mean([r['p_wait_gt_15min'] for r in res_none])
        pw15_full = np.mean([r['p_wait_gt_15min'] for r in res_full])

        # Under full activation, realized capacity should equal static capacity
        rcm_full = np.mean([r['realized_capacity_minutes'] for r in res_full])
        scm_full = np.mean([r['static_capacity_minutes'] for r in res_full])
        rc_ratio = rcm_full / scm_full if scm_full > 0 else 0

    util_match = abs(util_none - util_full) < 0.01
    pw15_match = abs(pw15_none - pw15_full) < 0.005

    print(f"    No-activation:   util={util_none:.4f}, P(W>15)={pw15_none:.4f}")
    print(f"    Full-activation: util={util_full:.4f}, P(W>15)={pw15_full:.4f}")
    print(f"    Realized/Static capacity ratio: {rc_ratio:.6f}")
    print(f"    util match: {'PASS' if util_match else 'FAIL'} "
          f"(delta={abs(util_none - util_full):.4f})")
    print(f"    pw15 match: {'PASS' if pw15_match else 'FAIL'} "
          f"(delta={abs(pw15_none - pw15_full):.4f})")

    results['test_A'] = {
        'util_none': util_none, 'util_full': util_full,
        'pw15_none': pw15_none, 'pw15_full': pw15_full,
        'realized_static_ratio': rc_ratio,
        'pass_util': util_match, 'pass_pw15': pw15_match,
        'n_reps_clean_none': len(res_none), 'n_reps_drain_hit_none': n_hits_none,
        'n_reps_clean_full': len(res_full), 'n_reps_drain_hit_full': n_hits_full,
    }

    # Test B: Directional — reduce capacity at 3 AM
    print("\n  Test B: Reduce capacity at 3 AM (directional)")
    sched_b = np.full(24, s)
    sched_b[3] = 1  # Only 1 charger at 3 AM

    config_b = load_config_empirical(param_path, nhpp_path, mgs_path, station,
                           n_chargers=s, faults_enabled=True,
                           sim_days=sim_days, random_seed=42)
    config_b.activation_schedule = sched_b

    res_b_all = simulate_station(config_b, n_replications=n_reps, verbose=False)
    res_b, n_hits_b = _clean_validation_reps(
        res_b_all, 'validation test B / 3AM-drop')
    if len(res_b) == 0 or len(res_full) == 0:
        util_b = pw15_b = ach_b = ach_full = 0.0
    else:
        util_b = np.mean([r['mean_utilization'] for r in res_b])
        pw15_b = np.mean([r['p_wait_gt_15min'] for r in res_b])
        ach_b = np.mean([r['active_charger_hours'] for r in res_b])
        ach_full = np.mean([r['active_charger_hours'] for r in res_full])

    print(f"    Full:     util={util_full:.4f}, P(W>15)={pw15_full:.4f}, "
          f"ACH={ach_full:.1f}/sim")
    print(f"    3AM-drop: util={util_b:.4f}, P(W>15)={pw15_b:.4f}, "
          f"ACH={ach_b:.1f}/sim")
    print(f"    ACH delta: {ach_b - ach_full:.1f} (expect negative)")
    print(f"    Util direction: {'PASS (higher)' if util_b > util_full else 'CHECK'}")

    results['test_B'] = {
        'util_full': util_full, 'util_reduced': util_b,
        'pw15_full': pw15_full, 'pw15_reduced': pw15_b,
        'ach_full': ach_full, 'ach_reduced': ach_b,
        'pass_direction': util_b > util_full,
        'n_reps_clean': len(res_b), 'n_reps_drain_hit': n_hits_b,
    }

    # Test C: Realized capacity >= static under full activation
    print("\n  Test C: Realized capacity check")
    # Under full activation with no capacity changes, realized should ≈ static
    rc_pass = abs(rc_ratio - 1.0) < 0.001
    print(f"    Realized/Static = {rc_ratio:.6f}, "
          f"{'PASS' if rc_pass else 'FAIL (distortion detected)'}")

    results['test_C'] = {'ratio': rc_ratio, 'pass': rc_pass}

    overall = all([
        results['test_A']['pass_util'], results['test_A']['pass_pw15'],
        results['test_B']['pass_direction'], results['test_C']['pass']
    ])
    results['overall'] = 'PASS' if overall else 'FAIL'
    print(f"\n  Overall validation: {results['overall']}")

    return results


# =====================================================================
# 9C: GREEDY HISTORICAL REPLAY
# =====================================================================

def run_greedy_replay(week8_dir: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run greedy scheduler on Week 8 representative station-days.

    This is a historical replay comparator. Same sessions, same observed
    service durations, same fleet sizes as Week 8 FCFS and LP.

    Greedy tie-break rule:
      1. Inflexible sessions first (priority 0).
      2. Among flexible: earliest deadline first.
      3. If deadlines equal: earlier arrival first (FCFS).
      4. If arrivals equal: lower session_id first.

    No fault buffer. No ML demand anticipation. No stochastic tier
    assignment. The greedy scheduler only reorders the queue when
    multiple sessions are simultaneously waiting.

    If a flexible session waits past its deadline, it is promoted to
    priority 0 (treated as inflexible). The count of deadline promotions
    is logged.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 9C: GREEDY HISTORICAL REPLAY")
    print("=" * 70)

    rep_days_path = week8_dir / 'representative_days.csv'
    if not rep_days_path.exists():
        print(f"  [SKIP] {rep_days_path} not found.")
        return pd.DataFrame(), pd.DataFrame()

    rep_days = pd.read_csv(rep_days_path)
    print(f"  Loaded representative_days.csv: {len(rep_days)} rows")

    # Load Week 8 LP and FCFS results for comparison
    lp_path = week8_dir / 'lp_results.csv'
    fcfs_path = week8_dir / 'fcfs_replay_results.csv'
    lp_df = pd.read_csv(lp_path) if lp_path.exists() else pd.DataFrame()
    fcfs_df = pd.read_csv(fcfs_path) if fcfs_path.exists() else pd.DataFrame()

    greedy_rows = []

    for station in REPRESENTATIVE_STATIONS:
        s_star = S_STAR_FAULTS_ON[station]
        label = STATION_LABELS.get(station, station)

        station_days = rep_days[rep_days['station_name'] == station]
        day_labels = station_days['day_label'].unique() if 'day_label' in station_days.columns else []

        for day_label in day_labels:
            day_data = station_days[station_days['day_label'] == day_label].copy()

            # Exclude faults (same as Week 8)
            if 'is_fault' in day_data.columns:
                nonfault = day_data[~day_data['is_fault'].astype(bool)].copy()
            else:
                nonfault = day_data.copy()

            if len(nonfault) == 0:
                continue

            nonfault = nonfault.sort_values(
                ['arrival_time_min', 'session_id']).reset_index(drop=True)

            # Greedy queue simulation (deterministic replay)
            result = _greedy_queue_replay(nonfault, s_star)
            result['station_name'] = station
            result['day_label'] = day_label
            result['n_chargers'] = s_star
            result['n_sessions'] = len(nonfault)
            result['n_total_with_faults'] = len(day_data)
            greedy_rows.append(result)

            print(f"  {label} / {day_label}: {len(nonfault)} sessions, "
                  f"mean_wait={result['mean_wait_min']:.2f} min, "
                  f"promotions={result['n_deadline_promotions']}")

    greedy_df = pd.DataFrame(greedy_rows)
    greedy_df.to_csv(output_dir / 'greedy_results.csv', index=False)
    print(f"\n  Saved: greedy_results.csv ({len(greedy_df)} rows)")

    # Build 3-way comparison
    comp_df = _build_three_way_comparison(greedy_df, lp_df, fcfs_df, output_dir)

    return greedy_df, comp_df


def _greedy_queue_replay(sessions: pd.DataFrame, n_chargers: int) -> dict:
    """Replay sessions through a greedy priority queue.

    Sessions are processed in arrival-time order. When a charger is free,
    the highest-priority queued session is served. Priority:
      1. Inflexible (or deadline-promoted) sessions first.
      2. Among flexible: earliest deadline first.
      3. Ties: earlier arrival, then lower session_id.

    This is a deterministic replay with observed service durations.
    """
    # Build session list with priority info
    sess_list = []
    for _, row in sessions.iterrows():
        arrival = row.get('arrival_time_min', 0)
        duration = row.get('service_duration_min',
                   row.get('duration_min',
                   row.get('charging_duration_min', 30)))

        # Flexibility info from Week 7 tier labels
        flex_tier = row.get('flexibility_tier',
                   row.get('flex_tier', 'inflexible'))
        is_flexible = flex_tier in ['likely_flexible', 'possibly_flexible']

        # Shift window (minutes)
        if flex_tier == 'likely_flexible':
            shift_window = 120  # 4 slots × 30 min
        elif flex_tier == 'possibly_flexible':
            shift_window = 60   # 2 slots × 30 min
        else:
            shift_window = 0

        deadline = arrival + shift_window

        sess_list.append({
            'session_id': row.get('session_id', 0),
            'arrival': arrival,
            'duration': max(duration, 0.5),
            'is_flexible': is_flexible,
            'deadline': deadline,
            'shift_window': shift_window,
        })

    # Sort by arrival time
    sess_list.sort(key=lambda x: (x['arrival'], x['session_id']))

    # Simulate greedy queue
    charger_free_at = [0.0] * n_chargers  # when each charger becomes free
    waits = []
    n_deadline_promotions = 0

    # Process events in arrival order
    queue = []
    event_idx = 0
    current_time = 0.0

    while event_idx < len(sess_list) or queue:
        # Add all sessions that have arrived by current_time
        while event_idx < len(sess_list) and sess_list[event_idx]['arrival'] <= current_time:
            queue.append(sess_list[event_idx])
            event_idx += 1

        # Find earliest free charger
        earliest_free = min(charger_free_at)
        earliest_charger = charger_free_at.index(earliest_free)

        if not queue:
            # No one waiting — advance to next arrival
            if event_idx < len(sess_list):
                current_time = sess_list[event_idx]['arrival']
                continue
            else:
                break

        # If charger not yet free, advance time
        if earliest_free > current_time:
            # Check if a new session arrives before charger frees
            if event_idx < len(sess_list) and sess_list[event_idx]['arrival'] < earliest_free:
                current_time = sess_list[event_idx]['arrival']
                continue
            else:
                current_time = earliest_free

        # Charger is free — serve highest priority from queue
        # Check deadline promotions
        for s in queue:
            if s['is_flexible'] and current_time > s['deadline']:
                s['is_flexible'] = False  # promote to inflexible
                n_deadline_promotions += 1

        # Priority sort: inflexible first, then earliest deadline, then FCFS
        queue.sort(key=lambda x: (
            1 if x['is_flexible'] else 0,  # inflexible = 0 (first)
            x['deadline'],                  # earliest deadline
            x['arrival'],                   # FCFS
            x['session_id'],               # final tie-break
        ))

        session = queue.pop(0)
        service_start = max(current_time, earliest_free, session['arrival'])
        wait = service_start - session['arrival']
        charger_free_at[earliest_charger] = service_start + session['duration']
        waits.append(wait)

    waits = np.array(waits) if waits else np.array([0.0])

    return {
        'mean_wait_min': float(waits.mean()),
        'median_wait_min': float(np.median(waits)),
        'p95_wait_min': float(np.percentile(waits, 95)) if len(waits) > 1 else 0,
        'p_wait_gt_10min': float((waits > 10).mean()),
        'p_wait_gt_15min': float((waits > 15).mean()),
        'max_wait_min': float(waits.max()),
        'n_deadline_promotions': n_deadline_promotions,
        'n_sessions_with_wait': int((waits > 0).sum()),
    }


def _build_three_way_comparison(greedy_df: pd.DataFrame,
                                 lp_df: pd.DataFrame,
                                 fcfs_df: pd.DataFrame,
                                 output_dir: Path) -> pd.DataFrame:
    """Build 3-way comparison: FCFS vs LP vs Greedy.

    Handles column-name variations from Week 8 outputs: the LP/FCFS
    files may use 'mean_wait_min' or 'mean_wait', and 'p_wait_gt_15min'
    or 'p_wait_gt_15'. The comparison tries known aliases.
    """
    if greedy_df.empty:
        return pd.DataFrame()

    # Column aliases: {canonical_name: [possible_names_in_week8_files]}
    _ALIASES = {
        'mean_wait_min': ['mean_wait_min', 'mean_wait', 'avg_wait_min'],
        'p_wait_gt_15min': ['p_wait_gt_15min', 'p_wait_gt_15', 'pw15'],
    }

    def _get_val(row, canonical):
        """Try canonical name and aliases."""
        for alias in _ALIASES.get(canonical, [canonical]):
            v = row.get(alias, None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
        return np.nan

    def _find_col(df, canonical):
        """Check if any alias exists in the dataframe."""
        for alias in _ALIASES.get(canonical, [canonical]):
            if alias in df.columns:
                return alias
        return None

    rows = []
    metrics = ['mean_wait_min', 'p_wait_gt_15min']

    for _, grow in greedy_df.iterrows():
        station = grow['station_name']
        day_label = grow['day_label']

        for metric in metrics:
            greedy_val = grow.get(metric, np.nan)

            # Find matching FCFS
            fcfs_val = np.nan
            if not fcfs_df.empty:
                col = _find_col(fcfs_df, metric)
                if col is not None:
                    fcfs_match = fcfs_df[
                        (fcfs_df['station_name'] == station) &
                        (fcfs_df['day_label'] == day_label)
                    ]
                    if len(fcfs_match) > 0:
                        fcfs_val = _get_val(fcfs_match.iloc[0], metric)

            # Find matching LP
            lp_val = np.nan
            if not lp_df.empty:
                col = _find_col(lp_df, metric)
                if col is not None:
                    lp_match = lp_df[
                        (lp_df['station_name'] == station) &
                        (lp_df['day_label'] == day_label)
                    ]
                    if len(lp_match) > 0:
                        lp_val = _get_val(lp_match.iloc[0], metric)

            rows.append({
                'station_name': station,
                'day_label': day_label,
                'metric': metric,
                'fcfs_value': round(fcfs_val, 4) if not np.isnan(fcfs_val) else np.nan,
                'lp_value': round(lp_val, 4) if not np.isnan(lp_val) else np.nan,
                'greedy_value': round(greedy_val, 4) if not np.isnan(greedy_val) else np.nan,
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_dir / 'three_way_comparison.csv', index=False)
    print(f"  Saved: three_way_comparison.csv ({len(comp_df)} rows)")

    # Print summary
    wait_rows = comp_df[comp_df['metric'] == 'mean_wait_min']
    if len(wait_rows) > 0:
        print(f"\n  3-way mean wait summary (across {len(wait_rows)} station-days):")
        for method in ['fcfs_value', 'lp_value', 'greedy_value']:
            vals = wait_rows[method].dropna()
            if len(vals) > 0:
                print(f"    {method}: mean={vals.mean():.3f} min")

    return comp_df


# =====================================================================
# 9D: ACTIVATION FRONTIER SWEEPS
# =====================================================================

def get_activation_vector(sched_df: pd.DataFrame, station: str,
                          scenario: str, n_chargers: int) -> np.ndarray:
    """Extract and cap activation schedule for a given fleet size.

    The heuristic schedule was computed against a generous installed
    capacity. For actual simulation at fleet size n_chargers, each
    hour's activation is capped at n_chargers.
    """
    mask = (sched_df['station'] == station) & \
           (sched_df['demand_scenario'] == scenario)
    sdata = sched_df[mask].sort_values('hour')

    if len(sdata) != 24:
        return np.full(24, n_chargers)

    raw = sdata['s_heuristic'].values.copy()
    capped = np.clip(raw, 1, n_chargers)
    return capped


def run_activation_sweeps(sched_df: pd.DataFrame,
                          param_path: str, nhpp_path: str,
                          mgs_path: str,
                          quarterly_path: str,
                          output_dir: Path,
                          s_star_map: Dict[str, int],
                          recent_quarter_map: Dict[str, str],
                          n_reps: int = 50,
                          sim_days: int = 30,
                          verbose: bool = False) -> pd.DataFrame:
    """Run activation frontier sweeps for all representative stations.

    Configurations:
      - Full activation (all chargers always on) — faults ON
        Uses historical-average NHPP arrivals.
      - Heuristic-Historical activation — faults ON
        Uses historical-average NHPP arrivals + historical heuristic schedule.
      - Heuristic-RecentQ activation — faults ON
        Uses RECENT-QUARTER NHPP arrivals + recent-quarter heuristic schedule.
        This is a true recent-demand simulation regime, not just a
        recent schedule on historical arrivals.
      - Heuristic-Historical activation — faults OFF (targeted comparison)
        Uses historical-average NHPP arrivals.

    Replications where drain_cap_hit is True are excluded from aggregation
    and flagged in the per-replication output.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 9D: ACTIVATION FRONTIER SWEEPS")
    print("=" * 70)

    if not HAS_ENGINE or not HAS_SIMPY:
        print("  [SKIP] SimPy or sim_engine not available.")
        return pd.DataFrame()

    # Load quarterly NHPP rates for the RecentQ regime
    quarterly_df = pd.read_csv(quarterly_path)

    all_rows = []
    all_rep_rows = []

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        recent_q = recent_quarter_map[station]
        recent_scenario = f'recent_quarter_{recent_q}'

        # Load recent-quarter NHPP rates for this station
        recent_rates_df = quarterly_df[
            (quarterly_df['station'] == station) &
            (quarterly_df['quarter'] == recent_q)
        ].sort_values('hour')
        has_recent_rates = len(recent_rates_df) == 24
        if has_recent_rates:
            recent_nhpp_24 = recent_rates_df['lambda_mean'].values
        else:
            print(f"  [WARN] {station}: no recent-quarter rates for {recent_q}, "
                  f"skipping heuristic_recentQ")

        configs_to_run = [
            # (activation_label, fault_label, sched_scenario, faults_on, use_recent_arrivals)
            ('full', 'ON', None, True, False),
            ('heuristic_historical', 'ON', 'historical_avg', True, False),
            ('heuristic_historical', 'OFF', 'historical_avg', False, False),
        ]
        if has_recent_rates:
            configs_to_run.append(
                ('heuristic_recentQ', 'ON', recent_scenario, True, True))

        for activation_label, fault_label, sched_scenario, faults_on, use_recent in configs_to_run:
            print(f"\n  {label} / {activation_label} / faults={fault_label}")

            for n_ch in FLEET_SIZES:
                # Build config from the correct NHPP source
                config = load_config_empirical(
                    param_path, nhpp_path, mgs_path, station,
                    n_chargers=n_ch, faults_enabled=faults_on,
                    sim_days=sim_days, random_seed=42
                )

                # For RecentQ: override NHPP rates with recent-quarter rates
                if use_recent and has_recent_rates:
                    config.nhpp_rates = recent_nhpp_24

                # Set activation schedule
                if sched_scenario is not None:
                    act_vec = get_activation_vector(
                        sched_df, station, sched_scenario, n_ch)
                    config.activation_schedule = act_vec
                else:
                    config.activation_schedule = None

                # Run simulation
                reps = simulate_station(config, n_replications=n_reps,
                                        verbose=False)

                # Per-replication records (include drain_cap_hit)
                n_drain_hits = 0
                for r in reps:
                    dch = r.get('drain_cap_hit', False)
                    if dch:
                        n_drain_hits += 1
                    all_rep_rows.append({
                        'station': station,
                        'n_chargers': n_ch,
                        'activation_mode': activation_label,
                        'faults': fault_label,
                        'replication': r['replication'],
                        'mean_utilization': r['mean_utilization'],
                        'mean_utilization_static': r['mean_utilization_static'],
                        'mean_wait': r['mean_wait'],
                        'p_wait_gt_15min': r['p_wait_gt_15min'],
                        'active_charger_hours': r['active_charger_hours'],
                        'scheduled_charger_hours': r['scheduled_charger_hours'],
                        'n_sessions': r['n_sessions'],
                        'fault_fraction': r['fault_fraction'],
                        'throughput_per_hour': r['throughput_per_hour'],
                        'drain_cap_hit': dch,
                    })

                if n_drain_hits > 0:
                    print(f"    [WARN] s={n_ch}: {n_drain_hits}/{n_reps} reps "
                          f"hit drain cap — excluding from aggregation")

                # Aggregate only non-drain-hit replications
                clean_reps = [r for r in reps if not r.get('drain_cap_hit', False)]
                if len(clean_reps) == 0:
                    print(f"    [WARN] s={n_ch}: ALL reps hit drain cap, skipping")
                    continue

                vals = {k: np.array([r[k] for r in clean_reps])
                        for k in ['mean_utilization', 'mean_utilization_static',
                                   'mean_wait', 'p_wait_gt_15min',
                                   'active_charger_hours', 'scheduled_charger_hours',
                                   'n_sessions', 'fault_fraction',
                                   'throughput_per_hour', 'p_wait_gt_10min',
                                   'median_wait', 'p95_wait']}

                row = {
                    'station': station,
                    'n_chargers': n_ch,
                    'activation_mode': activation_label,
                    'faults': fault_label,
                    'n_reps_clean': len(clean_reps),
                    'n_reps_drain_hit': n_drain_hits,
                }
                for k, v in vals.items():
                    row[f'{k}_mean'] = float(v.mean())
                    row[f'{k}_std'] = float(v.std())
                    row[f'{k}_ci95'] = float(1.96 * v.std() / np.sqrt(len(v)))

                pw15 = vals['p_wait_gt_15min']
                row['p_wait_gt_15min_upper95'] = float(
                    pw15.mean() + 1.96 * pw15.std() / np.sqrt(len(pw15)))

                all_rows.append(row)

                if verbose or n_ch % 6 == 0:
                    print(f"    s={n_ch:2d}: util={row['mean_utilization_mean']:.3f}, "
                          f"P(W>15)={row['p_wait_gt_15min_mean']:.4f}, "
                          f"ACH={row['active_charger_hours_mean']:.1f}/sim"
                          f" ({len(clean_reps)} clean reps)")

    agg_df = pd.DataFrame(all_rows)
    agg_df.to_csv(output_dir / 'pareto_activation_results.csv', index=False)
    print(f"\n  Saved: pareto_activation_results.csv ({len(agg_df)} rows)")

    rep_df = pd.DataFrame(all_rep_rows)
    rep_df.to_csv(output_dir / 'pareto_activation_replications.csv', index=False)
    print(f"  Saved: pareto_activation_replications.csv ({len(rep_df)} rows)")

    return agg_df


# =====================================================================
# 9E: FIGURES
# =====================================================================

def plot_capstone_frontiers(agg_df: pd.DataFrame, week6_path: str,
                            output_dir: Path, sim_days: int = 30):
    """Primary result: Active charger-hours per day vs P(Wq > 15 min).

    This is the clean, policy-comparable figure. x-axis is the 'cost'
    of running the station (active charger-hours). y-axis is the
    project-wide service quality metric.
    """
    if agg_df.empty:
        return

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Load Week 6 baseline for reference
    w6_df = pd.DataFrame()
    if Path(week6_path).exists():
        w6_df = pd.read_csv(week6_path)

    colors = {
        'full': '#1f77b4',
        'heuristic_historical': '#ff7f0e',
        'heuristic_recentQ': '#2ca02c',
    }
    markers = {
        'full': 'o',
        'heuristic_historical': 's',
        'heuristic_recentQ': '^',
    }

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        short = STATION_SHORT.get(station, station[:10])
        s_star = S_STAR_FAULTS_ON[station]

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot faults-ON curves for each activation mode
        for mode in ['full', 'heuristic_historical', 'heuristic_recentQ']:
            data = agg_df[
                (agg_df['station'] == station) &
                (agg_df['activation_mode'] == mode) &
                (agg_df['faults'] == 'ON')
            ].sort_values('n_chargers')

            if data.empty:
                continue

            # x = active charger-hours per day
            x = data['active_charger_hours_mean'].values / sim_days
            y = data['p_wait_gt_15min_mean'].values

            mode_label = mode.replace('_', ' ').title()
            ax.plot(x, y, marker=markers.get(mode, 'o'), color=colors.get(mode),
                    label=f'{mode_label} (faults ON)', linewidth=1.5, markersize=5)

            # Annotate fleet sizes
            for i, n_ch in enumerate(data['n_chargers'].values):
                if n_ch % 4 == 0 or n_ch == 2:
                    ax.annotate(f's={n_ch}', (x[i], y[i]),
                                textcoords='offset points', xytext=(5, 5),
                                fontsize=6, alpha=0.6)

        # Heuristic-Historical faults OFF (dashed)
        data_off = agg_df[
            (agg_df['station'] == station) &
            (agg_df['activation_mode'] == 'heuristic_historical') &
            (agg_df['faults'] == 'OFF')
        ].sort_values('n_chargers')

        if not data_off.empty:
            x_off = data_off['active_charger_hours_mean'].values / sim_days
            y_off = data_off['p_wait_gt_15min_mean'].values
            ax.plot(x_off, y_off, marker='s', color='#ff7f0e', linestyle='--',
                    alpha=0.5, label='Heuristic Historical (faults OFF)',
                    linewidth=1, markersize=4)

        ax.axhline(0.05, color='red', linestyle=':', alpha=0.5,
                   label='5% target')
        ax.set_xlabel('Active charger-hours per day (realized)')
        ax.set_ylabel('P(Wq > 15 min)')
        ax.set_title(f'{label}: Activation Frontier\n'
                     f'(primary result — active charger-hours vs service quality)')
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=-0.01)

        fig.tight_layout()
        fig.savefig(fig_dir / f'capstone_frontier_{short}.png')
        plt.close(fig)
        print(f"  Saved: capstone_frontier_{short}.png")


def plot_secondary_frontiers(agg_df: pd.DataFrame, output_dir: Path):
    """Secondary: Utilization vs P(Wq > 15 min).

    Explicitly labeled: utilization computed against time-varying
    realized active capacity.
    """
    if agg_df.empty:
        return

    fig_dir = output_dir / 'figures'

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        short = STATION_SHORT.get(station, station[:10])

        fig, ax = plt.subplots(figsize=(8, 5))

        for mode in ['full', 'heuristic_historical', 'heuristic_recentQ']:
            data = agg_df[
                (agg_df['station'] == station) &
                (agg_df['activation_mode'] == mode) &
                (agg_df['faults'] == 'ON')
            ].sort_values('n_chargers')

            if data.empty:
                continue

            x = data['mean_utilization_mean'].values
            y = data['p_wait_gt_15min_mean'].values
            mode_label = mode.replace('_', ' ').title()
            ax.plot(x, y, marker='o', label=f'{mode_label}',
                    linewidth=1.5, markersize=5)

        ax.axhline(0.05, color='red', linestyle=':', alpha=0.5, label='5% target')
        ax.set_xlabel('Mean utilization\n'
                      '(denominator: time-varying realized active capacity)')
        ax.set_ylabel('P(Wq > 15 min)')
        ax.set_title(f'{label}: Utilization–Wait Tradeoff\n'
                     f'(secondary — utilization uses realized-capacity denominator)')
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=-0.01)
        ax.set_xlim(left=0)

        fig.tight_layout()
        fig.savefig(fig_dir / f'secondary_frontier_{short}.png')
        plt.close(fig)
        print(f"  Saved: secondary_frontier_{short}.png")


def plot_master_overlay(agg_df: pd.DataFrame, week6_path: str,
                        greedy_df: pd.DataFrame, lp_df: pd.DataFrame,
                        output_dir: Path, s_star_map: Dict[str, int],
                        sim_days: int = 30):
    """Master Pareto overlay with all policies.

    IMPORTANT CAVEAT: This figure is illustrative, not a clean Pareto
    comparison in one common experimental regime. The activation curves
    come from 30-day stochastic NHPP simulation, while the LP and
    Greedy points come from Week 8 representative-day historical replay.
    The LP/Greedy x-coordinates are borrowed from the stochastic sweep.
    This is noted in the figure title.
    """
    if agg_df.empty:
        return

    fig_dir = output_dir / 'figures'

    # Load Week 6 faults-OFF baseline
    w6_df = pd.DataFrame()
    fault_off_path = Path(week6_path).parent / 'fault_off_sweep_results.csv'
    if fault_off_path.exists():
        w6_df = pd.read_csv(fault_off_path)

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        short = STATION_SHORT.get(station, station[:10])
        s_star = S_STAR_FAULTS_ON[station]

        fig, ax = plt.subplots(figsize=(9, 6))

        # 1. Full activation FCFS (faults ON) — primary baseline
        full_on = agg_df[
            (agg_df['station'] == station) &
            (agg_df['activation_mode'] == 'full') &
            (agg_df['faults'] == 'ON')
        ].sort_values('n_chargers')

        if not full_on.empty:
            x = full_on['active_charger_hours_mean'].values / sim_days
            y = full_on['p_wait_gt_15min_mean'].values
            ax.plot(x, y, 'o-', color='#1f77b4', linewidth=2,
                    label='1. Full FCFS (faults ON)', markersize=5)

        # 2. Heuristic activation (faults ON)
        heur_on = agg_df[
            (agg_df['station'] == station) &
            (agg_df['activation_mode'] == 'heuristic_historical') &
            (agg_df['faults'] == 'ON')
        ].sort_values('n_chargers')

        if not heur_on.empty:
            x2 = heur_on['active_charger_hours_mean'].values / sim_days
            y2 = heur_on['p_wait_gt_15min_mean'].values
            ax.plot(x2, y2, 's-', color='#ff7f0e', linewidth=1.5,
                    label='2. Heuristic Historical (faults ON)', markersize=5)

        # 3. LP point at s* (from Week 8)
        if not lp_df.empty and not full_on.empty:
            s_star_row = full_on[full_on['n_chargers'] == s_star]
            if len(s_star_row) > 0:
                x_lp = s_star_row['active_charger_hours_mean'].values[0] / sim_days
                # LP P(W>15) — use Week 8 value if available
                lp_station = lp_df[lp_df['station_name'] == station]
                if not lp_station.empty and 'p_wait_gt_15min' in lp_station.columns:
                    y_lp = lp_station['p_wait_gt_15min'].mean()
                else:
                    y_lp = s_star_row['p_wait_gt_15min_mean'].values[0]
                ax.scatter([x_lp], [y_lp], marker='D', s=80, color='purple',
                           zorder=5, label=f'3. LP at s*={s_star}')

        # 4. Greedy point at s*
        if not greedy_df.empty and not full_on.empty:
            greedy_station = greedy_df[greedy_df['station_name'] == station]
            if not greedy_station.empty:
                s_star_row = full_on[full_on['n_chargers'] == s_star]
                if len(s_star_row) > 0:
                    x_gr = s_star_row['active_charger_hours_mean'].values[0] / sim_days
                    y_gr = greedy_station['p_wait_gt_15min'].mean()
                    ax.scatter([x_gr], [y_gr], marker='*', s=120, color='green',
                               zorder=5, label=f'4. Greedy at s*={s_star}')

        # 5. Full activation faults OFF (dashed, reference)
        full_off = agg_df[
            (agg_df['station'] == station) &
            (agg_df['activation_mode'] == 'full') &
            (agg_df['faults'] == 'OFF')
        ]
        # Try Week 6 fault-off data if not in current sweep.
        # NOTE: Week 6 data uses n_chargers * 24 as x (scheduled full-
        # capacity charger-hours), not realized ACH. This is approximate
        # on the realized-ACH x-axis, but acceptable for a reference trace
        # on an illustrative figure.
        if full_off.empty and not w6_df.empty:
            full_off_w6 = w6_df[w6_df['station'] == station]
            if not full_off_w6.empty and 'p_wait_gt_15min_mean' in full_off_w6.columns:
                x5 = full_off_w6['n_chargers'].values * 24.0
                y5 = full_off_w6['p_wait_gt_15min_mean'].values
                ax.plot(x5, y5, 'o--', color='gray', alpha=0.4, linewidth=1,
                        label='5. Full FCFS no-faults (W6, approx x)',
                        markersize=3)

        ax.axhline(0.05, color='red', linestyle=':', alpha=0.5, label='5% target')
        ax.set_xlabel('Active charger-hours per day (realized)')
        ax.set_ylabel('P(Wq > 15 min)')
        ax.set_title(f'{label}: Master Pareto Overlay\n'
                     f'(illustrative — activation curves from NHPP simulation, '
                     f'LP/Greedy from historical replay)')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(bottom=-0.01)

        fig.tight_layout()
        fig.savefig(fig_dir / f'master_overlay_{short}.png')
        plt.close(fig)
        print(f"  Saved: master_overlay_{short}.png")


# =====================================================================
# 9F: SENSITIVITY ANALYSIS
# =====================================================================

def run_sensitivity(sched_df: pd.DataFrame,
                    param_path: str, nhpp_path: str,
                    mgs_path: str,
                    output_dir: Path,
                    n_reps: int = 30,
                    sim_days: int = 30) -> pd.DataFrame:
    """Sensitivity analysis: fault rate, wait threshold, repair median.

    Focal station: Gov Agency (only non-zero fault tax).
    Secondary: Bus Station (highest volume).

    Replications where drain_cap_hit fires are excluded from aggregation,
    matching the same discipline as run_activation_sweeps().

    Note: sensitivity_results.csv mixes two kinds of outputs:
      - 'simulated': fault_rate and repair_median dimensions run the
        simulator and report performance metrics (P(Wq>15), util, ACH).
      - 'schedule_only': wait_threshold dimension only recomputes the
        heuristic schedule and reports design outputs (total scheduled
        charger-hours, peak s_heuristic). These are NOT simulated.
    The 'output_type' column distinguishes them.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 9F: SENSITIVITY ANALYSIS")
    print("=" * 70)

    if not HAS_ENGINE or not HAS_SIMPY:
        print("  [SKIP] SimPy or sim_engine not available.")
        return pd.DataFrame()

    focal_stations = [
        'Xiuzhou_Government Agency',
        'Tongxiang_Bus Station',
    ]

    all_rows = []

    def _aggregate_with_drain_filter(reps, label_str):
        """Filter drain-hit reps and aggregate, matching sweep discipline."""
        clean = [r for r in reps if not r.get('drain_cap_hit', False)]
        n_hits = len(reps) - len(clean)
        if n_hits > 0:
            print(f"      [WARN] {label_str}: {n_hits}/{len(reps)} reps "
                  f"hit drain cap — excluded")
        if len(clean) == 0:
            print(f"      [WARN] {label_str}: ALL reps hit drain cap")
            return None, n_hits
        vals = {k: np.array([r[k] for r in clean])
                for k in ['p_wait_gt_15min', 'mean_utilization',
                           'active_charger_hours']}
        return vals, n_hits

    # --- Dimension 1: Fault rate ---
    print("\n  Dimension 1: Fault rate (simulated)")
    fault_rate_mults = [0.5, 1.0, 1.5]  # 9.3%, 18.6%, 27.9%

    for station in focal_stations:
        label = STATION_LABELS.get(station, station)
        s_star = S_STAR_FAULTS_ON[station]

        for mult in fault_rate_mults:
            scaled_rates = {
                k: min(v * mult, 0.95)
                for k, v in DEFAULT_FAULT_RATES.items()
            }

            for n_ch in [s_star - 2, s_star, s_star + 2]:
                if n_ch < 2:
                    continue

                config = load_config_empirical(
                    param_path, nhpp_path, mgs_path, station,
                    n_chargers=n_ch, faults_enabled=True,
                    sim_days=sim_days, random_seed=42
                )
                config.fault_rates = scaled_rates

                act_vec = get_activation_vector(
                    sched_df, station, 'historical_avg', n_ch)
                config.activation_schedule = act_vec

                reps = simulate_station(config, n_replications=n_reps,
                                        verbose=False)

                run_label = f"{label} s={n_ch} fault={mult*18.6:.0f}%"
                vals, n_drain_hits = _aggregate_with_drain_filter(reps, run_label)

                if vals is not None:
                    all_rows.append({
                        'station': station,
                        'sensitivity_dim': 'fault_rate',
                        'output_type': 'simulated',
                        'parameter_value': round(mult * 18.6, 1),
                        'n_chargers': n_ch,
                        'activation_mode': 'heuristic_historical',
                        'p_wait_gt_15min_mean': float(vals['p_wait_gt_15min'].mean()),
                        'p_wait_gt_15min_std': float(vals['p_wait_gt_15min'].std()),
                        'mean_utilization_mean': float(vals['mean_utilization'].mean()),
                        'active_charger_hours_mean': float(vals['active_charger_hours'].mean()),
                        'n_reps_clean': len(vals['p_wait_gt_15min']),
                        'n_reps_drain_hit': n_drain_hits,
                    })

                    print(f"    {run_label}: "
                          f"P(W>15)={vals['p_wait_gt_15min'].mean():.4f}")

    # --- Dimension 2: Wait threshold τ ---
    print("\n  Dimension 2: Wait threshold (schedule only — not simulated)")
    tau_values = [5, 10, 15, 20]

    for station in focal_stations:
        label = STATION_LABELS.get(station, station)
        s_star = S_STAR_FAULTS_ON.get(station, 24)

        nhpp_df = pd.read_csv(nhpp_path)
        station_rates = nhpp_df[nhpp_df['station'] == station].sort_values('hour')
        if len(station_rates) != 24:
            continue
        lam_24 = station_rates['lambda_mean'].values

        mixes = load_station_mixes(mgs_path)
        mix = mixes.get(station, {'DC_Fast': 0.33, 'Level_2': 0.34, 'Mixed': 0.33})
        mu_eff, _ = compute_effective_mu(mix)

        mgs_df = pd.read_csv(mgs_path)
        cv_row = mgs_df[(mgs_df['station'] == station) &
                        (mgs_df['sizing_scenario'] == 'historical_avg')]
        cv = cv_row.iloc[0]['cv'] if len(cv_row) > 0 else 1.4
        n_installed = max(int(s_star) + 6, 30)

        for tau in tau_values:
            sched_tau = compute_heuristic_schedule(
                lam_24, mu_eff, cv, n_installed, target_p=0.05, tau_min=tau)

            all_rows.append({
                'station': station,
                'sensitivity_dim': 'wait_threshold',
                'output_type': 'schedule_only',
                'parameter_value': tau,
                'n_chargers': -1,
                'activation_mode': f'heuristic_tau{tau}',
                'n_chargers_installed': int(n_installed),
                'total_scheduled_charger_hours': int(sched_tau.sum()),
                'peak_s_heuristic': int(sched_tau.max()),
                'min_s_heuristic': int(sched_tau.min()),
            })

            print(f"    {label} tau={tau}min: "
                  f"total_sch={sched_tau.sum()}/day, "
                  f"peak={sched_tau.max()}")

    # --- Dimension 3: Fault repair median (Gov Agency only) ---
    print("\n  Dimension 3: Fault repair median (simulated, Gov Agency)")
    station = 'Xiuzhou_Government Agency'
    s_star = S_STAR_FAULTS_ON[station]
    repair_medians = [15, 30, 60]

    for median in repair_medians:
        config = load_config_empirical(
            param_path, nhpp_path, mgs_path, station,
            n_chargers=s_star, faults_enabled=True,
            sim_days=sim_days, random_seed=42
        )
        config.fault_repair_median_min = float(median)
        act_vec = get_activation_vector(
            sched_df, station, 'historical_avg', s_star)
        config.activation_schedule = act_vec

        reps = simulate_station(config, n_replications=n_reps, verbose=False)

        run_label = f"Gov Agency s={s_star} repair={median}min"
        vals, n_drain_hits = _aggregate_with_drain_filter(reps, run_label)

        if vals is not None:
            all_rows.append({
                'station': station,
                'sensitivity_dim': 'repair_median',
                'output_type': 'simulated',
                'parameter_value': median,
                'n_chargers': s_star,
                'activation_mode': 'heuristic_historical',
                'p_wait_gt_15min_mean': float(vals['p_wait_gt_15min'].mean()),
                'p_wait_gt_15min_std': float(vals['p_wait_gt_15min'].std()),
                'mean_utilization_mean': float(vals['mean_utilization'].mean()),
                'active_charger_hours_mean': float(vals['active_charger_hours'].mean()),
                'n_reps_clean': len(vals['p_wait_gt_15min']),
                'n_reps_drain_hit': n_drain_hits,
            })

            print(f"    {run_label}: "
                  f"P(W>15)={vals['p_wait_gt_15min'].mean():.4f}")

    # --- Dimension 4: Arrival overdispersion (Gamma-Poisson/Cox process) ---
    print("\n  Dimension 4: Arrival overdispersion (simulated)")
    for station in focal_stations:
        s_star = S_STAR_FAULTS_ON[station]
        for alpha in [0.0, 0.25, 0.50]:
            config = load_config_empirical(
                param_path, nhpp_path, mgs_path, station,
                n_chargers=s_star, faults_enabled=True,
                sim_days=sim_days, random_seed=42)
            config.arrival_mode = 'nhpp_gamma_poisson'
            config.arrival_overdispersion_alpha = alpha
            config.activation_schedule = get_activation_vector(
                sched_df, station, 'historical_avg', s_star)
            reps = simulate_station(config, n_replications=n_reps,
                                    verbose=False)
            run_label = f"{STATION_LABELS.get(station, station)} alpha={alpha:.2f}"
            vals, n_drain_hits = _aggregate_with_drain_filter(reps, run_label)
            if vals is None:
                continue
            all_rows.append({
                'station': station,
                'sensitivity_dim': 'arrival_overdispersion_alpha',
                'output_type': 'simulated',
                'parameter_value': alpha,
                'n_chargers': s_star,
                'activation_mode': 'heuristic_historical',
                'p_wait_gt_15min_mean': float(
                    vals['p_wait_gt_15min'].mean()),
                'p_wait_gt_15min_std': float(
                    vals['p_wait_gt_15min'].std()),
                'mean_utilization_mean': float(
                    vals['mean_utilization'].mean()),
                'active_charger_hours_mean': float(
                    vals['active_charger_hours'].mean()),
                'n_reps_clean': len(vals['p_wait_gt_15min']),
                'n_reps_drain_hit': n_drain_hits,
            })

    sens_df = pd.DataFrame(all_rows)
    sens_df.to_csv(output_dir / 'sensitivity_results.csv', index=False)
    print(f"\n  Saved: sensitivity_results.csv ({len(sens_df)} rows)")

    return sens_df


# =====================================================================
# 9D+: SIMULATION-TUNED ACTIVATION SCHEDULES
# =====================================================================

def run_sim_tuned_schedules(sched_df: pd.DataFrame,
                            param_path: str, nhpp_path: str,
                            mgs_path: str,
                            output_dir: Path,
                            target_p: float = 0.05,
                            n_reps: int = 30,
                            sim_days: int = 30,
                            max_margin: int = 8) -> pd.DataFrame:
    """Compute simulation-tuned activation schedules for stations where
    the Erlang-C heuristic fails to meet the P(Wq>15min) target.

    Approach: take the heuristic schedule as a starting point and
    binary-search for the smallest uniform margin m (0 to max_margin)
    such that schedule(h) + m achieves P(Wq>15min) < target_p in the
    full simulator.

    A uniform margin is used rather than per-hour tuning because hours
    interact through queue carryover and session spillover. Per-hour
    tuning would require iterative re-simulation, which is expensive
    and hard to interpret. A uniform margin gives a clean single-number
    result: "the heuristic requires +m chargers everywhere to meet the
    service target in simulation."

    Only runs for stations where the heuristic schedule failed to meet
    the target in the frontier sweeps (detected from activation results).
    """
    print("\n" + "=" * 70)
    print("COMPONENT 9D+: SIMULATION-TUNED ACTIVATION SCHEDULES")
    print("=" * 70)

    if not HAS_ENGINE or not HAS_SIMPY:
        print("  [SKIP] SimPy or sim_engine not available.")
        return pd.DataFrame()

    if sched_df is None or sched_df.empty:
        schedule_path = output_dir / 'activation_schedules.csv'
        if not schedule_path.exists():
            print("  [SKIP] No activation schedules found.")
            return pd.DataFrame()
        sched_df = pd.read_csv(schedule_path)
        print(f"  Loaded existing activation schedules: {len(sched_df)} rows")

    # Load existing sweep results to identify failing stations
    sweep_path = output_dir / 'pareto_activation_results.csv'
    if not sweep_path.exists():
        print("  [SKIP] No sweep results found.")
        return pd.DataFrame()

    agg_df = pd.read_csv(sweep_path)

    results = []

    for station in REPRESENTATIVE_STATIONS:
        label = STATION_LABELS.get(station, station)
        short = STATION_SHORT.get(station, station[:10])

        # Check if heuristic_historical met the target at any fleet size
        heur_data = agg_df[
            (agg_df['station'] == station) &
            (agg_df['activation_mode'] == 'heuristic_historical') &
            (agg_df['faults'] == 'ON')
        ]
        if heur_data.empty:
            continue

        sched_hist = sched_df[
            (sched_df['station'] == station) &
            (sched_df['demand_scenario'] == 'historical_avg')
        ].sort_values('hour')

        if len(sched_hist) != 24:
            print(f"    [WARN] No 24-hour heuristic schedule found. Skipping.")
            continue

        heur_best_row = heur_data.loc[heur_data['p_wait_gt_15min_mean'].idxmin()]
        heur_ach = float(heur_best_row['active_charger_hours_mean'] / sim_days)
        base_schedule = sched_hist['s_heuristic'].values.copy()
        base_total = int(base_schedule.sum())
        base_peak = int(base_schedule.max())

        best_pw15 = heur_data['p_wait_gt_15min_mean'].min()
        if best_pw15 < target_p:
            print(f"\n  {label}: heuristic already meets target "
                  f"(best P(W>15)={best_pw15:.4f}). Skipping.")
            results.append({
                'station': station,
                'demand_scenario': 'historical_avg',
                'heuristic_best_pw15': best_pw15,
                'margin_needed': 0,
                'tuned_pw15': best_pw15,
                'heuristic_ach_per_day': heur_ach,
                'tuned_ach_per_day': heur_ach,
                'ach_increase': 0.0,
                'status': 'heuristic_sufficient',
                'tuned_schedule_peak': base_peak,
                'tuned_schedule_total': base_total,
            })
            continue

        print(f"\n  {label}: heuristic fails (best P(W>15)={best_pw15:.4f}). "
              f"Binary-searching margin...")

        # Use a generous installed capacity for tuning
        n_installed = max(int(base_schedule.max()) + max_margin + 2, 30)

        # Binary search for minimum margin
        lo, hi = 0, max_margin
        best_margin = None
        best_result_pw15 = None
        best_ach = None

        while lo <= hi:
            mid = (lo + hi) // 2
            test_schedule = np.clip(base_schedule + mid, 1, n_installed)

            config = load_config_empirical(
                param_path, nhpp_path, mgs_path, station,
                n_chargers=n_installed, faults_enabled=True,
                sim_days=sim_days, random_seed=42
            )
            config.activation_schedule = test_schedule

            reps = simulate_station(config, n_replications=n_reps, verbose=False)

            # Filter drain hits
            clean = [r for r in reps if not r.get('drain_cap_hit', False)]
            if len(clean) == 0:
                print(f"    margin=+{mid}: ALL reps hit drain cap")
                lo = mid + 1
                continue

            pw15 = np.mean([r['p_wait_gt_15min'] for r in clean])
            ach = np.mean([r['active_charger_hours'] for r in clean]) / sim_days

            print(f"    margin=+{mid}: P(W>15)={pw15:.4f}, "
                  f"ACH={ach:.1f}/day ({len(clean)} clean reps)")

            if pw15 < target_p:
                best_margin = mid
                best_result_pw15 = pw15
                best_ach = ach
                hi = mid - 1  # Try smaller margin
            else:
                lo = mid + 1  # Need more margin

        if best_margin is not None:
            print(f"    → Minimum margin: +{best_margin} chargers/hour")
            print(f"    → Tuned P(W>15) = {best_result_pw15:.4f}")
            print(f"    → Heuristic ACH: {heur_ach:.1f}/day, "
                  f"Tuned ACH: {best_ach:.1f}/day "
                  f"(+{best_ach - heur_ach:.1f})")

            results.append({
                'station': station,
                'demand_scenario': 'historical_avg',
                'heuristic_best_pw15': best_pw15,
                'margin_needed': best_margin,
                'tuned_pw15': best_result_pw15,
                'heuristic_ach_per_day': heur_ach,
                'tuned_ach_per_day': best_ach,
                'ach_increase': best_ach - heur_ach,
                'status': 'tuned',
                'tuned_schedule_peak': int(base_peak + best_margin),
                'tuned_schedule_total': int((base_schedule + best_margin).sum()),
            })
        else:
            print(f"    → Could not meet target within +{max_margin} margin")
            results.append({
                'station': station,
                'demand_scenario': 'historical_avg',
                'heuristic_best_pw15': best_pw15,
                'margin_needed': -1,
                'tuned_pw15': np.nan,
                'heuristic_ach_per_day': heur_ach,
                'tuned_ach_per_day': np.nan,
                'ach_increase': np.nan,
                'status': 'margin_exceeded',
                'tuned_schedule_peak': int(base_peak + max_margin),
                'tuned_schedule_total': int((base_schedule + max_margin).sum()),
            })

    tuned_df = pd.DataFrame(results)
    if not tuned_df.empty:
        tuned_df.to_csv(output_dir / 'sim_tuned_schedules.csv', index=False)
        print(f"\n  Saved: sim_tuned_schedules.csv ({len(tuned_df)} rows)")

    return tuned_df


# =====================================================================
# 9G: METADATA
# =====================================================================

def save_metadata(output_dir: Path, validation: dict,
                   sched_df: pd.DataFrame, greedy_df: pd.DataFrame,
                   agg_df: pd.DataFrame, sens_df: pd.DataFrame,
                   tuned_df: pd.DataFrame = None,
                   sim_days: int = 30) -> dict:
    """Assemble and save week9_metadata.json."""

    metadata = {
        'week': 9,
        'simulation_days': int(sim_days),
        'warmup_days': 7,
        'components': [
            '9A: Sim engine validation (activation + realized capacity)',
            '9B: Heuristic activation schedule computation',
            '9C: Greedy historical replay + 3-way comparison',
            '9D: Activation frontier sweeps',
            '9D+: Simulation-tuned activation schedules',
            '9E: Frontier figures',
            '9F: Sensitivity analysis',
        ],
        'design_decisions': {
            'activation_heuristic': (
                'Hourly Erlang-C / Allen-Cunneen computation is an '
                'analytically-motivated initializer, not a derived optimum. '
                'Hour-isolation assumption makes it systematically optimistic. '
                'The simulator is the binding evaluation.'
            ),
            'utilization_denominator': (
                'Realized active-server-minutes (event-driven integral): '
                'max(scheduled_capacity, n_users) at each state-change event. '
                'Both realized and static denominators are reported.'
            ),
            'greedy_replay': (
                'Historical replay on Week 8 station-days only. Same sessions, '
                'same observed durations, same fleet sizes. No fault buffer, '
                'no ML anticipation, no stochastic tier assignment.'
            ),
            'greedy_tiebreak': (
                '1. Inflexible first. 2. Earliest deadline among flexible. '
                '3. Earlier arrival (FCFS). 4. Lower session_id.'
            ),
            'recent_quarter': (
                'Most recent non-degenerate quarter per station. '
                'Tech Park: 2021Q3 (2021Q4 has zero arrivals). '
                'Others: 2021Q4. NOT synchronized calendar quarters.'
            ),
            'primary_figure_axis': (
                'Active charger-hours per day (realized) vs P(Wq > 15 min). '
                'Utilization-vs-wait is secondary and labeled as using '
                'time-varying realized active capacity denominator.'
            ),
            'activation_cap': (
                's_heuristic is NOT capped at historical s*. '
                'Capped at n_chargers (installed fleet size for that sweep point).'
            ),
            'fault_interaction': (
                'Targeted faults-OFF comparison run for Heuristic-Historical '
                'activation mode (not just Full activation). This gives a '
                'direct handle on whether activation changes the fault tax.'
            ),
        },
        'validation': validation,
        'representative_stations': REPRESENTATIVE_STATIONS,
        's_star_faults_on': S_STAR_FAULTS_ON,
        'recent_quarters': RECENT_QUARTER,
        'n_activation_schedule_rows': len(sched_df) if sched_df is not None else 0,
        'n_greedy_station_days': len(greedy_df) if greedy_df is not None else 0,
        'n_frontier_sweep_rows': len(agg_df) if agg_df is not None else 0,
        'n_sensitivity_rows': len(sens_df) if sens_df is not None else 0,
        'n_sim_tuned_rows': len(tuned_df) if tuned_df is not None else 0,
        'files_produced': [
            'activation_schedules.csv',
            'greedy_results.csv',
            'three_way_comparison.csv',
            'pareto_activation_results.csv',
            'pareto_activation_replications.csv',
            'sensitivity_results.csv',
            'sim_tuned_schedules.csv',
            'week9_metadata.json',
            'figures/activation_bars_*.png',
            'figures/capstone_frontier_*.png',
            'figures/secondary_frontier_*.png',
            'figures/master_overlay_*.png',
        ],
    }

    with open(output_dir / 'week9_metadata.json', 'w') as f:
        json.dump(to_builtin(metadata), f, indent=2)
    print(f"\n  Saved: week9_metadata.json")

    return metadata


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Week 9: Demand-Responsive Activation + Greedy Scheduler')
    parser.add_argument('--data-dir', default=str(DATA_DIR),
                        help='Path to data directory')
    parser.add_argument('--output-dir',
                        default=str(RESULTS_DIR / 'week9_results'),
                        help='Output directory')
    parser.add_argument('--week6-dir',
                        default=str(RESULTS_DIR / 'week6_results'),
                        help='Path to Week 6 results')
    parser.add_argument('--week8-dir',
                        default=str(RESULTS_DIR / 'week8_results'),
                        help='Path to Week 8 results')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip sim engine validation')
    parser.add_argument('--skip-sweeps', action='store_true',
                        help='Skip activation frontier sweeps')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip sensitivity analysis')
    parser.add_argument('--n-reps', type=int, default=50,
                        help='Replications per configuration')
    parser.add_argument('--sim-days', type=int, default=30,
                        help='Simulated days per replication')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    data_dir = (PROJECT_ROOT / args.data_dir).resolve() \
        if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    week6_dir = (PROJECT_ROOT / args.week6_dir).resolve() \
        if not Path(args.week6_dir).is_absolute() else Path(args.week6_dir)
    week8_dir = (PROJECT_ROOT / args.week8_dir).resolve() \
        if not Path(args.week8_dir).is_absolute() else Path(args.week8_dir)
    week3_dir = RESULTS_DIR / 'week3_results'
    week4_dir = RESULTS_DIR / 'week4_results'

    # Resolve upstream artifacts against this repo's actual layout.
    param_path = resolve_input_path(
        'parameter_summary.json',
        data_dir / 'parameter_summary.json',
        RESULTS_DIR / 'parameter_summary.json',
        week4_dir / 'parameter_summary.json',
    )
    nhpp_path = resolve_input_path(
        'nhpp_rate_functions.csv',
        data_dir / 'nhpp_rate_functions.csv',
        RESULTS_DIR / 'nhpp_rate_functions.csv',
        week3_dir / 'nhpp_rate_functions.csv',
    )
    quarterly_path = resolve_input_path(
        'nhpp_rate_functions_quarterly.csv',
        data_dir / 'nhpp_rate_functions_quarterly.csv',
        RESULTS_DIR / 'nhpp_rate_functions_quarterly.csv',
        week3_dir / 'nhpp_rate_functions_quarterly.csv',
    )
    mgs_path = resolve_input_path(
        'mgs_comparison_4rep.csv',
        data_dir / 'mgs_comparison_4rep.csv',
        RESULTS_DIR / 'mgs_comparison_4rep.csv',
        week4_dir / 'mgs_comparison_4rep.csv',
    )
    fault_tax_path = resolve_input_path(
        'fault_tax_results.csv',
        week6_dir / 'fault_tax_results.csv',
        RESULTS_DIR / 'fault_tax_results.csv',
    )

    # Load upstream constants from artifacts instead of hard-coding
    global S_STAR_FAULTS_ON, RECENT_QUARTER, SERVICE_MEANS
    S_STAR_FAULTS_ON = load_s_star_from_file(fault_tax_path)
    RECENT_QUARTER = detect_recent_quarters(quarterly_path)
    SERVICE_MEANS = load_service_means_from_file(param_path)

    print("=" * 70)
    print("WEEK 9: DEMAND-RESPONSIVE ACTIVATION + GREEDY SCHEDULER")
    print("=" * 70)
    print(f"  Data dir:   {data_dir}")
    print(f"  Week 6 dir: {week6_dir}")
    print(f"  Week 8 dir: {week8_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Reps: {args.n_reps}, Sim days: {args.sim_days}")
    print(f"  sim_engine: {SIM_ENGINE_DIR / 'sim_engine.py'}")
    print(f"  parameter_summary: {param_path}")
    print(f"  nhpp_rates: {nhpp_path}")
    print(f"  nhpp_quarterly: {quarterly_path}")
    print(f"  mgs: {mgs_path}")
    print(f"  fault_tax: {fault_tax_path}")
    print(f"  s* (from file): {S_STAR_FAULTS_ON}")
    print(f"  Recent quarters: {RECENT_QUARTER}")

    run_only_tuned = (
        args.skip_validation and args.skip_sweeps and args.skip_sensitivity
    )
    if run_only_tuned:
        print("  Mode: reuse existing schedules/sweeps and run 9D+ only")

    # ── 9A: Validation ───────────────────────────────────────────
    validation = {}
    if not args.skip_validation:
        validation = run_validation_tests(param_path, nhpp_path, mgs_path,
                                         output_dir,
                                         sim_days=args.sim_days)
    else:
        print("\n  [SKIP] Validation")

    # ── 9B: Heuristic activation schedules ───────────────────────
    sched_df = pd.DataFrame()
    schedule_path = output_dir / 'activation_schedules.csv'
    if run_only_tuned and schedule_path.exists():
        sched_df = pd.read_csv(schedule_path)
        print(f"\n  Loaded existing activation_schedules.csv: {len(sched_df)} rows")
    else:
        sched_df = build_activation_schedules(
            nhpp_path, quarterly_path, mgs_path, output_dir)

    # ── 9C: Greedy replay ────────────────────────────────────────
    if run_only_tuned:
        print("\n  [SKIP] Greedy historical replay (9D+ only mode)")
        greedy_df, comp_df = pd.DataFrame(), pd.DataFrame()
    else:
        greedy_df, comp_df = run_greedy_replay(week8_dir, output_dir)

    # ── 9D: Activation frontier sweeps ───────────────────────────
    agg_df = pd.DataFrame()
    if not args.skip_sweeps:
        agg_df = run_activation_sweeps(
            sched_df, param_path, nhpp_path, mgs_path, quarterly_path, output_dir,
            s_star_map=S_STAR_FAULTS_ON,
            recent_quarter_map=RECENT_QUARTER,
            n_reps=args.n_reps, sim_days=args.sim_days,
            verbose=args.verbose)
    else:
        print("\n  [SKIP] Activation frontier sweeps")
        sweep_path = output_dir / 'pareto_activation_results.csv'
        if sweep_path.exists():
            agg_df = pd.read_csv(sweep_path)
            print(f"  Loaded existing: {len(agg_df)} rows")

            # Staleness guard: warn if the reloaded sweep CSV is older
            # than the analysis script (proxy for pre-patch artifacts).
            sweep_mtime = os.path.getmtime(sweep_path)
            script_mtime = os.path.getmtime(__file__)
            if sweep_mtime < script_mtime:
                print("  " + "!" * 60)
                print("  WARNING: pareto_activation_results.csv is older than "
                      "this script.")
                print("  It may have been generated with the heuristic charger "
                      "mix (pre-patch).")
                print("  9D+ tuning results will be UNRELIABLE if the sweep "
                      "data uses a")
                print("  different charger-type mix. Re-run without "
                      "--skip-sweeps to regenerate.")
                print("  " + "!" * 60)

    # ── 9E: Figures ──────────────────────────────────────────────
    if not agg_df.empty and not run_only_tuned:
        print("\n" + "=" * 70)
        print("COMPONENT 9E: FIGURES")
        print("=" * 70)

        w6_sweep_path = str(week6_dir / 'pareto_sweep_results.csv')
        plot_capstone_frontiers(agg_df, w6_sweep_path, output_dir,
                                sim_days=args.sim_days)
        plot_secondary_frontiers(agg_df, output_dir)

        lp_df = pd.DataFrame()
        lp_path = week8_dir / 'lp_results.csv'
        if lp_path.exists():
            lp_df = pd.read_csv(lp_path)

        plot_master_overlay(agg_df, w6_sweep_path, greedy_df, lp_df,
                            output_dir, s_star_map=S_STAR_FAULTS_ON,
                            sim_days=args.sim_days)
    elif run_only_tuned:
        print("\n  [SKIP] Figures (9D+ only mode)")

    # ── 9F: Sensitivity ─────────────────────────────────────────
    sens_df = pd.DataFrame()
    if not args.skip_sensitivity:
        sens_df = run_sensitivity(
            sched_df, param_path, nhpp_path, mgs_path, output_dir,
            n_reps=min(args.n_reps, 30), sim_days=args.sim_days)
    else:
        print("\n  [SKIP] Sensitivity analysis")

    # ── 9D+: Simulation-tuned schedules ─────────────────────────
    tuned_df = pd.DataFrame()
    if not agg_df.empty:
        tuned_df = run_sim_tuned_schedules(
            sched_df, param_path, nhpp_path, mgs_path, output_dir,
            target_p=0.05, n_reps=min(args.n_reps, 30),
            sim_days=args.sim_days)
    else:
        print("\n  [SKIP] Simulation-tuned schedules")

    # ── 9G: Metadata ─────────────────────────────────────────────
    save_metadata(output_dir, validation, sched_df, greedy_df, agg_df,
                  sens_df, tuned_df, sim_days=args.sim_days)

    print("\n" + "=" * 70)
    print("WEEK 9 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
