#!/usr/bin/env python3
"""
Week 10 Lookup Regression and Independent-Seed Robustness Checks
=========================================================

Purpose:
    For 10 carefully chosen configurations, run fresh SimPy simulations
    (50 reps × 7 warm-up days + 30 measured days each) and compare the
    lookup-based simulator would return. This validates that the simulator
    faithfully represents the underlying simulation engine.

Validation matrix (10 configs):
    #  Station          Fleet  Activation             Faults  Purpose
    1  Expressway A       4    full                   ON      Exact sweep, baseline s*
    2  Expressway A       6    full                   ON      Exact higher-fleet sweep point
    3  Tech Park          6    heuristic_historical   ON      Activation at s*
    4  Tech Park          8    heuristic_historical   ON      Exact higher-fleet sweep point
    5  Gov Agency         8    heuristic_historical   ON      Activation at s*, L2-heavy
    6  Gov Agency         8    heuristic_historical   OFF     Faults-OFF comparator
    7  Gov Agency        10    heuristic_historical   ON      Exact higher-fleet L2 point
    8  Bus Station       10    full                   ON      Exact sweep, high-volume
    9  Bus Station       10    heuristic_recentQ      ON      Recent-quarter test
   10  Bus Station       16    heuristic_historical   ON      Exact over-provisioned regime

Error criteria:
    - Utilization: <10% relative error
    - P(W>15min): absolute error <2 pp when the regime is near-zero
      (P(W>15min) < 2%); otherwise <10% relative error
    - Both metrics must pass for a configuration to be marked PASS

Outputs:
    - validation_results.csv   (per-config comparison)
    - validation_summary.txt   (pass/fail summary)

Usage:
    python week10_validation.py --data-dir ./Data --output-dir ./Results/week10_results

Dependencies:
    - sim_engine.py (in data-dir or on sys.path)
    - parameter_summary.json
    - nhpp_rate_functions.csv
    - nhpp_rate_functions_quarterly.csv
    - activation_schedules.csv
    - pareto_activation_results.csv
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from project_paths import (CODE_DIR, DATA_DIR, PROJECT_ROOT, RESULTS_DIR,
                           to_builtin)

def resolve_input_path(label: str, *candidates: Path) -> str:
    """Return the first existing candidate path, else the first candidate."""
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    first = candidates[0]
    print(f"  [WARN] Could not resolve {label}; expected one of:")
    for candidate in candidates:
        print(f"    - {candidate}")
    return str(first)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_signature(path: Path) -> dict:
    p = Path(path)
    return {
        "path": str(p.resolve()),
        "sha256": sha256_file(p),
        "mtime": os.path.getmtime(p),
        "size_bytes": os.path.getsize(p),
    }


def warn_if_older(target_path: str, upstream_paths: List[str], target_label: str):
    target_mtime = os.path.getmtime(target_path)
    stale_against = [
        p for p in upstream_paths
        if Path(p).exists() and target_mtime < os.path.getmtime(p)
    ]
    if not stale_against:
        return

    print("  " + "!" * 60)
    print(f"  WARNING: {target_label} may be stale.")
    for p in stale_against:
        print(f"    older than: {p}")
    print("  Re-run the upstream week(s) and validation before using final verdicts.")
    print("  " + "!" * 60)


# ============================================================================
# VALIDATION MATRIX
# ============================================================================

VALIDATION_CONFIGS = [
    # (id, station, fleet, activation_mode, faults, purpose)
    (1,  "Xiuzhou_Expressway Service District A", 4,  "full",                  "ON",  "Exact sweep point, baseline s*"),
    (2,  "Xiuzhou_Expressway Service District A", 6,  "full",                  "ON",  "Exact higher-fleet sweep point"),
    (3,  "Nanhu_Technology Park",                 6,  "heuristic_historical",  "ON",  "Activation at s*"),
    (4,  "Nanhu_Technology Park",                 8,  "heuristic_historical",  "ON",  "Exact higher-fleet sweep point"),
    (5,  "Xiuzhou_Government Agency",             8,  "heuristic_historical",  "ON",  "Activation at s*, L2-heavy"),
    (6,  "Xiuzhou_Government Agency",             8,  "heuristic_historical",  "OFF", "Faults-OFF comparator"),
    (7,  "Xiuzhou_Government Agency",            10,  "heuristic_historical",  "ON",  "Exact higher-fleet L2 point"),
    (8,  "Tongxiang_Bus Station",                10,  "full",                  "ON",  "Exact sweep, high-volume"),
    (9,  "Tongxiang_Bus Station",                10,  "heuristic_recentQ",     "ON",  "Recent-quarter test"),
    (10, "Tongxiang_Bus Station",                16,  "heuristic_historical",  "ON",  "Exact over-provisioned regime"),
]

# Recent quarter mapping (from Week 9 detect_recent_quarters)
RECENT_QUARTER_MAP = {
    "Xiuzhou_Expressway Service District A": "2021Q4",
    "Nanhu_Technology Park": "2021Q3",
    "Xiuzhou_Government Agency": "2021Q4",
    "Tongxiang_Bus Station": "2021Q4",
}


# ============================================================================
# SIMULATOR LOOKUP ENGINE (mirrors the JSX logic exactly)
# ============================================================================

def load_pareto_data(path: str) -> pd.DataFrame:
    """Load pareto_activation_results.csv."""
    return pd.read_csv(path)


def simulator_lookup(pareto_df: pd.DataFrame, station: str,
                     mode: str, faults: str, fleet: int
                     ) -> Optional[Dict]:
    """
    Reproduce the simulator's exact-fleet lookup logic.

    Rules (from locked design / current JSX):
      - Exact lookup for fleet sizes that exist in the data
      - Unsupported fleet sizes return no lookup value
      - No interpolation across fleet sizes or categories
    """
    mask = ((pareto_df['station'] == station) &
            (pareto_df['activation_mode'] == mode) &
            (pareto_df['faults'] == faults))
    subset = pareto_df[mask].copy()

    if subset.empty:
        return None

    # Exact lookup
    exact = subset[subset['n_chargers'] == fleet]
    if len(exact) == 1:
        row = exact.iloc[0]
        return {
            'util_realized': row['mean_utilization_mean'],
            'p_wait_15': row['p_wait_gt_15min_mean'],
            'mean_wait': row['mean_wait_mean'],
            'active_ch_hrs': row['active_charger_hours_mean'],
            'n_sessions': row['n_sessions_mean'],
            'interpolated': False,
            'method': f'exact (s={fleet})',
        }

    return None


# ============================================================================
# SIMPY RUNNER
# ============================================================================

def run_simpy_config(config_id: int, station: str, fleet: int,
                     activation_mode: str, faults: str,
                     param_path: str, nhpp_path: str,
                     mgs_path: str,
                     quarterly_path: str, sched_df: pd.DataFrame,
                     n_reps: int = 50, sim_days: int = 30,
                     validation_seed: int = 987654,
                     verbose: bool = False) -> Dict:
    """
    Run a single validation configuration through SimPy.

    Returns dict with aggregated metrics (mean over clean reps).

    Uses the empirical charger-type mix from mgs_comparison_4rep.csv
    to match Week 9's load_config_empirical() approach, ensuring the
    "fresh SimPy" side of validation uses the same mix assumptions as
    the sweep data it validates against.
    """
    from sim_engine import load_config, simulate_station

    faults_on = (faults == "ON")

    # Build base config
    config = load_config(
        param_path, nhpp_path, station,
        n_chargers=fleet, faults_enabled=faults_on,
        sim_days=sim_days, random_seed=validation_seed
    )

    # Patch in empirical charger-type mix (matching Week 9 post-patch)
    mgs_df = pd.read_csv(mgs_path)
    mix_row = mgs_df[(mgs_df['station'] == station) &
                     (mgs_df['sizing_scenario'] == 'historical_avg')]
    if len(mix_row) > 0:
        config.charger_type_mix = json.loads(
            mix_row.iloc[0]['station_mix_json'])
    else:
        raise ValueError(
            f"run_simpy_config: no empirical charger-type mix found "
            f"for '{station}' in {mgs_path}. Cannot validate against "
            f"sweep data that uses empirical mixes."
        )

    # For heuristic_recentQ: override NHPP rates with recent-quarter rates
    if activation_mode == "heuristic_recentQ":
        recent_q = RECENT_QUARTER_MAP[station]
        qdf = pd.read_csv(quarterly_path)
        recent_rates = qdf[(qdf['station'] == station) &
                           (qdf['quarter'] == recent_q)].sort_values('hour')
        if len(recent_rates) == 24:
            config.nhpp_rates = recent_rates['lambda_mean'].values
        else:
            print(f"  [WARN] Config {config_id}: recent-quarter rates "
                  f"not found for {station}/{recent_q}, using historical")

    # Set activation schedule
    if activation_mode == "full":
        config.activation_schedule = None  # all chargers always on
    elif activation_mode in ("heuristic_historical", "heuristic_recentQ"):
        if activation_mode == "heuristic_historical":
            scenario = "historical_avg"
        else:
            recent_q = RECENT_QUARTER_MAP[station]
            scenario = f"recent_quarter_{recent_q}"

        mask = ((sched_df['station'] == station) &
                (sched_df['demand_scenario'] == scenario))
        sdata = sched_df[mask].sort_values('hour')

        if len(sdata) == 24:
            raw = sdata['s_heuristic'].values.copy()
            act_vec = np.clip(raw, 1, fleet)
            config.activation_schedule = act_vec
        else:
            print(f"  [WARN] Config {config_id}: schedule not found for "
                  f"{station}/{scenario}, using full activation")
            config.activation_schedule = None

    # Run simulation
    t0 = time.time()
    reps = simulate_station(config, n_replications=n_reps, verbose=verbose)
    elapsed = time.time() - t0

    # Filter out drain-cap hits
    clean = [r for r in reps if not r.get('drain_cap_hit', False)]
    n_clean = len(clean)
    n_excluded = len(reps) - n_clean

    if n_clean == 0:
        print(f"  [WARN] Config {config_id}: ALL {len(reps)} reps hit "
              f"drain cap. No clean results.")
        return {
            'n_reps_clean': 0,
            'n_reps_excluded': n_excluded,
            'util_realized': np.nan,
            'p_wait_15': np.nan,
            'mean_wait': np.nan,
            'active_ch_hrs': np.nan,
            'n_sessions': np.nan,
            'elapsed_sec': elapsed,
        }

    # Aggregate
    util_vals = [r['mean_utilization'] for r in clean]
    pw15_vals = [r['p_wait_gt_15min'] for r in clean]
    mw_vals = [r['mean_wait'] for r in clean]
    ach_vals = [r.get('active_charger_hours', r.get('realized_capacity_minutes', 0) / 60)
                for r in clean]
    ns_vals = [r['n_sessions'] for r in clean]

    return {
        'n_reps_clean': n_clean,
        'n_reps_excluded': n_excluded,
        'util_realized': float(np.mean(util_vals)),
        'p_wait_15': float(np.mean(pw15_vals)),
        'mean_wait': float(np.mean(mw_vals)),
        'active_ch_hrs': float(np.mean(ach_vals)),
        'n_sessions': float(np.mean(ns_vals)),
        'elapsed_sec': elapsed,
    }


# ============================================================================
# ERROR COMPUTATION AND PASS/FAIL
# ============================================================================

REL_ERROR_THRESHOLD = 0.10   # 10% relative error
ABS_ERROR_THRESHOLD = 0.005  # 0.5 percentage points near zero
NEAR_ZERO_THRESHOLD = 0.02
NEAR_ZERO_REL_THRESHOLD = 0.25


def compute_errors(sim_val: float, lookup_val: float,
                   is_probability: bool = False
                   ) -> Tuple[float, float, str, str]:
    """
    Compute relative and absolute error, and determine pass/fail.

    Rules:
      - Utilization: relative error only
      - P(W>15min): absolute error only in the near-zero regime
        (max(sim, lookup) < 2%); otherwise relative error only

    Returns (rel_error, abs_error, criterion, verdict).
    """
    abs_err = abs(sim_val - lookup_val)

    if sim_val == 0 and lookup_val == 0:
        criterion = "absolute (near-zero)" if is_probability else "relative"
        return 0.0, 0.0, criterion, "PASS"

    # Relative error: use max(sim, lookup) as denominator to avoid
    # division by zero when one is zero but the other is small
    denom = max(abs(sim_val), abs(lookup_val))
    rel_err = abs_err / denom if denom > 0 else 0.0

    if is_probability and max(sim_val, lookup_val) < NEAR_ZERO_THRESHOLD:
        criterion = "absolute AND relative (near-zero)"
        verdict = ("PASS" if abs_err < ABS_ERROR_THRESHOLD
                   and rel_err < NEAR_ZERO_REL_THRESHOLD else "FAIL")
    else:
        criterion = "relative"
        verdict = "PASS" if rel_err < REL_ERROR_THRESHOLD else "FAIL"

    return rel_err, abs_err, criterion, verdict


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Week 10 lookup regression and robustness checks")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory with parameter files and sim_engine.py")
    parser.add_argument("--output-dir", type=str,
                        default=str(RESULTS_DIR / 'week10_results'),
                        help="Output directory for results")
    parser.add_argument("--n-reps", type=int, default=50,
                        help="Replications per configuration (default: 50)")
    parser.add_argument("--sim-days", type=int, default=30,
                        help="Measured days per replication after warm-up (default: 30)")
    parser.add_argument(
        "--validation-seed", type=int, default=987654,
        help="Independent seed for robustness runs (default: 987654)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-replication progress")
    parser.add_argument("--skip-simpy", action="store_true",
                        help="Skip SimPy runs, only compute lookup values "
                             "(for testing the lookup logic)")
    args = parser.parse_args()

    data_dir = (PROJECT_ROOT / args.data_dir).resolve() \
        if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    week3_dir = RESULTS_DIR / "week3_results"
    week4_dir = RESULTS_DIR / "week4_results"
    week5_dir = RESULTS_DIR / "week5_results"
    week6_dir = RESULTS_DIR / "week6_results"
    week8_dir = RESULTS_DIR / "week8_results"
    week9_dir = RESULTS_DIR / "week9_results"

    sim_engine_path = resolve_input_path(
        "sim_engine.py",
        data_dir / "sim_engine.py",
        week5_dir / "sim_engine.py",
        CODE_DIR / "sim_engine.py",
    )

    # Add the selected sim_engine directory first, then fallback locations.
    for path in [Path(sim_engine_path).parent, data_dir, CODE_DIR, week5_dir]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # Resolve upstream artifacts against this repo's actual layout.
    param_path = resolve_input_path(
        "parameter_summary.json",
        data_dir / "parameter_summary.json",
        RESULTS_DIR / "parameter_summary.json",
        week4_dir / "parameter_summary.json",
    )
    nhpp_path = resolve_input_path(
        "nhpp_rate_functions.csv",
        data_dir / "nhpp_rate_functions.csv",
        RESULTS_DIR / "nhpp_rate_functions.csv",
        week3_dir / "nhpp_rate_functions.csv",
    )
    mgs_path = resolve_input_path(
        "mgs_comparison_4rep.csv",
        data_dir / "mgs_comparison_4rep.csv",
        RESULTS_DIR / "mgs_comparison_4rep.csv",
        week4_dir / "mgs_comparison_4rep.csv",
    )
    quarterly_path = resolve_input_path(
        "nhpp_rate_functions_quarterly.csv",
        data_dir / "nhpp_rate_functions_quarterly.csv",
        RESULTS_DIR / "nhpp_rate_functions_quarterly.csv",
        week3_dir / "nhpp_rate_functions_quarterly.csv",
    )
    sched_path = resolve_input_path(
        "activation_schedules.csv",
        data_dir / "activation_schedules.csv",
        RESULTS_DIR / "activation_schedules.csv",
        week9_dir / "activation_schedules.csv",
    )
    pareto_path = resolve_input_path(
        "pareto_activation_results.csv",
        data_dir / "pareto_activation_results.csv",
        RESULTS_DIR / "pareto_activation_results.csv",
        week9_dir / "pareto_activation_results.csv",
    )
    week6_fault_tax_path = resolve_input_path(
        "fault_tax_results.csv",
        data_dir / "fault_tax_results.csv",
        RESULTS_DIR / "fault_tax_results.csv",
        week6_dir / "fault_tax_results.csv",
    )
    week6_meta_path = resolve_input_path(
        "week6_metadata.json",
        data_dir / "week6_metadata.json",
        week6_dir / "week6_metadata.json",
    )
    week8_meta_path = resolve_input_path(
        "week8_metadata.json",
        data_dir / "week8_metadata.json",
        week8_dir / "week8_metadata.json",
    )
    week9_meta_path = resolve_input_path(
        "week9_metadata.json",
        data_dir / "week9_metadata.json",
        week9_dir / "week9_metadata.json",
    )

    for p in [param_path, nhpp_path, mgs_path, quarterly_path, sched_path, pareto_path]:
        if not Path(p).exists():
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)

    print("=" * 70)
    print("Week 10: Lookup Regression + Independent-Seed Robustness")
    print("=" * 70)
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Reps/config: {args.n_reps}")
    print(f"  Sim days:    {args.sim_days}")
    print(f"  sim_engine:  {sim_engine_path}")
    print()

    # Load data
    pareto_df = load_pareto_data(pareto_path)
    sched_df = pd.read_csv(sched_path)

    print(f"  Loaded pareto_activation_results: {len(pareto_df)} rows")
    print(f"  Loaded activation_schedules: {len(sched_df)} rows")

    # Staleness guards: the lookup side reads pareto_activation_results.csv
    # as-is, while the fresh SimPy side uses the current engine and upstream
    # artifacts. Warn whenever the lookup CSV predates key dependencies.
    upstream_for_pareto = [
        mgs_path,
        sched_path,
        sim_engine_path,
        str(CODE_DIR / "week9_analysis.py"),
    ]
    # week9_metadata.json is written after pareto_activation_results.csv in a
    # normal Week 9 run, so it is provenance, not an upstream freshness guard.
    for optional_path in [week6_fault_tax_path, week6_meta_path, week8_meta_path]:
        if Path(optional_path).exists():
            upstream_for_pareto.append(optional_path)
    warn_if_older(pareto_path, upstream_for_pareto,
                  "pareto_activation_results.csv")
    warn_if_older(sched_path, [mgs_path, week6_fault_tax_path,
                               str(CODE_DIR / "week9_analysis.py")],
                  "activation_schedules.csv")
    print()

    provenance = {
        "sim_engine": file_signature(Path(sim_engine_path)),
        "inputs": {
            "parameter_summary": file_signature(Path(param_path)),
            "nhpp_rate_functions": file_signature(Path(nhpp_path)),
            "mgs_comparison_4rep": file_signature(Path(mgs_path)),
            "nhpp_rate_functions_quarterly": file_signature(Path(quarterly_path)),
            "activation_schedules": file_signature(Path(sched_path)),
            "pareto_activation_results": file_signature(Path(pareto_path)),
        },
        "optional_upstream": {},
        "run_parameters": {
            "n_reps": args.n_reps,
            "sim_days": args.sim_days,
            "validation_seed": args.validation_seed,
            "skip_simpy": args.skip_simpy,
        },
    }
    for label, path in {
        "week6_fault_tax": week6_fault_tax_path,
        "week6_metadata": week6_meta_path,
        "week8_metadata": week8_meta_path,
        "week9_metadata": week9_meta_path,
    }.items():
        if Path(path).exists():
            provenance["optional_upstream"][label] = file_signature(Path(path))

    # ── Run validation ──
    results = []

    for (cfg_id, station, fleet, act_mode, faults, purpose) in VALIDATION_CONFIGS:
        print(f"Config {cfg_id:2d}: {station[:20]:20s}  s={fleet:2d}  "
              f"{act_mode:24s}  faults={faults:3s}")
        print(f"          Purpose: {purpose}")

        # Step 1: Simulator lookup
        lookup = simulator_lookup(pareto_df, station, act_mode, faults, fleet)
        if lookup is None:
            print(f"          LOOKUP: No data (unsupported configuration)")
            results.append({
                'config_id': cfg_id,
                'station': station,
                'fleet': fleet,
                'activation_mode': act_mode,
                'faults': faults,
                'purpose': purpose,
                'lookup_method': 'N/A',
                'lookup_util': np.nan,
                'lookup_pw15': np.nan,
                'simpy_util': np.nan,
                'simpy_pw15': np.nan,
                'util_rel_err': np.nan,
                'pw15_rel_err': np.nan,
                'pw15_abs_err': np.nan,
                'util_verdict': 'SKIP',
                'pw15_verdict': 'SKIP',
                'overall_verdict': 'SKIP',
                'simpy_reps_clean': 0,
                'simpy_reps_excluded': 0,
            })
            print()
            continue

        print(f"          LOOKUP: util={lookup['util_realized']:.5f}  "
              f"P(W>15)={lookup['p_wait_15']:.6f}  [{lookup['method']}]")

        # Step 2: SimPy run
        if args.skip_simpy:
            print(f"          SIMPY:  [skipped]")
            simpy_result = {
                'util_realized': np.nan, 'p_wait_15': np.nan,
                'mean_wait': np.nan, 'n_reps_clean': 0,
                'n_reps_excluded': 0, 'elapsed_sec': 0,
            }
        else:
            simpy_result = run_simpy_config(
                cfg_id, station, fleet, act_mode, faults,
                param_path, nhpp_path, mgs_path, quarterly_path, sched_df,
                n_reps=args.n_reps, sim_days=args.sim_days,
                validation_seed=args.validation_seed,
                verbose=args.verbose
            )
            print(f"          SIMPY:  util={simpy_result['util_realized']:.5f}  "
                  f"P(W>15)={simpy_result['p_wait_15']:.6f}  "
                  f"[{simpy_result['n_reps_clean']} clean, "
                  f"{simpy_result['n_reps_excluded']} excluded, "
                  f"{simpy_result['elapsed_sec']:.1f}s]")

        # Step 3: Compare
        if not args.skip_simpy and simpy_result['n_reps_clean'] > 0:
            util_rel, util_abs, util_rule, util_v = compute_errors(
                simpy_result['util_realized'], lookup['util_realized'])
            pw_rel, pw_abs, pw_rule, pw_v = compute_errors(
                simpy_result['p_wait_15'], lookup['p_wait_15'],
                is_probability=True)

            overall = "PASS" if util_v == "PASS" and pw_v == "PASS" else "FAIL"

            compare_msg = (
                f"          COMPARE: util_rel_err={util_rel:.4f} [{util_rule}; {util_v}]  "
                f"pw15_rel_err={pw_rel:.4f} pw15_abs_err={pw_abs:.4f} [{pw_rule}; {pw_v}]  "
                  f"-> {overall}")
        else:
            util_rel, util_abs, util_rule, util_v = np.nan, np.nan, "N/A", "SKIP"
            pw_rel, pw_abs, pw_rule, pw_v = np.nan, np.nan, "N/A", "SKIP"
            overall = "SKIP"


        if not args.skip_simpy and simpy_result['n_reps_clean'] > 0:
            print(compare_msg)

        results.append({
            'config_id': cfg_id,
            'station': station,
            'fleet': fleet,
            'activation_mode': act_mode,
            'faults': faults,
            'purpose': purpose,
            'lookup_method': lookup['method'],
            'lookup_util': lookup['util_realized'],
            'lookup_pw15': lookup['p_wait_15'],
            'lookup_mean_wait': lookup['mean_wait'],
            'simpy_util': simpy_result['util_realized'],
            'simpy_pw15': simpy_result['p_wait_15'],
            'simpy_mean_wait': simpy_result.get('mean_wait', np.nan),
            'util_rel_err': util_rel,
            'util_rule': util_rule,
            'pw15_rel_err': pw_rel,
            'pw15_abs_err': pw_abs,
            'pw15_rule': pw_rule,
            'util_verdict': util_v,
            'pw15_verdict': pw_v,
            'overall_verdict': overall,
            'simpy_reps_clean': simpy_result.get('n_reps_clean', 0),
            'simpy_reps_excluded': simpy_result.get('n_reps_excluded', 0),
            'simpy_elapsed_sec': simpy_result.get('elapsed_sec', 0),
        })
        print()

    # ── Save results ──
    results_df = pd.DataFrame(results)
    results_path = output_dir / "validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path} ({len(results_df)} rows)")

    provenance_path = output_dir / "validation_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(to_builtin(provenance), f, indent=2)
    print(f"Saved: {provenance_path}")

    # ── Summary ──
    n_pass = sum(1 for r in results if r['overall_verdict'] == 'PASS')
    n_fail = sum(1 for r in results if r['overall_verdict'] == 'FAIL')
    n_skip = sum(1 for r in results if r['overall_verdict'] == 'SKIP')

    if not args.skip_simpy:
        summary_lines = [
            "=" * 50,
            "LOOKUP REGRESSION / ROBUSTNESS SUMMARY",
            "=" * 50,
            f"Total configs:  {len(results)}",
            f"PASS:           {n_pass}",
            f"FAIL:           {n_fail}",
            f"SKIP:           {n_skip}",
            "",
            "Criteria:",
            f"  Utilization: relative error <{REL_ERROR_THRESHOLD*100:.0f}%",
            f"  P(W>15): relative error <{REL_ERROR_THRESHOLD*100:.0f}% "
            f"when >= {NEAR_ZERO_THRESHOLD*100:.0f}%",
            f"  Near-zero P(W>15): absolute error "
            f"<{ABS_ERROR_THRESHOLD*100:.1f} pp AND relative error "
            f"<{NEAR_ZERO_REL_THRESHOLD*100:.0f}%",
            "",
        ]

        if n_fail > 0:
            summary_lines.append("FAILED CONFIGS:")
            for r in results:
                if r['overall_verdict'] == 'FAIL':
                    summary_lines.append(
                        f"  Config {r['config_id']}: {r['station'][:20]} "
                        f"s={r['fleet']} {r['activation_mode']} "
                        f"faults={r['faults']}")
                    summary_lines.append(
                        f"    util: lookup={r['lookup_util']:.5f} "
                        f"simpy={r['simpy_util']:.5f} "
                        f"rel_err={r['util_rel_err']:.4f} "
                        f"rule={r['util_rule']} [{r['util_verdict']}]")
                    summary_lines.append(
                        f"    pw15: lookup={r['lookup_pw15']:.6f} "
                        f"simpy={r['simpy_pw15']:.6f} "
                        f"rel_err={r['pw15_rel_err']:.4f} "
                        f"abs_err={r['pw15_abs_err']:.4f} "
                        f"rule={r['pw15_rule']} [{r['pw15_verdict']}]")
            summary_lines.append("")

        total_time = sum(r.get('simpy_elapsed_sec', 0) for r in results)
        summary_lines.append(f"Total SimPy runtime: {total_time:.0f}s "
                             f"({total_time/60:.1f} min)")
        summary_lines.append("")
        summary_lines.append(
            f"Overall: {'PASS' if n_fail == 0 and n_skip == 0 else 'REVIEW NEEDED'}")
    else:
        summary_lines = [
            "=" * 50,
            "VALIDATION SUMMARY (LOOKUP ONLY)",
            "=" * 50,
            f"Total configs:  {len(results)}",
            f"Lookup OK:      {len(results) - n_skip}",
            f"Lookup missing: {n_skip}",
            "",
            "SimPy runs were skipped (--skip-simpy).",
            "No pass/fail verdicts were computed.",
        ]

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    summary_path = output_dir / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
