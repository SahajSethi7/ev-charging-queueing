"""
Week 7: Phase 2c Wrap-Up + Phase 3 Flexibility Analysis
=========================================================
Components:
  7A. Phase 2 narrative synthesis
      - M/M/s underestimation quantification (from existing CSVs)
      - Fault capacity penalty summary
      - ML fleet sizing impact assessment (minor new aggregation)
  7B. Jiaxing flexibility proxy construction
      - Charger-type benchmark power (P90 of non-fault observed power)
      - utilization_ratio = realized_avg_power / benchmark_power
      - Fixed-threshold flexibility tiers: likely / possibly / unlikely flexible
      - Stratification by hour, station, charger type
  7C. ACN cross-validation
      - Explicit slack from ACN doneChargingTime / disconnectTime
      - Implied slack using nominal power benchmark
      - Distribution comparison (plausibility reference, NOT a fallback)

Assumptions:
  - Flexibility proxy uses charger-type P90 observed power as the
    benchmark denominator (per-post rated power not available).
  - Fixed thresholds: utilization_ratio < 0.5 → likely_flexible,
    0.5–0.8 → possibly_flexible, ≥ 0.8 → unlikely_flexible.
  - 'user_stop_proxy' column from patched week1_wrapup.py determines
    which sessions are candidates; inflexible/fault sessions retain
    their original tier regardless of utilization_ratio.
  - ACN is a plausibility reference. If distributions differ substantially
    from Jiaxing, retain the Jiaxing proxy with sensitivity at
    25/50/75/100% of the proxy-flexible fraction.

Input files:
  - jiaxing_clean.parquet    (441k sessions, local)
  - acn_clean.parquet        (ACN JPL, ~11k sessions)
  - parameter_summary.json   (service time fits, fleet sizing)
  - mgs_comparison_4rep.csv  (M/M/s vs M/G/s fleet gap)
  - analytical_gap.csv       (simulated P(W>15) at analytical s*)
  - fault_tax_results.csv    (fault capacity penalty)
  - xgboost_results.json     (XGBoost metrics, Week 4/5 artifacts)
  - lstm_comparison.json     (LSTM negative result, Week 5 artifact)
  - nhpp_rate_functions.csv  (NHPP rates for peak lambda)

Output files:
  - phase2_conclusions.md
  - flexibility_analysis.csv
  - flexibility_summary.json
  - acn_flexibility.csv
  - flexibility_crossvalidation.json
  - figures/flexibility_*.png
  - week7_metadata.json

Usage:
  python week7_analysis.py --data-dir ./data --output-dir ./week7_results
                           [--week3-dir ./week3_results]
                           [--week4-dir ./week4_results]
                           [--week5-dir ./week5_results]
                           [--week6-dir ./week6_results]
                           [--skip-7a] [--skip-7b] [--skip-7c]

Date: Week 7, Mar 2026
"""

import argparse
import json
import os
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR, to_builtin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


# =====================================================================
# CONSTANTS
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

# s* from Week 6 faults-ON Pareto frontier (P(W>15min) < 5%)
S_STAR_FAULTS_ON = {
    'Xiuzhou_Expressway Service District A': 4,
    'Nanhu_Technology Park': 6,
    'Xiuzhou_Government Agency': 8,
    'Tongxiang_Bus Station': 10,
}

# Flexibility tier thresholds (utilization_ratio = realized_power / benchmark_power)
FLEX_THRESHOLD_LIKELY = 0.5     # < 0.5 → likely_flexible
FLEX_THRESHOLD_POSSIBLY = 0.8   # 0.5–0.8 → possibly_flexible, ≥ 0.8 → unlikely_flexible

# Max shift slots (30-min slots) per flexibility tier
SHIFT_SLOTS = {
    'likely_flexible': 4,       # ±2 hours (one-sided: 0 to +2hr)
    'possibly_flexible': 2,     # ±1 hour (one-sided: 0 to +1hr)
    'unlikely_flexible': 0,
    'inflexible': 0,
    'fault': -1,                # excluded from scheduling
}

# Minimum sample size for benchmark power computation
MIN_BENCHMARK_SAMPLE = 100


# =====================================================================
# CONSOLE ENCODING SAFETY (Windows)
# =====================================================================

def configure_console_output():
    """Ensure console output doesn't crash on Windows with special characters."""
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass


def detect_parquet_engine() -> Optional[str]:
    """Return an available parquet engine, or None if neither is installed."""
    if find_spec('pyarrow') is not None:
        return 'pyarrow'
    if find_spec('fastparquet') is not None:
        return 'fastparquet'
    return None


def read_table(data_dir: Path, stem: str,
               parse_dates: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Path]:
    """
    Read a table using parquet first, then CSV fallback.
    Mirrors the resilient loading pattern used in earlier weeks.
    """
    parquet_path = data_dir / f'{stem}.parquet'
    csv_path = data_dir / f'{stem}.csv'
    engine = detect_parquet_engine()

    if parquet_path.exists() and engine is not None:
        try:
            return pd.read_parquet(parquet_path, engine=engine), parquet_path
        except Exception as exc:
            print(f"  WARNING: Failed reading {parquet_path}: {exc}")
            if not csv_path.exists():
                raise
    elif parquet_path.exists() and engine is None:
        print(f"  WARNING: No parquet engine installed; looking for CSV fallback for {stem}.")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for col in parse_dates or []:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df, csv_path

    tried = [str(parquet_path), str(csv_path)]
    raise FileNotFoundError(f"{stem} not found. Tried: {', '.join(tried)}")


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name that exists in the dataframe."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_existing_path(*candidates: Path) -> Optional[Path]:
    """Return the first existing path among the candidates."""
    for path in candidates:
        if path.exists():
            return path
    return None


# =====================================================================
# DATA LOADING
# =====================================================================

def load_jiaxing(data_dir: str) -> pd.DataFrame:
    """Load the main Jiaxing session-level parquet."""
    data_path = Path(data_dir)
    df, src_path = read_table(data_path, 'jiaxing_clean',
                              parse_dates=['start_time', 'end_time'])
    print(f"  Loading {src_path} ...")

    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")

    if 'date_dt' not in df.columns:
        date_col = first_existing(df, ['date', 'session_date'])
        if date_col is not None:
            df['date_dt'] = pd.to_datetime(df[date_col], errors='coerce')
        elif 'start_time' in df.columns:
            df['date_dt'] = pd.to_datetime(df['start_time'], errors='coerce').dt.normalize()

    if 'hour_of_day' not in df.columns and 'start_time' in df.columns:
        df['hour_of_day'] = pd.to_datetime(df['start_time'], errors='coerce').dt.hour

    if 'charging_duration_min' not in df.columns:
        duration_col = first_existing(df, ['duration_min', 'service_time_min'])
        if duration_col is not None:
            df['charging_duration_min'] = pd.to_numeric(df[duration_col], errors='coerce')
        elif 'start_time' in df.columns and 'end_time' in df.columns:
            start = pd.to_datetime(df['start_time'], errors='coerce')
            end = pd.to_datetime(df['end_time'], errors='coerce')
            df['charging_duration_min'] = (end - start).dt.total_seconds() / 60.0

    if 'energy_kwh' not in df.columns:
        energy_col = first_existing(df, ['kWhDelivered', 'energy'])
        if energy_col is not None:
            df['energy_kwh'] = pd.to_numeric(df[energy_col], errors='coerce')

    if 'effective_rate_kw' not in df.columns and {
        'energy_kwh', 'charging_duration_min'
    }.issubset(df.columns):
        duration_hr = pd.to_numeric(df['charging_duration_min'], errors='coerce') / 60.0
        df['effective_rate_kw'] = pd.to_numeric(df['energy_kwh'], errors='coerce') / duration_hr.replace(0, np.nan)

    if 'flag_zero_energy' not in df.columns and 'energy_kwh' in df.columns:
        df['flag_zero_energy'] = (
            pd.to_numeric(df['energy_kwh'], errors='coerce').fillna(0) <= 0
        ).astype(int)

    if 'is_abnormal' not in df.columns:
        abnormal_col = first_existing(df, ['is_fault', 'fault'])
        if abnormal_col is not None:
            df['is_abnormal'] = pd.to_numeric(df[abnormal_col], errors='coerce').fillna(0).astype(int)
        else:
            df['is_abnormal'] = 0

    # Normalize the flexibility column name
    # Patched week1_wrapup produces 'user_stop_proxy';
    # older versions may produce 'flexibility_tier'.
    if 'user_stop_proxy' in df.columns:
        df['flex_raw'] = df['user_stop_proxy']
        print(f"  Using 'user_stop_proxy' as raw flexibility column.")
    elif 'flexibility_tier' in df.columns:
        # Map old names to new: Flexible→user_stop_proxy, Inflexible→inflexible, Fault→fault
        mapping = {'Flexible': 'user_stop_proxy', 'Inflexible': 'inflexible', 'Fault': 'fault'}
        df['flex_raw'] = df['flexibility_tier'].map(mapping).fillna(df['flexibility_tier'])
        print(f"  Mapped 'flexibility_tier' → 'flex_raw' using legacy mapping.")
    else:
        print(f"  WARNING: No flexibility column found. Defaulting all to 'inflexible'.")
        df['flex_raw'] = 'inflexible'

    return df


def load_acn(data_dir: str) -> Optional[pd.DataFrame]:
    """Load ACN-Data JPL parquet."""
    data_path = Path(data_dir)
    acn_path = find_existing_path(data_path / 'acn_clean.parquet', data_path / 'acn_clean.csv')
    if acn_path is None:
        print(f"  WARNING: acn_clean not found in {data_path}. ACN cross-validation will be skipped.")
        return None

    if acn_path.suffix.lower() == '.parquet':
        engine = detect_parquet_engine()
        if engine is None:
            print("  WARNING: No parquet engine available for acn_clean.parquet.")
            return None
        df = pd.read_parquet(acn_path, engine=engine)
    else:
        df = pd.read_csv(acn_path)

    print(f"  Loading {acn_path} ...")
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")

    for col in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'duration_min' not in df.columns and {'connectionTime', 'disconnectTime'}.issubset(df.columns):
        df['duration_min'] = (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 60.0

    if 'hour_of_day' not in df.columns and 'connectionTime' in df.columns:
        df['hour_of_day'] = df['connectionTime'].dt.hour

    return df


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV, return None if missing."""
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found.")
    return None


# =====================================================================
# 7A: PHASE 2 NARRATIVE SYNTHESIS
# =====================================================================

def run_phase2_synthesis(week3_dir: Path, week4_dir: Path, week5_dir: Path, week6_dir: Path,
                         output_dir: Path) -> dict:
    """
    Assemble the three Phase 2 quantitative claims from existing outputs.

    Claims:
      1. M/M/s underestimation: simulated P(W>15) at M/M/s s* vs 5% target
      2. Fault capacity penalty: fault tax per station
      3. Demand growth impact on fleet sizing: does accounting for recent
         demand (via the recent_quarter scenario) change s*?
         Note: this is a scenario comparison, not an ML-vs-analytical test.
         XGBoost's value is capturing demand growth via lag features;
         the recent_quarter scenario reflects the same growth effect.

    Returns a summary dict for metadata.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 7A: PHASE 2 NARRATIVE SYNTHESIS")
    print("=" * 70)

    lines = []
    lines.append("# Phase 2 Conclusions\n")
    lines.append("*Quantitative claims assembled from Weeks 4–6 outputs.*\n")
    summary = {}

    # ── Claim 1: M/M/s underestimation ──────────────────────────────
    lines.append("\n## Claim 1: M/M/s and M/G/s Recommendations Are Close, but a Direct Simulation Gap Is Not Evaluated\n")

    mgs_path = week4_dir / 'mgs_comparison_4rep.csv'
    gap_path = week6_dir / 'analytical_gap.csv'

    mgs_df = load_csv_safe(mgs_path)
    gap_df = load_csv_safe(gap_path)

    claim1_data = []
    if mgs_df is not None and gap_df is not None:
        # Week 4 provides analytical M/M/s and M/G/s recommendations.
        # Week 6 provides a step-2 simulation sweep. If s_star_mms is not
        # on that grid, bracket it with lower / upper sweep points rather
        # than snapping downward to the nearest available point.
        for station in REPRESENTATIVE_STATIONS:
            mgs_row = mgs_df[
                (mgs_df['station'] == station) &
                (mgs_df['sizing_scenario'] == 'historical_avg')
            ]
            if mgs_row.empty:
                continue
            mgs_row = mgs_row.iloc[0]
            s_mms = int(mgs_row['s_star_mms_5pct'])
            s_mgs = int(mgs_row['s_star_mgs_5pct'])
            s_gap = int(mgs_row['s_gap_5pct'])

            # Week 6 intentionally stores only a metric-safety status here.
            # Its full-day NHPP simulation metric, P(W>15 min), cannot be
            # subtracted from Week 4's peak-hour analytical P(W>0) reference.
            # Do not revive the obsolete row-by-charger comparison below.
            status_row = gap_df[gap_df['station'] == station]
            comparison_status = (
                status_row.iloc[0].get('comparison_status', 'not_evaluated')
                if not status_row.empty else 'not_evaluated'
            )
            label = STATION_LABELS.get(station, station)
            claim1_data.append({
                'station': station,
                'label': label,
                's_mms': s_mms,
                's_mgs': s_mgs,
                's_gap': s_gap,
                'direct_simulation_gap': 'not_evaluated_metric_mismatch',
                'week6_comparison_status': comparison_status,
            })
            lines.append(
                f"- **{label}**: Week 4 recommends s*={s_mms} under M/M/s and "
                f"s*={s_mgs} under the M/G/s approximation (gap={s_gap}). "
                "Week 6 does not compute a direct analytical-versus-simulation "
                "gap because the available metrics differ (peak-hour P(W>0) "
                "versus full-day P(W>15min)).\n"
            )
            continue

            station_gap = gap_df[gap_df['station'] == station].sort_values('n_chargers').copy()
            if station_gap.empty:
                continue

            exact_row = station_gap[station_gap['n_chargers'] == s_mms]
            lower_row = station_gap[station_gap['n_chargers'] <= s_mms].tail(1)
            upper_row = station_gap[station_gap['n_chargers'] >= s_mms].head(1)

            label = STATION_LABELS.get(station, station)
            claim_row = {
                'station': station,
                'label': label,
                's_mms': s_mms,
                's_mgs': s_mgs,
                's_gap': s_gap,
                'interpretation': None,
            }

            if not exact_row.empty:
                sim_pw15 = float(exact_row.iloc[0]['sim_p_wait_15'])
                interpretation = (
                    'mms_underestimates' if sim_pw15 > 0.05 else 'mms_adequate_or_conservative'
                )
                claim_row.update({
                    'sim_pw15_at_s_mms': sim_pw15,
                    'interpretation': interpretation,
                })
                lines.append(
                    f"- **{label}**: M/M/s recommends s*={s_mms}; M/G/s recommends s*={s_mgs} "
                    f"(gap={s_gap}). Simulated P(W>15min) at s={s_mms} is {sim_pw15:.1%} "
                    f"vs 5% target, so the M/M/s recommendation is "
                    f"{'too small' if sim_pw15 > 0.05 else 'adequate/conservative'} "
                    f"under historical-average demand.\n"
                )
                claim1_data.append(claim_row)
                continue

            lower_s = int(lower_row.iloc[0]['n_chargers']) if not lower_row.empty else None
            lower_pw15 = float(lower_row.iloc[0]['sim_p_wait_15']) if not lower_row.empty else None
            upper_s = int(upper_row.iloc[0]['n_chargers']) if not upper_row.empty else None
            upper_pw15 = float(upper_row.iloc[0]['sim_p_wait_15']) if not upper_row.empty else None

            claim_row.update({
                'lower_sweep_s': lower_s,
                'lower_sweep_sim_pw15': lower_pw15,
                'upper_sweep_s': upper_s,
                'upper_sweep_sim_pw15': upper_pw15,
            })

            if lower_pw15 is not None and lower_pw15 <= 0.05:
                interpretation = 'mms_conservative_on_coarse_grid'
                lines.append(
                    f"- **{label}**: M/M/s recommends s*={s_mms}; M/G/s recommends s*={s_mgs} "
                    f"(gap={s_gap}). The step-2 simulation grid does not include s={s_mms}, "
                    f"but even the nearest lower sweep point already meets the target: "
                    f"s={lower_s} gives P(W>15min)={lower_pw15:.1%}"
                    + (
                        f", while the upper point s={upper_s} gives {upper_pw15:.1%}."
                        if upper_pw15 is not None and upper_s != lower_s else "."
                    )
                    + " This indicates the M/M/s recommendation is conservative on the coarse grid.\n"
                )
            elif lower_pw15 is not None and upper_pw15 is not None and lower_pw15 > 0.05 >= upper_pw15:
                interpretation = 'coarse_grid_brackets_target'
                lines.append(
                    f"- **{label}**: M/M/s recommends s*={s_mms}; M/G/s recommends s*={s_mgs} "
                    f"(gap={s_gap}). The step-2 simulation grid brackets the target around "
                    f"the M/M/s recommendation: s={lower_s} gives {lower_pw15:.1%}, while "
                    f"s={upper_s} gives {upper_pw15:.1%}. Exact adequacy at s={s_mms} is not "
                    f"identified from the coarse sweep alone.\n"
                )
            else:
                interpretation = 'insufficient_simulation_support'
                lines.append(
                    f"- **{label}**: M/M/s recommends s*={s_mms}; M/G/s recommends s*={s_mgs} "
                    f"(gap={s_gap}). Available simulation sweep points are insufficient to "
                    f"evaluate the exact historical-demand performance at s={s_mms}.\n"
                )

            claim_row['interpretation'] = interpretation
            claim1_data.append(claim_row)
    else:
        lines.append("- *Data files missing. Cannot compute Claim 1.*\n")

    if claim1_data:
        lines.append(
            "\nThe Week 4 M/G/s approximation changes the historical-average recommendation "
            "by only 0-1 chargers across these stations. A direct comparison against the "
            "Week 6 simulation is intentionally not reported because their probability metrics "
            "are not the same.\n"
        )

    summary['claim1_mms_underestimation'] = claim1_data

    # ── Claim 2: Fault capacity penalty ─────────────────────────────
    lines.append("\n## Claim 2: Fault Capacity Penalty\n")

    ft_path = week6_dir / 'fault_tax_results.csv'
    ft_df = load_csv_safe(ft_path)

    claim2_data = []
    if ft_df is not None:
        for _, row in ft_df.iterrows():
            label = row.get('label', row['station'])
            tax = int(row['fault_tax'])
            s_on = int(row['s_star_faults_on'])
            s_off = int(row['s_star_faults_off'])
            frac = row.get('empirical_fault_fraction', np.nan)
            claim2_data.append({
                'station': row['station'], 'label': label,
                'fault_tax': tax, 's_on': s_on, 's_off': s_off,
                'fault_fraction': float(frac),
            })
            lines.append(
                f"- **{label}**: s*(faults ON)={s_on}, s*(faults OFF)={s_off}, "
                f"fault tax=+{tax} chargers. "
                f"Empirical fault fraction: {frac:.1%}.\n"
            )
        # Sensitivity note
        lines.append(
            "\nSensitivity (Gov Agency, L2-heavy): s* steps from 6 to 8 "
            "at ~15% fault rate. Below 10% fault rate, the fault tax disappears "
            "at step-2 grid resolution.\n"
        )
    else:
        lines.append("- *fault_tax_results.csv missing.*\n")

    summary['claim2_fault_penalty'] = claim2_data

    # ── Claim 3: Demand growth impact on fleet sizing ─────────────────
    lines.append("\n## Claim 3: Demand Growth Impact on Fleet Sizing\n")

    xgb_path = find_existing_path(
        week4_dir / 'xgboost_results.json',
        week5_dir / 'xgboost_results.json',
    )
    lstm_path = find_existing_path(
        week5_dir / 'lstm_comparison.json',
        week4_dir / 'lstm_comparison.json',
    )
    param_path = week4_dir / 'parameter_summary.json'

    # NHPP rates are a Week 3 artifact; search week3_dir first, then week4_dir
    nhpp_path = None
    for candidate in [week3_dir / 'nhpp_rate_functions.csv',
                      week4_dir / 'nhpp_rate_functions.csv',
                      week3_dir / 'nhpp_rate_functions_quarterly.csv']:
        if candidate.exists():
            nhpp_path = candidate
            break

    claim3 = {'lstm_negative_result': True, 'demand_growth_changes_fleet': False}

    # LSTM negative result
    if lstm_path is not None:
        lstm = load_json(lstm_path)
        xgb_mae = lstm['models']['xgboost']['mae']
        lstm_mae = lstm['models']['lstm_h1']['mae']
        pct = lstm['lstm_vs_xgboost']['lstm_improvement_pct']
        lines.append(
            f"LSTM vs XGBoost (horizon-1, Sep-Dec 2021): "
            f"XGBoost MAE={xgb_mae:.3f}, LSTM MAE={lstm_mae:.3f} "
            f"({pct:+.1f}%). LSTM does not outperform XGBoost. "
            f"XGBoost is the adopted forecasting model.\n\n"
        )
    elif xgb_path is not None:
        xgb = load_json(xgb_path)
        lines.append(
            f"XGBoost test MAE={xgb['models']['xgboost']['mae']:.3f}. "
            f"LSTM comparison file not found.\n\n"
        )

    # Fleet sizing impact: compare historical_avg vs recent_quarter
    # scenario peak lambdas from parameter_summary.json.
    # This is a demand-growth scenario comparison, not an ML-vs-NHPP test.
    # It quantifies the sizing implication of recent observed demand only;
    # forecasting-model MAE does not by itself establish a fleet-sizing benefit.
    if param_path.exists():
        params = load_json(param_path)

        lines.append(
            "### Peak lambda comparison: historical average vs recent quarter\n\n"
            "The recent-quarter scenario uses NHPP rates from the final quarter "
            "of the dataset, reflecting observed demand growth. If s* changes "
            "between scenarios, that is a demand-trend effect; no fleet-sizing "
            "effect is inferred from forecasting-model MAE alone.\n\n"
        )

        for station in REPRESENTATIVE_STATIONS:
            key_hist = f"{station}__historical_avg"
            if key_hist not in params.get('fleet_sizing', {}):
                continue
            hist_peak = params['fleet_sizing'][key_hist]['peak_lambda']
            s_mms_hist = params['fleet_sizing'][key_hist]['s_star_mms']

            key_recent = f"{station}__recent_quarter"
            if key_recent in params.get('fleet_sizing', {}):
                recent_peak = params['fleet_sizing'][key_recent]['peak_lambda']
                s_mms_recent = params['fleet_sizing'][key_recent]['s_star_mms']
            else:
                recent_peak = hist_peak
                s_mms_recent = s_mms_hist

            label = STATION_LABELS.get(station, station)
            fleet_changes = s_mms_recent != s_mms_hist
            if fleet_changes:
                claim3['demand_growth_changes_fleet'] = True
            lines.append(
                f"- **{label}**: historical peak lambda={hist_peak:.2f} "
                f"(s*={s_mms_hist}), recent-quarter peak={recent_peak:.2f} "
                f"(s*={s_mms_recent}). "
                f"{'Fleet size CHANGES.' if fleet_changes else 'Fleet size unchanged.'}\n"
            )

        if claim3['demand_growth_changes_fleet']:
            lines.append(
                "\nAccounting for recent demand growth changes the fleet "
                "recommendation at some stations. This is a demand trend "
                "effect, not a benefit specific to ML over growth-adjusted NHPP.\n"
            )
        else:
            lines.append(
                "\nFleet recommendation does not change between historical-average "
                "and recent-quarter scenarios for any representative station "
                "under M/M/s sizing.\n"
            )
    else:
        lines.append("- *parameter_summary.json missing. Cannot compute Claim 3.*\n")

    summary['claim3_demand_growth_impact'] = claim3

    # ── Save ────────────────────────────────────────────────────────
    md_path = output_dir / 'phase2_conclusions.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {md_path}")

    return summary


# =====================================================================
# 7B: JIAXING FLEXIBILITY PROXY
# =====================================================================

def compute_benchmark_power(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, int]]:
    """
    Compute charger-type benchmark power using the nominal power hierarchy:
      Level 1: per-post rated power (NOT AVAILABLE in this dataset)
      Level 2: charger-type P90 of observed average power (non-fault, non-zero-energy)
      Level 3: hardcoded fallback (DC_Fast=60, Level_2=7, Mixed=22)

    Returns:
      benchmark_kw:  {charger_type: power_kw}
      benchmark_src: {charger_type: source_description}
      benchmark_n:   {charger_type: sample_size}
    """
    print("\n  Computing benchmark power per charger type ...")

    benchmark_kw = {}
    benchmark_src = {}
    benchmark_n = {}

    # Filter: non-fault, non-zero-energy, positive effective rate
    valid = df[
        (df['is_abnormal'] == 0) &
        (df['flag_zero_energy'] == 0) &
        (df['effective_rate_kw'] > 0) &
        (df['effective_rate_kw'].notna())
    ].copy()

    # Charger type was itself derived from effective rate in Week 1, so using
    # type-specific rate quantiles here would be circular. Use station-level
    # P90 benchmarks and a global-data fallback instead.
    global_p90 = (float(valid['effective_rate_kw'].quantile(0.90))
                  if len(valid) else 22.0)
    for station, station_rows in df.groupby('station_name'):
        subset = valid.loc[
            valid['station_name'] == station, 'effective_rate_kw']
        if len(subset) >= MIN_BENCHMARK_SAMPLE:
            value = float(subset.quantile(0.90))
            source = f'station_P90 (n={len(subset):,})'
        else:
            value = global_p90
            source = (f'global_P90_fallback '
                      f'(station n={len(subset)} < {MIN_BENCHMARK_SAMPLE})')
        benchmark_kw[str(station)] = value
        benchmark_src[str(station)] = source
        benchmark_n[str(station)] = int(len(subset))
        print(f"    {station}: {value:.2f} kW [{source}]")

    return benchmark_kw, benchmark_src, benchmark_n


def assign_flexibility_tiers(df: pd.DataFrame,
                              benchmark_kw: Dict[str, float],
                              stations: List[str]) -> pd.DataFrame:
    """
    Assign flexibility tiers to sessions at the representative stations.

    For sessions with flex_raw == 'user_stop_proxy':
      - Compute utilization_ratio = realized_avg_power / benchmark_power
      - Apply fixed thresholds to sub-tier
    For sessions with flex_raw == 'inflexible' or 'fault':
      - Retain original classification

    Returns a DataFrame with per-session tier assignments.
    """
    print("\n  Assigning flexibility tiers ...")

    required_cols = ['station_name', 'charging_duration_min', 'energy_kwh', 'charger_type', 'flex_raw']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required Jiaxing columns for flexibility analysis: {missing}")

    # Filter to representative stations
    mask = df['station_name'].isin(stations)
    sub = df[mask].copy()
    print(f"    Sessions at representative stations: {len(sub):,}")

    # Compute realized average power (kW)
    duration_hr = sub['charging_duration_min'] / 60.0
    sub['realized_avg_power_kw'] = np.where(
        duration_hr > 0,
        sub['energy_kwh'] / duration_hr,
        np.nan
    )

    # Map benchmark power
    sub['benchmark_power_kw'] = sub['station_name'].map(benchmark_kw)

    # Compute utilization ratio
    sub['utilization_ratio'] = np.where(
        (sub['benchmark_power_kw'] > 0) & (sub['realized_avg_power_kw'].notna()),
        sub['realized_avg_power_kw'] / sub['benchmark_power_kw'],
        np.nan
    )

    # Assign tiers
    def _assign_tier(row):
        raw = row['flex_raw']

        if raw == 'fault':
            return 'fault'
        if raw == 'inflexible':
            return 'inflexible'

        # raw == 'user_stop_proxy': sub-tier by utilization ratio
        ur = row['utilization_ratio']
        if pd.isna(ur):
            # Cannot compute ratio (zero duration, missing energy, etc.)
            # Conservative: treat as unlikely_flexible
            return 'unlikely_flexible'
        if ur < FLEX_THRESHOLD_LIKELY:
            return 'likely_flexible'
        if ur < FLEX_THRESHOLD_POSSIBLY:
            return 'possibly_flexible'
        return 'unlikely_flexible'

    sub['flexibility_tier'] = sub.apply(_assign_tier, axis=1)
    sub['max_shift_slots'] = sub['flexibility_tier'].map(SHIFT_SLOTS)

    # Log distribution
    tier_counts = sub['flexibility_tier'].value_counts()
    N = len(sub)
    print(f"\n    Flexibility tier distribution ({N:,} sessions):")
    for tier in ['likely_flexible', 'possibly_flexible', 'unlikely_flexible',
                 'inflexible', 'fault']:
        count = tier_counts.get(tier, 0)
        print(f"      {tier:22s}  {count:>8,}  ({100*count/N:.1f}%)")

    return sub


def build_flexibility_output(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Build the flexibility_analysis.csv output with the locked schema.
    """
    # Build a session_id if not present
    if 'session_id' not in sub.columns:
        # Use index or construct from station + start_time
        sub = sub.copy()
        sub['session_id'] = sub.index.astype(str)

    cols = [
        'session_id', 'station_name', 'date_dt', 'hour_of_day',
        'charger_type', 'flex_raw', 'benchmark_power_kw',
        'utilization_ratio', 'flexibility_tier', 'max_shift_slots',
    ]
    # Only include columns that exist
    out_cols = [c for c in cols if c in sub.columns]
    return sub[out_cols].copy()


def build_flexibility_summary(sub: pd.DataFrame,
                               benchmark_kw: Dict[str, float],
                               benchmark_src: Dict[str, str],
                               benchmark_n: Dict[str, int]) -> dict:
    """
    Build flexibility_summary.json with the locked schema.
    """
    tier_counts = sub['flexibility_tier'].value_counts().to_dict()
    N = len(sub)
    tier_fractions = {k: v / N for k, v in tier_counts.items()}

    # Utilization ratio stats for the user_stop_proxy population
    proxy_mask = sub['flex_raw'] == 'user_stop_proxy'
    ur = sub.loc[proxy_mask, 'utilization_ratio'].dropna()

    return {
        'benchmark_power_source': benchmark_src,
        'benchmark_power_kw': {k: round(v, 2) for k, v in benchmark_kw.items()},
        'benchmark_sample_sizes': benchmark_n,
        'thresholds': {
            'likely_flexible': FLEX_THRESHOLD_LIKELY,
            'possibly_flexible': FLEX_THRESHOLD_POSSIBLY,
        },
        'tier_counts': {k: int(v) for k, v in tier_counts.items()},
        'tier_fractions': {k: round(v, 4) for k, v in tier_fractions.items()},
        'shift_slots': SHIFT_SLOTS,
        'stations_analyzed': sorted(sub['station_name'].unique().tolist()),
        'utilization_ratio_stats': {
            'n': int(len(ur)),
            'mean': round(float(ur.mean()), 4) if len(ur) > 0 else None,
            'median': round(float(ur.median()), 4) if len(ur) > 0 else None,
            'p10': round(float(ur.quantile(0.10)), 4) if len(ur) > 0 else None,
            'p25': round(float(ur.quantile(0.25)), 4) if len(ur) > 0 else None,
            'p75': round(float(ur.quantile(0.75)), 4) if len(ur) > 0 else None,
            'p90': round(float(ur.quantile(0.90)), 4) if len(ur) > 0 else None,
        },
        'note': (
            f"utilization_ratio < {FLEX_THRESHOLD_LIKELY} -> likely_flexible; "
            f"{FLEX_THRESHOLD_LIKELY}-{FLEX_THRESHOLD_POSSIBLY} -> possibly_flexible; "
            f">= {FLEX_THRESHOLD_POSSIBLY} -> unlikely_flexible (treated as inflexible "
            f"for scheduling). Only sessions with flex_raw='user_stop_proxy' are sub-tiered; "
            f"inflexible and fault sessions retain their original classification."
        ),
    }


def plot_flexibility(sub: pd.DataFrame, output_dir: Path):
    """
    Generate flexibility analysis figures.

    Fig 1: utilization_ratio distribution (CDF) for user_stop_proxy sessions
    Fig 2: flexibility tier composition by hour of day
    Fig 3: flexibility tier by station and charger type
    """
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # ── Fig 1: Utilization ratio CDF ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    proxy_mask = sub['flex_raw'] == 'user_stop_proxy'
    ur = sub.loc[proxy_mask, 'utilization_ratio'].dropna()

    if len(ur) > 0:
        # Histogram
        ax = axes[0]
        ax.hist(ur.clip(0, 2), bins=80, density=True, alpha=0.7,
                color='#2166ac', edgecolor='none')
        ax.axvline(FLEX_THRESHOLD_LIKELY, color='#e41a1c', ls='--', lw=1.5,
                   label=f'Likely flex (<{FLEX_THRESHOLD_LIKELY})')
        ax.axvline(FLEX_THRESHOLD_POSSIBLY, color='#ff7f00', ls='--', lw=1.5,
                   label=f'Possibly flex (<{FLEX_THRESHOLD_POSSIBLY})')
        ax.set_xlabel('Utilization Ratio (realized power / benchmark power)')
        ax.set_ylabel('Density')
        ax.set_title('Utilization Ratio Distribution\n(user_stop_proxy sessions)')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 2.0)

        # CDF
        ax = axes[1]
        sorted_ur = np.sort(ur.values)
        cdf = np.arange(1, len(sorted_ur) + 1) / len(sorted_ur)
        ax.plot(sorted_ur, cdf, color='#2166ac', lw=1.5)
        ax.axvline(FLEX_THRESHOLD_LIKELY, color='#e41a1c', ls='--', lw=1.5)
        ax.axvline(FLEX_THRESHOLD_POSSIBLY, color='#ff7f00', ls='--', lw=1.5)
        frac_likely = (ur < FLEX_THRESHOLD_LIKELY).mean()
        frac_poss = ((ur >= FLEX_THRESHOLD_LIKELY) & (ur < FLEX_THRESHOLD_POSSIBLY)).mean()
        ax.annotate(f'Likely: {frac_likely:.1%}', xy=(FLEX_THRESHOLD_LIKELY / 2, 0.5),
                    fontsize=9, ha='center', color='#e41a1c')
        ax.annotate(f'Possibly: {frac_poss:.1%}',
                    xy=((FLEX_THRESHOLD_LIKELY + FLEX_THRESHOLD_POSSIBLY) / 2, 0.7),
                    fontsize=9, ha='center', color='#ff7f00')
        ax.set_xlabel('Utilization Ratio')
        ax.set_ylabel('CDF')
        ax.set_title('CDF of Utilization Ratio')
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(fig_dir / 'flexibility_utilization_ratio.png')
    plt.close(fig)
    print(f"    Saved: flexibility_utilization_ratio.png")

    # ── Fig 2: Tier composition by hour ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tier_order = ['likely_flexible', 'possibly_flexible', 'unlikely_flexible',
                  'inflexible', 'fault']
    tier_colors = {
        'likely_flexible': '#2ca02c',
        'possibly_flexible': '#98df8a',
        'unlikely_flexible': '#ffbb78',
        'inflexible': '#aec7e8',
        'fault': '#d62728',
    }

    # Stacked area by hour
    ax = axes[0]
    hourly_tier = pd.crosstab(sub['hour_of_day'], sub['flexibility_tier'],
                               normalize='index') * 100
    hourly_tier = hourly_tier.reindex(columns=tier_order, fill_value=0)
    hourly_tier.plot.area(ax=ax, stacked=True,
                          color=[tier_colors[t] for t in tier_order],
                          alpha=0.8, linewidth=0)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('% of Sessions')
    ax.set_title('Flexibility Tier by Hour')
    ax.set_xlim(0, 23)
    ax.legend(fontsize=8, loc='upper right')

    # Bar chart: tier counts by station
    ax = axes[1]
    station_tier = pd.crosstab(
        sub['station_name'].map(STATION_LABELS),
        sub['flexibility_tier'],
        normalize='index'
    ) * 100
    station_tier = station_tier.reindex(columns=tier_order, fill_value=0)
    station_tier.plot.barh(ax=ax, stacked=True,
                           color=[tier_colors[t] for t in tier_order],
                           alpha=0.8)
    ax.set_xlabel('% of Sessions')
    ax.set_title('Flexibility Tier by Station')
    ax.legend(fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig(fig_dir / 'flexibility_by_hour_station.png')
    plt.close(fig)
    print(f"    Saved: flexibility_by_hour_station.png")

    # ── Fig 3: Tier by charger type ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ctype_tier = pd.crosstab(sub['charger_type'], sub['flexibility_tier'],
                              normalize='index') * 100
    ctype_tier = ctype_tier.reindex(columns=tier_order, fill_value=0)
    ctype_tier.plot.bar(ax=ax, stacked=True,
                        color=[tier_colors[t] for t in tier_order],
                        alpha=0.8)
    ax.set_xlabel('Charger Type')
    ax.set_ylabel('% of Sessions')
    ax.set_title('Flexibility Tier by Charger Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'flexibility_by_charger_type.png')
    plt.close(fig)
    print(f"    Saved: flexibility_by_charger_type.png")


def run_flexibility_analysis(df: pd.DataFrame,
                              output_dir: Path) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Run the full 7B flexibility analysis.

    Returns:
      flex_df:      per-session flexibility assignments (representative stations, clean schema)
      flex_summary: summary dict for JSON output
      enriched_sub: full sub-dataframe with benchmark columns (for use in 7C)
    """
    print("\n" + "=" * 70)
    print("COMPONENT 7B: JIAXING FLEXIBILITY PROXY")
    print("=" * 70)

    # Step 1: Compute benchmark power
    benchmark_kw, benchmark_src, benchmark_n = compute_benchmark_power(df)

    # Step 2: Assign tiers
    sub = assign_flexibility_tiers(df, benchmark_kw, REPRESENTATIVE_STATIONS)

    # Step 3: Build outputs
    flex_df = build_flexibility_output(sub)
    flex_summary = build_flexibility_summary(sub, benchmark_kw, benchmark_src, benchmark_n)

    # Step 4: Save
    flex_df.to_csv(output_dir / 'flexibility_analysis.csv', index=False)
    print(f"\n  Saved: flexibility_analysis.csv ({len(flex_df):,} rows)")

    with open(output_dir / 'flexibility_summary.json', 'w', encoding='utf-8') as f:
        json.dump(to_builtin(flex_summary), f, indent=2)
    print(f"  Saved: flexibility_summary.json")

    # Step 5: Plots
    plot_flexibility(sub, output_dir)

    # Step 6: Key findings
    print("\n  Key findings:")
    for tier in ['likely_flexible', 'possibly_flexible', 'unlikely_flexible']:
        frac = flex_summary['tier_fractions'].get(tier, 0)
        print(f"    {tier}: {frac:.1%}")

    # Peak-hour flexibility (hours 10-14, where TOU is Peak/Super-peak)
    peak_mask = sub['hour_of_day'].isin(range(10, 15))
    if peak_mask.any():
        peak_flex = sub.loc[peak_mask, 'flexibility_tier'].isin(
            ['likely_flexible', 'possibly_flexible']).mean()
        print(f"    Flexible (likely+possibly) during peak hours (10-14): {peak_flex:.1%}")

    return flex_df, flex_summary, sub


# =====================================================================
# 7C: ACN CROSS-VALIDATION
# =====================================================================

def run_acn_crossval(acn: Optional[pd.DataFrame],
                     flex_summary: dict,
                     jiaxing_sub: pd.DataFrame,
                     output_dir: Path) -> dict:
    """
    Compare Jiaxing flexibility proxy against ACN explicit slack.

    ACN provides explicit fields:
      - connectionTime: when the car was plugged in
      - doneChargingTime: when charging completed
      - disconnectTime: when the car was unplugged
      - kWhDelivered: energy delivered

    Explicit slack = disconnectTime - doneChargingTime.

    This is a plausibility reference, NOT a fallback.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 7C: ACN CROSS-VALIDATION")
    print("=" * 70)

    crossval = {
        'decision': 'heuristic_screen_with_sensitivity_only',
        'sensitivity_fractions': [0.25, 0.50, 0.75, 1.00],
    }

    if acn is None:
        print("  ACN data not available. Skipping cross-validation.")
        crossval['note'] = 'ACN data not loaded; cross-validation skipped.'
        with open(output_dir / 'flexibility_crossvalidation.json', 'w',
                  encoding='utf-8') as f:
            json.dump(crossval, f, indent=2)
        return crossval

    jiaxing_available = (not jiaxing_sub.empty) and ('flex_raw' in jiaxing_sub.columns)
    if not jiaxing_available:
        print("  Jiaxing flexibility output not available. Running ACN-only summary.")

    # ── Compute ACN explicit slack ──────────────────────────────────
    print("\n  Computing ACN explicit slack ...")

    # Parse times
    for col in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
        if col in acn.columns:
            acn[col] = pd.to_datetime(acn[col], errors='coerce')

    # Explicit slack
    acn_valid = acn.dropna(subset=['disconnectTime', 'doneChargingTime']).copy()
    acn_valid['explicit_slack_min'] = (
        (acn_valid['disconnectTime'] - acn_valid['doneChargingTime'])
        .dt.total_seconds() / 60.0
    )
    # Filter: non-negative slack, reasonable range
    acn_valid = acn_valid[
        (acn_valid['explicit_slack_min'] >= 0) &
        (acn_valid['explicit_slack_min'] < 1440)  # < 24 hours
    ].copy()

    print(f"    ACN sessions with valid explicit slack: {len(acn_valid):,}")

    if len(acn_valid) == 0:
        print("  WARNING: No valid ACN sessions for slack computation.")
        crossval['note'] = 'No valid ACN slack computed.'
        with open(output_dir / 'flexibility_crossvalidation.json', 'w',
                  encoding='utf-8') as f:
            json.dump(crossval, f, indent=2)
        return crossval

    # ACN explicit slack stats
    slack = acn_valid['explicit_slack_min']
    crossval['acn_n_valid'] = int(len(acn_valid))
    crossval['acn_median_explicit_slack_min'] = round(float(slack.median()), 1)
    crossval['acn_mean_explicit_slack_min'] = round(float(slack.mean()), 1)
    crossval['acn_fraction_slack_gt_30min'] = round(float((slack > 30).mean()), 4)
    crossval['acn_fraction_slack_gt_60min'] = round(float((slack > 60).mean()), 4)
    crossval['acn_fraction_slack_gt_120min'] = round(float((slack > 120).mean()), 4)

    print(f"    ACN explicit slack: median={crossval['acn_median_explicit_slack_min']} min, "
          f"mean={crossval['acn_mean_explicit_slack_min']} min")
    print(f"    Fraction > 30 min: {crossval['acn_fraction_slack_gt_30min']:.1%}")
    print(f"    Fraction > 60 min: {crossval['acn_fraction_slack_gt_60min']:.1%}")
    print(f"    Fraction > 120 min: {crossval['acn_fraction_slack_gt_120min']:.1%}")

    # ── Compute ACN implied slack using nominal power ───────────────
    # Use a simple ACN benchmark: L2 at 6.6 kW (standard US Level 2)
    # ACN JPL is predominantly Level 2 workplace charging.
    acn_benchmark_kw = 6.6

    acn_valid['min_charge_time_min'] = np.where(
        acn_valid['kWhDelivered'] > 0,
        acn_valid['kWhDelivered'] / acn_benchmark_kw * 60.0,
        0
    )
    acn_valid['implied_slack_min'] = (
        acn_valid['duration_min'] - acn_valid['min_charge_time_min']
    )
    # Clip to non-negative (implied slack can be negative if actual power > benchmark)
    acn_valid['implied_slack_min'] = acn_valid['implied_slack_min'].clip(lower=0)

    crossval['acn_benchmark_power_kw'] = acn_benchmark_kw
    crossval['acn_median_implied_slack_min'] = round(
        float(acn_valid['implied_slack_min'].median()), 1)

    # Same-session validation of the implied-slack construction. This tests
    # the proxy principle on ACN; it does not make ACN ground truth for Jiaxing.
    spearman = sp_stats.spearmanr(
        acn_valid['implied_slack_min'],
        acn_valid['explicit_slack_min'], nan_policy='omit')
    explicit_positive = acn_valid['explicit_slack_min'] > 30
    implied_positive = acn_valid['implied_slack_min'] > 30
    tp = int((explicit_positive & implied_positive).sum())
    fp = int((~explicit_positive & implied_positive).sum())
    fn = int((explicit_positive & ~implied_positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    crossval['same_session_validation'] = {
        'spearman_rho': round(float(spearman.statistic), 4),
        'spearman_pvalue': float(spearman.pvalue),
        'mae_slack_min': round(float(np.mean(np.abs(
            acn_valid['implied_slack_min']
            - acn_valid['explicit_slack_min']))), 2),
        'threshold_min': 30,
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'scope': ('Validates implied versus explicit slack on the same ACN '
                  'sessions; transfer to Jiaxing remains unvalidated.'),
    }

    # ── Compute Jiaxing implied slack ───────────────────────────────
    print("\n  Computing Jiaxing implied slack ...")

    if jiaxing_available:
        # Use the user_stop_proxy sessions from representative stations
        jx_proxy = jiaxing_sub[jiaxing_sub['flex_raw'] == 'user_stop_proxy'].copy()
    else:
        jx_proxy = pd.DataFrame()

    if not jiaxing_available:
        crossval['jiaxing_implied_slack_note'] = 'Jiaxing flexibility analysis unavailable'
    elif 'benchmark_power_kw' not in jx_proxy.columns:
        print("  WARNING: benchmark_power_kw not in jiaxing sub. Cannot compute implied slack.")
        crossval['jiaxing_implied_slack_note'] = 'benchmark_power_kw missing'
    elif len(jx_proxy) > 0:
        jx_proxy['min_charge_time_min'] = np.where(
            (jx_proxy['energy_kwh'] > 0) & (jx_proxy['benchmark_power_kw'] > 0),
            jx_proxy['energy_kwh'] / jx_proxy['benchmark_power_kw'] * 60.0,
            0
        )
        jx_proxy['implied_slack_min'] = (
            jx_proxy['charging_duration_min'] - jx_proxy['min_charge_time_min']
        )
        jx_proxy['implied_slack_min'] = jx_proxy['implied_slack_min'].clip(lower=0)

        jx_slack = jx_proxy['implied_slack_min']
        crossval['jiaxing_n_proxy'] = int(len(jx_proxy))
        crossval['jiaxing_median_implied_slack_min'] = round(float(jx_slack.median()), 1)
        crossval['jiaxing_mean_implied_slack_min'] = round(float(jx_slack.mean()), 1)
        crossval['jiaxing_fraction_implied_slack_gt_30min'] = round(
            float((jx_slack > 30).mean()), 4)
        crossval['jiaxing_fraction_implied_slack_gt_60min'] = round(
            float((jx_slack > 60).mean()), 4)

        print(f"    Jiaxing implied slack: median={crossval['jiaxing_median_implied_slack_min']} min, "
              f"mean={crossval['jiaxing_mean_implied_slack_min']} min")
        print(f"    Fraction > 30 min: {crossval['jiaxing_fraction_implied_slack_gt_30min']:.1%}")
        print(f"    Fraction > 60 min: {crossval['jiaxing_fraction_implied_slack_gt_60min']:.1%}")
    else:
        crossval['jiaxing_implied_slack_note'] = 'No user_stop_proxy sessions'

    # ── Distribution comparison ─────────────────────────────────────
    print("\n  Distribution comparison ...")

    # Context note: ACN is workplace (JPL), Jiaxing is mixed-use.
    # Slack distributions will likely differ due to use-case, not proxy quality.
    crossval['decision'] = 'heuristic_screen_with_sensitivity_only'
    crossval['distribution_comparison_note'] = (
        "ACN-Data JPL is workplace charging (users park all day, unplug at departure). "
        "Jiaxing is mixed-use (expressway, urban, institutional). "
        "ACN slack reflects parking behavior, not charging flexibility. "
        "The distributions are expected to differ structurally. "
        "Same-session ACN implied-versus-explicit slack statistics validate only "
        "the proxy construction on ACN, not its transfer to Jiaxing. "
        "Retained decision: use the Jiaxing proxy as a heuristic screen with "
        "sensitivity analysis "
        "at 25/50/75/100% of the proxy-flexible fraction."
    )

    # ── Save ACN output CSV ─────────────────────────────────────────
    acn_out_cols = ['explicit_slack_min', 'implied_slack_min']
    if 'hour_of_day' in acn_valid.columns:
        acn_out_cols.insert(0, 'hour_of_day')
    # Add a session_id
    acn_valid = acn_valid.copy()
    acn_valid['session_id'] = acn_valid.index.astype(str)
    acn_out_cols.insert(0, 'session_id')
    acn_out = acn_valid[[c for c in acn_out_cols if c in acn_valid.columns]]
    acn_out.to_csv(output_dir / 'acn_flexibility.csv', index=False)
    print(f"\n  Saved: acn_flexibility.csv ({len(acn_out):,} rows)")

    # ── Save crossvalidation JSON ───────────────────────────────────
    with open(output_dir / 'flexibility_crossvalidation.json', 'w',
              encoding='utf-8') as f:
        json.dump(crossval, f, indent=2)
    print(f"  Saved: flexibility_crossvalidation.json")

    # ── Plot: CDF comparison ────────────────────────────────────────
    fig_dir = output_dir / 'figures'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ACN explicit vs implied slack CDFs
    ax = axes[0]
    if len(acn_valid) > 0:
        explicit = np.sort(acn_valid['explicit_slack_min'].values)
        implied = np.sort(acn_valid['implied_slack_min'].dropna().values)
        ax.plot(explicit, np.arange(1, len(explicit)+1)/len(explicit),
                color='#2166ac', lw=1.5, label='ACN explicit slack')
        ax.plot(implied, np.arange(1, len(implied)+1)/len(implied),
                color='#e41a1c', lw=1.5, ls='--', label='ACN implied slack (6.6 kW)')
    ax.set_xlabel('Slack (minutes)')
    ax.set_ylabel('CDF')
    ax.set_title('ACN Slack: Explicit vs Implied')
    ax.set_xlim(0, 600)
    ax.legend(fontsize=9)

    # Right: Jiaxing implied slack vs ACN explicit slack
    ax = axes[1]
    if len(acn_valid) > 0:
        ax.plot(explicit, np.arange(1, len(explicit)+1)/len(explicit),
                color='#2166ac', lw=1.5, label='ACN explicit slack')
    if 'implied_slack_min' in jx_proxy.columns and len(jx_proxy) > 0:
        jx_vals = np.sort(jx_proxy['implied_slack_min'].dropna().values)
        ax.plot(jx_vals, np.arange(1, len(jx_vals)+1)/len(jx_vals),
                color='#4daf4a', lw=1.5, label='Jiaxing implied slack')
    ax.set_xlabel('Slack (minutes)')
    ax.set_ylabel('CDF')
    ax.set_title('Cross-Validation: Jiaxing vs ACN')
    ax.set_xlim(0, 600)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / 'flexibility_crossvalidation.png')
    plt.close(fig)
    print(f"  Saved: flexibility_crossvalidation.png")

    return crossval


# =====================================================================
# METADATA
# =====================================================================

def save_metadata(output_dir: Path, phase2_summary: dict,
                  flex_summary: dict, crossval: dict) -> dict:
    """Save week7_metadata.json."""

    metadata = {
        'week': 7,
        'components': [
            '7A: Phase 2 Narrative Synthesis',
            '7B: Jiaxing Flexibility Proxy',
            '7C: ACN Cross-Validation',
        ],
        'phase2_claims': phase2_summary,
        'flexibility_summary': {
            'benchmark_power_source': flex_summary.get('benchmark_power_source', {}),
            'benchmark_power_kw': flex_summary.get('benchmark_power_kw', {}),
            'thresholds': flex_summary.get('thresholds', {}),
            'tier_fractions': flex_summary.get('tier_fractions', {}),
        },
        'crossvalidation_decision': crossval.get('decision', 'unknown'),
        'files_produced': [
            'phase2_conclusions.md',
            'flexibility_analysis.csv',
            'flexibility_summary.json',
            'acn_flexibility.csv',
            'flexibility_crossvalidation.json',
            'week7_metadata.json',
        ],
        'figures_produced': [],
    }

    # List figures
    fig_dir = output_dir / 'figures'
    if fig_dir.exists():
        metadata['figures_produced'] = sorted(
            f for f in os.listdir(fig_dir) if f.endswith('.png')
        )

    with open(output_dir / 'week7_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(to_builtin(metadata), f, indent=2)
    print(f"\n  Saved: week7_metadata.json")

    return metadata


# =====================================================================
# MAIN
# =====================================================================

def main():
    configure_console_output()

    parser = argparse.ArgumentParser(
        description='Week 7: Phase 2c Wrap-Up + Flexibility Analysis')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory with jiaxing_clean and acn_clean parquet files')
    parser.add_argument('--week3-dir', type=str,
                        default=str(RESULTS_DIR / 'week3_results'),
                        help='Week 3 output directory (nhpp_rate_functions.csv)')
    parser.add_argument('--week4-dir', type=str,
                        default=str(RESULTS_DIR / 'week4_results'),
                        help='Week 4 output directory (parameter_summary, xgboost, etc.)')
    parser.add_argument('--week5-dir', type=str,
                        default=str(RESULTS_DIR / 'week5_results'),
                        help='Week 5 output directory (lstm_comparison, ML comparisons)')
    parser.add_argument('--week6-dir', type=str,
                        default=str(RESULTS_DIR / 'week6_results'),
                        help='Week 6 output directory (pareto sweeps, fault tax, etc.)')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'week7_results'),
                        help='Output directory for Week 7 results')
    parser.add_argument('--skip-7a', action='store_true',
                        help='Skip Phase 2 synthesis')
    parser.add_argument('--skip-7b', action='store_true',
                        help='Skip Jiaxing flexibility proxy')
    parser.add_argument('--skip-7c', action='store_true',
                        help='Skip ACN cross-validation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    week3_dir = Path(args.week3_dir)
    week4_dir = Path(args.week4_dir)
    week5_dir = Path(args.week5_dir)
    week6_dir = Path(args.week6_dir)

    print("=" * 70)
    print("WEEK 7: PHASE 2c WRAP-UP + FLEXIBILITY ANALYSIS")
    print("=" * 70)
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Week 3 dir: {week3_dir}")
    print(f"  Week 4 dir: {week4_dir}")
    print(f"  Week 5 dir: {week5_dir}")
    print(f"  Week 6 dir: {week6_dir}")
    print(f"  Output dir: {output_dir}")

    # ── 7A: Phase 2 Synthesis ───────────────────────────────────────
    phase2_summary = {}
    if not args.skip_7a:
        phase2_summary = run_phase2_synthesis(week3_dir, week4_dir, week5_dir, week6_dir, output_dir)
    else:
        print("\n  [SKIP] 7A: Phase 2 synthesis")

    # ── 7B: Flexibility Analysis ────────────────────────────────────
    flex_df = pd.DataFrame()
    flex_summary = {}
    jiaxing_sub = pd.DataFrame()

    if not args.skip_7b:
        # Load Jiaxing data
        print("\n  Loading Jiaxing data ...")
        df = load_jiaxing(args.data_dir)

        flex_df, flex_summary, jiaxing_sub = run_flexibility_analysis(df, output_dir)
    else:
        print("\n  [SKIP] 7B: Flexibility analysis")

    # ── 7C: ACN Cross-Validation ────────────────────────────────────
    crossval = {}
    if not args.skip_7c:
        acn = load_acn(args.data_dir)
        crossval = run_acn_crossval(acn, flex_summary, jiaxing_sub, output_dir)
    else:
        print("\n  [SKIP] 7C: ACN cross-validation")

    # ── Metadata ────────────────────────────────────────────────────
    save_metadata(output_dir, phase2_summary, flex_summary, crossval)

    # ── Done ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WEEK 7 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
