"""
Week 8: Phase 3a — LP Scheduling vs FCFS Baseline
===================================================
Components:
  8B. Representative day extraction (5 days × 4 stations)
  8C. MILP formulation and solve (PuLP, 30-min slots)
  8D. FCFS replay on same days (SimPy) + LP vs FCFS comparison
  8A. TOU costing on day-level schedules
  8X. Benchmark power sensitivity (P75/P90/P95 tier counts)

Locked design decisions:
  - Primary comparison: same sessions, same fleet size (s* faults-ON),
    FCFS replay vs LP schedule. Everything else is secondary context.
  - Faulted sessions excluded from BOTH FCFS replay and LP.
  - LP shift window is one-sided: arrival_slot ≤ t ≤ arrival_slot + max_shift.
  - Occupancy constraint is time-indexed: for every slot τ, sessions
    active in τ ≤ s.
  - TOU cost uses service start time price, not arrival hour.
    No multi-hour proration (stated simplification).
  - FCFS replay orders by arrival_time_min, then session_id as tie-breaker.
  - 30-minute slots (T=48 per day).

Input files:
  - jiaxing_clean.parquet        (441k sessions, local)
  - flexibility_analysis.csv     (from Week 7)
  - flexibility_summary.json     (from Week 7)
  - parameter_summary.json       (service time fits, fleet sizing)
  - fault_tax_results.csv        (s* values from Week 6)

Output files:
  - representative_day_selection.json
  - representative_days.csv
  - fcfs_replay_results.csv
  - lp_results.csv
  - lp_vs_fcfs_comparison.csv
  - benchmark_sensitivity.csv
  - week8_metadata.json

Usage:
  python week8_analysis.py --data-dir ./data --output-dir ./week8_results
                           [--week6-dir ./week6_results]
                           [--week7-dir ./week7_results]
                           [--week4-dir ./week4_results]

Date: Week 8, Mar 2026
"""

import argparse
import heapq
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR, to_builtin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import simpy
    HAS_SIMPY = True
except ImportError:
    HAS_SIMPY = False
    print("[WARN] SimPy not installed. FCFS replay will be skipped.")

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("[WARN] PuLP not installed. LP scheduling will be skipped.")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

SCRIPT_DIR = Path(__file__).resolve().parent


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

# s* from Week 6 faults-ON Pareto frontier
S_STAR_FAULTS_ON = {
    'Xiuzhou_Expressway Service District A': 4,
    'Nanhu_Technology Park': 6,
    'Xiuzhou_Government Agency': 8,
    'Tongxiang_Bus Station': 10,
}

SLOT_MINUTES = 30
T_SLOTS = 48  # 24 hours / 30 min
DELAY_TIEBREAKER_WEIGHT = 1e-6
TOU_PRICE_FILENAME = 'Time-of-use_Price.csv'

# Day type labels
DAY_TYPES = [
    'high_demand_summer_weekday',
    'low_demand_winter_weekday',
    'rainy_day',
    'holiday',
    'weekend',
]


# =====================================================================
# CONSOLE ENCODING SAFETY
# =====================================================================

def configure_console_output():
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass


# =====================================================================
# DATA LOADING
# =====================================================================

def load_jiaxing(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    for name in ['jiaxing_clean.parquet', 'jiaxing_clean.csv']:
        p = data_path / name
        if p.exists():
            print(f"  Loading {p} ...")
            if name.endswith('.parquet'):
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, parse_dates=['start_time', 'end_time'])
            print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")
            return df
    raise FileNotFoundError(f"jiaxing_clean not found in {data_path}")


def load_flexibility(week7_dir: Path) -> Tuple[pd.DataFrame, dict]:
    flex_path = week7_dir / 'flexibility_analysis.csv'
    summ_path = week7_dir / 'flexibility_summary.json'

    if not flex_path.exists():
        raise FileNotFoundError(f"flexibility_analysis.csv not found in {week7_dir}")

    flex_df = pd.read_csv(flex_path)
    print(f"  Loaded flexibility_analysis.csv: {len(flex_df):,} rows")

    flex_summary = {}
    if summ_path.exists():
        with open(summ_path, 'r', encoding='utf-8') as f:
            flex_summary = json.load(f)

    return flex_df, flex_summary


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_tou_price_lookup(df: pd.DataFrame) -> Dict[int, float]:
    return build_validated_tou_price_lookup(df)
    """Build hour → mean TOU price lookup from session data."""
    if 'tou_electricity_price' in df.columns and 'hour_of_day' in df.columns:
        lookup = df.groupby('hour_of_day')['tou_electricity_price'].mean().to_dict()
        print(f"  TOU price lookup: {len(lookup)} hours, "
              f"range [{min(lookup.values()):.3f}, {max(lookup.values()):.3f}]")
        return lookup
    print("  WARN: Cannot build TOU lookup. Using uniform price.")
    return {h: 0.70 for h in range(24)}


def _parse_hhmm(value: str) -> Tuple[int, int]:
    hour, minute = str(value).strip().split(':')
    return int(hour), int(minute)


def _hours_covered_by_period(period: str) -> List[int]:
    start_s, end_s = str(period).split('-')
    start_hour, _ = _parse_hhmm(start_s)
    end_hour, end_minute = _parse_hhmm(end_s)

    if end_hour == 0 and end_minute == 0:
        last_hour = 23
    elif end_minute == 0:
        last_hour = end_hour - 1
    else:
        last_hour = end_hour

    if last_hour < start_hour:
        return list(range(start_hour, 24)) + list(range(0, last_hour + 1))
    return list(range(start_hour, last_hour + 1))


def _lookup_from_tou_csv(path: Path) -> Dict[int, float]:
    price_df = pd.read_csv(path)
    period_col = next((c for c in price_df.columns if 'period' in c.lower()), None)
    price_col = next((c for c in price_df.columns if 'price' in c.lower()), None)
    if period_col is None or price_col is None:
        raise ValueError(f"{path} must contain time-period and price columns")

    lookup = {}
    for _, row in price_df.iterrows():
        price = float(row[price_col])
        for hour in _hours_covered_by_period(row[period_col]):
            lookup[int(hour)] = price
    return lookup


def validate_tou_lookup(lookup: Dict[int, float]) -> Dict[int, float]:
    missing = [h for h in range(24) if h not in lookup]
    nonfinite = [
        h for h in range(24)
        if h in lookup and not np.isfinite(float(lookup[h]))
    ]
    if missing or nonfinite:
        raise ValueError(
            "Invalid TOU lookup. Missing hours: "
            f"{missing}; non-finite hours: {nonfinite}"
        )
    return {h: float(lookup[h]) for h in range(24)}


def build_validated_tou_price_lookup(df: pd.DataFrame) -> Dict[int, float]:
    """Build a validated 24-hour TOU price lookup."""
    tou_path = SCRIPT_DIR / TOU_PRICE_FILENAME
    if tou_path.exists():
        lookup = _lookup_from_tou_csv(tou_path)
        source = str(tou_path)
    elif 'tou_electricity_price' in df.columns and 'hour_of_day' in df.columns:
        lookup = (
            df.groupby('hour_of_day')['tou_electricity_price']
            .mean()
            .dropna()
            .to_dict()
        )
        source = 'session tou_electricity_price means'
    else:
        raise ValueError(
            "Cannot build TOU lookup: no Time-of-use_Price.csv and no "
            "session-level TOU price columns."
        )

    lookup = validate_tou_lookup(lookup)
    values = list(lookup.values())
    print(f"  TOU price lookup ({source}): 24 hours, "
          f"range [{min(values):.3f}, {max(values):.3f}], "
          f"hour23={lookup[23]:.4f}")
    return lookup


def tou_price_for_slot(slot: int, tou_lookup: Dict[int, float]) -> float:
    hour = int((slot * SLOT_MINUTES) // 60) % 24
    return float(tou_lookup[hour])


# =====================================================================
# 8B: REPRESENTATIVE DAY EXTRACTION
# =====================================================================

def select_representative_days(df: pd.DataFrame,
                                stations: List[str]) -> Dict[str, str]:
    """
    Select 5 representative dates from the Jiaxing dataset.
    Uses the same dates across all stations for comparability.

    Selection:
      - high_demand_summer_weekday: weekday in Jun-Aug 2021 with highest total arrivals
      - low_demand_winter_weekday: weekday in Dec 2020-Feb 2021 with lowest non-zero arrivals
      - rainy_day: a day with is_rainy==True and moderate demand
      - holiday: a day during CNY or National Day
      - weekend: Saturday/Sunday with typical demand
    """
    print("\n" + "=" * 70)
    print("COMPONENT 8B: REPRESENTATIVE DAY SELECTION")
    print("=" * 70)

    sub = df[df['station_name'].isin(stations)].copy()

    # Ensure date column
    if 'date_dt' in sub.columns:
        sub['_date'] = pd.to_datetime(sub['date_dt'])
    elif 'start_time' in sub.columns:
        sub['_date'] = pd.to_datetime(sub['start_time']).dt.normalize()
    else:
        raise ValueError("Cannot determine session date")

    sub['_dow'] = sub['_date'].dt.dayofweek  # 0=Mon, 6=Sun
    sub['_month'] = sub['_date'].dt.month
    sub['_year'] = sub['_date'].dt.year
    sub['_is_weekday'] = sub['_dow'] < 5

    daily = sub.groupby('_date').agg(
        n_sessions=('_date', 'size'),
        is_weekday=('_is_weekday', 'first'),
        month=('_month', 'first'),
        year=('_year', 'first'),
        dow=('_dow', 'first'),
    ).reset_index()

    # Add is_rainy if available
    if 'is_rainy' in sub.columns:
        rainy_days = sub.groupby('_date')['is_rainy'].max().reset_index()
        rainy_days.columns = ['_date', '_is_rainy']
        daily = daily.merge(rainy_days, on='_date', how='left')
        daily['_is_rainy'] = daily['_is_rainy'].fillna(0).astype(bool)
    else:
        daily['_is_rainy'] = False

    selected = {}

    # 1. High-demand summer weekday (Jun-Aug 2021)
    summer_wd = daily[(daily['month'].isin([6, 7, 8])) & (daily['year'] == 2021) & daily['is_weekday']]
    if len(summer_wd) > 0:
        idx = summer_wd['n_sessions'].idxmax()
        selected['high_demand_summer_weekday'] = str(summer_wd.loc[idx, '_date'].date())
    else:
        # Fallback: highest weekday overall
        wd = daily[daily['is_weekday']]
        if len(wd) > 0:
            idx = wd['n_sessions'].idxmax()
            selected['high_demand_summer_weekday'] = str(wd.loc[idx, '_date'].date())

    # 2. Low-demand winter weekday (Dec 2020 - Feb 2021)
    winter_wd = daily[
        (((daily['month'] == 12) & (daily['year'] == 2020)) |
         ((daily['month'].isin([1, 2])) & (daily['year'] == 2021))) &
        daily['is_weekday'] &
        (daily['n_sessions'] > 0)
    ]
    if len(winter_wd) > 0:
        idx = winter_wd['n_sessions'].idxmin()
        selected['low_demand_winter_weekday'] = str(winter_wd.loc[idx, '_date'].date())
    else:
        wd = daily[daily['is_weekday'] & (daily['n_sessions'] > 0)]
        if len(wd) > 0:
            idx = wd['n_sessions'].idxmin()
            selected['low_demand_winter_weekday'] = str(wd.loc[idx, '_date'].date())

    # 3. Rainy day (moderate demand, is_rainy=True)
    rainy = daily[daily['_is_rainy'] & (daily['n_sessions'] > 0)].copy()
    if len(rainy) > 0:
        median_demand = daily['n_sessions'].median()
        rainy['_dist'] = (rainy['n_sessions'] - median_demand).abs()
        idx = rainy['_dist'].idxmin()
        selected['rainy_day'] = str(rainy.loc[idx, '_date'].date())
    else:
        # Fallback: pick a random weekday
        wd = daily[daily['is_weekday']]
        if len(wd) > 0:
            selected['rainy_day'] = str(wd.iloc[len(wd)//2]['_date'].date())

    # 4. Holiday (CNY ~Feb, National Day ~Oct)
    # CNY 2021: Feb 11-17; National Day 2021: Oct 1-7
    cny_mask = (daily['_date'] >= '2021-02-11') & (daily['_date'] <= '2021-02-17')
    nat_mask = (daily['_date'] >= '2021-10-01') & (daily['_date'] <= '2021-10-07')
    holidays = daily[cny_mask | nat_mask]
    if len(holidays) > 0:
        # Pick the one closest to median demand
        median_demand = daily['n_sessions'].median()
        holidays = holidays.copy()
        holidays['_dist'] = (holidays['n_sessions'] - median_demand).abs()
        idx = holidays['_dist'].idxmin()
        selected['holiday'] = str(holidays.loc[idx, '_date'].date())
    else:
        selected['holiday'] = selected.get('low_demand_winter_weekday', '')

    # 5. Weekend (typical demand Saturday or Sunday)
    weekends = daily[~daily['is_weekday'] & (daily['n_sessions'] > 0)]
    if len(weekends) > 0:
        median_we = weekends['n_sessions'].median()
        weekends = weekends.copy()
        weekends['_dist'] = (weekends['n_sessions'] - median_we).abs()
        idx = weekends['_dist'].idxmin()
        selected['weekend'] = str(weekends.loc[idx, '_date'].date())
    else:
        selected['weekend'] = selected.get('high_demand_summer_weekday', '')

    for label, date in selected.items():
        n = daily.loc[daily['_date'] == date, 'n_sessions'].values
        n_str = f"{int(n[0]):,}" if len(n) > 0 else "?"
        print(f"  {label:35s} → {date}  ({n_str} sessions)")

    return selected


def extract_representative_days(df: pd.DataFrame,
                                 flex_df: pd.DataFrame,
                                 selected_dates: Dict[str, str],
                                 tou_lookup: Dict[int, float],
                                 stations: List[str],
                                 output_dir: Path) -> pd.DataFrame:
    """
    Extract session-level data for the 5 selected days × 4 stations.
    Merge flexibility tiers from Week 7.
    """
    print("\n  Extracting session data for representative days ...")

    # Ensure date column
    if 'date_dt' in df.columns:
        df['_date_str'] = pd.to_datetime(df['date_dt']).dt.strftime('%Y-%m-%d')
    elif 'start_time' in df.columns:
        df['_date_str'] = pd.to_datetime(df['start_time']).dt.strftime('%Y-%m-%d')

    all_dates = list(selected_dates.values())
    mask = df['station_name'].isin(stations) & df['_date_str'].isin(all_dates)
    sub = df[mask].copy()

    # Invert the date→label mapping
    date_to_label = {v: k for k, v in selected_dates.items()}
    sub['day_label'] = sub['_date_str'].map(date_to_label)

    # Compute arrival_time_min (minutes from midnight)
    if 'start_time' in sub.columns:
        st = pd.to_datetime(sub['start_time'])
        sub['arrival_time_min'] = st.dt.hour * 60 + st.dt.minute + st.dt.second / 60.0
    elif 'hour_of_day' in sub.columns:
        sub['arrival_time_min'] = sub['hour_of_day'] * 60.0
    else:
        sub['arrival_time_min'] = 0.0

    # Use the slot containing the observed arrival time.
    sub['arrival_slot'] = np.floor(sub['arrival_time_min'] / SLOT_MINUTES).astype(int).clip(lower=0)

    # Service duration
    if 'charging_duration_min' in sub.columns:
        sub['service_duration_min'] = sub['charging_duration_min'].clip(lower=0.5)
    else:
        sub['service_duration_min'] = 30.0  # fallback

    sub['duration_slots'] = np.ceil(sub['service_duration_min'] / SLOT_MINUTES).astype(int).clip(lower=1)

    # Energy
    if 'energy_kwh' not in sub.columns:
        sub['energy_kwh'] = 0.0

    # Fault flag
    if 'is_abnormal' in sub.columns:
        sub['is_fault'] = sub['is_abnormal'] == 1
    else:
        sub['is_fault'] = False

    # TOU price at arrival (metadata only)
    if 'hour_of_day' in sub.columns:
        sub['tou_price_at_arrival'] = (
            sub['hour_of_day'].astype(int).map(tou_lookup)
        )
    else:
        sub['tou_price_at_arrival'] = sub['arrival_slot'].map(
            lambda slot: tou_price_for_slot(int(slot), tou_lookup)
        )

    # Merge flexibility tiers. Week 7 uses the original Jiaxing row index as
    # session_id when no explicit session id exists, so recreate that here.
    sub = sub.copy()
    if 'session_id' not in sub.columns:
        sub['session_id'] = sub.index.astype(str)
    else:
        sub['session_id'] = sub['session_id'].astype(str)

    if 'session_id' in flex_df.columns:
        flex_merge = flex_df.copy()
        flex_merge['session_id'] = flex_merge['session_id'].astype(str)
        merged = sub.merge(
            flex_merge[['session_id', 'flexibility_tier', 'max_shift_slots']],
            on='session_id', how='left'
        )
    else:
        print("  WARNING: flexibility_analysis.csv has no session_id column; defaulting all sessions to inflexible.")
        merged = sub.copy()

    # Fill missing flexibility tiers
    if 'flexibility_tier' not in merged.columns:
        merged['flexibility_tier'] = 'inflexible'
    if 'max_shift_slots' not in merged.columns:
        merged['max_shift_slots'] = 0

    matched = int(merged['flexibility_tier'].notna().sum()) if 'flexibility_tier' in merged.columns else 0
    match_rate = matched / len(merged) if len(merged) > 0 else 0.0
    print(f"  Flexibility merge coverage: {matched:,}/{len(merged):,} ({match_rate:.1%})")
    if match_rate < 0.95:
        print("  WARNING: Low flexibility merge coverage. LP may become artificially inflexible.")

    merged['flexibility_tier'] = merged['flexibility_tier'].fillna('inflexible')
    merged['max_shift_slots'] = merged['max_shift_slots'].fillna(0).astype(int)

    # For fault sessions, ensure they are marked
    merged.loc[merged['is_fault'], 'flexibility_tier'] = 'fault'
    merged.loc[merged['is_fault'], 'max_shift_slots'] = -1

    # Build a stable session_id if not present
    if 'session_id' not in merged.columns:
        merged['session_id'] = merged.index.astype(str)

    # Select output columns
    out_cols = [
        'station_name', '_date_str', 'day_label', 'session_id',
        'arrival_time_min', 'arrival_slot', 'charger_type',
        'service_duration_min', 'duration_slots', 'energy_kwh',
        'flexibility_tier', 'max_shift_slots', 'tou_price_at_arrival',
        'is_fault',
    ]
    out = merged[[c for c in out_cols if c in merged.columns]].copy()
    out = out.rename(columns={'_date_str': 'date_dt'})
    out['is_carry_in'] = False

    # Include chargers already occupied at midnight by sessions that began on
    # the preceding day. These synthetic rows affect capacity/utilization but
    # are excluded from selected-day customer metrics.
    if 'start_time' in df.columns and 'end_time' in df.columns:
        starts_all = pd.to_datetime(df['start_time'], errors='coerce')
        ends_all = pd.to_datetime(df['end_time'], errors='coerce')
        carry_rows = []
        for day_label, date_text in selected_dates.items():
            midnight = pd.Timestamp(date_text)
            for station in stations:
                carry_mask = (
                    (df['station_name'] == station)
                    & (starts_all < midnight)
                    & (ends_all > midnight)
                )
                for idx in df.index[carry_mask]:
                    remaining = max(
                        0.5,
                        (ends_all.loc[idx] - midnight).total_seconds() / 60.0,
                    )
                    carry_rows.append({
                        'station_name': station,
                        'date_dt': date_text,
                        'day_label': day_label,
                        'session_id': f'carry_in_{idx}_{date_text}',
                        'arrival_time_min': 0.0,
                        'arrival_slot': 0,
                        'charger_type': df.loc[idx].get('charger_type', 'Unknown'),
                        'service_duration_min': remaining,
                        'duration_slots': max(
                            1, int(np.ceil(remaining / SLOT_MINUTES))),
                        'energy_kwh': 0.0,
                        'flexibility_tier': 'inflexible',
                        'max_shift_slots': 0,
                        'tou_price_at_arrival': tou_price_for_slot(
                            0, tou_lookup),
                        'is_fault': False,
                        'is_carry_in': True,
                    })
        if carry_rows:
            out = pd.concat([out, pd.DataFrame(carry_rows)],
                            ignore_index=True, sort=False)

    # Sort: arrival_time_min, then session_id (deterministic FCFS order)
    out = out.sort_values(
        ['station_name', 'date_dt', 'is_carry_in',
         'arrival_time_min', 'session_id'],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)

    out.to_csv(output_dir / 'representative_days.csv', index=False)
    print(f"  Saved: representative_days.csv ({len(out):,} rows)")

    # Save selection metadata
    sessions_per_day = {}
    measured_out = out[~out['is_carry_in']]
    for _, row in measured_out.groupby(
            ['date_dt', 'station_name']).size().reset_index(name='n').iterrows():
        sessions_per_day.setdefault(row['date_dt'], {})[row['station_name']] = int(row['n'])

    sel_meta = {
        'dates': selected_dates,
        'selection_criteria': (
            'high_demand_summer_weekday: weekday Jun-Aug 2021, max arrivals; '
            'low_demand_winter_weekday: weekday Dec 2020-Feb 2021, min arrivals; '
            'rainy_day: is_rainy=True, closest to median demand; '
            'holiday: CNY or National Day 2021, closest to median demand; '
            'weekend: Sat/Sun, closest to median weekend demand.'
        ),
        'sessions_per_day': sessions_per_day,
    }
    with open(output_dir / 'representative_day_selection.json', 'w', encoding='utf-8') as f:
        json.dump(sel_meta, f, indent=2)
    print(f"  Saved: representative_day_selection.json")

    return out


# =====================================================================
# 8C: MILP SCHEDULING
# =====================================================================

def solve_lp_day(sessions: pd.DataFrame, s: int,
                  tou_lookup: Dict[int, float],
                  verbose: bool = False) -> dict:
    """
    Solve the time-indexed MILP for one station-day.

    Sessions must be non-fault only.

    Returns dict with LP metrics + per-session assignments.
    """
    if not HAS_PULP:
        return {'solver_status': 'SKIP_NO_PULP'}

    n = len(sessions)
    if n == 0:
        return {'solver_status': 'EMPTY', 'n_sessions_nonfault': 0}

    measurement_mask = (~sessions.get(
        'is_carry_in', pd.Series(False, index=sessions.index))
        .fillna(False).astype(bool).values)
    n_measurement = int(measurement_mask.sum())

    # ── Pre-compute per-session parameters ──────────────────────────
    arrival_times_min = sessions['arrival_time_min'].values.astype(float)
    # Never allow a session to start before its actual arrival.
    arrival_slots = np.ceil(arrival_times_min / SLOT_MINUTES).astype(int)
    arrival_slots = np.maximum(arrival_slots, 0)
    duration_slots = sessions['duration_slots'].values.astype(int)
    max_shifts = sessions['max_shift_slots'].values.astype(int)
    energy = np.nan_to_num(sessions['energy_kwh'].values.astype(float), nan=0.0)
    n_schedulable = int((max_shifts > 0).sum())
    n_likely = int(((sessions['flexibility_tier'] == 'likely_flexible').values
                    & measurement_mask).sum())
    n_possibly = int(((sessions['flexibility_tier'] == 'possibly_flexible').values
                      & measurement_mask).sum())

    # ── Pre-LP infeasibility diagnostic ─────────────────────────────
    # Construct a feasible FCFS baseline in slots. Non-flexible customers keep
    # this queued start, while flexible customers may be delayed further by
    # their declared discretionary window.
    server_available = [0] * s
    heapq.heapify(server_available)
    baseline_start_slots = np.zeros(n, dtype=int)
    for i in range(n):
        available = heapq.heappop(server_available)
        start = max(int(arrival_slots[i]), int(available))
        baseline_start_slots[i] = start
        heapq.heappush(server_available, start + int(duration_slots[i]))

    max_allowed_starts = baseline_start_slots + np.maximum(max_shifts, 0)
    max_slot_needed = int(np.max(max_allowed_starts + duration_slots)) + 1
    horizon = max(T_SLOTS, max_slot_needed)
    slot_load_noop = np.zeros(horizon, dtype=int)
    for i in range(n):
        t0 = baseline_start_slots[i]
        for dt in range(duration_slots[i]):
            if t0 + dt < len(slot_load_noop):
                slot_load_noop[t0 + dt] += 1
    peak_slot_load = int(slot_load_noop[:horizon].max())
    capacity_ratio = peak_slot_load / s if s > 0 else float('inf')

    # ── Build MILP ──────────────────────────────────────────────────
    prob = pulp.LpProblem("scheduling", pulp.LpMinimize)
    plan_horizon = max(T_SLOTS, max_slot_needed) + 1

    # Decision variables: x[i][tau] = 1 if session i starts in slot tau
    x = {}
    for i in range(n):
        a = baseline_start_slots[i]
        shift = max(0, int(max_shifts[i]))
        hi = min(a + shift + 1, plan_horizon)
        allowed = range(a, hi)
        for tau in allowed:
            x[i, tau] = pulp.LpVariable(f"x_{i}_{tau}", cat='Binary')

    # Constraint 1: each session starts in exactly one slot
    for i in range(n):
        a = baseline_start_slots[i]
        shift = max(0, int(max_shifts[i]))
        hi = min(a + shift + 1, plan_horizon)
        allowed = range(a, hi)
        prob += pulp.lpSum(x[i, tau] for tau in allowed) == 1, f"assign_{i}"

    # Constraint 2: time-indexed occupancy
    # For every slot tau, sum of sessions active in tau <= s
    for tau in range(plan_horizon):
        active_terms = []
        for i in range(n):
            a = baseline_start_slots[i]
            shift = max(0, int(max_shifts[i]))
            d = duration_slots[i]
            # Session i started at slot t is active in tau if t <= tau < t + d
            # So t ranges from max(a, tau - d + 1) to min(a + shift, tau)
            t_lo = max(a, tau - d + 1)
            t_hi = min(a + shift, tau)
            for t in range(t_lo, t_hi + 1):
                if (i, t) in x:
                    active_terms.append(x[i, t])
        if active_terms:
            prob += pulp.lpSum(active_terms) <= s, f"cap_{tau}"

    # Objective: minimize TOU energy cost, with a tiny delay tie-breaker.
    obj_terms = []
    for i in range(n):
        a = baseline_start_slots[i]
        shift = max(0, int(max_shifts[i]))
        hi = min(a + shift + 1, plan_horizon)
        for tau in range(a, hi):
            price = tou_price_for_slot(tau, tou_lookup)
            energy_cost = price * energy[i]
            discretionary_delay = (
                tau - baseline_start_slots[i]) * SLOT_MINUTES
            delay_penalty = discretionary_delay * DELAY_TIEBREAKER_WEIGHT
            obj_terms.append((energy_cost + delay_penalty) * x[i, tau])
    prob += pulp.lpSum(obj_terms) if obj_terms else 0, "tou_cost_with_delay_tiebreak"

    # ── Solve ───────────────────────────────────────────────────────
    t0_time = time.time()
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
    prob.solve(solver)
    solve_time = time.time() - t0_time

    status = pulp.LpStatus[prob.status]
    is_optimal = status == 'Optimal'
    is_feasible = is_optimal or (prob.status == 1)

    if verbose:
        print(f"    Solver: {status} ({solve_time:.1f}s), "
              f"{n} sessions, {n_schedulable} schedulable")

    # ── Extract solution ────────────────────────────────────────────
    # Read assigned slots
    assigned_slots = np.full(n, -1, dtype=int)
    for i in range(n):
        a = baseline_start_slots[i]
        shift = max(0, int(max_shifts[i]))
        hi = min(a + shift + 1, plan_horizon)
        for tau in range(a, hi):
            if (i, tau) in x and pulp.value(x[i, tau]) is not None and pulp.value(x[i, tau]) > 0.5:
                assigned_slots[i] = tau
                break
        if assigned_slots[i] < 0:
            assigned_slots[i] = a

    if not is_feasible:
        return {
            'solver_status': status,
            'solution_status': str(pulp.constants.LpStatus.get(prob.status, 'Unknown')),
            'solve_time_sec': round(solve_time, 2),
            'is_feasible': False,
            'is_optimal': False,
            'n_sessions_nonfault': n_measurement,
            'n_schedulable': n_schedulable,
            'n_likely_flexible': n_likely,
            'n_possibly_flexible': n_possibly,
            'peak_slot_load': peak_slot_load,
            'capacity_ratio': round(capacity_ratio, 3),
        }

    # Compute metrics from the LP assignment
    assigned_start_min = assigned_slots * SLOT_MINUTES
    real_delays_all = assigned_start_min - arrival_times_min
    if np.any(real_delays_all < -1e-9):
        raise RuntimeError("MILP produced a service start before arrival")
    discretionary_delays = (
        assigned_slots - baseline_start_slots) * SLOT_MINUTES
    real_delays = real_delays_all[measurement_mask]
    shifted_mask = (discretionary_delays > 0) & measurement_mask
    n_shifted = int(shifted_mask.sum())
    shifted_delays = discretionary_delays[shifted_mask]
    mean_shift = float(shifted_delays.mean()) if len(shifted_delays) > 0 else 0.0
    mean_wait = float(real_delays.mean())
    median_wait = float(np.median(real_delays))
    p_wait_gt_15 = float((real_delays > 15).mean())

    # Utilization: compute charger-minutes occupied
    busy_min = 0.0
    sim_duration_min = T_SLOTS * SLOT_MINUTES  # 1440 min = 24 hours
    for i in range(n):
        start_min = assigned_slots[i] * SLOT_MINUTES
        end_min = start_min + sessions.iloc[i]['service_duration_min']
        # Clip to day window
        effective_end = min(end_min, sim_duration_min)
        busy_min += max(0, effective_end - start_min)
    total_capacity = s * sim_duration_min
    utilization = busy_min / total_capacity if total_capacity > 0 else 0.0

    # TOU cost: price at service start hour × energy
    tou_cost = 0.0
    for i in range(n):
        price = tou_price_for_slot(int(assigned_slots[i]), tou_lookup)
        if measurement_mask[i]:
            tou_cost += price * energy[i]

    return {
        'solver_status': status,
        'solution_status': 'Optimal' if is_optimal else 'Feasible',
        'solve_time_sec': round(solve_time, 2),
        'is_feasible': True,
        'is_optimal': is_optimal,
        'n_sessions_nonfault': n_measurement,
        'n_schedulable': n_schedulable,
        'n_likely_flexible': n_likely,
        'n_possibly_flexible': n_possibly,
        'peak_slot_load': peak_slot_load,
        'capacity_ratio': round(capacity_ratio, 3),
        'mean_wait_min': round(mean_wait, 3),
        'median_wait_min': round(median_wait, 3),
        'p_wait_gt_15min': round(p_wait_gt_15, 4),
        'mean_utilization': round(utilization, 4),
        'tou_cost_yuan': round(tou_cost, 2),
        'objective': 'tou_cost_min_with_delay_tiebreak',
        'n_shifted': n_shifted,
        'mean_shift_min': round(mean_shift, 1),
        'throughput': round(n_measurement / (sim_duration_min / 60), 3),
        'assigned_slots': assigned_slots.tolist(),
    }


def run_lp_scheduling(rep_days: pd.DataFrame,
                       tou_lookup: Dict[int, float],
                       output_dir: Path,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Run MILP scheduling for each (station, day) pair.
    Faulted sessions are excluded.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 8C: MILP SCHEDULING")
    print("=" * 70)

    results = []

    for station in REPRESENTATIVE_STATIONS:
        s_star = S_STAR_FAULTS_ON[station]
        label = STATION_LABELS.get(station, station)

        for day_label in DAY_TYPES:
            mask = (
                (rep_days['station_name'] == station) &
                (rep_days['day_label'] == day_label)
            )
            day_data = rep_days[mask].copy()

            if len(day_data) == 0:
                print(f"  {label} / {day_label}: no sessions, skipping.")
                continue

            # Exclude faults
            n_total = len(day_data)
            n_fault = int(day_data['is_fault'].sum())
            nonfault = day_data[~day_data['is_fault']].copy().reset_index(drop=True)
            n_nonfault = len(nonfault)

            print(f"  {label} / {day_label}: {n_total} total, "
                  f"{n_fault} faults excluded, {n_nonfault} for LP, s*={s_star}")

            result = solve_lp_day(nonfault, s_star, tou_lookup, verbose=verbose)
            result['station_name'] = station
            result['day_label'] = day_label
            result['date_dt'] = day_data['date_dt'].iloc[0] if 'date_dt' in day_data.columns else ''
            result['n_chargers'] = s_star
            result['n_sessions_total'] = n_total
            result['n_fault_excluded'] = n_fault
            results.append(result)

    # Build results DataFrame (exclude per-session assigned_slots from CSV)
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'assigned_slots'}
        rows.append(row)
    lp_df = pd.DataFrame(rows)

    lp_df.to_csv(output_dir / 'lp_results.csv', index=False)
    print(f"\n  Saved: lp_results.csv ({len(lp_df)} rows)")

    return lp_df, results  # return raw results with assigned_slots for later use


# =====================================================================
# 8D: FCFS REPLAY
# =====================================================================

def fcfs_replay_day(sessions: pd.DataFrame, s: int,
                     tou_lookup: Dict[int, float]) -> dict:
    """
    Replay a fixed set of sessions through SimPy FCFS queue.
    Sessions must be non-fault, sorted by arrival_time_min then session_id.
    """
    if not HAS_SIMPY:
        return {'solver_status': 'SKIP_NO_SIMPY'}

    n = len(sessions)
    if n == 0:
        return {'n_sessions_nonfault': 0}

    arrival_times = sessions['arrival_time_min'].values
    durations = sessions['service_duration_min'].values
    energies = np.nan_to_num(sessions['energy_kwh'].values.astype(float), nan=0.0)
    measurement_mask = (~sessions.get(
        'is_carry_in', pd.Series(False, index=sessions.index))
        .fillna(False).astype(bool).values)
    n_measurement = int(measurement_mask.sum())

    env = simpy.Environment()
    chargers = simpy.Resource(env, capacity=s)

    wait_times = np.full(n, np.nan, dtype=float)
    service_starts = np.full(n, np.nan, dtype=float)

    def customer(env, idx):
        arrival = arrival_times[idx]
        if env.now < arrival:
            yield env.timeout(arrival - env.now)

        with chargers.request() as req:
            yield req
            service_start = env.now
            wait = service_start - arrival
            wait_times[idx] = wait
            service_starts[idx] = service_start

            duration = max(0.5, durations[idx])
            yield env.timeout(duration)

    # Launch all customers in order
    def arrival_process(env):
        for i in range(n):
            arrival = arrival_times[i]
            if env.now < arrival:
                yield env.timeout(arrival - env.now)
            env.process(customer(env, i))

    env.process(arrival_process(env))
    # Run with drain cap
    env.run(until=1440 + 1440)  # 24h + 24h drain

    # Compute metrics
    completed = np.isfinite(service_starts)
    measured_completed = completed & measurement_mask
    waits = wait_times[measured_completed]
    starts = service_starts
    sim_duration_min = 1440.0

    # Utilization: clipped busy time
    busy_min = 0.0
    for i in np.flatnonzero(completed):
        s_start = starts[i]
        s_end = s_start + max(0.5, durations[i])
        if s_start < sim_duration_min:
            effective_end = min(s_end, sim_duration_min)
            busy_min += max(0, effective_end - s_start)
    total_capacity = s * sim_duration_min
    utilization = busy_min / total_capacity if total_capacity > 0 else 0.0

    # TOU cost: price at service start hour
    tou_cost = 0.0
    for i in np.flatnonzero(measured_completed):
        start_slot = int(np.floor(starts[i] / SLOT_MINUTES))
        price = tou_price_for_slot(start_slot, tou_lookup)
        tou_cost += price * energies[i]

    return {
        'n_sessions_nonfault': n_measurement,
        'mean_wait_min': round(float(waits.mean()), 3) if len(waits) > 0 else 0,
        'median_wait_min': round(float(np.median(waits)), 3) if len(waits) > 0 else 0,
        'p_wait_gt_15min': round(float((waits > 15).mean()), 4) if len(waits) > 0 else 0,
        'mean_utilization': round(utilization, 4),
        'tou_cost_yuan': round(tou_cost, 2),
        'throughput': round(n_measurement / (sim_duration_min / 60), 3),
    }


def run_fcfs_replay(rep_days: pd.DataFrame,
                     tou_lookup: Dict[int, float],
                     output_dir: Path) -> pd.DataFrame:
    """
    Run FCFS replay for each (station, day) pair.
    Same sessions as LP (faults excluded).
    """
    print("\n" + "=" * 70)
    print("COMPONENT 8D: FCFS REPLAY")
    print("=" * 70)

    results = []

    for station in REPRESENTATIVE_STATIONS:
        s_star = S_STAR_FAULTS_ON[station]
        label = STATION_LABELS.get(station, station)

        for day_label in DAY_TYPES:
            mask = (
                (rep_days['station_name'] == station) &
                (rep_days['day_label'] == day_label)
            )
            day_data = rep_days[mask].copy()
            if len(day_data) == 0:
                continue

            n_total = len(day_data)
            n_fault = int(day_data['is_fault'].sum())
            nonfault = day_data[~day_data['is_fault']].copy()
            # Sort by arrival time, then session_id (deterministic)
            nonfault = nonfault.sort_values(['arrival_time_min', 'session_id']).reset_index(drop=True)

            print(f"  {label} / {day_label}: {len(nonfault)} non-fault sessions, s*={s_star}")

            result = fcfs_replay_day(nonfault, s_star, tou_lookup)
            result['station_name'] = station
            result['day_label'] = day_label
            result['date_dt'] = day_data['date_dt'].iloc[0] if 'date_dt' in day_data.columns else ''
            result['n_chargers'] = s_star
            result['n_sessions_total'] = n_total
            result['n_fault_excluded'] = n_fault
            results.append(result)

    fcfs_df = pd.DataFrame(results)
    fcfs_df.to_csv(output_dir / 'fcfs_replay_results.csv', index=False)
    print(f"\n  Saved: fcfs_replay_results.csv ({len(fcfs_df)} rows)")

    return fcfs_df


# =====================================================================
# 8A: LP vs FCFS COMPARISON + TOU
# =====================================================================

def build_comparison(lp_df: pd.DataFrame, fcfs_df: pd.DataFrame,
                      output_dir: Path) -> pd.DataFrame:
    """
    Build the LP vs FCFS comparison table.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 8A: LP vs FCFS COMPARISON")
    print("=" * 70)

    metrics = ['mean_wait_min', 'p_wait_gt_15min', 'mean_utilization', 'tou_cost_yuan']

    rows = []
    for _, lp_row in lp_df.iterrows():
        station = lp_row['station_name']
        day_label = lp_row['day_label']

        fcfs_match = fcfs_df[
            (fcfs_df['station_name'] == station) &
            (fcfs_df['day_label'] == day_label)
        ]
        if fcfs_match.empty:
            continue

        fcfs_row = fcfs_match.iloc[0]

        for metric in metrics:
            lp_val = lp_row.get(metric, np.nan)
            fcfs_val = fcfs_row.get(metric, np.nan)

            if pd.isna(lp_val) or pd.isna(fcfs_val):
                continue

            delta = lp_val - fcfs_val
            pct_change = (delta / fcfs_val * 100) if fcfs_val != 0 else 0.0

            rows.append({
                'station_name': station,
                'day_label': day_label,
                'metric': metric,
                'fcfs_value': round(fcfs_val, 4),
                'lp_value': round(lp_val, 4),
                'delta': round(delta, 4),
                'pct_change': round(pct_change, 2),
            })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_dir / 'lp_vs_fcfs_comparison.csv', index=False)
    print(f"  Saved: lp_vs_fcfs_comparison.csv ({len(comp_df)} rows)")

    # Print summary
    wait_comp = comp_df[comp_df['metric'] == 'mean_wait_min']
    if len(wait_comp) > 0:
        avg_delta = wait_comp['delta'].mean()
        print(f"\n  Mean wait change (LP - FCFS), avg across all days: {avg_delta:+.2f} min")

    return comp_df


# =====================================================================
# 8E: REDUCED-FLEET SWEEP
# =====================================================================

def run_reduced_fleet_sweep(rep_days: pd.DataFrame,
                             tou_lookup: Dict[int, float],
                             output_dir: Path,
                             verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LP + FCFS at reduced fleet sizes (s*-2, s*-4) to test whether
    scheduling helps once congestion is present.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 8E: REDUCED-FLEET SWEEP")
    print("=" * 70)

    lp_rows = []
    fcfs_rows = []

    for station in REPRESENTATIVE_STATIONS:
        s_star = S_STAR_FAULTS_ON[station]
        label = STATION_LABELS.get(station, station)

        for s_offset in [-2, -4]:
            s_test = max(1, s_star + s_offset)
            if s_test >= s_star:
                continue

            for day_label in DAY_TYPES:
                mask = (
                    (rep_days['station_name'] == station) &
                    (rep_days['day_label'] == day_label)
                )
                day_data = rep_days[mask].copy()
                if len(day_data) == 0:
                    continue

                n_total = len(day_data)
                n_fault = int(day_data['is_fault'].sum())
                nonfault = day_data[~day_data['is_fault']].copy()
                nonfault = nonfault.sort_values(['arrival_time_min', 'session_id']).reset_index(drop=True)

                if len(nonfault) == 0:
                    continue

                print(f"  {label} / {day_label} / s={s_test}: {len(nonfault)} non-fault sessions")

                lp_result = solve_lp_day(nonfault, s_test, tou_lookup, verbose=verbose)
                lp_result['station_name'] = station
                lp_result['day_label'] = day_label
                lp_result['date_dt'] = day_data['date_dt'].iloc[0] if 'date_dt' in day_data.columns else ''
                lp_result['n_chargers'] = s_test
                lp_result['n_sessions_total'] = n_total
                lp_result['n_fault_excluded'] = n_fault
                lp_result['fleet_label'] = f"s_star_{s_offset:+d}"
                lp_rows.append({k: v for k, v in lp_result.items() if k != 'assigned_slots'})

                fcfs_result = fcfs_replay_day(nonfault, s_test, tou_lookup)
                fcfs_result['station_name'] = station
                fcfs_result['day_label'] = day_label
                fcfs_result['date_dt'] = lp_result.get('date_dt', '')
                fcfs_result['n_chargers'] = s_test
                fcfs_result['n_sessions_total'] = n_total
                fcfs_result['n_fault_excluded'] = n_fault
                fcfs_result['fleet_label'] = f"s_star_{s_offset:+d}"
                fcfs_rows.append(fcfs_result)

    lp_sweep = pd.DataFrame(lp_rows)
    fcfs_sweep = pd.DataFrame(fcfs_rows)

    lp_sweep.to_csv(output_dir / 'reduced_fleet_lp.csv', index=False)
    fcfs_sweep.to_csv(output_dir / 'reduced_fleet_fcfs.csv', index=False)
    print(f"\n  Saved: reduced_fleet_lp.csv ({len(lp_sweep)} rows)")
    print(f"  Saved: reduced_fleet_fcfs.csv ({len(fcfs_sweep)} rows)")

    if len(lp_sweep) > 0 and len(fcfs_sweep) > 0:
        comp_rows = []
        metrics = ['mean_wait_min', 'p_wait_gt_15min', 'mean_utilization', 'tou_cost_yuan']

        for _, lp_row in lp_sweep.iterrows():
            fcfs_match = fcfs_sweep[
                (fcfs_sweep['station_name'] == lp_row['station_name']) &
                (fcfs_sweep['day_label'] == lp_row['day_label']) &
                (fcfs_sweep['fleet_label'] == lp_row['fleet_label'])
            ]
            if fcfs_match.empty:
                continue

            fcfs_row = fcfs_match.iloc[0]
            for metric in metrics:
                lp_val = lp_row.get(metric, np.nan)
                fcfs_val = fcfs_row.get(metric, np.nan)
                if pd.isna(lp_val) or pd.isna(fcfs_val):
                    continue

                delta = lp_val - fcfs_val
                pct = (delta / fcfs_val * 100) if fcfs_val != 0 else 0.0
                comp_rows.append({
                    'station_name': lp_row['station_name'],
                    'day_label': lp_row['day_label'],
                    'fleet_label': lp_row['fleet_label'],
                    'n_chargers': lp_row['n_chargers'],
                    'metric': metric,
                    'fcfs_value': round(fcfs_val, 4),
                    'lp_value': round(lp_val, 4),
                    'delta': round(delta, 4),
                    'pct_change': round(pct, 2),
                })

        comp_sweep = pd.DataFrame(comp_rows)
        comp_sweep.to_csv(output_dir / 'reduced_fleet_comparison.csv', index=False)
        print(f"  Saved: reduced_fleet_comparison.csv ({len(comp_sweep)} rows)")

        wait_rows = comp_sweep[comp_sweep['metric'] == 'mean_wait_min']
        if len(wait_rows) > 0:
            for fleet_label in wait_rows['fleet_label'].unique():
                sub = wait_rows[wait_rows['fleet_label'] == fleet_label]
                avg = sub['delta'].mean()
                print(f"  {fleet_label}: avg LP-FCFS wait delta = {avg:+.2f} min")

    return lp_sweep, fcfs_sweep


# =====================================================================
# 8X: BENCHMARK POWER SENSITIVITY
# =====================================================================

def run_benchmark_sensitivity(df: pd.DataFrame,
                               stations: List[str],
                               output_dir: Path) -> pd.DataFrame:
    """
    Re-compute flexibility tiers at P75, P90, P95 benchmark powers.
    Report how the schedulable population changes.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 8X: BENCHMARK POWER SENSITIVITY")
    print("=" * 70)

    # Identify the flex_raw column
    if 'user_stop_proxy' in df.columns:
        flex_col = 'user_stop_proxy'
    elif 'flexibility_tier' in df.columns:
        mapping = {'Flexible': 'user_stop_proxy', 'Inflexible': 'inflexible', 'Fault': 'fault'}
        df['_flex_raw'] = df['flexibility_tier'].map(mapping).fillna(df['flexibility_tier'])
        flex_col = '_flex_raw'
    else:
        df['_flex_raw'] = 'inflexible'
        flex_col = '_flex_raw'

    sub = df[df['station_name'].isin(stations)].copy()

    # Non-fault, non-zero-energy sessions for benchmark computation
    valid = sub[
        (sub.get('is_abnormal', pd.Series(0, index=sub.index)) == 0) &
        (sub.get('flag_zero_energy', pd.Series(0, index=sub.index)) == 0) &
        (sub.get('effective_rate_kw', pd.Series(np.nan, index=sub.index)) > 0)
    ].copy()

    # Compute benchmark at different percentiles
    percentiles = [0.75, 0.90, 0.95]
    results = []

    for pctile in percentiles:
        pct_label = f"P{int(pctile*100)}"

        benchmark_kw = {}
        for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
            vals = valid.loc[valid['charger_type'] == ctype, 'effective_rate_kw']
            if len(vals) >= 100:
                benchmark_kw[ctype] = float(vals.quantile(pctile))
            else:
                fallback = {'DC_Fast': 60.0, 'Level_2': 7.0, 'Mixed': 22.0}
                benchmark_kw[ctype] = fallback.get(ctype, 30.0)

        # Compute utilization ratios
        duration_hr = sub['charging_duration_min'] / 60.0
        realized_power = np.where(duration_hr > 0, sub['energy_kwh'] / duration_hr, np.nan)
        bench_power = sub['charger_type'].map(benchmark_kw)
        util_ratio = np.where(
            (bench_power > 0) & (~np.isnan(realized_power)),
            realized_power / bench_power,
            np.nan
        )
        util_ratio = pd.Series(util_ratio, index=sub.index)

        # Count tiers within user_stop_proxy population
        proxy_mask = sub[flex_col] == 'user_stop_proxy'
        proxy_ur = util_ratio.loc[proxy_mask]
        n_proxy = int(proxy_mask.sum())

        n_likely = int((proxy_ur < 0.5).sum())
        n_possibly = int(((proxy_ur >= 0.5) & (proxy_ur < 0.8)).sum())
        n_schedulable = n_likely + n_possibly

        n_all = len(sub)

        for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
            ctype_proxy = proxy_mask & (sub['charger_type'] == ctype)
            ctype_ur = util_ratio.loc[ctype_proxy]
            n_ct_likely = int((ctype_ur < 0.5).sum())
            n_ct_possibly = int(((ctype_ur >= 0.5) & (ctype_ur < 0.8)).sum())
            n_ct_proxy = int(ctype_proxy.sum())

            results.append({
                'benchmark_level': pct_label,
                'charger_type': ctype,
                'benchmark_kw': round(benchmark_kw[ctype], 2),
                'n_likely_flexible': n_ct_likely,
                'n_possibly_flexible': n_ct_possibly,
                'n_schedulable': n_ct_likely + n_ct_possibly,
                'n_user_stop_proxy': n_ct_proxy,
                'pct_schedulable_within_user_stop_proxy': round(
                    (n_ct_likely + n_ct_possibly) / n_ct_proxy * 100, 1
                ) if n_ct_proxy > 0 else 0.0,
            })

        # Overall row
        results.append({
            'benchmark_level': pct_label,
            'charger_type': 'ALL',
            'benchmark_kw': 0,
            'n_likely_flexible': n_likely,
            'n_possibly_flexible': n_possibly,
            'n_schedulable': n_schedulable,
            'n_user_stop_proxy': n_proxy,
            'pct_schedulable_within_user_stop_proxy': round(
                n_schedulable / n_proxy * 100, 1
            ) if n_proxy > 0 else 0.0,
        })

        # Add all-sessions fraction
        for r in results:
            if r['benchmark_level'] == pct_label:
                r['pct_schedulable_all_sessions'] = round(
                    r['n_schedulable'] / n_all * 100, 2
                ) if n_all > 0 else 0.0

        print(f"  {pct_label}: benchmark DC={benchmark_kw.get('DC_Fast', 0):.1f}, "
              f"L2={benchmark_kw.get('Level_2', 0):.1f}, Mixed={benchmark_kw.get('Mixed', 0):.1f}  "
              f"→ {n_schedulable:,} schedulable ({n_schedulable/n_all*100:.1f}% all, "
              f"{n_schedulable/n_proxy*100:.1f}% proxy)" if n_proxy > 0 else "")

    sens_df = pd.DataFrame(results)
    sens_df.to_csv(output_dir / 'benchmark_sensitivity.csv', index=False)
    print(f"\n  Saved: benchmark_sensitivity.csv ({len(sens_df)} rows)")

    return sens_df


# =====================================================================
# METADATA
# =====================================================================

def save_metadata(output_dir: Path, lp_df: pd.DataFrame,
                  fcfs_df: pd.DataFrame, comp_df: pd.DataFrame,
                  selected_dates: dict,
                  tou_lookup: Dict[int, float]) -> dict:
    metadata = {
        'week': 8,
        'components': [
            '8B: Representative Day Extraction',
            '8C: MILP Scheduling',
            '8D: FCFS Replay + Comparison',
            '8A: TOU Costing',
            '8X: Benchmark Power Sensitivity',
        ],
        'design_decisions': {
            'primary_comparison': 'Same sessions, same fleet size, same day. FCFS replay vs LP schedule.',
            'fault_exclusion': 'Faulted sessions excluded from BOTH FCFS and LP. s* from Week 6 faults-ON frontier.',
            'lp_objective': 'Minimize TOU energy cost; delay is a 1e-6 tie-breaker only.',
            'tou_costing': 'Validated 24-hour TOU price at service start hour only. No multi-hour proration.',
            'tou_price_lookup': {str(k): float(v) for k, v in tou_lookup.items()},
            'fcfs_ordering': 'arrival_time_min ascending, then session_id as tie-breaker.',
            'slot_resolution': f'{SLOT_MINUTES} min ({T_SLOTS} slots per day)',
            'lp_shift_window': 'One-sided: arrival_slot <= t <= arrival_slot + max_shift_slots',
        },
        'representative_dates': selected_dates,
        'infeasible_days': [],
        'files_produced': [
            'representative_day_selection.json',
            'representative_days.csv',
            'lp_results.csv',
            'fcfs_replay_results.csv',
            'lp_vs_fcfs_comparison.csv',
            'reduced_fleet_lp.csv',
            'reduced_fleet_fcfs.csv',
            'reduced_fleet_comparison.csv',
            'benchmark_sensitivity.csv',
            'week8_metadata.json',
        ],
    }

    # Log infeasible days
    if lp_df is not None and 'is_feasible' in lp_df.columns:
        infeasible = lp_df[lp_df['is_feasible'] == False]
        for _, row in infeasible.iterrows():
            metadata['infeasible_days'].append({
                'station': row['station_name'],
                'day_label': row['day_label'],
                'n_chargers': row.get('n_chargers', 0),
                'peak_slot_load': row.get('peak_slot_load', 0),
                'capacity_ratio': row.get('capacity_ratio', 0),
                'note': (f"Infeasible: peak slot load {row.get('peak_slot_load', '?')} "
                         f"exceeds {row.get('n_chargers', '?')} chargers "
                         f"(ratio={row.get('capacity_ratio', '?')})."),
            })

    with open(output_dir / 'week8_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(to_builtin(metadata), f, indent=2)
    print(f"\n  Saved: week8_metadata.json")

    return metadata


# =====================================================================
# MAIN
# =====================================================================

def main():
    configure_console_output()

    parser = argparse.ArgumentParser(
        description='Week 8: Phase 3a — LP Scheduling vs FCFS Baseline')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory with jiaxing_clean parquet')
    parser.add_argument('--week4-dir', type=str,
                        default=str(RESULTS_DIR / 'week4_results'),
                        help='Week 4 output directory (parameter_summary)')
    parser.add_argument('--week6-dir', type=str,
                        default=str(RESULTS_DIR / 'week6_results'),
                        help='Week 6 output directory (fault_tax_results)')
    parser.add_argument('--week7-dir', type=str,
                        default=str(RESULTS_DIR / 'week7_results'),
                        help='Week 7 output directory (flexibility_analysis)')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'week8_results'),
                        help='Output directory for Week 8 results')
    parser.add_argument('--skip-lp', action='store_true',
                        help='Skip LP scheduling (load from CSV)')
    parser.add_argument('--skip-fcfs', action='store_true',
                        help='Skip FCFS replay (load from CSV)')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip benchmark sensitivity')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose solver output')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    week7_dir = Path(args.week7_dir)

    print("=" * 70)
    print("WEEK 8: PHASE 3a — LP SCHEDULING vs FCFS BASELINE")
    print("=" * 70)
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Week 7 dir: {week7_dir}")
    print(f"  Output dir: {output_dir}")

    # ── Load data ───────────────────────────────────────────────────
    print("\n  Loading data ...")
    df = load_jiaxing(args.data_dir)
    flex_df, flex_summary = load_flexibility(week7_dir)
    tou_lookup = build_validated_tou_price_lookup(df)

    # ── 8B: Representative day extraction ───────────────────────────
    selected_dates = select_representative_days(df, REPRESENTATIVE_STATIONS)
    rep_days = extract_representative_days(
        df, flex_df, selected_dates, tou_lookup,
        REPRESENTATIVE_STATIONS, output_dir
    )

    # ── 8C: LP scheduling ──────────────────────────────────────────
    lp_df = pd.DataFrame()
    lp_raw = []
    if not args.skip_lp:
        lp_df, lp_raw = run_lp_scheduling(
            rep_days, tou_lookup, output_dir, verbose=args.verbose
        )
    else:
        print("\n  [SKIP] LP scheduling (loading from CSV)")
        lp_path = output_dir / 'lp_results.csv'
        if lp_path.exists():
            lp_df = pd.read_csv(lp_path)

    # ── 8D: FCFS replay ────────────────────────────────────────────
    fcfs_df = pd.DataFrame()
    if not args.skip_fcfs:
        fcfs_df = run_fcfs_replay(rep_days, tou_lookup, output_dir)
    else:
        print("\n  [SKIP] FCFS replay (loading from CSV)")
        fcfs_path = output_dir / 'fcfs_replay_results.csv'
        if fcfs_path.exists():
            fcfs_df = pd.read_csv(fcfs_path)

    # ── 8A: Comparison ──────────────────────────────────────────────
    comp_df = pd.DataFrame()
    if len(lp_df) > 0 and len(fcfs_df) > 0:
        comp_df = build_comparison(lp_df, fcfs_df, output_dir)
    else:
        print("\n  [SKIP] Comparison (LP or FCFS results missing)")

    # ── 8X: Benchmark sensitivity ───────────────────────────────────
    if not args.skip_lp and not args.skip_fcfs:
        run_reduced_fleet_sweep(rep_days, tou_lookup, output_dir, verbose=args.verbose)
    else:
        print("\n  [SKIP] Reduced-fleet sweep (requires live LP + FCFS run)")

    if not args.skip_sensitivity:
        run_benchmark_sensitivity(df, REPRESENTATIVE_STATIONS, output_dir)
    else:
        print("\n  [SKIP] Benchmark sensitivity")

    # ── Metadata ────────────────────────────────────────────────────
    save_metadata(output_dir, lp_df, fcfs_df, comp_df, selected_dates,
                  tou_lookup)

    # ── Done ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WEEK 8 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
