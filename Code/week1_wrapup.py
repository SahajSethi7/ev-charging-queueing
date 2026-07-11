"""
Week 1 — Wrap-Up Script
========================
Runs AFTER ingest_jiaxing.py has produced jiaxing_clean.parquet.

Handles all remaining Week 1 tasks:
  Wednesday:  station grouping, flexibility_tier, TOU tier, charger_type
  Thursday:   hourly/daily aggregation, lag features, inter-arrival times
  Friday:     final validation, save 4 parquet files, ACN load, quality report

Usage:
    cd path/to/ev-charging-queueing
    python Code/week1_wrapup.py
"""

import json
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

# ============================================================================
# PATHS
# ============================================================================
CLEAN_PATH   = DATA_DIR / "jiaxing_clean.parquet"
CLEAN_CSV_PATH = DATA_DIR / "jiaxing_clean.csv"
OUTPUT_DIR   = DATA_DIR
OUTPUTS_DIR  = RESULTS_DIR

# Output parquet files (Week 1 deliverables)
HOURLY_PATH  = OUTPUT_DIR / "jiaxing_hourly.parquet"
DAILY_PATH   = OUTPUT_DIR / "jiaxing_daily.parquet"
IAT_PATH     = OUTPUT_DIR / "jiaxing_iat.parquet"
FINAL_PATH   = OUTPUT_DIR / "jiaxing_clean.parquet"   # overwrite with new cols

ACN_RAW_DIR  = DATA_DIR
ACN_OUT_PATH = OUTPUT_DIR / "acn_clean.parquet"

REPORT_PATH  = OUTPUTS_DIR / "data_quality_report.md"


# ============================================================================
# COMPATIBILITY HELPERS
# ============================================================================

def configure_console_output() -> None:
    """Avoid UnicodeEncodeError on cp1252-like terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass


def detect_parquet_engine() -> Optional[str]:
    if find_spec('pyarrow') is not None:
        return 'pyarrow'
    if find_spec('fastparquet') is not None:
        return 'fastparquet'
    return None


def save_table(df: pd.DataFrame, parquet_path: Path) -> Path:
    """
    Save to parquet when possible, otherwise CSV fallback.
    Returns the actual saved path.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    engine = detect_parquet_engine()
    if engine is None:
        csv_path = parquet_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        return csv_path

    df.to_parquet(parquet_path, index=False, engine=engine)
    return parquet_path


def first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward/forward compatibility with ingest_jiaxing outputs:
    - parquet or CSV origins
    - weather columns with _x/_y suffixes
    - alternative TOU price names
    """
    # Canonical weather columns expected by this script
    weather_aliases = {
        'temperature': ['temperature', 'temperature_y', 'temperature_x', 'temperature_c'],
        'humidity': ['humidity', 'humidity_y', 'humidity_x', 'relative_humidity'],
        'precipitation': ['precipitation', 'precipitation_y', 'precipitation_x', 'precipitation_mm'],
    }
    for canonical, candidates in weather_aliases.items():
        if canonical not in df.columns:
            src = first_existing(df, candidates)
            if src is not None:
                df[canonical] = df[src]

    # Canonical TOU price column
    if 'tou_electricity_price' not in df.columns:
        src = first_existing(
            df,
            ['tou_electricity_price', 'tou_electricity_price_yuan_kwh', 'electricity_price_yuan_kwh']
        )
        if src is not None:
            df['tou_electricity_price'] = df[src]

    # Numeric coercion for CSV-based loads
    numeric_cols = [
        'station_id', 'energy_kwh', 'elec_fee', 'service_fee', 'total_amount',
        'actual_payment', 'is_abnormal', 'is_user_stop', 'is_nonsystem_fault',
        'is_system_issue', 'is_ev_fault', 'is_other_fault', 'hour_of_day',
        'day_of_week', 'month', 'year', 'is_weekend', 'charging_duration_min',
        'effective_rate_kw', 'price_per_kwh', 'elec_price_per_kwh',
        'service_price_per_kwh', 'tou_electricity_price', 'temperature',
        'humidity', 'precipitation',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse datetime columns when loaded from CSV
    dt_cols = ['order_created_time', 'start_time', 'end_time', 'payment_time']
    for col in dt_cols:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Normalize boolean-like flag columns from CSV text
    bool_map = {
        'true': True, 'false': False, '1': True, '0': False,
        'yes': True, 'no': False, 'y': True, 'n': False
    }
    for col in [c for c in df.columns if c.startswith('flag_')] + ['is_null_session']:
        if col not in df.columns:
            continue
        if df[col].dtype == bool:
            continue
        series = df[col].astype(str).str.strip().str.lower()
        mapped = series.map(bool_map)
        if mapped.notna().mean() >= 0.8:
            df[col] = mapped.fillna(False).astype(bool)

    return df


configure_console_output()


# ============================================================================
# 1. LOAD
# ============================================================================

def load_clean() -> pd.DataFrame:
    configure_console_output()
    print("=" * 70)
    print("WEEK 1 WRAP-UP")
    print("=" * 70)

    parquet_engine = detect_parquet_engine()
    if CLEAN_PATH.exists() and parquet_engine is not None:
        df = pd.read_parquet(CLEAN_PATH, engine=parquet_engine)
        loaded_from = CLEAN_PATH
    elif CLEAN_CSV_PATH.exists():
        df = pd.read_csv(CLEAN_CSV_PATH)
        loaded_from = CLEAN_CSV_PATH
    elif CLEAN_PATH.exists():
        sys.exit(
            "[FATAL] Found jiaxing_clean.parquet but no parquet engine is installed. "
            "Install pyarrow/fastparquet, or rerun ingest_jiaxing.py to generate CSV fallback."
        )
    else:
        sys.exit(
            f"[FATAL] Neither {CLEAN_PATH} nor {CLEAN_CSV_PATH} found. "
            "Run ingest_jiaxing.py first."
        )

    df = harmonize_schema(df)
    print(f"[OK] Loaded {loaded_from.name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ============================================================================
# 2. STATION GROUPING (Wednesday AM)
# ============================================================================

def create_station_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """
    station_id is charger-level (92 unique).  The paper says 13 stations.
    Group by (district, location_info) to recover station-level identity.
    """
    print(f"\n{'=' * 70}")
    print("STATION GROUPING")
    print(f"{'=' * 70}")

    if not all(c in df.columns for c in ['district', 'location_info', 'station_id']):
        if 'station_name' in df.columns:
            n_stations = df['station_name'].nunique()
            print("  ⚠ Missing district/location_info/station_id; using existing station_name.")
            print(f"  Existing station_name: {n_stations} unique stations")
            return df
        raise KeyError(
            "Missing required columns for station grouping. Need either "
            "['district','location_info','station_id'] or existing 'station_name'."
        )

    # Check how many groups (district, location_info) produces
    groups = df.groupby(['district', 'location_info'])['station_id'].nunique()
    n_groups = len(groups)
    print(f"  (district, location_info) groups: {n_groups}")
    print(f"  Charger counts per group:")
    for (dist, loc), n_chargers in groups.items():
        print(f"    {dist:15s}  {loc:25s}  {n_chargers} chargers")

    # Create station_name as "District_LocationInfo"
    df['station_name'] = df['district'] + '_' + df['location_info']
    n_stations = df['station_name'].nunique()
    print(f"\n  Created station_name: {n_stations} unique stations")

    if n_stations != 13:
        print(f"  ⚠ Expected 13 stations, got {n_stations}.")
        print(f"    If this is wrong, consult the paper's Table 2/3 for the")
        print(f"    correct station groupings and update this function.")
    else:
        print(f"  ✓ Matches expected 13 stations.")

    # Print session distribution
    print(f"\n  Sessions per station:")
    for name, count in df['station_name'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    {name:45s}  {count:>8,}  ({pct:.1f}%)")

    return df


# ============================================================================
# 3. ABNORMAL / FAULT ANALYSIS (Wednesday AM)
# ============================================================================

def analyze_faults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break down is_abnormal by station, district, end_cause.
    Document the 18.58% vs 30.9% discrepancy.
    """
    print(f"\n{'=' * 70}")
    print("FAULT ANALYSIS")
    print(f"{'=' * 70}")

    N = len(df)
    if 'is_abnormal' not in df.columns:
        print("  ⚠ is_abnormal not present; skipping fault analysis.")
        return df

    print(f"  is_abnormal rate: {df['is_abnormal'].mean():.2%}")

    # Break down by end_cause
    if 'end_cause' in df.columns:
        print(f"\n  end_cause distribution:")
        for cause, count in df['end_cause'].value_counts().items():
            abn_rate = df.loc[df['end_cause'] == cause, 'is_abnormal'].mean()
            print(f"    {cause:40s}  {count:>8,}  ({100*count/N:.1f}%)  "
                  f"abnormal_rate={abn_rate:.1%}")

    # Break down by district
    print(f"\n  is_abnormal rate by district:")
    for dist, sub in df.groupby('district'):
        print(f"    {dist:15s}  {sub['is_abnormal'].mean():.2%}")

    # Break down by station
    if 'station_name' in df.columns:
        print(f"\n  is_abnormal rate by station (top 5 / bottom 5):")
        station_rates = df.groupby('station_name')['is_abnormal'].mean().sort_values()
        for name, rate in list(station_rates.head(3).items()) + \
                          [('...', np.nan)] + \
                          list(station_rates.tail(3).items()):
            if pd.isna(rate):
                print(f"    {'...':45s}")
            else:
                print(f"    {name:45s}  {rate:.2%}")

    return df


# ============================================================================
# 4. USER-STOP PROXY (Wednesday AM)
# ============================================================================

def create_flexibility_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify sessions by termination type as an OBSERVATIONAL PROXY
    for potential scheduling flexibility.

    Categories:
      - 'user_stop_proxy':  User stopped charging (is_user_stop=1 AND is_abnormal=0).
                            Upper bound on shiftable demand — reason unknown.
      - 'inflexible':       Charging completed normally / is_full_charge=1.
      - 'fault':            is_abnormal=1.

    IMPORTANT: 'user_stop_proxy' ≠ willingness-to-delay.
    Phase 3 scheduling must run sensitivity at 25/50/75/100% of proxy rate.
    """
    print(f"\n{'=' * 70}")
    print("USER-STOP PROXY (observational termination classification)")
    print(f"{'=' * 70}")

    if 'is_abnormal' not in df.columns:
        print("  ⚠ is_abnormal not found. Defaulting user_stop_proxy='inflexible'.")
        df['user_stop_proxy'] = 'inflexible'
        return df

    if 'is_user_stop' not in df.columns:
        if 'end_cause' in df.columns:
            df['is_user_stop'] = (
                df['end_cause']
                .astype(str)
                .str.lower()
                .str.contains('user stop|user stops charging', regex=True, na=False)
                .astype(int)
            )
            print("  ⚠ is_user_stop missing; inferred from end_cause text.")
        else:
            df['is_user_stop'] = 0
            print("  ⚠ is_user_stop missing; defaulted to 0.")

    conditions = [
        df['is_abnormal'] == 1,
        df['is_user_stop'] == 1,
    ]
    choices = ['fault', 'user_stop_proxy']
    df['user_stop_proxy'] = np.select(conditions, choices, default='inflexible')

    tier_counts = df['user_stop_proxy'].value_counts()
    N = len(df)
    for tier, count in tier_counts.items():
        print(f"  {tier:20s}  {count:>8,}  ({100*count/N:.1f}%)")

    print(f"\n  NOTE: 'user_stop_proxy' is observational, not confirmed flexibility.")
    print(f"  Phase 3 must vary the effective shiftable fraction.")

    return df


# ============================================================================
# 5. TOU TIER FROM ACTUAL PRICES (Wednesday PM)
# ============================================================================

def derive_tou_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive TOU tier from actual per-session electricity unit price.
    Thresholds from the weekly plan:
      Valley:     < ¥0.50
      Flat:       ¥0.50 – ¥0.80
      Peak:       ¥0.80 – ¥1.05
      Super-peak: > ¥1.05
    """
    print(f"\n{'=' * 70}")
    print("TOU TIER DERIVATION")
    print(f"{'=' * 70}")

    # Use the TOU electricity price merged from the schedule
    price_col = 'tou_electricity_price'
    if price_col not in df.columns:
        print(f"  ⚠ {price_col} not found. Cannot derive TOU tier.")
        return df

    conditions = [
        df[price_col] < 0.50,
        (df[price_col] >= 0.50) & (df[price_col] < 0.80),
        (df[price_col] >= 0.80) & (df[price_col] <= 1.05),
        df[price_col] > 1.05,
    ]
    choices = ['Valley', 'Flat', 'Peak', 'Super-peak']
    df['tou_tier'] = np.select(conditions, choices, default='Unknown')

    tier_counts = df['tou_tier'].value_counts()
    N = len(df)
    print(f"  TOU tier distribution:")
    for tier, count in tier_counts.items():
        mean_price = df.loc[df['tou_tier'] == tier, price_col].mean()
        print(f"    {tier:12s}  {count:>8,}  ({100*count/N:.1f}%)  "
              f"mean_price=¥{mean_price:.4f}")

    # Cross-check: TOU tier vs hour_of_day
    print(f"\n  TOU tier by hour (spot check):")
    cross = pd.crosstab(df['hour_of_day'], df['tou_tier'])
    # Show just a few representative hours
    for h in [2, 8, 12, 18, 22]:
        if h in cross.index:
            dominant = cross.loc[h].idxmax()
            print(f"    Hour {h:2d}: dominant tier = {dominant}")

    return df


# ============================================================================
# 6. CHARGER TYPE (Wednesday PM)
# ============================================================================

def derive_charger_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive charger_type from effective_rate_kw:
      DC Fast:  > 30 kW
      Level 2:  < 22 kW
      Mixed:    22–30 kW (ambiguous zone)
    """
    print(f"\n{'=' * 70}")
    print("CHARGER TYPE DERIVATION")
    print(f"{'=' * 70}")

    if 'effective_rate_kw' not in df.columns:
        print("  ⚠ effective_rate_kw not found.")
        return df

    conditions = [
        df['effective_rate_kw'] > 30,
        df['effective_rate_kw'] < 22,
    ]
    choices = ['DC_Fast', 'Level_2']
    df['charger_type'] = np.select(conditions, choices, default='Mixed')

    # NaN rates (from zero-duration) → 'Unknown'
    df.loc[df['effective_rate_kw'].isna(), 'charger_type'] = 'Unknown'

    type_counts = df['charger_type'].value_counts()
    N = len(df)
    print(f"  Charger type distribution:")
    for ctype, count in type_counts.items():
        print(f"    {ctype:10s}  {count:>8,}  ({100*count/N:.1f}%)")

    return df


# ============================================================================
# 7. WEATHER: is_rainy (Thursday PM)
# ============================================================================

def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_rainy = (precipitation > 0)."""
    precip_col = first_existing(df, ['precipitation', 'precipitation_y', 'precipitation_x'])
    if precip_col is not None:
        df['is_rainy'] = (pd.to_numeric(df[precip_col], errors='coerce') > 0).astype(int)
        if precip_col != 'precipitation':
            df['precipitation'] = pd.to_numeric(df[precip_col], errors='coerce')
        print(f"\n  is_rainy: {df['is_rainy'].mean():.1%} of sessions")
    else:
        df['is_rainy'] = 0
        print("\n  ⚠ precipitation not available; set is_rainy=0.")
    return df


# ============================================================================
# 8. HOURLY AGGREGATION + LAG FEATURES (Thursday AM)
# ============================================================================

def build_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (station_name, date, hour_of_day) → hourly arrival counts.
    Builds a COMPLETE station × date × hour grid (zero-filling absent bins)
    before computing lag features, ensuring clock-aligned shifts.
    """
    print(f"\n{'=' * 70}")
    print("HOURLY AGGREGATION")
    print(f"{'=' * 70}")

    station_col = 'station_name' if 'station_name' in df.columns else 'station_id'

    # Ensure date is proper date type
    if 'date' not in df.columns and 'start_time' in df.columns:
        df['date'] = pd.to_datetime(df['start_time'], errors='coerce').dt.date
    df['date_dt'] = pd.to_datetime(df['date']) if df['date'].dtype == object else df['date']

    agg_spec = {
        'arrivals': ('start_time', 'count'),
        'mean_energy': ('energy_kwh', 'mean'),
        'mean_duration': ('charging_duration_min', 'mean'),
    }
    if 'temperature' in df.columns:
        agg_spec['mean_temperature'] = ('temperature', 'mean')
    if 'humidity' in df.columns:
        agg_spec['mean_humidity'] = ('humidity', 'mean')
    if 'precipitation' in df.columns:
        agg_spec['total_precipitation'] = ('precipitation', 'max')
    if 'is_abnormal' in df.columns:
        agg_spec['fault_count'] = ('is_abnormal', 'sum')

    hourly_obs = df.groupby([station_col, 'date_dt', 'hour_of_day']).agg(**agg_spec).reset_index()

    # Stable schema
    for col, default in [('mean_temperature', np.nan), ('mean_humidity', np.nan),
                         ('total_precipitation', np.nan), ('fault_count', 0)]:
        if col not in hourly_obs.columns:
            hourly_obs[col] = default

    # ---- ZERO-FILL: complete station × date × hour grid ----
    print("  Building complete station × date × hour grid...")
    all_stations = sorted(hourly_obs[station_col].unique())
    date_min = hourly_obs['date_dt'].min()
    date_max = hourly_obs['date_dt'].max()
    all_dates = pd.date_range(date_min, date_max, freq='D')
    all_hours = list(range(24))

    full_index = pd.MultiIndex.from_product(
        [all_stations, all_dates, all_hours],
        names=[station_col, 'date_dt', 'hour_of_day']
    )
    full_grid = pd.DataFrame(index=full_index).reset_index()
    full_grid['date_dt'] = pd.to_datetime(full_grid['date_dt'])

    hourly = full_grid.merge(
        hourly_obs, on=[station_col, 'date_dt', 'hour_of_day'], how='left'
    )
    hourly['arrivals'] = hourly['arrivals'].fillna(0)
    hourly['fault_count'] = hourly['fault_count'].fillna(0)

    # Forward/back fill weather within station (daily values)
    for wcol in ['mean_temperature', 'mean_humidity', 'total_precipitation']:
        if wcol in hourly.columns:
            hourly[wcol] = (hourly.groupby(station_col)[wcol]
                           .transform(lambda x: x.ffill().bfill()))

    n_filled = len(hourly) - len(hourly_obs)
    print(f"  Grid: {len(hourly):,} total rows ({n_filled:,} zero-fill bins added)")

    # Datetime index for lag computation
    hourly['datetime'] = pd.to_datetime(hourly['date_dt']) + \
                         pd.to_timedelta(hourly['hour_of_day'], unit='h')
    hourly = hourly.sort_values([station_col, 'datetime']).reset_index(drop=True)

    # Lag features (per station, clock-aligned on complete grid)
    print("  Computing lag features on zero-filled grid...")
    for lag_name, lag_periods in [('lag_1', 1), ('lag_24', 24), ('lag_168', 168)]:
        hourly[lag_name] = hourly.groupby(station_col)['arrivals'].shift(lag_periods)

    # Rolling features — SHIFT(1) before rolling to exclude current row.
    # At time t, window covers arrivals[t-1] back through arrivals[t-169].
    hourly['rolling_7d_mean'] = (
        hourly.groupby(station_col)['arrivals']
        .transform(lambda x: x.shift(1).rolling(168, min_periods=24).mean())
    )
    hourly['rolling_7d_std'] = (
        hourly.groupby(station_col)['arrivals']
        .transform(lambda x: x.shift(1).rolling(168, min_periods=24).std())
    )

    # AUDIT CHECK: rolling must not use current row
    _lag1_null = hourly['lag_1'].isna()
    _roll_null = hourly['rolling_7d_mean'].isna()
    _violation = _lag1_null & ~_roll_null
    assert _violation.sum() == 0, (
        f"LEAKAGE CHECK FAILED: {_violation.sum()} rows where lag_1 is NaN "
        f"but rolling_7d_mean is not"
    )
    print(f"  ✓ Rolling leakage check passed")

    # TOU tier for hourly
    if 'tou_tier' in df.columns:
        tou_mode = df.groupby('hour_of_day')['tou_tier'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        ).to_dict()
        hourly['tou_tier'] = hourly['hour_of_day'].map(tou_mode)

    if 'tou_electricity_price' in df.columns:
        tou_price = df.groupby('hour_of_day')['tou_electricity_price'].mean().to_dict()
        hourly['tou_electricity_price'] = hourly['hour_of_day'].map(tou_price)

    # Temporal features
    hourly['day_of_week'] = hourly['datetime'].dt.dayofweek
    hourly['month'] = hourly['datetime'].dt.month
    hourly['is_weekend'] = hourly['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical hour encoding
    hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour_of_day'] / 24)
    hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour_of_day'] / 24)

    print(f"  Hourly table: {hourly.shape[0]:,} rows × {hourly.shape[1]} cols")
    print(f"  Stations: {hourly[station_col].nunique()}")
    print(f"  Date range: {hourly['datetime'].min()} → {hourly['datetime'].max()}")
    print(f"  Arrivals/hour: mean={hourly['arrivals'].mean():.1f}, "
          f"max={hourly['arrivals'].max()}")

    null_lags = hourly[['lag_1', 'lag_24', 'lag_168']].isna().sum()
    print(f"  Lag nulls (expected at start of series): {null_lags.to_dict()}")

    return hourly


# ============================================================================
# 9. DAILY AGGREGATION (Thursday PM)
# ============================================================================

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (station_name, date) → daily arrival counts for Poisson input."""
    print(f"\n{'=' * 70}")
    print("DAILY AGGREGATION")
    print(f"{'=' * 70}")

    station_col = 'station_name' if 'station_name' in df.columns else 'station_id'

    if 'date' not in df.columns and 'start_time' in df.columns:
        df['date'] = pd.to_datetime(df['start_time'], errors='coerce').dt.date
    df['date_dt'] = pd.to_datetime(df['date']) if df['date'].dtype == object else df['date']

    agg_spec = {
        'arrivals': ('start_time', 'count'),
        'total_energy': ('energy_kwh', 'sum'),
        'mean_duration': ('charging_duration_min', 'mean'),
    }
    if 'is_abnormal' in df.columns:
        agg_spec['fault_count'] = ('is_abnormal', 'sum')
    if 'temperature' in df.columns:
        agg_spec['mean_temperature'] = ('temperature', 'mean')
    if 'precipitation' in df.columns:
        agg_spec['total_precipitation'] = ('precipitation', 'max')

    daily = df.groupby([station_col, 'date_dt']).agg(**agg_spec).reset_index()

    if 'fault_count' not in daily.columns:
        daily['fault_count'] = 0
    if 'mean_temperature' not in daily.columns:
        daily['mean_temperature'] = np.nan
    if 'total_precipitation' not in daily.columns:
        daily['total_precipitation'] = np.nan

    daily['day_of_week'] = daily['date_dt'].dt.dayofweek
    daily['month'] = daily['date_dt'].dt.month
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)

    print(f"  Daily table: {daily.shape[0]:,} rows × {daily.shape[1]} cols")
    print(f"  Arrivals/day/station: mean={daily['arrivals'].mean():.1f}, "
          f"max={daily['arrivals'].max()}")

    # Also compute total daily (all stations pooled) for reference
    daily_total = daily.groupby('date_dt')['arrivals'].sum()
    print(f"  Total arrivals/day: mean={daily_total.mean():.0f}, "
          f"max={daily_total.max()}")

    return daily


# ============================================================================
# 10. INTER-ARRIVAL TIMES (Thursday PM)
# ============================================================================

def build_iat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-arrival times per station:
    sort by start_time, compute consecutive differences in minutes.
    """
    print(f"\n{'=' * 70}")
    print("INTER-ARRIVAL TIMES")
    print(f"{'=' * 70}")

    station_col = 'station_name' if 'station_name' in df.columns else 'station_id'

    sorted_df = df.sort_values([station_col, 'start_time']).copy()
    sorted_df['iat_min'] = (
        sorted_df.groupby(station_col)['start_time']
        .diff()
        .dt.total_seconds() / 60.0
    )

    # Keep only valid IATs (drop first per station = NaN)
    for col, dt_attr in [('hour_of_day', 'hour'), ('day_of_week', 'dayofweek')]:
        if col not in sorted_df.columns:
            sorted_df[col] = getattr(sorted_df['start_time'].dt, dt_attr)
    if 'is_weekend' not in sorted_df.columns:
        sorted_df['is_weekend'] = sorted_df['day_of_week'].isin([5, 6]).astype(int)

    iat = sorted_df[
        [station_col, 'start_time', 'iat_min', 'hour_of_day', 'day_of_week', 'is_weekend']
    ].dropna(subset=['iat_min']).copy()

    # Add date for grouping
    iat['date'] = iat['start_time'].dt.date

    print(f"  IAT records: {len(iat):,}")
    print(f"  IAT (minutes): mean={iat['iat_min'].mean():.2f}, "
          f"median={iat['iat_min'].median():.2f}, "
          f"std={iat['iat_min'].std():.2f}")
    print(f"  Dispersion index (Var/Mean): "
          f"{iat['iat_min'].var() / iat['iat_min'].mean():.2f}")

    # Per-station summary
    print(f"\n  Per-station IAT summary:")
    for station, sub in iat.groupby(station_col):
        lam = 1.0 / sub['iat_min'].mean() if sub['iat_min'].mean() > 0 else 0
        print(f"    {str(station):45s}  n={len(sub):>6,}  "
              f"mean={sub['iat_min'].mean():>7.1f} min  "
              f"λ={lam:.4f}/min ({lam*60:.1f}/hr)")

    return iat


# ============================================================================
# 11. ACN DATA LOADING (Friday PM)
# ============================================================================

def load_acn() -> pd.DataFrame | None:
    """
    Load ACN-Data JPL (acndata_sessions.json).
    Returns None if file not found (non-blocking).
    """
    print(f"\n{'=' * 70}")
    print("ACN DATA LOADING")
    print(f"{'=' * 70}")

    # Check common locations
    acn_candidates = [
        ACN_RAW_DIR / "acndata_sessions.json",
        PROJECT_ROOT / "acndata_sessions.json",
        ACN_RAW_DIR / "acn_jpl.json",
    ]

    acn_path = None
    for p in acn_candidates:
        if p.exists():
            acn_path = p
            break

    if acn_path is None:
        print(f"  ⚠ ACN data not found. Looked in:")
        for p in acn_candidates:
            print(f"    {p}")
        print(f"  → Download from https://ev.caltech.edu/dataset")
        print(f"  → Place as {acn_candidates[0]}")
        print(f"  → This is non-blocking; ACN processing can happen in Week 2.")
        return None

    print(f"  Loading: {acn_path}")
    try:
        with open(acn_path, 'r') as f:
            raw = json.load(f)

        # ACN JSON structure: {"_items": [...]} or list of sessions
        if isinstance(raw, dict) and '_items' in raw:
            sessions = raw['_items']
        elif isinstance(raw, list):
            sessions = raw
        else:
            print(f"  ⚠ Unexpected JSON structure: {type(raw)}, keys={list(raw.keys())[:5]}")
            return None

        acn = pd.DataFrame(sessions)
        print(f"  [OK] ACN loaded: {acn.shape[0]:,} rows × {acn.shape[1]} cols")

        # Parse timestamps
        for col in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
            if col in acn.columns:
                acn[col] = pd.to_datetime(acn[col], errors='coerce')

        # Compute duration
        if 'connectionTime' in acn.columns and 'disconnectTime' in acn.columns:
            acn['duration_min'] = (
                (acn['disconnectTime'] - acn['connectionTime'])
                .dt.total_seconds() / 60.0
            )

        # Compute charging rate
        if 'kWhDelivered' in acn.columns and 'duration_min' in acn.columns:
            dur_hr = acn['duration_min'] / 60.0
            acn['effective_rate_kw'] = acn['kWhDelivered'] / dur_hr.replace(0, np.nan)

        # Hour and day features
        if 'connectionTime' in acn.columns:
            acn['hour_of_day'] = acn['connectionTime'].dt.hour
            acn['day_of_week'] = acn['connectionTime'].dt.dayofweek
            acn['is_weekend'] = acn['day_of_week'].isin([5, 6]).astype(int)

        # Save
        actual_path = save_table(acn, ACN_OUT_PATH)
        print(f"  [OK] Saved: {actual_path}")

        return acn

    except Exception as e:
        print(f"  ⚠ Error loading ACN: {e}")
        return None


# ============================================================================
# 12. FINAL VALIDATION (Friday AM)
# ============================================================================

def final_validation(df: pd.DataFrame, hourly: pd.DataFrame,
                     daily: pd.DataFrame, iat: pd.DataFrame) -> list:
    """Run final validation checks.  Returns list of issues."""
    print(f"\n{'=' * 70}")
    print("FINAL VALIDATION")
    print(f"{'=' * 70}")

    issues = []

    # No NaTs in critical timestamp fields
    for col in ['start_time', 'end_time']:
        if col in df.columns:
            n_nat = df[col].isna().sum()
            status = "✓" if n_nat == 0 else "⚠"
            print(f"  {status} {col} NaTs: {n_nat:,}")
            if n_nat > 0:
                issues.append(f"{col} has {n_nat} NaTs")

    # No negative durations remaining (they're flagged, not removed)
    if 'charging_duration_min' in df.columns:
        n_neg = (df['charging_duration_min'] < 0).sum()
        status = "✓" if n_neg == 0 else "⚠"
        print(f"  {status} Negative durations: {n_neg:,}")
        if n_neg > 0:
            issues.append(f"{n_neg} negative durations")

    # All derived columns present
    required_cols = ['charging_duration_min', 'effective_rate_kw', 'price_per_kwh',
                     'user_stop_proxy', 'tou_tier', 'charger_type',
                     'hour_of_day', 'day_of_week', 'is_weekend', 'is_rainy',
                     'station_name']
    for col in required_cols:
        present = col in df.columns
        status = "✓" if present else "⚠"
        print(f"  {status} Column present: {col}")
        if not present:
            issues.append(f"Missing column: {col}")

    # Hourly table sanity
    print(f"  ✓ Hourly table: {hourly.shape[0]:,} rows")
    print(f"  ✓ Daily table: {daily.shape[0]:,} rows")
    print(f"  ✓ IAT table: {iat.shape[0]:,} rows")

    # Summary
    if issues:
        print(f"\n  ⚠ {len(issues)} issue(s) found:")
        for i in issues:
            print(f"    - {i}")
    else:
        print(f"\n  ✓ All validation checks passed.")

    return issues


# ============================================================================
# 13. QUALITY REPORT (Friday PM)
# ============================================================================

def write_quality_report(df: pd.DataFrame, hourly: pd.DataFrame,
                         daily: pd.DataFrame, iat: pd.DataFrame,
                         issues: list) -> None:
    """Write the comprehensive data_quality_report.md."""

    N = len(df)
    lines = []

    lines.append("# Data Quality Report — Jiaxing EV Charging Dataset")
    lines.append(f"\n*Generated by week1_wrapup.py*\n")

    # --- Dataset overview ---
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- **Source:** Figshare DOI 10.6084/m9.figshare.28182251")
    lines.append(f"- **Sessions:** {N:,}")
    lines.append(f"- **Time range:** {df['start_time'].min()} → {df['start_time'].max()}")
    lines.append(f"- **Districts:** {df['district'].nunique()} "
                 f"({', '.join(df['district'].unique())})")
    if 'station_name' in df.columns:
        lines.append(f"- **Stations:** {df['station_name'].nunique()}")
    lines.append(f"- **Charger posts:** {df['station_id'].nunique()}")
    lines.append(f"- **Unique users:** {df['user_id'].nunique():,}")

    # --- Cleaning decisions ---
    lines.append("\n## 2. Cleaning Decisions\n")
    lines.append("| Decision | Count | % | Rationale |")
    lines.append("|----------|------:|--:|-----------|")
    lines.append(f"| Rows preserved (no removal) | {N:,} | 100% | "
                 f"Flag-don't-drop policy |")

    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    for col in flag_cols:
        n = int(df[col].sum())
        pct = 100 * n / N
        lines.append(f"| Flagged: {col} | {n:,} | {pct:.1f}% | "
                     f"Preserved with flag |")

    if 'tou_electricity_price' in df.columns:
        n_tou_missing = int(df['tou_electricity_price'].isna().sum())
        lines.append(f"| TOU price missing | {n_tou_missing:,} rows | {100*n_tou_missing/N:.1f}% | "
                     f"Missing hour mapping in TOU schedule |")

    # --- Station grouping ---
    lines.append("\n## 3. Station Grouping\n")
    lines.append(f"`station_id` (Charging Post ID) is charger-level: "
                 f"{df['station_id'].nunique()} unique values.")
    lines.append(f"`station_name` = district + location_info → "
                 f"{df['station_name'].nunique() if 'station_name' in df.columns else '?'} "
                 f"stations.\n")

    if 'station_name' in df.columns:
        lines.append("| Station | Sessions | % |")
        lines.append("|---------|--------:|--:|")
        for name, count in df['station_name'].value_counts().items():
            lines.append(f"| {name} | {count:,} | {100*count/N:.1f}% |")

    # --- Fault rate ---
    lines.append("\n## 4. Fault Rate Analysis\n")
    lines.append(f"- `is_abnormal` rate: **{df['is_abnormal'].mean():.2%}**")
    lines.append(f"- Expected from sample: ~30.9%")
    lines.append(f"- The fault indicator columns (`is_nonsystem_fault`, `is_system_issue`, "
                 f"`is_ev_fault`, `is_other_fault`) are a **decomposition** of `is_abnormal`, "
                 f"not additional faults.")
    lines.append(f"- **Decision:** Use 18.58% as the fault rate for Phase 2 simulation.")

    # --- User-stop proxy ---
    if 'user_stop_proxy' in df.columns:
        lines.append("\n## 5. User-Stop Proxy (Termination Classification)\n")
        lines.append("*'user_stop_proxy' is an observational upper bound on scheduling "
                     "flexibility, not confirmed willingness-to-delay.*\n")
        lines.append("| Category | Count | % |")
        lines.append("|----------|------:|--:|")
        for tier, count in df['user_stop_proxy'].value_counts().items():
            lines.append(f"| {tier} | {count:,} | {100*count/N:.1f}% |")
    elif 'flexibility_tier' in df.columns:
        # Backward compat if old parquet loaded
        lines.append("\n## 5. Flexibility Tiers (LEGACY)\n")
        lines.append("| Tier | Count | % |")
        lines.append("|------|------:|--:|")
        for tier, count in df['flexibility_tier'].value_counts().items():
            lines.append(f"| {tier} | {count:,} | {100*count/N:.1f}% |")

    # --- TOU ---
    if 'tou_tier' in df.columns:
        lines.append("\n## 6. TOU Tier Distribution\n")
        lines.append("| Tier | Count | % | Mean Price (¥/kWh) |")
        lines.append("|------|------:|--:|---:|")
        for tier in ['Valley', 'Flat', 'Peak', 'Super-peak']:
            sub = df[df['tou_tier'] == tier]
            if len(sub) > 0:
                mp = sub['tou_electricity_price'].mean() if 'tou_electricity_price' in sub.columns else np.nan
                lines.append(f"| {tier} | {len(sub):,} | "
                             f"{100*len(sub)/N:.1f}% | {mp:.4f} |")

    # --- Charger type ---
    if 'charger_type' in df.columns:
        lines.append("\n## 7. Charger Type Distribution\n")
        lines.append("| Type | Count | % |")
        lines.append("|------|------:|--:|")
        for ctype, count in df['charger_type'].value_counts().items():
            lines.append(f"| {ctype} | {count:,} | {100*count/N:.1f}% |")

    # --- Aggregation tables ---
    lines.append("\n## 8. Output Files\n")
    lines.append(f"- `jiaxing_clean.(parquet|csv)`: {df.shape[0]:,} rows × {df.shape[1]} cols (session-level)")
    lines.append(f"- `jiaxing_hourly.(parquet|csv)`: {hourly.shape[0]:,} rows × {hourly.shape[1]} cols (LSTM input)")
    lines.append(f"- `jiaxing_daily.(parquet|csv)`: {daily.shape[0]:,} rows × {daily.shape[1]} cols (Poisson input)")
    lines.append(f"- `jiaxing_iat.(parquet|csv)`: {iat.shape[0]:,} rows × {iat.shape[1]} cols (inter-arrival times)")

    # --- Validation issues ---
    lines.append("\n## 9. Validation Issues\n")
    if issues:
        for i in issues:
            lines.append(f"- ⚠ {i}")
    else:
        lines.append("- ✓ All checks passed.")

    # Write
    report = '\n'.join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[OK] Quality report: {REPORT_PATH}")


# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    configure_console_output()
    df = load_clean()

    # Wednesday tasks
    df = create_station_grouping(df)
    df = analyze_faults(df)
    df = create_flexibility_tier(df)
    df = derive_tou_tier(df)
    df = derive_charger_type(df)
    df = add_weather_flags(df)

    # Thursday tasks
    hourly = build_hourly(df)
    daily = build_daily(df)
    iat = build_iat(df)

    # Friday tasks
    issues = final_validation(df, hourly, daily, iat)

    # Save all 4 parquet files
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUT FILES")
    print(f"{'=' * 70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_saved = save_table(df, FINAL_PATH)
    print(f"  [OK] {final_saved.name}: {df.shape}")

    hourly_saved = save_table(hourly, HOURLY_PATH)
    print(f"  [OK] {hourly_saved.name}: {hourly.shape}")

    daily_saved = save_table(daily, DAILY_PATH)
    print(f"  [OK] {daily_saved.name}: {daily.shape}")

    iat_saved = save_table(iat, IAT_PATH)
    print(f"  [OK] {iat_saved.name}: {iat.shape}")

    # ACN (non-blocking)
    load_acn()

    # Quality report
    write_quality_report(df, hourly, daily, iat, issues)

    # Checkpoint summary
    print(f"\n{'=' * 70}")
    print("WEEK 1 CHECKPOINT")
    print(f"{'=' * 70}")
    print(f"  ✓ {final_saved.name}: {df.shape[0]:,} rows, all features")
    sn = df['station_name'].nunique() if 'station_name' in df.columns else '?'
    print(f"  {'✓' if sn == 13 else '⚠'} Stations: {sn} "
          f"(expected 13)")
    print(f"  ✓ Fault rate: {df['is_abnormal'].mean():.1%} "
          f"(documented, differs from sample)")
    if 'tou_tier' in df.columns:
        print(f"  ✓ TOU tiers: {df['tou_tier'].nunique()} tiers defined")
    if 'charger_type' in df.columns:
        print(f"  ✓ Charger types: "
              f"{df['charger_type'].value_counts().to_dict()}")
    print(f"  ✓ data_quality_report.md written")
    acn_loaded = ACN_OUT_PATH.exists() or ACN_OUT_PATH.with_suffix('.csv').exists()
    print(f"  {'✓' if acn_loaded else '⚠'} ACN data: "
          f"{'loaded' if acn_loaded else 'not found (do in Week 2)'}")

    print(f"\n{'=' * 70}")
    print("WEEK 1 COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
