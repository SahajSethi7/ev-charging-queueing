"""
Week 1 — Phase 0a: Data Ingestion and Cleaning for Jiaxing EV Charging Dataset
================================================================================

Purpose:
    Load the three raw Jiaxing CSVs (Charging, Weather, TOU Price), validate,
    parse timestamps, merge, compute derived features, flag anomalies, and
    produce a clean Parquet file for all downstream analysis.

Data source:
    Figshare DOI: 10.6084/m9.figshare.28182251
    Expected: ~441K sessions, 13 stations, 3 districts, ~2 years.

Working directory:
    Repository root

Input files (in Data/):
    Charging_Data.csv     — 441K charging sessions
    Weather_Data.csv      — daily weather by district
    Time-of-use_Price.csv — TOU electricity price schedule

Output (created by this script):
    Data\\jiaxing_clean.parquet

Usage:
    cd path/to/ev-charging-queueing
    python Code/ingest_jiaxing.py

    Or with overrides:
    python Code/ingest_jiaxing.py --charging path/to/Charging_Data.csv
"""

import argparse
import json
import sys
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR


# ============================================================================
# 0. ASSUMPTIONS (verify manually after first run)
# ============================================================================
"""
A1. Charging_Data.csv: ~441,077 rows.  Columns likely include (from paper
    Table 4): Order number, User ID, Charging station name, District,
    Order creation time, Start charging time, End charging time, Payment time,
    Transaction power (kWh), Total amount (Yuan), Electricity fee, Service fee,
    Electricity unit price, Service unit price, end_cause, is_abnormal,
    is_full_charge.

A2. Weather_Data.csv: daily weather per district (temperature, humidity,
    precipitation, possibly wind).  Key = (district, date).
    This is a SEPARATE file — it must be merged onto charging sessions
    by district + date.

A3. Time-of-use_Price.csv: maps hour-of-day → TOU tier and price.
    From paper Table 5, Jiaxing TOU has Valley / Flat / Peak / Super-peak.
    This is a LOOKUP table, not per-session — merge by hour of day.

A4. Column names may be in English (Figshare export) or Chinese.
    The script auto-detects and maps either way.

A5. Timestamps: second-level precision, likely YYYY-MM-DD HH:MM:SS.
    May have trailing whitespace or tab characters from CSV export.

A6. ~30.9% abnormal sessions (from sample analysis).
    Zero-energy sessions expected.
    Some negative/zero durations from timestamp recording errors.

A7. Encoding: UTF-8 or GB18030 (script tries both).
"""


# ============================================================================
# 1. PATHS AND CONFIGURATION
# ============================================================================

# --- Input files ---
DEFAULT_CHARGING_PATH = DATA_DIR / "Charging_Data.csv"
DEFAULT_WEATHER_PATH  = DATA_DIR / "Weather_Data.csv"
DEFAULT_TOU_PATH      = DATA_DIR / "Time-of-use_Price.csv"

# --- Output ---
OUTPUT_DIR    = DATA_DIR
OUTPUT_PATH   = OUTPUT_DIR / "jiaxing_clean.parquet"
REPORT_PATH   = RESULTS_DIR / "data_quality_report.txt"

# --- Expected counts ---
EXPECTED_ROW_COUNT_RANGE = (400_000, 500_000)
EXPECTED_STATION_COUNT   = 13
EXPECTED_DISTRICT_COUNT  = 3

# --- Anomaly thresholds ---
MIN_DURATION_MIN       = 0        # flag if duration <= 0
MAX_DURATION_MIN       = 1440     # flag if duration > 24h
MIN_EFFECTIVE_RATE_KW  = 0.1      # flag if < 0.1 kW
MAX_EFFECTIVE_RATE_KW  = 200      # flag if > 200 kW
ZERO_ENERGY_THRESHOLD  = 0.01     # kWh; below = zero-energy session


# ============================================================================
# 2. COLUMN NAME NORMALIZATION
# ============================================================================

# This map covers BOTH possible English and Chinese column names from Figshare.
# After first run, inspect the raw columns printed by Step 0 (bootstrap) and
# add any missing mappings here.

RENAME_MAP = {
    # --- Identifiers ---
    'Order number':              'order_id',
    'order number':              'order_id',
    'Order Number':              'order_id',
    'User ID':                   'user_id',
    'user ID':                   'user_id',
    'user_id':                   'user_id',
    'UserID':                    'user_id',
    'Charging Post ID':          'station_id',
    # --- Location ---
    'Charging station name':     'station_name',
    'charging station name':     'station_name',
    'Station Name':              'station_name',
    'station_name':              'station_name',
    'Location Information':      'location_info',
    'District':                  'district',
    'district':                  'district',
    'District Name':             'district',
    # --- Timestamps ---
    'Order creation time':       'order_created_time',
    'order creation time':       'order_created_time',
    'Start charging time':       'start_time',
    'start charging time':       'start_time',
    'Start Time':                'start_time',
    'End charging time':         'end_time',
    'end charging time':         'end_time',
    'End Time':                  'end_time',
    'Payment time':              'payment_time',
    'payment time':              'payment_time',
    # --- Energy ---
    'Transaction power':         'energy_kwh',      # paper calls this kWh
    'transaction power':         'energy_kwh',
    'Transaction Power':         'energy_kwh',
    'Transaction power/kwh':     'energy_kwh',
    'Charging capacity':         'energy_kwh',
    # --- Pricing ---
    'Total amount':              'total_amount',
    'total amount':              'total_amount',
    'Transaction Amount/Yuan':   'total_amount',
    'Electricity fee':           'elec_fee',
    'electricity fee':           'elec_fee',
    'Electricity cost/Yuan':     'elec_fee',
    'Service fee':               'service_fee',
    'service fee':               'service_fee',
    'Service charge/Yuan':       'service_fee',
    'Actual Payment/Yuan':       'actual_payment',
    'Electricity unit price':    'elec_unit_price',
    'electricity unit price':    'elec_unit_price',
    'Service unit price':        'service_unit_price',
    'service unit price':        'service_unit_price',
    # --- Session metadata ---
    'is_abnormal':               'is_abnormal',
    'Is_abnormal':               'is_abnormal',
    'end_cause':                 'end_cause',
    'End_cause':                 'end_cause',
    'End cause':                 'end_cause',
    'end cause':                 'end_cause',
    'is_full_charge':            'is_full_charge',
    'Is_full_charge':            'is_full_charge',
    'is_user_stop':              'is_user_stop',
    'is_nonsystem_fault':        'is_nonsystem_fault',
    'is_system_issue':           'is_system_issue',
    'is_EV_fault':               'is_ev_fault',
    'is_other_fault':            'is_other_fault',
    # --- Weather (if embedded in charging file — unlikely but safe) ---
    'Temperature':               'temperature',
    'temperature':               'temperature',
    'Temperature/℃':             'temperature',
    'Temperature(℃)':            'temperature',
    'Average temperature/℃':     'temperature',
    'Temperature(?)':            'temperature',
    'Temperature/?':             'temperature',
    'Humidity':                  'humidity',
    'humidity':                  'humidity',
    'Relative humidity':         'humidity',
    'Relative Humidity(%)':      'humidity',
    'Relative humidity/%':       'humidity',
    'Precipitation':             'precipitation',
    'precipitation':             'precipitation',
    'Precipitation(mm)':         'precipitation',
    'Precipitation/mm':          'precipitation',
    'Date':                      'date',
    'Weather':                   'weather_desc',
    'weather':                   'weather_desc',
    # --- TOU (if embedded) ---
    'Time Period':               'time_period',
    'Electricity Price(Yuan/kWh)': 'tou_electricity_price',
    'TOU tier':                  'tou_tier',
    'tou_tier':                  'tou_tier',
}


def normalize_columns(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Normalize column names using RENAME_MAP.  Unmapped cols → snake_case."""
    renamed = {}
    unmapped = []
    for raw_col in df.columns:
        stripped = raw_col.strip()
        if stripped in RENAME_MAP:
            renamed[raw_col] = RENAME_MAP[stripped]
        else:
            clean = re.sub(r'\W+', '_', stripped.lower()).strip('_')
            if not clean:
                clean = f'col_{len(renamed)}'
            renamed[raw_col] = clean
            unmapped.append(f"  '{stripped}' → '{clean}'  (NOT in RENAME_MAP)")

    df = df.rename(columns=renamed)

    if unmapped:
        print(f"\n  ⚠ [{label}] Unmapped columns (review and add to RENAME_MAP):")
        for u in unmapped:
            print(f"    {u}")

    return df


# ============================================================================
# 2b. COMPATIBILITY HELPERS
# ============================================================================

def configure_console_output() -> None:
    """
    Prevent UnicodeEncodeError on Windows consoles (e.g., cp1252) by replacing
    characters that cannot be encoded in the active terminal encoding.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                # Non-fatal: keep default stream behavior if reconfigure fails.
                pass


def detect_parquet_engine() -> Optional[str]:
    """Return the first available parquet engine."""
    if find_spec('pyarrow') is not None:
        return 'pyarrow'
    if find_spec('fastparquet') is not None:
        return 'fastparquet'
    return None


def parse_hour_value(value) -> float:
    """Parse hour representations like 8, '08', or '08:00' to 0-23."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value) % 24
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        if float(value).is_integer():
            return int(value) % 24
        return np.nan

    text = str(value).strip()
    if not text:
        return np.nan
    if text.isdigit():
        return int(text) % 24

    m = re.match(r'^(\d{1,2})\s*:', text)
    if m:
        return int(m.group(1)) % 24
    return np.nan


def expand_tou_hour_ranges(tou: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """
    Expand TOU time ranges like '00:00-08:00' to one row per hour.
    Keeps the original non-period columns as values for each expanded hour.
    """
    rows = []
    value_cols = [c for c in tou.columns if c != period_col]
    pattern = re.compile(r'^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$')

    for _, record in tou.iterrows():
        period_text = str(record[period_col]).strip()
        m = pattern.match(period_text)
        if not m:
            continue

        start_h, start_m, end_h, end_m = map(int, m.groups())
        # Most TOU files use hour-aligned ranges; if minutes are present but not
        # aligned, we still map based on the containing hours.
        start_h %= 24
        end_h %= 24
        if end_h > start_h:
            hours = list(range(start_h, end_h))
        elif end_h < start_h:
            hours = list(range(start_h, 24)) + list(range(0, end_h))
        else:
            # 00:00-00:00 style range means all-day in many schedules.
            hours = list(range(24))

        for h in hours:
            row = {'hour_of_day': h}
            for c in value_cols:
                row[c] = record[c]
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['hour_of_day', *value_cols])

    expanded = pd.DataFrame(rows)
    expanded = expanded.drop_duplicates(subset=['hour_of_day'], keep='last')
    return expanded.sort_values('hour_of_day').reset_index(drop=True)


# Ensure any later print() calls won't fail on cp1252-like terminals.
configure_console_output()


# ============================================================================
# 3. LOADING UTILITIES
# ============================================================================

def load_csv(filepath: Path, label: str) -> pd.DataFrame:
    """Load a CSV with encoding fallback."""
    if not filepath.exists():
        sys.exit(f"[FATAL] {label} not found: {filepath}")

    for enc in ['utf-8', 'gb18030', 'utf-8-sig', 'latin-1']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"[OK] {label}: loaded {df.shape[0]:,} rows × {df.shape[1]} cols  "
                  f"(encoding={enc})")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    sys.exit(f"[FATAL] Could not parse {label} with any encoding.")


# ============================================================================
# 4. STEP 0 — BOOTSTRAP: print raw schemas (run once, then refine)
# ============================================================================

def bootstrap_inspect(df: pd.DataFrame, label: str) -> None:
    """Print raw column names, dtypes, nulls, and sample values."""
    print(f"\n{'=' * 70}")
    print(f"BOOTSTRAP INSPECTION: {label}")
    print(f"{'=' * 70}")
    print(f"  Shape: {df.shape}")
    print(f"\n  {'Column':<45} {'Dtype':<12} {'Null%':>6}  Sample")
    print(f"  {'-'*45} {'-'*12} {'-'*6}  {'-'*30}")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        sample = str(df[col].dropna().iloc[0]) if df[col].notna().any() else 'ALL NULL'
        sample = sample[:30]
        print(f"  {col:<45} {str(df[col].dtype):<12} {null_pct:5.1f}%  {sample}")


# ============================================================================
# 5. SCHEMA VALIDATION (Charging Data)
# ============================================================================

def validate_charging(df: pd.DataFrame) -> list:
    """Validate the charging dataframe.  Returns list of warning strings."""
    warnings = []
    N = len(df)

    print(f"\n{'=' * 70}")
    print("SCHEMA VALIDATION: Charging Data")
    print(f"{'=' * 70}")

    # Row count
    lo, hi = EXPECTED_ROW_COUNT_RANGE
    print(f"  Rows: {N:,}")
    if not (lo <= N <= hi):
        msg = f"Row count {N:,} outside expected [{lo:,}, {hi:,}]"
        print(f"  ⚠ {msg}")
        warnings.append(msg)

    # Station count — dataset uses station_id (Charging Post ID), not station_name
    station_col = 'station_name' if 'station_name' in df.columns else \
                  'station_id' if 'station_id' in df.columns else None
    if station_col:
        ns = df[station_col].nunique()
        print(f"  Stations ({station_col}): {ns}")
        if ns != EXPECTED_STATION_COUNT:
            msg = f"Station count {ns} ≠ expected {EXPECTED_STATION_COUNT}"
            print(f"  ⚠ {msg}")
            warnings.append(msg)
    else:
        msg = "No station_name or station_id column found"
        print(f"  ⚠ {msg}")
        warnings.append(msg)

    # District count
    if 'district' in df.columns:
        nd = df['district'].nunique()
        print(f"  Districts: {nd}")
        if nd != EXPECTED_DISTRICT_COUNT:
            msg = f"District count {nd} ≠ expected {EXPECTED_DISTRICT_COUNT}"
            print(f"  ⚠ {msg}")
            warnings.append(msg)

    # Duplicate order IDs
    if 'order_id' in df.columns:
        n_dup = df['order_id'].duplicated().sum()
        print(f"  Duplicate order_ids: {n_dup:,}")
        if n_dup > 0:
            warnings.append(f"{n_dup:,} duplicate order IDs")

    print(f"  Warnings: {len(warnings)}")
    return warnings


# ============================================================================
# 6. TIMESTAMP PARSING
# ============================================================================

TIMESTAMP_COLS = ['order_created_time', 'start_time', 'end_time', 'payment_time']


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all timestamp columns, stripping whitespace/tabs."""
    print(f"\n{'=' * 70}")
    print("TIMESTAMP PARSING")
    print(f"{'=' * 70}")

    for col in TIMESTAMP_COLS:
        if col not in df.columns:
            print(f"  ⚠ '{col}' not found — skipping")
            continue

        raw = df[col].copy()
        n_null_before = raw.isna().sum()

        # Strip whitespace and tab chars (common in Chinese CSV exports)
        if raw.dtype == object:
            raw = raw.str.strip().str.replace(r'[\t\r]+', '', regex=True)

        parsed = pd.to_datetime(raw, errors='coerce')
        n_failed = parsed.isna().sum() - n_null_before

        df[col] = parsed

        # Summary
        ts_min = parsed.min()
        ts_max = parsed.max()
        print(f"  {col:<30}  OK  "
              f"null_before={n_null_before:,}  "
              f"parse_fail={n_failed:,}  "
              f"range=[{ts_min}, {ts_max}]")

        if n_failed > 0:
            bad = raw[parsed.isna() & raw.notna()].head(5).tolist()
            print(f"    Sample unparsed: {bad}")

    return df


# ============================================================================
# 7. WEATHER MERGE
# ============================================================================

def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather data onto charging sessions by (district, date).
    Weather_Data.csv is daily per district.
    """
    print(f"\n{'=' * 70}")
    print("WEATHER MERGE")
    print(f"{'=' * 70}")

    # Normalize weather columns
    weather = normalize_columns(weather, label="Weather")
    bootstrap_inspect(weather, "Weather (after normalize)")

    # --- Drop embedded weather columns from charging data BEFORE merge ---
    # Charging_Data.csv already contains per-session weather (temperature,
    # humidity, precipitation), but these are the SAME daily district-level
    # values from Weather_Data.csv duplicated onto each session.  Merging
    # without dropping creates _x/_y suffix pairs.
    # Strategy: drop the charging-embedded copies; keep Weather_Data as the
    # single canonical source.
    weather_cols_in_charging = [c for c in ['temperature', 'humidity', 'precipitation']
                                if c in df.columns]
    if weather_cols_in_charging:
        print(f"  Dropping embedded weather columns from charging data to avoid "
              f"duplicates: {weather_cols_in_charging}")
        df = df.drop(columns=weather_cols_in_charging)

    # We need a date column in weather and in charging.
    # Weather likely has a 'date' column; charging needs one derived from start_time.
    if 'start_time' not in df.columns:
        print("  ⚠ No start_time in charging data — cannot merge weather.")
        return df

    df['date'] = df['start_time'].dt.date

    # Auto-detect the date column in weather
    weather_date_candidates = [c for c in weather.columns
                               if 'date' in c.lower() or 'time' in c.lower()
                               or 'day' in c.lower()]
    if not weather_date_candidates:
        print("  ⚠ No date column found in weather data. Columns:")
        print(f"    {list(weather.columns)}")
        print("  → Skipping weather merge.  ADD MAPPING MANUALLY.")
        return df

    weather_date_col = weather_date_candidates[0]
    weather[weather_date_col] = pd.to_datetime(
        weather[weather_date_col].astype(str).str.strip(), errors='coerce'
    ).dt.date

    # Auto-detect district column in weather
    weather_district_candidates = [c for c in weather.columns
                                   if 'district' in c.lower()
                                   or 'region' in c.lower()
                                   or 'area' in c.lower()]

    if weather_district_candidates:
        # Merge on (district, date)
        w_dist_col = weather_district_candidates[0]
        weather_renamed = weather.rename(columns={
            weather_date_col: 'date',
            w_dist_col: 'district'
        })
        dup_mask = weather_renamed.duplicated(subset=['district', 'date'], keep=False)
        if dup_mask.any():
            n_dup_rows = int(dup_mask.sum())
            n_dup_keys = int(weather_renamed.loc[dup_mask, ['district', 'date']]
                             .drop_duplicates().shape[0])
            print(f"  ⚠ Weather has {n_dup_rows:,} duplicate rows across "
                  f"{n_dup_keys:,} (district, date) keys; keeping last per key")
            weather_renamed = weather_renamed.drop_duplicates(
                subset=['district', 'date'], keep='last'
            )

        n_before = len(df)
        df['date'] = df['start_time'].dt.date
        df = df.merge(weather_renamed, on=['district', 'date'], how='left')
        n_after = len(df)
        weather_probe_cols = ['temperature', 'humidity', 'precipitation']
        probe_col = next((c for c in weather_probe_cols if c in df.columns), None)
        n_unmatched = df[probe_col].isna().sum() if probe_col is not None else -1
        print(f"  Merged on (district, date): {n_before:,} → {n_after:,} rows")
        print(f"  Unmatched (no weather): {n_unmatched:,}")
        if n_after != n_before:
            print(f"  ⚠ Row count changed!  Possible duplicate (district,date) in weather.")
    else:
        # Merge on date only (weather may be city-wide, not per-district)
        weather_renamed = weather.rename(columns={weather_date_col: 'date'})
        dup_mask = weather_renamed.duplicated(subset=['date'], keep=False)
        if dup_mask.any():
            n_dup_rows = int(dup_mask.sum())
            n_dup_keys = int(weather_renamed.loc[dup_mask, ['date']].drop_duplicates().shape[0])
            print(f"  ⚠ Weather has {n_dup_rows:,} duplicate rows across {n_dup_keys:,} "
                  "date keys; keeping last per date")
            weather_renamed = weather_renamed.drop_duplicates(subset=['date'], keep='last')

        n_before = len(df)
        df = df.merge(weather_renamed, on='date', how='left')
        n_after = len(df)
        print(f"  Merged on (date) only: {n_before:,} → {n_after:,} rows")
        if n_after != n_before:
            print(f"  ⚠ Row count changed!  Possible duplicate dates in weather.")

    return df


# ============================================================================
# 8. TOU PRICE MERGE
# ============================================================================

def merge_tou(df: pd.DataFrame, tou: pd.DataFrame) -> pd.DataFrame:
    """
    Merge TOU price schedule.
    TOU_Price.csv is a lookup: hour → tier, price.
    Merge by hour_of_day derived from start_time.
    """
    print(f"\n{'=' * 70}")
    print("TOU PRICE MERGE")
    print(f"{'=' * 70}")

    tou = normalize_columns(tou, label="TOU")
    bootstrap_inspect(tou, "TOU (after normalize)")

    if 'start_time' not in df.columns:
        print("  ⚠ No start_time — cannot derive hour for TOU merge.")
        return df

    # Derive hour from start_time
    df['hour_of_day'] = df['start_time'].dt.hour

    # TOU file likely maps hour → tier/price.  Auto-detect the hour column.
    tou_hour_candidates = [c for c in tou.columns
                           if 'hour' in c.lower() or 'time' in c.lower()
                           or 'period' in c.lower()]

    if not tou_hour_candidates:
        print("  ⚠ No hour/time column in TOU data. Columns:")
        print(f"    {list(tou.columns)}")
        print("  → Skipping TOU merge.  Manual intervention needed.")
        print("  → If TOU is a simple table (e.g. 6 rows for time ranges),")
        print("    you'll need to expand it to 24 hourly rows.  See note below.")
        return df

    tou_hour_col = tou_hour_candidates[0]

    hour_text = tou[tou_hour_col].astype(str).str.strip()
    range_like = hour_text.str.match(r'^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$').mean() >= 0.5

    if range_like:
        tou_renamed = expand_tou_hour_ranges(tou, tou_hour_col)
        if tou_renamed.empty:
            print("  ⚠ TOU appears to use time ranges but they could not be parsed.")
            print("  → Skipping TOU merge. Inspect TOU_Price format manually.")
            return df
        print(f"  Expanded TOU ranges to {tou_renamed['hour_of_day'].nunique()} hourly rows")
    else:
        tou_renamed = tou.rename(columns={tou_hour_col: 'hour_of_day'})
        tou_renamed['hour_of_day'] = tou_renamed['hour_of_day'].map(parse_hour_value)

        bad_hours = tou_renamed['hour_of_day'].isna().sum()
        if bad_hours:
            print(f"  ⚠ Dropping {bad_hours:,} TOU rows with unparseable hour values")
            tou_renamed = tou_renamed[tou_renamed['hour_of_day'].notna()].copy()

        if tou_renamed.empty:
            print("  ⚠ No valid TOU hour mappings after parsing. Skipping merge.")
            return df

        tou_renamed['hour_of_day'] = tou_renamed['hour_of_day'].astype(int)
        if tou_renamed['hour_of_day'].duplicated().any():
            print("  ⚠ Duplicate hour mappings found in TOU; keeping the last row per hour")
            tou_renamed = tou_renamed.drop_duplicates(subset=['hour_of_day'], keep='last')

    if tou_renamed['hour_of_day'].nunique() < 24:
        covered = set(tou_renamed['hour_of_day'].unique())
        missing = sorted(set(range(24)) - covered)
        print(f"  ⚠ TOU provides {len(covered)} unique hours (expected 24).")
        print(f"    Missing hours: {missing}")
        print(f"    This means sessions starting at hour(s) {missing} will have "
              f"null TOU price.")
        # Attempt to fill missing hours by inheriting from the previous hour
        if missing and len(covered) >= 20:
            for mh in missing:
                prev_h = (mh - 1) % 24
                if prev_h in covered:
                    fill_row = tou_renamed[tou_renamed['hour_of_day'] == prev_h].iloc[0].copy()
                    fill_row['hour_of_day'] = mh
                    tou_renamed = pd.concat([tou_renamed, fill_row.to_frame().T],
                                            ignore_index=True)
                    print(f"    → Filled hour {mh} with same price as hour {prev_h}")
            tou_renamed = tou_renamed.sort_values('hour_of_day').reset_index(drop=True)

    # Prefix TOU columns to avoid collision
    tou_cols_to_merge = [c for c in tou_renamed.columns if c != 'hour_of_day']
    tou_renamed = tou_renamed.rename(
        columns={c: f'tou_{c}' if not c.startswith('tou_') else c
                 for c in tou_cols_to_merge}
    )

    n_before = len(df)
    df = df.merge(tou_renamed, on='hour_of_day', how='left')
    n_after = len(df)
    print(f"  Merged on hour_of_day: {n_before:,} → {n_after:,} rows")

    return df


def expand_tou_ranges(tou: pd.DataFrame) -> pd.DataFrame:
    """
    HELPER: If your TOU file has time RANGES (e.g., "00:00-06:00", Valley, 0.35),
    expand it to one row per hour.

    This is a template — you must adapt it after inspecting the actual file.

    Example input (from paper Table 5):
        Period      | Hours              | Elec Price (Yuan/kWh) | Service
        Valley      | 22:00-08:00        | 0.3583                | 0.65
        Flat        | 08:00-11:00,       | 0.6283                | 0.65
                    | 13:00-17:00        |                       |
        Peak        | 11:00-13:00,       | 1.0583                | 0.65
                    | 17:00-22:00        |                       |

    You will likely need to create this mapping manually as a dict:
    """
    # EXAMPLE — adapt to your actual TOU structure:
    hour_to_tier = {}
    # Valley: 22:00-08:00
    for h in list(range(22, 24)) + list(range(0, 8)):
        hour_to_tier[h] = {'tou_tier': 'Valley', 'tou_elec_price': 0.3583}
    # Flat: 08:00-11:00, 13:00-17:00
    for h in list(range(8, 11)) + list(range(13, 17)):
        hour_to_tier[h] = {'tou_tier': 'Flat', 'tou_elec_price': 0.6283}
    # Peak: 11:00-13:00, 17:00-22:00
    for h in list(range(11, 13)) + list(range(17, 22)):
        hour_to_tier[h] = {'tou_tier': 'Peak', 'tou_elec_price': 1.0583}

    rows = [{'hour_of_day': h, **v} for h, v in sorted(hour_to_tier.items())]
    return pd.DataFrame(rows)


# ============================================================================
# 9. DERIVED FEATURES
# ============================================================================

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
      - charging_duration_min  = (end_time - start_time) in minutes
      - effective_rate_kw      = energy_kwh / (duration in hours)
      - price_per_kwh          = total_amount / energy_kwh
      - elec_price_per_kwh     = elec_fee / energy_kwh
      - service_price_per_kwh  = service_fee / energy_kwh
    Also validates fee decomposition: elec + service ≈ total.
    """
    print(f"\n{'=' * 70}")
    print("DERIVED FEATURES")
    print(f"{'=' * 70}")

    N = len(df)

    # Coerce numeric columns that are frequently exported as strings.
    for col in ['energy_kwh', 'total_amount', 'elec_fee', 'service_fee']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Duration ---
    if 'start_time' in df.columns and 'end_time' in df.columns:
        df['charging_duration_min'] = (
            (df['end_time'] - df['start_time']).dt.total_seconds() / 60.0
        )
        d = df['charging_duration_min']
        print(f"  charging_duration_min:  "
              f"mean={d.mean():.1f}  median={d.median():.1f}  "
              f"min={d.min():.1f}  max={d.max():.1f}")
    else:
        print("  ⚠ Cannot compute duration (missing start_time or end_time)")

    # --- Effective rate ---
    if 'energy_kwh' in df.columns and 'charging_duration_min' in df.columns:
        duration_hr = df['charging_duration_min'] / 60.0
        duration_hr_safe = duration_hr.replace(0, np.nan)
        df['effective_rate_kw'] = df['energy_kwh'] / duration_hr_safe
        r = df['effective_rate_kw']
        print(f"  effective_rate_kw:      "
              f"mean={r.mean():.1f}  median={r.median():.1f}")
    else:
        print("  ⚠ Cannot compute effective_rate_kw")

    # --- Price per kWh ---
    energy_safe = None
    if 'energy_kwh' in df.columns:
        energy_safe = df['energy_kwh'].replace(0, np.nan)

    if 'total_amount' in df.columns and energy_safe is not None:
        df['price_per_kwh'] = df['total_amount'] / energy_safe
        p = df['price_per_kwh']
        print(f"  price_per_kwh:          "
              f"mean={p.mean():.2f}  median={p.median():.2f}")
    else:
        print("  ⚠ Cannot compute price_per_kwh")

    # --- Per-component prices ---
    if 'elec_fee' in df.columns and energy_safe is not None:
        df['elec_price_per_kwh'] = df['elec_fee'] / energy_safe
    if 'service_fee' in df.columns and energy_safe is not None:
        df['service_price_per_kwh'] = df['service_fee'] / energy_safe

    # --- Fee decomposition check ---
    if all(c in df.columns for c in ['elec_fee', 'service_fee', 'total_amount']):
        df['fee_residual'] = df['total_amount'] - df['elec_fee'] - df['service_fee']
        large = (df['fee_residual'].abs() > 0.01).sum()
        print(f"  Fee check: {large:,} rows where |total - elec - service| > ¥0.01 "
              f"({100*large/N:.2f}%)")

    # --- Temporal features ---
    if 'start_time' in df.columns:
        df['hour_of_day'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek   # 0=Mon
        df['month']       = df['start_time'].dt.month
        df['year']        = df['start_time'].dt.year
        df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
        print("  Temporal features:      hour_of_day, day_of_week, month, year, is_weekend")

    return df


# ============================================================================
# 10. ANOMALY FLAGGING
# ============================================================================

def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (do NOT remove) anomalous sessions.
    Adds boolean flag_* columns.  Downstream code decides filtering.
    """
    print(f"\n{'=' * 70}")
    print("ANOMALY FLAGS")
    print(f"{'=' * 70}")

    flags = {}
    N = len(df)

    # --- Negative or zero durations ---
    if 'charging_duration_min' in df.columns:
        m = df['charging_duration_min'] <= MIN_DURATION_MIN
        df['flag_nonpositive_duration'] = m
        flags['nonpositive_duration'] = int(m.sum())
        print(f"  Duration ≤ 0 min:            {m.sum():>8,}  ({100*m.sum()/N:.2f}%)")

    # --- Duration > 24h ---
    if 'charging_duration_min' in df.columns:
        m = df['charging_duration_min'] > MAX_DURATION_MIN
        df['flag_excessive_duration'] = m
        flags['excessive_duration'] = int(m.sum())
        print(f"  Duration > 24h:              {m.sum():>8,}  ({100*m.sum()/N:.2f}%)")

    # --- Zero-energy sessions ---
    if 'energy_kwh' in df.columns:
        # Missing energy is unknown, not a zero-energy observation.
        m = (df['energy_kwh'].notna()
             & (df['energy_kwh'] < ZERO_ENERGY_THRESHOLD))
        df['flag_zero_energy'] = m
        df['is_null_session'] = m
        flags['zero_energy'] = int(m.sum())
        print(f"  Energy < {ZERO_ENERGY_THRESHOLD} kWh:          {m.sum():>8,}  ({100*m.sum()/N:.2f}%)")

    # --- Implausible effective rate ---
    if 'effective_rate_kw' in df.columns:
        lo = df['effective_rate_kw'] < MIN_EFFECTIVE_RATE_KW
        hi = df['effective_rate_kw'] > MAX_EFFECTIVE_RATE_KW
        df['flag_rate_too_low']  = lo
        df['flag_rate_too_high'] = hi
        flags['rate_too_low']  = int(lo.sum())
        flags['rate_too_high'] = int(hi.sum())
        print(f"  Rate < {MIN_EFFECTIVE_RATE_KW} kW:              {lo.sum():>8,}  ({100*lo.sum()/N:.2f}%)")
        print(f"  Rate > {MAX_EFFECTIVE_RATE_KW} kW:            {hi.sum():>8,}  ({100*hi.sum()/N:.2f}%)")

    # --- Composite flag ---
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    if flag_cols:
        df['flag_any'] = df[flag_cols].any(axis=1)
        n_any = df['flag_any'].sum()
        print(f"\n  ANY flag set:                {n_any:>8,}  ({100*n_any/N:.2f}%)")

    # --- Dataset's own is_abnormal ---
    if 'is_abnormal' in df.columns:
        abn_num = pd.to_numeric(df['is_abnormal'], errors='coerce')
        if abn_num.notna().any():
            abn = int(abn_num.fillna(0).sum())
        else:
            abn = int(
                df['is_abnormal']
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({'1', 'true', 'yes', 'y'})
                .sum()
            )
        abn_rate = abn / N
        print(f"  is_abnormal (from dataset):  {abn:>8,}  ({100*abn_rate:.2f}%)")

        # BLOCKER CHECK from weekly plan: sample showed ~30.9%.
        # If full dataset differs significantly, flag it.
        if abn_rate < 0.20 or abn_rate > 0.45:
            print(f"  ⚠ BLOCKER: is_abnormal rate ({100*abn_rate:.1f}%) differs "
                  f"substantially from sample expectation (~30.9%).")
            print(f"    The dataset has additional fault indicator columns "
                  f"(is_user_stop, is_nonsystem_fault, is_system_issue, "
                  f"is_ev_fault, is_other_fault).")
            print(f"    is_abnormal likely captures only a SUBSET of faults.")
            # Compute the broader fault rate from individual fault columns
            fault_cols = ['is_nonsystem_fault', 'is_system_issue',
                          'is_ev_fault', 'is_other_fault']
            available_fault_cols = [c for c in fault_cols if c in df.columns]
            if available_fault_cols:
                any_fault = df[available_fault_cols].any(axis=1).sum()
                print(f"    Any fault column set:      {any_fault:>8,}  "
                      f"({100*any_fault/N:.2f}%)")
                all_non_normal = (df[available_fault_cols].any(axis=1) |
                                  (df.get('is_abnormal', 0) == 1)).sum()
                print(f"    is_abnormal OR any fault:  {all_non_normal:>8,}  "
                      f"({100*all_non_normal/N:.2f}%)")
            print(f"    → Document this in data_quality_report.md")

    print(f"\n  Flag summary: {json.dumps(flags, indent=2)}")
    return df


# ============================================================================
# 11. SAVE
# ============================================================================

def save_clean(df: pd.DataFrame, path: Path) -> None:
    """Save cleaned DataFrame as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = detect_parquet_engine()
    if engine is None:
        fallback = path.with_suffix('.csv')
        df.to_csv(fallback, index=False)
        size_mb = fallback.stat().st_size / 1e6
        print("\n[WARN] No parquet engine installed (pyarrow/fastparquet).")
        print(f"[OK] Saved CSV fallback: {fallback}")
        print(f"     {len(df):,} rows × {len(df.columns)} cols, {size_mb:.1f} MB")
        return

    df.to_parquet(path, index=False, engine=engine)
    size_mb = path.stat().st_size / 1e6
    print(f"\n[OK] Saved: {path} (engine={engine})")
    print(f"     {len(df):,} rows × {len(df.columns)} cols, {size_mb:.1f} MB")


# ============================================================================
# 12. QUALITY REPORT
# ============================================================================

def write_quality_report(df: pd.DataFrame, warnings: list, path: Path) -> None:
    """Write a plain-text quality report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    N = len(df)

    lines = []
    lines.append("=" * 70)
    lines.append("DATA QUALITY REPORT — Jiaxing EV Charging Dataset")
    lines.append("=" * 70)
    lines.append(f"\nDataset shape: {df.shape}")

    if 'start_time' in df.columns:
        lines.append(f"Time range: {df['start_time'].min()} → {df['start_time'].max()}")
    if 'station_name' in df.columns:
        lines.append(f"Stations (station_name): {df['station_name'].nunique()}")
    elif 'station_id' in df.columns:
        lines.append(f"Stations (station_id): {df['station_id'].nunique()}")
    if 'district' in df.columns:
        lines.append(f"Districts: {df['district'].nunique()}")
    if 'is_abnormal' in df.columns:
        abn_rate = df['is_abnormal'].mean()
        lines.append(f"is_abnormal rate: {100*abn_rate:.2f}%  "
                     f"{'(matches sample ~31%)' if 0.25 < abn_rate < 0.36 else '(DIFFERS from sample ~31%)'}")

    lines.append(f"\nSchema warnings ({len(warnings)}):")
    for w in warnings:
        lines.append(f"  - {w}")

    lines.append(f"\nNull counts (non-zero only):")
    for col in df.columns:
        n = df[col].isna().sum()
        if n > 0:
            lines.append(f"  {col:40s} {n:>8,}  ({100*n/N:.2f}%)")

    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    if flag_cols:
        lines.append(f"\nAnomaly flags:")
        for col in flag_cols:
            n = df[col].sum()
            lines.append(f"  {col:40s} {n:>8,}  ({100*n/N:.2f}%)")

    lines.append(f"\nColumn listing ({len(df.columns)}):")
    for col in df.columns:
        lines.append(f"  {col:40s} {str(df[col].dtype)}")

    report_text = '\n'.join(lines)
    print(f"\n{report_text}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[OK] Quality report saved: {path}")


# ============================================================================
# 13. MAIN PIPELINE
# ============================================================================

def main(charging_path: Path, weather_path: Path, tou_path: Path,
         output_path: Path, report_path: Path,
         duplicate_policy: str = 'error') -> None:

    configure_console_output()

    print("=" * 70)
    print("JIAXING EV CHARGING — DATA INGESTION & CLEANING PIPELINE")
    print("=" * 70)
    print(f"  Charging: {charging_path}")
    print(f"  Weather:  {weather_path}")
    print(f"  TOU:      {tou_path}")
    print(f"  Output:   {output_path}")

    # ---- Load all three files ----
    df_charging = load_csv(charging_path, "Charging_Data")
    df_weather  = load_csv(weather_path,  "Weather_Data")
    df_tou      = load_csv(tou_path,      "TOU_Price")

    # ---- Bootstrap: print raw schemas ----
    bootstrap_inspect(df_charging, "Charging_Data (RAW)")
    bootstrap_inspect(df_weather,  "Weather_Data (RAW)")
    bootstrap_inspect(df_tou,      "TOU_Price (RAW)")

    # ---- Normalize charging columns ----
    df = normalize_columns(df_charging, label="Charging")

    # ---- Schema validation ----
    warnings = validate_charging(df)

    if 'order_id' in df.columns:
        duplicate_mask = df['order_id'].notna() & df['order_id'].duplicated(
            keep=False)
        if duplicate_mask.any():
            n_rows = int(duplicate_mask.sum())
            if duplicate_policy == 'error':
                raise ValueError(
                    f"Found {n_rows:,} rows with duplicated order_id values. "
                    "Inspect them and rerun with --duplicate-policy "
                    "drop-first or drop-last only if that rule is justified.")
            keep = 'first' if duplicate_policy == 'drop-first' else 'last'
            before = len(df)
            df = df.drop_duplicates(subset=['order_id'], keep=keep).copy()
            warnings.append(
                f"Removed {before - len(df):,} duplicated order rows using "
                f"policy={duplicate_policy}")

    # ---- Parse timestamps ----
    df = parse_timestamps(df)

    # ---- Merge weather ----
    df = merge_weather(df, df_weather)

    # ---- Merge TOU ----
    df = merge_tou(df, df_tou)

    # ---- Derived features ----
    df = compute_derived_features(df)

    # ---- Anomaly flags ----
    df = flag_anomalies(df)

    # ---- Save ----
    save_clean(df, output_path)

    # ---- Quality report ----
    write_quality_report(df, warnings, report_path)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nNext steps:")
    print(f"  1. Review console output above for ⚠ warnings")
    print(f"  2. If TOU merge was skipped, manually expand the TOU ranges")
    print(f"     using expand_tou_ranges() and re-run")
    print(f"  3. Run the 5 sanity checks in week1_sanity_checks.md")
    print(f"  4. Load the clean parquet to verify:")
    print(f"     df = pd.read_parquet(r'{output_path}')")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Ingest and clean Jiaxing EV charging dataset (3-file merge)."
    )
    parser.add_argument('--charging', type=Path, default=DEFAULT_CHARGING_PATH,
                        help="Path to Charging_Data.csv")
    parser.add_argument('--weather', type=Path, default=DEFAULT_WEATHER_PATH,
                        help="Path to Weather_Data.csv")
    parser.add_argument('--tou', type=Path, default=DEFAULT_TOU_PATH,
                        help="Path to Time-of-use_Price.csv")
    parser.add_argument('--output', type=Path, default=OUTPUT_PATH,
                        help="Output Parquet path")
    parser.add_argument('--report', type=Path, default=REPORT_PATH,
                        help="Quality report output path")
    parser.add_argument(
        '--duplicate-policy',
        choices=['error', 'drop-first', 'drop-last'],
        default='error',
        help='How to handle duplicated non-null order IDs (default: error).')

    args = parser.parse_args()
    main(args.charging, args.weather, args.tou, args.output, args.report,
         args.duplicate_policy)
