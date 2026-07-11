"""
Week 3: Arrival Process Characterization
=========================================
Phase 1 of EV Charging Optimization SOP

Components:
  1. Formal Poisson goodness-of-fit tests per station
     - Chi-squared on hourly arrival counts
     - Kolmogorov-Smirnov on inter-arrival times
     - Anderson-Darling on inter-arrival times
  2. Conditional dispersion analysis
     - Unconditional DI (pooled across hours)
     - Conditional DI (within hour-of-day bins)
     - Conditional DI (within hour-of-day × day-type bins)
  3. NHPP fitting with piecewise-constant hourly rate functions
     - Per-station λ(h) for h = 0..23
     - Confidence intervals via bootstrap
     - Seasonal decomposition (quarterly rate multipliers)
  4. TOU price explanatory power
     - Poisson GLM: arrivals ~ hour_dummies + tou_price
     - Likelihood ratio test for price effect beyond hour-of-day

Assumptions:
  - Analysis is per-station (never pooled unless explicitly stated)
  - Zero-energy sessions ARE included in arrival counts
  - Fault sessions ARE included in arrival counts
  - Inter-arrival times use all arrivals (including zero-energy and fault)
  - Service time fitting is NOT in this script (Week 4)

Input files (from Week 1):
  - jiaxing_clean.parquet   (session-level, all 441k rows)
  - jiaxing_hourly.parquet  (station-hour aggregation)
  - jiaxing_iat.parquet     (inter-arrival times per station)

Output files:
  - week3_results/poisson_gof_tests.csv
  - week3_results/conditional_dispersion.csv
  - week3_results/nhpp_rate_functions.csv
  - week3_results/nhpp_rate_functions_quarterly.csv
  - week3_results/tou_glm_results.csv
  - week3_results/figures/*.png

Usage:
  python week3_analysis.py --data-dir ./data --output-dir ./week3_results

Date: Week 3, Feb 2026
"""

import argparse
import warnings
import json
import os
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2, kstest, anderson, expon, poisson
try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod.families import Poisson as PoissonFamily
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[INFO] statsmodels not available. Using scipy-based Poisson GLM (IRLS).")

warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


# =============================================================================
# DATA LOADING
# =============================================================================

def configure_console_output():
    """Avoid UnicodeEncodeError on cp1252-like terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass


def detect_parquet_engine():
    """Return available parquet engine, or None."""
    if find_spec('pyarrow') is not None:
        return 'pyarrow'
    if find_spec('fastparquet') is not None:
        return 'fastparquet'
    return None


def load_data(data_dir):
    """Load all required data files from Week 1 output. Supports Parquet and CSV."""
    data_dir = Path(data_dir)

    def read_file(name_base):
        """Try parquet first, then CSV."""
        parquet_path = data_dir / f'{name_base}.parquet'
        csv_path = data_dir / f'{name_base}.csv'
        engine = detect_parquet_engine()
        if parquet_path.exists() and engine is not None:
            try:
                return pd.read_parquet(parquet_path, engine=engine), parquet_path.name
            except Exception as e:
                print(f"[WARN] Failed reading {parquet_path.name} with {engine}: {e}")
        elif parquet_path.exists() and engine is None:
            print(f"[WARN] No parquet engine installed. Looking for CSV fallback for {name_base}.")
        if csv_path.exists():
            return pd.read_csv(csv_path), csv_path.name
        return None, None

    sessions, sessions_src = read_file('jiaxing_clean')
    if sessions is None:
        raise FileNotFoundError(
            f"Cannot find jiaxing_clean.parquet or jiaxing_clean.csv in {data_dir}")

    hourly, hourly_src = read_file('jiaxing_hourly')
    if hourly is None:
        print("[INFO] jiaxing_hourly not found. Computing from sessions.")
        hourly = compute_hourly_counts(sessions)
        hourly_src = 'computed'

    iat, iat_src = read_file('jiaxing_iat')
    if iat is None:
        print("[INFO] jiaxing_iat not found. Computing IATs from session data.")
        iat = compute_iat(sessions)
        iat_src = 'computed'

    # Normalize station labels (trim accidental trailing whitespace).
    for frame in [sessions, hourly, iat]:
        station_col = identify_columns(frame, 'station')
        if station_col is not None:
            frame[station_col] = frame[station_col].astype(str).str.strip()

    # Ensure consistent IAT unit/column expected by this script.
    if 'iat_seconds' not in iat.columns and 'iat_min' in iat.columns:
        iat['iat_seconds'] = pd.to_numeric(iat['iat_min'], errors='coerce') * 60.0
    elif 'iat_seconds' in iat.columns and 'iat_min' not in iat.columns:
        iat['iat_min'] = pd.to_numeric(iat['iat_seconds'], errors='coerce') / 60.0

    # Parse common datetime fields when loading from CSV.
    for col in ['start_time', 'end_time', 'order_created_time', 'payment_time', 'date']:
        if col in sessions.columns:
            sessions[col] = pd.to_datetime(sessions[col], errors='coerce')
    for col in ['date', 'date_dt']:
        if col in hourly.columns:
            hourly[col] = pd.to_datetime(hourly[col], errors='coerce')
    if 'date' in iat.columns:
        iat['date'] = pd.to_datetime(iat['date'], errors='coerce')

    # Ensure hourly table includes explicit zero-arrival bins for all station×date×hour.
    hourly = ensure_complete_hourly_grid(hourly, sessions)

    print(f"[DATA] Sessions: {len(sessions):,} rows ({sessions_src})")
    print(f"[DATA] Hourly:   {len(hourly):,} rows ({hourly_src})")
    print(f"[DATA] IAT:      {len(iat):,} rows ({iat_src})")

    return sessions, hourly, iat


def ensure_complete_hourly_grid(hourly, sessions):
    """
    Ensure hourly table has one row for every station × date × hour (0..23).
    Week 1 hourly output can omit zero-arrival hours; Week 3 Poisson/NHPP
    analysis requires those bins explicitly present as 0.
    """
    station_col = identify_columns(hourly, 'station')
    hour_col = identify_columns(hourly, 'hour')
    date_col = identify_columns(hourly, 'date')
    arrivals_col = identify_columns(hourly, 'arrivals')

    if station_col is None or hour_col is None or date_col is None or arrivals_col is None:
        print("[INFO] Hourly schema incomplete for Poisson/NHPP. Recomputing hourly grid from sessions.")
        return compute_hourly_counts(sessions)

    h = hourly[[station_col, date_col, hour_col, arrivals_col]].copy()
    h.columns = ['station', 'date', 'hour', 'arrivals']
    h['date'] = pd.to_datetime(h['date'], errors='coerce').dt.normalize()
    h['hour'] = pd.to_numeric(h['hour'], errors='coerce')
    h['arrivals'] = pd.to_numeric(h['arrivals'], errors='coerce').fillna(0)
    h = h.dropna(subset=['station', 'date', 'hour'])
    h['hour'] = h['hour'].astype(int)
    h['station'] = h['station'].astype(str).str.strip()
    h = h[(h['hour'] >= 0) & (h['hour'] <= 23)]

    # Collapse any accidental duplicate keys.
    h = h.groupby(['station', 'date', 'hour'], as_index=False)['arrivals'].sum()

    stations = sorted(h['station'].unique())
    dates = sorted(h['date'].unique())
    full_index = pd.MultiIndex.from_product(
        [stations, dates, list(range(24))], names=['station', 'date', 'hour']
    )
    full = (
        h.set_index(['station', 'date', 'hour'])
         .reindex(full_index, fill_value=0)
         .reset_index()
    )

    # Preserve TOU covariates if present (hour-based lookup).
    price_col = identify_columns(hourly, 'tou_price')
    if price_col is not None:
        p = hourly[[hour_col, price_col]].copy()
        p.columns = ['hour', 'tou_price']
        p['hour'] = pd.to_numeric(p['hour'], errors='coerce')
        p['tou_price'] = pd.to_numeric(p['tou_price'], errors='coerce')
        p = p.dropna(subset=['hour', 'tou_price']).groupby('hour', as_index=False)['tou_price'].mean()
        full = full.merge(p, on='hour', how='left')

    tou_tier_col = identify_columns(hourly, 'tou_tier')
    if tou_tier_col is not None:
        t = hourly[[hour_col, tou_tier_col]].copy()
        t.columns = ['hour', 'tou_tier']
        t['hour'] = pd.to_numeric(t['hour'], errors='coerce')
        t = t.dropna(subset=['hour', 'tou_tier'])
        if len(t) > 0:
            t = t.groupby('hour')['tou_tier'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
            ).reset_index()
            full = full.merge(t, on='hour', how='left')

    # Add convenient aliases used across functions.
    full['date_dt'] = full['date']
    full['hour_of_day'] = full['hour']

    expected = len(stations) * len(dates) * 24
    if len(full) > len(h):
        print(f"[INFO] Expanded hourly grid: {len(h):,} -> {len(full):,} rows "
              f"(expected={expected:,}; filled missing bins with 0 arrivals).")
    return full


def compute_iat(sessions):
    """Compute inter-arrival times per station from session-level data."""
    # Identify the start time column
    time_col = None
    for candidate in ['start_time', 'start_datetime', 'StartTime', 'begin_time']:
        if candidate in sessions.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(f"Cannot find start time column. Available: {list(sessions.columns)}")

    # Identify the station column
    station_col = None
    for candidate in ['station', 'station_name', 'station_id', 'Station']:
        if candidate in sessions.columns:
            station_col = candidate
            break
    if station_col is None:
        raise ValueError(f"Cannot find station column. Available: {list(sessions.columns)}")

    sessions = sessions.sort_values([station_col, time_col])
    records = []
    for station, group in sessions.groupby(station_col):
        times = pd.to_datetime(group[time_col]).sort_values()
        diffs = times.diff().dt.total_seconds().dropna()
        # Preserve simultaneous arrivals. Zero IATs are valid evidence against
        # a continuous exponential model and must not depend on input source.
        diffs = diffs[diffs >= 0]
        for d in diffs:
            records.append({'station': station, 'iat_seconds': d})

    return pd.DataFrame(records)


def identify_columns(df, col_type):
    """Flexibly identify column names across possible naming conventions."""
    candidates = {
        'station': ['station', 'station_name', 'station_id', 'Station'],
        'start_time': ['start_time', 'start_datetime', 'StartTime', 'begin_time'],
        'hour': ['hour', 'Hour', 'hour_of_day'],
        'arrivals': ['arrivals', 'arrival_count', 'count', 'n_arrivals', 'num_sessions'],
        'date': ['date', 'date_dt', 'Date', 'day'],
        'tou_price': ['tou_electricity_price', 'tou_electricity_price_yuan_kwh',
                      'tou_price', 'price', 'electricity_price', 'tou_tier_price',
                      'price_per_kwh', 'avg_price'],
        'tou_tier': ['tou_tier', 'tou_period', 'price_tier'],
        'is_abnormal': ['is_abnormal', 'is_fault', 'fault'],
    }
    for c in candidates.get(col_type, []):
        if c in df.columns:
            return c
    return None


# =============================================================================
# COMPONENT 1: POISSON GOODNESS-OF-FIT TESTS
# =============================================================================

def chi_squared_poisson_test(counts):
    """
    Chi-squared goodness-of-fit test for Poisson distribution on count data.

    Tests H0: counts ~ Poisson(λ_hat) where λ_hat = sample mean.

    Returns dict with test statistic, p-value, df, and bin details.
    """
    n = len(counts)
    if n < 30:
        return {'statistic': np.nan, 'p_value': np.nan, 'df': np.nan,
                'n_bins': np.nan, 'note': 'Insufficient data (n < 30)'}

    lambda_hat = counts.mean()

    # Create bins: 0, 1, 2, ..., max_k, with last bin = "≥ max_k"
    # Merge bins with expected frequency < 5
    max_k = int(counts.max())
    # Start with individual count bins
    observed = np.array([np.sum(counts == k) for k in range(max_k + 1)])
    expected = np.array([n * poisson.pmf(k, lambda_hat) for k in range(max_k + 1)])

    # Add overflow bin
    overflow_obs = 0  # already counted
    overflow_exp = n * (1 - poisson.cdf(max_k, lambda_hat))

    # Merge from the top until expected >= 5
    obs_bins = list(observed)
    exp_bins = list(expected)
    obs_bins.append(overflow_obs)
    exp_bins.append(overflow_exp)

    # Merge small bins from the right
    while len(exp_bins) > 2 and exp_bins[-1] < 5:
        obs_bins[-2] += obs_bins[-1]
        exp_bins[-2] += exp_bins[-1]
        obs_bins.pop()
        exp_bins.pop()

    # Merge small bins from the left
    while len(exp_bins) > 2 and exp_bins[0] < 5:
        obs_bins[1] += obs_bins[0]
        exp_bins[1] += exp_bins[0]
        obs_bins.pop(0)
        exp_bins.pop(0)

    obs_bins = np.array(obs_bins, dtype=float)
    exp_bins = np.array(exp_bins, dtype=float)

    # Remove bins with zero expected (shouldn't happen after merging, but safety)
    mask = exp_bins > 0
    obs_bins = obs_bins[mask]
    exp_bins = exp_bins[mask]

    n_bins = len(obs_bins)
    df = n_bins - 1 - 1  # -1 for constraint (sum), -1 for estimated λ

    if df < 1:
        return {'statistic': np.nan, 'p_value': np.nan, 'df': np.nan,
                'n_bins': n_bins, 'note': 'Insufficient bins after merging'}

    chi2_stat = np.sum((obs_bins - exp_bins) ** 2 / exp_bins)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'df': df,
        'n_bins': n_bins,
        'lambda_hat': lambda_hat,
        'note': ''
    }


def ks_exponential_test(iat_values):
    """
    Kolmogorov-Smirnov test for exponential distribution on inter-arrival times.

    H0: IATs ~ Exponential(1/λ_hat)
    If arrivals are Poisson, IATs should be exponential.
    """
    if len(iat_values) < 30:
        return {'statistic': np.nan, 'p_value': np.nan,
                'note': 'Insufficient data (n < 30)'}

    values = np.asarray(iat_values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) < 30 or float(values.mean()) <= 0:
        return {'statistic': np.nan, 'p_value': np.nan,
                'note': 'Insufficient positive-scale data'}
    rng = np.random.default_rng(202603)
    if len(values) > 5000:
        values = rng.choice(values, size=5000, replace=False)

    # Fit exponential and obtain a fitted-parameter p-value by parametric
    # bootstrap (the ordinary KS reference distribution is not valid here).
    loc = 0  # exponential starts at 0
    scale = np.mean(values)

    stat = float(kstest(values, 'expon', args=(loc, scale)).statistic)
    bootstrap_stats = []
    for _ in range(200):
        sample = rng.exponential(scale, size=len(values))
        fitted_scale = float(sample.mean())
        bootstrap_stats.append(float(kstest(
            sample, 'expon', args=(0, fitted_scale)).statistic))
    p_value = ((1 + np.sum(np.asarray(bootstrap_stats) >= stat))
               / (len(bootstrap_stats) + 1))

    return {
        'statistic': stat,
        'p_value': p_value,
        'n': len(values),
        'mean_iat_min': scale / 60,  # convert to minutes
        'note': 'Parametric-bootstrap p-value; deterministic subsample capped at 5000'
    }


def ad_exponential_test(iat_values):
    """
    Anderson-Darling test for exponential distribution.
    More powerful than KS for tail departures.
    """
    if len(iat_values) < 30:
        return {'statistic': np.nan, 'critical_5pct': np.nan,
                'reject_5pct': np.nan, 'note': 'Insufficient data'}

    result = anderson(iat_values, dist='expon')
    statistic = float(result.statistic)
    saturated = not np.isfinite(statistic)

    # Anderson returns critical values at [15%, 10%, 5%, 2.5%, 1%]
    # Index 2 = 5% significance level
    note = ''
    if saturated:
        note = (
            'AD statistic overflow/saturation; use rejection flag only, '
            'not statistic magnitude.'
        )
    return {
        'statistic': statistic,
        'critical_1pct': result.critical_values[4],
        'critical_5pct': result.critical_values[2],
        'critical_10pct': result.critical_values[1],
        'reject_1pct': True if saturated else statistic > result.critical_values[4],
        'reject_5pct': True if saturated else statistic > result.critical_values[2],
        'reject_10pct': True if saturated else statistic > result.critical_values[1],
        'statistic_saturated': saturated,
        'note': note,
    }


def run_poisson_gof_tests(sessions, hourly, iat, output_dir):
    """
    Run all three Poisson GOF tests per station.

    Returns DataFrame with one row per station and columns for each test.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 1: POISSON GOODNESS-OF-FIT TESTS")
    print("=" * 70)

    station_col_h = identify_columns(hourly, 'station')
    arrivals_col = identify_columns(hourly, 'arrivals')
    station_col_iat = identify_columns(iat, 'station') or 'station'

    # Fallback: if hourly doesn't have an arrivals column, try to compute
    if arrivals_col is None:
        # Check if hourly has a count-like column
        for c in hourly.columns:
            if hourly[c].dtype in ['int64', 'float64'] and hourly[c].min() >= 0:
                if 'count' in c.lower() or 'arrival' in c.lower() or 'session' in c.lower():
                    arrivals_col = c
                    break
        if arrivals_col is None:
            print("[WARN] Cannot identify arrivals column in hourly data.")
            print(f"  Available columns: {list(hourly.columns)}")
            print("  Will compute hourly counts from session data.")
            hourly = compute_hourly_counts(sessions)
            station_col_h = 'station'
            arrivals_col = 'arrivals'

    stations = sorted(hourly[station_col_h].unique())
    print(f"[INFO] Testing {len(stations)} stations")

    results = []

    for station in stations:
        print(f"\n--- Station: {station} ---")
        row = {'station': station}

        # 1a. Chi-squared test on hourly counts
        mask = hourly[station_col_h] == station
        counts = hourly.loc[mask, arrivals_col].values
        chi2_result = chi_squared_poisson_test(counts)
        row['chi2_statistic'] = chi2_result['statistic']
        row['chi2_p_value'] = chi2_result['p_value']
        row['chi2_df'] = chi2_result.get('df', np.nan)
        row['chi2_lambda'] = chi2_result.get('lambda_hat', np.nan)
        row['chi2_reject_5pct'] = chi2_result['p_value'] < 0.05 if not np.isnan(chi2_result['p_value']) else np.nan

        reject_str = "REJECT" if row['chi2_reject_5pct'] else "FAIL TO REJECT"
        print(f"  Chi2: stat={chi2_result['statistic']:.1f}, p={chi2_result['p_value']:.4e} -> {reject_str}")

        # 1b. KS test on inter-arrival times
        iat_vals = iat.loc[iat[station_col_iat] == station, 'iat_seconds'].values
        if len(iat_vals) > 0:
            ks_result = ks_exponential_test(iat_vals)
            row['ks_statistic'] = ks_result['statistic']
            row['ks_p_value'] = ks_result['p_value']
            row['ks_n'] = ks_result.get('n', len(iat_vals))
            row['ks_mean_iat_min'] = ks_result.get('mean_iat_min', np.nan)
            row['ks_reject_5pct'] = ks_result['p_value'] < 0.05 if not np.isnan(ks_result['p_value']) else np.nan

            reject_str = "REJECT" if row.get('ks_reject_5pct') else "FAIL TO REJECT"
            print(f"  KS:   stat={ks_result['statistic']:.4f}, p={ks_result['p_value']:.4e} -> {reject_str}")
        else:
            row['ks_statistic'] = np.nan
            row['ks_p_value'] = np.nan
            row['ks_n'] = 0
            row['ks_mean_iat_min'] = np.nan
            row['ks_reject_5pct'] = np.nan
            print(f"  KS:   No IAT data for this station")

        # 1c. Anderson-Darling test on inter-arrival times
        if len(iat_vals) > 0:
            ad_result = ad_exponential_test(iat_vals)
            row['ad_statistic'] = ad_result['statistic']
            row['ad_critical_5pct'] = ad_result.get('critical_5pct', np.nan)
            row['ad_reject_5pct'] = ad_result.get('reject_5pct', np.nan)
            row['ad_reject_1pct'] = ad_result.get('reject_1pct', np.nan)
            row['ad_statistic_saturated'] = ad_result.get('statistic_saturated', False)
            row['ad_note'] = ad_result.get('note', '')

            reject_str = "REJECT" if row['ad_reject_5pct'] else "FAIL TO REJECT"
            stat_str = (
                "saturated"
                if row['ad_statistic_saturated']
                else f"{ad_result['statistic']:.2f}"
            )
            print(f"  AD:   stat={stat_str}, crit(5%)={ad_result.get('critical_5pct', np.nan):.2f} -> {reject_str}")
        else:
            row['ad_statistic'] = np.nan
            row['ad_critical_5pct'] = np.nan
            row['ad_reject_5pct'] = np.nan
            row['ad_reject_1pct'] = np.nan
            row['ad_statistic_saturated'] = False
            row['ad_note'] = ''

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'poisson_gof_tests.csv', index=False)
    print(f"\n[SAVED] poisson_gof_tests.csv")

    # Summary
    n_reject_chi2 = df['chi2_reject_5pct'].sum()
    n_reject_ks = df['ks_reject_5pct'].sum()
    n_reject_ad = df['ad_reject_5pct'].sum()
    print(f"\n[SUMMARY] Rejection at α=0.05:")
    print(f"  Chi-squared:     {int(n_reject_chi2)}/{len(stations)} stations reject H0")
    print(f"  KS (exponential): {int(n_reject_ks)}/{len(stations)} stations reject H0")
    print(f"  Anderson-Darling: {int(n_reject_ad)}/{len(stations)} stations reject H0")

    return df


def compute_hourly_counts(sessions):
    """Compute hourly arrival counts per station from session-level data."""
    station_col = identify_columns(sessions, 'station')
    time_col = identify_columns(sessions, 'start_time')

    if station_col is None or time_col is None:
        raise ValueError("Cannot identify station or start_time columns in sessions data.")

    df = sessions[[station_col, time_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['date'] = df[time_col].dt.date
    df['hour'] = df[time_col].dt.hour

    hourly = df.groupby([station_col, 'date', 'hour']).size().reset_index(name='arrivals')
    hourly.rename(columns={station_col: 'station'}, inplace=True)

    # Fill missing hours with 0 arrivals
    stations = hourly['station'].unique()
    dates = hourly['date'].unique()
    hours = range(24)
    full_index = pd.MultiIndex.from_product([stations, dates, hours],
                                             names=['station', 'date', 'hour'])
    hourly = hourly.set_index(['station', 'date', 'hour']).reindex(full_index, fill_value=0).reset_index()

    return hourly


# =============================================================================
# COMPONENT 2: CONDITIONAL DISPERSION ANALYSIS
# =============================================================================

def dispersion_index(counts):
    """Compute variance/mean ratio (dispersion index) for count data."""
    mean = counts.mean()
    var = counts.var(ddof=1)
    if mean == 0:
        return np.nan
    return var / mean


def run_conditional_dispersion(sessions, hourly, output_dir):
    """
    Compute dispersion indices at three levels of conditioning:
      1. Unconditional (all hours pooled)
      2. Conditional on hour-of-day
      3. Conditional on hour-of-day × day-type (weekday/weekend)

    If Poisson holds conditionally, the within-bin DI should be ≈ 1.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 2: CONDITIONAL DISPERSION ANALYSIS")
    print("=" * 70)

    station_col = identify_columns(hourly, 'station')
    arrivals_col = identify_columns(hourly, 'arrivals')
    hour_col = identify_columns(hourly, 'hour')

    # If hourly doesn't have what we need, recompute
    if station_col is None or arrivals_col is None or hour_col is None:
        print("[INFO] Recomputing hourly counts from sessions.")
        hourly = compute_hourly_counts(sessions)
        station_col, arrivals_col, hour_col = 'station', 'arrivals', 'hour'

    # Add day-of-week info if not present
    date_col = identify_columns(hourly, 'date')
    if date_col is not None:
        hourly['_date'] = pd.to_datetime(hourly[date_col])
        hourly['_dow'] = hourly['_date'].dt.dayofweek
        hourly['_day_type'] = np.where(hourly['_dow'] < 5, 'weekday', 'weekend')
    else:
        # Try to infer from session data
        print("[WARN] No date column in hourly data; computing day_type from sessions.")
        hourly['_day_type'] = 'unknown'

    stations = sorted(hourly[station_col].unique())
    results = []

    for station in stations:
        mask = hourly[station_col] == station
        df_s = hourly.loc[mask].copy()

        row = {'station': station}

        # 1. Unconditional DI
        counts_all = df_s[arrivals_col].values
        row['di_unconditional'] = dispersion_index(counts_all)
        row['mean_unconditional'] = counts_all.mean()
        row['var_unconditional'] = counts_all.var(ddof=1)
        row['n_obs_unconditional'] = len(counts_all)

        # 2. Conditional on hour-of-day: compute DI within each hour, then average
        di_by_hour = []
        for h in range(24):
            counts_h = df_s.loc[df_s[hour_col] == h, arrivals_col].values
            if len(counts_h) > 1 and counts_h.mean() > 0:
                di_by_hour.append(dispersion_index(counts_h))
        row['di_cond_hour_mean'] = np.mean(di_by_hour) if di_by_hour else np.nan
        row['di_cond_hour_median'] = np.median(di_by_hour) if di_by_hour else np.nan
        row['di_cond_hour_max'] = np.max(di_by_hour) if di_by_hour else np.nan
        row['di_cond_hour_min'] = np.min(di_by_hour) if di_by_hour else np.nan
        row['n_hours_tested'] = len(di_by_hour)

        # 3. Conditional on hour × day_type
        if '_day_type' in df_s.columns and df_s['_day_type'].nunique() > 1:
            di_by_hour_day = []
            for h in range(24):
                for dt in ['weekday', 'weekend']:
                    counts_hd = df_s.loc[(df_s[hour_col] == h) & (df_s['_day_type'] == dt),
                                          arrivals_col].values
                    if len(counts_hd) > 1 and counts_hd.mean() > 0:
                        di_by_hour_day.append(dispersion_index(counts_hd))
            row['di_cond_hour_day_mean'] = np.mean(di_by_hour_day) if di_by_hour_day else np.nan
            row['di_cond_hour_day_median'] = np.median(di_by_hour_day) if di_by_hour_day else np.nan
            row['n_bins_hour_day'] = len(di_by_hour_day)
        else:
            row['di_cond_hour_day_mean'] = np.nan
            row['di_cond_hour_day_median'] = np.nan
            row['n_bins_hour_day'] = 0

        # Interpretation flag
        row['poisson_plausible_cond_hour'] = (
            row['di_cond_hour_mean'] < 2.0 if not np.isnan(row['di_cond_hour_mean']) else np.nan
        )

        results.append(row)
        print(f"  {station}: DI_uncond={row['di_unconditional']:.1f}, "
              f"DI_cond_hour_mean={row['di_cond_hour_mean']:.2f}, "
              f"DI_cond_hour_day_mean={row['di_cond_hour_day_mean']:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'conditional_dispersion.csv', index=False)
    print(f"\n[SAVED] conditional_dispersion.csv")

    # Summary
    n_plausible = df['poisson_plausible_cond_hour'].sum()
    print(f"\n[SUMMARY] Conditional Poisson plausibility (DI < 2.0 after hour conditioning):")
    print(f"  {int(n_plausible)}/{len(stations)} stations have mean conditional DI < 2.0")
    print(f"  Range of conditional DI means: {df['di_cond_hour_mean'].min():.2f} to {df['di_cond_hour_mean'].max():.2f}")

    return df


# =============================================================================
# COMPONENT 3: NHPP FITTING
# =============================================================================

def fit_nhpp_rate_functions(sessions, hourly, output_dir, bootstrap_iters=300):
    """
    Fit piecewise-constant NHPP rate functions λ(h) for each station.

    Method: For each station, compute mean arrivals per hour-of-day across all dates.
    This gives a 24-element rate vector. Bootstrap CIs are computed by resampling dates.

    Also computes quarterly rate multipliers to capture the 2.7x growth trend.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 3: NHPP RATE FUNCTION FITTING")
    print("=" * 70)
    print(f"[INFO] Bootstrap iterations per station-hour: {bootstrap_iters}")

    station_col = identify_columns(hourly, 'station')
    arrivals_col = identify_columns(hourly, 'arrivals')
    hour_col = identify_columns(hourly, 'hour')
    date_col = identify_columns(hourly, 'date')

    if station_col is None or arrivals_col is None:
        hourly = compute_hourly_counts(sessions)
        station_col, arrivals_col, hour_col, date_col = 'station', 'arrivals', 'hour', 'date'

    stations = sorted(hourly[station_col].unique())

    # ---- Overall NHPP rate functions (time-averaged) ----
    rate_records = []
    for station_idx, station in enumerate(stations):
        df_s = hourly[hourly[station_col] == station]
        for h in range(24):
            counts = df_s.loc[df_s[hour_col] == h, arrivals_col].values
            mean_rate = counts.mean()
            std_rate = counts.std(ddof=1)
            n_days = len(counts)

            # Bootstrap 95% CI
            if n_days >= 10 and bootstrap_iters > 0:
                rng = np.random.default_rng(42 + station_idx * 100 + h)
                # Vectorized bootstrap: much faster than Python loops.
                sample_idx = rng.integers(0, n_days, size=(bootstrap_iters, n_days))
                boot_means = counts[sample_idx].mean(axis=1)
                ci_low = np.percentile(boot_means, 2.5)
                ci_high = np.percentile(boot_means, 97.5)
            else:
                ci_low = mean_rate - 1.96 * std_rate / np.sqrt(max(n_days, 1))
                ci_high = mean_rate + 1.96 * std_rate / np.sqrt(max(n_days, 1))

            rate_records.append({
                'station': station,
                'hour': h,
                'lambda_mean': mean_rate,
                'lambda_std': std_rate,
                'lambda_ci_low': max(0, ci_low),
                'lambda_ci_high': ci_high,
                'n_days': n_days,
            })

    rate_df = pd.DataFrame(rate_records)
    rate_df.to_csv(output_dir / 'nhpp_rate_functions.csv', index=False)
    print(f"[SAVED] nhpp_rate_functions.csv ({len(rate_df)} rows)")

    # ---- Quarterly rate multipliers (to capture growth trend) ----
    if date_col is not None:
        hourly['_date'] = pd.to_datetime(hourly[date_col])
        hourly['_quarter'] = hourly['_date'].dt.to_period('Q').astype(str)

        quarterly_records = []
        for station in stations:
            df_s = hourly[hourly[station_col] == station]
            overall_mean = df_s[arrivals_col].mean()
            if overall_mean == 0:
                continue

            for quarter, df_q in df_s.groupby('_quarter'):
                for h in range(24):
                    counts = df_q.loc[df_q[hour_col] == h, arrivals_col].values
                    if len(counts) > 0:
                        quarterly_records.append({
                            'station': station,
                            'quarter': quarter,
                            'hour': h,
                            'lambda_mean': counts.mean(),
                            'n_days': len(counts),
                        })

        quarterly_df = pd.DataFrame(quarterly_records)
        quarterly_df.to_csv(output_dir / 'nhpp_rate_functions_quarterly.csv', index=False)
        print(f"[SAVED] nhpp_rate_functions_quarterly.csv ({len(quarterly_df)} rows)")
    else:
        quarterly_df = pd.DataFrame()
        print("[WARN] No date column; quarterly rate functions not computed.")

    # Print summary
    for station in stations:
        df_s = rate_df[rate_df['station'] == station]
        peak_hour = df_s.loc[df_s['lambda_mean'].idxmax(), 'hour']
        peak_rate = df_s['lambda_mean'].max()
        trough_hour = df_s.loc[df_s['lambda_mean'].idxmin(), 'hour']
        trough_rate = df_s['lambda_mean'].min()
        ratio = peak_rate / trough_rate if trough_rate > 0 else np.inf
        print(f"  {station}: peak={peak_rate:.1f}/hr @ h={int(peak_hour)}, "
              f"trough={trough_rate:.1f}/hr @ h={int(trough_hour)}, ratio={ratio:.1f}x")

    return rate_df, quarterly_df


# =============================================================================
# COMPONENT 4: TOU PRICE GLM
# =============================================================================

class PoissonGLMResult:
    """Lightweight container for Poisson GLM results (scipy fallback)."""
    def __init__(self, params, bse, llf, aic, deviance):
        self.params = params
        self.bse = bse
        self.llf = llf
        self.aic = aic
        self.deviance = deviance


def fit_poisson_glm_irls(y, X, max_iter=50, tol=1e-8):
    """
    Fit Poisson GLM via Iteratively Reweighted Least Squares (IRLS).
    This is a fallback when statsmodels is not available.

    Parameters:
        y: array of shape (n,) — response (counts)
        X: array of shape (n, p) — design matrix (with intercept)

    Returns:
        PoissonGLMResult object with params, bse, llf, aic, deviance
    """
    n, p = X.shape
    beta = np.zeros(p)  # initial guess

    for iteration in range(max_iter):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -20, 20))  # prevent overflow
        mu = np.maximum(mu, 1e-10)  # prevent log(0)

        # Working weights and adjusted dependent variable
        W = np.diag(mu)
        z = eta + (y - mu) / mu  # working response

        # Weighted least squares step
        try:
            XtWX = X.T @ W @ X
            XtWz = X.T @ W @ z
            beta_new = np.linalg.solve(XtWX + 1e-10 * np.eye(p), XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final values
    eta = X @ beta
    mu = np.exp(np.clip(eta, -20, 20))
    mu = np.maximum(mu, 1e-10)

    # Log-likelihood
    llf = np.sum(y * np.log(mu) - mu - np.array([np.sum(np.log(np.arange(1, int(yi) + 1))) for yi in y]))

    # Deviance
    y_safe = np.maximum(y, 1e-10)
    deviance = 2 * np.sum(y * np.log(y_safe / mu) - (y - mu))

    # Standard errors from Fisher information
    try:
        W = np.diag(mu)
        cov = np.linalg.inv(X.T @ W @ X + 1e-10 * np.eye(p))
        bse = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        bse = np.full(p, np.nan)

    # AIC
    aic = -2 * llf + 2 * p

    return PoissonGLMResult(beta, bse, llf, aic, deviance)


def run_tou_glm(sessions, hourly, output_dir):
    """
    Test whether TOU price adds explanatory power for arrivals beyond hour-of-day.

    Model 1 (restricted): arrivals ~ hour_dummies (23 dummies for hours 1–23)
    Model 2 (full):        arrivals ~ hour_dummies + tou_price

    Likelihood ratio test compares Model 2 vs Model 1.

    This is done per-station because rate levels differ.
    Also done pooled as a power check.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 4: TOU PRICE GLM")
    print("=" * 70)

    station_col = identify_columns(hourly, 'station')
    arrivals_col = identify_columns(hourly, 'arrivals')
    hour_col = identify_columns(hourly, 'hour')
    price_col = identify_columns(hourly, 'tou_price')

    # If no price column in hourly, try to merge from sessions
    if price_col is None:
        print("[INFO] No price column in hourly data. Attempting to merge from sessions.")
        price_col_sess = identify_columns(sessions, 'tou_price')
        hour_col_sess = identify_columns(sessions, 'hour')
        if hour_col_sess is None:
            # Compute hour from start time
            time_col = identify_columns(sessions, 'start_time')
            if time_col:
                sessions['_hour'] = pd.to_datetime(sessions[time_col]).dt.hour
                hour_col_sess = '_hour'

        if price_col_sess is not None and hour_col_sess is not None:
            # Get mean price per hour (TOU is deterministic by hour, so this is fine)
            price_by_hour = sessions.groupby(hour_col_sess)[price_col_sess].mean().reset_index()
            price_by_hour.columns = ['_merge_hour', '_tou_price']
            hourly = hourly.merge(price_by_hour, left_on=hour_col, right_on='_merge_hour', how='left')
            price_col = '_tou_price'
        else:
            # Construct TOU schedule from known Jiaxing structure
            print("[INFO] Constructing TOU prices from known Jiaxing schedule.")
            # Jiaxing TOU: Valley ~0.38, Peak ~0.90, Super-peak ~1.21
            # Approximate hour mapping (from Week 1 report)
            tou_map = {}
            valley_hours = [0, 1, 2, 3, 4, 5, 6, 7]  # rough; will be adjusted
            peak_hours = [8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            super_peak_hours = [12, 13]
            for h in range(24):
                if h in super_peak_hours:
                    tou_map[h] = 1.21
                elif h in peak_hours:
                    tou_map[h] = 0.90
                else:
                    tou_map[h] = 0.38
            hourly['_tou_price'] = hourly[hour_col].map(tou_map)
            price_col = '_tou_price'
            print("[WARN] Using approximate TOU schedule. Results should be validated against actual prices.")

    if station_col is None or arrivals_col is None:
        hourly = compute_hourly_counts(sessions)
        station_col, arrivals_col, hour_col = 'station', 'arrivals', 'hour'

    stations = sorted(hourly[station_col].unique())
    results = []

    for station in stations:
        df_s = hourly[hourly[station_col] == station].copy()
        df_s = df_s.dropna(subset=[arrivals_col, price_col, hour_col])

        if len(df_s) < 100:
            results.append({
                'station': station, 'lr_statistic': np.nan, 'lr_p_value': np.nan,
                'price_coef': np.nan, 'price_se': np.nan, 'note': 'Insufficient data'
            })
            continue

        y = df_s[arrivals_col].values.astype(float)

        # Price is not separately identifiable if it is a deterministic
        # function of hour. Do not fit or interpret a rank-deficient LR test.
        price_variation_within_hour = df_s.groupby(hour_col)[price_col].nunique()
        if len(price_variation_within_hour) == 0 or \
                price_variation_within_hour.max() <= 1:
            results.append({
                'station': station,
                'lr_statistic': np.nan,
                'lr_p_value': np.nan,
                'price_coef': np.nan,
                'price_se': np.nan,
                'price_significant_5pct': np.nan,
                'collinear_flag': True,
                'model1_aic': np.nan,
                'model2_aic': np.nan,
                'aic_improvement': np.nan,
                'model1_deviance': np.nan,
                'model2_deviance': np.nan,
                'note': ('Not identifiable: price is deterministic within '
                         'hour; exogenous within-hour price variation required'),
            })
            print(f"  {station}: PRICE EFFECT NOT IDENTIFIABLE FROM HOUR")
            continue

        # Hour dummies (drop hour 0 as reference)
        hour_dummies = pd.get_dummies(df_s[hour_col], prefix='h', drop_first=True, dtype=float)

        # Model 1: hour dummies only
        X1 = np.column_stack([np.ones(len(y)), hour_dummies.values])
        try:
            if HAS_STATSMODELS:
                model1 = GLM(y, X1, family=PoissonFamily()).fit()
            else:
                model1 = fit_poisson_glm_irls(y, X1)
        except Exception as e:
            results.append({
                'station': station, 'lr_statistic': np.nan, 'lr_p_value': np.nan,
                'price_coef': np.nan, 'price_se': np.nan, 'note': f'Model1 failed: {e}'
            })
            continue

        # Model 2: hour dummies + price
        price_vals = df_s[price_col].values.astype(float).reshape(-1, 1)
        X2 = np.column_stack([np.ones(len(y)), hour_dummies.values, price_vals])
        try:
            if HAS_STATSMODELS:
                model2 = GLM(y, X2, family=PoissonFamily()).fit()
            else:
                model2 = fit_poisson_glm_irls(y, X2)
        except Exception as e:
            results.append({
                'station': station, 'lr_statistic': np.nan, 'lr_p_value': np.nan,
                'price_coef': np.nan, 'price_se': np.nan, 'note': f'Model2 failed: {e}'
            })
            continue

        # Likelihood ratio test
        lr_stat = max(0, -2 * (model1.llf - model2.llf))  # clamp to 0 for numerical safety
        lr_p = 1 - chi2.cdf(lr_stat, df=1) if lr_stat > 0 else 1.0

        # Price coefficient is the last one
        price_coef = model2.params[-1]
        price_se = model2.bse[-1]

        # Detect collinearity: price is a deterministic function of hour,
        # so 23 hour dummies already span the 3 TOU tiers. Flag this.
        collinear = False
        note = ''
        if abs(price_coef) > 100 or price_se > 100 or np.isnan(price_se):
            collinear = True
            note = 'Collinear: price is a deterministic function of hour dummies'
        if lr_stat < 1e-6 and lr_p > 0.99:
            collinear = True
            note = 'Collinear: LR~0, price adds no information beyond hour dummies'

        results.append({
            'station': station,
            'lr_statistic': lr_stat,
            'lr_p_value': lr_p,
            'price_coef': np.nan if collinear else price_coef,
            'price_se': np.nan if collinear else price_se,
            'price_significant_5pct': False if collinear else lr_p < 0.05,
            'collinear_flag': collinear,
            'model1_aic': model1.aic,
            'model2_aic': model2.aic,
            'aic_improvement': model1.aic - model2.aic,
            'model1_deviance': model1.deviance,
            'model2_deviance': model2.deviance,
            'note': note
        })

        if collinear:
            print(f"  {station}: COLLINEAR (price spanned by hour dummies; LR={lr_stat:.2e})")
        else:
            sig_str = "SIGNIFICANT" if lr_p < 0.05 else "not significant"
            print(f"  {station}: LR={lr_stat:.2f}, p={lr_p:.4f} ({sig_str}), "
                  f"price_coef={price_coef:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'tou_glm_results.csv', index=False)
    print(f"\n[SAVED] tou_glm_results.csv")

    # Summary
    valid = df.dropna(subset=['lr_p_value'])
    n_stations = len(df)
    n_collinear = df['collinear_flag'].sum() if 'collinear_flag' in df.columns else 0
    n_sig = valid['price_significant_5pct'].sum() if 'price_significant_5pct' in valid.columns else 0
    print(f"\n[SUMMARY] TOU price explanatory power:")
    if n_collinear > 0:
        print(f"  {int(n_collinear)}/{n_stations} stations flagged as collinear "
              f"(price is a deterministic function of hour)")
    print(f"  {int(n_sig)}/{len(valid)} stations show significant price effect at alpha=0.05")
    if n_stations > 0 and n_collinear == n_stations:
        print(f"  [NOTE] Price is perfectly determined by hour-of-day in the Jiaxing TOU schedule.")
        print(f"         23 hour dummies already capture all TOU tier variation.")
        print(f"         This confirms that hour-of-day alone is sufficient for arrival modeling.")
        print(f"         No additional price variable is needed in the NHPP specification.")
    elif 'price_coef' in valid.columns:
        non_collinear = valid[~valid.get('collinear_flag', False)]
        if len(non_collinear) > 0:
            mean_coef = non_collinear['price_coef'].mean()
            print(f"  Mean price coefficient (non-collinear): {mean_coef:.4f} "
                  f"({'positive = more arrivals at higher price' if mean_coef > 0 else 'negative = fewer arrivals at higher price'})")

    return df


# =============================================================================
# FIGURES
# =============================================================================

def plot_gof_summary(gof_df, output_dir):
    """Plot summary of GOF test results across stations."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    stations = gof_df['station'].values
    x = np.arange(len(stations))

    # Chi-squared p-values
    ax = axes[0]
    pvals = gof_df['chi2_p_value'].values
    colors = ['#d62728' if p < 0.05 else '#2ca02c' for p in pvals]
    ax.barh(x, -np.log10(np.clip(pvals, 1e-300, 1)), color=colors)
    ax.axvline(-np.log10(0.05), color='black', linestyle='--', label='α=0.05')
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel('-log₁₀(p-value)')
    ax.set_title('Chi-squared Test\n(Hourly Counts vs Poisson)')
    ax.legend(fontsize=9)

    # KS p-values
    ax = axes[1]
    pvals = gof_df['ks_p_value'].values
    colors = ['#d62728' if p < 0.05 else '#2ca02c' for p in pvals]
    ax.barh(x, -np.log10(np.clip(pvals, 1e-300, 1)), color=colors)
    ax.axvline(-np.log10(0.05), color='black', linestyle='--', label='α=0.05')
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel('-log₁₀(p-value)')
    ax.set_title('K-S Test\n(IATs vs Exponential)')
    ax.legend(fontsize=9)

    # AD test statistics vs critical values
    ax = axes[2]
    ad_stats_raw = gof_df['ad_statistic'].values.astype(float)
    ad_crits = gof_df['ad_critical_5pct'].values
    finite_stats = ad_stats_raw[np.isfinite(ad_stats_raw)]
    finite_crits = ad_crits[np.isfinite(ad_crits)]
    fallback_cap = (
        max(float(np.max(finite_stats)) if len(finite_stats) else 0.0,
            float(np.max(finite_crits)) if len(finite_crits) else 1.0)
        * 1.10
    )
    ad_stats = np.where(np.isfinite(ad_stats_raw), ad_stats_raw, fallback_cap)
    colors = ['#d62728' if s > c else '#2ca02c' for s, c in zip(ad_stats_raw, ad_crits)]
    ax.barh(x, ad_stats, color=colors)
    # Plot critical values as points
    ax.scatter(ad_crits, x, color='black', marker='|', s=100, zorder=5, label='Critical (5%)')
    saturated_mask = ~np.isfinite(ad_stats_raw)
    for idx in np.where(saturated_mask)[0]:
        ax.text(ad_stats[idx], idx, ' saturated', va='center', fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel('AD Statistic (saturated values capped for display)')
    ax.set_title('Anderson-Darling Test\n(IATs vs Exponential)')
    ax.legend(fontsize=9)

    plt.suptitle('Poisson Goodness-of-Fit Tests by Station', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_gof_summary.png')
    plt.close()
    print(f"[SAVED] figures/fig1_gof_summary.png")


def plot_conditional_dispersion(disp_df, output_dir):
    """Plot unconditional vs conditional dispersion indices."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    stations = disp_df['station'].values
    x = np.arange(len(stations))
    width = 0.25

    ax.barh(x - width, disp_df['di_unconditional'].values, width,
            label='Unconditional (all hours pooled)', color='#d62728', alpha=0.8)
    ax.barh(x, disp_df['di_cond_hour_mean'].values, width,
            label='Conditional on hour-of-day', color='#ff7f0e', alpha=0.8)
    if 'di_cond_hour_day_mean' in disp_df.columns:
        vals = disp_df['di_cond_hour_day_mean'].values
        ax.barh(x + width, vals, width,
                label='Conditional on hour × day-type', color='#2ca02c', alpha=0.8)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='Poisson (DI=1)')
    ax.axvline(2.0, color='gray', linestyle=':', linewidth=1, label='DI=2 threshold')
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel('Dispersion Index (Variance / Mean)')
    ax.set_title('Dispersion Index: Unconditional vs Conditional')
    ax.legend(loc='lower right', fontsize=9)
    # Use log scale if range is large
    if disp_df['di_unconditional'].max() > 50:
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_conditional_dispersion.png')
    plt.close()
    print(f"[SAVED] figures/fig2_conditional_dispersion.png")


def plot_nhpp_rate_functions(rate_df, output_dir, max_stations_per_fig=7):
    """Plot NHPP rate functions λ(h) for each station."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    stations = sorted(rate_df['station'].unique())
    n_stations = len(stations)

    # Split into panels if many stations
    n_figs = (n_stations + max_stations_per_fig - 1) // max_stations_per_fig

    for fig_idx in range(n_figs):
        start = fig_idx * max_stations_per_fig
        end = min(start + max_stations_per_fig, n_stations)
        subset = stations[start:end]

        n_cols = min(3, len(subset))
        n_rows = (len(subset) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for i, station in enumerate(subset):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            df_s = rate_df[rate_df['station'] == station].sort_values('hour')
            hours = df_s['hour'].values
            rates = df_s['lambda_mean'].values
            ci_low = df_s['lambda_ci_low'].values
            ci_high = df_s['lambda_ci_high'].values

            ax.step(hours, rates, where='mid', color='#1f77b4', linewidth=2)
            ax.fill_between(hours, ci_low, ci_high, alpha=0.2, color='#1f77b4', step='mid')
            ax.set_title(f'{station}', fontsize=10)
            ax.set_xlabel('Hour')
            ax.set_ylabel('λ(h) [arrivals/hour]')
            ax.set_xlim(-0.5, 23.5)
            ax.set_xticks([0, 4, 8, 12, 16, 20])
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(subset), n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(f'NHPP Rate Functions λ(h) with 95% CI (Part {fig_idx + 1})',
                     fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(fig_dir / f'fig3_nhpp_rates_part{fig_idx + 1}.png')
        plt.close()
        print(f"[SAVED] figures/fig3_nhpp_rates_part{fig_idx + 1}.png")


def plot_tou_glm(glm_df, output_dir):
    """Plot TOU GLM results."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    valid = glm_df.dropna(subset=['price_coef'])
    if len(valid) == 0:
        print("[WARN] No valid GLM results to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    stations = valid['station'].values
    x = np.arange(len(stations))

    # Price coefficient with error bars
    ax = axes[0]
    coefs = valid['price_coef'].values
    ses = valid['price_se'].values
    colors = ['#d62728' if p < 0.05 else '#7f7f7f'
              for p in valid['lr_p_value'].values]
    ax.barh(x, coefs, xerr=1.96 * ses, color=colors, alpha=0.8, capsize=3)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel('Price Coefficient (Poisson GLM)')
    ax.set_title('TOU Price Effect\n(controlling for hour-of-day)')

    # LR test p-values
    ax = axes[1]
    pvals = valid['lr_p_value'].values
    colors = ['#d62728' if p < 0.05 else '#7f7f7f' for p in pvals]
    ax.barh(x, -np.log10(np.clip(pvals, 1e-300, 1)), color=colors, alpha=0.8)
    ax.axvline(-np.log10(0.05), color='black', linestyle='--', label='α=0.05')
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel('-log₁₀(p-value)')
    ax.set_title('Likelihood Ratio Test\n(Price added to Hour-only model)')
    ax.legend(fontsize=9)

    plt.suptitle('TOU Price Explanatory Power (Poisson GLM)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_tou_glm.png')
    plt.close()
    print(f"[SAVED] figures/fig4_tou_glm.png")


def plot_iat_distributions(iat, output_dir, n_stations=4):
    """Plot IAT distributions for a subset of stations with exponential overlay."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    station_col = identify_columns(iat, 'station') or 'station'
    stations = sorted(iat[station_col].unique())

    # Pick representative stations (first, middle, last, and one more)
    if len(stations) > n_stations:
        indices = np.linspace(0, len(stations) - 1, n_stations, dtype=int)
        subset = [stations[i] for i in indices]
    else:
        subset = stations

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, station in enumerate(subset[:4]):
        ax = axes[i]
        vals = iat.loc[iat[station_col] == station, 'iat_seconds'].values / 60  # to minutes
        vals = vals[vals < np.percentile(vals, 99)]  # trim extreme tail for visibility

        ax.hist(vals, bins=60, density=True, alpha=0.6, color='#1f77b4', label='Observed')

        # Overlay fitted exponential
        mu = vals.mean()
        x_exp = np.linspace(0, vals.max(), 200)
        y_exp = expon.pdf(x_exp, scale=mu)
        ax.plot(x_exp, y_exp, 'r-', linewidth=2, label=f'Exponential(μ={mu:.1f}min)')

        ax.set_title(f'{station}', fontsize=11)
        ax.set_xlabel('Inter-arrival time (minutes)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.set_xlim(0, np.percentile(vals, 99))

    plt.suptitle('Inter-Arrival Time Distributions vs Exponential', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig5_iat_distributions.png')
    plt.close()
    print(f"[SAVED] figures/fig5_iat_distributions.png")


def plot_growth_trend_nhpp(quarterly_df, output_dir):
    """Plot quarterly evolution of rate functions for representative stations."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    if quarterly_df is None or len(quarterly_df) == 0:
        print("[WARN] No quarterly data; skipping growth trend plot.")
        return

    stations = sorted(quarterly_df['station'].unique())
    # Pick 4 representative stations
    if len(stations) > 4:
        indices = np.linspace(0, len(stations) - 1, 4, dtype=int)
        subset = [stations[i] for i in indices]
    else:
        subset = stations[:4]

    quarters = sorted(quarterly_df['quarter'].unique())
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(quarters)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, station in enumerate(subset):
        ax = axes[i]
        df_s = quarterly_df[quarterly_df['station'] == station]

        for j, quarter in enumerate(quarters):
            df_q = df_s[df_s['quarter'] == quarter].sort_values('hour')
            if len(df_q) > 0:
                ax.plot(df_q['hour'], df_q['lambda_mean'],
                        color=cmap[j], alpha=0.8, linewidth=1.5, label=quarter)

        ax.set_title(f'{station}', fontsize=11)
        ax.set_xlabel('Hour')
        ax.set_ylabel('λ(h)')
        ax.set_xlim(-0.5, 23.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    plt.suptitle('Quarterly Evolution of NHPP Rate Functions\n(capturing 2.7× growth trend)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig6_quarterly_nhpp.png')
    plt.close()
    print(f"[SAVED] figures/fig6_quarterly_nhpp.png")


# =============================================================================
# SANITY CHECKS
# =============================================================================

def run_sanity_checks(gof_df, disp_df, rate_df, glm_df, sessions, hourly):
    """
    Post-analysis sanity checks as required by project method discipline.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    # Check 1: Number of stations
    n_stations = len(gof_df)
    checks_total += 1
    if n_stations == 13:
        print(f"  [PASS] Station count: {n_stations} (expected 13)")
        checks_passed += 1
    else:
        print(f"  [WARN] Station count: {n_stations} (expected 13)")

    # Check 2: All chi2 tests should reject (given massive overdispersion from Week 2)
    n_reject = gof_df['chi2_reject_5pct'].sum()
    checks_total += 1
    if n_reject == n_stations:
        print(f"  [PASS] Chi2 rejects for all {n_stations} stations (consistent with DI >> 1)")
        checks_passed += 1
    else:
        print(f"  [FLAG] Chi2 rejects for {int(n_reject)}/{n_stations} stations. "
              f"Non-rejection would be surprising given Week 2 DIs of 87-1141.")

    # Check 3: Conditional DI should be lower than unconditional
    checks_total += 1
    if (disp_df['di_cond_hour_mean'] < disp_df['di_unconditional']).all():
        print(f"  [PASS] Conditional DI < Unconditional DI for all stations")
        checks_passed += 1
    else:
        violations = (disp_df['di_cond_hour_mean'] >= disp_df['di_unconditional']).sum()
        print(f"  [FLAG] {violations} station(s) have conditional DI >= unconditional. "
              f"This would indicate within-hour overdispersion beyond hour-of-day effects.")

    # Check 4: NHPP rate functions should sum to approximately the daily arrival rate
    station_col = identify_columns(hourly, 'station')
    arrivals_col = identify_columns(hourly, 'arrivals')
    if station_col and arrivals_col:
        checks_total += 1
        all_close = True
        for station in rate_df['station'].unique():
            nhpp_daily = rate_df.loc[rate_df['station'] == station, 'lambda_mean'].sum()
            # Get empirical daily mean
            hourly_station = hourly[hourly[station_col] == station]
            date_col = identify_columns(hourly_station, 'date')
            if date_col:
                daily_mean = hourly_station.groupby(date_col)[arrivals_col].sum().mean()
            else:
                daily_mean = hourly_station[arrivals_col].sum() / max(1, hourly_station[arrivals_col].count() / 24)
            if abs(nhpp_daily - daily_mean) > 0.1 * daily_mean:
                print(f"  [FLAG] {station}: NHPP sum={nhpp_daily:.1f}, empirical daily={daily_mean:.1f}")
                all_close = False
        if all_close:
            print(f"  [PASS] NHPP rate sums match empirical daily means (within 10%)")
            checks_passed += 1

    # Check 5: Price coefficients — collinearity expected
    if 'collinear_flag' in glm_df.columns:
        n_collinear = glm_df['collinear_flag'].sum()
        checks_total += 1
        if n_collinear == len(glm_df):
            print(f"  [PASS] All {len(glm_df)} stations show collinearity "
                  f"(price determined by hour; expected with Jiaxing TOU structure)")
            checks_passed += 1
        elif 'price_coef' in glm_df.columns:
            valid_coefs = glm_df[~glm_df['collinear_flag']].dropna(subset=['price_coef'])
            n_positive = (valid_coefs['price_coef'] > 0).sum()
            print(f"  [INFO] {int(n_collinear)} collinear, {len(valid_coefs)} non-collinear. "
                  f"Price coef positive for {n_positive}/{len(valid_coefs)} non-collinear stations.")
            checks_passed += 1
    elif 'price_coef' in glm_df.columns:
        valid_coefs = glm_df.dropna(subset=['price_coef'])
        n_positive = (valid_coefs['price_coef'] > 0).sum()
        checks_total += 1
        print(f"  [INFO] Price coefficient positive for {n_positive}/{len(valid_coefs)} stations "
              f"(positive = more arrivals at higher prices, consistent with r=+0.518)")
        if n_positive > len(valid_coefs) / 2:
            checks_passed += 1

    # Check 6: Flag unusually strong results
    checks_total += 1
    if 'lr_p_value' in glm_df.columns:
        n_extreme = (glm_df['lr_p_value'] < 1e-10).sum()
        if n_extreme > len(glm_df) / 2:
            print(f"  [FLAG] {n_extreme} stations have extremely low GLM p-values (< 1e-10). "
                  f"This could indicate genuine effect or model misspecification (overdispersion).")
        else:
            print(f"  [PASS] GLM p-values in expected range")
            checks_passed += 1

    print(f"\n[SANITY] {checks_passed}/{checks_total} checks passed")

    # Failure modes
    print("\n--- POTENTIAL FAILURE MODES ---")
    print("  1. Chi-squared test has low power if hourly data has many zero-count bins")
    print("  2. KS p-values use a fitted-parameter parametric bootstrap; finite bootstrap "
          "resolution and deterministic subsampling still limit precision")
    print("  3. Poisson GLM assumes equidispersion; overdispersion inflates significance")
    print("     -> If all stations show significant price effect, re-check with Negative Binomial GLM")
    print("  4. Conditional DI may remain >1 due to day-to-day variation not captured by hour-of-day")
    print("  5. TOU prices are deterministic per hour, so price variable is perfectly collinear with")
    print("     a subset of hour dummies. The GLM tests whether price adds anything BEYOND the")
    print("     hour-of-day pattern, which is a stringent test.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    configure_console_output()
    parser = argparse.ArgumentParser(description='Week 3: Arrival Process Characterization')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing Week 1 output files (parquet/csv)')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'week3_results'),
                        help='Output directory for results')
    parser.add_argument('--bootstrap-iters', type=int, default=300,
                        help='Bootstrap iterations for NHPP CI per station-hour (default: 300)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("WEEK 3: ARRIVAL PROCESS CHARACTERIZATION")
    print("=" * 70)

    # Load data
    sessions, hourly, iat = load_data(args.data_dir)

    # Component 1: Poisson GOF tests
    gof_df = run_poisson_gof_tests(sessions, hourly, iat, output_dir)

    # Component 2: Conditional dispersion
    disp_df = run_conditional_dispersion(sessions, hourly, output_dir)

    # Component 3: NHPP fitting
    rate_df, quarterly_df = fit_nhpp_rate_functions(
        sessions, hourly, output_dir, bootstrap_iters=max(0, args.bootstrap_iters)
    )

    # Component 4: TOU GLM
    glm_df = run_tou_glm(sessions, hourly, output_dir)

    # Generate all figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_gof_summary(gof_df, output_dir)
    plot_conditional_dispersion(disp_df, output_dir)
    plot_nhpp_rate_functions(rate_df, output_dir)
    plot_tou_glm(glm_df, output_dir)
    plot_iat_distributions(iat, output_dir)
    if quarterly_df is not None and len(quarterly_df) > 0:
        plot_growth_trend_nhpp(quarterly_df, output_dir)

    # Sanity checks
    run_sanity_checks(gof_df, disp_df, rate_df, glm_df, sessions, hourly)

    # Save metadata
    metadata = {
        'week': 3,
        'component': 'Arrival Process Characterization',
        'bootstrap_iters': int(max(0, args.bootstrap_iters)),
        'n_stations': len(gof_df),
        'n_sessions_total': len(sessions),
        'files_produced': [
            'poisson_gof_tests.csv',
            'conditional_dispersion.csv',
            'nhpp_rate_functions.csv',
            'nhpp_rate_functions_quarterly.csv',
            'tou_glm_results.csv',
        ],
        'figures_produced': [f for f in os.listdir(output_dir / 'figures') if f.endswith('.png')],
    }
    with open(output_dir / 'week3_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("WEEK 3 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    print(f"Files: {', '.join(metadata['files_produced'])}")
    print(f"Figures: {len(metadata['figures_produced'])} PNG files")


if __name__ == '__main__':
    main()
