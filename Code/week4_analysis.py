"""
Week 4: Service Time Fitting + ACN Cross-Validation + Phase 2 Kickoff
======================================================================
Phase 1b completion + Phase 2a start

Components:
  1. Service time distribution fitting per charger type
     - Exponential, Gamma, Lognormal, Weibull via MLE (floc=0)
     - AIC/BIC model selection
     - QQ plots (3 charger types x 4 distributions)
     - 2-component mixture exploration for L2
     - Mixed vs DC Fast mergeability test
  2. ACN JPL cross-validation
     - K-S, dispersion, conditional Poisson tests on ACN data
     - Service time fitting on ACN L2 sessions
     - Comparison table: Jiaxing vs ACN distributional patterns
  3. Erlang-C and M/G/s fleet sizing
     - Erlang-C P(wait) for s = 1..50 per representative station
     - M/G/s approximation: P(wait)_MGs approx P(wait)_MMs x (1+CV^2)/2
     - Fleet size gap: s* where P(wait) < 5% under M/M/s vs M/G/s
  4. XGBoost hourly arrival forecasting
     - Feature matrix: hour, dow, month, is_weekend, TOU_tier,
       temperature, precipitation, lag_1, lag_24, lag_168, rolling_7d_mean
     - Train/test: last 4 months (Sep-Dec 2021) as test
     - 5-fold CV on training set
     - Baseline models: last-week-same-hour, historical mean
     - Feature importance + residual analysis

Assumptions:
  - Service time fitting uses NON-FAULT, NON-ZERO-ENERGY sessions only
  - Arrival modeling includes ALL sessions (zero-energy + faults)
  - Per-station analysis unless explicitly stated
  - M/G/s uses station charger-type mix + global type-level fitted moments (CV source logged)
  - XGBoost holdout is Sep-Dec 2021; absolute errors may be elevated under demand growth
  - Charger types: DC_Fast (>30kW), Level_2 (<22kW), Mixed (22-30kW)
  - Locked EDA values: DC Fast mean=35min CV=0.57, L2 mean=92min CV=1.39,
    Mixed mean=43min CV=0.60

Input files:
  - jiaxing_clean.parquet   (session-level, 441k rows)
  - jiaxing_hourly.parquet  (station-hour aggregation)
  - jiaxing_iat.parquet     (inter-arrival times)
  - acn_clean.parquet       (ACN JPL dataset, if available)
  - week3_results/nhpp_rate_functions.csv  (NHPP rates from Week 3)

Output files:
  - week4_results/service_time_fits.csv
  - week4_results/service_time_summary.json   (parameter_summary.json input)
  - week4_results/acn_crossval.csv
  - week4_results/erlang_c_results.csv        (full s=1..50 sweep, one row per station-s)
  - week4_results/mgs_comparison.csv          (per-station s* comparison M/M/s vs M/G/s)
  - week4_results/xgboost_results.json
  - week4_results/xgboost_feature_importance.csv
  - week4_results/parameter_summary.json      (consolidated Phase 1 output)
  - week4_results/figures/*.png

Usage:
  python week4_analysis.py --data-dir ./data --week3-dir ./week3_results --output-dir ./week4_results

Date: Week 4, Mar 2026
"""

import argparse
import json
import os
import sys
import warnings
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR, to_builtin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import (
    expon, gamma, lognorm, weibull_min,
    kstest, anderson, poisson, chi2,
)
from scipy.optimize import minimize

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[INFO] statsmodels not available. Erlang-C will use scipy only.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[INFO] xgboost not available. XGBoost component will be skipped.")

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[INFO] scikit-learn not available. ML evaluation will be limited.")


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


# =============================================================================
# UTILITY FUNCTIONS
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


def identify_columns(df, col_type):
    """Flexibly identify column names across possible naming conventions."""
    candidates = {
        'station': ['station', 'station_name', 'station_id', 'Station'],
        'start_time': ['start_time', 'start_datetime', 'StartTime', 'begin_time',
                        'connectionTime'],
        'end_time': ['end_time', 'end_datetime', 'EndTime', 'disconnectTime'],
        'hour': ['hour', 'Hour', 'hour_of_day'],
        'arrivals': ['arrivals', 'arrival_count', 'count', 'n_arrivals', 'num_sessions'],
        'date': ['date', 'date_dt', 'Date', 'day'],
        'duration': ['charging_duration_min', 'duration_min', 'duration_minutes',
                      'service_time_min'],
        'energy': ['energy_kwh', 'kWhDelivered', 'total_energy_kwh', 'energy'],
        'charger_type': ['charger_type', 'charger_category', 'evse_type'],
        'is_abnormal': ['is_abnormal', 'is_fault', 'fault'],
        'flag_zero_energy': ['flag_zero_energy', 'zero_energy', 'is_zero_energy'],
        'tou_price': ['tou_electricity_price', 'tou_electricity_price_yuan_kwh',
                       'tou_price', 'price', 'electricity_price'],
        'tou_tier': ['tou_tier', 'tou_period', 'price_tier'],
        'temperature': ['temperature', 'temp', 'avg_temp', 'temperature_avg'],
        'precipitation': ['precipitation', 'precip', 'rainfall'],
        'flexibility': ['flexibility_tier', 'flexibility'],
        'dow': ['day_of_week', 'dow', 'dayofweek'],
        'is_weekend': ['is_weekend'],
        'month': ['month'],
        'iat_seconds': ['iat_seconds', 'iat_sec'],
    }
    for c in candidates.get(col_type, []):
        if c in df.columns:
            return c
    return None


def read_file(data_dir, name_base):
    """Try parquet first, then CSV."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / f'{name_base}.parquet'
    csv_path = data_dir / f'{name_base}.csv'
    engine = detect_parquet_engine()
    if parquet_path.exists() and engine is not None:
        try:
            return pd.read_parquet(parquet_path, engine=engine), parquet_path.name
        except Exception as e:
            print(f"[WARN] Failed reading {parquet_path.name}: {e}")
    elif parquet_path.exists() and engine is None:
        print(f"[WARN] No parquet engine. Looking for CSV fallback for {name_base}.")
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path.name
    return None, None


def save_csv(df, output_dir, filename):
    """Save DataFrame to CSV."""
    path = Path(output_dir) / filename
    df.to_csv(path, index=False)
    print(f"  [SAVED] {path}")
    return path


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jiaxing(data_dir):
    """Load Jiaxing session-level data."""
    sessions, src = read_file(data_dir, 'jiaxing_clean')
    if sessions is None:
        raise FileNotFoundError(
            f"Cannot find jiaxing_clean.parquet or .csv in {data_dir}")

    # Parse datetime columns
    for col in ['start_time', 'end_time', 'order_created_time', 'payment_time', 'date']:
        if col in sessions.columns:
            sessions[col] = pd.to_datetime(sessions[col], errors='coerce')

    # Ensure key derived columns exist
    if 'hour_of_day' not in sessions.columns:
        time_col = identify_columns(sessions, 'start_time')
        if time_col:
            sessions['hour_of_day'] = pd.to_datetime(
                sessions[time_col], errors='coerce').dt.hour

    if 'day_of_week' not in sessions.columns:
        time_col = identify_columns(sessions, 'start_time')
        if time_col:
            sessions['day_of_week'] = pd.to_datetime(
                sessions[time_col], errors='coerce').dt.dayofweek

    if 'month' not in sessions.columns:
        time_col = identify_columns(sessions, 'start_time')
        if time_col:
            sessions['month'] = pd.to_datetime(
                sessions[time_col], errors='coerce').dt.month

    if 'is_weekend' not in sessions.columns and 'day_of_week' in sessions.columns:
        sessions['is_weekend'] = sessions['day_of_week'].isin([5, 6]).astype(int)

    print(f"[DATA] Jiaxing sessions: {len(sessions):,} rows ({src})")
    return sessions


def load_hourly(data_dir):
    """Load Jiaxing hourly aggregation."""
    hourly, src = read_file(data_dir, 'jiaxing_hourly')
    if hourly is None:
        print("[WARN] jiaxing_hourly not found. Will compute from sessions.")
        return None
    # Parse dates
    for col in ['date', 'date_dt']:
        if col in hourly.columns:
            hourly[col] = pd.to_datetime(hourly[col], errors='coerce')
    print(f"[DATA] Hourly: {len(hourly):,} rows ({src})")
    return hourly


def load_acn(data_dir):
    """Load ACN JPL dataset."""
    acn, src = read_file(data_dir, 'acn_clean')
    if acn is None:
        print("[INFO] ACN data not found. ACN cross-validation will be skipped.")
        return None

    for col in ['connectionTime', 'disconnectTime', 'doneChargingTime',
                'start_time', 'end_time']:
        if col in acn.columns:
            acn[col] = pd.to_datetime(acn[col], errors='coerce')

    print(f"[DATA] ACN: {len(acn):,} rows ({src})")
    return acn


def load_nhpp_rates(week3_dir):
    """Load NHPP rate functions from Week 3."""
    path = Path(week3_dir) / 'nhpp_rate_functions.csv'
    if path.exists():
        rates = pd.read_csv(path)
        print(f"[DATA] NHPP rates: {len(rates):,} rows")
        return rates
    print("[INFO] NHPP rate functions not found. Erlang-C will use empirical peak rates.")
    return None


# =============================================================================
# COMPONENT 1: SERVICE TIME DISTRIBUTION FITTING
# =============================================================================

def estimate_clamped_sampler_moments(dist_name, params, seed,
                                     n_draws=200_000):
    """Estimate moments of the exact sampler used by the DES (0.5..720 min)."""
    rng = np.random.default_rng(seed)
    if dist_name == 'mixture_2lognorm':
        component_1 = rng.random(n_draws) < params.get('pi1', 0.5)
        draws = np.empty(n_draws)
        draws[component_1] = rng.lognormal(
            params.get('mu1', 4.0), params.get('sigma1', 1.0),
            component_1.sum())
        draws[~component_1] = rng.lognormal(
            params.get('mu2', 4.0), params.get('sigma2', 0.5),
            (~component_1).sum())
    elif dist_name == 'gamma':
        draws = rng.gamma(params.get('a', 1.0), params.get('scale', 35.0),
                          n_draws)
    elif dist_name == 'weibull':
        draws = (rng.weibull(params.get('c', 1.0), n_draws)
                 * params.get('scale', 44.0))
    elif dist_name == 'lognormal':
        draws = rng.lognormal(
            np.log(params.get('scale', 50.0)), params.get('s', 1.0),
            n_draws)
    else:
        draws = rng.exponential(params.get('scale', 50.0), n_draws)
    draws = np.clip(draws, 0.5, 720.0)
    return {
        'clamp_min': 0.5,
        'clamp_max': 720.0,
        'estimated_mean_min': float(draws.mean()),
        'estimated_std_min': float(draws.std()),
        'estimated_cv': float(draws.std() / draws.mean()),
        'estimation_draws': int(n_draws),
        'estimation_seed': int(seed),
    }

def filter_service_time_data(sessions):
    """
    Filter to non-fault, non-zero-energy sessions for service time fitting.

    Rules:
      - Exclude is_abnormal == 1  (faults)
      - Exclude flag_zero_energy == 1  (zero-energy sessions)
      - Exclude duration <= 0 or duration > 720 min (12h cap, same as EDA)
      - Include all charger types
    """
    df = sessions.copy()
    n_start = len(df)

    # Exclude faults
    abn_col = identify_columns(df, 'is_abnormal')
    if abn_col:
        mask_fault = df[abn_col] == 1
        n_fault = mask_fault.sum()
        df = df[~mask_fault]
    else:
        n_fault = 0
        print("  [WARN] is_abnormal column not found. Cannot exclude faults.")

    # Exclude zero-energy
    ze_col = identify_columns(df, 'flag_zero_energy')
    if ze_col:
        mask_ze = df[ze_col] == 1
        n_ze = mask_ze.sum()
        df = df[~mask_ze]
    else:
        # Fallback: use energy column
        energy_col = identify_columns(df, 'energy')
        if energy_col:
            mask_ze = pd.to_numeric(df[energy_col], errors='coerce').fillna(0) <= 0
            n_ze = mask_ze.sum()
            df = df[~mask_ze]
        else:
            n_ze = 0
            print("  [WARN] Cannot identify zero-energy sessions.")

    # Duration filter
    dur_col = identify_columns(df, 'duration')
    if dur_col is None:
        raise ValueError(f"Cannot find duration column. Available: {list(df.columns)}")

    df[dur_col] = pd.to_numeric(df[dur_col], errors='coerce')
    mask_dur = (df[dur_col] > 0) & (df[dur_col] <= 720)
    n_dur_excluded = (~mask_dur).sum()
    df = df[mask_dur]

    n_final = len(df)
    print(f"  Filtering for service time fitting:")
    print(f"    Start:              {n_start:>10,}")
    print(f"    - Faults:           {n_fault:>10,}")
    print(f"    - Zero-energy:      {n_ze:>10,}")
    print(f"    - Duration invalid: {n_dur_excluded:>10,}")
    print(f"    = Remaining:        {n_final:>10,}  ({100*n_final/n_start:.1f}%)")

    return df, dur_col


def fit_single_distribution(data, dist_name, dist_obj):
    """
    Fit a single scipy distribution to data via MLE with floc=0.

    Returns dict with parameters, log-likelihood, AIC, BIC, KS statistic.
    """
    n = len(data)
    if n < 50:
        return None

    try:
        if dist_name == 'exponential':
            # Exponential has 1 parameter (scale = 1/lambda)
            params = dist_obj.fit(data, floc=0)
            k = 1  # only scale
        elif dist_name == 'gamma':
            params = dist_obj.fit(data, floc=0)
            k = 2  # shape, scale
        elif dist_name == 'lognormal':
            params = dist_obj.fit(data, floc=0)
            k = 2  # s (shape/sigma), scale (exp(mu))
        elif dist_name == 'weibull':
            params = dist_obj.fit(data, floc=0)
            k = 2  # c (shape), scale
        else:
            params = dist_obj.fit(data, floc=0)
            k = len(params) - 1  # subtract loc

        # Log-likelihood
        log_lik = np.sum(dist_obj.logpdf(data, *params))

        # Guard against -inf
        if not np.isfinite(log_lik):
            return None

        # AIC and BIC
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        # KS test
        ks_stat, ks_p = kstest(data, dist_obj.cdf, args=params)

        return {
            'dist_name': dist_name,
            'params': params,
            'k': k,
            'log_likelihood': log_lik,
            'aic': aic,
            'bic': bic,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'n': n,
        }

    except Exception as e:
        print(f"    [WARN] {dist_name} fit failed: {e}")
        return None


def fit_mixture_2component(data, max_iter=200, tol=1e-6):
    """
    Fit a 2-component lognormal mixture via EM algorithm.

    This is specifically for L2 chargers where single distributions
    may not capture the short-session + long-session bimodality.

    Returns dict with mixture parameters and fit metrics.
    """
    n = len(data)
    if n < 200:
        return None

    log_data = np.log(data)

    # Initialize with K-means-style split
    median_val = np.median(log_data)
    mask_low = log_data <= median_val
    mask_high = log_data > median_val

    # Initial parameters
    pi1 = mask_low.sum() / n
    mu1 = log_data[mask_low].mean()
    sigma1 = max(log_data[mask_low].std(), 0.1)
    mu2 = log_data[mask_high].mean()
    sigma2 = max(log_data[mask_high].std(), 0.1)

    prev_log_lik = -np.inf

    for iteration in range(max_iter):
        # E-step: compute responsibilities
        log_p1 = np.log(pi1 + 1e-300) + stats.norm.logpdf(log_data, mu1, sigma1)
        log_p2 = np.log(1 - pi1 + 1e-300) + stats.norm.logpdf(log_data, mu2, sigma2)

        # Log-sum-exp for numerical stability
        log_sum = np.logaddexp(log_p1, log_p2)
        gamma1 = np.exp(log_p1 - log_sum)

        # Current log-likelihood
        log_lik = np.sum(log_sum)

        if abs(log_lik - prev_log_lik) < tol:
            break
        prev_log_lik = log_lik

        # M-step
        n1 = gamma1.sum()
        n2 = n - n1

        if n1 < 10 or n2 < 10:
            # Degenerate: one component collapsed
            return None

        pi1 = n1 / n
        mu1 = np.sum(gamma1 * log_data) / n1
        sigma1 = max(np.sqrt(np.sum(gamma1 * (log_data - mu1)**2) / n1), 0.01)
        mu2 = np.sum((1 - gamma1) * log_data) / n2
        sigma2 = max(np.sqrt(np.sum((1 - gamma1) * (log_data - mu2)**2) / n2), 0.01)

    # Compute AIC/BIC (5 params: pi, mu1, sigma1, mu2, sigma2)
    k = 5
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    # Ensure component 1 is the "short" component
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1
        pi1 = 1 - pi1

    return {
        'pi1': pi1,
        'mu1': mu1, 'sigma1': sigma1,
        'mu2': mu2, 'sigma2': sigma2,
        'mean1_min': np.exp(mu1 + sigma1**2 / 2),
        'mean2_min': np.exp(mu2 + sigma2**2 / 2),
        'log_likelihood': log_lik,
        'aic': aic,
        'bic': bic,
        'n': n,
        'n_iterations': iteration + 1,
    }


def run_service_time_fitting(sessions, output_dir):
    """
    Component 1: Fit service time distributions per charger type.

    For each charger type (DC_Fast, Level_2, Mixed):
      - Fit Exponential, Gamma, Lognormal, Weibull
      - Compute AIC, BIC, KS statistic
      - Generate QQ plots
      - For Level_2: also try 2-component mixture

    Decision output: best distribution per type, and whether Mixed approx DC_Fast.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 1: SERVICE TIME DISTRIBUTION FITTING")
    print("=" * 70)

    # Filter data
    svc_df, dur_col = filter_service_time_data(sessions)

    # Identify charger type column
    ct_col = identify_columns(svc_df, 'charger_type')
    if ct_col is None:
        print("  [ERROR] charger_type column not found. Cannot fit by type.")
        return None, None

    charger_types = ['DC_Fast', 'Level_2', 'Mixed']
    distributions = {
        'exponential': expon,
        'gamma': gamma,
        'lognormal': lognorm,
        'weibull': weibull_min,
    }

    all_results = []
    summary = {}

    for ctype in charger_types:
        mask = svc_df[ct_col] == ctype
        data = svc_df.loc[mask, dur_col].dropna().values

        if len(data) < 50:
            print(f"\n  {ctype}: insufficient data (n={len(data)}). Skipping.")
            continue

        print(f"\n  {ctype} (n={len(data):,}):")
        print(f"    Mean={data.mean():.1f} min, Median={np.median(data):.1f} min, "
              f"CV={data.std()/data.mean():.3f}, Skew={stats.skew(data):.2f}")

        type_summary = {
            'n': int(len(data)),
            'mean_min': float(data.mean()),
            'median_min': float(np.median(data)),
            'std_min': float(data.std()),
            'cv': float(data.std() / data.mean()),
            'skewness': float(stats.skew(data)),
            'p25': float(np.percentile(data, 25)),
            'p75': float(np.percentile(data, 75)),
            'p95': float(np.percentile(data, 95)),
            'fits': {},
        }

        best_aic = np.inf
        best_dist = None

        for dist_name, dist_obj in distributions.items():
            result = fit_single_distribution(data, dist_name, dist_obj)
            if result is None:
                print(f"    {dist_name:12s}: FAILED")
                continue

            all_results.append({
                'charger_type': ctype,
                'distribution': dist_name,
                'n': result['n'],
                'k': result['k'],
                'log_likelihood': result['log_likelihood'],
                'aic': result['aic'],
                'bic': result['bic'],
                'ks_statistic': result['ks_statistic'],
                'ks_p_value': result['ks_p_value'],
            })

            # Store parameters in summary
            param_dict = {}
            if dist_name == 'exponential':
                param_dict = {'loc': float(result['params'][0]),
                              'scale': float(result['params'][1])}
            elif dist_name == 'gamma':
                param_dict = {'a': float(result['params'][0]),
                              'loc': float(result['params'][1]),
                              'scale': float(result['params'][2])}
            elif dist_name == 'lognormal':
                param_dict = {'s': float(result['params'][0]),
                              'loc': float(result['params'][1]),
                              'scale': float(result['params'][2])}
            elif dist_name == 'weibull':
                param_dict = {'c': float(result['params'][0]),
                              'loc': float(result['params'][1]),
                              'scale': float(result['params'][2])}

            type_summary['fits'][dist_name] = {
                'params': param_dict,
                'aic': float(result['aic']),
                'bic': float(result['bic']),
                'ks_statistic': float(result['ks_statistic']),
            }

            marker = ""
            if result['aic'] < best_aic:
                best_aic = result['aic']
                best_dist = dist_name

            print(f"    {dist_name:12s}: AIC={result['aic']:.1f}  BIC={result['bic']:.1f}  "
                  f"KS={result['ks_statistic']:.4f}")

        # Mark best
        if best_dist:
            type_summary['best_distribution'] = best_dist
            type_summary['best_aic'] = float(best_aic)
            print(f"    --> Best by AIC: {best_dist}")

        # L2 mixture model
        if ctype == 'Level_2':
            print(f"\n    Fitting 2-component lognormal mixture for {ctype}...")
            mix_result = fit_mixture_2component(data)
            if mix_result is not None:
                all_results.append({
                    'charger_type': ctype,
                    'distribution': 'mixture_2lognorm',
                    'n': mix_result['n'],
                    'k': 5,
                    'log_likelihood': mix_result['log_likelihood'],
                    'aic': mix_result['aic'],
                    'bic': mix_result['bic'],
                    'ks_statistic': np.nan,  # no simple KS for mixture
                    'ks_p_value': np.nan,
                })
                type_summary['fits']['mixture_2lognorm'] = {
                    'params': {
                        'pi1': float(mix_result['pi1']),
                        'mu1': float(mix_result['mu1']),
                        'sigma1': float(mix_result['sigma1']),
                        'mu2': float(mix_result['mu2']),
                        'sigma2': float(mix_result['sigma2']),
                    },
                    'aic': float(mix_result['aic']),
                    'bic': float(mix_result['bic']),
                    'mean1_min': float(mix_result['mean1_min']),
                    'mean2_min': float(mix_result['mean2_min']),
                }
                print(f"    mixture_2lognorm: AIC={mix_result['aic']:.1f}  "
                      f"BIC={mix_result['bic']:.1f}")
                print(f"      Component 1: pi={mix_result['pi1']:.2f}, "
                      f"mean={mix_result['mean1_min']:.1f} min")
                print(f"      Component 2: pi={1-mix_result['pi1']:.2f}, "
                      f"mean={mix_result['mean2_min']:.1f} min")

                # Check if mixture beats single best
                if mix_result['aic'] < best_aic:
                    type_summary['best_distribution'] = 'mixture_2lognorm'
                    type_summary['best_aic'] = float(mix_result['aic'])
                    print(f"    --> Mixture beats single-distribution best ({best_dist})")
                else:
                    print(f"    --> Single {best_dist} still preferred over mixture")
            else:
                print(f"    mixture_2lognorm: FAILED (degenerate or insufficient data)")

        best_name = type_summary.get('best_distribution')
        best_params = type_summary.get('fits', {}).get(
            best_name, {}).get('params', {})
        if best_name and best_params:
            sampler_seed = {
                'DC_Fast': 4101, 'Level_2': 4102, 'Mixed': 4103,
            }.get(ctype, 4199)
            type_summary['simulation_sampler'] = \
                estimate_clamped_sampler_moments(
                    best_name, best_params, sampler_seed)
            print("    DES clamped-sampler mean: "
                  f"{type_summary['simulation_sampler']['estimated_mean_min']:.1f} min")

        summary[ctype] = type_summary

    # Mixed vs DC_Fast mergeability check
    if 'DC_Fast' in summary and 'Mixed' in summary:
        print(f"\n  Mergeability check: Mixed vs DC_Fast")
        dc_cv = summary['DC_Fast']['cv']
        mx_cv = summary['Mixed']['cv']
        dc_best = summary['DC_Fast'].get('best_distribution', 'unknown')
        mx_best = summary['Mixed'].get('best_distribution', 'unknown')
        cv_diff = abs(dc_cv - mx_cv)
        same_best = dc_best == mx_best

        # KS two-sample test between DC_Fast and Mixed durations
        dc_data = svc_df.loc[svc_df[ct_col] == 'DC_Fast', dur_col].dropna().values
        mx_data = svc_df.loc[svc_df[ct_col] == 'Mixed', dur_col].dropna().values
        ks_2samp_stat, ks_2samp_p = stats.ks_2samp(dc_data, mx_data)

        mergeable = same_best and cv_diff < 0.15 and ks_2samp_p > 0.01
        summary['merge_decision'] = {
            'dc_cv': float(dc_cv),
            'mixed_cv': float(mx_cv),
            'cv_difference': float(cv_diff),
            'same_best_dist': same_best,
            'ks_2sample_stat': float(ks_2samp_stat),
            'ks_2sample_p': float(ks_2samp_p),
            'recommendation': 'merge' if mergeable else 'keep_separate',
        }
        print(f"    CV difference: {cv_diff:.3f}")
        print(f"    Same best distribution: {same_best} ({dc_best} vs {mx_best})")
        print(f"    KS 2-sample: D={ks_2samp_stat:.4f}, p={ks_2samp_p:.4e}")
        print(f"    --> Recommendation: {'MERGE' if mergeable else 'KEEP SEPARATE'}")

    # Save results
    results_df = pd.DataFrame(all_results)
    save_csv(results_df, output_dir, 'service_time_fits.csv')

    summary_path = Path(output_dir) / 'service_time_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(to_builtin(summary), f, indent=2)
    print(f"  [SAVED] {summary_path}")

    return results_df, summary


def plot_service_time_qq(sessions, summary, output_dir):
    """
    QQ plots: 3 charger types x 4 distributions (3x4 grid).
    """
    print("\n  Generating QQ plots...")
    svc_df, dur_col = filter_service_time_data(sessions)
    ct_col = identify_columns(svc_df, 'charger_type')

    charger_types = ['DC_Fast', 'Level_2', 'Mixed']
    dist_map = {
        'exponential': expon,
        'gamma': gamma,
        'lognormal': lognorm,
        'weibull': weibull_min,
    }

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for row, ctype in enumerate(charger_types):
        data = svc_df.loc[svc_df[ct_col] == ctype, dur_col].dropna().values
        if len(data) < 50:
            continue

        for col, (dist_name, dist_obj) in enumerate(dist_map.items()):
            ax = axes[row, col]
            try:
                params = dist_obj.fit(data, floc=0)

                # QQ plot: theoretical quantiles vs sample quantiles
                n = len(data)
                theoretical_q = dist_obj.ppf(
                    np.linspace(0.01, 0.99, min(500, n)), *params)
                sample_q = np.quantile(data, np.linspace(0.01, 0.99, min(500, n)))

                ax.scatter(theoretical_q, sample_q, alpha=0.3, s=5, c='steelblue')
                # 45-degree line
                lims = [0, max(theoretical_q.max(), sample_q.max()) * 1.05]
                ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7)

                is_best = (summary.get(ctype, {}).get('best_distribution') == dist_name)
                title_suffix = " [BEST]" if is_best else ""
                ax.set_title(f"{ctype} / {dist_name}{title_suffix}", fontsize=9)

            except Exception:
                ax.text(0.5, 0.5, 'Fit failed', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f"{ctype} / {dist_name}", fontsize=9)

            if col == 0:
                ax.set_ylabel('Sample Quantiles (min)')
            if row == 2:
                ax.set_xlabel('Theoretical Quantiles (min)')

    fig.suptitle("Service Time QQ Plots: 3 Charger Types x 4 Distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = Path(output_dir) / 'figures' / 'qq_service_time_grid.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  [SAVED] {fig_path}")


def plot_service_time_fitted_pdf(sessions, summary, output_dir):
    """
    Overlay fitted PDFs on empirical histograms, one subplot per charger type.
    """
    print("  Generating fitted PDF overlay plots...")
    svc_df, dur_col = filter_service_time_data(sessions)
    ct_col = identify_columns(svc_df, 'charger_type')

    charger_types = ['DC_Fast', 'Level_2', 'Mixed']
    dist_map = {
        'exponential': expon,
        'gamma': gamma,
        'lognormal': lognorm,
        'weibull': weibull_min,
    }
    colors = {'exponential': '#e74c3c', 'gamma': '#2ecc71',
              'lognormal': '#3498db', 'weibull': '#9b59b6'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, ctype in enumerate(charger_types):
        ax = axes[i]
        data = svc_df.loc[svc_df[ct_col] == ctype, dur_col].dropna().values
        if len(data) < 50:
            continue

        # Histogram
        ax.hist(data, bins=100, density=True, alpha=0.4, color='gray',
                edgecolor='white', linewidth=0.3, label='Empirical')

        x = np.linspace(0.1, np.percentile(data, 99), 500)

        best_dist = summary.get(ctype, {}).get('best_distribution', '')

        for dist_name, dist_obj in dist_map.items():
            try:
                params = dist_obj.fit(data, floc=0)
                pdf_vals = dist_obj.pdf(x, *params)
                lw = 2.5 if dist_name == best_dist else 1.2
                ls = '-' if dist_name == best_dist else '--'
                label = f"{dist_name}" + (" [best]" if dist_name == best_dist else "")
                ax.plot(x, pdf_vals, color=colors[dist_name], linewidth=lw,
                        linestyle=ls, label=label)
            except Exception:
                pass

        cv = summary.get(ctype, {}).get('cv', 0)
        ax.set_title(f"{ctype} (n={len(data):,}, CV={cv:.2f})", fontsize=11)
        ax.set_xlabel('Duration (min)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_xlim(0, np.percentile(data, 99))

    fig.suptitle("Service Time: Empirical vs Fitted Distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = Path(output_dir) / 'figures' / 'service_time_fitted_pdf.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  [SAVED] {fig_path}")


# =============================================================================
# COMPONENT 2: ACN CROSS-VALIDATION
# =============================================================================

def run_acn_crossval(acn, output_dir):
    """
    Component 2: Cross-validate key findings against ACN JPL data.

    Tests:
      - IAT dispersion index (unconditional + conditional on hour)
      - K-S test for exponential IATs
      - Service time distribution fitting (ACN is L2 only)
      - Comparison summary table
    """
    if acn is None:
        print("\n" + "=" * 70)
        print("COMPONENT 2: ACN CROSS-VALIDATION - SKIPPED (no data)")
        print("=" * 70)
        results_df = pd.DataFrame([{
            'dataset': 'ACN_JPL',
            'test': 'status',
            'statistic': np.nan,
            'p_value': np.nan,
            'n': 0,
            'status': 'skipped_no_data',
            'note': 'acn_clean dataset not found; cross-validation skipped',
        }])
        save_csv(results_df, output_dir, 'acn_crossval.csv')
        return results_df

    print("\n" + "=" * 70)
    print("COMPONENT 2: ACN CROSS-VALIDATION")
    print("=" * 70)

    results = []

    # Identify columns
    time_col = identify_columns(acn, 'start_time')
    dur_col = identify_columns(acn, 'duration')
    energy_col = identify_columns(acn, 'energy')

    if time_col is None:
        print("  [ERROR] Cannot find start time column in ACN data.")
        results_df = pd.DataFrame([{
            'dataset': 'ACN_JPL',
            'test': 'status',
            'statistic': np.nan,
            'p_value': np.nan,
            'n': int(len(acn)),
            'status': 'error_missing_time_column',
            'note': 'Cannot find start-time column in ACN dataset',
        }])
        save_csv(results_df, output_dir, 'acn_crossval.csv')
        return results_df

    acn[time_col] = pd.to_datetime(acn[time_col], errors='coerce')
    acn = acn.dropna(subset=[time_col])
    acn = acn.sort_values(time_col)

    print(f"  ACN sessions: {len(acn):,}")
    print(f"  Date range: {acn[time_col].min()} to {acn[time_col].max()}")

    # --- Inter-arrival time analysis ---
    print(f"\n  Computing IATs...")
    times = acn[time_col].sort_values()
    iats = times.diff().dt.total_seconds().dropna()
    iats = iats[iats > 0].values

    if len(iats) > 100:
        # Unconditional dispersion index
        # Use hourly arrival counts
        acn['_hour'] = acn[time_col].dt.hour
        acn['_date'] = acn[time_col].dt.date
        hourly_counts = acn.groupby(['_date', '_hour']).size().reset_index(name='count')

        global_mean = hourly_counts['count'].mean()
        global_var = hourly_counts['count'].var()
        global_di = global_var / global_mean if global_mean > 0 else np.nan

        print(f"  Unconditional DI (hourly counts): {global_di:.1f}")

        # K-S test on IATs
        lambda_hat = 1.0 / np.mean(iats)
        ks_stat, ks_p = kstest(iats, 'expon', args=(0, 1/lambda_hat))
        print(f"  K-S test (exponential IAT): D={ks_stat:.4f}, p={ks_p:.4e}")

        results.append({
            'dataset': 'ACN_JPL',
            'test': 'unconditional_DI',
            'statistic': global_di,
            'p_value': np.nan,
            'n': len(hourly_counts),
        })
        results.append({
            'dataset': 'ACN_JPL',
            'test': 'KS_exponential_IAT',
            'statistic': ks_stat,
            'p_value': ks_p,
            'n': len(iats),
        })

        # Conditional DI (per hour)
        print(f"\n  Conditional DI by hour of day:")
        conditional_dis = []
        for h in range(24):
            h_counts = hourly_counts.loc[hourly_counts['_hour'] == h, 'count']
            if len(h_counts) > 10:
                h_mean = h_counts.mean()
                h_var = h_counts.var()
                h_di = h_var / h_mean if h_mean > 0 else np.nan
                conditional_dis.append({'hour': h, 'di': h_di, 'mean': h_mean})

        if conditional_dis:
            cond_df = pd.DataFrame(conditional_dis)
            mean_cond_di = cond_df['di'].mean()
            min_cond_di = cond_df['di'].min()
            max_cond_di = cond_df['di'].max()
            print(f"    Conditional DI range: {min_cond_di:.2f} - {max_cond_di:.2f}")
            print(f"    Mean conditional DI: {mean_cond_di:.2f}")

            results.append({
                'dataset': 'ACN_JPL',
                'test': 'conditional_DI_mean',
                'statistic': mean_cond_di,
                'p_value': np.nan,
                'n': len(cond_df),
            })

    # --- Service time fitting (ACN is workplace L2) ---
    if dur_col:
        print(f"\n  ACN service time fitting (workplace L2)...")
        acn_dur = pd.to_numeric(acn[dur_col], errors='coerce').dropna()

        # Filter: exclude zero/negative and >720 min
        acn_dur = acn_dur[(acn_dur > 0) & (acn_dur <= 720)]

        if len(acn_dur) > 100:
            acn_data = acn_dur.values
            acn_cv = acn_data.std() / acn_data.mean()
            print(f"    n={len(acn_data):,}, mean={acn_data.mean():.1f} min, "
                  f"CV={acn_cv:.3f}, skew={stats.skew(acn_data):.2f}")

            results.append({
                'dataset': 'ACN_JPL',
                'test': 'service_time_CV',
                'statistic': acn_cv,
                'p_value': np.nan,
                'n': len(acn_data),
            })

            # Fit same 4 distributions
            dist_map = {
                'exponential': expon,
                'gamma': gamma,
                'lognormal': lognorm,
                'weibull': weibull_min,
            }
            for dist_name, dist_obj in dist_map.items():
                fit = fit_single_distribution(acn_data, dist_name, dist_obj)
                if fit:
                    results.append({
                        'dataset': 'ACN_JPL',
                        'test': f'service_fit_{dist_name}',
                        'statistic': fit['aic'],
                        'p_value': fit['ks_p_value'],
                        'n': fit['n'],
                    })
                    print(f"    {dist_name:12s}: AIC={fit['aic']:.1f}, "
                          f"KS={fit['ks_statistic']:.4f}")
        else:
            print(f"    Insufficient ACN duration data (n={len(acn_dur)})")
    else:
        print(f"  [WARN] No duration column in ACN data. Service time fitting skipped.")

    if len(results) == 0:
        results.append({
            'dataset': 'ACN_JPL',
            'test': 'status',
            'statistic': np.nan,
            'p_value': np.nan,
            'n': int(len(acn)),
            'status': 'completed_no_tests',
            'note': 'ACN dataset loaded, but no valid tests were produced',
        })
    # Save
    results_df = pd.DataFrame(results)
    save_csv(results_df, output_dir, 'acn_crossval.csv')

    return results_df


# =============================================================================
# COMPONENT 3: ERLANG-C AND M/G/s FLEET SIZING
# =============================================================================

def erlang_c(s, rho_total):
    """
    Erlang-C formula: probability that an arriving customer must wait.

    Parameters:
        s: number of servers
        rho_total: total offered load (lambda / mu). Must be < s for stability.

    Returns:
        P(wait) = C(s, rho_total)
    """
    if rho_total >= s:
        return 1.0  # System is unstable

    a = rho_total  # offered load

    # Compute P0 (probability system is empty)
    # P0 = [sum_{k=0}^{s-1} a^k/k! + a^s/s! * s/(s-a)]^{-1}

    # Use log-space to avoid overflow for large s
    log_terms = []
    for k in range(s):
        log_term = k * np.log(a) - np.sum(np.log(np.arange(1, k + 1))) if k > 0 else 0.0
        log_terms.append(log_term)

    log_last = (s * np.log(a)
                - np.sum(np.log(np.arange(1, s + 1)))
                + np.log(s) - np.log(s - a))
    log_terms.append(log_last)

    # Log-sum-exp
    max_log = max(log_terms)
    log_sum = max_log + np.log(sum(np.exp(t - max_log) for t in log_terms))

    log_P0 = -log_sum

    # P(wait) = [a^s / s! * s/(s-a)] * P0
    log_Pw = (s * np.log(a)
              - np.sum(np.log(np.arange(1, s + 1)))
              + np.log(s) - np.log(s - a)
              + log_P0)

    return min(np.exp(log_Pw), 1.0)


def wait_tail_approximations(s, lambda_per_min, mu_per_min,
                             cv_squared, tau_min=15.0):
    """Return M/M/s and moment-matched M/G/s waiting-tail estimates.

    Allen-Cunneen is applied only to mean waiting time. The M/G/s tail is
    then an explicitly labelled exponential moment-match approximation; the
    DES remains the binding service-level evaluation.
    """
    offered_load = lambda_per_min / mu_per_min
    if offered_load >= s:
        return 1.0, 1.0, float('inf'), float('inf'), 1.0

    p_wait_zero = erlang_c(s, offered_load)
    clearance_rate = s * mu_per_min - lambda_per_min
    mean_wait_mms = p_wait_zero / clearance_rate
    mean_wait_mgs = mean_wait_mms * (1.0 + cv_squared) / 2.0

    p_tail_mms = p_wait_zero * np.exp(-clearance_rate * tau_min)
    if p_wait_zero <= 0 or mean_wait_mgs <= 0:
        p_tail_mgs = 0.0
    else:
        conditional_mean_mgs = mean_wait_mgs / p_wait_zero
        p_tail_mgs = p_wait_zero * np.exp(-tau_min / conditional_mean_mgs)
    return (float(min(p_tail_mms, 1.0)),
            float(min(p_tail_mgs, 1.0)),
            float(mean_wait_mms), float(mean_wait_mgs),
            float(p_wait_zero))


def select_representative_stations(sessions, nhpp_rates):
    """
    Select 4 representative stations:
      1. Expressway (high fault rate, short sessions)
      2. High-volume urban (most arrivals)
      3. Institutional (long sessions)
      4. Mixed archetype

    Uses station-level statistics to identify candidates.
    """
    station_col = identify_columns(sessions, 'station')
    dur_col = identify_columns(sessions, 'duration')
    abn_col = identify_columns(sessions, 'is_abnormal')

    if station_col is None:
        return []

    stats_list = []
    for station, group in sessions.groupby(station_col):
        s = {'station': station, 'n_sessions': len(group)}

        if dur_col:
            durations = pd.to_numeric(group[dur_col], errors='coerce').dropna()
            s['mean_duration'] = durations.mean() if len(durations) > 0 else np.nan
        if abn_col:
            s['fault_rate'] = group[abn_col].mean()

        stats_list.append(s)

    sdf = pd.DataFrame(stats_list)
    if len(sdf) < 4:
        return sdf['station'].tolist()[:4]

    reps = []

    # 1. Expressway: highest fault rate
    if 'fault_rate' in sdf.columns:
        exp_idx = sdf['fault_rate'].idxmax()
        reps.append(sdf.loc[exp_idx, 'station'])
        sdf_remaining = sdf.drop(exp_idx)
    else:
        sdf_remaining = sdf

    # 2. High-volume urban: most sessions
    vol_idx = sdf_remaining['n_sessions'].idxmax()
    reps.append(sdf_remaining.loc[vol_idx, 'station'])
    sdf_remaining = sdf_remaining.drop(vol_idx)

    # 3. Institutional: longest mean duration
    if 'mean_duration' in sdf_remaining.columns:
        dur_idx = sdf_remaining['mean_duration'].idxmax()
        reps.append(sdf_remaining.loc[dur_idx, 'station'])
        sdf_remaining = sdf_remaining.drop(dur_idx)

    # 4. Mixed: median by session count
    if len(sdf_remaining) > 0:
        median_idx = (sdf_remaining['n_sessions'] - sdf_remaining['n_sessions'].median()).abs().idxmin()
        reps.append(sdf_remaining.loc[median_idx, 'station'])

    print(f"  Representative stations: {reps}")
    return reps


def run_erlang_c_analysis(sessions, nhpp_rates, svc_summary, output_dir):
    """
    Component 3: Erlang-C and M/G/s fleet sizing comparison.

    For each representative station:
      - Compute peak-hour arrival rate (from NHPP rates or empirical)
      - Compute service rate mu from fitted service time distribution
      - Sweep s = 1..50 chargers
      - Compare P(wait) under M/M/s (Erlang-C) vs M/G/s (corrected)
      - Find s* where P(wait) < 5% under each model
    """
    print("\n" + "=" * 70)
    print("COMPONENT 3: ERLANG-C AND M/G/s FLEET SIZING")
    print("=" * 70)

    station_col = identify_columns(sessions, 'station')
    dur_col = identify_columns(sessions, 'duration')
    ct_col = identify_columns(sessions, 'charger_type')
    abn_col = identify_columns(sessions, 'is_abnormal')

    if station_col is None or dur_col is None:
        print("  [ERROR] Required columns not found.")
        return None

    # Select representative stations
    rep_stations = select_representative_stations(sessions, nhpp_rates)
    if not rep_stations:
        print("  [ERROR] Could not identify representative stations.")
        return None

    # Get per-type service parameters from summary
    charger_cv = {}
    charger_mean = {}  # mean service time in minutes
    for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
        if ctype in svc_summary:
            mean_min = float(svc_summary[ctype]['mean_min'])
            cv = float(svc_summary[ctype]['cv'])
            charger_mean[ctype] = mean_min
            charger_cv[ctype] = cv

    all_results = []

    # Global study end date — anchor for recent-quarter across all stations
    time_col = identify_columns(sessions, 'start_time')
    if time_col:
        global_max_date = pd.to_datetime(
            sessions[time_col], errors='coerce').dt.date.max()
    else:
        global_max_date = None

    for station in rep_stations:
        s_mask = sessions[station_col] == station

        # =================================================================
        # SCENARIO-BASED PEAK LAMBDA
        # Three scenarios to address 2.7x demand growth:
        #   historical_avg  — mean across all dates (descriptive only)
        #   recent_quarter  — last 90 days before global study end
        #   stress_case     — recent_quarter × 1.2 (planning margin)
        # =================================================================
        peak_lambdas = {}  # scenario_name -> peak hourly arrival rate

        # --- From NHPP rates file (historical_avg) ---
        if nhpp_rates is not None:
            station_rates = nhpp_rates[nhpp_rates.iloc[:, 0] == station]
            if len(station_rates) > 0:
                rate_cols = [c for c in station_rates.columns if 'rate' in c.lower()
                             or 'lambda' in c.lower() or 'mean' in c.lower()]
                if rate_cols:
                    peak_lambdas['historical_avg'] = float(
                        station_rates[rate_cols[0]].max())

        # --- From raw data with zero-fill grid ---
        if time_col:
            station_sessions = sessions[s_mask].copy()
            station_sessions['_hour'] = pd.to_datetime(
                station_sessions[time_col], errors='coerce').dt.hour
            station_sessions['_date'] = pd.to_datetime(
                station_sessions[time_col], errors='coerce').dt.date

            hourly_cts = station_sessions.groupby(
                ['_date', '_hour']).size().reset_index(name='count')

            # Build complete date × hour grid for this station.
            # Without this, groupby('_hour').mean() conditions on ≥1 arrival
            # and biases lambda upward.
            station_date_min = hourly_cts['_date'].min()
            station_date_max = hourly_cts['_date'].max()
            station_dates = pd.date_range(
                station_date_min, station_date_max, freq='D').date
            full_station_grid = pd.DataFrame(
                [(d, h) for d in station_dates for h in range(24)],
                columns=['_date', '_hour']
            )
            full_station_grid = full_station_grid.merge(
                hourly_cts, on=['_date', '_hour'], how='left'
            )
            full_station_grid['count'] = full_station_grid['count'].fillna(0)

            # Historical average from raw data (fallback if NHPP file missing)
            if 'historical_avg' not in peak_lambdas:
                hour_means_all = full_station_grid.groupby('_hour')['count'].mean()
                peak_lambdas['historical_avg'] = float(hour_means_all.max())

            # Recent quarter: last 90 days anchored to GLOBAL study end
            anchor_date = global_max_date if global_max_date else station_date_max
            # Both dates are Python ``date`` objects.  Keep the subtraction in
            # the standard library to avoid NumPy's deprecated generic timedelta.
            recent_cutoff = anchor_date - timedelta(days=90)
            recent = full_station_grid[
                pd.to_datetime(full_station_grid['_date']) >= pd.Timestamp(recent_cutoff)
            ]

            if len(recent) >= 24 * 7:  # at least 1 week of complete hourly data
                recent_hour_means = recent.groupby('_hour')['count'].mean()
                recent_peak = float(recent_hour_means.max())
                # Anomaly guard: cap at 3× historical
                hist_peak = peak_lambdas['historical_avg']
                if hist_peak > 0 and recent_peak > 3 * hist_peak:
                    recent_peak = hist_peak * 2.0
                    print(f"    [WARN] {station}: recent quarter peak capped "
                          f"at 2× historical (anomaly guard)")
                peak_lambdas['recent_quarter'] = recent_peak
            else:
                peak_lambdas['recent_quarter'] = peak_lambdas['historical_avg']
                print(f"    [NOTE] {station}: insufficient recent data, "
                      f"using historical average as recent_quarter fallback")

            # Stress case
            peak_lambdas['stress_case'] = peak_lambdas['recent_quarter'] * 1.2
        else:
            print(f"  [WARN] Cannot compute arrival rate for {station}. Skipping.")
            continue

        if not peak_lambdas:
            continue

        # --- Charger type and service parameters (unchanged) ---
        station_mix = {}
        if ct_col:
            station_types = sessions.loc[s_mask, ct_col].dropna().value_counts()
            dominant_type = station_types.index[0] if len(station_types) > 0 else 'DC_Fast'
            if len(station_types) > 0:
                station_mix = (station_types / station_types.sum()).to_dict()
        else:
            dominant_type = 'DC_Fast'

        valid_components = []
        for ctype, p_share in station_mix.items():
            if ctype in charger_mean and ctype in charger_cv:
                mean_k = float(charger_mean[ctype])
                cv_k = float(charger_cv[ctype])
                if np.isfinite(mean_k) and mean_k > 0 and np.isfinite(cv_k):
                    valid_components.append((str(ctype), float(p_share), mean_k, cv_k))

        if len(valid_components) > 0:
            share_sum = sum(comp[1] for comp in valid_components)
            normalized_mix = {
                ctype: share / share_sum
                for ctype, share, _, _ in valid_components
            } if share_sum > 0 else {}
            mix_coverage = float(share_sum)

            mean_s = 0.0
            second_moment_s = 0.0
            for ctype, _, mean_k, cv_k in valid_components:
                w = normalized_mix.get(ctype, 0.0)
                var_k = (cv_k * mean_k) ** 2
                mean_s += w * mean_k
                second_moment_s += w * (var_k + mean_k ** 2)

            var_s = max(second_moment_s - mean_s ** 2, 0.0)
            mu = 1.0 / mean_s if mean_s > 0 else 1.0 / 35.0
            cv_sq = var_s / (mean_s ** 2) if mean_s > 0 else 1.0
            cv = float(np.sqrt(cv_sq))
            cv_source = 'station_type_mixture'
            service_param_scope = 'station_mixture_from_global_type_fits'
            station_mix_json = json.dumps(normalized_mix, sort_keys=True)
        else:
            mean_min = charger_mean.get(dominant_type, 35.0)
            cv = float(charger_cv.get(dominant_type, 1.0))
            cv_sq = cv ** 2
            mu = 1.0 / mean_min if mean_min > 0 else 1.0 / 35.0
            cv_source = 'dominant_type_fallback'
            service_param_scope = 'dominant_type_from_global_type_fit'
            station_mix_json = json.dumps({str(dominant_type): 1.0}, sort_keys=True)
            mix_coverage = 0.0

        # --- Erlang-C sweep for EACH scenario ---
        for scenario_name, peak_lambda in peak_lambdas.items():
            lambda_per_min = peak_lambda / 60.0
            rho_total = lambda_per_min / mu

            print(f"\n  Station: {station} | Scenario: {scenario_name}")
            print(f"    Peak arrivals/hour: {peak_lambda:.2f}")
            print(f"    Dominant type: {dominant_type} (mu={1/mu:.1f} min, CV={cv:.2f})")
            print(f"    Offered load (rho): {rho_total:.2f}")

            s_mms_star = None
            s_mgs_star = None

            for s in range(1, 51):
                (pw_mms, pw_mgs, mean_wait_mms, mean_wait_mgs,
                 p_wait_zero_mms) = wait_tail_approximations(
                    s, lambda_per_min, mu, cv_sq, tau_min=15.0)

                all_results.append({
                    'station': station,
                    'sizing_scenario': scenario_name,
                    'dominant_type': dominant_type,
                    'peak_lambda_per_hour': peak_lambda,
                    'mu_per_min': mu,
                    'cv': cv,
                    'cv_source': cv_source,
                    'service_param_scope': service_param_scope,
                    'station_mix_json': station_mix_json,
                    'mix_coverage': mix_coverage,
                    'rho_total': rho_total,
                    's': s,
                    'p_wait_zero_mms': p_wait_zero_mms,
                    'p_wait_mms': pw_mms,
                    'p_wait_mgs': pw_mgs,
                    'mean_wait_mms_min': mean_wait_mms,
                    'mean_wait_mgs_min': mean_wait_mgs,
                    'probability_metric': 'p_wait_gt_15min',
                })

                if s_mms_star is None and pw_mms < 0.05:
                    s_mms_star = s
                if s_mgs_star is None and pw_mgs < 0.05:
                    s_mgs_star = s

            print(f"    s* (M/M/s, P(W>15 min)<5%): {s_mms_star if s_mms_star else '>50'}")
            print(f"    s* (M/G/s approx, P(W>15 min)<5%): {s_mgs_star if s_mgs_star else '>50'}")
            if s_mms_star and s_mgs_star:
                gap = s_mgs_star - s_mms_star
                direction = "MORE" if gap > 0 else ("FEWER" if gap < 0 else "SAME")
                print(f"    Gap: M/G/s needs {abs(gap)} {direction} chargers than M/M/s")

    results_df = pd.DataFrame(all_results)
    save_csv(results_df, output_dir, 'erlang_c_results_4rep.csv')

    # Build mgs_comparison: one row per (station, scenario)
    comparison_rows = []
    for (station, scenario), sdf in results_df.groupby(['station', 'sizing_scenario']):
        mms_star = sdf.loc[sdf['p_wait_mms'] < 0.05, 's'].min()
        mgs_star = sdf.loc[sdf['p_wait_mgs'] < 0.05, 's'].min()
        mms_star_10 = sdf.loc[sdf['p_wait_mms'] < 0.10, 's'].min()
        mgs_star_10 = sdf.loc[sdf['p_wait_mgs'] < 0.10, 's'].min()
        mms_star_20 = sdf.loc[sdf['p_wait_mms'] < 0.20, 's'].min()
        mgs_star_20 = sdf.loc[sdf['p_wait_mgs'] < 0.20, 's'].min()

        rho = sdf['rho_total'].iloc[0]
        cv_val = sdf['cv'].iloc[0]
        dom_type = sdf['dominant_type'].iloc[0]
        peak_lam = sdf['peak_lambda_per_hour'].iloc[0]
        cv_src = sdf['cv_source'].iloc[0] if 'cv_source' in sdf.columns else 'unknown'
        svc_scope = (sdf['service_param_scope'].iloc[0]
                     if 'service_param_scope' in sdf.columns else 'unknown')
        mix_json = (sdf['station_mix_json'].iloc[0]
                    if 'station_mix_json' in sdf.columns else '{}')
        mix_cov = (float(sdf['mix_coverage'].iloc[0])
                   if 'mix_coverage' in sdf.columns else np.nan)

        s_gap_5pct = (int(mgs_star) - int(mms_star)
                      if pd.notna(mms_star) and pd.notna(mgs_star) else None)

        comparison_rows.append({
            'station': station,
            'sizing_scenario': scenario,
            'dominant_type': dom_type,
            'peak_lambda_per_hour': peak_lam,
            'rho_total': rho,
            'cv': cv_val,
            'cv_source': cv_src,
            'service_param_scope': svc_scope,
            'station_mix_json': mix_json,
            'mix_coverage': mix_cov,
            'cv_squared': cv_val ** 2,
            'mgs_correction_factor': (1 + cv_val ** 2) / 2,
            'probability_metric': 'p_wait_gt_15min',
            'mgs_tail_method': 'Allen-Cunneen mean + exponential moment match',
            's_star_mms_5pct': int(mms_star) if pd.notna(mms_star) else None,
            's_star_mgs_5pct': int(mgs_star) if pd.notna(mgs_star) else None,
            's_gap_5pct': s_gap_5pct,
            's_star_mms_10pct': int(mms_star_10) if pd.notna(mms_star_10) else None,
            's_star_mgs_10pct': int(mgs_star_10) if pd.notna(mgs_star_10) else None,
            's_star_mms_20pct': int(mms_star_20) if pd.notna(mms_star_20) else None,
            's_star_mgs_20pct': int(mgs_star_20) if pd.notna(mgs_star_20) else None,
        })

    mgs_comparison_df = pd.DataFrame(comparison_rows)
    save_csv(mgs_comparison_df, output_dir, 'mgs_comparison_4rep.csv')

    return results_df


def plot_erlang_c_comparison(results_df, output_dir):
    """Plot P(wait) vs s for M/M/s and M/G/s at representative stations."""
    if results_df is None or len(results_df) == 0:
        return

    print("\n  Generating Erlang-C comparison plots...")
    stations = results_df['station'].unique()
    n_stations = len(stations)

    fig, axes = plt.subplots(1, n_stations, figsize=(5 * n_stations, 5))
    if n_stations == 1:
        axes = [axes]

    for i, station in enumerate(stations):
        ax = axes[i]
        sdf = results_df[results_df['station'] == station]

        ax.plot(sdf['s'], sdf['p_wait_mms'], 'b-o', markersize=3,
                label='M/M/s (Erlang-C)', linewidth=1.5)
        ax.plot(sdf['s'], sdf['p_wait_mgs'], 'r-s', markersize=3,
                label='M/G/s (corrected)', linewidth=1.5)
        ax.axhline(0.05, color='gray', linestyle='--', alpha=0.7,
                    label='P(wait) = 5%')

        cv = sdf['cv'].iloc[0]
        rho = sdf['rho_total'].iloc[0]
        dom = sdf['dominant_type'].iloc[0]

        # Truncate long station names
        short_name = station[:30] + '...' if len(str(station)) > 30 else station
        ax.set_title(f"{short_name}\n{dom}, CV={cv:.2f}, rho={rho:.1f}", fontsize=9)
        ax.set_xlabel('Number of Chargers (s)')
        ax.set_ylabel('P(wait)')
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Set x limits to a reasonable range
        rho_ceil = int(np.ceil(rho))
        ax.set_xlim(max(1, rho_ceil - 2), min(50, rho_ceil + 15))

    fig.suptitle("Fleet Sizing: M/M/s vs M/G/s P(wait) Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = Path(output_dir) / 'figures' / 'erlang_c_comparison.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  [SAVED] {fig_path}")


# =============================================================================
# COMPONENT 4: XGBOOST DEMAND FORECASTING
# =============================================================================

def build_feature_matrix(hourly, sessions):
    """
    Build feature matrix for hourly arrival prediction.

    Features:
      - hour, dow, month, is_weekend
      - TOU_tier (one-hot)
      - temperature, precipitation
      - lag_1, lag_24, lag_168 (lagged arrivals)
      - rolling_7d_mean
      - station indicator (one-hot or label-encoded)

    Target: hourly arrival count
    """
    print("  Building feature matrix...")

    if hourly is None:
        print("  [ERROR] Hourly data not available.")
        return None, None, None

    df = hourly.copy()

    # Identify key columns
    station_col = identify_columns(df, 'station')
    hour_col = identify_columns(df, 'hour')
    arrivals_col = identify_columns(df, 'arrivals')
    date_col = identify_columns(df, 'date')

    if arrivals_col is None:
        print("  [ERROR] Cannot find arrivals column in hourly data.")
        return None, None, None

    # Normalize column names
    rename_map = {}
    if station_col and station_col != 'station':
        rename_map[station_col] = 'station'
    if hour_col and hour_col != 'hour':
        rename_map[hour_col] = 'hour'
    if arrivals_col and arrivals_col != 'arrivals':
        rename_map[arrivals_col] = 'arrivals'
    if date_col and date_col != 'date':
        rename_map[date_col] = 'date'
    if rename_map:
        df = df.rename(columns=rename_map)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['arrivals'] = pd.to_numeric(df['arrivals'], errors='coerce').fillna(0)
    df = df.dropna(subset=['date', 'hour'])

    # Temporal features
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)

    # Cyclical encoding for hour and dow
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)

    # TOU tier from hour (deterministic mapping from Jiaxing schedule)
    tou_map = {}
    # Valley: 23:00-07:00
    for h in [23, 0, 1, 2, 3, 4, 5, 6]:
        tou_map[h] = 0  # Valley
    # Flat: 07:00-08:00, 11:00-13:00, 17:00-19:00, 21:00-23:00
    for h in [7, 11, 12, 17, 18, 21, 22]:
        tou_map[h] = 1  # Flat
    # Peak: 08:00-11:00, 13:00-17:00, 19:00-21:00
    for h in [8, 9, 10, 13, 14, 15, 16, 19, 20]:
        tou_map[h] = 2  # Peak
    df['tou_tier_num'] = df['hour'].map(tou_map).fillna(1)

    # Build one weather row per date. It is merged onto the complete grid
    # below, so empty demand hours receive that day's weather directly.
    daily_weather = None
    if sessions is not None:
        temp_col = identify_columns(sessions, 'temperature')
        precip_col = identify_columns(sessions, 'precipitation')
        if temp_col or precip_col:
            # Compute daily averages
            s_time = identify_columns(sessions, 'start_time')
            if s_time:
                weather = sessions[[s_time]].copy()
                weather['_date'] = pd.to_datetime(
                    sessions[s_time], errors='coerce').dt.normalize()
                if temp_col:
                    weather['temperature'] = pd.to_numeric(
                        sessions[temp_col], errors='coerce')
                if precip_col:
                    weather['precipitation'] = pd.to_numeric(
                        sessions[precip_col], errors='coerce')

                daily_weather = weather.groupby('_date').agg({
                    **({'temperature': 'mean'} if temp_col else {}),
                    **({'precipitation': 'mean'} if precip_col else {}),
                }).reset_index()
                daily_weather.columns = ['date'] + list(daily_weather.columns[1:])

    # Keep missing weather explicit until the chronological train/test split.
    if 'temperature' not in df.columns:
        df['temperature'] = np.nan
    if 'precipitation' not in df.columns:
        df['precipitation'] = np.nan

    # -------------------------------------------------------------------------
    # ZERO-FILL: Rebuild a complete station x date x hour grid before lags.
    #
    # week1_wrapup builds the hourly table by grouping observed sessions only,
    # so hours with zero arrivals are absent. Using .shift(24) on that sparse
    # frame would point to the 24th preceding *observed* row, not the same
    # clock-hour yesterday. We rebuild the full grid here (matching Week 3's
    # approach) so that lag_24 and lag_168 are always clock-aligned.
    # -------------------------------------------------------------------------
    print("  Rebuilding full station x date x hour grid (zero-filling absent bins)...")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'hour'])

    all_stations = df['station'].unique()
    all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    all_hours = list(range(24))

    full_index = pd.MultiIndex.from_product(
        [all_stations, all_dates, all_hours],
        names=['station', 'date', 'hour']
    )
    full_grid = pd.DataFrame(index=full_index).reset_index()
    full_grid['date'] = pd.to_datetime(full_grid['date'])

    # Columns to carry from df onto the grid (non-lag, non-temporal)
    carry_cols = [
        'arrivals', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'dow', 'month', 'is_weekend', 'tou_tier_num',
        'temperature', 'precipitation',
    ]
    # Keep only columns that actually exist in df
    carry_cols = [c for c in carry_cols if c in df.columns]

    # Merge observed values onto the full grid
    df_obs = df[['station', 'date', 'hour'] + carry_cols].drop_duplicates(
        subset=['station', 'date', 'hour'])
    full_grid = full_grid.merge(df_obs, on=['station', 'date', 'hour'], how='left')
    if daily_weather is not None:
        # Drop sparse weather columns from the observed-hour merge and attach
        # daily values consistently to every station-hour on that date.
        full_grid = full_grid.drop(
            columns=['temperature', 'precipitation'], errors='ignore')
        full_grid = full_grid.merge(daily_weather, on='date', how='left')

    # Fill arrivals zero for absent bins; fill temporal features from the grid
    full_grid['arrivals'] = full_grid['arrivals'].fillna(0)

    # Re-derive temporal features for zero-fill rows where they are NaN
    full_grid['dow'] = full_grid['dow'].fillna(full_grid['date'].dt.dayofweek)
    full_grid['month'] = full_grid['month'].fillna(full_grid['date'].dt.month)
    full_grid['is_weekend'] = full_grid['is_weekend'].fillna(
        full_grid['date'].dt.dayofweek.isin([5, 6]).astype(float))
    full_grid['hour_sin'] = full_grid['hour_sin'].fillna(
        np.sin(2 * np.pi * full_grid['hour'] / 24))
    full_grid['hour_cos'] = full_grid['hour_cos'].fillna(
        np.cos(2 * np.pi * full_grid['hour'] / 24))
    full_grid['dow_sin'] = full_grid['dow_sin'].fillna(
        np.sin(2 * np.pi * full_grid['dow'] / 7))
    full_grid['dow_cos'] = full_grid['dow_cos'].fillna(
        np.cos(2 * np.pi * full_grid['dow'] / 7))

    # TOU tier: deterministic from hour, so zero-fill rows get correct value
    tou_map_fill = {}
    for h in [23, 0, 1, 2, 3, 4, 5, 6]:
        tou_map_fill[h] = 0
    for h in [7, 11, 12, 17, 18, 21, 22]:
        tou_map_fill[h] = 1
    for h in [8, 9, 10, 13, 14, 15, 16, 19, 20]:
        tou_map_fill[h] = 2
    full_grid['tou_tier_num'] = full_grid['tou_tier_num'].fillna(
        full_grid['hour'].map(tou_map_fill))

    for wcol in ['temperature', 'precipitation']:
        if wcol not in full_grid.columns:
            full_grid[wcol] = np.nan

    n_filled = len(full_grid) - len(df_obs)
    print(f"  Grid: {len(full_grid):,} rows total "
          f"({n_filled:,} zero-fill bins added across all stations)")

    df = full_grid.copy()

    # Lag features (computed per station on the complete, clock-aligned grid)
    print("  Computing lag features on zero-filled grid...")
    df = df.sort_values(['station', 'date', 'hour'])

    lag_features = []
    for station, group in df.groupby('station'):
        g = group.sort_values(['date', 'hour']).copy()
        # .shift() on a complete grid is now guaranteed clock-aligned
        g['lag_1'] = g['arrivals'].shift(1)
        g['lag_24'] = g['arrivals'].shift(24)
        g['lag_168'] = g['arrivals'].shift(168)
        # SHIFT(1) before rolling: window at time t uses arrivals[t-1..t-169]
        g['rolling_7d_mean'] = g['arrivals'].shift(1).rolling(168, min_periods=24).mean()
        lag_features.append(g)

    df = pd.concat(lag_features, ignore_index=True)

    # Drop rows with NaN lags (first 168 hours per station)
    n_before = len(df)
    df = df.dropna(subset=['lag_1', 'lag_24', 'lag_168', 'rolling_7d_mean'])
    print(f"  Dropped {n_before - len(df):,} rows with NaN lags "
          f"(first 7 days per station)")

    # Station encoding (label encoding)
    station_codes = {s: i for i, s in enumerate(sorted(df['station'].unique()))}
    df['station_code'] = df['station'].map(station_codes)

    # Define feature columns and target
    feature_cols = [
        'hour', 'dow', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'tou_tier_num', 'temperature', 'precipitation',
        'lag_1', 'lag_24', 'lag_168', 'rolling_7d_mean',
        'station_code',
    ]

    target_col = 'arrivals'

    # Verify all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing features: {missing}. Removing from feature set.")
        feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"  Feature matrix: {len(df):,} rows x {len(feature_cols)} features")

    return df, feature_cols, target_col


def run_xgboost_forecasting(hourly, sessions, output_dir):
    """
    Component 4: XGBoost hourly arrival forecasting.

    Train/test split: last 4 months (Sep-Dec 2021) as test.
    Baselines: last-week-same-hour, historical mean.
    """
    if not HAS_XGBOOST:
        print("\n" + "=" * 70)
        print("COMPONENT 4: XGBOOST FORECASTING - SKIPPED (xgboost not installed)")
        print("=" * 70)
        print("  Install with: pip install xgboost scikit-learn")
        return None, None

    if not HAS_SKLEARN:
        print("\n" + "=" * 70)
        print("COMPONENT 4: XGBOOST FORECASTING - SKIPPED (sklearn not installed)")
        print("=" * 70)
        return None, None

    print("\n" + "=" * 70)
    print("COMPONENT 4: XGBOOST DEMAND FORECASTING")
    print("=" * 70)

    df, feature_cols, target_col = build_feature_matrix(hourly, sessions)
    if df is None:
        return None, None

    # Train/test split: Sep-Dec 2021 as test
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    test_start = pd.Timestamp('2021-09-01')
    train_mask = df['date'] < test_start
    test_mask = df['date'] >= test_start

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_col].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, target_col].values

    # Fit weather imputers on training data only; never backfill from future.
    for weather_col in ['temperature', 'precipitation']:
        if weather_col not in feature_cols:
            continue
        feature_idx = feature_cols.index(weather_col)
        train_values = X_train[:, feature_idx]
        train_median = float(np.nanmedian(train_values))
        if not np.isfinite(train_median):
            train_median = 0.0
        X_train[:, feature_idx] = np.where(
            np.isnan(X_train[:, feature_idx]), train_median,
            X_train[:, feature_idx])
        X_test[:, feature_idx] = np.where(
            np.isnan(X_test[:, feature_idx]), train_median,
            X_test[:, feature_idx])
        print(f"  {weather_col} NaN fill: training median={train_median:.2f}")

    print(f"\n  Train: {len(X_train):,} rows (before Sep 2021)")
    print(f"  Test:  {len(X_test):,} rows (Sep-Dec 2021)")

    if len(X_train) < 1000 or len(X_test) < 100:
        print("  [ERROR] Insufficient data for train/test split.")
        return None, None

    # Demand-shift diagnostics for the fixed split
    train_mean_arrivals = float(np.mean(y_train))
    test_mean_arrivals = float(np.mean(y_test))
    growth_ratio = (
        float(test_mean_arrivals / train_mean_arrivals)
        if train_mean_arrivals > 0 else np.nan
    )
    print(f"  Mean arrivals: train={train_mean_arrivals:.3f}, "
          f"test={test_mean_arrivals:.3f}, ratio={growth_ratio:.3f}")

    # --- Baseline models ---
    print("\n  Baseline models:")

    # Baseline 1: Last-week-same-hour
    baseline_lwsh = df.loc[test_mask, 'lag_168'].values
    mae_lwsh = mean_absolute_error(y_test, baseline_lwsh)
    rmse_lwsh = np.sqrt(mean_squared_error(y_test, baseline_lwsh))
    r2_lwsh = r2_score(y_test, baseline_lwsh)
    print(f"    Last-week-same-hour: MAE={mae_lwsh:.3f}, RMSE={rmse_lwsh:.3f}, R^2={r2_lwsh:.3f}")

    # Baseline 2: Historical mean per (station, hour, dow_type)
    df['_dow_type'] = df['is_weekend'].map({0: 'weekday', 1: 'weekend'})
    hist_mean = df.loc[train_mask].groupby(['station', 'hour', '_dow_type'])[target_col].mean()

    pred_hist = []
    for idx in df.loc[test_mask].index:
        key = (df.loc[idx, 'station'], df.loc[idx, 'hour'], df.loc[idx, '_dow_type'])
        pred_hist.append(hist_mean.get(key, y_train.mean()))

    pred_hist = np.array(pred_hist)
    mae_hist = mean_absolute_error(y_test, pred_hist)
    rmse_hist = np.sqrt(mean_squared_error(y_test, pred_hist))
    r2_hist = r2_score(y_test, pred_hist)
    print(f"    Historical mean:     MAE={mae_hist:.3f}, RMSE={rmse_hist:.3f}, R^2={r2_hist:.3f}")

    # --- XGBoost ---
    print("\n  Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # 5-fold time series CV on training set.
    # Sort by (date, hour, station) so fold boundaries cut at consistent
    # calendar times across all stations.
    print("  Running 5-fold time series CV...")
    train_sort_idx = (df.loc[train_mask]
                      .sort_values(['date', 'hour', 'station'])
                      .index)
    X_train_cv = df.loc[train_sort_idx, feature_cols].values
    y_train_cv = df.loc[train_sort_idx, target_col].values

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        model, X_train_cv, y_train_cv, cv=tscv,
        scoring='neg_mean_absolute_error', n_jobs=-1
    )
    cv_mae = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    print(f"    CV MAE: {cv_mae:.3f} +/- {cv_mae_std:.3f}")

    # Train on full training set
    model.fit(X_train, y_train, verbose=False)

    # Predict on test
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Arrivals can't be negative

    mae_xgb = mean_absolute_error(y_test, y_pred)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_xgb = r2_score(y_test, y_pred)
    mean_error = float(np.mean(y_pred - y_test))
    underprediction_rate = float(np.mean(y_pred < y_test))
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_err = np.where(y_test > 0, (y_pred - y_test) / y_test, np.nan)
        pct_err_mean = np.nanmean(pct_err)
    mean_pct_error = float(pct_err_mean * 100) if np.isfinite(pct_err_mean) else np.nan
    print(f"\n  XGBoost test performance:")
    print(f"    MAE:  {mae_xgb:.3f}")
    print(f"    RMSE: {rmse_xgb:.3f}")
    print(f"    R^2:   {r2_xgb:.3f}")
    print(f"    Mean error (pred-actual): {mean_error:.3f}")
    print(f"    Underprediction rate: {underprediction_rate:.3f}")

    # Monthly breakdown within the holdout period
    monthly_holdout_metrics = []
    test_eval = df.loc[test_mask, ['date']].copy()
    test_eval['actual'] = y_test
    test_eval['pred'] = y_pred
    test_eval['month'] = test_eval['date'].dt.to_period('M').astype(str)
    for month, g in test_eval.groupby('month'):
        if len(g) == 0:
            continue
        month_mae = float(mean_absolute_error(g['actual'], g['pred']))
        month_rmse = float(np.sqrt(mean_squared_error(g['actual'], g['pred'])))
        month_mean_error = float((g['pred'] - g['actual']).mean())
        monthly_holdout_metrics.append({
            'month': month,
            'n': int(len(g)),
            'mae': month_mae,
            'rmse': month_rmse,
            'mean_error': month_mean_error,
        })

    # Feature importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    print(f"\n  Top 5 features:")
    for _, row in feat_imp.head(5).iterrows():
        print(f"    {row['feature']:20s}: {row['importance']:.4f}")

    save_csv(feat_imp, output_dir, 'xgboost_feature_importance.csv')

    # Comparison table
    comparison = {
        'models': {
            'last_week_same_hour': {
                'mae': float(mae_lwsh),
                'rmse': float(rmse_lwsh),
                'r2': float(r2_lwsh),
            },
            'historical_mean': {
                'mae': float(mae_hist),
                'rmse': float(rmse_hist),
                'r2': float(r2_hist),
            },
            'xgboost': {
                'mae': float(mae_xgb),
                'rmse': float(rmse_xgb),
                'r2': float(r2_xgb),
                'cv_mae': float(cv_mae),
                'cv_mae_std': float(cv_mae_std),
            },
        },
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'n_features': len(feature_cols),
        'test_period': 'Sep-Dec 2021',
        'split': {
            'train_end_exclusive': '2021-09-01',
            'test_start_inclusive': '2021-09-01',
            'test_period': 'Sep-Dec 2021',
        },
        'demand_shift': {
            'train_mean_arrivals': train_mean_arrivals,
            'test_mean_arrivals': test_mean_arrivals,
            'growth_ratio_test_over_train': growth_ratio,
            'note': 'Fixed late-period holdout may inflate absolute errors under trend; '
                    'relative model comparisons remain informative.',
        },
        'holdout_bias_diagnostics': {
            'mean_error_pred_minus_actual': mean_error,
            'mean_pct_error': mean_pct_error,
            'underprediction_rate': underprediction_rate,
        },
        'monthly_holdout_metrics': monthly_holdout_metrics,
        'improvement_over_lwsh_mae': float((mae_lwsh - mae_xgb) / mae_lwsh * 100),
        'improvement_over_hist_mae': float((mae_hist - mae_xgb) / mae_hist * 100),
    }

    improvement_pct = comparison['improvement_over_lwsh_mae']
    comparison['fleet_sizing_impact'] = 'not_evaluated'
    comparison['fleet_sizing_impact_note'] = (
        'Forecast error alone does not establish operational impact. '
        'A holdout forecast-to-sizing/activation experiment is required.')
    print(f"\n  MAE improvement over LWSH baseline: {improvement_pct:.1f}%")
    print("  Fleet sizing impact: not evaluated by this forecasting experiment")

    # Save
    results_path = Path(output_dir) / 'xgboost_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    print(f"  [SAVED] {results_path}")

    return comparison, feat_imp


def plot_xgboost_results(hourly, sessions, comparison, feat_imp, output_dir):
    """Generate XGBoost diagnostic plots."""
    if comparison is None:
        return

    print("\n  Generating XGBoost plots...")

    # Plot 1: Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['last_week_same_hour', 'historical_mean', 'xgboost']
    labels = ['Last-Week\nSame-Hour', 'Historical\nMean', 'XGBoost']
    colors = ['#95a5a6', '#3498db', '#e74c3c']

    for i, metric in enumerate(['mae', 'rmse', 'r2']):
        ax = axes[i]
        values = [comparison['models'][m][metric] for m in models]
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_title(metric.upper(), fontsize=12)
        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle("Hourly Arrival Forecasting: Model Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = Path(output_dir) / 'figures' / 'xgboost_model_comparison.png'
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  [SAVED] {fig_path}")

    # Plot 2: Feature importance
    if feat_imp is not None and len(feat_imp) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(15, len(feat_imp))
        top_feats = feat_imp.head(top_n).iloc[::-1]  # reverse for horizontal bar

        ax.barh(top_feats['feature'], top_feats['importance'],
                color='steelblue', alpha=0.8)
        ax.set_xlabel('Feature Importance (gain)')
        ax.set_title('XGBoost Feature Importance (Top 15)')
        fig.tight_layout()
        fig_path = Path(output_dir) / 'figures' / 'xgboost_feature_importance.png'
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"  [SAVED] {fig_path}")


# =============================================================================
# PARAMETER SUMMARY (Consolidated Phase 1 Output)
# =============================================================================

def build_parameter_summary(svc_summary, erlang_results, xgb_results,
                            acn_results, nhpp_rates, output_dir):
    """
    Consolidate all Phase 1 + Phase 2 kickoff results into parameter_summary.json.

    This file feeds all downstream phases (SimPy, scheduling, dashboard).
    """
    print("\n" + "=" * 70)
    print("BUILDING PARAMETER SUMMARY")
    print("=" * 70)

    summary = {
        'phase': 'Phase 1b + Phase 2a',
        'dataset': 'Jiaxing (primary)',
        'n_sessions': 441077,
        'n_stations': 13,

        'service_time': {},
        'arrival_process': {
            'poisson_rejected': True,
            'model': 'NHPP_piecewise_constant',
            'note': 'Poisson rejected at all 13 stations. NHPP with hourly rates '
                    'is the arrival model for simulation.',
        },
        'fleet_sizing': {},
        'forecasting': {},
        'acn_validation': None,
    }

    # Service time parameters
    if svc_summary:
        for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
            if ctype in svc_summary:
                s = svc_summary[ctype]
                summary['service_time'][ctype] = {
                    'n': s.get('n'),
                    'mean_min': s.get('mean_min'),
                    'cv': s.get('cv'),
                    'best_distribution': s.get('best_distribution'),
                    'best_aic': s.get('best_aic'),
                    'fit_params': s.get('fits', {}).get(
                        s.get('best_distribution', ''), {}).get('params', {}),
                    'simulation_sampler': s.get('simulation_sampler', {}),
                }
        if 'merge_decision' in svc_summary:
            summary['service_time']['merge_decision'] = svc_summary['merge_decision']

    # NHPP rates reference
    if nhpp_rates is not None:
        summary['arrival_process']['nhpp_rates_file'] = 'nhpp_rate_functions.csv'
        summary['arrival_process']['n_rate_entries'] = len(nhpp_rates)

    # Erlang-C key findings (scenario-based)
    if erlang_results is not None and len(erlang_results) > 0:
        for (station, scenario), sdf in erlang_results.groupby(
                ['station', 'sizing_scenario']):
            mms_star = sdf.loc[sdf['p_wait_mms'] < 0.05, 's'].min()
            mgs_star = sdf.loc[sdf['p_wait_mgs'] < 0.05, 's'].min()
            cv_source = (sdf['cv_source'].iloc[0]
                         if 'cv_source' in sdf.columns else 'unknown')
            service_param_scope = (
                sdf['service_param_scope'].iloc[0]
                if 'service_param_scope' in sdf.columns else 'unknown'
            )
            # Key format: "StationName__scenario"
            key = f"{station}__{scenario}"
            summary['fleet_sizing'][key] = {
                's_star_mms': int(mms_star) if pd.notna(mms_star) else None,
                's_star_mgs': int(mgs_star) if pd.notna(mgs_star) else None,
                'sizing_scenario': scenario,
                'dominant_type': sdf['dominant_type'].iloc[0],
                'peak_lambda': float(sdf['peak_lambda_per_hour'].iloc[0]),
                'cv': float(sdf['cv'].iloc[0]),
                'cv_source': cv_source,
                'mgs_correction_factor': float(
                    (1 + float(sdf['cv'].iloc[0]) ** 2) / 2),
                'service_param_scope': service_param_scope,
            }

        # Scope metadata — use rsplit to handle station names containing '__'
        sized_stations = sorted(set(
            k.rsplit('__', 1)[0] for k in summary['fleet_sizing'].keys()
        ))
        summary['fleet_sizing_scope'] = {
            'type': 'representative_subset',
            'n_stations_sized': len(sized_stations),
            'n_stations_total': 13,
            'representative_stations': sized_stations,
            'note': 'Fleet sizing covers 4 representative stations selected '
                    'to span the archetype space (expressway, high-volume, '
                    'institutional, mixed). Do NOT extrapolate to all 13 '
                    'stations without running sizing for each individually.',
        }
        summary['fleet_sizing_assumptions'] = {
            'scenarios': {
                'historical_avg': 'Mean hourly rate across all dates '
                                  '(Jan 2020 – Dec 2021). Descriptive only.',
                'recent_quarter': 'Mean hourly rate over last 90 days '
                                  'before global study end. Operational sizing.',
                'stress_case': 'recent_quarter × 1.2. Planning margin.',
            },
            'service_param_scope': 'station-level charger-type mix '
                                   'with type-level fitted moments',
        }

    # ACN validation summary (explicit key; null when skipped due no data)
    if acn_results is not None and isinstance(acn_results, pd.DataFrame) and len(acn_results) > 0:
        status = 'completed'
        if 'status' in acn_results.columns:
            status_vals = acn_results['status'].dropna().astype(str).unique().tolist()
            if len(status_vals) > 0:
                status = status_vals[0]

        if status == 'skipped_no_data':
            summary['acn_validation'] = None
            summary['acn_validation_status'] = {
                'status': 'skipped_no_data',
                'reason': acn_results['note'].dropna().iloc[0]
                if 'note' in acn_results.columns and acn_results['note'].notna().any()
                else 'acn_clean dataset not found',
            }
        else:
            key_tests = ['unconditional_DI', 'conditional_DI_mean', 'KS_exponential_IAT', 'service_time_CV']
            key_stats = {}
            if 'test' in acn_results.columns and 'statistic' in acn_results.columns:
                for test_name in key_tests:
                    m = acn_results[acn_results['test'] == test_name]
                    if len(m) > 0:
                        key_stats[test_name] = float(m['statistic'].iloc[0])
            n_tests = int((acn_results['test'] != 'status').sum()) if 'test' in acn_results.columns else 0
            summary['acn_validation'] = {
                'status': status,
                'n_rows': int(len(acn_results)),
                'n_tests': n_tests,
                'key_stats': key_stats,
            }

    # XGBoost results
    if xgb_results:
        summary['forecasting'] = {
            'model': 'XGBoost',
            'test_mae': xgb_results['models']['xgboost']['mae'],
            'test_rmse': xgb_results['models']['xgboost']['rmse'],
            'test_r2': xgb_results['models']['xgboost']['r2'],
            'improvement_over_baseline': xgb_results['improvement_over_lwsh_mae'],
            'fleet_sizing_impact': xgb_results['fleet_sizing_impact'],
            'split': xgb_results.get('split', {}),
            'demand_shift': xgb_results.get('demand_shift', {}),
            'holdout_bias_diagnostics': xgb_results.get('holdout_bias_diagnostics', {}),
        }

    # Save
    path = Path(output_dir) / 'parameter_summary.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_builtin(summary), f, indent=2)
    print(f"  [SAVED] {path}")

    return summary


# =============================================================================
# SANITY CHECKS
# =============================================================================

def run_sanity_checks(svc_summary, erlang_results, xgb_results):
    """
    Verify results are consistent with known EDA values and each other.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    issues = []

    # Check 1: CV values should match EDA (within tolerance)
    expected_cv = {'DC_Fast': 0.57, 'Level_2': 1.39, 'Mixed': 0.60}
    if svc_summary:
        for ctype, expected in expected_cv.items():
            if ctype in svc_summary:
                actual = svc_summary[ctype]['cv']
                diff = abs(actual - expected)
                status = "OK" if diff < 0.15 else "MISMATCH"
                print(f"  [{status}] {ctype} CV: expected ~{expected}, got {actual:.3f} "
                      f"(diff={diff:.3f})")
                if diff >= 0.15:
                    issues.append(f"{ctype} CV mismatch: expected ~{expected}, got {actual:.3f}")

    # Check 2: Erlang-C stability
    if erlang_results is not None:
        unstable = erlang_results[erlang_results['p_wait_mms'] >= 0.99]
        n_unstable = len(unstable[unstable['s'] > 20])
        if n_unstable > 0:
            print(f"  [WARN] {n_unstable} station-s combinations still near-unstable at s>20")
            issues.append(f"{n_unstable} near-unstable configurations at s>20")
        else:
            print(f"  [OK] All representative stations reach stability before s=20")

    # Check 3: L2 should need MORE chargers under M/G/s (CV>1)
    if erlang_results is not None:
        l2_stations = erlang_results[erlang_results['dominant_type'] == 'Level_2']
        if len(l2_stations) > 0:
            for station in l2_stations['station'].unique():
                sdf = l2_stations[l2_stations['station'] == station]
                mms_star = sdf.loc[sdf['p_wait_mms'] < 0.05, 's'].min()
                mgs_star = sdf.loc[sdf['p_wait_mgs'] < 0.05, 's'].min()
                if pd.notna(mms_star) and pd.notna(mgs_star):
                    if mgs_star < mms_star:
                        issues.append(f"L2 station {station}: M/G/s should need MORE "
                                      f"chargers (got s*_mgs={mgs_star} < s*_mms={mms_star})")
                        print(f"  [WARN] L2 strict reversal at {station}")
                    elif mgs_star == mms_star:
                        print(f"  [NOTE] L2 station {station}: s*_mgs={mgs_star} equals "
                              f"s*_mms={mms_star} (discrete threshold tie)")
                    else:
                        print(f"  [OK] L2 station {station}: M/G/s needs more chargers (expected)")

    # Check 4: DC_Fast should need FEWER chargers under M/G/s (CV<1)
    if erlang_results is not None:
        dc_stations = erlang_results[erlang_results['dominant_type'] == 'DC_Fast']
        if len(dc_stations) > 0:
            for station in dc_stations['station'].unique():
                sdf = dc_stations[dc_stations['station'] == station]
                mms_star = sdf.loc[sdf['p_wait_mms'] < 0.05, 's'].min()
                mgs_star = sdf.loc[sdf['p_wait_mgs'] < 0.05, 's'].min()
                if pd.notna(mms_star) and pd.notna(mgs_star):
                    if mgs_star >= mms_star:
                        # This could be a finding if gap is 0
                        print(f"  [NOTE] DC station {station}: s*_mgs={mgs_star} >= "
                              f"s*_mms={mms_star} (gap may be 0 at small s)")
                    else:
                        print(f"  [OK] DC station {station}: M/G/s needs fewer chargers (expected)")

    # Check 5: XGBoost should beat baselines
    if xgb_results:
        xgb_mae = xgb_results['models']['xgboost']['mae']
        lwsh_mae = xgb_results['models']['last_week_same_hour']['mae']
        hist_mae = xgb_results['models']['historical_mean']['mae']
        if xgb_mae > lwsh_mae:
            print(f"  [WARN] XGBoost MAE ({xgb_mae:.3f}) worse than LWSH baseline "
                  f"({lwsh_mae:.3f})")
            issues.append("XGBoost underperforms last-week-same-hour baseline")
        else:
            print(f"  [OK] XGBoost beats LWSH baseline by "
                  f"{(lwsh_mae-xgb_mae)/lwsh_mae*100:.1f}%")

    if not issues:
        print("\n  All sanity checks passed.")
    else:
        print(f"\n  {len(issues)} issue(s) flagged:")
        for i in issues:
            print(f"    - {i}")

    return issues


# =============================================================================
# MAIN
# =============================================================================

def main():
    configure_console_output()
    parser = argparse.ArgumentParser(description='Week 4: Phase 1b + Phase 2 Kickoff')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing Week 1 output files')
    parser.add_argument('--week3-dir', type=str,
                        default=str(RESULTS_DIR / 'week3_results'),
                        help='Directory containing Week 3 output files')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'week4_results'),
                        help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("WEEK 4: SERVICE TIME FITTING + ACN CROSS-VALIDATION + PHASE 2 KICKOFF")
    print("=" * 70)

    # Load data
    sessions = load_jiaxing(args.data_dir)
    hourly = load_hourly(args.data_dir)
    acn = load_acn(args.data_dir)
    nhpp_rates = load_nhpp_rates(args.week3_dir)

    # Component 1: Service time distribution fitting
    svc_results, svc_summary = run_service_time_fitting(sessions, output_dir)
    if svc_summary:
        plot_service_time_qq(sessions, svc_summary, output_dir)
        plot_service_time_fitted_pdf(sessions, svc_summary, output_dir)

    # Component 2: ACN cross-validation
    acn_results = run_acn_crossval(acn, output_dir)

    # Component 3: Erlang-C and M/G/s
    if svc_summary is None:
        # Fallback: use EDA values
        svc_summary = {
            'DC_Fast': {'mean_min': 35, 'cv': 0.57},
            'Level_2': {'mean_min': 92, 'cv': 1.39},
            'Mixed': {'mean_min': 43, 'cv': 0.60},
        }
    erlang_results = run_erlang_c_analysis(sessions, nhpp_rates, svc_summary, output_dir)
    plot_erlang_c_comparison(erlang_results, output_dir)

    # Component 4: XGBoost forecasting
    xgb_results, feat_imp = run_xgboost_forecasting(hourly, sessions, output_dir)
    plot_xgboost_results(hourly, sessions, xgb_results, feat_imp, output_dir)

    # Consolidated parameter summary
    param_summary = build_parameter_summary(
        svc_summary, erlang_results, xgb_results,
        acn_results, nhpp_rates, output_dir
    )

    # Sanity checks
    issues = run_sanity_checks(svc_summary, erlang_results, xgb_results)

    # Save metadata
    metadata = {
        'week': 4,
        'components': [
            'Service Time Distribution Fitting',
            'ACN Cross-Validation',
            'Erlang-C and M/G/s Fleet Sizing',
            'XGBoost Demand Forecasting',
        ],
        'files_produced': [
            'service_time_fits.csv',
            'service_time_summary.json',
            'acn_crossval.csv',
            'erlang_c_results_4rep.csv',
            'mgs_comparison_4rep.csv',
            'xgboost_results.json',
            'xgboost_feature_importance.csv',
            'parameter_summary.json',
        ],
        'figures_produced': [
            f for f in os.listdir(output_dir / 'figures')
            if f.endswith('.png')
        ] if (output_dir / 'figures').exists() else [],
        'sanity_issues': issues,
    }
    with open(output_dir / 'week4_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("WEEK 4 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    print(f"Files: {', '.join(metadata['files_produced'])}")
    print(f"Figures: {len(metadata['figures_produced'])} PNG files")
    if issues:
        print(f"Sanity issues: {len(issues)}")


if __name__ == '__main__':
    main()
