"""
Week 5: LSTM Forecasting + SimPy Engine Skeleton
==================================================
Phase 2a — LSTM + Simulation Engine

Components:
  1. LSTM hourly arrival forecasting
     - Direct multi-output: (168 × F) → (24,) next-24-hour arrivals
     - Station representation: learned embedding (dim=8) for 13 stations
     - Global model (all stations pooled) trained first
     - Feature ablation: retrain without weather to quantify its value
     - Comparison table: LSTM vs XGBoost vs baselines
     - Fleet sizing relevance check

  2. SimPy discrete-event simulation engine (skeleton + NHPP mode)
     - ArrivalGenerator: Poisson / NHPP / ML-replay modes
     - ServiceProcess: per-type distribution sampling from fitted params
     - FaultInjector: session-start Bernoulli by charger type
     - SchedulingPolicy: FCFS (pluggable interface for later policies)
     - MetricsCollector: per-session wait, utilization, throughput

Assumptions:
  - LSTM target is direct multi-output (hours t+1 through t+24), NOT recursive
  - ML replay mode is piecewise-homogeneous Poisson with ML-estimated hourly
    means; it does NOT discover sub-hourly arrival structure
  - Fault model treats faults as session-start events; mid-session failure
    modeling deferred unless empirical records clearly require it
  - Hourly parquet has been zero-filled (complete station×date×hour grid)
  - Normalization uses train-set statistics only

Input files:
  - jiaxing_hourly.parquet  (zero-filled station-hour grid, ~228k rows)
  - parameter_summary.json  (service time params, NHPP reference)
  - service_time_summary.json (detailed fit params for SimPy sampling)
  - nhpp_rate_functions.csv  (13 stations × 24 hours = 312 rows)
  - xgboost_results.json    (baseline comparison numbers)

Output files:
  - week5_results/lstm_results.json
  - week5_results/lstm_comparison.json  (4-model comparison table)
  - week5_results/lstm_model.pt         (if LSTM trained successfully)
  - week5_results/scaler_params.json    (train-set normalization stats)
  - week5_results/figures/*.png
  - week5_results/week5_metadata.json
  - week5_results/sim_engine.py         (standalone SimPy module, copied)

Usage:
  python week5_analysis.py --data-dir ./data --week4-dir ./week4_results --output-dir ./week5_results

Date: Week 5, Mar 2026
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR, SIM_ENGINE_PATH, to_builtin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Optional dependency checks ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[INFO] PyTorch not available. LSTM training will be skipped.")

try:
    import simpy
    HAS_SIMPY = True
except ImportError:
    HAS_SIMPY = False
    print("[INFO] SimPy not available. Simulation engine will be skeleton only.")

from scipy import stats as sp_stats


# ============================================================================
# CONSOLE SAFETY (Windows encoding)
# ============================================================================

def configure_console_output():
    """Handle Windows console encoding issues."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass


def detect_parquet_engine() -> Optional[str]:
    """Return available parquet engine, or None."""
    if find_spec('pyarrow') is not None:
        return 'pyarrow'
    if find_spec('fastparquet') is not None:
        return 'fastparquet'
    return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_hourly(data_dir: str) -> Optional[pd.DataFrame]:
    """Load jiaxing_hourly.parquet (zero-filled grid)."""
    p = Path(data_dir)
    engine = detect_parquet_engine()
    for fname in ['jiaxing_hourly.parquet', 'jiaxing_hourly.csv']:
        fpath = p / fname
        if fpath.exists():
            if fname.endswith('.parquet'):
                if engine is None:
                    print("[WARN] No parquet engine installed. Looking for CSV fallback.")
                    continue
                try:
                    df = pd.read_parquet(fpath, engine=engine)
                except Exception as exc:
                    print(f"[WARN] Failed reading {fname} with {engine}: {exc}")
                    continue
            else:
                df = pd.read_csv(fpath)
            for col in ['date', 'date_dt', 'datetime']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"[OK] Loaded {fname}: {df.shape[0]:,} rows × {df.shape[1]} cols")
            return df
    print("[ERROR] jiaxing_hourly not found.")
    return None


def load_json(fpath: Path, label: str) -> Optional[dict]:
    """Load a JSON file."""
    if not fpath.exists():
        print(f"[WARN] {label} not found: {fpath}")
        return None
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] Loaded {label}: {fpath.name}")
    return data


def load_nhpp_rates(week3_dir: str) -> Optional[pd.DataFrame]:
    """Load NHPP rate functions CSV."""
    for parent in [week3_dir, '.']:
        fpath = Path(parent) / 'nhpp_rate_functions.csv'
        if fpath.exists():
            df = pd.read_csv(fpath)
            print(f"[OK] Loaded nhpp_rate_functions.csv: {len(df)} rows")
            return df
    # Also check week4_dir parent
    print("[WARN] nhpp_rate_functions.csv not found.")
    return None


# ============================================================================
# COMPONENT 1: LSTM FORECASTING
# ============================================================================

# ── 1a. Dataset construction ────────────────────────────────────────

def identify_hourly_columns(df: pd.DataFrame) -> dict:
    """Identify key columns in hourly parquet with name variations."""
    candidates = {
        'station': ['station_name', 'station_id', 'station'],
        'hour': ['hour_of_day', 'hour'],
        'arrivals': ['arrivals', 'arrival_count'],
        'date': ['date_dt', 'date', 'datetime'],
        'datetime': ['datetime', 'date_dt'],
        'temperature': ['mean_temperature', 'temperature'],
        'precipitation': ['total_precipitation', 'precipitation'],
        'dow': ['day_of_week', 'dow'],
        'month': ['month'],
        'is_weekend': ['is_weekend'],
        'hour_sin': ['hour_sin'],
        'hour_cos': ['hour_cos'],
        'tou_price': ['tou_electricity_price', 'tou_tier_num', 'tou_price'],
    }
    found = {}
    for key, names in candidates.items():
        for name in names:
            if name in df.columns:
                found[key] = name
                break
    return found


def prepare_lstm_data(hourly: pd.DataFrame, output_dir: Path) -> Optional[dict]:
    """
    Build LSTM sliding-window dataset from zero-filled hourly parquet.

    Architecture decisions (locked):
      - Input: 168 hours (1 week lookback)
      - Output: 24 hours (direct multi-output, NOT recursive)
      - Station: learned embedding (dim=8), index passed separately
      - Features per timestep: arrivals, hour_sin, hour_cos, dow (7 one-hot),
        tou_price, temperature, precipitation → F ≈ 12-13
      - Train: first 18 months, Val: next 2 months, Test: last 4 months
        (test matches XGBoost holdout exactly: Sep-Dec 2021)

    Returns dict with arrays and metadata, or None on failure.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 1a: LSTM DATA PREPARATION")
    print("=" * 70)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not installed.")
        return None

    cols = identify_hourly_columns(hourly)
    required = ['station', 'hour', 'arrivals']
    for r in required:
        if r not in cols:
            print(f"  [ERROR] Cannot find '{r}' column in hourly data.")
            return None

    df = hourly.copy()
    df[cols['station']] = df[cols['station']].astype(str).str.strip()

    # ── Ensure datetime column ─────────────────────────────────────
    if 'datetime' in cols:
        df['_dt'] = pd.to_datetime(df[cols['datetime']])
    elif 'date' in cols:
        df['_dt'] = pd.to_datetime(df[cols['date']]) + \
                    pd.to_timedelta(df[cols['hour']], unit='h')
    else:
        print("  [ERROR] Cannot construct datetime.")
        return None

    df = df.sort_values([cols['station'], '_dt']).reset_index(drop=True)

    # ── Station encoding ───────────────────────────────────────────
    stations = sorted(df[cols['station']].unique())
    station_to_idx = {s: i for i, s in enumerate(stations)}
    df['station_idx'] = df[cols['station']].map(station_to_idx)
    n_stations = len(stations)
    print(f"  Stations: {n_stations}")

    # ── Feature matrix construction ────────────────────────────────
    feature_cols = []

    # Arrivals (will be the primary feature and target)
    feature_cols.append(cols['arrivals'])

    # Cyclical hour encoding
    if 'hour_sin' in cols and 'hour_cos' in cols:
        feature_cols.extend([cols['hour_sin'], cols['hour_cos']])
    else:
        df['_hour_sin'] = np.sin(2 * np.pi * df[cols['hour']] / 24)
        df['_hour_cos'] = np.cos(2 * np.pi * df[cols['hour']] / 24)
        feature_cols.extend(['_hour_sin', '_hour_cos'])

    # Day-of-week one-hot (7 dims)
    if 'dow' in cols:
        dow_vals = df[cols['dow']].values.astype(int)
    else:
        dow_vals = df['_dt'].dt.dayofweek.values.astype(int)
    dow_onehot = np.eye(7)[dow_vals]
    for i in range(7):
        col_name = f'_dow_{i}'
        df[col_name] = dow_onehot[:, i]
        feature_cols.append(col_name)

    # TOU price
    if 'tou_price' in cols:
        df[cols['tou_price']] = pd.to_numeric(df[cols['tou_price']], errors='coerce')
        feature_cols.append(cols['tou_price'])
        df[cols['tou_price']] = df[cols['tou_price']].fillna(0)

    # Weather features (will also build ablation set without these)
    # NOTE: imputation deferred until after split assignment to avoid
    # leaking val/test statistics into training inputs.
    weather_cols = []
    if 'temperature' in cols:
        df[cols['temperature']] = pd.to_numeric(df[cols['temperature']], errors='coerce')
        feature_cols.append(cols['temperature'])
        weather_cols.append(cols['temperature'])
    if 'precipitation' in cols:
        df[cols['precipitation']] = pd.to_numeric(df[cols['precipitation']], errors='coerce')
        feature_cols.append(cols['precipitation'])
        weather_cols.append(cols['precipitation'])

    n_features = len(feature_cols)
    print(f"  Features per timestep: {n_features}")
    print(f"  Feature columns: {feature_cols}")
    print(f"  Weather columns (for ablation): {weather_cols}")

    # ── Train / Val / Test split by date ───────────────────────────
    # Dataset: Jan 2020 - Dec 2021 (24 months)
    # Train: Jan 2020 - Jun 2021 (18 months)
    # Val: Jul 2021 - Aug 2021 (2 months)
    # Test: Sep 2021 - Dec 2021 (4 months, matches XGBoost)
    train_end = pd.Timestamp('2021-07-01')
    val_end = pd.Timestamp('2021-09-01')

    df['_split'] = 'train'
    df.loc[df['_dt'] >= train_end, '_split'] = 'val'
    df.loc[df['_dt'] >= val_end, '_split'] = 'test'

    split_counts = df['_split'].value_counts()
    print(f"\n  Split sizes:")
    for s in ['train', 'val', 'test']:
        print(f"    {s}: {split_counts.get(s, 0):,} rows")

    # ── Weather imputation using TRAIN-ONLY statistics ─────────────
    # Must happen after split assignment to avoid future data leakage.
    imputation_fills = {}
    if 'temperature' in cols:
        train_temp_mean = float(
            df.loc[df['_split'] == 'train', cols['temperature']].mean())
        if not np.isfinite(train_temp_mean):
            train_temp_mean = 0.0
        df[cols['temperature']] = df[cols['temperature']].fillna(train_temp_mean)
        imputation_fills[cols['temperature']] = train_temp_mean
        print(f"  Temperature fill (train mean): {train_temp_mean:.2f}")
    if 'precipitation' in cols:
        # Precipitation: fill with 0 (no rain is the safe default)
        df[cols['precipitation']] = df[cols['precipitation']].fillna(0)
        imputation_fills[cols['precipitation']] = 0.0

    # ── Normalization using TRAIN-ONLY statistics (global, not per-station) ──
    scaler_params = {'_imputation_fills': imputation_fills}
    df_norm = df.copy()

    for col in feature_cols:
        train_vals = df.loc[df['_split'] == 'train', col]
        mean_val = float(train_vals.mean())
        std_val = float(train_vals.std())
        if not np.isfinite(mean_val):
            mean_val = 0.0
        if std_val < 1e-8:
            std_val = 1.0  # constant feature, avoid div-by-zero
        if not np.isfinite(std_val):
            std_val = 1.0
        scaler_params[col] = {'mean': mean_val, 'std': std_val}
        normalized = (pd.to_numeric(df[col], errors='coerce') - mean_val) / std_val
        df_norm[col] = normalized.fillna(0.0)

    # Save scaler params
    scaler_path = output_dir / 'scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  [SAVED] {scaler_path}")

    # ── Build sliding windows per station ──────────────────────────
    # Convention (locked):
    #   Input:  rows [t-168, t-1] — the 168 hours ending BEFORE the forecast origin
    #   Target: rows [t, t+23]   — the 24 hours starting AT the forecast origin
    #   So the model sees up to timestamp tau and predicts tau+1h through tau+24h
    #   where tau is the datetime of row t-1.
    #
    # Split assignment: based on the ENTIRE target block [t, t+23].
    #   All 24 target rows must belong to the same split.
    #   Samples whose target crosses a split boundary are DROPPED.
    LOOKBACK = 168   # 1 week
    HORIZON = 24     # next 24 hours (direct multi-output)

    arrivals_col = cols['arrivals']
    arrivals_idx_in_features = feature_cols.index(arrivals_col)

    windows_X = []      # (N, 168, F)
    windows_y = []      # (N, 24)
    windows_station = [] # (N,) station index
    windows_split = []   # (N,) 'train'/'val'/'test'
    n_boundary_dropped = 0

    for station in stations:
        sdf = df_norm[df_norm[cols['station']] == station].sort_values('_dt')
        feature_matrix = sdf[feature_cols].values  # (T, F)
        arrivals_raw = df.loc[sdf.index, arrivals_col].values  # un-normalized for target
        split_labels = sdf['_split'].values
        station_idx = station_to_idx[station]

        T = len(sdf)
        for t in range(LOOKBACK, T - HORIZON + 1):
            # Target block split check: all 24 target rows must share the same split
            target_splits = split_labels[t: t + HORIZON]
            unique_splits = set(target_splits)
            if len(unique_splits) > 1:
                n_boundary_dropped += 1
                continue

            target_split = target_splits[0]

            x = feature_matrix[t - LOOKBACK: t]  # (168, F) rows [t-168 .. t-1]
            y = arrivals_raw[t: t + HORIZON]       # (24,) rows [t .. t+23]

            windows_X.append(x)
            windows_y.append(y)
            windows_station.append(station_idx)
            windows_split.append(target_split)

    if not windows_X:
        print("  [ERROR] No valid sliding windows were constructed.")
        return None

    X = np.array(windows_X, dtype=np.float32)
    y = np.array(windows_y, dtype=np.float32)
    S = np.array(windows_station, dtype=np.int64)
    splits = np.array(windows_split)

    print(f"\n  Sliding windows: {len(X):,} total")
    print(f"  Boundary-dropped (target crosses split): {n_boundary_dropped}")
    for sp in ['train', 'val', 'test']:
        print(f"    {sp}: {(splits == sp).sum():,}")

    # ── Ablation: build feature index without weather ──────────────
    weather_indices = [feature_cols.index(w) for w in weather_cols
                       if w in feature_cols]
    non_weather_indices = [i for i in range(n_features)
                           if i not in weather_indices]

    result = {
        'X': X, 'y': y, 'S': S, 'splits': splits,
        'feature_cols': feature_cols,
        'weather_cols': weather_cols,
        'weather_indices': weather_indices,
        'non_weather_indices': non_weather_indices,
        'n_features': n_features,
        'n_stations': n_stations,
        'stations': stations,
        'station_to_idx': station_to_idx,
        'scaler_params': scaler_params,
        'arrivals_idx_in_features': arrivals_idx_in_features,
        'lookback': LOOKBACK,
        'horizon': HORIZON,
    }

    return result


# ── 1b. PyTorch Dataset + Model ─────────────────────────────────────

if HAS_TORCH:

    class ArrivalDataset(Dataset):
        """PyTorch dataset for LSTM arrival forecasting."""

        def __init__(self, X, y, S):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
            self.S = torch.from_numpy(S)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.S[idx]

    class ArrivalLSTM(nn.Module):
        """
        2-layer LSTM with station embedding for hourly arrival forecasting.

        Architecture (locked):
          - Station embedding: dim=8, 13 stations
          - Input: (batch, 168, F + 8)  [features + station embedding tiled]
          - LSTM: 2 layers, hidden_size, dropout=0.2
          - Output head: Linear(hidden_size, 24)  [direct multi-output]
          - Forward pass produces (batch, 24) predictions for hours t+1..t+24
        """

        def __init__(self, n_features: int, n_stations: int,
                     hidden_size: int = 64, n_layers: int = 2,
                     embed_dim: int = 8, dropout: float = 0.2):
            super().__init__()
            self.station_embed = nn.Embedding(n_stations, embed_dim)
            self.lstm = nn.LSTM(
                input_size=n_features + embed_dim,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 24)

        def forward(self, x, station_idx):
            """
            x: (batch, 168, F)
            station_idx: (batch,) long tensor
            Returns: (batch, 24) predicted arrival counts
            """
            B, T, F = x.shape
            emb = self.station_embed(station_idx)    # (B, embed_dim)
            emb = emb.unsqueeze(1).expand(B, T, -1)  # (B, T, embed_dim)
            x_cat = torch.cat([x, emb], dim=-1)       # (B, T, F + embed_dim)
            out, _ = self.lstm(x_cat)                  # (B, T, hidden)
            last = out[:, -1, :]                       # (B, hidden) — last timestep
            pred = self.head(last)                     # (B, 24)
            return pred


# ── 1c. Training loop ───────────────────────────────────────────────

def train_lstm(data: dict, output_dir: Path,
               hidden_size: int = 64, lr: float = 1e-3,
               batch_size: int = 128, max_epochs: int = 80,
               patience: int = 10, use_weather: bool = True,
               label: str = 'full', device_preference: str = 'cuda') -> Optional[dict]:
    """
    Train LSTM on the prepared data.

    Returns dict with model, metrics, and training history.
    """
    if not HAS_TORCH:
        return None

    # Reproducibility seeds
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cuda_available = torch.cuda.is_available()
    cuda_build = getattr(torch.version, 'cuda', None)
    if device_preference not in {'cuda', 'cpu', 'auto'}:
        raise ValueError(f"Unsupported device preference: {device_preference}")
    if device_preference == 'cuda' and not cuda_available:
        print("  [ERROR] CUDA was requested but is unavailable. "
              f"Installed PyTorch={torch.__version__}, CUDA build={cuda_build}.")
        print("          Install a CUDA-enabled PyTorch build for the GTX 1650, then rerun.")
        return None
    use_cuda = cuda_available and device_preference != 'cpu'
    if device_preference == 'auto' and not cuda_available:
        print("  [WARN] CUDA is unavailable; using the explicitly permitted auto CPU fallback. "
              f"Installed PyTorch={torch.__version__}, CUDA build={cuda_build}.")
    if use_cuda:
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda
    print(f"\n  Training LSTM ({label}) on {device}...")
    if use_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(device)} | "
              f"CUDA build: {cuda_build}")

    # Select features
    if use_weather:
        X_data = data['X']
        n_feat = data['n_features']
    else:
        indices = data['non_weather_indices']
        X_data = data['X'][:, :, indices]
        n_feat = len(indices)

    splits = data['splits']

    # Build datasets
    train_mask = splits == 'train'
    val_mask = splits == 'val'
    test_mask = splits == 'test'

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        print("  [ERROR] LSTM split construction failed:"
              f" train={int(train_mask.sum())},"
              f" val={int(val_mask.sum())},"
              f" test={int(test_mask.sum())}")
        return None

    train_ds = ArrivalDataset(X_data[train_mask], data['y'][train_mask],
                               data['S'][train_mask])
    val_ds = ArrivalDataset(X_data[val_mask], data['y'][val_mask],
                             data['S'][val_mask])
    test_ds = ArrivalDataset(X_data[test_mask], data['y'][test_mask],
                              data['S'][test_mask])

    loader_kwargs = {
        'num_workers': 0,
        'pin_memory': use_cuda,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, **loader_kwargs)
    eval_batch_size = min(batch_size * 2, 256)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                             **loader_kwargs)

    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")
    print(f"  Test samples:  {len(test_ds):,}")
    print(f"  Features:      {n_feat} ({'with' if use_weather else 'without'} weather)")
    print(f"  AMP enabled:   {use_amp}")
    print(f"  DataLoader:    num_workers={loader_kwargs['num_workers']}, "
          f"pin_memory={loader_kwargs['pin_memory']}")

    # Model
    model = ArrivalLSTM(
        n_features=n_feat,
        n_stations=data['n_stations'],
        hidden_size=hidden_size,
        n_layers=2,
        embed_dim=8,
        dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    train_losses = []
    val_losses = []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.time()
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            sb = sb.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=use_amp):
                pred = model(xb, sb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                sb = sb.to(device, non_blocking=use_cuda)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=use_amp):
                    pred = model(xb, sb)
                    val_loss += loss_fn(pred, yb).item()
                n_val += 1
        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_time = time.time() - epoch_t0
        elapsed = time.time() - t0
        avg_epoch_time = elapsed / epoch
        eta = avg_epoch_time * max(max_epochs - epoch, 0)
        print(f"    Epoch {epoch:3d}/{max_epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, best_epoch={best_epoch}, "
              f"no_improve={epoch - best_epoch}, "
              f"lr={optimizer.param_groups[0]['lr']:.1e}, "
              f"epoch_time={epoch_time:.1f}s, elapsed={elapsed/60:.1f}m, "
              f"eta={eta/60:.1f}m")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"    Early stopping at epoch {epoch} (best={best_epoch})")
            break

    elapsed = time.time() - t0
    print(f"  Training complete: {elapsed:.1f}s, best_epoch={best_epoch}")

    # Load best model
    if best_state is None:
        print("  [ERROR] No valid LSTM checkpoint was captured during training.")
        return None
    model.load_state_dict(best_state)
    model.eval()

    # ── Test evaluation ────────────────────────────────────────────
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb, sb in test_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            sb = sb.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=use_amp):
                pred = model(xb, sb)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(yb.numpy())

    if not all_preds:
        print("  [ERROR] No test predictions were generated.")
        return None

    preds = np.concatenate(all_preds)   # (N_test, 24)
    targets = np.concatenate(all_targets)

    # ── Per-horizon metrics ────────────────────────────────────────
    # IMPORTANT: Week 4 XGBoost/baselines are single-step (horizon-1)
    # forecasters. To make a fair comparison, we report:
    #   (a) horizon-1 metrics (column 0 of the 24-output) — comparable to XGBoost
    #   (b) all-horizon metrics (flattened) — LSTM-specific, NOT comparable to XGBoost
    #   (c) per-horizon breakdown for analysis
    per_horizon_mae = []
    per_horizon_rmse = []
    for h in range(24):
        h_err = np.abs(preds[:, h] - targets[:, h])
        per_horizon_mae.append(float(h_err.mean()))
        per_horizon_rmse.append(float(np.sqrt(np.mean((preds[:, h] - targets[:, h])**2))))

    # Horizon-1 (comparable to XGBoost single-step)
    h1_mae = per_horizon_mae[0]
    h1_rmse = per_horizon_rmse[0]
    h1_preds = preds[:, 0]
    h1_targets = targets[:, 0]
    h1_ss_res = np.sum((h1_targets - h1_preds) ** 2)
    h1_ss_tot = np.sum((h1_targets - h1_targets.mean()) ** 2)
    h1_r2 = float(1 - h1_ss_res / h1_ss_tot) if h1_ss_tot > 0 else 0.0

    # All-horizon (LSTM-specific, not directly comparable to XGBoost)
    p_flat = preds.flatten()
    t_flat = targets.flatten()
    all_mae = float(np.mean(np.abs(p_flat - t_flat)))
    all_rmse = float(np.sqrt(np.mean((p_flat - t_flat) ** 2)))
    all_ss_res = np.sum((t_flat - p_flat) ** 2)
    all_ss_tot = np.sum((t_flat - t_flat.mean()) ** 2)
    all_r2 = float(1 - all_ss_res / all_ss_tot) if all_ss_tot > 0 else 0.0

    print(f"\n  Test metrics ({label}):")
    print(f"    Horizon-1 (XGBoost-comparable): MAE={h1_mae:.4f}, "
          f"RMSE={h1_rmse:.4f}, R2={h1_r2:.4f}")
    print(f"    All-horizon (24h avg):          MAE={all_mae:.4f}, "
          f"RMSE={all_rmse:.4f}, R2={all_r2:.4f}")

    # Save model
    if label == 'full':
        model_path = output_dir / 'lstm_model.pt'
        torch.save({
            'model_state_dict': best_state,
            'n_features': n_feat,
            'n_stations': data['n_stations'],
            'hidden_size': hidden_size,
            'embed_dim': 8,
        }, model_path)
        print(f"  [SAVED] {model_path}")

    result = {
        'label': label,
        'use_weather': use_weather,
        'n_features': n_feat,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'train_time_s': elapsed,
        # Horizon-1 metrics (comparable to XGBoost single-step)
        'test_mae_h1': h1_mae,
        'test_rmse_h1': h1_rmse,
        'test_r2_h1': h1_r2,
        # All-horizon metrics (LSTM-specific, NOT comparable to XGBoost)
        'test_mae_all': all_mae,
        'test_rmse_all': all_rmse,
        'test_r2_all': all_r2,
        # Per-horizon breakdown
        'per_horizon_mae': per_horizon_mae,
        'per_horizon_rmse': per_horizon_rmse,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'preds': preds,
        'targets': targets,
    }

    return result


# ── 1d. Comparison + plots ──────────────────────────────────────────

def build_comparison_table(lstm_full: Optional[dict],
                           lstm_no_weather: Optional[dict],
                           xgb_results: Optional[dict],
                           output_dir: Path) -> dict:
    """
    Build the 4-model comparison table:
      Historical Mean, Last-Week-Same-Hour, XGBoost, LSTM.

    Also computes fleet sizing relevance.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)

    comparison = {'models': {}}

    # XGBoost and baselines from Week 4 (single-step hourly forecasters)
    if xgb_results:
        for model_name in ['last_week_same_hour', 'historical_mean', 'xgboost']:
            if model_name in xgb_results.get('models', {}):
                comparison['models'][model_name] = xgb_results['models'][model_name]

    # LSTM: two metric sets
    #   - horizon-1 (h1): comparable to XGBoost/baselines (single-step)
    #   - all-horizon: LSTM-specific 24h forecast quality (NOT comparable)
    if lstm_full:
        comparison['models']['lstm_h1'] = {
            'mae': lstm_full['test_mae_h1'],
            'rmse': lstm_full['test_rmse_h1'],
            'r2': lstm_full['test_r2_h1'],
            'n_params': lstm_full['n_params'],
            'best_epoch': lstm_full['best_epoch'],
            'train_time_s': lstm_full['train_time_s'],
            'note': 'Horizon-1 only; comparable to XGBoost single-step.',
        }
        comparison['lstm_all_horizon'] = {
            'mae': lstm_full['test_mae_all'],
            'rmse': lstm_full['test_rmse_all'],
            'r2': lstm_full['test_r2_all'],
            'per_horizon_mae': lstm_full['per_horizon_mae'],
            'note': 'Average across all 24 forecast horizons. NOT directly '
                    'comparable to XGBoost single-step metrics.',
        }

    # LSTM no weather (ablation) — also horizon-1 for consistency
    if lstm_no_weather:
        comparison['models']['lstm_no_weather_h1'] = {
            'mae': lstm_no_weather['test_mae_h1'],
            'rmse': lstm_no_weather['test_rmse_h1'],
            'r2': lstm_no_weather['test_r2_h1'],
        }

    # ── Comparative analysis (horizon-1 only for fair comparison) ──
    xgb_mae = comparison['models'].get('xgboost', {}).get('mae')
    lstm_h1_mae = comparison['models'].get('lstm_h1', {}).get('mae')

    if xgb_mae and lstm_h1_mae:
        improvement = (xgb_mae - lstm_h1_mae) / xgb_mae * 100
        comparison['lstm_vs_xgboost'] = {
            'comparison_basis': 'horizon-1 (single-step, apples-to-apples)',
            'xgb_mae': xgb_mae,
            'lstm_h1_mae': lstm_h1_mae,
            'lstm_improvement_pct': improvement,
            'lstm_wins': lstm_h1_mae < xgb_mae,
            'primary_model': 'lstm' if lstm_h1_mae < xgb_mae else 'xgboost',
            'note': (
                f'LSTM horizon-1 {"outperforms" if lstm_h1_mae < xgb_mae else "underperforms"} '
                f'XGBoost by {abs(improvement):.1f}% MAE on the Sep-Dec 2021 test set '
                f'(single-step comparison).'
            ),
        }
        print(f"\n  LSTM vs XGBoost (h1): {comparison['lstm_vs_xgboost']['note']}")

        # Fleet sizing relevance — derived from fair h1 comparison only
        comparison['fleet_sizing_impact'] = 'not_evaluated'
        comparison['fleet_sizing_impact_note'] = (
            'The model comparison measures forecast accuracy only. '
            'Operational impact requires forecast-fed holdout simulation.')
        print("  Fleet sizing impact: not evaluated by model MAE alone")

    # Weather ablation (horizon-1)
    lstm_nw_h1_mae = comparison['models'].get('lstm_no_weather_h1', {}).get('mae')
    if lstm_h1_mae and lstm_nw_h1_mae:
        weather_delta = (lstm_nw_h1_mae - lstm_h1_mae) / lstm_h1_mae * 100
        comparison['weather_ablation'] = {
            'full_h1_mae': lstm_h1_mae,
            'no_weather_h1_mae': lstm_nw_h1_mae,
            'weather_contribution_pct': weather_delta,
            'note': (
                f'Removing weather features changes h1 MAE by {weather_delta:+.2f}%. '
                f'Weather is {"meaningful" if abs(weather_delta) > 1 else "negligible"}.'
            ),
        }
        print(f"  Weather ablation: {comparison['weather_ablation']['note']}")

    # Print comparison table (horizon-1 section)
    print(f"\n  Fair comparison (all single-step / horizon-1):")
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for mname in ['historical_mean', 'last_week_same_hour', 'xgboost', 'lstm_h1']:
        mvals = comparison['models'].get(mname, {})
        if not mvals:
            continue
        mae_s = f"{mvals.get('mae', 0):.4f}" if 'mae' in mvals else 'N/A'
        rmse_s = f"{mvals.get('rmse', 0):.4f}" if 'rmse' in mvals else 'N/A'
        r2_s = f"{mvals.get('r2', 0):.4f}" if 'r2' in mvals else 'N/A'
        print(f"  {mname:<25} {mae_s:>8} {rmse_s:>8} {r2_s:>8}")

    if 'lstm_all_horizon' in comparison:
        ah = comparison['lstm_all_horizon']
        print(f"\n  LSTM all-horizon (24h avg, NOT comparable to above):")
        print(f"    MAE={ah['mae']:.4f}, RMSE={ah['rmse']:.4f}, R2={ah['r2']:.4f}")

    # Save
    comp_path = output_dir / 'lstm_comparison.json'
    # Remove numpy arrays before saving
    saveable = {k: v for k, v in comparison.items()
                if not isinstance(v, np.ndarray)}
    with open(comp_path, 'w') as f:
        json.dump(to_builtin(saveable), f, indent=2)
    print(f"\n  [SAVED] {comp_path}")

    return comparison


def plot_lstm_results(lstm_full: Optional[dict], lstm_no_weather: Optional[dict],
                      comparison: dict, output_dir: Path):
    """Generate LSTM diagnostic plots."""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    if lstm_full is None:
        print("  [SKIP] No LSTM results to plot.")
        return

    # ── Plot 1: Training curves ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(lstm_full['train_losses'], label='Train', alpha=0.7)
    ax1.plot(lstm_full['val_losses'], label='Validation', alpha=0.7)
    ax1.axvline(lstm_full['best_epoch'] - 1, color='gray', linestyle='--',
                alpha=0.5, label=f"Best epoch ({lstm_full['best_epoch']})")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('LSTM Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Model comparison bar chart ─────────────────────────
    models_to_plot = []
    labels = []
    maes = []
    colors_list = []
    color_map = {
        'historical_mean': '#95a5a6',
        'last_week_same_hour': '#3498db',
        'xgboost': '#e74c3c',
        'lstm_h1': '#2ecc71',
        'lstm_no_weather_h1': '#f39c12',
    }
    label_map = {
        'historical_mean': 'Hist. Mean',
        'last_week_same_hour': 'LWSH',
        'xgboost': 'XGBoost',
        'lstm_h1': 'LSTM (h1)',
        'lstm_no_weather_h1': 'LSTM\n(no weather, h1)',
    }

    for mname in ['historical_mean', 'last_week_same_hour', 'xgboost',
                   'lstm_h1', 'lstm_no_weather_h1']:
        if mname in comparison.get('models', {}):
            m = comparison['models'][mname]
            if 'mae' in m:
                labels.append(label_map.get(mname, mname))
                maes.append(m['mae'])
                colors_list.append(color_map.get(mname, 'gray'))

    if maes:
        bars = ax2.bar(labels, maes, color=colors_list, alpha=0.8)
        for bar, val in zip(bars, maes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.set_ylabel('MAE')
        ax2.set_title('Test MAE Comparison')
        ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig_path = fig_dir / 'lstm_comparison.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {fig_path}")

    # ── Plot 3: Predicted vs actual for 2 sample weeks ─────────────
    preds = lstm_full['preds']
    targets = lstm_full['targets']

    if len(preds) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        # Pick two sample windows: one early, one late in test set
        n_windows = len(preds)
        sample_indices = [0, n_windows // 2]

        for ax_idx, w_idx in enumerate(sample_indices):
            ax = axes[ax_idx]
            hours = np.arange(24)
            ax.plot(hours, targets[w_idx], 'b-o', markersize=3, label='Actual')
            ax.plot(hours, preds[w_idx], 'r-s', markersize=3, label='Predicted')
            ax.set_xlabel('Forecast Horizon (hours)')
            ax.set_ylabel('Arrivals')
            ax.set_title(f'Sample Forecast Window {ax_idx + 1} (index {w_idx})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig_path = fig_dir / 'lstm_sample_forecasts.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  [SAVED] {fig_path}")


# ============================================================================
# COMPONENT 2: SimPy ENGINE SKELETON
# ============================================================================

PATCHED_ENGINE_MARKERS = [
    'fault_repair_median_min',
    'activation_schedule',
    'capacity_event_log',
    'fault_repair_duration',
    'SeedSequence',
    'completed_successfully',
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _is_patched_sim_engine(source: str) -> bool:
    """Return True only for the Week 6+ patched simulation engine."""
    return all(marker in source for marker in PATCHED_ENGINE_MARKERS)


def write_sim_engine(output_dir: Path, param_summary: dict,
                     svc_summary: dict, nhpp_rates: Optional[pd.DataFrame]):
    """
    Preserve/copy the patched standalone SimPy engine as sim_engine.py.

    Week 6 introduced critical fixes to the engine: repair-duration faults,
    demand-preserving requeue, activation schedules, and realized-capacity
    metrics. Rerunning Week 5 must not regenerate the older embedded template.
    """
    print("\n" + "=" * 70)
    print("COMPONENT 2: PRESERVING PATCHED SimPy ENGINE (sim_engine.py)")
    print("=" * 70)

    canonical_path = SIM_ENGINE_PATH
    target_path = output_dir / 'sim_engine.py'

    source_path = canonical_path if canonical_path.exists() else target_path
    if not source_path.exists():
        raise FileNotFoundError(
            "Patched sim_engine.py not found. Expected "
            f"{canonical_path} or existing target {target_path}."
        )

    engine_source = source_path.read_text(encoding='utf-8')
    if not _is_patched_sim_engine(engine_source):
        missing = [
            marker for marker in PATCHED_ENGINE_MARKERS
            if marker not in engine_source
        ]
        raise RuntimeError(
            "Refusing to write stale sim_engine.py; patched markers missing: "
            + ', '.join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if target_path.resolve() != source_path.resolve():
        target_path.write_text(engine_source, encoding='utf-8')
        action = "COPIED"
    else:
        action = "UNCHANGED"

    print(f"  [{action}] {target_path}")
    print(f"  Source: {source_path}")
    print(f"  SHA256: {_sha256_text(engine_source)}")

    if HAS_SIMPY:
        print("\n  Running M/M/s analytical validation...")
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "sim_engine", str(target_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {target_path}")
            sim_mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sim_mod
            spec.loader.exec_module(sim_mod)
            val_result = sim_mod.validate_mms(
                s=5, lam_per_hour=3.0, mean_service_min=60.0,
                n_reps=100, sim_days=30)
            val_path = output_dir / 'mms_validation.json'
            with open(val_path, 'w') as f:
                json.dump(val_result, f, indent=2)
            print(f"  [SAVED] {val_path}")
        except Exception as e:
            print(f"  [WARN] Validation failed: {e}")
    else:
        print("  [SKIP] SimPy not installed; validation deferred to Week 6.")

    return target_path

    # Extract the 4 representative stations from parameter_summary
    rep_stations = param_summary.get('fleet_sizing_scope', {}).get(
        'representative_stations', [])

    # Extract service time parameters for the engine's docstring
    dc_params = param_summary.get('service_time', {}).get('DC_Fast', {}).get(
        'fit_params', {})
    l2_params = param_summary.get('service_time', {}).get('Level_2', {}).get(
        'fit_params', {})
    mx_params = param_summary.get('service_time', {}).get('Mixed', {}).get(
        'fit_params', {})

    engine_code = r'''"""
SimPy Discrete-Event Simulation Engine for EV Charging Stations
================================================================
Generated by week5_analysis.py — do NOT edit parameter values here.
They are loaded from parameter_summary.json and nhpp_rate_functions.csv.

Architecture:
  - ArrivalGenerator: Poisson(constant-λ), NHPP(piecewise-constant),
    or ML-replay (piecewise-homogeneous Poisson with ML-estimated
    hourly means — does NOT discover sub-hourly arrival structure).
  - ServiceProcess: samples from fitted distribution per charger type.
  - FaultInjector: session-start Bernoulli by charger type.
    Mid-session failure modeling deferred unless empirical records
    clearly require it.
  - SchedulingPolicy: FCFS (interface accepts pluggable policy objects
    for greedy/LP schedulers in Weeks 8–9).
  - MetricsCollector: per-session and per-charger metrics.

Usage:
    from sim_engine import simulate_station, load_config

    config = load_config('parameter_summary.json',
                         'nhpp_rate_functions.csv',
                         station='Nanhu_Technology Park')
    results = simulate_station(config, n_chargers=10, sim_days=30,
                               n_replications=50)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field

try:
    import simpy
    HAS_SIMPY = True
except ImportError:
    HAS_SIMPY = False

from scipy.stats import gamma as gamma_dist
from scipy.stats import weibull_min as weibull_dist


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StationConfig:
    """Configuration for a single station simulation."""
    station_name: str
    n_chargers: int

    # Arrival parameters
    arrival_mode: str = 'nhpp'  # 'poisson', 'nhpp', 'ml_replay'
    nhpp_rates: Optional[np.ndarray] = None  # (24,) hourly rates
    constant_lambda: float = 1.0  # for Poisson mode
    ml_hourly_predictions: Optional[np.ndarray] = None  # (24*sim_days,) for replay

    # Service time parameters (loaded from parameter_summary.json)
    service_params: Optional[dict] = None
    charger_type_mix: Optional[dict] = None  # e.g. {'DC_Fast': 0.6, 'Level_2': 0.3, 'Mixed': 0.1}

    # Fault parameters
    faults_enabled: bool = True
    fault_rates: Optional[dict] = None  # per charger type

    # Scheduling
    scheduling_policy: str = 'fcfs'

    # Activation control: how many chargers active per hour (24-vector)
    # None = all active
    activation_schedule: Optional[np.ndarray] = None

    # Simulation
    sim_days: int = 30
    random_seed: Optional[int] = None


# ============================================================================
# SERVICE TIME SAMPLING
# ============================================================================

def sample_service_time(charger_type: str, service_params: dict,
                        rng: np.random.Generator) -> float:
    """
    Draw a service duration (minutes) from the fitted distribution
    for the given charger type.

    Parameters come from parameter_summary.json -> service_time -> {type} -> fit_params.
    """
    if charger_type not in service_params:
        # Fallback: exponential with type mean
        means = {'DC_Fast': 34.9, 'Level_2': 96.5, 'Mixed': 44.1}
        return rng.exponential(means.get(charger_type, 50.0))

    params = service_params[charger_type]
    dist_name = params.get('best_distribution', 'exponential')
    fp = params.get('fit_params', {})

    if dist_name == 'gamma':
        # scipy gamma: shape=a, loc=0, scale=scale
        a = fp.get('a', 1.0)
        scale = fp.get('scale', params.get('mean_min', 35.0))
        return float(gamma_dist.rvs(a, loc=0, scale=scale, random_state=rng))

    elif dist_name == 'weibull':
        # scipy weibull_min: c=shape, loc=0, scale=scale
        c = fp.get('c', 1.0)
        scale = fp.get('scale', params.get('mean_min', 44.0))
        return float(weibull_dist.rvs(c, loc=0, scale=scale, random_state=rng))

    elif dist_name == 'mixture_2lognorm':
        # 2-component lognormal mixture
        pi1 = fp.get('pi1', 0.5)
        mu1, sigma1 = fp.get('mu1', 4.0), fp.get('sigma1', 1.0)
        mu2, sigma2 = fp.get('mu2', 4.0), fp.get('sigma2', 0.5)
        if rng.random() < pi1:
            return float(rng.lognormal(mu1, sigma1))
        else:
            return float(rng.lognormal(mu2, sigma2))

    elif dist_name == 'lognormal':
        s = fp.get('s', 1.0)
        scale = fp.get('scale', params.get('mean_min', 50.0))
        # scipy lognormal: s=sigma, scale=exp(mu)
        return float(rng.lognormal(np.log(scale), s))

    else:
        # Exponential fallback
        mean = params.get('mean_min', 50.0)
        return float(rng.exponential(mean))


def choose_charger_type(charger_type_mix: Optional[dict],
                        rng: np.random.Generator) -> str:
    """Draw a charger type for this session from the station's type mix."""
    if not charger_type_mix:
        return 'DC_Fast'  # default
    types = list(charger_type_mix.keys())
    probs = np.array([charger_type_mix[t] for t in types], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(types, p=probs)


# ============================================================================
# FAULT INJECTION
# ============================================================================

# Default per-type fault rates.
# Source: Jiaxing dataset EDA (Week 2), computed as fraction of sessions
# with is_abnormal=1 within each charger_type group:
#   DC_Fast: 9,451 / 135,359 = 6.98% ~ 7%
#   Level_2: 40,730 / 131,329 = 31.02% ~ 31%
#   Mixed:   6,577 / 82,221  = 8.00% ~ 8%
# These are dataset-level averages; per-station rates vary.
DEFAULT_FAULT_RATES = {
    'DC_Fast': 0.07,
    'Level_2': 0.31,
    'Mixed': 0.08,
}


def is_session_faulted(charger_type: str, fault_rates: Optional[dict],
                       rng: np.random.Generator) -> bool:
    """
    Determine if a session is faulted (session-start Bernoulli).

    Mid-session failure modeling is deferred unless empirical records
    clearly require it.
    """
    rates = fault_rates or DEFAULT_FAULT_RATES
    rate = rates.get(charger_type, 0.186)  # global fallback
    return rng.random() < rate


def fault_duration(rng: np.random.Generator) -> float:
    """
    Duration (minutes) a faulted session occupies the charger.

    Based on dataset: zero-energy faults have median ~1.3 min.
    We use a lognormal(0.0, 0.8) to give median ~1.0 min, mean ~1.4 min.
    """
    return max(0.5, float(rng.lognormal(0.0, 0.8)))


# ============================================================================
# ARRIVAL GENERATORS
# ============================================================================

def nhpp_thinning_arrivals(env, nhpp_rates_24: np.ndarray,
                           rng: np.random.Generator):
    """
    Generate NHPP arrivals using the thinning (acceptance-rejection) algorithm.

    nhpp_rates_24: (24,) array of hourly arrival rates λ(h).
    The rate function is piecewise-constant within each hour.

    Algorithm:
      1. λ_max = max(nhpp_rates_24)
      2. Generate candidate inter-arrival times ~ Exp(λ_max)
      3. Accept each candidate with probability λ(current_hour) / λ_max

    Yields arrival times as a SimPy generator.
    """
    lambda_max = nhpp_rates_24.max()
    if lambda_max <= 0:
        return  # no arrivals

    while True:
        # Candidate inter-arrival time
        iat = rng.exponential(1.0 / lambda_max) * 60  # convert to minutes
        yield env.timeout(iat)

        # Current hour
        current_hour = int((env.now / 60) % 24)
        current_rate = nhpp_rates_24[current_hour]

        # Accept with probability λ(h) / λ_max
        if rng.random() < current_rate / lambda_max:
            yield current_hour  # signal: accepted arrival


def poisson_arrivals(env, rate_per_hour: float, rng: np.random.Generator):
    """Generate homogeneous Poisson arrivals."""
    if rate_per_hour <= 0:
        return
    mean_iat_min = 60.0 / rate_per_hour
    while True:
        iat = rng.exponential(mean_iat_min)
        yield env.timeout(iat)


def ml_replay_arrivals(env, hourly_predictions: np.ndarray,
                       rng: np.random.Generator):
    """
    ML-replay arrival mode.

    This is a piecewise-homogeneous Poisson simulator with ML-estimated
    hourly means. Within each simulated hour, arrivals are drawn from
    Poisson(λ_predicted) and distributed uniformly. This is a pragmatic
    way to inject demand forecasts into the simulator — it does NOT
    discover or reproduce sub-hourly arrival structure from the data.

    hourly_predictions: (n_hours,) predicted arrival counts per hour.
    """
    for hour_idx, lam in enumerate(hourly_predictions):
        hour_start = hour_idx * 60  # minutes
        if env.now > hour_start + 60:
            continue
        if env.now < hour_start:
            yield env.timeout(hour_start - env.now)

        # Draw number of arrivals this hour from Poisson(λ)
        n_arrivals = rng.poisson(max(0, lam))
        if n_arrivals > 0:
            # Distribute uniformly within the hour
            offsets = np.sort(rng.uniform(0, 60, size=n_arrivals))
            for offset in offsets:
                target_time = hour_start + offset
                if target_time > env.now:
                    yield env.timeout(target_time - env.now)
                yield hour_idx  # signal: arrival


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

@dataclass
class SessionRecord:
    """Record for one charging session."""
    session_id: int = 0
    arrival_time: float = 0.0
    service_start_time: float = 0.0
    wait_time: float = 0.0
    service_duration: float = 0.0
    departure_time: float = 0.0
    charger_type: str = ''
    is_fault: bool = False
    energy_proxy: float = 0.0  # placeholder


@dataclass
class MetricsCollector:
    """Collects simulation metrics."""
    sessions: List[SessionRecord] = field(default_factory=list)
    charger_busy_time: Optional[np.ndarray] = None
    charger_fault_time: Optional[np.ndarray] = None
    sim_duration_min: float = 0.0

    def add_session(self, record: SessionRecord):
        self.sessions.append(record)

    def compute_summary(self, n_chargers: int, sim_duration_min: float) -> dict:
        """
        Compute aggregate metrics from session records.

        sim_duration_min is the MEASUREMENT WINDOW (arrival horizon).
        Sessions that arrived before the cutoff but completed after it
        are included in wait/throughput stats, but their busy time is
        clipped to the measurement window for utilization calculation.
        """
        self.sim_duration_min = sim_duration_min

        _EMPTY = {
            'n_sessions': 0,
            'n_faults': 0,
            'fault_fraction': 0.0,
            'mean_wait': 0.0,
            'median_wait': 0.0,
            'p95_wait': 0.0,
            'p_wait_gt_10min': 0.0,
            'p_wait_gt_15min': 0.0,
            'mean_utilization': 0.0,
            'throughput_per_hour': 0.0,
            'mean_service_duration': 0.0,
            'n_drained_after_horizon': 0,
        }

        if not self.sessions:
            return dict(_EMPTY)

        # Only sessions whose arrival_time < sim_duration_min
        in_window = [s for s in self.sessions
                     if s.arrival_time < sim_duration_min]
        if not in_window:
            return dict(_EMPTY)

        waits = np.array([s.wait_time for s in in_window])
        faults = np.array([s.is_fault for s in in_window])

        # Clip busy time to measurement window for utilization.
        # A session that started service at t and lasted d minutes
        # contributes min(d, sim_duration_min - t) to busy time
        # (if service_start_time < sim_duration_min).
        clipped_busy = 0.0
        for s in in_window:
            if s.service_start_time < sim_duration_min:
                effective_end = min(s.departure_time, sim_duration_min)
                clipped_busy += max(0, effective_end - s.service_start_time)
        total_capacity = n_chargers * sim_duration_min

        return {
            'n_sessions': len(in_window),
            'n_faults': int(faults.sum()),
            'fault_fraction': float(faults.mean()),
            'mean_wait': float(waits.mean()),
            'median_wait': float(np.median(waits)),
            'p95_wait': float(np.percentile(waits, 95)) if len(waits) > 0 else 0,
            'p_wait_gt_10min': float((waits > 10).mean()),
            'p_wait_gt_15min': float((waits > 15).mean()),
            'mean_utilization': float(clipped_busy / total_capacity)
                                if total_capacity > 0 else 0,
            'throughput_per_hour': float(len(in_window) /
                                        (sim_duration_min / 60))
                                  if sim_duration_min > 0 else 0,
            'mean_service_duration': float(
                np.mean([s.service_duration for s in in_window])),
            'n_drained_after_horizon': sum(
                1 for s in in_window
                if s.departure_time > sim_duration_min),
        }


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

def simulate_station(config: StationConfig,
                     n_replications: int = 50,
                     verbose: bool = False) -> List[dict]:
    """
    Run discrete-event simulation for one station configuration.

    Returns a list of summary dicts, one per replication.
    """
    if not HAS_SIMPY:
        raise ImportError("SimPy is required. Install with: pip install simpy")

    results = []
    sim_duration_min = config.sim_days * 24 * 60  # total simulation minutes

    for rep in range(n_replications):
        seed = (config.random_seed or 42) + rep
        rng = np.random.default_rng(seed)
        env = simpy.Environment()

        # Charger resource
        n_active = config.n_chargers  # TODO: activation_schedule per hour
        chargers = simpy.Resource(env, capacity=n_active)
        metrics = MetricsCollector()
        session_counter = [0]

        def customer_process(env, arrival_time, charger_type, config, metrics,
                             chargers, rng, session_id):
            """Process for a single customer session."""
            with chargers.request() as req:
                yield req
                service_start = env.now
                wait = service_start - arrival_time

                # Fault check
                fault_rates = config.fault_rates or DEFAULT_FAULT_RATES
                is_fault = (config.faults_enabled and
                            is_session_faulted(charger_type, fault_rates, rng))

                if is_fault:
                    duration = fault_duration(rng)
                else:
                    duration = sample_service_time(
                        charger_type, config.service_params or {}, rng)
                    # Clamp extreme values
                    duration = min(max(duration, 0.5), 720)

                yield env.timeout(duration)

                record = SessionRecord(
                    session_id=session_id,
                    arrival_time=arrival_time,
                    service_start_time=service_start,
                    wait_time=wait,
                    service_duration=duration,
                    departure_time=env.now,
                    charger_type=charger_type,
                    is_fault=is_fault,
                )
                metrics.add_session(record)

        def arrival_process(env, config, chargers, metrics, rng,
                            session_counter):
            """Main arrival process dispatcher."""
            if config.arrival_mode == 'nhpp' and config.nhpp_rates is not None:
                # NHPP thinning
                lambda_max = config.nhpp_rates.max()
                if lambda_max <= 0:
                    return
                while env.now < sim_duration_min:
                    iat = rng.exponential(1.0 / lambda_max) * 60
                    yield env.timeout(iat)
                    if env.now >= sim_duration_min:
                        break
                    current_hour = int((env.now / 60) % 24)
                    current_rate = config.nhpp_rates[current_hour]
                    if rng.random() < current_rate / lambda_max:
                        ctype = choose_charger_type(config.charger_type_mix, rng)
                        sid = session_counter[0]
                        session_counter[0] += 1
                        env.process(customer_process(
                            env, env.now, ctype, config, metrics,
                            chargers, rng, sid))

            elif config.arrival_mode == 'poisson':
                mean_iat_min = 60.0 / max(config.constant_lambda, 0.01)
                while env.now < sim_duration_min:
                    iat = rng.exponential(mean_iat_min)
                    yield env.timeout(iat)
                    if env.now >= sim_duration_min:
                        break
                    ctype = choose_charger_type(config.charger_type_mix, rng)
                    sid = session_counter[0]
                    session_counter[0] += 1
                    env.process(customer_process(
                        env, env.now, ctype, config, metrics,
                        chargers, rng, sid))

            elif config.arrival_mode == 'ml_replay' and \
                    config.ml_hourly_predictions is not None:
                # Piecewise-homogeneous Poisson with ML-estimated hourly means
                preds = config.ml_hourly_predictions
                for hour_idx in range(len(preds)):
                    hour_start = hour_idx * 60
                    if hour_start >= sim_duration_min:
                        break
                    lam = max(0, preds[hour_idx])
                    n_arr = rng.poisson(lam)
                    if n_arr > 0:
                        offsets = np.sort(rng.uniform(0, 60, size=n_arr))
                        for offset in offsets:
                            t = hour_start + offset
                            if t >= sim_duration_min:
                                break
                            if t > env.now:
                                yield env.timeout(t - env.now)
                            ctype = choose_charger_type(
                                config.charger_type_mix, rng)
                            sid = session_counter[0]
                            session_counter[0] += 1
                            env.process(customer_process(
                                env, env.now, ctype, config, metrics,
                                chargers, rng, sid))
                # Wait out remaining sim time
                if env.now < sim_duration_min:
                    yield env.timeout(sim_duration_min - env.now)

            else:
                print(f"[WARN] Unknown arrival mode: {config.arrival_mode}")
                return

        # Run simulation:
        #   1. Arrivals stop at sim_duration_min (the arrival_process loops
        #      already break when env.now >= sim_duration_min).
        #   2. After arrivals stop, let all in-progress and queued sessions
        #      complete — do NOT truncate with env.run(until=...).
        #   3. Compute metrics using sim_duration_min as the measurement
        #      window for utilization (busy time is clipped to the window).
        env.process(arrival_process(env, config, chargers, metrics, rng,
                                     session_counter))
        # Run WITHOUT a hard cutoff so queued/in-service customers drain.
        # Safety cap: drain for at most 2x the longest expected service time
        # (720 min) beyond the arrival horizon to avoid infinite hangs.
        drain_limit = sim_duration_min + 1440
        env.run(until=drain_limit)

        # Check if drain cap was hit (queue not fully drained)
        n_arrived = session_counter[0]
        n_completed = len(metrics.sessions)
        n_unfinished = n_arrived - n_completed
        drain_cap_hit = n_unfinished > 0

        summary = metrics.compute_summary(config.n_chargers, sim_duration_min)
        summary['replication'] = rep
        summary['seed'] = seed
        summary['n_unfinished_at_termination'] = n_unfinished
        summary['drain_cap_hit'] = drain_cap_hit
        results.append(summary)

        if verbose and rep % 10 == 0:
            drain_note = ' [DRAIN CAP HIT]' if drain_cap_hit else ''
            print(f"    Rep {rep}: {summary['n_sessions']} sessions, "
                  f"util={summary['mean_utilization']:.3f}, "
                  f"P(W>15)={summary['p_wait_gt_15min']:.3f}{drain_note}")

    return results


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(param_summary_path: str,
                nhpp_rates_path: str,
                station: str,
                n_chargers: int = 10,
                arrival_mode: str = 'nhpp',
                faults_enabled: bool = True,
                sim_days: int = 30,
                random_seed: int = 42) -> StationConfig:
    """
    Build a StationConfig from the saved parameter files.

    This is the standard way to set up a simulation run.
    """
    # Load parameter summary
    with open(param_summary_path, 'r') as f:
        params = json.load(f)

    # Load NHPP rates
    nhpp_df = pd.read_csv(nhpp_rates_path)
    station_rates = nhpp_df[nhpp_df['station'] == station].sort_values('hour')
    if len(station_rates) == 0:
        raise ValueError(f"Station '{station}' not found in NHPP rates. "
                         f"Available: {sorted(nhpp_df['station'].unique())}")
    nhpp_24 = station_rates['lambda_mean'].values  # (24,)

    # Service time params
    service_params = {}
    for ctype in ['DC_Fast', 'Level_2', 'Mixed']:
        if ctype in params.get('service_time', {}):
            service_params[ctype] = params['service_time'][ctype]

    # Charger type mix: rough heuristic from dominant_type metadata.
    # WARNING: This is NOT the empirical per-station charger mix from the
    # dataset. It is a coarse approximation based on which charger type
    # dominates the station. For accurate simulation, callers should pass
    # the empirical mix (from jiaxing_clean.parquet groupby station ×
    # charger_type) via the StationConfig.charger_type_mix field directly.
    charger_type_mix = {'DC_Fast': 0.4, 'Level_2': 0.4, 'Mixed': 0.2}
    # Try to read dominant_type from fleet_sizing
    for key, val in params.get('fleet_sizing', {}).items():
        if key.startswith(station + '__'):
            dom = val.get('dominant_type', '')
            if dom == 'DC_Fast':
                charger_type_mix = {'DC_Fast': 0.7, 'Level_2': 0.2, 'Mixed': 0.1}
            elif dom == 'Level_2':
                charger_type_mix = {'DC_Fast': 0.1, 'Level_2': 0.7, 'Mixed': 0.2}
            break

    config = StationConfig(
        station_name=station,
        n_chargers=n_chargers,
        arrival_mode=arrival_mode,
        nhpp_rates=nhpp_24,
        service_params=service_params,
        charger_type_mix=charger_type_mix,
        faults_enabled=faults_enabled,
        sim_days=sim_days,
        random_seed=random_seed,
    )

    return config


# ============================================================================
# ANALYTICAL COMPARISON (Erlang-C)
# ============================================================================

def erlang_c_pwait(s: int, lam: float, mu: float) -> float:
    """
    Compute Erlang-C P(wait) for M/M/s queue.
    Uses log-space to avoid overflow.

    s: number of servers
    lam: arrival rate (per unit time)
    mu: service rate (per unit time, per server)
    rho = lam / (s * mu) must be < 1 for stability
    """
    rho = lam / (s * mu)
    if rho >= 1.0:
        return 1.0

    a = lam / mu  # offered load

    # Log of (a^s / s!) * 1/(1-rho)
    log_numerator = s * np.log(a) - sum(np.log(k) for k in range(1, s+1)) \
                    - np.log(1 - rho)

    # Log of sum_{k=0}^{s-1} a^k / k!
    log_terms = []
    for k in range(s):
        log_term = k * np.log(a) - sum(np.log(j) for j in range(1, k+1))
        log_terms.append(log_term)

    log_max = max(log_terms + [log_numerator])
    sum_exp = sum(np.exp(t - log_max) for t in log_terms) + \
              np.exp(log_numerator - log_max)

    log_C = log_numerator - log_max - np.log(sum_exp)
    return float(np.exp(log_C))


# ============================================================================
# QUICK VALIDATION HELPER
# ============================================================================

def validate_mms(s: int = 5, lam_per_hour: float = 3.0,
                 mean_service_min: float = 60.0,
                 n_reps: int = 200, sim_days: int = 30,
                 verbose: bool = True) -> dict:
    """
    M/M/s validation: compare simulator against Erlang-C.

    Sets up a clean Poisson/Exponential scenario (no faults, no NHPP)
    and checks that simulated P(wait>0) matches the Erlang-C formula
    within ~5%.
    """
    if not HAS_SIMPY:
        print("[SKIP] SimPy not installed.")
        return {}

    mu_per_min = 1.0 / mean_service_min
    mu_per_hour = 60.0 / mean_service_min
    rho = lam_per_hour / (s * mu_per_hour)

    if verbose:
        print(f"\n  M/M/s validation: s={s}, λ={lam_per_hour}/hr, "
              f"μ={mu_per_hour:.3f}/hr, ρ={rho:.3f}")

    # Analytical
    p_wait_analytical = erlang_c_pwait(s, lam_per_hour, mu_per_hour)

    # Simulated
    config = StationConfig(
        station_name='_validation',
        n_chargers=s,
        arrival_mode='poisson',
        constant_lambda=lam_per_hour,
        service_params={
            'DC_Fast': {
                'best_distribution': 'exponential',
                'mean_min': mean_service_min,
                'fit_params': {'loc': 0, 'scale': mean_service_min},
            }
        },
        charger_type_mix={'DC_Fast': 1.0},
        faults_enabled=False,
        sim_days=sim_days,
        random_seed=12345,
    )

    results = simulate_station(config, n_replications=n_reps, verbose=False)
    sim_p_waits = [r['p_wait_gt_10min'] for r in results]
    # For M/M/s, P(wait>0) is Erlang-C; P(wait>10min) requires W_q distribution.
    # Better comparison: use mean_wait.
    sim_mean_waits = [r['mean_wait'] for r in results]
    sim_utils = [r['mean_utilization'] for r in results]

    # Analytical mean wait: W_q = C(s,a) / (s*mu - lambda)
    if rho < 1:
        W_q_analytical = p_wait_analytical / (s * mu_per_min - lam_per_hour / 60)
    else:
        W_q_analytical = float('inf')

    sim_mean_wait = np.mean(sim_mean_waits)
    sim_mean_util = np.mean(sim_utils)

    if verbose:
        print(f"  Analytical: P(wait)={p_wait_analytical:.4f}, "
              f"E[W_q]={W_q_analytical:.2f} min")
        print(f"  Simulated:  E[W_q]={sim_mean_wait:.2f} min "
              f"(±{np.std(sim_mean_waits):.2f}), "
              f"util={sim_mean_util:.3f} (expect {rho:.3f})")

        wait_error = abs(sim_mean_wait - W_q_analytical) / max(W_q_analytical, 0.01)
        util_error = abs(sim_mean_util - rho)
        print(f"  Wait error: {wait_error*100:.1f}%, Util error: {util_error:.4f}")

        if wait_error < 0.10 and util_error < 0.02:
            print(f"  ✓ PASS: within tolerance")
        else:
            print(f"  ⚠ CHECK: errors exceed expected tolerance")

    return {
        'analytical_p_wait': p_wait_analytical,
        'analytical_mean_wait': W_q_analytical,
        'simulated_mean_wait': sim_mean_wait,
        'simulated_mean_util': sim_mean_util,
        'analytical_rho': rho,
    }
'''

    engine_path = output_dir / 'sim_engine.py'
    with open(engine_path, 'w', encoding='utf-8') as f:
        f.write(engine_code)
    print(f"  [SAVED] {engine_path}")

    # ── Run M/M/s validation if SimPy is available ─────────────────
    if HAS_SIMPY:
        print("\n  Running M/M/s analytical validation...")
        # We exec the module to run the validation
        try:
            # Import the just-written module
            import importlib.util
            spec = importlib.util.spec_from_file_location("sim_engine",
                                                           str(engine_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {engine_path}")
            sim_mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sim_mod
            spec.loader.exec_module(sim_mod)
            val_result = sim_mod.validate_mms(s=5, lam_per_hour=3.0,
                                               mean_service_min=60.0,
                                               n_reps=100, sim_days=30)
            # Save validation result
            val_path = output_dir / 'mms_validation.json'
            with open(val_path, 'w') as f:
                json.dump(val_result, f, indent=2)
            print(f"  [SAVED] {val_path}")
        except Exception as e:
            print(f"  [WARN] Validation failed: {e}")
    else:
        print("  [SKIP] SimPy not installed; validation deferred to Week 6.")

    return engine_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    configure_console_output()
    parser = argparse.ArgumentParser(
        description='Week 5: LSTM Forecasting + SimPy Engine')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory with jiaxing_hourly.parquet')
    parser.add_argument('--week4-dir', type=str,
                        default=str(RESULTS_DIR / 'week4_results'),
                        help='Directory with Week 4 outputs')
    parser.add_argument('--week3-dir', type=str,
                        default=str(RESULTS_DIR / 'week3_results'),
                        help='Directory with nhpp_rate_functions.csv')
    parser.add_argument('--output-dir', type=str,
                        default=str(RESULTS_DIR / 'week5_results'),
                        help='Output directory')
    parser.add_argument('--skip-lstm', action='store_true',
                        help='Skip LSTM training (use for SimPy-only runs)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden size (default: 64)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size (default: 128; suitable for 4 GB GPUs)')
    parser.add_argument('--max-epochs', type=int, default=80,
                        help='Max training epochs (default: 80)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='cuda',
                        help='LSTM device: cuda (default), cpu, or auto fallback')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    print("=" * 70)
    print("WEEK 5: LSTM FORECASTING + SimPy ENGINE SKELETON")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────
    hourly = load_hourly(args.data_dir)
    param_summary = load_json(
        Path(args.week4_dir) / 'parameter_summary.json', 'parameter_summary')
    svc_summary = load_json(
        Path(args.week4_dir) / 'service_time_summary.json', 'service_time_summary')
    xgb_results = load_json(
        Path(args.week4_dir) / 'xgboost_results.json', 'xgboost_results')
    nhpp_rates = load_nhpp_rates(args.week3_dir)

    # ── Component 1: LSTM ──────────────────────────────────────────
    lstm_full = None
    lstm_no_weather = None

    if not args.skip_lstm and hourly is not None:
        data = prepare_lstm_data(hourly, output_dir)

        if data is not None:
            # Train full model (with weather)
            lstm_full = train_lstm(
                data, output_dir,
                hidden_size=args.hidden_size,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                use_weather=True,
                label='full',
                device_preference=args.device,
            )

            # Weather ablation
            lstm_no_weather = train_lstm(
                data, output_dir,
                hidden_size=args.hidden_size,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                use_weather=False,
                label='no_weather',
                device_preference=args.device,
            )

    # Comparison table
    comparison = build_comparison_table(
        lstm_full, lstm_no_weather, xgb_results, output_dir)

    # Plots
    plot_lstm_results(lstm_full, lstm_no_weather, comparison, output_dir)

    # Save LSTM results JSON
    if lstm_full:
        lstm_results = {
            'test_mae_h1': lstm_full['test_mae_h1'],
            'test_rmse_h1': lstm_full['test_rmse_h1'],
            'test_r2_h1': lstm_full['test_r2_h1'],
            'test_mae_all': lstm_full['test_mae_all'],
            'test_rmse_all': lstm_full['test_rmse_all'],
            'test_r2_all': lstm_full['test_r2_all'],
            'per_horizon_mae': lstm_full['per_horizon_mae'],
            'best_epoch': lstm_full['best_epoch'],
            'n_params': lstm_full['n_params'],
            'train_time_s': lstm_full['train_time_s'],
        }
        with open(output_dir / 'lstm_results.json', 'w') as f:
            json.dump(lstm_results, f, indent=2)
        print(f"\n[SAVED] lstm_results.json")

    # ── Component 2: SimPy Engine ──────────────────────────────────
    if param_summary:
        write_sim_engine(output_dir, param_summary, svc_summary or {},
                         nhpp_rates)
    else:
        print("\n[WARN] Cannot write SimPy engine: parameter_summary.json missing.")

    # ── Metadata ───────────────────────────────────────────────────
    files_produced = [f for f in os.listdir(output_dir)
                      if not f.startswith('.') and f != 'figures']
    figures_produced = []
    fig_dir = output_dir / 'figures'
    if fig_dir.exists():
        figures_produced = [f for f in os.listdir(fig_dir) if f.endswith('.png')]

    metadata = {
        'week': 5,
        'components': [
            'LSTM Forecasting' + (' (skipped)' if args.skip_lstm else ''),
            'SimPy Engine Skeleton',
        ],
        'files_produced': files_produced,
        'figures_produced': figures_produced,
        'lstm_trained': lstm_full is not None,
        'simpy_available': HAS_SIMPY,
        'torch_available': HAS_TORCH,
    }
    with open(output_dir / 'week5_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("WEEK 5 ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    print(f"Files: {', '.join(files_produced)}")
    print(f"Figures: {len(figures_produced)} PNG files")
    if lstm_full:
        print(f"LSTM test MAE (h1, XGBoost-comparable): "
              f"{lstm_full['test_mae_h1']:.4f}")
        print(f"LSTM test MAE (all-horizon, 24h avg):   "
              f"{lstm_full['test_mae_all']:.4f}")
    print(f"SimPy engine: {'validated' if HAS_SIMPY else 'skeleton only'}")


if __name__ == '__main__':
    main()
