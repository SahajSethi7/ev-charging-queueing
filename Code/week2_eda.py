"""
Week 2 EDA — Jiaxing EV Charging Dataset
=========================================
Self-contained script producing ~20 plots with written interpretations.
Each plot answers a modeling-relevant question for Weeks 3 to 7.

Assumptions:
- Data files are in DATA_DIR (set below).
- All cleaning from Week 1 is trusted; we do not re-clean.
- Zero-energy sessions included in arrival counts, excluded from service-time fits.
- Station grouping (92 posts → 13 stations) is frozen.

Usage:
    python week2_eda.py --data-dir ./Data --output-dir ./Results/week2_results
"""

import argparse
import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from project_paths import DATA_DIR, RESULTS_DIR
try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────
DEFAULT_DATA_DIR = DATA_DIR
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "week2_results"

STATION_ORDER = [
    "Tongxiang_Bus Station", "Tongxiang_Park B", "Tongxiang_Shopping Mall",
    "Xiuzhou_Expressway Service District C", "Xiuzhou_Government Agency",
    "Xiuzhou_Wholesale Market", "Nanhu_Technology Park", "Tongxiang_Park A",
    "Nanhu_Tourist Attraction", "Nanhu_Financial Industrial Park",
    "Tongxiang_Industrial Park", "Xiuzhou_Expressway Service District A",
    "Xiuzhou_Expressway Service District B"
]

# Short labels for plots
STATION_SHORT = {
    "Tongxiang_Bus Station": "TX Bus",
    "Tongxiang_Park B": "TX Park B",
    "Tongxiang_Shopping Mall": "TX Mall",
    "Xiuzhou_Expressway Service District C": "XZ Exp C",
    "Xiuzhou_Government Agency": "XZ Gov",
    "Xiuzhou_Wholesale Market": "XZ Market",
    "Nanhu_Technology Park": "NH Tech",
    "Tongxiang_Park A": "TX Park A",
    "Nanhu_Tourist Attraction": "NH Tourist",
    "Nanhu_Financial Industrial Park": "NH Finance",
    "Tongxiang_Industrial Park": "TX Indust",
    "Xiuzhou_Expressway Service District A": "XZ Exp A",
    "Xiuzhou_Expressway Service District B": "XZ Exp B",
}

TOU_COLORS = {"Valley": "#2196F3", "Peak": "#FF9800", "Super-peak": "#F44336"}
CHARGER_COLORS = {"DC_Fast": "#E53935", "Level_2": "#1E88E5", "Mixed": "#43A047"}
FLEX_COLORS = {"Inflexible": "#1976D2", "Flexible": "#4CAF50", "Fault": "#F44336"}


def ensure_flexibility_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Week 1 observational proxy for EDA plots."""
    if '_flexibility_label' in df.columns:
        return df
    source = ('user_stop_proxy' if 'user_stop_proxy' in df.columns
              else 'flexibility_tier')
    if source not in df.columns:
        raise KeyError('Expected user_stop_proxy from Week 1 output')
    mapping = {
        'inflexible': 'Inflexible',
        'user_stop_proxy': 'Flexible',
        'fault': 'Fault',
        'Inflexible': 'Inflexible',
        'Flexible': 'Flexible',
        'Fault': 'Fault',
    }
    df = df.copy()
    df['_flexibility_label'] = df[source].map(mapping).fillna('Inflexible')
    return df

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

PLOT_NUM = 0
INTERPRETATIONS = []


def configure_console_output():
    """Avoid UnicodeEncodeError on cp1252-like terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass


def detect_parquet_engine() -> Optional[str]:
    if find_spec("pyarrow") is not None:
        return "pyarrow"
    if find_spec("fastparquet") is not None:
        return "fastparquet"
    return None


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(series):
    if series.dtype == bool:
        return series
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    if mapped.notna().mean() >= 0.8:
        return mapped.fillna(False).astype(bool)
    return series


def load_table(data_dir, stem):
    parquet = data_dir / f"{stem}.parquet"
    csv = data_dir / f"{stem}.csv"
    engine = detect_parquet_engine()

    if parquet.exists() and engine is not None:
        return pd.read_parquet(parquet, engine=engine), parquet
    if csv.exists():
        return pd.read_csv(csv), csv
    if parquet.exists() and engine is None:
        raise RuntimeError(
            f"Found {parquet.name} but no parquet engine is installed. "
            "Install pyarrow/fastparquet or provide CSV files."
        )
    raise FileNotFoundError(f"Missing both {parquet.name} and {csv.name} in {data_dir}")


def harmonize_tables(df, hourly, daily, iat):
    """Normalize schema/type differences from Week 1 parquet vs CSV outputs."""
    # Canonical weather columns in session table.
    weather_aliases = {
        "temperature": ["temperature", "temperature_y", "temperature_x"],
        "humidity": ["humidity", "humidity_y", "humidity_x", "relative_humidity"],
        "precipitation": ["precipitation", "precipitation_y", "precipitation_x", "precipitation_mm"],
    }
    for canonical, aliases in weather_aliases.items():
        if canonical not in df.columns:
            src = first_existing(df, aliases)
            if src is not None:
                df[canonical] = df[src]

    if "tou_electricity_price" not in df.columns:
        src = first_existing(df, ["tou_electricity_price", "tou_electricity_price_yuan_kwh"])
        if src is not None:
            df["tou_electricity_price"] = df[src]

    for frame in (df, hourly, daily, iat):
        if "station_name" in frame.columns:
            frame["station_name"] = frame["station_name"].astype(str).str.strip()

    # Date/time columns
    for col in ["start_time", "end_time", "order_created_time", "payment_time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["date_dt"]:
        if col in daily.columns:
            daily[col] = pd.to_datetime(daily[col], errors="coerce")
        if col in hourly.columns:
            hourly[col] = pd.to_datetime(hourly[col], errors="coerce")
    if "datetime" in hourly.columns:
        hourly["datetime"] = pd.to_datetime(hourly["datetime"], errors="coerce")
    if "start_time" in iat.columns:
        iat["start_time"] = pd.to_datetime(iat["start_time"], errors="coerce")
    if "date" in iat.columns:
        iat["date"] = pd.to_datetime(iat["date"], errors="coerce")

    # CSV-loaded booleans.
    for col in [c for c in df.columns if c.startswith("flag_")] + ["is_null_session"]:
        if col in df.columns:
            df[col] = to_bool_series(df[col])

    # Core numeric coercions.
    num_cols_df = [
        "hour_of_day", "day_of_week", "is_weekend", "is_abnormal", "energy_kwh",
        "charging_duration_min", "effective_rate_kw", "flag_zero_energy",
        "tou_electricity_price",
    ]
    for col in num_cols_df:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for frame, cols in [
        (hourly, ["hour_of_day", "arrivals", "fault_count", "tou_electricity_price"]),
        (daily, ["arrivals", "fault_count", "mean_temperature", "total_precipitation"]),
        (iat, ["hour_of_day", "day_of_week", "is_weekend", "iat_min"]),
    ]:
        for col in cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Build missing core columns when possible.
    if "date" not in df.columns and "start_time" in df.columns:
        df["date"] = pd.to_datetime(df["start_time"], errors="coerce").dt.normalize()
    if "flag_zero_energy" not in df.columns and "energy_kwh" in df.columns:
        df["flag_zero_energy"] = (pd.to_numeric(df["energy_kwh"], errors="coerce").fillna(0) < 0.01).astype(int)
    if "station_name" not in df.columns and all(c in df.columns for c in ["district", "location_info"]):
        df["station_name"] = (df["district"].astype(str) + "_" + df["location_info"].astype(str)).str.strip()

    return df, hourly, daily, iat


def _violin_or_boxplot(data, x, y, order, ax, color, inner="box"):
    if sns is not None:
        sns.violinplot(data=data, x=x, y=y, order=order, color=color, ax=ax, inner=inner)
        return
    grouped = [data.loc[data[x] == label, y].dropna().values for label in order]
    ax.boxplot(grouped, positions=np.arange(1, len(order) + 1), widths=0.6, patch_artist=True)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order)


def _heatmap(data, ax, cmap, xticklabels=None, yticklabels=None, cbar_kws=None, linewidths=0.5, linecolor="white"):
    if sns is not None:
        sns.heatmap(
            data, ax=ax, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels,
            cbar_kws=cbar_kws, linewidths=linewidths, linecolor=linecolor
        )
        return
    arr = np.asarray(data, dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, origin="upper")
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_plot(fig, name, output_dir):
    """Save figure and increment counter."""
    global PLOT_NUM
    PLOT_NUM += 1
    fname = f"plot_{PLOT_NUM:02d}_{name}.png"
    fig.savefig(output_dir / fname)
    plt.close(fig)
    return fname


def add_interpretation(title, text, fname):
    """Store interpretation for final report."""
    INTERPRETATIONS.append({"title": title, "text": text, "file": fname})


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_data(data_dir):
    """Load all four Week 1 output files (parquet or CSV fallback)."""
    data_dir = Path(data_dir)

    print("Loading session-level data...")
    df, df_src = load_table(data_dir, "jiaxing_clean")
    print(f"  -> {len(df):,} sessions, {df.shape[1]} columns ({df_src.name})")

    print("Loading hourly data...")
    hourly, hourly_src = load_table(data_dir, "jiaxing_hourly")
    print(f"  -> {len(hourly):,} rows ({hourly_src.name})")

    print("Loading daily data...")
    daily, daily_src = load_table(data_dir, "jiaxing_daily")
    print(f"  -> {len(daily):,} rows ({daily_src.name})")

    print("Loading IAT data...")
    iat, iat_src = load_table(data_dir, "jiaxing_iat")
    print(f"  -> {len(iat):,} rows ({iat_src.name})")

    df, hourly, daily, iat = harmonize_tables(df, hourly, daily, iat)

    return df, hourly, daily, iat


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: TIME-SERIES ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def section1_timeseries(df, daily, output_dir):
    """
    Q1.1: Secular trend in daily arrivals?
    Q1.2: Weekday vs weekend?
    Q1.3: Anomalous periods (CNY, lockdowns)?
    Q1.4: Per-station trend differences?
    """
    print("\n=== Section 1: Time-Series Analysis ===")

    # ── Plot 1: Overall daily arrivals with rolling averages ──────────
    total_daily = daily.groupby("date_dt")["arrivals"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(total_daily.index, total_daily.values, alpha=0.3, linewidth=0.5,
            color="#90CAF9", label="Daily")
    ax.plot(total_daily.rolling(7, center=True).mean(),
            color="#1565C0", linewidth=1.2, label="7-day MA")
    ax.plot(total_daily.rolling(30, center=True).mean(),
            color="#E53935", linewidth=1.5, label="30-day MA")

    # Mark CNY periods (approximate)
    cny_periods = [
        ("2020-01-24", "2020-02-02", "CNY 2020"),
        ("2021-02-11", "2021-02-17", "CNY 2021"),
    ]
    for start, end, label in cny_periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color="red", zorder=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.95, label,
                fontsize=8, color="red", ha="left", va="top")

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Arrivals (all stations)")
    ax.set_title("Plot 1: Daily Arrival Volume — Full Time Span")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    fname = save_plot(fig, "daily_arrivals_trend", output_dir)

    # Compute summary stats for interpretation
    h1_mean = total_daily["2020-01":"2020-06"].mean()
    h2_mean = total_daily["2021-07":"2021-12"].mean()

    add_interpretation(
        "Plot 1: Daily Arrival Trend",
        f"Total daily arrivals across all 13 stations. "
        f"First-half 2020 mean: {h1_mean:.0f}/day; second-half 2021 mean: {h2_mean:.0f}/day. "
        f"CNY periods (red bands) show sharp dips. "
        f"Check for secular growth trend, COVID lockdown effects, and seasonal cycles. "
        f"The 30-day MA reveals whether demand is stationary (needed for Poisson testing) "
        f"or has a trend (which would require detrending for Week 3). "
        f"Any visible step changes or regime shifts must be noted for simulation.",
        fname
    )

    # ── Plot 2: Weekday vs Weekend box plot by month ──────────────────
    daily_total = daily.groupby(["date_dt", "is_weekend"]).agg(
        arrivals=("arrivals", "sum")
    ).reset_index()

    daily_total["month_str"] = daily_total["date_dt"].dt.to_period("M").astype(str)
    daily_total["day_type"] = daily_total["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    fig, ax = plt.subplots(figsize=(14, 5))
    weekday_data = daily_total[daily_total["day_type"] == "Weekday"].groupby("date_dt")["arrivals"].sum()
    weekend_data = daily_total[daily_total["day_type"] == "Weekend"].groupby("date_dt")["arrivals"].sum()

    ax.plot(weekday_data.rolling(14, center=True).mean(),
            color="#1565C0", linewidth=1.2, label="Weekday (14-day MA)")
    ax.plot(weekend_data.rolling(14, center=True).mean(),
            color="#E53935", linewidth=1.2, label="Weekend (14-day MA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Arrivals")
    ax.set_title("Plot 2: Weekday vs Weekend Arrival Volume Over Time")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    fname = save_plot(fig, "weekday_weekend_trend", output_dir)

    wd_mean = weekday_data.mean()
    we_mean = weekend_data.mean()

    add_interpretation(
        "Plot 2: Weekday vs Weekend",
        f"Weekday mean: {wd_mean:.0f}/day; Weekend mean: {we_mean:.0f}/day "
        f"(ratio: {wd_mean/we_mean:.2f}x). "
        f"If the gap is stable over time, day_of_week is a strong covariate for ML. "
        f"If the gap narrows or reverses seasonally, interaction terms may be needed. "
        f"Large weekday/weekend differences also affect the hourly NHPP rate function — "
        f"separate rate functions for weekday/weekend may be necessary.",
        fname
    )

    # ── Plot 3: Day-of-week distribution (violin) ────────────────────
    # Aggregate total arrivals per date first
    daily_by_date = daily.groupby("date_dt").agg(
        arrivals=("arrivals", "sum"),
        day_of_week=("day_of_week", "first")
    ).reset_index()

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_by_date["dow_label"] = daily_by_date["day_of_week"].map(
        dict(enumerate(dow_labels))
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    _violin_or_boxplot(
        data=daily_by_date, x="dow_label", y="arrivals",
        order=dow_labels, color="#90CAF9", ax=ax, inner="box"
    )
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Daily Arrivals")
    ax.set_title("Plot 3: Arrival Distribution by Day of Week")
    fname = save_plot(fig, "dow_violin", output_dir)

    dow_means = daily_by_date.groupby("dow_label")["arrivals"].mean()
    peak_dow = dow_means.idxmax()
    trough_dow = dow_means.idxmin()

    add_interpretation(
        "Plot 3: Day-of-Week Distribution",
        f"Peak day: {peak_dow} ({dow_means[peak_dow]:.0f} avg); "
        f"Trough day: {trough_dow} ({dow_means[trough_dow]:.0f} avg). "
        f"The shape of the violin (symmetric vs skewed) indicates whether "
        f"day-of-week means are reliable or have high variance. "
        f"This directly feeds the NHPP specification — if all weekdays are similar, "
        f"a binary weekday/weekend indicator suffices; "
        f"if Monday and Friday differ, per-day rate functions may be needed.",
        fname
    )

    # ── Plot 4: Per-station daily arrivals (small multiples) ─────────
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    for i, station in enumerate(STATION_ORDER):
        ax = axes[i]
        sdata = daily[daily["station_name"] == station].set_index("date_dt")["arrivals"]
        ax.plot(sdata.index, sdata.values, alpha=0.2, linewidth=0.4, color="#90CAF9")
        ax.plot(sdata.rolling(30, center=True).mean(),
                color="#1565C0", linewidth=1.0)
        ax.set_title(STATION_SHORT[station], fontsize=9, pad=3)
        ax.tick_params(labelsize=7)
        if i >= 12:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))

    # Hide unused subplots
    for j in range(len(STATION_ORDER), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Plot 4: Per-Station Daily Arrivals (30-day MA)", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = save_plot(fig, "per_station_daily", output_dir)

    add_interpretation(
        "Plot 4: Per-Station Trends",
        "Individual station trends reveal whether the overall pattern (Plot 1) is driven "
        "by all stations uniformly or by a few dominant ones. "
        "Stations with different trend shapes (e.g., one growing while others plateau) "
        "indicate that pooled trend removal would be inappropriate. "
        "Expressway stations may show different seasonality (holiday travel) vs urban stations. "
        "Any station with a step change (new charger installation, closure) must be flagged.",
        fname
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: HOURLY PATTERNS & TOU COUPLING
# ══════════════════════════════════════════════════════════════════════

def section2_hourly_tou(df, hourly, output_dir):
    """
    Q2.1: Hourly arrival profile shape?
    Q2.2: Profile by station type?
    Q2.3: TOU price anti-correlation?
    Q2.4: Hour × day-of-week heatmap?
    """
    print("\n=== Section 2: Hourly Patterns & TOU Coupling ===")

    # ── Plot 5: Overall hourly profile with TOU overlay ──────────────
    hourly_avg = hourly.groupby("hour_of_day")["arrivals"].mean().reindex(range(24), fill_value=0)

    # Get TOU price by hour (from session data)
    tou_by_hour = df.groupby("hour_of_day")["tou_electricity_price"].mean().reindex(range(24))
    if tou_by_hour.notna().any():
        tou_by_hour = tou_by_hour.ffill().bfill()
    else:
        tou_by_hour = pd.Series([0.0] * 24, index=range(24))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # TOU background bands
    tou_schedule = df.groupby("hour_of_day")["tou_tier"].agg(
        lambda x: x.value_counts().index[0]
    ).reindex(range(24)).fillna("Peak")
    for hour in range(24):
        tier = tou_schedule.get(hour, "Peak")
        ax1.axvspan(hour - 0.5, hour + 0.5, alpha=0.08,
                    color=TOU_COLORS.get(tier, "gray"), zorder=0)

    bar_colors = [TOU_COLORS.get(tou_schedule.get(h, "Peak"), "gray") for h in range(24)]
    ax1.bar(range(24), hourly_avg.values, color=bar_colors, alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Mean Arrivals per Station-Hour", color="#1565C0")
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    ax2.plot(range(24), tou_by_hour.values, color="#E53935", linewidth=2,
             marker="o", markersize=4, label="TOU Price (¥/kWh)")
    ax2.set_ylabel("TOU Price (¥/kWh)", color="#E53935")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TOU_COLORS["Valley"], alpha=0.5, label="Valley"),
        Patch(facecolor=TOU_COLORS["Peak"], alpha=0.5, label="Peak"),
        Patch(facecolor=TOU_COLORS["Super-peak"], alpha=0.5, label="Super-peak"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left")
    ax2.legend(loc="upper right")

    ax1.set_title("Plot 5: Hourly Arrival Profile with TOU Price Overlay")
    fname = save_plot(fig, "hourly_tou_overlay", output_dir)

    peak_hour = hourly_avg.idxmax()
    trough_hour = hourly_avg.idxmin()
    valid = ~(np.isnan(hourly_avg.values) | np.isnan(tou_by_hour.values))
    corr = np.corrcoef(hourly_avg.values[valid], tou_by_hour.values[valid])[0, 1] if valid.sum() > 1 else np.nan

    add_interpretation(
        "Plot 5: Hourly Profile + TOU",
        f"Peak arrival hour: {peak_hour}:00; Trough: {trough_hour}:00. "
        f"Correlation between hourly arrivals and TOU price: r={corr:.3f}. "
        f"Bar colors show TOU tier. If arrivals peak during Peak/Super-peak hours, "
        f"users are NOT price-responsive — they charge when they need to. "
        f"This is critical for scheduling: low price-responsiveness means the "
        f"scheduler must actively shift sessions, not rely on organic behavior. "
        f"The shape of this profile becomes the NHPP rate function λ(t).",
        fname
    )

    # ── Plot 6: Hourly profiles by station type (faceted) ────────────
    # Classify stations into types
    station_types = {
        "Expressway": [s for s in STATION_ORDER if "Expressway" in s],
        "Commercial": [s for s in STATION_ORDER if any(k in s for k in
                       ["Mall", "Market", "Bus"])],
        "Office/Gov": [s for s in STATION_ORDER if any(k in s for k in
                       ["Government", "Technology", "Financial", "Industrial"])],
        "Park/Tourist": [s for s in STATION_ORDER if any(k in s for k in
                        ["Park", "Tourist"])],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for idx, (stype, stations) in enumerate(station_types.items()):
        ax = axes[idx]
        for station in stations:
            sdata = hourly[hourly["station_name"] == station].groupby(
                "hour_of_day")["arrivals"].mean().reindex(range(24), fill_value=0)
            ax.plot(range(24), sdata.values, alpha=0.6, linewidth=1.2,
                    label=STATION_SHORT.get(station, station))
        ax.set_title(f"{stype} Stations", fontsize=10)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean Arrivals/Hour")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=7)

    fig.suptitle("Plot 6: Hourly Profiles by Station Type", fontsize=13, y=1.01)
    fig.tight_layout()
    fname = save_plot(fig, "hourly_by_station_type", output_dir)

    add_interpretation(
        "Plot 6: Station-Type Hourly Profiles",
        "Expressway stations may show flatter profiles (transient users, less time-of-day structure). "
        "Commercial stations likely peak during business/shopping hours. "
        "Office stations should peak during commute hours. "
        "If station types have fundamentally different hourly shapes, a single NHPP rate function "
        "is inappropriate — per-station or per-type rate functions are needed. "
        "This also informs whether station pooling is valid for M/M/s analysis.",
        fname
    )

    # ── Plot 7: TOU tier arrival share by hour ───────────────────────
    # Cross-tab: for each hour, what fraction of sessions fall in each TOU tier?
    # (This is informative only if TOU tier varies within hours — which it doesn't
    # since TOU is deterministic by hour. So instead: show arrivals by TOU tier.)

    # More useful: arrival VOLUME during each TOU tier
    tou_hourly = df.groupby(["hour_of_day", "tou_tier"]).size().unstack(fill_value=0)
    tou_hourly_pct = tou_hourly.div(tou_hourly.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Absolute counts by TOU tier
    tou_hourly_mean = df.groupby(["hour_of_day", "tou_tier"]).size().reset_index(name="count")
    tou_pivot = tou_hourly_mean.pivot_table(index="hour_of_day", columns="tou_tier",
                                             values="count", fill_value=0)
    tier_order = ["Valley", "Peak", "Super-peak"]
    tou_pivot = tou_pivot.reindex(columns=[t for t in tier_order if t in tou_pivot.columns])
    tou_pivot.plot.bar(stacked=True, ax=ax1,
                       color=[TOU_COLORS[t] for t in tou_pivot.columns],
                       edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Total Sessions")
    ax1.set_title("Sessions by TOU Tier")
    ax1.legend(title="TOU Tier")

    # Right: Mean energy by TOU tier
    energy_by_tou = df[df["flag_zero_energy"] == 0].groupby("tou_tier")["energy_kwh"].describe()
    tiers = ["Valley", "Peak", "Super-peak"]
    means = [energy_by_tou.loc[t, "mean"] if t in energy_by_tou.index else 0 for t in tiers]
    medians = [energy_by_tou.loc[t, "50%"] if t in energy_by_tou.index else 0 for t in tiers]

    x = np.arange(len(tiers))
    ax2.bar(x - 0.15, means, 0.3, label="Mean", color="#1565C0", alpha=0.8)
    ax2.bar(x + 0.15, medians, 0.3, label="Median", color="#FF9800", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.set_xlabel("TOU Tier")
    ax2.set_ylabel("Energy (kWh)")
    ax2.set_title("Energy per Session by TOU Tier\n(excl. zero-energy)")
    ax2.legend()

    fig.suptitle("Plot 7: TOU Tier Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = save_plot(fig, "tou_tier_analysis", output_dir)

    add_interpretation(
        "Plot 7: TOU Tier Analysis",
        "Left panel shows absolute session counts — since TOU tiers are time-deterministic, "
        "this is equivalent to 'which hours have most traffic.' "
        "Right panel shows whether users charge *more energy* during cheaper periods. "
        "If Valley sessions have higher mean energy, users may be deliberately doing full charges "
        "during off-peak. If energy is similar across tiers, there is no energy-shifting behavior. "
        "This directly bounds the scheduling optimizer's potential savings.",
        fname
    )

    # ── Plot 8: Hour × Day-of-week heatmap (top 4 stations) ─────────
    top4 = STATION_ORDER[:4]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    for i, station in enumerate(top4):
        ax = axes[i]
        sdata = hourly[hourly["station_name"] == station]
        pivot = sdata.pivot_table(index="hour_of_day", columns="day_of_week",
                                   values="arrivals", aggfunc="mean")
        pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        _heatmap(
            pivot, ax=ax, cmap="YlOrRd",
            xticklabels=list(pivot.columns), yticklabels=list(pivot.index),
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(STATION_SHORT[station], fontsize=10)
        ax.set_xlabel("Day of Week")
        if i == 0:
            ax.set_ylabel("Hour of Day")
        else:
            ax.set_ylabel("")

    fig.suptitle("Plot 8: Hour × Day Heatmaps (Top 4 Stations)", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = save_plot(fig, "hour_dow_heatmaps", output_dir)

    add_interpretation(
        "Plot 8: Hour × Day Heatmaps",
        "Each cell shows mean arrivals for that (hour, day) combination at a station. "
        "Diagonal patterns indicate that peak hours shift across days. "
        "Strong weekday/weekend contrast means is_weekend is a useful ML feature. "
        "If certain (hour, day) cells are consistently empty, the Poisson test for those intervals "
        "will have near-zero counts — these should be handled carefully. "
        "The hotspot pattern directly shapes the NHPP rate function granularity.",
        fname
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: SERVICE TIME & ENERGY DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════

def section3_service_time(df, output_dir):
    """
    Q3.1: Service time distributions by charger type?
    Q3.2: Energy distribution DC vs L2?
    Q3.3: Energy vs duration relationship?
    Q3.4: Zero-energy session profile?
    """
    print("\n=== Section 3: Service Time & Energy Distributions ===")

    # Exclude zero-energy for service time analysis
    df_svc = df[df["flag_zero_energy"] == 0].copy()
    # Also exclude extreme durations for cleaner visualization (but report the filter)
    n_before = len(df_svc)
    df_svc = df_svc[(df_svc["charging_duration_min"] > 0) &
                     (df_svc["charging_duration_min"] < 720)]  # cap at 12 hours
    n_after = len(df_svc)
    pct_excluded = (n_before - n_after) / n_before * 100

    # ── Plot 9: Service time distributions by charger type ───────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    charger_types = ["DC_Fast", "Level_2", "Mixed"]

    summary_stats = {}
    for i, ctype in enumerate(charger_types):
        ax = axes[i]
        data = df_svc[df_svc["charger_type"] == ctype]["charging_duration_min"]
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        ax.hist(data, bins=80, density=True, alpha=0.7,
                color=CHARGER_COLORS[ctype], edgecolor="white", linewidth=0.3)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1,
                   label=f"Median: {data.median():.0f} min")
        ax.axvline(data.mean(), color="red", linestyle="-", linewidth=1,
                   label=f"Mean: {data.mean():.0f} min")
        ax.set_title(f"{ctype}\n(n={len(data):,})", fontsize=10)
        ax.set_xlabel("Duration (min)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 480)

        summary_stats[ctype] = {
            "n": len(data), "mean": data.mean(), "median": data.median(),
            "std": data.std(), "cv": data.std() / data.mean(),
            "skew": data.skew(), "p75": data.quantile(0.75),
        }

    fig.suptitle(f"Plot 9: Service Time Distributions by Charger Type\n"
                 f"(excl. zero-energy and >12h sessions; {pct_excluded:.1f}% excluded)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fname = save_plot(fig, "service_time_by_charger", output_dir)

    interp_parts = []
    for ct, s in summary_stats.items():
        interp_parts.append(f"{ct}: mean={s['mean']:.0f}min, median={s['median']:.0f}min, "
                           f"CV={s['cv']:.2f}, skew={s['skew']:.2f}")

    add_interpretation(
        "Plot 9: Service Time Distributions",
        f"Service time distributions (excluding zero-energy, excluding >{12}h; "
        f"{pct_excluded:.1f}% of non-zero sessions excluded). "
        f"{'; '.join(interp_parts)}. "
        f"Right-skewed distributions with CV>1 indicate that M/M/s (which assumes exponential "
        f"service times with CV=1) will underestimate congestion. "
        f"M/G/s with fitted distributions (lognormal or Weibull) will be more accurate. "
        f"The mean/median gap indicates the impact of long-tail sessions.",
        fname
    )

    # ── Plot 10: Energy distributions by charger type ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, ctype in enumerate(charger_types):
        ax = axes[i]
        data = df_svc[df_svc["charger_type"] == ctype]["energy_kwh"]
        if len(data) == 0:
            continue

        ax.hist(data, bins=80, density=True, alpha=0.7,
                color=CHARGER_COLORS[ctype], edgecolor="white", linewidth=0.3)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1,
                   label=f"Median: {data.median():.1f} kWh")
        ax.set_title(f"{ctype}", fontsize=10)
        ax.set_xlabel("Energy (kWh)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, data.quantile(0.99))

    fig.suptitle("Plot 10: Energy Distributions by Charger Type (excl. zero-energy)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fname = save_plot(fig, "energy_by_charger", output_dir)

    add_interpretation(
        "Plot 10: Energy Distributions",
        "Energy per session informs the scheduling optimizer's load-shifting potential. "
        "If DC Fast sessions cluster tightly (low variance), their service demand is predictable. "
        "If L2 sessions have heavy tails, some sessions occupy chargers for long periods. "
        "The energy distribution also determines whether a fixed kWh capacity or variable "
        "duration-based model is more appropriate for simulation.",
        fname
    )

    # ── Plot 11: Energy vs Duration scatter ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    sample_size = min(20000, len(df_svc))
    df_sample = df_svc.sample(sample_size, random_state=42)

    for ctype in charger_types:
        mask = df_sample["charger_type"] == ctype
        ax.scatter(df_sample.loc[mask, "charging_duration_min"],
                   df_sample.loc[mask, "energy_kwh"],
                   alpha=0.15, s=8, color=CHARGER_COLORS[ctype],
                   label=f"{ctype} (n={mask.sum():,})")

    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(f"Plot 11: Energy vs Duration by Charger Type (n={sample_size:,} sample)")
    ax.legend()
    ax.set_xlim(0, 480)
    ax.set_ylim(0, df_sample["energy_kwh"].quantile(0.99))
    fname = save_plot(fig, "energy_vs_duration", output_dir)

    add_interpretation(
        "Plot 11: Energy vs Duration",
        "If the scatter shows tight linear bands, effective charging rate is consistent "
        "within each charger type. Spread indicates variable rates (partial charging, "
        "battery SoC effects, power degradation). "
        "Points along the x-axis (high duration, low energy) represent sessions where the "
        "car is parked but not charging — these inflate service time relative to actual demand. "
        "For M/G/s modeling, we need the *occupancy* duration, not just the charging duration.",
        fname
    )

    # ── Plot 12: Zero-energy session profile ─────────────────────────
    df_zero = df[df["flag_zero_energy"] == 1].copy()
    df_nonzero = df[df["flag_zero_energy"] == 0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Hourly distribution comparison
    ax = axes[0]
    zero_hourly = df_zero.groupby("hour_of_day").size() / len(df_zero) * 100
    nonzero_hourly = df_nonzero.groupby("hour_of_day").size() / len(df_nonzero) * 100
    ax.bar(np.arange(24) - 0.2, nonzero_hourly.values, 0.4,
           label="Normal", color="#1565C0", alpha=0.7)
    ax.bar(np.arange(24) + 0.2, zero_hourly.values, 0.4,
           label="Zero-energy", color="#F44336", alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("% of Sessions")
    ax.set_title("Hourly Distribution")
    ax.legend()
    ax.set_xticks(range(0, 24, 3))

    # Station concentration
    ax = axes[1]
    zero_by_station = df_zero.groupby("station_name").size() / df.groupby("station_name").size() * 100
    zero_by_station = zero_by_station.reindex(STATION_ORDER).fillna(0)
    short_labels = [STATION_SHORT[s] for s in STATION_ORDER]
    ax.barh(short_labels, zero_by_station.values, color="#F44336", alpha=0.7)
    ax.set_xlabel("Zero-Energy Rate (%)")
    ax.set_title("Zero-Energy % by Station")
    ax.axvline(12.4, color="black", linestyle="--", linewidth=0.8, label="Overall: 12.4%")
    ax.legend(fontsize=8)

    # Duration distribution of zero-energy sessions
    ax = axes[2]
    zero_dur = df_zero["charging_duration_min"].clip(upper=60)
    ax.hist(zero_dur, bins=60, color="#F44336", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Duration (min, capped at 60)")
    ax.set_ylabel("Count")
    ax.set_title("Zero-Energy Session Duration")
    ax.axvline(zero_dur.median(), color="black", linestyle="--",
               label=f"Median: {zero_dur.median():.1f} min")
    ax.legend(fontsize=8)

    fig.suptitle(f"Plot 12: Zero-Energy Session Profile (n={len(df_zero):,}, {len(df_zero)/len(df)*100:.1f}%)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fname = save_plot(fig, "zero_energy_profile", output_dir)

    add_interpretation(
        "Plot 12: Zero-Energy Sessions",
        f"N={len(df_zero):,} sessions ({len(df_zero)/len(df)*100:.1f}%) delivered zero energy. "
        f"Left: hourly distribution — if these mirror normal sessions, they are random connector failures. "
        f"If they cluster at specific hours, there may be a systematic cause. "
        f"Center: station-level rates — high concentration at specific stations suggests equipment issues. "
        f"Right: duration distribution — very short durations (<1 min) = tap-and-leave; "
        f"longer durations = connector failure with car parked. "
        f"These sessions count as arrivals (occupy queue slot) but not service demand (no energy). "
        f"For simulation, they should have near-zero service time but still occupy a charger.",
        fname
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: FAULT & FLEXIBILITY STRUCTURE
# ══════════════════════════════════════════════════════════════════════

def section4_faults_flexibility(df, hourly, output_dir):
    """
    Q4.1: Fault rate variation across stations?
    Q4.2: Temporal clustering of faults?
    Q4.3: Flexibility × TOU cross-tabulation?
    Q4.4: Fault / zero-energy overlap?
    """
    print("\n=== Section 4: Fault & Flexibility Structure ===")
    df = ensure_flexibility_labels(df)

    # ── Plot 13: Fault rate by station ───────────────────────────────
    fault_by_station = df.groupby("station_name").agg(
        total=("is_abnormal", "count"),
        faults=("is_abnormal", "sum")
    )
    fault_by_station["fault_rate"] = fault_by_station["faults"] / fault_by_station["total"] * 100
    fault_by_station = fault_by_station.reindex(STATION_ORDER)

    fig, ax = plt.subplots(figsize=(12, 6))
    short_labels = [STATION_SHORT[s] for s in STATION_ORDER]
    colors = ["#F44336" if r > 25 else "#FF9800" if r > 15 else "#4CAF50"
              for r in fault_by_station["fault_rate"]]
    ax.barh(short_labels, fault_by_station["fault_rate"], color=colors, alpha=0.8)
    ax.axvline(18.6, color="black", linestyle="--", linewidth=0.8, label="Overall: 18.6%")
    ax.set_xlabel("Fault Rate (%)")
    ax.set_title("Plot 13: Fault Rate by Station")
    ax.legend()

    # Annotate counts
    for i, (_, row) in enumerate(fault_by_station.iterrows()):
        ax.text(row["fault_rate"] + 0.3, i,
                f"n={row['faults']:.0f}/{row['total']:.0f}",
                va="center", fontsize=7)

    fig.tight_layout()
    fname = save_plot(fig, "fault_rate_by_station", output_dir)

    max_fault_station = fault_by_station["fault_rate"].idxmax()
    min_fault_station = fault_by_station["fault_rate"].idxmin()

    add_interpretation(
        "Plot 13: Fault Rate by Station",
        f"Highest: {STATION_SHORT[max_fault_station]} ({fault_by_station.loc[max_fault_station, 'fault_rate']:.1f}%); "
        f"Lowest: {STATION_SHORT[min_fault_station]} ({fault_by_station.loc[min_fault_station, 'fault_rate']:.1f}%). "
        f"Stations with >25% fault rates may need station-specific fault models in simulation. "
        f"If fault rates correlate with charger type (DC Fast stations having higher faults), "
        f"this should be reflected in the fault injection model. "
        f"Wide variation (>3x) across stations means a uniform 18.6% rate is inappropriate for "
        f"per-station simulation.",
        fname
    )

    # ── Plot 14: Fault rate by hour and charger type ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # By hour
    fault_hourly = df.groupby("hour_of_day").agg(
        total=("is_abnormal", "count"),
        faults=("is_abnormal", "sum")
    )
    fault_hourly["rate"] = fault_hourly["faults"] / fault_hourly["total"] * 100
    ax1.bar(range(24), fault_hourly["rate"], color="#F44336", alpha=0.7)
    ax1.axhline(18.6, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Fault Rate (%)")
    ax1.set_title("Fault Rate by Hour")
    ax1.set_xticks(range(0, 24, 3))

    # By charger type
    fault_charger = df.groupby("charger_type").agg(
        total=("is_abnormal", "count"),
        faults=("is_abnormal", "sum")
    )
    fault_charger["rate"] = fault_charger["faults"] / fault_charger["total"] * 100
    charger_order = ["DC_Fast", "Level_2", "Mixed"]
    fault_charger = fault_charger.reindex([c for c in charger_order if c in fault_charger.index])
    ax2.bar(range(len(fault_charger)), fault_charger["rate"],
            color=[CHARGER_COLORS.get(c, "gray") for c in fault_charger.index],
            alpha=0.8)
    ax2.set_xticks(range(len(fault_charger)))
    ax2.set_xticklabels(fault_charger.index)
    ax2.axhline(18.6, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Fault Rate (%)")
    ax2.set_title("Fault Rate by Charger Type")

    fig.suptitle("Plot 14: Temporal and Equipment Fault Patterns", fontsize=12, y=1.02)
    fig.tight_layout()
    fname = save_plot(fig, "fault_hourly_charger", output_dir)

    add_interpretation(
        "Plot 14: Fault Temporal/Equipment Patterns",
        "Left: If fault rate is flat across hours, faults are independent of demand level "
        "(equipment-driven, not congestion-driven). If faults spike during high-demand hours, "
        "capacity stress may cause faults — this has simulation implications. "
        "Right: If DC Fast has higher fault rates, the fault injection model should condition "
        "on charger type. Uniform fault rate by charger type simplifies the simulation model.",
        fname
    )

    # ── Plot 15: Flexibility tier × TOU tier × hour ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    flex_tiers = ["Inflexible", "Flexible", "Fault"]

    # Flexibility × TOU cross-tab
    ax = axes[0]
    ct = pd.crosstab(df["_flexibility_label"], df["tou_tier"], normalize="columns") * 100
    ct = ct.reindex(index=flex_tiers, columns=["Valley", "Peak", "Super-peak"])
    ct.plot.bar(ax=ax, color=[TOU_COLORS[t] for t in ct.columns], alpha=0.8)
    ax.set_ylabel("% of TOU Tier Sessions")
    ax.set_title("Flexibility Tier by TOU Tier")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="TOU")

    # Flexibility distribution by hour
    ax = axes[1]
    flex_hourly = pd.crosstab(df["hour_of_day"], df["_flexibility_label"],
                               normalize="index") * 100
    flex_hourly = flex_hourly.reindex(columns=flex_tiers)
    flex_hourly.plot.area(ax=ax, stacked=True,
                          color=[FLEX_COLORS[t] for t in flex_tiers], alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("% of Sessions")
    ax.set_title("Flexibility Composition by Hour")
    ax.set_xlim(0, 23)
    ax.legend(fontsize=8)

    # Flexible sessions: potential for shifting
    ax = axes[2]
    flex_only = df[df["_flexibility_label"] == "Flexible"]
    flex_tou_counts = flex_only.groupby("tou_tier").size()
    flex_tou_pct = flex_tou_counts / flex_tou_counts.sum() * 100
    tier_order = ["Valley", "Peak", "Super-peak"]
    flex_tou_pct = flex_tou_pct.reindex(tier_order).fillna(0)
    ax.bar(tier_order, flex_tou_pct.values,
           color=[TOU_COLORS[t] for t in tier_order], alpha=0.8)
    ax.set_ylabel("% of Flexible Sessions")
    ax.set_title("Where Flexible Sessions Occur")

    fig.suptitle("Plot 15: Flexibility & TOU Interaction", fontsize=12, y=1.03)
    fig.tight_layout()
    fname = save_plot(fig, "flexibility_tou", output_dir)

    flex_peak_pct = flex_tou_pct.get("Peak", 0) + flex_tou_pct.get("Super-peak", 0)

    add_interpretation(
        "Plot 15: Flexibility × TOU",
        f"Flexible sessions during Peak+Super-peak: {flex_peak_pct:.1f}%. "
        f"These are the sessions the scheduler could potentially shift to Valley. "
        f"Left panel: If Flexible fraction is higher during Peak/Super-peak, "
        f"users who can be interrupted tend to charge during expensive hours — "
        f"this is exactly the target population for scheduling. "
        f"Center panel: Hourly flexibility composition reveals when shifting potential exists. "
        f"Right panel: Quantifies the upper bound on sessions available for TOU optimization.",
        fname
    )

    # ── Plot 16: Fault / zero-energy overlap ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))

    # Venn-style counts
    fault_mask = df["is_abnormal"] == 1
    zero_mask = df["flag_zero_energy"] == 1

    both = (fault_mask & zero_mask).sum()
    fault_only = (fault_mask & ~zero_mask).sum()
    zero_only = (~fault_mask & zero_mask).sum()
    neither = (~fault_mask & ~zero_mask).sum()

    labels = ["Fault only", "Zero-energy only", "Both", "Neither"]
    counts = [fault_only, zero_only, both, neither]
    pcts = [c / len(df) * 100 for c in counts]
    colors_bar = ["#F44336", "#FF9800", "#9C27B0", "#4CAF50"]

    bars = ax.barh(labels, pcts, color=colors_bar, alpha=0.8)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("% of All Sessions")
    ax.set_title("Plot 16: Fault × Zero-Energy Overlap")
    fig.tight_layout()
    fname = save_plot(fig, "fault_zero_overlap", output_dir)

    add_interpretation(
        "Plot 16: Fault / Zero-Energy Overlap",
        f"Fault-only: {fault_only:,} ({fault_only/len(df)*100:.1f}%); "
        f"Zero-energy-only: {zero_only:,} ({zero_only/len(df)*100:.1f}%); "
        f"Both: {both:,} ({both/len(df)*100:.1f}%); "
        f"Neither: {neither:,} ({neither/len(df)*100:.1f}%). "
        f"High overlap means faults largely explain zero-energy sessions. "
        f"Low overlap means zero-energy has a different mechanism (user behavior, not equipment). "
        f"For simulation: if most zero-energy sessions are also faults, a single fault model "
        f"covers both; otherwise, separate zero-energy injection is needed.",
        fname
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: WEATHER RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════

def section5_weather(df, daily, output_dir):
    """
    Q5.1: Daily arrivals vs temperature/precipitation?
    Q5.2: Seasonality confound?
    """
    print("\n=== Section 5: Weather Relationships ===")

    # Aggregate to daily total (across all stations)
    daily_agg = daily.groupby("date_dt").agg(
        arrivals=("arrivals", "sum"),
        mean_temp=("mean_temperature", "mean"),
        total_precip=("total_precipitation", "sum"),
        month=("month", "first"),
    ).reset_index()

    # ── Plot 17: Weather scatter plots ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Temperature vs arrivals (colored by month)
    ax = axes[0]
    scatter = ax.scatter(daily_agg["mean_temp"], daily_agg["arrivals"],
                         c=daily_agg["month"], cmap="coolwarm", alpha=0.4, s=12)
    plt.colorbar(scatter, ax=ax, label="Month")
    ax.set_xlabel("Mean Temperature (°C)")
    ax.set_ylabel("Total Daily Arrivals")
    ax.set_title("Temperature vs Arrivals")

    r_temp = daily_agg[["mean_temp", "arrivals"]].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f"r = {r_temp:.3f}", transform=ax.transAxes,
            fontsize=9, va="top")

    # Precipitation vs arrivals
    ax = axes[1]
    ax.scatter(daily_agg["total_precip"], daily_agg["arrivals"],
               alpha=0.3, s=12, color="#1565C0")
    ax.set_xlabel("Total Precipitation (mm)")
    ax.set_ylabel("Total Daily Arrivals")
    ax.set_title("Precipitation vs Arrivals")

    r_precip = daily_agg[["total_precip", "arrivals"]].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f"r = {r_precip:.3f}", transform=ax.transAxes,
            fontsize=9, va="top")

    # Rainy vs non-rainy day comparison
    ax = axes[2]
    daily_agg["rainy"] = daily_agg["total_precip"] > 0.5
    rain_groups = daily_agg.groupby("rainy")["arrivals"]
    box_data = [rain_groups.get_group(False).values, rain_groups.get_group(True).values] \
        if True in rain_groups.groups and False in rain_groups.groups else [daily_agg["arrivals"].values]
    bp = ax.boxplot(box_data, tick_labels=["Dry", "Rainy"] if len(box_data) == 2 else ["All"],
                    patch_artist=True)
    if len(box_data) == 2:
        bp["boxes"][0].set_facecolor("#FFA726")
        bp["boxes"][1].set_facecolor("#42A5F5")
    ax.set_ylabel("Total Daily Arrivals")
    ax.set_title("Rainy vs Dry Days")

    fig.suptitle("Plot 17: Weather Effects on Daily Arrivals", fontsize=12, y=1.02)
    fig.tight_layout()
    fname = save_plot(fig, "weather_effects", output_dir)

    add_interpretation(
        "Plot 17: Weather Effects",
        f"Temperature-arrivals correlation: r={r_temp:.3f}; "
        f"Precipitation-arrivals correlation: r={r_precip:.3f}. "
        f"CAUTION: The temperature correlation may reflect seasonality (month coloring in left panel). "
        f"If warm months happen to have more arrivals due to secular growth, "
        f"this is a confound, not a weather effect. "
        f"The rainy/dry comparison (right) is less confounded since rain varies day-to-day. "
        f"If the rainy-day effect is <5%, weather is not worth adding to the ML model "
        f"as a primary feature.",
        fname
    )

    # ── Plot 18: Detrended weather effect ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residualize: subtract monthly mean from arrivals and temperature
    monthly_means = daily_agg.groupby("month").agg(
        arr_mean=("arrivals", "mean"),
        temp_mean=("mean_temp", "mean"),
    )
    daily_agg = daily_agg.merge(monthly_means, left_on="month", right_index=True)
    daily_agg["arr_resid"] = daily_agg["arrivals"] - daily_agg["arr_mean"]
    daily_agg["temp_resid"] = daily_agg["mean_temp"] - daily_agg["temp_mean"]

    ax = axes[0]
    ax.scatter(daily_agg["temp_resid"], daily_agg["arr_resid"],
               alpha=0.3, s=10, color="#1565C0")
    r_detrend = daily_agg[["temp_resid", "arr_resid"]].corr().iloc[0, 1]
    ax.set_xlabel("Temperature Residual (°C, detrended)")
    ax.set_ylabel("Arrival Residual (detrended)")
    ax.set_title(f"Detrended: Temp vs Arrivals (r={r_detrend:.3f})")

    # Heavy rain effect (>10mm)
    ax = axes[1]
    daily_agg["rain_cat"] = pd.cut(daily_agg["total_precip"],
                                     bins=[-0.1, 0.5, 5, 20, 200],
                                     labels=["None", "Light", "Moderate", "Heavy"])
    rain_means = daily_agg.groupby("rain_cat")["arr_resid"].agg(["mean", "sem", "count"])
    ax.bar(range(len(rain_means)), rain_means["mean"],
           yerr=rain_means["sem"] * 1.96,
           color=["#FFA726", "#42A5F5", "#1565C0", "#0D47A1"], alpha=0.8,
           capsize=4)
    ax.set_xticks(range(len(rain_means)))
    ax.set_xticklabels([f"{cat}\n(n={n:.0f})" for cat, n in
                         zip(rain_means.index, rain_means["count"])], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Arrival Residual (detrended)")
    ax.set_title("Detrended Arrivals by Precipitation Category")

    fig.suptitle("Plot 18: Detrended Weather Analysis", fontsize=12, y=1.03)
    fig.tight_layout()
    fname = save_plot(fig, "weather_detrended", output_dir)

    add_interpretation(
        "Plot 18: Detrended Weather",
        f"After removing monthly means (controlling for seasonality): "
        f"Temperature-arrivals correlation drops to r={r_detrend:.3f}. "
        f"If the detrended correlation is near zero, the raw correlation was purely seasonal confound. "
        f"Right panel: Precipitation categories after detrending — if heavy rain days show "
        f"statistically significant negative residuals, rain is a real (if modest) effect. "
        f"This determines whether precipitation should be an ML feature (Week 5) or can be omitted.",
        fname
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: STATION HETEROGENEITY SUMMARY
# ══════════════════════════════════════════════════════════════════════

def section6_heterogeneity(df, daily, iat, output_dir):
    """
    Q6.1: Key metrics comparison across stations?
    Q6.2: Can stations be clustered?
    """
    print("\n=== Section 6: Station Heterogeneity ===")
    df = ensure_flexibility_labels(df)

    # Build station summary table
    df_svc = df[df["flag_zero_energy"] == 0]
    session_key = "order_id" if "order_id" in df.columns else "start_time"
    n_days = max(pd.to_datetime(df["date"], errors="coerce").nunique(), 1) if "date" in df.columns else 1

    station_summary = df.groupby("station_name").agg(
        total_sessions=(session_key, "count"),
        mean_arrivals_day=(session_key, lambda x: len(x) / n_days),
        fault_rate=("is_abnormal", "mean"),
        zero_energy_rate=("flag_zero_energy", "mean"),
        flex_rate=("_flexibility_label", lambda x: (x == "Flexible").mean()),
    )

    svc_stats = df_svc.groupby("station_name").agg(
        mean_duration=("charging_duration_min", "mean"),
        median_duration=("charging_duration_min", "median"),
        mean_energy=("energy_kwh", "mean"),
        pct_dc_fast=("charger_type", lambda x: (x == "DC_Fast").mean()),
    )

    # IAT dispersion per station
    iat_stats = iat.groupby("station_name")["iat_min"].agg(
        iat_mean="mean",
        iat_var="var"
    )
    iat_stats["dispersion_index"] = iat_stats["iat_var"] / iat_stats["iat_mean"]

    station_summary = station_summary.join(svc_stats).join(iat_stats)
    station_summary = station_summary.reindex(STATION_ORDER)

    # ── Plot 19: Station comparison radar-style (heatmap) ────────────
    # Normalize key metrics to 0–1 for heatmap
    metrics_to_plot = [
        "mean_arrivals_day", "fault_rate", "zero_energy_rate", "flex_rate",
        "mean_duration", "mean_energy", "pct_dc_fast", "dispersion_index"
    ]
    metric_labels = [
        "Arrivals/day", "Fault Rate", "Zero-Energy %", "Flex %",
        "Mean Duration", "Mean Energy", "% DC Fast", "IAT Dispersion"
    ]

    norm_data = station_summary[metrics_to_plot].copy()
    for col in norm_data.columns:
        rng = norm_data[col].max() - norm_data[col].min()
        if rng > 0:
            norm_data[col] = (norm_data[col] - norm_data[col].min()) / rng
        else:
            norm_data[col] = 0.5

    fig, ax = plt.subplots(figsize=(14, 8))
    short_idx = [STATION_SHORT[s] for s in norm_data.index]

    _heatmap(
        norm_data.values, ax=ax, cmap="YlOrRd",
        xticklabels=metric_labels, yticklabels=short_idx,
        linewidths=0.5, linecolor="white"
    )

    # Annotate with raw values
    for i, station in enumerate(norm_data.index):
        for j, col in enumerate(metrics_to_plot):
            val = station_summary.loc[station, col]
            if col in ["fault_rate", "zero_energy_rate", "flex_rate", "pct_dc_fast"]:
                text = f"{val*100:.0f}%"
            elif col == "dispersion_index":
                text = f"{val:.0f}"
            elif col == "mean_arrivals_day":
                text = f"{val:.0f}"
            elif col == "mean_duration":
                text = f"{val:.0f}m"
            elif col == "mean_energy":
                text = f"{val:.1f}"
            else:
                text = f"{val:.1f}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                    fontsize=7, color="black" if norm_data.values[i, j] < 0.65 else "white")

    ax.set_title("Plot 19: Station Heterogeneity Heatmap (normalized, raw values annotated)")
    fig.tight_layout()
    fname = save_plot(fig, "station_heterogeneity_heatmap", output_dir)

    add_interpretation(
        "Plot 19: Station Heterogeneity",
        "This heatmap reveals the structural diversity across stations. "
        "Key patterns to check: (1) Do high-traffic stations have lower fault rates (better maintained)? "
        "(2) Do expressway stations cluster together (high DC Fast %, high fault rate, short duration)? "
        "(3) Is IAT dispersion uniformly high, or do some stations approach Poisson-like behavior? "
        "Stations with similar profiles could potentially be pooled for M/M/s analysis; "
        "stations with distinct profiles require per-station treatment. "
        "This is the master reference for simulation parameterization.",
        fname
    )

    # ── Plot 20: IAT dispersion index by station ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    disp = station_summary["dispersion_index"].sort_values(ascending=True)
    short_labels = [STATION_SHORT[s] for s in disp.index]
    colors = ["#F44336" if d > 100 else "#FF9800" if d > 10 else "#4CAF50"
              for d in disp.values]

    ax.barh(short_labels, disp.values, color=colors, alpha=0.8)
    ax.axvline(1.0, color="black", linewidth=2, linestyle="-", label="Poisson (DI=1)")
    ax.axvline(10, color="orange", linewidth=1, linestyle="--", label="Mild overdispersion")
    ax.set_xlabel("Dispersion Index (Variance / Mean of IAT)")
    ax.set_title("Plot 20: IAT Dispersion Index by Station")
    ax.legend()
    ax.set_xscale("log")

    # Annotate
    for i, (val, label) in enumerate(zip(disp.values, short_labels)):
        ax.text(val * 1.1, i, f"{val:.0f}", va="center", fontsize=8)

    fig.tight_layout()
    fname = save_plot(fig, "iat_dispersion_by_station", output_dir)

    min_di = disp.min()
    max_di = disp.max()
    all_above_1 = (disp > 1).all()

    add_interpretation(
        "Plot 20: IAT Dispersion by Station",
        f"Dispersion Index range: {min_di:.0f} – {max_di:.0f}. "
        f"All stations above DI=1: {all_above_1}. "
        f"DI=1 means Poisson; DI>>1 means overdispersed (arrivals are burstier than Poisson). "
        f"IMPORTANT CAVEAT: This is the *unconditional* dispersion index using all IATs. "
        f"The Poisson hypothesis is about homogeneous intervals. "
        f"After conditioning on hour-of-day (NHPP), the within-interval dispersion may be much lower. "
        f"Week 3 must test Poisson *per hour-of-day stratum*, not on raw IATs. "
        f"This plot sets the expectation: raw Poisson will almost certainly be rejected; "
        f"the question is whether NHPP with hourly rates rescues it.",
        fname
    )

    # Save the station summary table
    station_summary.to_csv(output_dir / "station_summary_table.csv")

    return station_summary


# ══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_report(output_dir, station_summary):
    """Generate a markdown report with all interpretations."""
    lines = [
        "# Week 2 EDA Report — Jiaxing EV Charging Dataset\n",
        f"*Generated by week2_eda.py — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "---\n",
    ]

    for item in INTERPRETATIONS:
        lines.append(f"## {item['title']}\n")
        lines.append(f"![{item['title']}]({item['file']})\n")
        lines.append(f"{item['text']}\n")
        lines.append("---\n")

    # Station summary table
    lines.append("## Station Summary Table\n")
    lines.append("| Station | Sessions | Arr/day | Fault% | Zero-E% | "
                 "Flex% | Duration | Energy | DC% | IAT DI |\n")
    lines.append("|---------|----------|---------|--------|---------|"
                 "-------|----------|--------|-----|--------|\n")
    for station in STATION_ORDER:
        if station in station_summary.index:
            s = station_summary.loc[station]
            lines.append(
                f"| {STATION_SHORT[station]} | {s['total_sessions']:,.0f} | "
                f"{s['mean_arrivals_day']:.0f} | {s['fault_rate']*100:.1f}% | "
                f"{s['zero_energy_rate']*100:.1f}% | {s['flex_rate']*100:.1f}% | "
                f"{s['mean_duration']:.0f}m | {s['mean_energy']:.1f}kWh | "
                f"{s['pct_dc_fast']*100:.0f}% | {s['dispersion_index']:.0f} |\n"
            )

    # Key findings for downstream phases
    lines.append("\n## Key Findings for Downstream Modeling\n")
    lines.append(
        "1. **Poisson Validity (Week 3):** IAT dispersion indices are uniformly >>1 across all "
        "stations, confirming raw Poisson will be rejected. NHPP with hourly rate functions "
        "is the appropriate next step. Per-station testing is mandatory.\n\n"
        "2. **NHPP Specification (Week 3):** Hourly profiles show strong time-of-day periodicity. "
        "Weekday/weekend separation is likely necessary. The TOU price overlay determines "
        "whether price should be a covariate in the rate function.\n\n"
        "3. **ML Feature Design (Week 5):** Key features: hour_of_day, day_of_week, is_weekend, "
        "lag_1, lag_24, lag_168, rolling_7d_mean. Weather features should be included only if "
        "detrended analysis shows significant effects. Station-level features capture "
        "structural heterogeneity.\n\n"
        "4. **Service Time for M/G/s (Week 4):** Distributions are right-skewed with CV > 1, "
        "confirming M/M/s (CV=1 assumption) will underestimate congestion. "
        "Separate fits needed for DC Fast and Level 2. Mixed category needs decision.\n\n"
        "5. **Simulation Inputs (Week 6):** Per-station arrival rates (NHPP), per-charger-type "
        "service time distributions, per-station fault rates (not uniform 18.6%), "
        "and zero-energy session handling (short occupancy, no energy).\n\n"
        "6. **Scheduling Potential (Week 7):** Bounded by the fraction of Flexible sessions "
        "occurring during Peak/Super-peak hours. TOU price ratio of 3.19x sets the maximum "
        "per-session savings. Actual savings depend on behavioral flexibility distribution.\n"
    )

    # Sanity checks
    lines.append("\n## Sanity Checks\n")
    lines.append(
        "- [ ] Total sessions in all plots = 441,077\n"
        "- [ ] Station counts sum correctly\n"
        "- [ ] Fault rate in Plot 13 averages to ~18.6% when weighted by station size\n"
        "- [ ] Zero-energy count matches Week 1 report (54,621)\n"
        "- [ ] Hourly profile sums to ~603/day across all stations\n"
        "- [ ] No plot shows data outside Jan 2020 – Dec 2021 range\n"
        "- [ ] Service time plots exclude zero-energy sessions\n"
        "- [ ] Weather detrended correlation is smaller than raw correlation\n"
    )

    # Failure modes
    lines.append("\n## Failure Modes to Watch\n")
    lines.append(
        "1. **Aggregation artifact in hourly profile:** Pooling across stations with different "
        "peak hours would flatten the overall profile. Check per-station profiles match.\n"
        "2. **Survivor bias in service time:** If we only measure completed sessions, "
        "we miss sessions still in progress at data cutoff (right censoring).\n"
        "3. **Weather confound:** Temperature correlation may entirely reflect seasonality. "
        "The detrended analysis is the valid test.\n"
        "4. **CNY distortion:** If CNY periods aren't excluded from summary statistics, "
        "they pull down means and inflate variance.\n"
        "5. **Charger type boundary:** The 22/30 kW cutoffs for DC/L2/Mixed are arbitrary. "
        "If the distribution is smooth across this boundary, the classification may be misleading.\n"
    )

    report_path = output_dir / "week2_eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nReport saved to: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    configure_console_output()
    parser = argparse.ArgumentParser(description="Week 2 EDA — Jiaxing EV Charging")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory containing Week 1 output files (parquet/csv)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for output plots and report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    if sns is None:
        print("[WARN] seaborn not installed; using matplotlib fallback for violin/heatmap plots.")
    print("=" * 60)

    # Load data
    df, hourly, daily, iat = load_data(data_dir)

    # Sanity check
    print(f"\n── Quick Sanity Check ──")
    print(f"Sessions: {len(df):,}")
    if "station_name" in df.columns:
        print(f"Stations: {df['station_name'].nunique()}")
    if "date" in df.columns:
        print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    if "is_abnormal" in df.columns:
        print(f"Fault rate: {df['is_abnormal'].mean()*100:.2f}%")
    if "flag_zero_energy" in df.columns:
        print(f"Zero-energy: {df['flag_zero_energy'].sum():,} ({df['flag_zero_energy'].mean()*100:.1f}%)")

    # Run all sections
    section1_timeseries(df, daily, output_dir)
    section2_hourly_tou(df, hourly, output_dir)
    section3_service_time(df, output_dir)
    section4_faults_flexibility(df, hourly, output_dir)
    section5_weather(df, daily, output_dir)
    station_summary = section6_heterogeneity(df, daily, iat, output_dir)

    # Generate report
    report_path = generate_report(output_dir, station_summary)

    print(f"\n{'='*60}")
    print(f"EDA complete. {PLOT_NUM} plots generated.")
    print(f"Report: {report_path}")
    print(f"Station summary: {output_dir / 'station_summary_table.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
