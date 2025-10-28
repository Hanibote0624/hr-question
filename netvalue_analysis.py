#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NetValue analysis & plotting script

Usage:
    python netvalue_analysis.py --csv /path/to/NetValue.csv --days 365 --rf 0.0

Produces:
  - NetValue_plot.png: NAV curves for two strategies
  - NetValue_metrics.csv: metrics table

Notes:
  - Assumes two numeric NAV columns in the CSV (plus an optional date/datetime column).
  - Annualization uses TRADING_DAYS_PER_YEAR as provided via --days (default 365).
"""
import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def maybe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_col = None
    for col in df.columns:
        low = str(col).lower()
        if "date" in low or "time" in low or "datetime" in low:
            date_col = col
            break
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if parsed.notna().mean() > 0.9:
            df = df.copy()
            df[date_col] = parsed
            df = df.set_index(date_col).sort_index()
            return df
    df = df.copy()
    df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    return df

def infer_nav_columns(df: pd.DataFrame):
    likely_date_cols = []
    for col in df.columns:
        low = str(col).lower()
        if "date" in low or "time" in low or "datetime" in low:
            likely_date_cols.append(col)
    candidates = [c for c in df.columns if c not in likely_date_cols and pd.api.types.is_numeric_dtype(df[c])]
    if len(candidates) < 2:
        for col in df.columns:
            if col not in likely_date_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        candidates = [c for c in df.columns if c not in likely_date_cols and pd.api.types.is_numeric_dtype(df[c])]
    if len(candidates) < 2:
        raise ValueError("Need at least two numeric NAV columns.")
    return candidates[0], candidates[1]

def compute_drawdown_stats(nav: pd.Series):
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    mdd = dd.min()
    new_peak = nav == running_max
    durations = (~new_peak).astype(int)
    durations = durations.groupby((new_peak).cumsum()).cumsum()
    max_duration = int(durations.max()) if len(durations) else 0
    return float(mdd), max_duration

def compute_metrics(nav: pd.Series, name: str, trading_days: int, rf_annual: float = 0.0):
    nav = nav.dropna().astype(float)
    if nav.iloc[0] <= 0:
        raise ValueError(f"First NAV value for {name} must be positive.")
    rets = nav.pct_change().dropna()
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_days = len(nav) - 1
    if n_days <= 0:
        ann_return = np.nan
    else:
        ann_return = (1.0 + total_return) ** (trading_days / n_days) - 1.0
    ann_vol = rets.std(ddof=1) * math.sqrt(trading_days) if len(rets) > 1 else np.nan
    sharpe = (ann_return - rf_annual) / ann_vol if (ann_vol and not np.isnan(ann_vol) and ann_vol != 0) else np.nan
    mdd, mdd_dur = compute_drawdown_stats(nav)
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan
    up_days = (rets > 0).sum()
    down_days = (rets < 0).sum()
    return {
        "name": name,
        "start_date": str(nav.index.min().date()) if hasattr(nav.index.min(), "date") else str(nav.index.min()),
        "end_date": str(nav.index.max().date()) if hasattr(nav.index.max(), "date") else str(nav.index.max()),
        "obs_days": int(n_days + 1),
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol) if not np.isnan(ann_vol) else np.nan,
        "sharpe_ratio_rf0": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": float(mdd),
        "max_drawdown_duration_days": int(mdd_dur),
        "calmar_ratio": float(calmar) if not np.isnan(calmar) else np.nan,
        "positive_days_pct": float(up_days / len(rets)) if len(rets) else np.nan,
        "negative_days_pct": float(down_days / len(rets)) if len(rets) else np.nan,
    }

def main(args):
    raw = pd.read_csv(args.csv)
    df = maybe_parse_dates(raw)
    col_a, col_b = infer_nav_columns(df)
    nav_df = df[[col_a, col_b]].copy()
    nav_df.columns = ["Strategy_A", "Strategy_B"]
    nav_df = nav_df.dropna()
    nav_df = nav_df[(nav_df > 0).all(axis=1)]
    metrics_a = compute_metrics(nav_df["Strategy_A"], "Strategy_A", args.days, args.rf)
    metrics_b = compute_metrics(nav_df["Strategy_B"], "Strategy_B", args.days, args.rf)
    metrics_df = pd.DataFrame([metrics_a, metrics_b])
    metrics_df.to_csv("NetValue_metrics.csv", index=False)
    # Plot (normalized for comparability)
    nav_plot = nav_df / nav_df.iloc[0]
    plt.figure(figsize=(9,5))
    plt.plot(nav_plot.index, nav_plot["Strategy_A"], label="Strategy_A")
    plt.plot(nav_plot.index, nav_plot["Strategy_B"], label="Strategy_B")
    plt.title("Net Value (Normalized to 1.0 at start)")
    plt.xlabel("Date")
    plt.ylabel("Net Value (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("NetValue_plot.png", dpi=150)
    plt.close()
    # Print correlation
    rets = nav_df.pct_change().dropna()
    if rets.shape[1] == 2:
        print("Daily return correlation:", rets.corr().iloc[0, 1])
    else:
        print("Daily return correlation: N/A")
    print("Done. Outputs: NetValue_metrics.csv, NetValue_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to NetValue.csv")
    parser.add_argument("--days", type=int, default=365, help="Trading days per year")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate (decimal)")
    args = parser.parse_args()
    main(args)
