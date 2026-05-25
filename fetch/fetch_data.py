"""
data/fetch_data.py
──────────────────
Fetches OHLCV data from yfinance for any supported timeframe,
preprocesses it, and creates the 5-class directional labels.

yfinance hourly data limits (free, no API key needed):
  - 1h  : up to 730 days history
  - 1d  : unlimited history (2000+)
  - 4h  : not natively supported by yfinance — use "1h" and resample

Labels:
  0 → Strong Sell  (next return < -strong_threshold)
  1 → Sell         (next return < -weak_threshold)
  2 → Hold         (|next return| <= weak_threshold)
  3 → Buy          (next return >  weak_threshold)
  4 → Strong Buy   (next return >  strong_threshold)

Run:
    python data/fetch_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Timeframe helpers ─────────────────────────────────────────────────────────

# yfinance interval strings and their max history in days
_YF_INTERVALS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,   # alias for 1h
    "1h":  730,
    "90m": 60,
    "1d":  None,  # unlimited
    "5d":  None,
    "1wk": None,
    "1mo": None,
}

def _resolve_interval(timeframe: str) -> str:
    """Map config timeframe string to yfinance interval string."""
    mapping = {
        "1h":  "1h",
        "4h":  "1h",   # fetch 1h then resample to 4h
        "1d":  "1d",
        "15m": "15m",
        "5m":  "5m",
        "1m":  "1m",
    }
    return mapping.get(timeframe.lower(), timeframe.lower())

def _max_history_days(yf_interval: str) -> int | None:
    """Return the maximum lookback in days for a given yfinance interval."""
    return _YF_INTERVALS.get(yf_interval)

def _clamp_start(start_str: str | None, yf_interval: str) -> str:
    """
    yfinance enforces hard history limits for intraday intervals.
    Clamp the requested start date to stay within the allowed window,
    with a small buffer to avoid edge-case API rejections.
    """
    max_days = _max_history_days(yf_interval)
    today    = datetime.today()

    if max_days is not None:
        earliest_allowed = today - timedelta(days=max_days - 2)
        if start_str is None:
            return earliest_allowed.strftime("%Y-%m-%d")
        requested = datetime.strptime(start_str, "%Y-%m-%d")
        if requested < earliest_allowed:
            clamped = earliest_allowed.strftime("%Y-%m-%d")
            log.warning(
                f"Requested start {start_str} is beyond yfinance's {max_days}-day "
                f"limit for '{yf_interval}' interval. Clamping to {clamped}."
            )
            return clamped
        return start_str

    return start_str or "2010-01-01"


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance.

    For 4h: fetches 1h bars and resamples to 4h (yfinance has no native 4h).
    For intraday intervals: start date is automatically clamped to the
    maximum allowed history window.
    """
    yf_interval = _resolve_interval(interval)
    start       = _clamp_start(start, yf_interval)
    end         = end or datetime.today().strftime("%Y-%m-%d")

    log.info(f"Downloading {ticker}  interval={interval}  {start} → {end}")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}' with interval='{yf_interval}'. "
            "Check the ticker symbol and your internet connection."
        )

    # Flatten MultiIndex columns (yfinance >= 0.2.x quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.sort_index()

    # Remove any rows with zero/negative close (bad data)
    bad = (df["close"] <= 0) | df["close"].isna()
    if bad.any():
        log.warning(f"Dropping {bad.sum()} rows with invalid close prices.")
        df = df[~bad]

    # Resample 1h → 4h if requested
    if interval.lower() == "4h":
        log.info("Resampling 1h bars to 4h...")
        df = (
            df.resample("4h")
            .agg({
                "open":   "first",
                "high":   "max",
                "low":    "min",
                "close":  "last",
                "volume": "sum",
            })
            .dropna()
        )

    log.info(
        f"Fetched {len(df)} bars  |  "
        f"{df.index[0]}  →  {df.index[-1]}"
    )
    return df


# ── Label creation ────────────────────────────────────────────────────────────

def make_labels(
    df: pd.DataFrame,
    strong_threshold: float = 0.005,
    weak_threshold: float   = 0.001,
) -> pd.DataFrame:
    """
    Compute next-bar log return and map to 5-class label.

    We predict the NEXT bar's direction, so labels are shifted by -1
    (current bar's features → next bar's outcome).
    """
    df = df.copy()
    df["log_return"]      = np.log(df["close"] / df["close"].shift(1))
    df["next_log_return"] = df["log_return"].shift(-1)

    def _classify(r: float) -> int:
        if   r >  strong_threshold: return 4   # Strong Buy
        elif r >  weak_threshold:   return 3   # Buy
        elif r < -strong_threshold: return 0   # Strong Sell
        elif r < -weak_threshold:   return 1   # Sell
        else:                       return 2   # Hold

    df["label"] = df["next_log_return"].apply(_classify)

    names = {0: "StrongSell", 1: "Sell", 2: "Hold", 3: "Buy", 4: "StrongBuy"}
    dist  = df["label"].value_counts().sort_index()
    log.info("Label distribution:")
    for k, cnt in dist.items():
        log.info(f"  {names[k]:>10} ({k}): {cnt:5d}  ({100*cnt/len(df):.1f}%)")

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw OHLCV + label dataframe.
    - Forward-fill any remaining gaps.
    - Drop the last bar (no next_log_return available).
    - Sanity checks.
    """
    df = df.ffill()
    df = df.dropna(subset=["next_log_return", "label"])

    assert (df["high"] >= df["low"]).all(),  "High < Low found in data!"
    assert (df["close"] > 0).all(),          "Non-positive close prices found!"
    assert df["label"].between(0, 4).all(),  "Labels outside [0, 4]!"

    log.info(f"After preprocessing: {len(df)} rows remain")
    return df


# ── Train / Val / Test split ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict chronological split — NO shuffle."""
    n         = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    log.info(
        f"Split →  Train: {len(train):,}  ({train.index[0]} → {train.index[-1]})\n"
        f"          Val:  {len(val):,}   ({val.index[0]} → {val.index[-1]})\n"
        f"          Test: {len(test):,}  ({test.index[0]} → {test.index[-1]})"
    )
    return train, val, test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    dc  = cfg["data"]

    raw_path  = Path(dc["raw_path"])
    proc_path = Path(dc["processed_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch
    df = fetch_ohlcv(
        ticker   = dc["ticker"],
        start    = dc.get("start_date"),
        end      = dc.get("end_date"),
        interval = dc.get("timeframe", "1d"),
    )
    df.to_parquet(raw_path)
    log.info(f"Raw data saved → {raw_path}")

    # 2. Label
    df = make_labels(df, dc["strong_threshold"], dc["weak_threshold"])

    # 3. Preprocess
    df = preprocess(df)

    # 4. Save processed
    df.to_parquet(proc_path)
    log.info(f"Processed data saved → {proc_path}")

    # 5. Split preview
    train, val, test = chronological_split(df, dc["train_ratio"], dc["val_ratio"])

    # 6. Summary
    print("\n" + "="*60)
    print("  DATA SUMMARY")
    print("="*60)
    print(f"  Ticker    : {dc['ticker']}")
    print(f"  Timeframe : {dc.get('timeframe', '1d')}")
    print(f"  Total bars: {len(df):,}")
    print(f"  Train     : {len(train):,}")
    print(f"  Val       : {len(val):,}")
    print(f"  Test      : {len(test):,}")
    print()
    print(df[["open", "high", "low", "close", "log_return", "label"]].describe().round(6))
    print("="*60)
    print("Done! Next step → python features/feature_engineering.py")


if __name__ == "__main__":
    main()