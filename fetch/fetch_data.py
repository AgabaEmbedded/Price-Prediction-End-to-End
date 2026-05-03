"""
data/fetch_data.py
──────────────────
Fetches EUR/USD OHLCV data from yfinance, preprocesses it,
and creates the 5-class directional labels.

Labels:
  0 → Strong Sell  (return < -strong_threshold)
  1 → Sell         (return < -weak_threshold)
  2 → Hold         (|return| <= weak_threshold)
  3 → Buy          (return >  weak_threshold)
  4 → Strong Buy   (return >  strong_threshold)

Run:
    python data/fetch_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from datetime import datetime

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


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV from yfinance."""
    log.info(f"Downloading {ticker} from {start} to {end or 'today'}")
    end = end or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check your internet connection.")

    # yfinance returns MultiIndex columns when downloading single ticker in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.sort_index()

    log.info(f"Fetched {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── Label creation ────────────────────────────────────────────────────────────

def make_labels(
    df: pd.DataFrame,
    strong_threshold: float = 0.005,
    weak_threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Compute next-day log return and map to 5-class label.

    We predict the NEXT day's direction, so labels are shifted by -1
    (today's features → tomorrow's outcome).
    """
    # Log return of next trading day close vs today's close
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["next_log_return"] = df["log_return"].shift(-1)   # target

    def _classify(r: float) -> int:
        if r > strong_threshold:
            return 4   # Strong Buy
        elif r > weak_threshold:
            return 3   # Buy
        elif r < -strong_threshold:
            return 0   # Strong Sell
        elif r < -weak_threshold:
            return 1   # Sell
        else:
            return 2   # Hold

    df["label"] = df["next_log_return"].apply(_classify)

    # Distribution summary
    dist = df["label"].value_counts().sort_index()
    names = {0: "StrongSell", 1: "Sell", 2: "Hold", 3: "Buy", 4: "StrongBuy"}
    log.info("Label distribution:")
    for k, v in dist.items():
        log.info(f"  {names[k]:>10}  ({k}): {v:5d}  ({100*v/len(df):.1f}%)")

    return df


# ── Basic preprocessing ───────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw OHLCV + label dataframe.
    - Drop weekends / non-trading rows (already handled by yfinance)
    - Forward-fill any remaining gaps (holidays etc.)
    - Drop the last row (no next-day label available)
    - Sanity checks
    """
    # Forward fill (handles holidays where market was closed mid-week)
    df = df.ffill()

    # Drop the last row — its label (next_log_return) is NaN
    df = df.dropna(subset=["next_log_return", "label"])

    # Sanity checks
    assert (df["high"] >= df["low"]).all(), "High < Low found in data!"
    assert (df["close"] > 0).all(), "Non-positive close prices found!"
    assert df["label"].between(0, 4).all(), "Labels outside [0,4] range!"

    log.info(f"After preprocessing: {len(df)} rows remain")
    return df


# ── Train / Val / Test split ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split — NO shuffle.
    This is critical for time series to avoid data leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    log.info(
        f"Split → Train: {len(train)} ({train.index[0].date()}→{train.index[-1].date()})  "
        f"Val: {len(val)} ({val.index[0].date()}→{val.index[-1].date()})  "
        f"Test: {len(test)} ({test.index[0].date()}→{test.index[-1].date()})"
    )
    return train, val, test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    dc = cfg["data"]

    # Paths
    raw_path = Path(dc["raw_path"])
    proc_path = Path(dc["processed_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetch
    df = fetch_ohlcv(dc["ticker"], dc["start_date"], dc["end_date"])
    df.to_parquet(raw_path)
    log.info(f"Raw data saved → {raw_path}")

    # 2. Label
    df = make_labels(df, dc["strong_threshold"], dc["weak_threshold"])

    # 3. Preprocess
    df = preprocess(df)

    # 4. Save processed (with labels, without features — features added later)
    df.to_parquet(proc_path)
    log.info(f"Processed data saved → {proc_path}")

    # 5. Quick split preview
    train, val, test = chronological_split(df, dc["train_ratio"], dc["val_ratio"])

    # 6. Print a quick summary
    print("\n" + "="*60)
    print("  DATA SUMMARY")
    print("="*60)
    print(df[["open", "high", "low", "close", "volume", "log_return", "label"]].describe().round(4))
    print("="*60)
    print("Done! Next step → python features/feature_engineering.py")


if __name__ == "__main__":
    main()
