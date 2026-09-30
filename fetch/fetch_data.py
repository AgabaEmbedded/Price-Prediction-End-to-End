"""
data/fetch_data.py
──────────────────
Fetches OHLCV data from MT5 and creates 3-class labels aligned to the
1:2 RR backtest logic:

  0 → Sell  (next day achieves a 2× ATR downward move before 1× ATR up)
  1 → Hold  (next day achieves neither a 2× ATR move in either direction)
  2 → Buy   (next day achieves a 2× ATR upward move before 1× ATR down)

Why this matters:
  Previously the model predicted general direction (up/down) and we hoped
  that also implied a 2× ATR move — a weak connection. Now the model is
  trained to predict EXACTLY what the backtest needs: will the market
  achieve the TP (2× ATR) before the SL (1× ATR) tomorrow?

  Labels are computed by walking the next day's 1h intraday bars (same
  logic as the backtest) so training and live execution are consistent.

Run:
    python data/fetch_data.py
"""

from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import MetaTrader5 as mt5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

LABEL_NAMES = {0: "Sell", 1: "Hold", 2: "Buy"}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── MT5 fetch helpers ─────────────────────────────────────────────────────────

def _init_mt5(cfg: dict) -> bool:
    path = cfg.get("trade", {}).get("IC_MT5_PATH",
           r"C:\Program Files\MetaTrader 5\terminal64.exe")
    if not mt5.initialize(path=path):
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    return True


def fetch_mt5_ohlcv(
    symbol: str,
    start_date: str,
    end_date: datetime = None,
    timeframe=mt5.TIMEFRAME_D1,
    cfg: dict = None,
) -> pd.DataFrame:
    """Fetch OHLCV from MT5 terminal for any timeframe."""
    cfg = cfg or load_config()
    if not _init_mt5(cfg):
        return pd.DataFrame()

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = end_date if end_date is not None else datetime.now()

    log.info(f"Fetching MT5 [{symbol}]  {start_date} → {end_dt.strftime('%Y-%m-%d')}")
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        log.error(f"No data for '{symbol}'. Check symbol name in Market Watch.")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={
        "time":        "Date",
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    })
    df.set_index("Date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    log.info(f"  Pulled {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str | None,
    interval: str = "1d",
    cfg: dict = None,
) -> pd.DataFrame:
    """Fetch OHLCV from MT5, clean and standardise column names."""
    cfg = cfg or load_config()
    end = end or datetime.today().strftime("%Y-%m-%d")

    tf_map = {
        "1d": mt5.TIMEFRAME_D1,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "15m": mt5.TIMEFRAME_M15,
        "5m":  mt5.TIMEFRAME_M5,
    }
    mt5_tf = tf_map.get(interval.lower(), mt5.TIMEFRAME_D1)
    symbol = ticker.split("=")[0]

    df = fetch_mt5_ohlcv(
        symbol=symbol,
        start_date=start,
        end_date=datetime.strptime(end, "%Y-%m-%d"),
        timeframe=mt5_tf,
        cfg=cfg,
    )

    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'.")

    # Drop weekends (MT5 sometimes includes Sunday open bar)
    df = df[df.index.dayofweek < 5]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.sort_index()

    log.info(f"Fetched {len(df)} rows  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── ATR ───────────────────────────────────────────────────────────────────────

def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    14-period ATR on daily bars.
    Uses previous day's ATR to avoid lookahead (shift by 1).
    """
    h, l, c  = df["high"], df["low"], df["close"]
    prev_c   = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    # Shift by 1: today's label uses yesterday's ATR (known at bar open)
    return atr.shift(1)


# ── Label creation ────────────────────────────────────────────────────────────

def _label_one_day(
    next_day_1h: pd.DataFrame,
    entry_price: float,
    sl_dist: float,   # 1× ATR
    tp_dist: float,   # 2× ATR
) -> int:
    """
    Walk the next day's 1h bars and check which is hit first.

    Returns:
      2 = Buy  (TP hit first going up)
      0 = Sell (TP hit first going down)
      1 = Hold (neither TP hit by end of day)

    Conservative: if both hit in same bar, SL wins (same as backtest).
    """
    tp_up   = entry_price + tp_dist
    sl_up   = entry_price - sl_dist
    tp_down = entry_price - tp_dist
    sl_down = entry_price + sl_dist

    for _, bar in next_day_1h.iterrows():
        h, l = bar["high"], bar["low"]

        # ── Check Buy scenario: TP=tp_up, SL=sl_up
        buy_tp = h >= tp_up
        buy_sl = l <= sl_up

        # ── Check Sell scenario: TP=tp_down, SL=sl_down
        sell_tp = l <= tp_down
        sell_sl = h >= sl_down

        # Priority: if a TP is cleanly hit without its SL → label that direction
        # If both BUY_TP and SELL_TP hit in same bar → Hold (ambiguous spike)
        if buy_tp and sell_tp:
            return 1   # Hold — both TPs hit, ambiguous

        if buy_tp and not buy_sl:
            return 2   # Buy TP hit cleanly

        if sell_tp and not sell_sl:
            return 0   # Sell TP hit cleanly

        # Both SLs hit → Hold
        if buy_sl and sell_sl:
            return 1

        # Only one SL hit without TP → that direction failed → Hold
        if buy_sl and not buy_tp:
            # BUY scenario failed; but SELL might still play out in later bars
            # Don't return yet — continue checking for sell TP
            pass
        if sell_sl and not sell_tp:
            pass

    # End of day — neither TP reached
    return 1


def make_labels(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    atr_period: int   = 14,
    sl_mult:    float = 1.0,
    tp_mult:    float = 2.0,
) -> pd.DataFrame:
    """
    Create 3-class labels aligned with the 1:2 RR backtest:
      2 = Buy  — next day hits 2× ATR upward move before 1× ATR loss
      0 = Sell — next day hits 2× ATR downward move before 1× ATR loss
      1 = Hold — next day hits neither TP in either direction

    Requires intraday (1h) bars to walk through the next day bar-by-bar,
    matching exactly what the backtest does at execution time.
    """
    df   = daily_df.copy()
    atr  = compute_atr_series(df, atr_period)

    df["log_return"]      = np.log(df["close"] / df["close"].shift(1))
    df["atr"]             = atr
    df["next_log_return"] = df["log_return"].shift(-1)   # kept for reference

    labels = []
    dates  = df.index.tolist()

    log.info("Computing intraday-walk labels (this may take a moment)...")

    daily_bars = []

    for i in range(len(dates) - 1):
        today      = dates[i]
        next_day   = dates[i + 1]
        atr_val    = df.loc[today, "atr"]
        entry      = df.loc[next_day, "open"]   # enter at next day's open
        
        # Get 1h bars for next_day
        next_day_bars = hourly_df[
            hourly_df.index.normalize().date == next_day.date()
            if hasattr(hourly_df.index, 'normalize')
            else pd.Series(hourly_df.index).apply(lambda x: x.date()) == next_day.date()
        ]
        today_bars = hourly_df[
            hourly_df.index.normalize().date == today.date()
            if hasattr(hourly_df.index, 'normalize')
            else pd.Series(hourly_df.index).apply(lambda x: x.date()) == today.date()
        ]
        
        today_bars_close = today_bars["close"].values.T        
            
        if len(today_bars_close) < 24:
            log.warning(f"Today {today.date()} has {len(today_bars_close)} intraday bars (expected 24).")
            today_bars_close = np.pad(today_bars_close, (0, 24 - len(today_bars_close)), constant_values=np.nan)
        elif len(today_bars_close) > 24:
            log.warning(f"Today {today.date()} has {len(today_bars_close)} intraday bars (expected 24).")
            today_bars_close = today_bars_close[:24]

        daily_bars.append(today_bars_close)
        





        if pd.isna(atr_val) or atr_val <= 0 or entry <= 0 or next_day_bars.empty:
            labels.append(np.nan)
            continue

        sl_dist = atr_val * sl_mult
        tp_dist = atr_val * tp_mult

        label = _label_one_day(next_day_bars, entry, sl_dist, tp_dist)
        labels.append(label)
    daily_bars_arr = np.array(daily_bars)
    daily_bars_df = pd.DataFrame(daily_bars_arr, columns= [f"h{i}" for i in range(1, 25)], index=dates[:-1])
    df = pd.concat([df, daily_bars_df], axis=1)
    
    # Last row has no next day — NaN
    labels.append(np.nan)

    df["label"] = labels

    # Distribution summary
    dist = df["label"].dropna().value_counts().sort_index()
    log.info("Label distribution:")
    total = dist.sum()
    for k, v in dist.items():
        log.info(f"  {LABEL_NAMES[int(k)]:>5} ({int(k)}): {v:5d}  ({100*v/total:.1f}%)")

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.ffill()
    df = df.dropna(subset=["next_log_return", "label"])

    assert (df["high"] >= df["low"]).all(),      "High < Low found!"
    assert (df["close"] > 0).all(),              "Non-positive close prices!"
    assert df["label"].isin([0, 1, 2]).all(),    "Labels outside {0,1,2}!"

    log.info(f"After preprocessing: {len(df)} rows remain")
    return df


# ── Train / Val / Test split ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n         = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

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
    cfg  = load_config()
    dc   = cfg["data"]
    atr_period = cfg.get("features", {}).get("atr_period", 14)
    sl_mult    = cfg.get("backtest", {}).get("atr_sl_mult", 1.0)
    tp_mult    = cfg.get("backtest", {}).get("atr_tp_mult", 2.0)

    raw_path       = Path(dc["raw_path"])
    init_proc_path = Path(dc["processed_path"])
    tickers        = dc["tickers"]

    for ticker in tickers:
        ticker_id = ticker.lower().split("=")[0]
        raw_path  = Path(raw_path).with_name(f"{ticker_id}_raw.parquet")
        proc_path = Path(init_proc_path).with_name(f"{ticker_id}_processed.parquet")
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 1. Fetch daily bars ───────────────────────────────────────────────
        daily_df = fetch_ohlcv(
            ticker,
            dc["start_date"],
            dc["end_date"],
            interval=dc.get("timeframe", "1d"),
            cfg=cfg,
        )
        daily_df.to_parquet(raw_path)
        log.info(f"Raw daily data saved → {raw_path}")

        # ── 2. Fetch 1h bars for label computation ───────────────────────────
        # We need intraday bars to walk through each next day and check
        # which of TP/SL is hit first — matching the backtest logic exactly.
        log.info(f"Fetching 1h bars for intraday label computation...")
        hourly_df = fetch_ohlcv(
            ticker,
            dc["start_date"],
            dc["end_date"],
            interval="1h",
            cfg=cfg,
        )

        # ── 3. Label ─────────────────────────────────────────────────────────
        daily_df = make_labels(
            daily_df,
            hourly_df,
            atr_period=atr_period,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
        )

        # ── 4. Preprocess ─────────────────────────────────────────────────────
        daily_df = preprocess(daily_df)

        # ── 5. Save ───────────────────────────────────────────────────────────
        daily_df.to_parquet(proc_path)
        log.info(f"Processed {ticker_id} data saved → {proc_path}")

        # ── 6. Split preview ──────────────────────────────────────────────────
        train, val, test = chronological_split(daily_df, dc["train_ratio"], dc["val_ratio"])

        print("\n" + "="*60)
        print(f"{ticker_id.upper()}  DATA SUMMARY")
        print("="*60)
        print(daily_df[["open", "high", "low", "close", "log_return", "label"]].describe().round(6))
        print("="*60)

    print("Done! Next step → python features/feature_engineering.py")


if __name__ == "__main__":
    main()