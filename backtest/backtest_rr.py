"""
backtest/backtest_rr.py
───────────────────────
Tests model predictions on a 1:2 risk-to-reward ratio.

Prediction classes:
  0 = Sell  → open SHORT trade
  1 = Hold  → skip the day, no trade placed
  2 = Buy   → open LONG trade

Logic per trading day:
  - Prediction made on day D is for day D+1's price action
  - Hold (1) predictions are skipped — balance unchanged
  - On Buy/Sell days, enter at D+1's open price
  - Stop Loss  = 1× ATR from entry  (risk)
  - Take Profit = 2× ATR from entry  (reward)
  - Walk through 1h bars from MT5 to check which is hit first: TP or SL
  - If neither is hit by end of day, close at daily close (partial P&L)

Account:
  - Starting balance: $200
  - Lot size: 0.01 (micro lot)
  - For major forex pairs: 1 pip = $0.10 on 0.01 lot

Run:
    python backtest/backtest_rr.py --predictions predictions/eurusd_predictions.csv --ticker EURUSD

    # Run all pairs using paths from config:
    python backtest/backtest_rr.py --all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MT5 data fetch — intraday 1h bars for the backtest period
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mt5_intraday(symbol: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    Fetch 1h OHLCV bars from MetaTrader 5 for a date range.
    Returns DataFrame with columns: open, high, low, close, volume
    indexed by UTC datetime.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError(
            "MetaTrader5 package not installed. Run: pip install MetaTrader5"
        )

    cfg = load_config()
    terminal_path = cfg.get("trade", {}).get("IC_MT5_PATH", None)

    if not mt5.initialize(path=terminal_path if terminal_path else None):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    # MT5 symbol for forex (no =X suffix, use broker format e.g. "EURUSD")
    mt5_symbol = symbol.upper().replace("=X", "").replace("/", "")

    import MetaTrader5 as mt5
    rates = mt5.copy_rates_range(
        mt5_symbol,
        mt5.TIMEFRAME_H1,
        date_from,
        date_to + timedelta(days=1),   # inclusive
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(
            f"No MT5 data returned for {mt5_symbol}. "
            "Check that MT5 is running and the symbol is available."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "tick_volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]]

    # Drop weekend bars (broker sometimes sends Sunday open)
    df = df[df.index.dayofweek < 5]

    log.info(f"MT5 [{mt5_symbol}]: {len(df)} 1h bars  {df.index[0]} → {df.index[-1]}")
    return df


def fetch_mt5_daily(symbol: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from MT5 to get ATR for SL/TP sizing.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 package not installed.")

    cfg = load_config()
    terminal_path = cfg.get("trade", {}).get("IC_MT5_PATH", None)

    if not mt5.initialize(path=terminal_path if terminal_path else None):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    mt5_symbol = symbol.upper().replace("=X", "").replace("/", "")

    rates = mt5.copy_rates_range(
        mt5_symbol,
        mt5.TIMEFRAME_D1,
        date_from - timedelta(days=30),  # extra history for ATR
        date_to + timedelta(days=1),
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError(f"No MT5 daily data for {mt5_symbol}.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df.index.dayofweek < 5]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ATR calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(daily_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute 14-period ATR on daily bars. Index = date."""
    h, l, c = daily_df["high"], daily_df["low"], daily_df["close"]
    prev_c  = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    # Normalise index to date only for clean merging
    atr.index = atr.index.normalize()
    return atr


# ─────────────────────────────────────────────────────────────────────────────
# Pip value per lot
# ─────────────────────────────────────────────────────────────────────────────

# For 0.01 lot (micro lot), value per pip in USD
# Pip = 0.0001 for most pairs, 0.01 for JPY pairs
PIP_SIZE = {
    "EURUSD": 0.0001, "EURGBP": 0.0001, "GBPUSD": 0.0001,
    "USDJPY": 0.01,   "USDCAD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCHF": 0.0001,
}
# USD value per pip per 0.01 lot
PIP_VALUE_PER_MICROLOT = {
    "EURUSD": 0.10, "EURGBP": 0.10, "GBPUSD": 0.10,
    "USDJPY": 0.10, "USDCAD": 0.10, "AUDUSD": 0.10,
    "NZDUSD": 0.10, "USDCHF": 0.10,
}

def price_to_usd(symbol: str, price_move: float, lot_size: float = 0.01) -> float:
    """
    Convert a price move (in quote currency units) to USD P&L.
    For non-USD quote pairs this is approximate without a live rate.
    """
    sym = symbol.upper().replace("=X", "").replace("/", "")
    pip = PIP_SIZE.get(sym, 0.0001)
    pv  = PIP_VALUE_PER_MICROLOT.get(sym, 0.10)
    pips = price_move / pip
    return pips * pv * (lot_size / 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Single-day trade simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_day(
    direction:   int,        # 2=Buy, 0=Sell  (1=Hold should never reach here)
    trade_date:  pd.Timestamp,
    hourly_df:   pd.DataFrame,
    entry_price: float,
    sl_distance: float,      # in price units (1× ATR)
    tp_distance: float,      # in price units (2× ATR)
    symbol:      str,
    lot_size:    float,
) -> dict:
    """
    Walk through the 1h bars of trade_date, starting from the open,
    and check whether TP or SL is hit first.

    Labels:
      2 = Buy  → LONG trade
      0 = Sell → SHORT trade
      1 = Hold → caller must skip before calling this function

    Returns a dict with keys:
        outcome   : "TP" | "SL" | "EOD" (end of day, neither hit)
        pnl_price : raw price move (+ or -)
        pnl_usd   : converted to USD
        exit_price: price at exit
        exit_time : timestamp of exit bar
        bars_held : number of 1h bars in trade
    """
    is_buy = (direction == 2)   # 2=Buy, 0=Sell

    if is_buy:
        tp_price = entry_price + tp_distance
        sl_price = entry_price - sl_distance
    else:
        tp_price = entry_price - tp_distance
        sl_price = entry_price + sl_distance

    # Get all 1h bars for this calendar day
    day_bars = hourly_df[
        hourly_df.index.normalize() == trade_date.normalize()
    ]

    if day_bars.empty:
        return {
            "outcome": "NO_DATA", "pnl_price": 0.0,
            "pnl_usd": 0.0, "exit_price": entry_price,
            "exit_time": trade_date, "bars_held": 0,
        }

    for i, (ts, bar) in enumerate(day_bars.iterrows()):
        # Check if TP or SL is touched within this bar's range
        if is_buy:
            tp_hit = bar["high"] >= tp_price
            sl_hit = bar["low"]  <= sl_price
        else:
            tp_hit = bar["low"]  <= tp_price
            sl_hit = bar["high"] >= sl_price

        # Both hit in same bar → conservative: assume SL hit first
        # (worst-case assumption — realistic for risk management)
        if sl_hit and tp_hit:
            sl_hit = True
            tp_hit = False

        if tp_hit:
            pnl_price = tp_distance if is_buy else -(-tp_distance)
            pnl_price = tp_distance  # always positive at TP
            return {
                "outcome":    "TP",
                "pnl_price":  tp_distance,
                "pnl_usd":    price_to_usd(symbol, tp_distance, lot_size),
                "exit_price": tp_price,
                "exit_time":  ts,
                "bars_held":  i + 1,
            }

        if sl_hit:
            return {
                "outcome":    "SL",
                "pnl_price":  -sl_distance,
                "pnl_usd":    -price_to_usd(symbol, sl_distance, lot_size),
                "exit_price": sl_price,
                "exit_time":  ts,
                "bars_held":  i + 1,
            }

    # End of day — neither hit: close at last bar's close
    last_bar   = day_bars.iloc[-1]
    eod_price  = last_bar["close"]
    pnl_price  = (eod_price - entry_price) if is_buy else (entry_price - eod_price)

    return {
        "outcome":    "EOD",
        "pnl_price":  pnl_price,
        "pnl_usd":    price_to_usd(symbol, abs(pnl_price), lot_size) * np.sign(pnl_price),
        "exit_price": eod_price,
        "exit_time":  day_bars.index[-1],
        "bars_held":  len(day_bars),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    predictions_path: str,
    symbol:           str,
    starting_balance: float = 200.0,
    lot_size:         float = 0.01,
    atr_period:       int   = 14,
    atr_sl_mult:      float = 1.0,   # SL = 1× ATR
    atr_tp_mult:      float = 2.0,   # TP = 2× ATR  → 1:2 RR
    output_dir:       str   = "backtest_results",
) -> pd.DataFrame:
    """
    Run the full backtest and return a trade log DataFrame.
    """
    out_dir = Path(output_dir) / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────────
    pred_df = pd.read_csv(predictions_path)

    # Normalise column names (handle variations)
    pred_df.columns = [c.lower().strip() for c in pred_df.columns]
    if "date" not in pred_df.columns:
        # Try to use the first column as date
        pred_df = pred_df.rename(columns={pred_df.columns[0]: "date"})
    # AFTER — make everything UTC-aware from the start
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.tz_localize("UTC")
    pred_df = pred_df.sort_values("date").reset_index(drop=True)
    pred_df["trade_date"] = pred_df["date"].shift(-1)
    pred_df = pred_df.dropna(subset=["trade_date"]).copy()
    pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"])#.dt.tz_localize("UTC")

    log.info(f"Loaded {len(pred_df)} predictions  "
             f"{pred_df['date'].min().date()} → {pred_df['date'].max().date()}")

    date_from = pred_df["trade_date"].min().to_pydatetime()
    date_to   = pred_df["trade_date"].max().to_pydatetime()

    # ── Fetch MT5 data ────────────────────────────────────────────────────────
    log.info("Fetching MT5 daily bars for ATR...")
    daily_df  = fetch_mt5_daily(symbol, date_from, date_to)
    atr_series = compute_atr(daily_df, atr_period)
    

    log.info("Fetching MT5 1h bars for intraday simulation...")
    hourly_df = fetch_mt5_intraday(symbol, date_from, date_to)

    # ── Trade loop ────────────────────────────────────────────────────────────
    balance    = starting_balance
    trade_log  = []

    for _, row in pred_df.iterrows():
        trade_date = row["date"]
        direction  = int(row["predicted"])

        # ── Skip Hold days — no trade placed, balance unchanged ───────────────
        if direction == 1:
            log.info(f"  {trade_date.date()}  HOLD   —  skipping")
            continue

        # Get the daily bar for the trade date (for entry price)
        trade_day_daily = daily_df[
            daily_df.index.normalize() == trade_date.normalize()
        ]

        if trade_day_daily.empty:
            log.warning(f"No daily bar for {trade_date.date()} — skipping.")
            continue

        entry_price = float(trade_day_daily.iloc[0]["open"])

        # ATR from the previous daily close (don't use future ATR)
        prev_atr_dates = atr_series[atr_series.index < trade_date.normalize()]
        if prev_atr_dates.empty:
            log.warning(f"No ATR available before {trade_date.date()} — skipping.")
            continue
        atr_value = float(prev_atr_dates.iloc[-1])

        sl_distance = atr_value * atr_sl_mult
        tp_distance = atr_value * atr_tp_mult

        # Simulate the day
        result = simulate_day(
            direction=direction,
            trade_date=trade_date,
            hourly_df=hourly_df,
            entry_price=entry_price,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            symbol=symbol,
            lot_size=lot_size,
        )

        balance += result["pnl_usd"]

        # direction string: 2=Buy, 0=Sell
        dir_str = "BUY" if direction == 2 else "SELL"

        trade_log.append({
            "signal_date":  row["date"].date(),
            "trade_date":   trade_date.date(),
            "direction":    dir_str,
            "actual":       int(row.get("actual", -1)),
            "predicted":    direction,
            "correct":      int(row.get("actual", -1)) == direction,
            "entry_price":  round(entry_price, 5),
            "sl_price":     round(
                entry_price - sl_distance if direction == 2 else entry_price + sl_distance, 5
            ),
            "tp_price":     round(
                entry_price + tp_distance if direction == 2 else entry_price - tp_distance, 5
            ),
            "exit_price":   round(result["exit_price"], 5),
            "exit_time":    result["exit_time"],
            "outcome":      result["outcome"],
            "atr":          round(atr_value, 5),
            "sl_distance":  round(sl_distance, 5),
            "tp_distance":  round(tp_distance, 5),
            "pnl_price":    round(result["pnl_price"], 5),
            "pnl_usd":      round(result["pnl_usd"], 2),
            "balance":      round(balance, 2),
            "bars_held":    result["bars_held"],
        })

        log.info(
            f"  {trade_date.date()}  {dir_str:<4}  "
            f"{result['outcome']:<6}  P&L: ${result['pnl_usd']:+.2f}  "
            f"Balance: ${balance:.2f}"
        )

    trade_df = pd.DataFrame(trade_log)
    total_predictions = len(pred_df)   # includes Hold days

    # ── Save trade log ────────────────────────────────────────────────────────
    log_path = out_dir / "trade_log.csv"
    trade_df.to_csv(log_path, index=False)
    log.info(f"Trade log saved → {log_path}")

    hold_count = (pred_df["predicted"] == 1).sum()
    log.info(f"Summary: {total_predictions} signals — "
             f"{hold_count} Hold (skipped), {len(trade_df)} trades placed")

    return trade_df, total_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(trade_df: pd.DataFrame, symbol: str, starting_balance: float,
                total_predictions: int = 0):
    if trade_df.empty:
        print("No trades to summarise.")
        return

    total_trades  = len(trade_df)
    tp_trades     = (trade_df["outcome"] == "TP").sum()
    sl_trades     = (trade_df["outcome"] == "SL").sum()
    eod_trades    = (trade_df["outcome"] == "EOD").sum()
    no_data       = (trade_df["outcome"] == "NO_DATA").sum()
    buy_trades    = (trade_df["direction"] == "BUY").sum()
    sell_trades   = (trade_df["direction"] == "SELL").sum()
    held_days     = total_predictions - total_trades if total_predictions else "n/a"

    win_rate      = tp_trades / total_trades * 100 if total_trades else 0
    total_pnl     = trade_df["pnl_usd"].sum()
    final_bal     = trade_df["balance"].iloc[-1]
    max_balance   = trade_df["balance"].max()
    drawdown      = ((trade_df["balance"] - trade_df["balance"].cummax()) /
                     trade_df["balance"].cummax() * 100).min()

    gross_profit  = trade_df[trade_df["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_loss    = trade_df[trade_df["pnl_usd"] < 0]["pnl_usd"].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    correct_direction = trade_df["correct"].mean() * 100 if "correct" in trade_df.columns else None

    print("\n" + "═"*55)
    print(f"  BACKTEST RESULTS  —  {symbol.upper()}")
    print("═"*55)
    print(f"  Starting balance : ${starting_balance:.2f}")
    print(f"  Final balance    : ${final_bal:.2f}")
    print(f"  Total P&L        : ${total_pnl:+.2f}")
    print(f"  Return           : {(final_bal/starting_balance - 1)*100:+.1f}%")
    print()
    print(f"  Total signals    : {total_predictions or total_trades}")
    print(f"  Hold (skipped)   : {held_days}")
    print(f"  Trades placed    : {total_trades}")
    print(f"    BUY trades     : {buy_trades}")
    print(f"    SELL trades    : {sell_trades}")
    print()
    print(f"  TP hits          : {tp_trades}  ({tp_trades/total_trades*100:.1f}%)")
    print(f"  SL hits          : {sl_trades}  ({sl_trades/total_trades*100:.1f}%)")
    print(f"  EOD closes       : {eod_trades}  ({eod_trades/total_trades*100:.1f}%)")
    if no_data:
        print(f"  Skipped (no data): {no_data}")
    print()
    print(f"  TP Win rate      : {win_rate:.1f}%")
    if correct_direction is not None:
        print(f"  Direction acc.   : {correct_direction:.1f}%")
    print(f"  Profit factor    : {profit_factor:.2f}")
    print(f"  Max drawdown     : {drawdown:.1f}%")
    print(f"  Peak balance     : ${max_balance:.2f}")
    print("═"*55)

    # Break-even win rate at 1:2 RR = 33.3%
    print(f"\n  ℹ️  Break-even win rate at 1:2 RR = 33.3%")
    if win_rate >= 33.3:
        print(f"  ✅  Win rate {win_rate:.1f}% is ABOVE break-even")
    else:
        print(f"  ❌  Win rate {win_rate:.1f}% is BELOW break-even")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(trade_df: pd.DataFrame, symbol: str,
                 starting_balance: float, output_dir: str):
    if trade_df.empty:
        return

    out_dir = Path(output_dir) / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    dates   = pd.to_datetime(trade_df["trade_date"])
    balance = trade_df["balance"].values
    pnl     = trade_df["pnl_usd"].values

    # Colour map per outcome
    colour_map = {"TP": "#2ecc71", "SL": "#e74c3c", "EOD": "#3498db", "NO_DATA": "#95a5a6"}
    bar_colours = [colour_map.get(o, "#95a5a6") for o in trade_df["outcome"]]

    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#3a3d4d",
        "axes.labelcolor":  "#c8c8d0",
        "xtick.color":      "#c8c8d0",
        "ytick.color":      "#c8c8d0",
        "text.color":       "#e8e8f0",
        "grid.color":       "#2a2d3d",
        "grid.alpha":       0.5,
        "lines.linewidth":  1.8,
    })

    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                              gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.suptitle(f"{symbol.upper()}  —  Backtest  (1:2 RR, 0.01 lot, ATR-based SL/TP)",
                 fontsize=13, y=0.98)

    # ── Panel 1: Equity curve ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, balance, color="#2ecc71", linewidth=2, label="Balance", zorder=3)
    ax1.axhline(starting_balance, color="#7f8c8d", linestyle="--",
                linewidth=1, label=f"Start ${starting_balance:.0f}")
    ax1.fill_between(dates, starting_balance, balance,
                     where=(np.array(balance) >= starting_balance),
                     alpha=0.15, color="#2ecc71")
    ax1.fill_between(dates, starting_balance, balance,
                     where=(np.array(balance) < starting_balance),
                     alpha=0.15, color="#e74c3c")

    # Mark TP / SL on equity curve
    for outcome, colour, marker, label in [
        ("TP", "#2ecc71", "^", "TP hit"),
        ("SL", "#e74c3c", "v", "SL hit"),
        ("EOD", "#3498db", "o", "EOD close"),
    ]:
        mask = trade_df["outcome"] == outcome
        if mask.any():
            ax1.scatter(dates[mask], balance[mask],
                        c=colour, marker=marker, s=40,
                        zorder=5, label=label, alpha=0.8)

    ax1.set_ylabel("Account Balance ($)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.grid(True)

    # Drawdown shading
    running_max = np.maximum.accumulate(balance)
    drawdown    = (balance - running_max) / running_max * 100
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(dates, drawdown, 0, alpha=0.2, color="#e74c3c")
    ax1_twin.set_ylabel("Drawdown %", fontsize=9, color="#e74c3c")
    ax1_twin.tick_params(axis="y", colors="#e74c3c")
    ax1_twin.set_ylim(min(drawdown) * 3, 5)

    # ── Panel 2: Per-trade P&L bars ────────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(dates, pnl, color=bar_colours, width=0.7, alpha=0.85)
    ax2.axhline(0, color="#7f8c8d", linewidth=0.8)
    ax2.set_ylabel("Trade P&L ($)", fontsize=11)
    ax2.grid(axis="y")

    # Legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#2ecc71", label="TP hit"),
        Patch(color="#e74c3c", label="SL hit"),
        Patch(color="#3498db", label="EOD close"),
    ]
    ax2.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.3)

    # ── Panel 3: Rolling win rate (over trades placed, not calendar days) ──
    ax3 = axes[2]
    is_win = (trade_df["outcome"] == "TP").astype(float)
    rolling_wr = is_win.rolling(20, min_periods=5).mean() * 100
    ax3.plot(dates, rolling_wr, color="#3498db", linewidth=1.5,
             label="20-trade rolling win rate")
    ax3.axhline(33.3, color="#e74c3c", linestyle="--", linewidth=1,
                label="Break-even (33.3%)")
    ax3.axhline(50.0, color="#2ecc71", linestyle=":", linewidth=1,
                label="50% reference")
    ax3.set_ylabel("Win Rate %", fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax3.grid(True)

    # Annotate Hold % in title
    hold_pct = 0
    ax3.set_xlabel(
        f"(Hold days skipped — only trade days shown on x-axis)",
        fontsize=8, color="#7f8c8d"
    )

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = out_dir / "backtest_equity_curve.png"
    fig.savefig(plot_path, bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    log.info(f"Plot saved → {plot_path}")

    # ── Monthly P&L heatmap ────────────────────────────────────────────────
    _plot_monthly_pnl(trade_df, symbol, out_dir)


def _plot_monthly_pnl(trade_df: pd.DataFrame, symbol: str, out_dir: Path):
    df = trade_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["year"]  = df["trade_date"].dt.year
    df["month"] = df["trade_date"].dt.month

    monthly = df.groupby(["year", "month"])["pnl_usd"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, max(3, len(monthly) * 0.7)))

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#e74c3c", "#1a1d27", "#2ecc71"]
    )

    vmax = abs(monthly.values).max() or 1
    im   = ax.imshow(monthly.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="P&L ($)")

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels([month_names[m-1] for m in monthly.columns], fontsize=9)
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index, fontsize=9)

    for i in range(monthly.shape[0]):
        for j in range(monthly.shape[1]):
            val = monthly.values[i, j]
            ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > vmax*0.4 else "#c8c8d0")

    ax.set_title(f"{symbol.upper()} — Monthly P&L ($)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "monthly_pnl_heatmap.png"
    fig.savefig(path, bbox_inches="tight", facecolor="#0f1117", dpi=150)
    plt.close(fig)
    log.info(f"Monthly heatmap saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=False,
                        help="Path to predictions CSV (date, actual, predicted)")
    parser.add_argument("--ticker", required=False,
                        help="Symbol e.g. EURUSD or EURUSD=X")
    parser.add_argument("--all", action="store_true",
                        help="Run backtest for all tickers in config")
    parser.add_argument("--balance",  type=float, default=200.0,
                        help="Starting balance in USD (default: 200)")
    parser.add_argument("--lot",      type=float, default=0.01,
                        help="Lot size (default: 0.01 micro lot)")
    parser.add_argument("--atr-sl",   type=float, default=0.25,
                        help="ATR multiplier for Stop Loss (default: 1.0)")
    parser.add_argument("--atr-tp",   type=float, default=0.5,
                        help="ATR multiplier for Take Profit (default: 2.0 → 1:2 RR)")
    parser.add_argument("--output",   default="backtest_results",
                        help="Output directory (default: backtest_results/)")
    args = parser.parse_args()

    cfg = load_config()

    if args.all:
        tickers = cfg["data"]["tickers"]
        for ticker in tickers:
            ticker_id = ticker.lower().split("=")[0]
            pred_path = Path(cfg["output"]["predictions_dir"]) / ticker_id / "test_predictions.csv"
            if not pred_path.exists():
                log.warning(f"No predictions found for {ticker_id} at {pred_path} — skipping.")
                continue
            try:
                trade_df, total_predictions = run_backtest(
                    predictions_path=str(pred_path),
                    symbol=ticker_id,
                    starting_balance=args.balance,
                    lot_size=args.lot,
                    atr_sl_mult=args.atr_sl,
                    atr_tp_mult=args.atr_tp,
                    output_dir=args.output,
                )
                print_stats(trade_df, ticker_id, args.balance, total_predictions)
                plot_results(trade_df, ticker_id, args.balance, args.output)
            except Exception as e:
                log.error(f"Backtest failed for {ticker_id}: {e}")

    else:
        if not args.predictions or not args.ticker:
            parser.error("Provide --predictions and --ticker, or use --all")

        symbol = args.ticker.upper().replace("=X", "").replace("/", "")
        trade_df, total_predictions = run_backtest(
            predictions_path=args.predictions,
            symbol=symbol,
            starting_balance=args.balance,
            lot_size=args.lot,
            atr_sl_mult=args.atr_sl,
            atr_tp_mult=args.atr_tp,
            output_dir=args.output,
        )
        print_stats(trade_df, symbol, args.balance, total_predictions)
        plot_results(trade_df, symbol, args.balance, args.output)


if __name__ == "__main__":
    main()