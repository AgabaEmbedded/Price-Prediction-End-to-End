"""
features/feature_engineering.py
────────────────────────────────
Comprehensive time-series feature engineering — works correctly for BOTH
daily (1d) and intraday (1h, 4h) bar data.

Key improvements over v1 for intraday use:
  - All rolling windows scaled by bars_per_day so SMA-200 always means
    ~200 trading days of context regardless of timeframe.
  - Historical volatility annualisation uses correct bars_per_year scalar.
  - Hour-of-day and forex session features added (critical for 1h signals).
  - Intraday VWAP resets each calendar day (real intraday VWAP behaviour).
  - Calendar boundary flags (month_start etc.) fire once per day, not per bar.
  - Gap / overnight features gated behind is_intraday flag.
  - ROC periods expressed in calendar-equivalent days, not raw bar counts.

Feature groups:
  1.  Price action          (body/wick ratios, doji/spinning-top patterns)
  2.  Trend                 (SMA, EMA, MACD, ADX — window-scaled to timeframe)
  3.  Momentum              (RSI, Stochastic, Williams %R, CCI, MFI)
  4.  Volatility            (ATR, Bollinger Bands, Keltner squeeze, HVol)
  5.  Volume                (OBV, volume ratio, intraday VWAP)
  6.  Calendar / session    (hour-of-day cyclical, forex sessions, day-of-week)
  7.  Lag features          (past returns/RSI/ATR lags)
  8.  Rolling statistics    (return mean/std/skew/kurt over day-equivalent windows)
  9.  Sequence arrays       (for LSTM/Transformer — saved separately)

Run:
    python features/feature_engineering.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Timeframe metadata
# ─────────────────────────────────────────────────────────────────────────────

# (bars_per_trading_day, trading_days_per_year, is_intraday)
TIMEFRAME_META = {
    "1d":  (1,    252, False),
    "1h":  (6,    252, True),   # ~6 liquid hours/day for EUR/USD forex
    "4h":  (1.5,  252, True),
    "15m": (24,   252, True),
    "5m":  (72,   252, True),
}

def get_timeframe_meta(timeframe: str) -> tuple[float, int, bool]:
    tf = timeframe.lower()
    if tf not in TIMEFRAME_META:
        log.warning(f"Unknown timeframe '{tf}', defaulting to 1d scaling.")
        return TIMEFRAME_META["1d"]
    return TIMEFRAME_META[tf]

def _bars(days: int, bpd: float) -> int:
    """Convert a window in trading days to actual bar count."""
    return max(2, int(round(days * bpd)))


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _true_range(high, low, prev_close) -> pd.Series:
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))

def _stochastic(high, low, close, period: int = 14):
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    k  = 100 * (close - lo) / (hi - lo + 1e-10)
    d  = k.rolling(3).mean()
    return k, d

def _cci(high, low, close, period: int = 20) -> pd.Series:
    tp     = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad    = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def _williams_r(high, low, close, period: int = 14) -> pd.Series:
    hi = high.rolling(period).max()
    lo = low.rolling(period).min()
    return -100 * (hi - close) / (hi - lo + 1e-10)

def _adx(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr   = _true_range(high, low, prev_close)
    atr  = tr.ewm(span=period, adjust=False).mean()
    up   = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di  = 100 * pd.Series(plus_dm,  index=high.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).cumsum()

def _mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    tp  = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - 100 / (1 + pos / (neg + 1e-10))

def _intraday_vwap(high, low, close, volume) -> pd.Series:
    """
    True intraday VWAP that resets at the start of each calendar day.
    For daily data or plain integer index, falls back to rolling 24-bar VWAP.
    """
    tp = (high + low + close) / 3
    try:
        df_tmp = pd.DataFrame(
            {"tp": tp, "vol": volume, "tp_vol": tp * volume},
            index=high.index,
        )
        df_tmp["_date"] = df_tmp.index.date
        df_tmp["cum_tpvol"] = df_tmp.groupby("_date")["tp_vol"].cumsum()
        df_tmp["cum_vol"]   = df_tmp.groupby("_date")["vol"].cumsum()
        return df_tmp["cum_tpvol"] / (df_tmp["cum_vol"] + 1e-10)
    except Exception:
        return (tp * volume).rolling(24).sum() / (volume.rolling(24).sum() + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Main feature engineering function
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Takes the processed OHLCV + label DataFrame and returns a DataFrame
    with all engineered features appended.
    """
    fc        = cfg["features"]
    timeframe = cfg["data"].get("timeframe", "1d")
    bpd, bpy, is_intraday = get_timeframe_meta(timeframe)

    log.info(f"Timeframe: {timeframe}  |  bars/day={bpd}  |  intraday={is_intraday}")

    feat   = df.copy()
    o, h, l, c, v = feat["open"], feat["high"], feat["low"], feat["close"], feat["volume"]
    prev_c = c.shift(1)
    lr     = feat["log_return"]

    # ── 1. Price Action ───────────────────────────────────────────────────────
    log.info("  → Price action features")
    feat["body"]           = c - o
    feat["body_abs"]       = feat["body"].abs()
    feat["wick_upper"]     = h - feat[["open", "close"]].max(axis=1)
    feat["wick_lower"]     = feat[["open", "close"]].min(axis=1) - l
    feat["range"]          = h - l
    feat["body_to_range"]  = feat["body_abs"]   / (feat["range"] + 1e-10)
    feat["upper_wick_pct"] = feat["wick_upper"] / (feat["range"] + 1e-10)
    feat["lower_wick_pct"] = feat["wick_lower"] / (feat["range"] + 1e-10)
    feat["is_bullish"]     = (c > o).astype(int)
    feat["is_doji"]        = (feat["body_to_range"] < 0.1).astype(int)
    feat["is_spinning_top"] = (
        (feat["body_to_range"] < 0.3) &
        (feat["upper_wick_pct"] > 0.3) &
        (feat["lower_wick_pct"] > 0.3)
    ).astype(int)

    # Gap (session open vs previous close)
    feat["gap"]     = o - prev_c
    feat["gap_pct"] = feat["gap"] / (prev_c + 1e-10)
    if not is_intraday:
        feat["has_gap_up"]   = (feat["gap_pct"] >  0.0005).astype(int)
        feat["has_gap_down"] = (feat["gap_pct"] < -0.0005).astype(int)

    # ── 2. Trend Indicators ───────────────────────────────────────────────────
    log.info("  → Trend indicators")

    sma_days = fc.get("sma_windows", [5, 10, 20, 50, 100, 200])
    for d in sma_days:
        n = _bars(d, bpd)
        feat[f"sma_{d}d"]       = _sma(c, n)
        feat[f"close_to_sma_{d}d"] = c / (feat[f"sma_{d}d"] + 1e-10) - 1

    ema_days = fc.get("ema_windows", [5, 12, 26, 50])
    for d in ema_days:
        n = _bars(d, bpd)
        feat[f"ema_{d}d"]       = _ema(c, n)
        feat[f"close_to_ema_{d}d"] = c / (feat[f"ema_{d}d"] + 1e-10) - 1

    # MACD
    mf = _bars(fc.get("macd_fast",   12), bpd)
    ms = _bars(fc.get("macd_slow",   26), bpd)
    mg = _bars(fc.get("macd_signal",  9), bpd)
    feat["macd_line"]        = _ema(c, mf) - _ema(c, ms)
    feat["macd_signal_line"] = _ema(feat["macd_line"], mg)
    feat["macd_hist"]        = feat["macd_line"] - feat["macd_signal_line"]
    feat["macd_cross"]       = (
        np.sign(feat["macd_hist"]) != np.sign(feat["macd_hist"].shift(1))
    ).astype(int)

    # ADX
    adx_p = _bars(fc.get("atr_period", 14), bpd)
    adx, plus_di, minus_di = _adx(h, l, c, adx_p)
    feat["adx"]          = adx
    feat["plus_di"]      = plus_di
    feat["minus_di"]     = minus_di
    feat["di_diff"]      = plus_di - minus_di
    feat["adx_trending"] = (adx > 25).astype(int)

    # SMA crossovers
    sma_5d   = feat[f"sma_5d"]   if "sma_5d"   in feat.columns else _sma(c, _bars(5,   bpd))
    sma_20d  = feat[f"sma_20d"]  if "sma_20d"  in feat.columns else _sma(c, _bars(20,  bpd))
    sma_50d  = feat[f"sma_50d"]  if "sma_50d"  in feat.columns else _sma(c, _bars(50,  bpd))
    sma_200d = feat[f"sma_200d"] if "sma_200d" in feat.columns else _sma(c, _bars(200, bpd))

    feat["sma5_above_sma20"]   = (sma_5d   > sma_20d).astype(int)
    feat["sma20_above_sma50"]  = (sma_20d  > sma_50d).astype(int)
    feat["sma50_above_sma200"] = (sma_50d  > sma_200d).astype(int)
    feat["golden_cross"] = (
        (sma_50d > sma_200d) & (sma_50d.shift(1) <= sma_200d.shift(1))
    ).astype(int)
    feat["death_cross"] = (
        (sma_50d < sma_200d) & (sma_50d.shift(1) >= sma_200d.shift(1))
    ).astype(int)

    # ── 3. Momentum Oscillators ───────────────────────────────────────────────
    log.info("  → Momentum oscillators")
    rsi_p = _bars(fc.get("rsi_period", 14), bpd)
    feat["rsi"]          = _rsi(c, rsi_p)
    feat["rsi_overbought"] = (feat["rsi"] > 70).astype(int)
    feat["rsi_oversold"]   = (feat["rsi"] < 30).astype(int)
    feat["rsi_mid"]        = (feat["rsi"] > 50).astype(int)

    stoch_p = _bars(fc.get("stoch_period", 14), bpd)
    stoch_k, stoch_d = _stochastic(h, l, c, stoch_p)
    feat["stoch_k"]         = stoch_k
    feat["stoch_d"]         = stoch_d
    feat["stoch_kd_diff"]   = stoch_k - stoch_d
    feat["stoch_overbought"] = (stoch_k > 80).astype(int)
    feat["stoch_oversold"]   = (stoch_k < 20).astype(int)

    cci_p = _bars(fc.get("cci_period", 20), bpd)
    feat["cci"]        = _cci(h, l, c, cci_p)
    wr_p   = _bars(fc.get("williams_period", 14), bpd)
    feat["williams_r"] = _williams_r(h, l, c, wr_p)
    feat["mfi"]        = _mfi(h, l, c, v, rsi_p)

    # ROC — 1 day, 1 week, 1 month in bar-equivalent terms
    for d in [1, 5, 21]:
        n = _bars(d, bpd)
        feat[f"roc_{d}d"] = (c / c.shift(n) - 1) * 100

    # ── 4. Volatility ─────────────────────────────────────────────────────────
    log.info("  → Volatility indicators")
    atr_p = _bars(fc.get("atr_period", 14), bpd)
    tr    = _true_range(h, l, prev_c)
    feat["atr"]     = tr.ewm(span=atr_p, adjust=False).mean()
    feat["atr_pct"] = feat["atr"] / (c + 1e-10)

    bb_p   = _bars(fc.get("bb_period", 20), bpd)
    bb_mid = _sma(c, bb_p)
    bb_std = c.rolling(bb_p).std()
    bb_k   = fc.get("bb_std", 2)
    feat["bb_upper"]   = bb_mid + bb_k * bb_std
    feat["bb_lower"]   = bb_mid - bb_k * bb_std
    feat["bb_mid"]     = bb_mid
    feat["bb_width"]   = (feat["bb_upper"] - feat["bb_lower"]) / (bb_mid + 1e-10)
    feat["bb_pct_b"]   = (c - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-10)
    feat["bb_squeeze"] = (feat["bb_width"] < feat["bb_width"].rolling(bb_p).mean()).astype(int)

    # Historical volatility — annualised correctly for this timeframe
    # Daily:  * sqrt(252);  Hourly: * sqrt(252 * 6)
    ann_factor = np.sqrt(bpy * bpd)
    for d in [5, 10, 21]:
        n = _bars(d, bpd)
        feat[f"hvol_{d}d"] = lr.rolling(n).std() * ann_factor

    # Keltner Channel
    kc_p   = _bars(20, bpd)
    kc_mid = _ema(c, kc_p)
    feat["kc_upper"]   = kc_mid + 2 * feat["atr"]
    feat["kc_lower"]   = kc_mid - 2 * feat["atr"]
    feat["kc_pct"]     = (c - feat["kc_lower"]) / (feat["kc_upper"] - feat["kc_lower"] + 1e-10)
    feat["squeeze_on"] = (
        (feat["bb_upper"] < feat["kc_upper"]) &
        (feat["bb_lower"] > feat["kc_lower"])
    ).astype(int)

    # ── 5. Volume Features ────────────────────────────────────────────────────
    log.info("  → Volume features")
    vol_p = _bars(20, bpd)
    feat["volume_sma20"]  = _sma(v, vol_p)
    feat["volume_ratio"]  = v / (feat["volume_sma20"] + 1e-10)
    feat["volume_spike"]  = (feat["volume_ratio"] > 2.0).astype(int)
    feat["obv"]           = _obv(c, v)
    feat["obv_ema"]       = _ema(feat["obv"], _bars(10, bpd))
    feat["obv_diff"]      = feat["obv"] - feat["obv_ema"]

    # Intraday VWAP (resets daily)
    vwap = _intraday_vwap(h, l, c, v)
    feat["vwap"]          = vwap
    feat["close_to_vwap"] = c / (vwap + 1e-10) - 1
    feat["above_vwap"]    = (c > vwap).astype(int)

    # ── 6. Calendar / Session Features ───────────────────────────────────────
    log.info("  → Calendar & session features")

    feat["day_of_week"]  = feat.index.dayofweek
    feat["day_of_month"] = feat.index.day
    feat["month"]        = feat.index.month
    feat["quarter"]      = feat.index.quarter

    feat["dow_sin"]   = np.sin(2 * np.pi * feat["day_of_week"] / 5)
    feat["dow_cos"]   = np.cos(2 * np.pi * feat["day_of_week"] / 5)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # Calendar boundary flags — for intraday data, only fire on the FIRST bar
    # of that calendar day to avoid dozens of identical flags per day
    if is_intraday:
        try:
            date_arr  = np.array(feat.index.date, dtype=object)
            prev_date = np.roll(date_arr, 1)
            prev_date[0] = None
            is_first_bar = pd.Series(date_arr != prev_date, index=feat.index)

            feat["is_day_open"]    = is_first_bar.astype(int)
            feat["is_month_start"] = (is_first_bar & feat.index.is_month_start).astype(int)
            feat["is_month_end"]   = (is_first_bar & feat.index.is_month_end).astype(int)
            feat["is_quarter_end"] = (is_first_bar & feat.index.is_quarter_end).astype(int)
        except Exception as e:
            log.warning(f"Date boundary features failed: {e}")
            feat["is_day_open"] = feat["is_month_start"] = 0
            feat["is_month_end"] = feat["is_quarter_end"] = 0
    else:
        feat["is_day_open"]    = 1
        feat["is_month_start"] = feat.index.is_month_start.astype(int)
        feat["is_month_end"]   = feat.index.is_month_end.astype(int)
        feat["is_quarter_end"] = feat.index.is_quarter_end.astype(int)

    # ── Hour-of-day & forex session (intraday only) ───────────────────────────
    # This is the most important new feature group for 1h performance.
    # London open (07-16 UTC) and London/NY overlap (12-16 UTC) account for
    # the majority of EUR/USD daily volume and directional moves.
    if is_intraday:
        try:
            hour = feat.index.hour

            feat["hour"] = hour
            # Cyclical encoding so hour 23 and hour 0 are adjacent
            feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)

            # Forex session flags (UTC times)
            feat["session_sydney"]     = ((hour >= 21) | (hour < 6)).astype(int)
            feat["session_tokyo"]      = ((hour >= 0) & (hour < 9)).astype(int)
            feat["session_london"]     = ((hour >= 7) & (hour < 16)).astype(int)
            feat["session_newyork"]    = ((hour >= 12) & (hour < 21)).astype(int)
            feat["session_overlap"]    = ((hour >= 12) & (hour < 16)).astype(int)  # highest volume
            feat["session_pre_london"] = ((hour >= 5) & (hour < 7)).astype(int)    # often sets tone
            feat["session_dead"]       = ((hour >= 20) & (hour < 24)).astype(int)  # low liquidity

            # Open-bar flags (momentum spikes at session opens)
            feat["london_open_bar"] = (hour == 7).astype(int)
            feat["ny_open_bar"]     = (hour == 12).astype(int)
            feat["tokyo_open_bar"]  = (hour == 0).astype(int)

            # Hours since London open (captures intraday trend decay)
            feat["hours_since_london_open"] = np.where(
                hour >= 7,
                np.minimum(hour - 7, 9),   # cap at 9h post-open
                0,
            )

        except Exception as e:
            log.warning(f"Session features failed: {e}. Filling with zeros.")
            for col in ["hour", "hour_sin", "hour_cos", "session_sydney",
                        "session_tokyo", "session_london", "session_newyork",
                        "session_overlap", "session_pre_london", "session_dead",
                        "london_open_bar", "ny_open_bar", "tokyo_open_bar",
                        "hours_since_london_open"]:
                feat[col] = 0
    else:
        # Daily data — fill session features with neutral constants
        for col in ["hour", "hour_sin", "hour_cos", "session_sydney",
                    "session_tokyo", "session_london", "session_newyork",
                    "session_overlap", "session_pre_london", "session_dead",
                    "london_open_bar", "ny_open_bar", "tokyo_open_bar",
                    "hours_since_london_open"]:
            feat[col] = 0

    # ── 7. Lag Features ───────────────────────────────────────────────────────
    log.info("  → Lag features")
    lag_days = fc.get("lag_periods", [1, 2, 3, 5, 10, 21])
    for d in lag_days:
        n = _bars(d, bpd)
        feat[f"return_lag_{d}d"] = lr.shift(n)
        feat[f"close_lag_{d}d"]  = c.shift(n)
        feat[f"rsi_lag_{d}d"]    = feat["rsi"].shift(n)
        feat[f"atr_lag_{d}d"]    = feat["atr"].shift(n)

    # ── 8. Rolling Statistics of Returns ─────────────────────────────────────
    log.info("  → Rolling statistics")
    roll_days = fc.get("rolling_windows", [5, 10, 21])
    for d in roll_days:
        n = _bars(d, bpd)
        feat[f"ret_mean_{d}d"]  = lr.rolling(n).mean()
        feat[f"ret_std_{d}d"]   = lr.rolling(n).std()
        feat[f"ret_skew_{d}d"]  = lr.rolling(n).skew()
        feat[f"ret_kurt_{d}d"]  = lr.rolling(n).kurt()
        feat[f"ret_min_{d}d"]   = lr.rolling(n).min()
        feat[f"ret_max_{d}d"]   = lr.rolling(n).max()
        feat[f"pos_bars_{d}d"]  = (lr > 0).rolling(n).sum() / n

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    before = len(feat)
    feat = feat.dropna()
    log.info(f"  Dropped {before - len(feat)} NaN rows → {len(feat)} rows remain")

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder for LSTM / Transformer
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build overlapping windows for sequence models.

    Returns:
        X:     (n_samples, seq_len, n_features)
        y:     (n_samples,)
        dates: (n_samples,)  — date of the LAST bar in each window
    """
    arr    = feat_df[feature_cols].values
    labels = feat_df[label_col].values
    dates  = feat_df.index.values

    X_list, y_list, date_list = [], [], []
    for i in range(seq_len, len(arr)):
        X_list.append(arr[i - seq_len : i])
        y_list.append(labels[i])
        date_list.append(dates[i])

    return (
        np.array(X_list,    dtype=np.float32),
        np.array(y_list,    dtype=np.int64),
        np.array(date_list),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature column selector
# ─────────────────────────────────────────────────────────────────────────────

# Columns that are targets or raw price levels — excluded from model input
NON_FEATURE_COLS = {
    "open", "high", "low", "close", "volume",
    "log_return", "next_log_return", "label",
}

# Raw price-level prefixes to exclude (we keep the normalised ratio versions)
_EXCLUDE_PREFIXES = (
    "sma_", "ema_", "bb_upper", "bb_lower", "bb_mid",
    "kc_upper", "kc_lower", "vwap",
)
# Suffixes that are allowed even if the column matches an exclude prefix
_KEEP_SUFFIXES = (
    "_pct", "_pct_b", "_width", "_diff", "_squeeze",
    "_ema", "_ratio", "to_vwap", "above_vwap",
    "to_sma", "to_ema", "_on",
)

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all engineered feature columns (excludes raw OHLCV and target)."""
    cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        # Exclude raw SMA/EMA/BB/KC price levels but keep ratio/pct variants
        if any(col.startswith(p) for p in _EXCLUDE_PREFIXES):
            if not any(col.endswith(s) for s in _KEEP_SUFFIXES):
                continue
        cols.append(col)
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    dc  = cfg["data"]
    fc  = cfg["features"]

    proc_path = Path(dc["processed_path"])
    out_path  = Path(fc["featured_path"])
    out_path.mkdir(parents=True, exist_ok=True)

    if not proc_path.exists():
        raise FileNotFoundError(f"{proc_path} not found. Run data/fetch_data.py first.")

    df = pd.read_parquet(proc_path)
    log.info(f"Loaded processed data: {len(df)} rows, {len(df.columns)} columns")

    feat_df      = engineer_features(df, cfg)
    feature_cols = get_feature_columns(feat_df)
    log.info(f"Total features engineered: {len(feature_cols)}")

    # Save featured dataset
    feat_path = out_path / "featured_eurusd.parquet"
    feat_df.to_parquet(feat_path)
    log.info(f"Featured dataset saved → {feat_path}")

    # Save feature column list
    col_path = out_path / "feature_columns.json"
    with open(col_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"Feature column list saved → {col_path}")

    # Build & save sequence arrays for DL models
    seq_len = fc["sequence_length"]
    X_seq, y_seq, dates_seq = build_sequences(feat_df, feature_cols, "label", seq_len)

    seq_path = out_path / "sequences.npz"
    np.savez_compressed(seq_path, X=X_seq, y=y_seq, dates=dates_seq.astype(str))
    log.info(f"Sequences saved → {seq_path}  shape: {X_seq.shape}")

    print("\n" + "="*60)
    print("  FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"  Timeframe : {dc.get('timeframe', '1d')}")
    print(f"  Rows      : {len(feat_df)}")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Sequences : {X_seq.shape}")
    print("="*60)
    print("  Next step → python training/search_models.py")


if __name__ == "__main__":
    main()