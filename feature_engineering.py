"""
features/feature_engineering.py
────────────────────────────────
Comprehensive time-series feature engineering for EUR/USD.

Feature groups:
  1.  Price action features        (OHLC relationships, body/wick ratios)
  2.  Trend indicators             (SMA, EMA, MACD, ADX)
  3.  Momentum oscillators         (RSI, Stochastic, Williams %R, CCI, MFI)
  4.  Volatility indicators        (ATR, Bollinger Bands, Keltner Channels)
  5.  Volume features              (OBV, Volume ratios)
  6.  Calendar / time features     (day-of-week, month, quarter effects)
  7.  Lag features                 (past returns and close prices)
  8.  Rolling statistics           (mean, std, skew, kurt of returns)
  9.  Sequence arrays              (for LSTM/Transformer — saved separately)

Run:
    python features/feature_engineering.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _true_range(high, low, prev_close) -> pd.Series:
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def _stochastic(high, low, close, period: int = 14) -> tuple[pd.Series, pd.Series]:
    lowest  = low.rolling(period).min()
    highest = high.rolling(period).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(3).mean()
    return k, d

def _cci(high, low, close, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad    = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def _williams_r(high, low, close, period: int = 14) -> pd.Series:
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest + 1e-10)

def _adx(high, low, close, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ADX, +DI, -DI"""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = _true_range(high, low, prev_close)
    atr = tr.ewm(span=period, adjust=False).mean()

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di  = 100 * pd.Series(plus_dm, index=high.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def _mfi(high, low, close, volume, period: int = 14) -> pd.Series:
    tp  = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(period).sum()
    return 100 - 100 / (1 + pos / (neg + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Main feature engineering function
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Takes the processed OHLCV + label DataFrame and returns a DataFrame
    with all engineered features appended.
    """
    fc = cfg["features"]
    feat = df.copy()

    o, h, l, c, v = feat["open"], feat["high"], feat["low"], feat["close"], feat["volume"]
    prev_c = c.shift(1)

    log.info("Engineering features...")

    # ── 1. Price Action ──────────────────────────────────────────────────────
    log.info("  → Price action features")
    feat["body"]           = c - o
    feat["body_abs"]       = feat["body"].abs()
    feat["wick_upper"]     = h - feat[["open", "close"]].max(axis=1)
    feat["wick_lower"]     = feat[["open", "close"]].min(axis=1) - l
    feat["range"]          = h - l
    feat["body_to_range"]  = feat["body_abs"] / (feat["range"] + 1e-10)
    feat["upper_wick_pct"] = feat["wick_upper"] / (feat["range"] + 1e-10)
    feat["lower_wick_pct"] = feat["wick_lower"] / (feat["range"] + 1e-10)
    feat["gap"]            = o - prev_c       # overnight gap
    feat["gap_pct"]        = feat["gap"] / (prev_c + 1e-10)
    feat["is_bullish"]     = (c > o).astype(int)

    # ── 2. Trend Indicators ──────────────────────────────────────────────────
    log.info("  → Trend indicators")
    for w in fc["sma_windows"]:
        feat[f"sma_{w}"] = _sma(c, w)
        feat[f"close_to_sma_{w}"] = c / (feat[f"sma_{w}"] + 1e-10) - 1

    for w in fc["ema_windows"]:
        feat[f"ema_{w}"] = _ema(c, w)
        feat[f"close_to_ema_{w}"] = c / (feat[f"ema_{w}"] + 1e-10) - 1

    # MACD
    macd_fast   = fc["macd_fast"]
    macd_slow   = fc["macd_slow"]
    macd_signal = fc["macd_signal"]
    feat["macd_line"]    = _ema(c, macd_fast) - _ema(c, macd_slow)
    feat["macd_signal"]  = _ema(feat["macd_line"], macd_signal)
    feat["macd_hist"]    = feat["macd_line"] - feat["macd_signal"]
    feat["macd_cross"]   = np.sign(feat["macd_hist"]) != np.sign(feat["macd_hist"].shift(1))
    feat["macd_cross"]   = feat["macd_cross"].astype(int)

    # ADX
    adx, plus_di, minus_di = _adx(h, l, c)
    feat["adx"]      = adx
    feat["plus_di"]  = plus_di
    feat["minus_di"] = minus_di
    feat["di_diff"]  = plus_di - minus_di

    # SMA crossovers
    feat["sma5_above_sma20"]   = (feat["sma_5"] > feat["sma_20"]).astype(int)
    feat["sma20_above_sma50"]  = (feat["sma_20"] > feat["sma_50"]).astype(int)
    feat["sma50_above_sma200"] = (feat["sma_50"] > feat["sma_200"]).astype(int)
    feat["golden_cross"]       = (
        (feat["sma_50"] > feat["sma_200"]) &
        (feat["sma_50"].shift(1) <= feat["sma_200"].shift(1))
    ).astype(int)
    feat["death_cross"] = (
        (feat["sma_50"] < feat["sma_200"]) &
        (feat["sma_50"].shift(1) >= feat["sma_200"].shift(1))
    ).astype(int)

    # ── 3. Momentum Oscillators ──────────────────────────────────────────────
    log.info("  → Momentum oscillators")
    feat["rsi"] = _rsi(c, fc["rsi_period"])
    feat["rsi_overbought"] = (feat["rsi"] > 70).astype(int)
    feat["rsi_oversold"]   = (feat["rsi"] < 30).astype(int)

    stoch_k, stoch_d = _stochastic(h, l, c, fc["stoch_period"])
    feat["stoch_k"] = stoch_k
    feat["stoch_d"] = stoch_d
    feat["stoch_kd_diff"] = stoch_k - stoch_d

    feat["cci"] = _cci(h, l, c, fc["cci_period"])
    feat["williams_r"] = _williams_r(h, l, c, fc["williams_period"])
    feat["mfi"] = _mfi(h, l, c, v)

    # Rate of Change
    for p in [5, 10, 21]:
        feat[f"roc_{p}"] = (c / c.shift(p) - 1) * 100

    # ── 4. Volatility ────────────────────────────────────────────────────────
    log.info("  → Volatility indicators")
    tr = _true_range(h, l, prev_c)
    feat["atr"]     = tr.ewm(span=fc["atr_period"], adjust=False).mean()
    feat["atr_pct"] = feat["atr"] / (c + 1e-10)   # normalised ATR

    bb_mid   = _sma(c, fc["bb_period"])
    bb_std   = c.rolling(fc["bb_period"]).std()
    feat["bb_upper"]  = bb_mid + fc["bb_std"] * bb_std
    feat["bb_lower"]  = bb_mid - fc["bb_std"] * bb_std
    feat["bb_mid"]    = bb_mid
    feat["bb_width"]  = (feat["bb_upper"] - feat["bb_lower"]) / (bb_mid + 1e-10)
    feat["bb_pct_b"]  = (c - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-10)
    feat["bb_squeeze"] = (feat["bb_width"] < feat["bb_width"].rolling(20).mean()).astype(int)

    # Historical volatility (rolling std of log returns)
    lr = feat["log_return"]
    for w in [5, 10, 21, 63]:
        feat[f"hvol_{w}d"] = lr.rolling(w).std() * np.sqrt(252)  # annualised

    # Keltner Channel
    kc_mid   = _ema(c, 20)
    feat["kc_upper"] = kc_mid + 2 * feat["atr"]
    feat["kc_lower"] = kc_mid - 2 * feat["atr"]
    feat["kc_pct"]   = (c - feat["kc_lower"]) / (feat["kc_upper"] - feat["kc_lower"] + 1e-10)

    # ── 5. Volume Features ───────────────────────────────────────────────────
    log.info("  → Volume features")
    feat["volume_sma20"]   = _sma(v, 20)
    feat["volume_ratio"]   = v / (feat["volume_sma20"] + 1e-10)
    feat["obv"]            = _obv(c, v)
    feat["obv_ema"]        = _ema(feat["obv"], 10)
    feat["obv_diff"]       = feat["obv"] - feat["obv_ema"]

    # Volume-weighted price momentum
    feat["vwap_proxy"] = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
    feat["close_to_vwap"] = c / (feat["vwap_proxy"] + 1e-10) - 1

    # ── 6. Calendar / Time Features ─────────────────────────────────────────
    log.info("  → Calendar features")
    feat["day_of_week"]   = feat.index.dayofweek          # 0=Mon … 4=Fri
    feat["day_of_month"]  = feat.index.day
    feat["month"]         = feat.index.month
    feat["quarter"]       = feat.index.quarter
    feat["week_of_year"]  = feat.index.isocalendar().week.astype(int)
    feat["is_month_start"] = feat.index.is_month_start.astype(int)
    feat["is_month_end"]   = feat.index.is_month_end.astype(int)
    feat["is_quarter_end"] = feat.index.is_quarter_end.astype(int)

    # Cyclical encoding for periodic features (avoids ordinal distance issues)
    feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 5)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 5)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # ── 7. Lag Features ──────────────────────────────────────────────────────
    log.info("  → Lag features")
    for lag in fc["lag_periods"]:
        feat[f"return_lag_{lag}"] = lr.shift(lag)
        feat[f"close_lag_{lag}"]  = c.shift(lag)
        feat[f"rsi_lag_{lag}"]    = feat["rsi"].shift(lag)

    # ── 8. Rolling Statistics of Returns ────────────────────────────────────
    log.info("  → Rolling statistics")
    for w in fc["rolling_windows"]:
        feat[f"ret_mean_{w}d"] = lr.rolling(w).mean()
        feat[f"ret_std_{w}d"]  = lr.rolling(w).std()
        feat[f"ret_skew_{w}d"] = lr.rolling(w).skew()
        feat[f"ret_kurt_{w}d"] = lr.rolling(w).kurt()
        feat[f"ret_min_{w}d"]  = lr.rolling(w).min()
        feat[f"ret_max_{w}d"]  = lr.rolling(w).max()
        feat[f"pos_days_{w}d"] = (lr > 0).rolling(w).sum() / w   # fraction of up days

    # ── 9. Cross-asset / macro proxies (yfinance available tickers) ──────────
    # These are optional but add value. Enable if you wish to fetch additional
    # tickers (DXY proxy, gold, US10Y). Left as a TODO to keep this script
    # self-contained. See comments below.
    #
    # TODO:
    #   dxy = yf.download("DX-Y.NYB", ...)["Close"]   # USD index
    #   gold = yf.download("GC=F", ...)["Close"]       # Gold
    #   us10y = yf.download("^TNX", ...)["Close"]      # US 10Y yield
    #   vix = yf.download("^VIX", ...)["Close"]        # VIX
    #   Then merge on date index and add lag/ratio features.

    # ── Drop NaN rows created by rolling windows ─────────────────────────────
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
        X: shape (n_samples, seq_len, n_features)
        y: shape (n_samples,)
        dates: shape (n_samples,)  — the date of the LAST step in each window
    """
    X_list, y_list, date_list = [], [], []
    arr = feat_df[feature_cols].values
    labels = feat_df[label_col].values
    dates = feat_df.index.values

    for i in range(seq_len, len(arr)):
        X_list.append(arr[i - seq_len : i])
        y_list.append(labels[i])
        date_list.append(dates[i])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), np.array(date_list)


# ─────────────────────────────────────────────────────────────────────────────
# Feature column selector
# ─────────────────────────────────────────────────────────────────────────────

NON_FEATURE_COLS = {
    "open", "high", "low", "close", "volume",
    "log_return", "next_log_return", "label",
    # raw SMA/EMA levels — we keep the *ratio* versions instead
    # (but feel free to include them)
}

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all engineered feature columns (excludes raw OHLCV and target)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    dc  = cfg["data"]
    fc  = cfg["features"]

    proc_path = Path(dc["processed_path"])
    if not proc_path.exists():
        raise FileNotFoundError(f"{proc_path} not found. Run data/fetch_data.py first.")

    df = pd.read_parquet(proc_path)
    log.info(f"Loaded processed data: {len(df)} rows, {len(df.columns)} columns")

    # Feature engineering
    feat_df = engineer_features(df, cfg)
    feature_cols = get_feature_columns(feat_df)
    log.info(f"Total features engineered: {len(feature_cols)}")

    # Save the full featured dataset
    feat_path = proc_path.parent / "featured_eurusd.parquet"
    feat_df.to_parquet(feat_path)
    log.info(f"Featured dataset saved → {feat_path}")

    # Save feature column list
    import json
    col_path = proc_path.parent / "feature_columns.json"
    with open(col_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"Feature column list saved → {col_path}")

    # Build & save sequence arrays for DL models
    seq_len = fc["sequence_length"]
    X_seq, y_seq, dates_seq = build_sequences(feat_df, feature_cols, "label", seq_len)

    seq_path = proc_path.parent / "sequences.npz"
    np.savez_compressed(seq_path, X=X_seq, y=y_seq, dates=dates_seq.astype(str))
    log.info(f"Sequence arrays saved → {seq_path}  (shape: {X_seq.shape})")

    print("\n" + "="*60)
    print(f"  FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"  Rows:     {len(feat_df)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Sequences: {X_seq.shape}")
    print("="*60)
    print("  Next step → python training/search_models.py")


if __name__ == "__main__":
    main()
