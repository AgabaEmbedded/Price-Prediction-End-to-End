"""
predict.py
──────────
Load a trained model from MLflow/disk and make a prediction for today.

Usage:
    # Load from MLflow Model Registry:
    python predict.py --model LightGBM --from-registry

    # Load from local file:
    python predict.py --model LightGBM --model-path saved_models/LightGBM_best.joblib

Output:
    Today's signal: BUY / SELL / HOLD / STRONG BUY / STRONG SELL
    with confidence breakdown.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from fetch.fetch_data import fetch_ohlcv, make_labels, preprocess
from features.feature_engineering import engineer_features, get_feature_columns
from utils.email_utils import send_email
from utils.trade_utils import manage_ic_markets_scheduled_trade
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
TRADE_DIRECTIONS = {0: "SELL", 1: "SELL", 2: "HOLD", 3: "BUY", 4: "BUY"}
SIGNAL_NAMES   = {0: "STRONG SELL 🔴", 1: "SELL 🟠", 2: "HOLD ⚪", 3: "BUY 🟢", 4: "STRONG BUY 💚"}
SIGNAL_ACTIONS = {0: "SHORT — large downward move expected",
                  1: "SHORT — small downward move expected",
                  2: "STAY OUT — no significant move expected",
                  3: "LONG  — small upward move expected",
                  4: "LONG  — large upward move expected"}


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_latest_features(cfg: dict, feature_cols: list[str]) -> np.ndarray:
    """Fetch the latest data and compute features for today."""
    dc = cfg["data"]
    end = datetime.today().strftime("%Y-%m-%d")
    # Fetch enough history for all rolling windows (200+ days minimum)
    start = (datetime.today() - timedelta(days=600)).strftime("%Y-%m-%d")

    df = fetch_ohlcv(dc["ticker"], start, end)
    df = make_labels(df, dc["strong_threshold"], dc["weak_threshold"])
    df = preprocess(df)
    feat_df = engineer_features(df, cfg)

    # Use the last available row (today's features to predict tomorrow)
    last_row = feat_df[feature_cols].iloc[-1].values.astype(np.float32)
    last_date = feat_df.index[-1]
    log.info(f"Latest data point: {last_date.date()}")
    return last_row.reshape(1, -1), last_date


def predict_sklearn(model_path: str, feature_cols: list[str], cfg: dict):
    import joblib
    bundle = joblib.load(model_path)
    model  = bundle["model"]

    X, last_date = get_latest_features(cfg, feature_cols)
    pred   = model.predict(X)[0]
    proba  = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred, proba, last_date


def predict_from_registry(feature_cols: list[str], cfg: dict):
    import mlflow
    from utils.mlflow_utils import setup_mlflow
    setup_mlflow(cfg)

    registered_name = f"eurusd_model"
    model_uri = f"models:/{registered_name}/latest"
    log.info(f"Loading model from registry: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    X, last_date = get_latest_features(cfg, feature_cols)
    X_df = pd.DataFrame(X, columns=feature_cols)
    pred = int(model.predict(X_df)[0])
    return pred, None, last_date


def print_signal(pred: int, proba: np.ndarray | None, last_date):
    print("\n" + "="*55)
    print(f"  EUR/USD DAILY SIGNAL  —  {datetime.today().strftime('%A %d %b %Y')}")
    print(f"  Model: Registry Latest  |  Based on data up to: {last_date.date()}")
    print("="*55)
    print(f"\n  📊  SIGNAL:  {SIGNAL_NAMES[pred]}")
    print(f"  📌  ACTION:  {SIGNAL_ACTIONS[pred]}\n")

    if proba is not None:
        print("  Probability breakdown:")
        labels = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
        bar_max = 30
        for i, (lbl, p) in enumerate(zip(labels, proba)):
            bar = "█" * int(p * bar_max)
            arrow = " ◄" if i == pred else ""
            print(f"    {lbl:>12}  {bar:<{bar_max}}  {p*100:5.1f}%{arrow}")

    print("\n  ⚠️  DISCLAIMER: This is a research model")
    print("     NOT financial advice. Always use proper")
    print("     risk management before trading.")
    print("="*55 + "\n")


def get_signal_message(pred: int, proba: np.ndarray | None, last_date) -> str:
    lines = []
    
    border = "=" * 30
    lines.append(border)
    lines.append(f"   EUR/USD DAILY SIGNAL  —  {datetime.today().strftime('%A %d %b %Y')}")
    lines.append(f"   Model: Registry Latest  |  Based on data up to: {last_date.date()}")
    lines.append(border)
    
    lines.append(f"\n   📊  SIGNAL:  {SIGNAL_NAMES[pred]}")
    lines.append(f"   📌  ACTION:  {SIGNAL_ACTIONS[pred]}\n")

    if proba is not None:
        lines.append("   Probability breakdown:")
        labels = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
        bar_max = 30
        for i, (lbl, p) in enumerate(zip(labels, proba)):
            bar = "█" * int(p * bar_max)
            arrow = " ◄" if i == pred else ""
            lines.append(f"    {lbl:>12}  {bar:<{bar_max}}  {p*100:5.1f}%{arrow}")

    lines.append("\n   ⚠️  DISCLAIMER: This is a research model for portfolio/learning")
    lines.append("     purposes only. NOT financial advice. Always use proper")
    lines.append("     risk management before trading.")
    lines.append(border)

    # Combine everything into one string
    return "\n".join(lines)

def is_nfp_friday(date_obj):
    # NFP is always the first Friday of the month
    if date_obj.weekday() == 3 and date_obj.day <= 7:
        return True
    return False

def main():
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model",         required=False, help="Model name (e.g. LightGBM)")
    parser.add_argument("--model-path",    default=None,  help="Local model file path")
    parser.add_argument("--from-registry", action="store_true", help="Load from MLflow registry")
    args = parser.parse_args()


    cfg = load_config()

    # Load feature columns
    col_path = Path(cfg["features"]["featured_path"]) / "feature_columns.json"
    size = cfg["trade"]["size"]
    IC_MT5_PATH = cfg["trade"]["IC_MT5_PATH"]
    with open(col_path) as f:
        feature_cols = json.load(f)

    if args.from_registry:
        pred, proba, last_date = predict_from_registry(feature_cols, cfg)
    elif args.model_path:
        pred, proba, last_date = predict_sklearn(args.model_path, feature_cols, cfg)
    else:
        # Default: look for local file
        default_path = Path(cfg["output"]["model_dir"]) / "model_best.joblib"
        if not default_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {default_path}. "
                "Pass --model-path or --from-registry."
            )
        pred, proba, last_date = predict_sklearn(str(default_path), feature_cols, cfg)

    message = get_signal_message(pred, proba, last_date)


    now = datetime.now()
    if is_nfp_friday(now):
        message = f"Today is {now.strftime('%A %d %b %Y')}, which is NFP Friday. The Non-Farm Payroll report is released today, often causing high volatility in the markets. To protect capital, we are skipping the trade generation for today. Please review the NFP report and adjust your trading strategy accordingly.\n\n{message}"          
        pred = 2


    print_signal(pred, proba, last_date)
    IC_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
  
    send_email(message)
    # Trigger management sequence 
    try:
        manage_ic_markets_scheduled_trade(
            symbol="EURUSD", 
            direction=TRADE_DIRECTIONS[pred], 
            volume=size, 
            terminal_path=IC_MT5_PATH
        )
    except Exception as e:
        log.error(f"Error during trade management: {e}")
        log.info("Continuing without trade execution.")



if __name__ == "__main__": 

    now = datetime.now()   
    
    MAINTENANCE_MODE = True 
    
    if MAINTENANCE_MODE:
        print("[ALERT] Triggering maintenance mode. Server will remain awake for debugging.")
        sys.exit(100)

    elif now.weekday() == 4:
        print("[INFO] Friday execution complete. Signaling batch script to run DVC Retraining.")
        sys.exit(99)
    else:

        try:
            main()
            sys.exit(0)
        except Exception as e:
            print(f"[CRITICAL ERROR] Main execution crashed: {e}")
            print("Safeguarding server: switching to maintenance mode to prevent data loss.")
            sys.exit(100)