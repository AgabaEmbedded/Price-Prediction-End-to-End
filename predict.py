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

from matplotlib import lines
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import time

from fetch.fetch_data import fetch_ohlcv, make_labels, preprocess
from features.feature_engineering import engineer_features, get_feature_columns
from utils.email_utils import send_email
from utils.trade_utils import manage_ic_markets_scheduled_trade
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
TRADE_DIRECTIONS = {0: "SELL", 1: "BUY", 2: "HOLD"}
SIGNAL_NAMES   = {0: "SELL 🔴", 1: "BUY 🟢", 2: "HOLD ⚪"}
SIGNAL_ACTIONS = {0: "SHORT — downward move expected",
                  1: "LONG  — upward move expected",
                  2: "STAY OUT — market may be volatile or uncertain",
                  }


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_latest_features(cfg: dict, feature_cols: list[str], ticker_id: str) -> np.ndarray:
    """Fetch the latest data and compute features for today."""
    dc = cfg["data"]
    index = dc["tickers"].index(ticker_id.upper()+"=X")
    end = datetime.today().strftime("%Y-%m-%d")
    # Fetch enough history for all rolling windows (200+ days minimum)
    start = (datetime.today() - timedelta(days=600)).strftime("%Y-%m-%d")

    df = fetch_ohlcv(dc["tickers"][index], start, end)
    df = make_labels(df, dc["strong_thresholds"][index], dc["weak_thresholds"][index])
    df = preprocess(df)
    feat_df = engineer_features(df, cfg, ticker_id)

    # Use the last available row (today's features to predict tomorrow)
    last_row = feat_df[feature_cols].iloc[-1].values.astype(np.float32)
    last_date = feat_df.index[-1]
    log.info(f"Latest data point: {last_date.date()}")
    return last_row.reshape(1, -1), last_date


def predict_sklearn(model_path: str, feature_cols: list[str], cfg: dict, ticker_id: str, model_name: str):
    import joblib
    bundle = joblib.load(model_path)
    model  = bundle["model"]

    X, last_date = get_latest_features(cfg, feature_cols, ticker_id)
    pred   = model.predict(X)[0]
    proba  = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred, proba, last_date


def predict_from_registry(feature_cols: list[str], cfg: dict, ticker_id: str, model_name: str):
    import mlflow
    from utils.mlflow_utils import setup_mlflow
    setup_mlflow(cfg)

    registered_name = f"{ticker_id}_{model_name}"
    model_uri = f"models:/{registered_name}/latest"
    log.info(f"Loading model from registry: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    X, last_date = get_latest_features(cfg, feature_cols, ticker_id)
    X_df = pd.DataFrame(X, columns=feature_cols)
    pred = int(model.predict(X_df)[0])
    return pred, None, last_date


def print_signal(pred_dict: dict):
    print("\n" + "="*55)
    print(f"  DAILY SIGNAL  —  {datetime.today().strftime('%A %d %b %Y')}")
    print(f"  Model: Registry Latests")
    print("="*55)

    for ticker_id, (pred, proba, last_date) in pred_dict.items():
        print(f"\n   Ticker: {ticker_id.upper()} |  Based on data up to: {last_date.date()}")
        print(f"\n   📊  SIGNAL:  {SIGNAL_NAMES[pred]}")
        print(f"   📌  ACTION:  {SIGNAL_ACTIONS[pred]}\n")

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

def get_signal_message(pred_dict: dict) -> str:
    lines = []
    
    border = "=" * 30
    lines.append(border)
    lines.append(f"   DAILY SIGNAL  —  {datetime.today().strftime('%A %d %b %Y')}")
    lines.append(f"   Model: Registry Latest")
    lines.append(border)
    
    for ticker_id, (pred, proba, last_date) in pred_dict.items():
        lines.append(f"\n   Ticker: {ticker_id.upper()} |  Based on data up to: {last_date.date()}")
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
    parser.add_argument("--ticker",         required=False, help="yfinance ticker for pair to predict on e.g EURUSD=X, GBPUSD=X, etc.")
    parser.add_argument("--model-path",    default=None,  help="Local model file path")
    #parser.add_argument("--from-registry", action="store_true", help="Load from MLflow registry")
    args = parser.parse_args()


    cfg = load_config()
    tickers = [args.ticker] if args.ticker else cfg["data"]["tickers"]
    model_names = cfg["training"]["models"]
    model_names = [model_names[tickers.index(tickers[0])]] if args.ticker else model_names
    print(model_names)
    size = cfg["trade"]["size"]
    IC_MT5_PATH = cfg["trade"]["IC_MT5_PATH"]

    pred_dict = {}

    for ticker, model_name in zip(tickers, model_names):
        ticker_id = ticker.split("=")[0].lower()
        print(f"[INFO] Processing ticker: {ticker_id} with model: {model_name}")
        # Load feature columns
                               

        # Load feature columns
        col_path = Path(cfg["features"]["featured_path"]) / f"{ticker_id}/feature_columns.json"
        print("[INFO] Loading feature columns from:", col_path)
    
    
        with open(col_path) as f:
            feature_cols = json.load(f)

        
        if args.model_path:
            pred, proba, last_date = predict_sklearn(args.model_path, feature_cols, cfg, ticker_id, model_name)
        else:
            pred, proba, last_date = predict_from_registry(feature_cols, cfg, ticker_id, model_name)
        pred_dict[ticker_id] = [pred, proba, last_date]

    print(pred_dict)
    message = get_signal_message(pred_dict)


    now = datetime.now()
    if is_nfp_friday(now):
        message = f"Today is {now.strftime('%A %d %b %Y')}, which is NFP Friday. The Non-Farm Payroll report is released today, often causing high volatility in the markets. To protect capital, we are skipping the trade generation for today. Please review the NFP report and adjust your trading strategy accordingly.\n\n{message}"          
        for ticker_id in pred_dict.keys():
            pred_dict[ticker_id][0] = 2
        


    print_signal(pred_dict)
    IC_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
  
    send_email(message)
    # Trigger management sequence
    for ticker_id, (pred, proba, last_date) in pred_dict.items():
        try:
            manage_ic_markets_scheduled_trade(
                symbol=ticker_id.upper(), 
                direction=TRADE_DIRECTIONS[pred], 
                volume=size, 
                terminal_path=IC_MT5_PATH
            )
        except Exception as e:
            log.error(f"Error during trade management: {e}")
            log.info("Continuing without trade execution.")
    #time.sleep(120)
    #Retry 1: Trigger management sequence to ensure trade execution
    #for ticker_id, (pred, proba, last_date) in pred_dict.items():
    #    try:
    #        manage_ic_markets_scheduled_trade(
    #            symbol=ticker_id.upper(), 
    #            direction=TRADE_DIRECTIONS[pred], 
    #            volume=size, 
    #            terminal_path=IC_MT5_PATH
    #        )
    #    except Exception as e:
    #        log.error(f"Error during trade management: {e}")
    #        log.info("Continuing without trade execution.")
    

    time.sleep(120)
    #Retry 1: Trigger management sequence to ensure trade execution
    for ticker_id, (pred, proba, last_date) in pred_dict.items():
        try:
            manage_ic_markets_scheduled_trade(
                symbol=ticker_id.upper(), 
                direction=TRADE_DIRECTIONS[pred], 
                volume=size, 
                terminal_path=IC_MT5_PATH
            )
        except Exception as e:
            log.error(f"Error during trade management: {e}")
            log.info("Continuing without trade execution.")



if __name__ == "__main__":

    now = datetime.now()   
    
    MAINTENANCE_MODE = False 
    
    if MAINTENANCE_MODE:
        print("[ALERT] Triggering maintenance mode. Server will remain awake for debugging.")
        sys.exit(100)

    elif now.weekday() == 4:
        print("[INFO] Signaling batch script to run DVC Retraining.")
        sys.exit(99)
    else:

        try:
            main()
            sys.exit(0)
        except Exception as e:
            print(f"[CRITICAL ERROR] Main execution crashed: {e}")
            print("Safeguarding server: switching to maintenance mode to prevent data loss.")
            sys.exit(100)