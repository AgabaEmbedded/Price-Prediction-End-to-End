import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import json
import mlflow

# Internal imports from your project structure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch.fetch_data import fetch_ohlcv, make_labels, preprocess
from features.feature_engineering import engineer_features
from utils.mlflow_utils import setup_mlflow

# --- Constants ---
SIGNAL_NAMES   = {0: "STRONG SELL 🔴", 1: "SELL 🟠", 2: "HOLD ⚪", 3: "BUY 🟢", 4: "STRONG BUY 💚"}
SIGNAL_ACTIONS = {0: "SHORT — large downward move expected",
                  1: "SHORT — small downward move expected",
                  2: "STAY OUT — no significant move expected",
                  3: "LONG  — small upward move expected",
                  4: "LONG  — large upward move expected"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- State Management ---
app_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and feature metadata on startup."""
    log.info("Initializing application resources...")
    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        
        col_path = Path(cfg["features"]["featured_path"]) / "feature_columns.json"
        with open(col_path) as f:
            feature_cols = json.load(f)
        
        # Setup MLflow connection
        setup_mlflow(cfg)
        
        app_data["cfg"] = cfg
        app_data["feature_cols"] = feature_cols
        log.info("Startup complete.")
    except Exception as e:
        log.error(f"Failed to initialize: {e}")
        raise e
    yield
    app_data.clear()

app = FastAPI(title="EUR/USD Prediction Service", lifespan=lifespan)

# --- Core Logic ---

def get_latest_features(cfg: dict, feature_cols: list[str]):
    dc = cfg["data"]
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=600)).strftime("%Y-%m-%d")

    df = fetch_ohlcv(dc["ticker"], start, end)
    df = make_labels(df, dc["strong_threshold"], dc["weak_threshold"])
    df = preprocess(df)
    feat_df = engineer_features(df, cfg)

    last_row = feat_df[feature_cols].iloc[-1].values.astype(np.float32)
    last_date = feat_df.index[-1]
    return last_row.reshape(1, -1), last_date

def get_signal_message(pred: int, proba: np.ndarray | None, last_date, model_name: str) -> str:
    lines = []
    border = "═" * 55
    lines.append(border)
    lines.append(f"   EUR/USD DAILY SIGNAL  —  {datetime.today().strftime('%A %d %b %Y')}")
    lines.append(f"   Model: {model_name}  |  Based on data up to: {last_date.date()}")
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

    lines.append("\n   ⚠️  DISCLAIMER: Research model for learning purposes. NOT financial advice.")
    lines.append(border)
    return "\n".join(lines)

# --- Endpoints ---

@app.get("/predict")
async def predict_signal(model_alias: str = Query("latest", help="The version or alias in MLflow")):
    """Fetch data, load model from registry, and return the formatted signal."""
    cfg = app_data["cfg"]
    feature_cols = app_data["feature_cols"]
    
    try:
        # 1. Load model from Registry
        model_uri = f"models:/eurusd_model/{model_alias}"
        log.info(f"Fetching model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 2. Get latest market data and engineer features
        X, last_date = get_latest_features(cfg, feature_cols)
        X_df = pd.DataFrame(X, columns=feature_cols)
        
        # 3. Generate Prediction
        pred = int(model.predict(X_df)[0])
        
        # Note: MLflow pyfunc doesn't always expose predict_proba. 
        # If your model supports it, you can attempt to access the underlying model:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_df)[0]

        # 4. Format Message
        message = get_signal_message(pred, proba, last_date, "eurusd_model")
        
        return {
            "model": "eurusd_model",
            "version": model_alias,
            "last_data_point": last_date.strftime("%Y-%m-%d"),
            "signal": SIGNAL_NAMES[pred],
            "formatted_message": message
        }

    except Exception as e:
        log.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)