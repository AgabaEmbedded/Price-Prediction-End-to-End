"""
training/search_models.py
─────────────────────────
Broad model search across all sklearn models + PyTorch DL models.
Supports multi-pair: runs search per ticker and saves a leaderboard per ticker.

Changes vs original:
  - Multi-pair: iterates over cfg["data"]["tickers"], loads per-ticker features
  - Binary labels (0=Sell, 1=Buy): n_classes read from cfg (default 2)
  - DL class_weights uses n_classes not hardcoded 5
  - Leaderboard saved per ticker: search_leaderboard_{ticker_id}.csv
  - Best params saved per ticker: best_params/{ticker_id}.json (matching train_final)
  - LogisticRegression: removed multi_class="multinomial" (invalid for binary)
  - XGBoost eval_metric: "logloss" for binary (was "mlogloss" for multiclass)

Run:
    python training/search_models.py
    python training/search_models.py --ticker EURUSD=X   # single pair
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from models.model_registry import (
    get_sklearn_models, LSTMClassifier, TransformerClassifier,
    TimeSeriesDataset, get_device,
)
from utils.mlflow_utils import setup_mlflow, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — per ticker
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data(cfg: dict, ticker_id: str):
    """Load train+val tabular data and sequence arrays for one ticker."""
    dc       = cfg["data"]
    fc       = cfg["features"]
    base     = Path(fc["featured_path"]) / ticker_id
    feat_path = base / "features.parquet"
    seq_path  = base / "sequences.npz"
    col_path  = base / "feature_columns.json"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found. Run features/feature_engineering.py first."
        )

    df = pd.read_parquet(feat_path)
    with open(col_path) as f:
        feature_cols = json.load(f)

    n         = len(df)
    search_end = int(n * (dc["train_ratio"] + dc["val_ratio"]))
    df_search  = df.iloc[:search_end]

    X_tab = df_search[feature_cols].values.astype(np.float32)
    y_tab = df_search["label"].values.astype(np.int64)

    seqs  = np.load(seq_path, allow_pickle=True)
    X_seq = seqs["X"][:search_end].astype(np.float32)
    y_seq = seqs["y"][:search_end].astype(np.int64)

    log.info(f"[{ticker_id}] Search data: tabular {X_tab.shape}, sequences {X_seq.shape}")
    return X_tab, y_tab, X_seq, y_seq, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn model search
# ─────────────────────────────────────────────────────────────────────────────

def search_sklearn_model(
    name: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    ticker_id: str,
) -> dict:
    ms_cfg = cfg["model_search"]
    tscv   = TimeSeriesSplit(n_splits=ms_cfg["cv_folds"])

    log.info(f"  [{ticker_id}] Searching: {name}")
    t0 = time.time()

    with mlflow.start_run(run_name=f"search_{ticker_id}_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_type", "sklearn")
        mlflow.set_tag("stage",      "model_search")
        mlflow.set_tag("ticker",     ticker_id)
        mlflow.log_param("model",      name)
        mlflow.log_param("n_cv_folds", ms_cfg["cv_folds"])

        cv_results = cross_validate(
            model, X, y,
            cv=tscv,
            scoring=["f1_macro", "accuracy", "balanced_accuracy"],
            n_jobs=ms_cfg["n_jobs"],
            return_train_score=True,
        )

        elapsed = time.time() - t0
        metrics = {
            "cv_f1_macro_mean":       float(cv_results["test_f1_macro"].mean()),
            "cv_f1_macro_std":        float(cv_results["test_f1_macro"].std()),
            "cv_accuracy_mean":       float(cv_results["test_accuracy"].mean()),
            "cv_balanced_acc_mean":   float(cv_results["test_balanced_accuracy"].mean()),
            "cv_train_f1_macro_mean": float(cv_results["train_f1_macro"].mean()),
            "elapsed_seconds":        elapsed,
        }
        mlflow.log_metrics(metrics)

    log.info(
        f"    {name}: f1={metrics['cv_f1_macro_mean']:.4f} "
        f"± {metrics['cv_f1_macro_std']:.4f}  ({elapsed:.1f}s)"
    )
    return {"name": name, "type": "sklearn", **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# DL model search
# ─────────────────────────────────────────────────────────────────────────────

def search_dl_model(
    name: str,
    model_cls,
    model_kwargs: dict,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    cfg: dict,
    ticker_id: str,
) -> dict:
    tc        = cfg["training"]["dl"]
    n_classes = cfg.get("training", {}).get("n_classes", 2)   # binary by default
    device    = get_device(tc["device"])

    split   = int(len(X_seq) * 0.80)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    n_feat  = X_train.shape[2]
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape).astype(np.float32)
    X_val   = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)

    train_dl_ = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=tc["batch_size"], shuffle=False)
    val_dl_   = DataLoader(TimeSeriesDataset(X_val,   y_val),   batch_size=tc["batch_size"], shuffle=False)

    # Class weights — use n_classes not hardcoded 5
    class_counts  = np.bincount(y_train, minlength=n_classes)
    class_weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32).to(device)

    # Inject n_classes into model kwargs
    model_kwargs = {**model_kwargs, "n_classes": n_classes}
    model     = model_cls(n_features=n_feat, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    log.info(f"  [{ticker_id}] Searching: {name}  (device={device}, n_classes={n_classes})")
    t0 = time.time()
    best_val_f1 = 0.0

    with mlflow.start_run(run_name=f"search_{ticker_id}_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_type", "pytorch")
        mlflow.set_tag("stage",      "model_search")
        mlflow.set_tag("ticker",     ticker_id)
        mlflow.log_params({
            **{k: v for k, v in model_kwargs.items() if k != "n_classes"},
            "batch_size": tc["batch_size"],
            "lr":         tc["learning_rate"],
            "n_classes":  n_classes,
        })

        for epoch in range(min(tc["max_epochs"], 30)):
            model.train()
            for xb, yb in train_dl_:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_dl_:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                    trues.extend(yb.numpy())

            val_f1 = f1_score(trues, preds, average="macro", zero_division=0)
            scheduler.step(1 - val_f1)
            best_val_f1 = max(best_val_f1, val_f1)
            mlflow.log_metric("val_f1_macro", val_f1, step=epoch)

        elapsed = time.time() - t0
        mlflow.log_metric("elapsed_seconds",   elapsed)
        mlflow.log_metric("cv_f1_macro_mean",  best_val_f1)

    log.info(f"    {name}: best_val_f1={best_val_f1:.4f}  ({elapsed:.1f}s)")
    return {
        "name": name, "type": "pytorch",
        "cv_f1_macro_mean": best_val_f1,
        "cv_f1_macro_std":  0.0,
        "elapsed_seconds":  elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker search
# ─────────────────────────────────────────────────────────────────────────────

def run_search_for_ticker(cfg: dict, ticker_id: str):
    n_classes = cfg.get("training", {}).get("n_classes", 2)
    seed      = cfg["training"]["random_seed"]

    X_tab, y_tab, X_seq, y_seq, feature_cols = load_train_data(cfg, ticker_id)
    results = []

    # ── Sklearn ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"  [{ticker_id}] SKLEARN MODEL SEARCH")
    log.info(f"{'='*60}")

    sklearn_models = get_sklearn_models(seed)
    for name, model in sklearn_models.items():
        try:
            r = search_sklearn_model(name, model, X_tab, y_tab, cfg, ticker_id)
            results.append(r)
        except Exception as e:
            log.error(f"  [{ticker_id}] {name} failed: {e}")

    # ── Deep Learning ─────────────────────────────────────────────────────────
    if cfg["model_search"]["include_deep_learning"]:
        log.info(f"\n{'='*60}")
        log.info(f"  [{ticker_id}] DEEP LEARNING MODEL SEARCH")
        log.info(f"{'='*60}")

        # n_classes injected in search_dl_model — don't hardcode here
        dl_configs = [
            ("LSTM",        LSTMClassifier,        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}),
            ("BiLSTM",      LSTMClassifier,        {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "bidirectional": True}),
            ("Transformer", TransformerClassifier, {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.3}),
        ]

        for name, cls, kwargs in dl_configs:
            try:
                r = search_dl_model(name, cls, kwargs, X_seq, y_seq, cfg, ticker_id)
                results.append(r)
            except Exception as e:
                log.error(f"  [{ticker_id}] {name} failed: {e}")

    # ── Leaderboard ───────────────────────────────────────────────────────────
    board = (
        pd.DataFrame(results)
        .sort_values("cv_f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )
    board.index += 1

    print(f"\n{'='*70}")
    print(f"  [{ticker_id.upper()}]  MODEL SEARCH LEADERBOARD  (F1 Macro)")
    print(f"{'='*70}")
    print(board[["name", "type", "cv_f1_macro_mean", "cv_f1_macro_std", "elapsed_seconds"]].to_string())
    print(f"{'='*70}")

    best = board.iloc[0]
    print(f"\n  🏆  [{ticker_id.upper()}] BEST: {best['name']}  (F1={best['cv_f1_macro_mean']:.4f})")

    # Save leaderboard per ticker
    board_path = Path(f"search_leaderboard_{ticker_id}.csv")
    board.to_csv(board_path, index=False)
    log.info(f"  Leaderboard saved → {board_path}")

    # Save best model name for HPO / train steps
    best_dir = Path("best_params")
    best_dir.mkdir(exist_ok=True)
    best_model_path = best_dir / f"{ticker_id}_best_model.json"
    with open(best_model_path, "w") as f:
        json.dump({"ticker": ticker_id, "best_model": best["name"],
                   "cv_f1": best["cv_f1_macro_mean"]}, f, indent=2)
    log.info(f"  Best model name saved → {best_model_path}")

    return best["name"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=False,
                        help="Single ticker to search (e.g. EURUSD=X). "
                             "Defaults to all tickers in config.")
    args = parser.parse_args()

    cfg = load_config()
    setup_mlflow(cfg)

    tickers = [args.ticker] if args.ticker else cfg["data"]["tickers"]

    summary = {}
    for ticker in tickers:
        ticker_id = ticker.lower().split("=")[0]
        try:
            best_model = run_search_for_ticker(cfg, ticker_id)
            summary[ticker_id] = best_model
        except Exception as e:
            log.error(f"Search failed for {ticker_id}: {e}")
            summary[ticker_id] = "FAILED"

    print(f"\n{'='*60}")
    print("  MULTI-PAIR SEARCH COMPLETE")
    print(f"{'='*60}")
    for tid, best in summary.items():
        print(f"  {tid.upper():<12} → best model: {best}")
    print(f"{'='*60}")
    print("\n  Next step:")
    print("    python training/hyperparameter_tuning.py")
    print("    python training/hyperparameter_tuning.py --ticker EURUSD=X --model LightGBM")


if __name__ == "__main__":
    main()