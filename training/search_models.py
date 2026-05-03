"""
training/search_models.py
─────────────────────────
Runs a broad model search across all sklearn models + PyTorch DL models.
Every run is tracked in MLflow (hosted on S3).

Uses TimeSeriesSplit cross-validation — no data leakage.

At the end, prints a ranked leaderboard and logs the best model name
as an MLflow tag so you can easily find it.

Run:
    python training/search_models.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.model_registry import (
    get_sklearn_models, LSTMClassifier, TransformerClassifier,
    TimeSeriesDataset, get_device,
)
from utils.mlflow_utils import setup_mlflow, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data(cfg: dict):
    """Load train+val tabular data and sequence arrays."""
    dc = cfg["data"]
    feat_path = Path(dc["processed_path"]).parent / "featured_eurusd.parquet"
    seq_path  = Path(dc["processed_path"]).parent / "sequences.npz"
    col_path  = Path(dc["processed_path"]).parent / "feature_columns.json"

    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found. Run features/feature_engineering.py first.")

    df = pd.read_parquet(feat_path)
    with open(col_path) as f:
        feature_cols = json.load(f)

    # Use train + val for model search (we keep test totally held out)
    n = len(df)
    train_end = int(n * (dc["train_ratio"] + dc["val_ratio"]))
    df_search = df.iloc[:train_end]

    X_tab = df_search[feature_cols].values.astype(np.float32)
    y_tab = df_search["label"].values.astype(np.int64)

    # Sequence data for DL models
    seqs = np.load(seq_path, allow_pickle=True)
    X_seq  = seqs["X"][:train_end]
    y_seq  = seqs["y"][:train_end]

    log.info(f"Search data: tabular {X_tab.shape}, sequences {X_seq.shape}")
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
) -> dict:
    """Run TimeSeriesCV and log results to MLflow."""
    ms_cfg = cfg["model_search"]
    tscv   = TimeSeriesSplit(n_splits=ms_cfg["cv_folds"])

    log.info(f"  Searching: {name}")
    t0 = time.time()

    with mlflow.start_run(run_name=f"search_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_type", "sklearn")
        mlflow.set_tag("stage", "model_search")

        # Cross-validate
        cv_results = cross_validate(
            model, X, y,
            cv=tscv,
            scoring=["f1_macro", "accuracy", "balanced_accuracy"],
            n_jobs=ms_cfg["n_jobs"],
            return_train_score=True,
        )

        elapsed = time.time() - t0
        metrics = {
            "cv_f1_macro_mean":          float(cv_results["test_f1_macro"].mean()),
            "cv_f1_macro_std":           float(cv_results["test_f1_macro"].std()),
            "cv_accuracy_mean":          float(cv_results["test_accuracy"].mean()),
            "cv_balanced_acc_mean":      float(cv_results["test_balanced_accuracy"].mean()),
            "cv_train_f1_macro_mean":    float(cv_results["train_f1_macro"].mean()),
            "elapsed_seconds":           elapsed,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_cv_folds", ms_cfg["cv_folds"])

        result = {"name": name, "type": "sklearn", **metrics}
        log.info(
            f"    {name}: f1_macro={metrics['cv_f1_macro_mean']:.4f} "
            f"± {metrics['cv_f1_macro_std']:.4f}  ({elapsed:.1f}s)"
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch model search (single train/val split — CV too slow for DL)
# ─────────────────────────────────────────────────────────────────────────────

def search_dl_model(
    name: str,
    model_cls,
    model_kwargs: dict,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    cfg: dict,
) -> dict:
    """Train a DL model on 80% of search data, eval on 20%, log to MLflow."""
    tc  = cfg["training"]["dl"]
    device = get_device(tc["device"])

    split = int(len(X_seq) * 0.80)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # Scale features
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    X_val_2d   = X_val.reshape(-1, n_feat)
    X_train = scaler.fit_transform(X_train_2d).reshape(X_train.shape).astype(np.float32)
    X_val   = scaler.transform(X_val_2d).reshape(X_val.shape).astype(np.float32)

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds   = TimeSeriesDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=tc["batch_size"], shuffle=False)

    # Class weights to handle imbalance
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32).to(device)

    model = model_cls(n_features=X_train.shape[2], **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    log.info(f"  Searching: {name}  (device={device})")
    t0 = time.time()
    best_val_f1 = 0.0

    with mlflow.start_run(run_name=f"search_{name}"):
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("model_type", "pytorch")
        mlflow.set_tag("stage", "model_search")
        mlflow.log_params({**model_kwargs, "batch_size": tc["batch_size"],
                           "lr": tc["learning_rate"]})

        for epoch in range(min(tc["max_epochs"], 30)):   # cap at 30 epochs for search
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    preds = model(xb.to(device)).argmax(1).cpu().numpy()
                    all_preds.extend(preds)
                    all_true.extend(yb.numpy())

            val_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
            val_acc = accuracy_score(all_true, all_preds)
            scheduler.step(1 - val_f1)
            best_val_f1 = max(best_val_f1, val_f1)

            mlflow.log_metric("val_f1_macro", val_f1, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        elapsed = time.time() - t0
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.log_metric("cv_f1_macro_mean", best_val_f1)   # for comparable leaderboard

        result = {
            "name": name,
            "type": "pytorch",
            "cv_f1_macro_mean": best_val_f1,
            "cv_f1_macro_std": 0.0,
            "elapsed_seconds": elapsed,
        }
        log.info(f"    {name}: best_val_f1={best_val_f1:.4f}  ({elapsed:.1f}s)")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Main search loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    setup_mlflow(cfg)

    X_tab, y_tab, X_seq, y_seq, feature_cols = load_train_data(cfg)

    results = []

    # ── Sklearn ───────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  SKLEARN MODEL SEARCH")
    log.info("="*60)
    sklearn_models = get_sklearn_models(cfg["training"]["random_seed"])
    for name, model in sklearn_models.items():
        try:
            r = search_sklearn_model(name, model, X_tab, y_tab, cfg)
            results.append(r)
        except Exception as e:
            log.error(f"  {name} failed: {e}")

    # ── Deep Learning ─────────────────────────────────────────────────────────
    if cfg["model_search"]["include_deep_learning"]:
        log.info("\n" + "="*60)
        log.info("  DEEP LEARNING MODEL SEARCH")
        log.info("="*60)

        n_features = X_seq.shape[2]
        dl_configs = [
            ("LSTM",        LSTMClassifier,        {"n_classes": 5, "hidden_size": 128, "num_layers": 2,  "dropout": 0.3}),
            ("BiLSTM",      LSTMClassifier,        {"n_classes": 5, "hidden_size": 128, "num_layers": 2,  "dropout": 0.3, "bidirectional": True}),
            ("Transformer", TransformerClassifier, {"n_classes": 5, "d_model": 64,      "nhead": 4,       "num_layers": 2, "dropout": 0.3}),
        ]

        for name, cls, kwargs in dl_configs:
            try:
                r = search_dl_model(name, cls, {**kwargs, "n_features": n_features}, X_seq, y_seq, cfg)
                results.append(r)
            except Exception as e:
                log.error(f"  {name} failed: {e}")

    # ── Leaderboard ───────────────────────────────────────────────────────────
    board = pd.DataFrame(results).sort_values("cv_f1_macro_mean", ascending=False)
    board = board.reset_index(drop=True)
    board.index += 1

    print("\n" + "="*70)
    print("  MODEL SEARCH LEADERBOARD  (ranked by CV F1 Macro)")
    print("="*70)
    print(board[["name", "type", "cv_f1_macro_mean", "cv_f1_macro_std", "elapsed_seconds"]].to_string())
    print("="*70)

    best = board.iloc[0]
    print(f"\n  🏆  BEST MODEL: {best['name']}  (F1={best['cv_f1_macro_mean']:.4f})")
    print(f"\n  Next step:")
    print(f"    python training/hyperparameter_tuning.py --model {best['name']}")
    print("="*70)

    # Save leaderboard locally
    board.to_csv("search_leaderboard.csv", index=False)
    log.info("Leaderboard saved → search_leaderboard.csv")


if __name__ == "__main__":
    main()
