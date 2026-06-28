"""
training/train_final.py
────────────────────────
Trains the final model on full train data (train + val),
evaluates on the held-out test set, generates all validation plots,
and saves the model to MLflow (artifacts stored on S3).

Usage:
    # With best params from HPO:
    python training/train_final.py --model LightGBM

    # Skip HPO — use defaults:
    python training/train_final.py --model LightGBM --no-hpo-params

    # Deep learning:
    python training/train_final.py --model LSTM
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import dagshub
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE

from models.model_registry import (
    get_sklearn_models, LSTMClassifier, TransformerClassifier,
    TimeSeriesDataset, get_device,
)
from utils.mlflow_utils import setup_mlflow, load_config, log_dataset_info
from utils.plotting import save_all_plots, plot_learning_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DL_MODELS = {"LSTM", "BiLSTM", "Transformer"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_splits(cfg: dict, ticker_id: str):
    dc = cfg["data"]
    fc = cfg["features"]
    feat_path = Path(fc["featured_path"]) /  f"{ticker_id}/features.parquet"
    seq_path  = Path(fc["featured_path"]) / f"{ticker_id}/sequences.npz"
    col_path  = Path(fc["featured_path"]) / f"{ticker_id}/feature_columns.json"

    df = pd.read_parquet(feat_path)
    with open(col_path) as f:
        feature_cols = json.load(f)

    n         = len(df)
    train_end = int(n * dc["train_ratio"])
    val_end   = int(n * (dc["train_ratio"] + dc["val_ratio"]))

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    # Tabular
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["label"].values.astype(np.int64)
    X_val   = df_val[feature_cols].values.astype(np.float32)
    y_val   = df_val["label"].values.astype(np.int64)
    X_test  = df_test[feature_cols].values.astype(np.float32)
    y_test  = df_test["label"].values.astype(np.int64)

    # Sequences for DL
    seqs = np.load(seq_path, allow_pickle=True)
    X_seq = seqs["X"].astype(np.float32)
    y_seq = seqs["y"].astype(np.int64)
    X_seq_train = X_seq[:train_end];  y_seq_train = y_seq[:train_end]
    X_seq_val   = X_seq[train_end:val_end]; y_seq_val = y_seq[train_end:val_end]
    X_seq_test  = X_seq[val_end:];    y_seq_test  = y_seq[val_end:]

    # Test metadata for plots
    test_dates   = df_test.index
    test_prices  = df_test["close"].values
    test_returns = df_test["next_log_return"].values

    log.info(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    return (
        X_train, y_train, X_val, y_val, X_test, y_test,
        X_seq_train, y_seq_train, X_seq_val, y_seq_val, X_seq_test, y_seq_test,
        feature_cols, df_train, df_val, df_test,
        test_dates, test_prices, test_returns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build final sklearn model with best params
# ─────────────────────────────────────────────────────────────────────────────

def build_sklearn_model(model_name: str, best_params: dict | None, cfg: dict):
    seed = cfg["training"]["random_seed"]

    if best_params is None:
        # Fall back to defaults
        return get_sklearn_models(seed)[model_name]

    # Reconstruct model with tuned params
    if model_name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**best_params, random_state=seed, n_jobs=-1, verbose=-1)

    elif model_name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(**best_params, random_state=seed, n_jobs=-1,
                                  use_label_encoder=False, eval_metric="mlogloss", verbosity=0)

    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**best_params, random_state=seed, n_jobs=-1)

    elif model_name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**best_params, random_state=seed)

    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**best_params, random_state=seed)),
        ])

    elif model_name == "SVM":
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(**best_params, random_state=seed, probability=True)),
        ])

    elif model_name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**best_params, random_state=seed, n_jobs=-1)

    else:
        return get_sklearn_models(seed)[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn training
# ─────────────────────────────────────────────────────────────────────────────

def train_sklearn(model_name: str, best_params: dict | None, cfg: dict, data: tuple, full_training: bool = False):
    (X_train, y_train, X_val, y_val, X_test, y_test,
     _, _, _, _, _, _,
     feature_cols, df_train, df_val, df_test,
     test_dates, test_prices, test_returns) = data

    seed = cfg["training"]["random_seed"]

    

    if full_training:
        log.info("Full training mode: using train + val=test for final model.")
        X_trainval = np.concatenate([X_train, X_val, X_test], axis=0)
        y_trainval = np.concatenate([y_train, y_val, y_test], axis=0)
    else:
        # Combine train + val for final model (test is truly held-out)
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)

    # Handle class imbalance with SMOTE on training portion only
    log.info("Applying SMOTE for class balancing...")
    try:
        sm = SMOTE(random_state=seed, k_neighbors=5)
        X_trainval_bal, y_trainval_bal = sm.fit_resample(X_trainval, y_trainval)
        log.info(f"  SMOTE: {len(y_trainval)} → {len(y_trainval_bal)} samples")
    except Exception as e:
        log.warning(f"SMOTE failed ({e}), training without resampling")
        X_trainval_bal, y_trainval_bal = X_trainval, y_trainval

    model = build_sklearn_model(model_name, best_params, cfg)

    log.info(f"Training {model_name}...")
    model.fit(X_trainval_bal, y_trainval_bal)

    # Evaluate
    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics_val  = _compute_metrics(y_val,  y_val_pred,  prefix="val")
    metrics_test = _compute_metrics(y_test, y_test_pred, prefix="test")

    log.info(f"\n  Validation:\n{classification_report(y_val, y_val_pred, target_names=['StrongSell','Sell','Hold','Buy','StrongBuy'])}")
    log.info(f"\n  Test:\n{classification_report(y_test, y_test_pred, target_names=['StrongSell','Sell','Hold','Buy','StrongBuy'])}")

    return model, y_val_pred, y_test_pred, {**metrics_val, **metrics_test}


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch training
# ─────────────────────────────────────────────────────────────────────────────

def train_dl(model_name: str, best_params: dict | None, cfg: dict, data: tuple):
    (_, _, _, _, _, _,
     X_seq_train, y_seq_train, X_seq_val, y_seq_val, X_seq_test, y_seq_test,
     feature_cols, df_train, df_val, df_test,
     test_dates, test_prices, test_returns) = data

    tc     = cfg["training"]["dl"]
    device = get_device(tc["device"])
    seed   = cfg["training"]["random_seed"]
    torch.manual_seed(seed)

    n_feat = X_seq_train.shape[2]

    # Scale
    scaler   = StandardScaler()
    Xtr_sc   = scaler.fit_transform(X_seq_train.reshape(-1, n_feat)).reshape(X_seq_train.shape).astype(np.float32)
    Xvl_sc   = scaler.transform(X_seq_val.reshape(-1, n_feat)).reshape(X_seq_val.shape).astype(np.float32)
    Xte_sc   = scaler.transform(X_seq_test.reshape(-1, n_feat)).reshape(X_seq_test.shape).astype(np.float32)

    # Class weights
    class_counts  = np.bincount(y_seq_train, minlength=5)
    class_weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32).to(device)

    # Resolve params
    p = best_params or {}
    batch_size = p.get("batch_size", tc["batch_size"])
    lr         = p.get("lr",         tc["learning_rate"])

    if model_name in ("LSTM", "BiLSTM"):
        model = LSTMClassifier(
            n_features=n_feat, n_classes=5,
            hidden_size=p.get("hidden_size", 128),
            num_layers=p.get("num_layers", 2),
            dropout=p.get("dropout", 0.3),
            bidirectional=(model_name == "BiLSTM"),
        ).to(device)
    elif model_name == "Transformer":
        model = TransformerClassifier(
            n_features=n_feat, n_classes=5,
            d_model=p.get("d_model", 64),
            nhead=p.get("nhead", 4),
            num_layers=p.get("num_layers", 2),
            dim_feedforward=p.get("dim_feedforward", 256),
            dropout=p.get("dropout", 0.3),
        ).to(device)

    train_dl_ = DataLoader(TimeSeriesDataset(Xtr_sc, y_seq_train), batch_size=batch_size, shuffle=False)
    val_dl_   = DataLoader(TimeSeriesDataset(Xvl_sc, y_seq_val),   batch_size=batch_size, shuffle=False)
    test_dl_  = DataLoader(TimeSeriesDataset(Xte_sc, y_seq_test),  batch_size=batch_size, shuffle=False)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1   = 0.0
    best_state    = None
    patience_left = tc["patience"]

    log.info(f"Training {model_name} on {device}...")
    for epoch in range(tc["max_epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_dl_:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl_:
                xb, yb = xb.to(device), yb.to(device)
                out  = model(xb)
                val_losses.append(criterion(out, yb).item())
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(yb.cpu().numpy())

        val_f1  = f1_score(trues, preds, average="macro", zero_division=0)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        mlflow.log_metric("train_loss", np.mean(train_losses), step=epoch)
        mlflow.log_metric("val_loss",   val_loss,              step=epoch)
        mlflow.log_metric("val_f1",     val_f1,                step=epoch)

        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = tc["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1:3d}  train_loss={np.mean(train_losses):.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

    # Load best checkpoint
    model.load_state_dict(best_state)

    # Final predictions
    model.eval()
    def predict(dl):
        all_preds = []
        with torch.no_grad():
            for xb, _ in dl:
                all_preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        return np.array(all_preds)

    y_val_pred  = predict(val_dl_)
    y_test_pred = predict(test_dl_)

    metrics_val  = _compute_metrics(y_seq_val,  y_val_pred,  prefix="val")
    metrics_test = _compute_metrics(y_seq_test, y_test_pred, prefix="test")

    return model, scaler, y_val_pred, y_test_pred, {**metrics_val, **metrics_test}, history


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_f1_macro":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
        f"{prefix}_f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_accuracy":        accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_acc":    balanced_accuracy_score(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         required=False, help="Model name")
    #parser.add_argument("--no-hpo-params", action="store_true",
    #                    help="Skip loading HPO params — use default hyperparameters")
    parser.add_argument("--ticker",         required=False, help="pair ticker symbol e.g EURUSD=X, GBPUSD=X, etc.")
    parser.add_argument("--full",         required=False, help="Set to True to train on all data (train + val + test) but metrics will not be representative as there will be data leakage. Default is False.")
    
    args = parser.parse_args()
    full_training = args.full.lower() == "true" if args.full else False
    
    cfg = load_config()
    setup_mlflow(cfg)

    tickers = [args.ticker] if args.ticker else cfg["data"]["tickers"]
    models = [args.model] if args.model else cfg["training"]["models"]

    for model_name, ticker in zip(models, tickers):
        ticker_id = ticker.lower().split('=')[0]

    #if args.model is None:
    #    if Path("search_leaderboard.csv").exists():
    #        model = pd.read_csv("search_leaderboard.csv").iloc[0]["name"]
    #        log.info(f"No model specified. Using best from HPO: {model}")
    #        args.model = model
    #    else:
    #        log.error("No model specified and search_leaderboard.csv not found. Please run hyperparameter tuning first or specify a model.")
    #        sys.exit(1)

    # Load best params if available
        best_params = None
        param_file = Path(f"best_params/{ticker_id}.json")
        #param_file = Path(f"best_params.json")
        if param_file.exists():
            with open(param_file) as f:
                best_params = json.load(f)["params"]
            log.info(f"Loaded best params from {param_file}: {best_params}")
        else:
            log.warning(f"No HPO params found at {param_file}. Using defaults.")

        # Load data
        data = load_all_splits(cfg, ticker_id)
        (X_train, y_train, X_val, y_val, X_test, y_test,
        X_seq_train, y_seq_train, X_seq_val, y_seq_val, X_seq_test, y_seq_test,
        feature_cols, df_train, df_val, df_test,
        test_dates, test_prices, test_returns) = data

        # Select y_true for test (tabular vs sequence models have slightly different lengths)
        is_dl = args.model in DL_MODELS
        y_true_test = y_seq_test if is_dl else y_test
        y_true_val  = y_seq_val  if is_dl else y_val

        with mlflow.start_run(run_name=f"final_{ticker_id}_{model_name}") as run:
            mlflow.set_tag("model_name",  model_name)
            mlflow.set_tag("model_type",  "pytorch" if is_dl else "sklearn")
            mlflow.set_tag("stage",       "final_training")
            mlflow.set_tag("hpo_applied", str(best_params is not None))
            mlflow.set_tag("ticker",     ticker_id)

            # Log data info
            log_dataset_info(df_train, "train", cfg)
            log_dataset_info(df_val,   "val",   cfg)
            log_dataset_info(df_test,  "test",  cfg)
            mlflow.log_param("n_features", len(feature_cols))
            if best_params:
                mlflow.log_params(best_params)

            # ── Train ──────────────────────────────────────────────────────────
            learning_history = None
            scaler = None

            if is_dl:
                model, scaler, y_val_pred, y_test_pred, metrics, learning_history = \
                    train_dl(model_name, best_params, cfg, data)
            else:
                model, y_val_pred, y_test_pred, metrics = \
                    train_sklearn(model_name, best_params, cfg, data, full_training=full_training)

            # ── Log metrics ────────────────────────────────────────────────────
            mlflow.log_metrics(metrics)
            log.info(f"\n  FINAL METRICS:")
            for k, v in metrics.items():
                log.info(f"    {k}: {v:.4f}")

            # ── Generate plots ─────────────────────────────────────────────────
            plots_dir = Path(f"{cfg['output']['plots_dir']}/{ticker_id}")
            plots_dir.mkdir(parents=True, exist_ok=True)

            save_all_plots(
                y_true_val=y_true_val,
                y_pred_val=y_val_pred,
                y_true_test=y_true_test,
                y_pred_test=y_test_pred,
                test_dates=test_dates,
                test_prices=test_prices,
                test_returns=test_returns,
                feature_names=feature_cols,
                model=model,
                plots_dir=str(plots_dir),
                learning_history=learning_history,
            )

            # Log all plots as MLflow artifacts
            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file), artifact_path="plots")

            # ── Save model to MLflow ────────────────────────────────────────────
            log.info(f"Saving {ticker_id} model to MLflow...")
            model_dir = Path(f"{cfg['output']['model_dir']}/{ticker_id}")
            model_dir.mkdir(parents=True, exist_ok=True)

            if is_dl:
                # Save as PyTorch
                model_path = model_dir / f"{ticker_id}_{model_name}_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scaler_mean":      scaler.mean_,
                    "scaler_scale":     scaler.scale_,
                    "model_name":       f"{ticker_id}_{model_name}",
                    "n_features":       X_seq_train.shape[2],
                    "n_classes":        5,
                    "best_params":      best_params,
                    "feature_cols":     feature_cols,
                    "metrics":          metrics,
                }, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")

                # Also register as MLflow model for serving
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model_torch",
                    registered_model_name=f"{ticker_id}_{model_name}",
                )

            else:
                # Sklearn — use mlflow.sklearn
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model_sklearn",
                    registered_model_name=f"{ticker_id}_{model_name}",
                )
                # Also save locally
                import joblib
                model_path = model_dir / f"model_best.joblib"
                joblib.dump({
                    "model":        model,
                    "feature_cols": feature_cols,
                    "metrics":      metrics,
                }, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")

            # ── Log feature columns ─────────────────────────────────────────────
            col_path = model_dir / "feature_columns.json"
            with open(col_path, "w") as f:
                json.dump(feature_cols, f, indent=2)
            mlflow.log_artifact(str(col_path))

            run_id = mlflow.active_run().info.run_id

        # ── Summary ──────────────────────────────────────────────────────────────
        print("\n" + "="*65)
        print(f"  {ticker_id.upper()}   TRAINING COMPLETE:  {model_name.upper()}")
        print("="*65)
        print(f"  Test F1 Macro:       {metrics['test_f1_macro']:.4f}")
        print(f"  Test Accuracy:       {metrics['test_accuracy']:.4f}")
        print(f"  Test Balanced Acc:   {metrics['test_balanced_acc']:.4f}")
        print(f"  MLflow Run ID:       {run_id}")
        print(f"  Plots saved:         {plots_dir}/")
        print(f"  Model saved:         {model_dir}/")
        print("="*65)
        print("  Model is registered in MLflow Model Registry.")
        print("  Artifacts (plots, model) are stored on mlflow.")
        print("="*65)


if __name__ == "__main__":
    main()
