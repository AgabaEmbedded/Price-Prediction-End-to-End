"""
training/hyperparameter_tuning.py
───────────────────────────────────
Bayesian hyperparameter optimisation with Optuna.
Every trial is logged as a child run in MLflow.

Usage:
    python training/hyperparameter_tuning.py --model LightGBM
    python training/hyperparameter_tuning.py --model LSTM

Supported model names (pass the exact name from the leaderboard):
    Sklearn: LogisticRegression, SVM, KNN, RandomForest, ExtraTrees,
             GradientBoosting, AdaBoost, Bagging, XGBoost, LightGBM
    DL:      LSTM, BiLSTM, Transformer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

from models.model_registry import (
    LSTMClassifier, TransformerClassifier,
    TimeSeriesDataset, get_device,
)
from utils.mlflow_utils import setup_mlflow, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (reused from search_models)
# ─────────────────────────────────────────────────────────────────────────────

def load_train_val_data(cfg: dict):
    dc = cfg["data"]
    fc = cfg["features"]
    feat_path = Path(fc["featured_path"]) / "featured_eurusd.parquet"
    seq_path  = Path(fc["featured_path"]) / "sequences.npz"
    col_path  = Path(fc["featured_path"]) / "feature_columns.json"

    df = pd.read_parquet(feat_path)
    with open(col_path) as f:
        feature_cols = json.load(f)

    n = len(df)
    train_end = int(n * dc["train_ratio"])
    val_end   = int(n * (dc["train_ratio"] + dc["val_ratio"]))

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["label"].values.astype(np.int64)

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]

    seqs = np.load(seq_path, allow_pickle=True)
    X_seq_train = seqs["X"][:train_end]
    y_seq_train = seqs["y"][:train_end]
    X_seq_val   = seqs["X"][train_end:val_end]
    y_seq_val   = seqs["y"][train_end:val_end]

    return (X_train, y_train, X_val, y_val,
            X_seq_train, y_seq_train, X_seq_val, y_seq_val,
            feature_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn objective factories
# ─────────────────────────────────────────────────────────────────────────────

def make_sklearn_objective(model_name: str, X: np.ndarray, y: np.ndarray, cfg: dict):
    """Returns an Optuna objective function for the given sklearn model."""
    n_folds = cfg["hpo"]["cv_folds"]
    seed    = cfg["training"]["random_seed"]
    tscv    = TimeSeriesSplit(n_splits=n_folds)

    def objective(trial: optuna.Trial) -> float:
        # ── Suggest hyperparameters per model ──────────────────────────────

        if model_name == "LightGBM":
            import lightgbm as lgb
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 200),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
                "class_weight": "balanced",
                "random_state": seed,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)

        elif model_name == "XGBoost":
            import xgboost as xgb
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma":            trial.suggest_float("gamma", 0, 5),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
                "n_jobs": -1,
                "verbosity": 0,
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
            }
            model = xgb.XGBClassifier(**params)

        elif model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth":    trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                "class_weight": "balanced",
                "random_state": seed,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        elif model_name == "GradientBoosting":
            from sklearn.ensemble import GradientBoostingClassifier
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth":     trial.suggest_int("max_depth", 2, 7),
                "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "random_state":  seed,
            }
            model = GradientBoostingClassifier(**params)

        elif model_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            params = {
                "C":       trial.suggest_float("C", 1e-4, 100, log=True),
                "solver":  trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": 2000,
                "multi_class": "multinomial",
                "class_weight": "balanced",
                "random_state": seed,
            }
            from sklearn.preprocessing import StandardScaler
            model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**params))])

        elif model_name == "SVM":
            from sklearn.svm import SVC
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            params = {
                "C":      trial.suggest_float("C", 1e-3, 100, log=True),
                "gamma":  trial.suggest_categorical("gamma", ["scale", "auto"]),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
                "class_weight": "balanced",
                "probability": True,
                "random_state": seed,
            }
            model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(**params))])

        elif model_name == "ExtraTrees":
            from sklearn.ensemble import ExtraTreesClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth":    trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "class_weight": "balanced",
                "random_state": seed,
                "n_jobs": -1,
            }
            model = ExtraTreesClassifier(**params)

        else:
            raise ValueError(f"No HPO search space defined for '{model_name}'.")

        # Cross-validate
        scores = cross_val_score(model, X, y, cv=tscv, scoring="f1_macro", n_jobs=-1)
        return float(scores.mean())

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# DL objective factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dl_objective(
    model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    cfg: dict,
):
    tc     = cfg["training"]["dl"]
    device = get_device(tc["device"])

    # Scale once outside the objective (expensive)
    n_feat  = X_train.shape[2]
    scaler  = StandardScaler()
    Xtr_2d  = X_train.reshape(-1, n_feat)
    Xvl_2d  = X_val.reshape(-1, n_feat)
    Xtr_sc  = scaler.fit_transform(Xtr_2d).reshape(X_train.shape).astype(np.float32)
    Xvl_sc  = scaler.transform(Xvl_2d).reshape(X_val.shape).astype(np.float32)

    class_counts  = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(1.0 / (class_counts + 1), dtype=torch.float32).to(device)

    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout    = trial.suggest_float("dropout", 0.1, 0.5)

        if model_name in ("LSTM", "BiLSTM"):
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
            num_layers  = trial.suggest_int("num_layers", 1, 4)
            model = LSTMClassifier(
                n_features=n_feat,
                n_classes=5,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=(model_name == "BiLSTM"),
            ).to(device)

        elif model_name == "Transformer":
            d_model = trial.suggest_categorical("d_model", [32, 64, 128])
            nhead   = trial.suggest_categorical("nhead", [2, 4, 8])
            if d_model % nhead != 0:
                raise optuna.exceptions.TrialPruned()
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dim_ff     = trial.suggest_categorical("dim_feedforward", [128, 256, 512])
            model = TransformerClassifier(
                n_features=n_feat,
                n_classes=5,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_ff,
                dropout=dropout,
            ).to(device)
        else:
            raise ValueError(f"Unknown DL model: {model_name}")

        train_dl = DataLoader(TimeSeriesDataset(Xtr_sc, y_train), batch_size=batch_size, shuffle=False)
        val_dl   = DataLoader(TimeSeriesDataset(Xvl_sc, y_val),   batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = 0.0
        patience_left = tc["patience"]

        for epoch in range(tc["max_epochs"]):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                    trues.extend(yb.numpy())

            val_f1 = f1_score(trues, preds, average="macro", zero_division=0)
            trial.report(val_f1, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_left = tc["patience"]
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        return best_f1

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# MLflow callback for Optuna
# ─────────────────────────────────────────────────────────────────────────────

class MLflowCallback:
    def __init__(self, model_name: str, experiment_name: str):
        self.model_name      = model_name
        self.experiment_name = experiment_name

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        with mlflow.start_run(run_name=f"hpo_{self.model_name}_trial{trial.number}"):
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("stage", "hpo")
            mlflow.set_tag("trial_number", str(trial.number))
            mlflow.log_params(trial.params)
            mlflow.log_metric("val_f1_macro", trial.value)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

DL_MODELS = {"LSTM", "BiLSTM", "Transformer"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name from leaderboard")
    args = parser.parse_args()

    cfg       = load_config()
    hpo_cfg   = cfg["hpo"]
    client    = setup_mlflow(cfg)
    exp_name  = cfg["mlflow"]["experiment_name"]

    (X_train, y_train, X_val, y_val,
     X_seq_train, y_seq_train,
     X_seq_val,   y_seq_val,
     feature_cols) = load_train_val_data(cfg)

    is_dl = args.model in DL_MODELS

    if is_dl:
        objective = make_dl_objective(
            args.model,
            X_seq_train, y_seq_train,
            X_seq_val,   y_seq_val,
            cfg,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    else:
        # Combine train + val for CV (the CV folds handle temporal splits)
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)
        objective = make_sklearn_objective(args.model, X_all, y_all, cfg)
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction=hpo_cfg["direction"],
        sampler=optuna.samplers.TPESampler(seed=cfg["training"]["random_seed"]),
        pruner=pruner,
        study_name=f"{args.model}_hpo",
    )

    mlflow_cb = MLflowCallback(args.model, exp_name)

    log.info(f"\nStarting HPO for {args.model}  ({hpo_cfg['n_trials']} trials)")
    study.optimize(
        objective,
        n_trials=hpo_cfg["n_trials"],
        timeout=hpo_cfg["timeout"],
        callbacks=[mlflow_cb],
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    log.info(f"\n  Best trial: #{best_trial.number}  f1={best_trial.value:.4f}")
    log.info(f"  Best params: {best_trial.params}")

    # Save best params to JSON for use by train_final.py
    best_params_path = Path(f"best_params_{args.model}.json")
    with open(best_params_path, "w") as f:
        json.dump({"model": args.model, "params": best_trial.params, "val_f1": best_trial.value}, f, indent=2)
    log.info(f"  Best params saved → {best_params_path}")

    # Log best overall to MLflow
    with mlflow.start_run(run_name=f"hpo_best_{args.model}"):
        mlflow.set_tag("model_name", args.model)
        mlflow.set_tag("stage", "hpo_best")
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_val_f1_macro", best_trial.value)
        mlflow.log_artifact(str(best_params_path))

    print("\n" + "="*60)
    print(f"  HPO COMPLETE:  {args.model}")
    print(f"  Best F1 Macro: {best_trial.value:.4f}")
    print(f"  Best params saved → {best_params_path}")
    print(f"\n  Next step:")
    print(f"    python training/train_final.py --model {args.model}")
    print("="*60)


if __name__ == "__main__":
    main()
