"""
models/model_registry.py
─────────────────────────
All model definitions in one place.

Sklearn models:
  - LogisticRegression
  - RandomForest
  - GradientBoosting (sklearn)
  - XGBoost (via sklearn API)
  - LightGBM
  - SVM
  - KNN
  - ExtraTrees
  - AdaBoost
  - BaggingClassifier

PyTorch models:
  - LSTMClassifier
  - TransformerClassifier
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional heavy deps — skip gracefully if not installed
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_sklearn_models(random_seed: int = 42) -> dict:
    """
    Returns a dict of {name: sklearn estimator}.
    Models that need scaling are wrapped in a Pipeline.
    """
    models = {}

    # ── Linear ──────────────────────────────────────────────────────────────
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=random_seed,
            class_weight="balanced",
        )),
    ])

    # ── SVM ──────────────────────────────────────────────────────────────────
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_seed,
        )),
    ])

    # ── KNN ──────────────────────────────────────────────────────────────────
    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(n_neighbors=15)),
    ])

    # ── Tree ensembles ────────────────────────────────────────────────────────
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=random_seed,
    )

    models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=random_seed,
        algorithm="SAMME",
    )

    models["Bagging"] = BaggingClassifier(
        n_estimators=100,
        random_state=random_seed,
        n_jobs=-1,
    )

    # ── Gradient Boosting libraries ──────────────────────────────────────────
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_seed,
            n_jobs=-1,
            verbosity=0,
        )

    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch models
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Stacked LSTM for sequence classification.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_classes)  — logits
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.norm  = nn.LayerNorm(hidden_size * directions)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden_size * directions, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        out = out[:, -1, :]            # take last timestep
        out = self.norm(out)
        out = self.drop(out)
        return self.head(out)


class TransformerClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_classes)  — logits
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_seq_len: int = 200,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        batch, seq, _ = x.shape
        pos = torch.arange(seq, device=x.device).unsqueeze(0)   # (1, seq)
        x = self.input_proj(x) + self.pos_enc(pos)              # (batch, seq, d_model)
        x = self.encoder(x)                                      # (batch, seq, d_model)
        x = x.mean(dim=1)                                        # global average pool
        x = self.norm(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Wraps numpy arrays into a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_cfg)
