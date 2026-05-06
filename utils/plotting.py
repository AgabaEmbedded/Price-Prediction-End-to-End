"""
utils/plotting.py
─────────────────
All validation, diagnostic, and trading-specific plots.

Functions:
  plot_confusion_matrix      — normalised confusion matrix
  plot_class_distribution    — label balance across splits
  plot_feature_importance    — for tree/GBDT models
  plot_learning_curves       — train vs val metric over time (DL)
  plot_predictions_over_time — predicted signal vs actual price
  plot_equity_curve          — simulated trading equity curve
  plot_per_class_metrics     — precision/recall/f1 per class
  save_all_plots             — convenience wrapper
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c8c8d0",
    "xtick.color":      "#c8c8d0",
    "ytick.color":      "#c8c8d0",
    "text.color":       "#e8e8f0",
    "grid.color":       "#2a2d3d",
    "grid.alpha":       0.6,
    "lines.linewidth":  1.8,
    "font.family":      "DejaVu Sans",
})

CLASS_NAMES  = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
CLASS_COLORS = ["#e74c3c", "#e67e22", "#95a5a6", "#2ecc71", "#27ae60"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved plot → {path}")


def _align_to_predictions(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    returns: np.ndarray,
    n_preds: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Align dates/prices/returns arrays to match the number of predictions.

    For DL sequence models, build_sequences() consumes the first `seq_len`
    rows to form the initial window, producing fewer predictions than the raw
    test split has rows (e.g. 608 rows → 578 predictions with seq_len=30).
    Each prediction corresponds to the LAST date in its window, so we align
    by taking the TAIL of the raw arrays.

    For sklearn models the lengths already match, so this is a no-op.
    """
    total = len(dates)
    if n_preds == total:
        return dates, prices, returns

    if n_preds > total:
        raise ValueError(
            f"n_preds ({n_preds}) > len(dates) ({total}). "
            "Something is wrong with the data pipeline."
        )

    offset = total - n_preds
    log.debug(f"  Aligning arrays: dropping first {offset} rows (sequence window offset)")
    return dates[offset:], prices[offset:], returns[offset:]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    path: str | None = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap="Blues", ax=ax, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (y, label) in zip(axes, [(y_train, "Train"), (y_val, "Val"), (y_test, "Test")]):
        counts = np.bincount(y, minlength=5)
        bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor="#0f1117", linewidth=1)
        ax.set_title(f"{label}  (n={len(y):,})", fontsize=11)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=8,
            )
    fig.suptitle("Label Distribution Across Splits", fontsize=13, y=1.02)
    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 30,
    path: str | None = None,
) -> plt.Figure | None:
    """Works for tree-based sklearn models, LightGBM, and XGBoost."""
    # Unwrap sklearn Pipeline if needed
    est = model
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_"):
                est = step
                break

    if not hasattr(est, "feature_importances_"):
        log.warning("Model does not expose feature_importances_ — skipping feature importance plot.")
        return None

    importances = pd.Series(est.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.28)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    ax.barh(top.index, top.values, color=colors, edgecolor="#0f1117")
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13)
    ax.grid(axis="x")
    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Learning Curves (DL)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    history: dict,
    path: str | None = None,
) -> plt.Figure:
    """
    Expects history dict with keys:
      train_loss, val_loss, val_f1  (required)
      train_f1                      (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#3498db")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#e74c3c")
    ax1.set_title("Loss Curves", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_f1"], label="Val F1 Macro", color="#2ecc71")
    if "train_f1" in history:
        ax2.plot(epochs, history["train_f1"], label="Train F1 Macro", color="#9b59b6")
    ax2.set_title("F1 Macro Score", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle("Learning Curves", fontsize=14, y=1.02)
    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Predictions Over Time
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions_over_time(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | None = None,
) -> plt.Figure:
    """
    Overlay buy/sell/hold signals on the price chart.
    All four arrays must be the same length — use _align_to_predictions()
    before calling this if dates/prices come from the raw test split.
    """
    # Convert to numpy for safe boolean indexing
    dates_arr  = np.array(dates)
    prices_arr = np.asarray(prices)
    y_pred_arr = np.asarray(y_pred)
    y_true_arr = np.asarray(y_true)

    if len(dates_arr) != len(y_pred_arr):
        raise ValueError(
            f"plot_predictions_over_time: dates length ({len(dates_arr)}) != "
            f"y_pred length ({len(y_pred_arr)}). Call _align_to_predictions() first."
        )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Price line ──────────────────────────────────────────────────────────
    ax1.plot(dates_arr, prices_arr, color="#7f8c8d", linewidth=1, alpha=0.8, label="Close Price")

    signal_colors  = {0: "#e74c3c", 1: "#e67e22", 2: "#95a5a6", 3: "#2ecc71", 4: "#27ae60"}
    signal_markers = {0: "v",       1: "v",       2: "o",       3: "^",       4: "^"}

    for cls in range(5):
        mask = y_pred_arr == cls
        if mask.any():
            ax1.scatter(
                dates_arr[mask], prices_arr[mask],
                c=signal_colors[cls], marker=signal_markers[cls],
                s=30, alpha=0.7, label=CLASS_NAMES[cls], zorder=5,
            )

    ax1.set_ylabel("EUR/USD Price", fontsize=11)
    ax1.set_title("Predicted Signals Overlaid on Price", fontsize=13)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
    ax1.grid(True)

    # ── Rolling accuracy ─────────────────────────────────────────────────────
    correct     = (y_pred_arr == y_true_arr).astype(float)
    rolling_acc = pd.Series(correct, index=dates_arr).rolling(30).mean()
    ax2.fill_between(dates_arr, rolling_acc, alpha=0.5, color="#3498db")
    ax2.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1, label="50% baseline")
    ax2.set_ylabel("30d Rolling Acc", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.grid(True)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-class Metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | None = None,
) -> plt.Figure:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(5)), zero_division=0,
    )

    x     = np.arange(5)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, precision, width, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x,         recall,    width, label="Recall",    color="#2ecc71", alpha=0.85)
    ax.bar(x + width, f1,        width, label="F1",        color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-Class Precision / Recall / F1", fontsize=13)
    ax.legend()
    ax.grid(axis="y")

    for i, s in enumerate(support):
        ax.text(i, -0.07, f"n={s}", ha="center", fontsize=8, color="#aaa")

    fig.tight_layout()
    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Equity Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curve(
    dates: pd.DatetimeIndex,
    actual_returns: np.ndarray,
    y_pred: np.ndarray,
    path: str | None = None,
) -> plt.Figure:
    """
    Simulate a simple long/short/flat strategy from predicted signals:
      4 / 3  → long  (+return)
      1 / 0  → short (-return)
      2      → flat  (0)

    Compares against buy-and-hold. No transaction costs.
    All arrays must be the same length — use _align_to_predictions() first.
    """
    dates_arr   = np.array(dates)
    returns_arr = np.asarray(actual_returns)
    y_pred_arr  = np.asarray(y_pred)

    signal = np.where(y_pred_arr >= 3,  1.0,
             np.where(y_pred_arr <= 1, -1.0, 0.0))

    strat_returns = signal * returns_arr
    bh_returns    = returns_arr

    strat_equity = np.exp(np.cumsum(strat_returns))
    bh_equity    = np.exp(np.cumsum(bh_returns))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(dates_arr, strat_equity, color="#2ecc71", label="Model Strategy", linewidth=2)
    ax1.plot(dates_arr, bh_equity,    color="#3498db", label="Buy & Hold",     linewidth=1.5, alpha=0.7)
    ax1.axhline(1.0, color="#7f8c8d", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Cumulative Return (1 = start)", fontsize=11)
    ax1.set_title("Simulated Equity Curve  (no transaction costs — indicative only)", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    roll_max = np.maximum.accumulate(strat_equity)
    drawdown = (strat_equity - roll_max) / (roll_max + 1e-10)
    ax2.fill_between(dates_arr, drawdown, 0, color="#e74c3c", alpha=0.5)
    ax2.set_ylabel("Drawdown", fontsize=9)
    ax2.grid(True)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    fig.tight_layout()

    total_ret = strat_equity[-1] - 1
    bh_ret    = bh_equity[-1] - 1
    max_dd    = drawdown.min()
    sharpe    = (strat_returns.mean() / (strat_returns.std() + 1e-10)) * np.sqrt(252)
    ann_ret   = np.exp(strat_returns.mean() * 252) - 1

    stats_str = (
        f"Strat: {total_ret*100:+.1f}%  |  BH: {bh_ret*100:+.1f}%  |  "
        f"Ann.Ret: {ann_ret*100:.1f}%  |  Sharpe: {sharpe:.2f}  |  MaxDD: {max_dd*100:.1f}%"
    )
    fig.text(0.5, -0.02, stats_str, ha="center", fontsize=9, color="#aaa")

    if path:
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def save_all_plots(
    y_true_val:  np.ndarray,
    y_pred_val:  np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    test_dates:  pd.DatetimeIndex,
    test_prices: np.ndarray,
    test_returns: np.ndarray,
    feature_names: list[str],
    model,
    plots_dir: str = "plots/",
    learning_history: dict | None = None,
) -> None:
    """
    Generate and save all validation plots.

    Automatically aligns test_dates / test_prices / test_returns to match
    the length of y_pred_test, so this works correctly for both sklearn
    models (no offset) and DL sequence models (seq_len offset).
    """
    log.info("Generating validation plots...")
    d = Path(plots_dir)

    # ── Align dates/prices/returns to prediction length ──────────────────────
    # DL sequence models produce fewer predictions than raw test rows because
    # build_sequences() needs seq_len rows for the first window.
    # _align_to_predictions() is a no-op when lengths already match (sklearn).
    n_preds = len(y_pred_test)
    dates_aligned, prices_aligned, returns_aligned = _align_to_predictions(
        test_dates, test_prices, test_returns, n_preds,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true_val, y_pred_val,
        title="Confusion Matrix — Validation Set",
        path=d / "confusion_matrix_val.png",
    )
    plot_confusion_matrix(
        y_true_test, y_pred_test,
        title="Confusion Matrix — Test Set",
        path=d / "confusion_matrix_test.png",
    )
    plot_per_class_metrics(
        y_true_test, y_pred_test,
        path=d / "per_class_metrics.png",
    )
    plot_predictions_over_time(
        dates_aligned, prices_aligned, y_true_test, y_pred_test,
        path=d / "predictions_over_time.png",
    )
    plot_equity_curve(
        dates_aligned, returns_aligned, y_pred_test,
        path=d / "equity_curve.png",
    )
    plot_feature_importance(
        model, feature_names,
        path=d / "feature_importance.png",
    )
    if learning_history:
        plot_learning_curves(
            learning_history,
            path=d / "learning_curves.png",
        )

    log.info(f"All plots saved to {d}/")