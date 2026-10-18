"""Evaluation metrics and performance plots for PPI link prediction.

Supervised models output interaction probabilities; topological heuristics
(L3, common neighbours, ...) output unbounded ranking scores. Both are
compared with threshold-free ranking metrics:

* **ROC-AUC** - probability that a random true interaction is ranked above a
  random non-interacting pair.
* **Average precision** (area under the precision-recall curve) - more
  informative than ROC-AUC when positives are rare, as in real interactome
  screening.
* **R-precision** - precision among the top *k* ranked pairs, where *k* is the
  number of true interactions.

Probabilistic models are additionally scored at a decision threshold
(precision, recall, F1, accuracy, Matthews correlation, confusion matrix).

Command line: recompute metrics and figures from a saved predictions file::

    python src/evaluate.py --predictions results/predictions.csv --output-dir results
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

if __package__ in (None, ""):  # allow `python src/evaluate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are written to disk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

#: Categorical colours, assigned in this fixed order (colour-vision-deficiency
#: safe for adjacent series).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}
STYLE = {
    "figure.facecolor": INK["surface"],
    "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"],
    "axes.edgecolor": INK["axis"],
    "axes.labelcolor": INK["secondary"],
    "axes.titlecolor": INK["primary"],
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": INK["grid"],
    "grid.linewidth": 0.6,
    "xtick.color": INK["muted"],
    "ytick.color": INK["muted"],
    "xtick.labelcolor": INK["secondary"],
    "ytick.labelcolor": INK["secondary"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK["primary"],
    "lines.linewidth": 1.8,
    "font.family": "sans-serif",
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "l3": "L3 paths",
    "common_neighbors": "Common neighbours",
    "jaccard": "Jaccard",
    "adamic_adar": "Adamic-Adar",
    "resource_allocation": "Resource allocation",
    "preferential_attachment": "Preferential attachment",
    "shortest_path": "Shortest path (inverted)",
}

#: Feature columns usable directly as unsupervised ranking scores, with the
#: sign that makes "higher = more likely to interact".
HEURISTIC_BASELINES: dict[str, float] = {
    "l3": 1.0,
    "adamic_adar": 1.0,
    "resource_allocation": 1.0,
    "common_neighbors": 1.0,
    "jaccard": 1.0,
    "preferential_attachment": 1.0,
    "shortest_path": -1.0,
}


@contextmanager
def plot_style() -> Iterator[None]:
    """Temporarily apply the project's matplotlib style."""
    with plt.rc_context(STYLE):
        yield


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").capitalize())


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def r_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision among the top-``k`` scored pairs, ``k`` = number of positives."""
    y_true = np.asarray(y_true)
    k = int(y_true.sum())
    if k == 0:
        return float("nan")
    top = np.argsort(-np.asarray(y_score), kind="stable")[:k]
    return float(y_true[top].mean())


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = 0.5,
) -> dict[str, float]:
    """Ranking metrics, plus threshold metrics when ``threshold`` is given.

    Parameters
    ----------
    y_true:
        Binary labels (1 = interaction).
    y_score:
        Scores where higher means "more likely to interact". Pass
        ``threshold=None`` for scores that are not probabilities.
    threshold:
        Probability cut-off for the threshold-dependent metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both classes")

    metrics: dict[str, float] = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "r_precision": r_precision(y_true, y_score),
    }
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update(
            {
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "mcc": matthews_corrcoef(y_true, y_pred),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return {k: float(v) for k, v in metrics.items()}


def heuristic_scores(X: pd.DataFrame, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Turn feature columns into unsupervised baseline scores."""
    names = names or [n for n in HEURISTIC_BASELINES if n in X.columns]
    return {name: HEURISTIC_BASELINES[name] * X[name].to_numpy(dtype=float) for name in names}


def evaluate_scores(
    y_true: np.ndarray,
    model_scores: Mapping[str, np.ndarray],
    baseline_scores: Mapping[str, np.ndarray] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Metric table for supervised models and unsupervised baselines.

    Returns
    -------
    DataFrame indexed by method, with a ``kind`` column (``model`` or
    ``heuristic``), sorted by ROC-AUC. Threshold metrics are NaN for
    heuristics.
    """
    rows = []
    for name, score in model_scores.items():
        rows.append({"method": name, "kind": "model", **compute_metrics(y_true, score, threshold)})
    for name, score in (baseline_scores or {}).items():
        rows.append({"method": name, "kind": "heuristic", **compute_metrics(y_true, score, None)})
    table = pd.DataFrame(rows).set_index("method")
    return table.sort_values("roc_auc", ascending=False)


def compute_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Drop in test ROC-AUC when each feature is shuffled (model-agnostic)."""
    result = permutation_importance(
        estimator, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _series_styles(names: list[str], kinds: Mapping[str, str]) -> dict[str, dict]:
    """Colour by entity in fixed palette order; heuristics drawn dashed."""
    return {
        name: {
            "color": PALETTE[i % len(PALETTE)],
            "linestyle": "-" if kinds.get(name, "model") == "model" else (0, (5, 2)),
        }
        for i, name in enumerate(names)
    }


def plot_roc_curves(
    y_true: np.ndarray,
    scores: Mapping[str, np.ndarray],
    kinds: Mapping[str, str] | None = None,
    title: str = "ROC curves on held-out interactions",
) -> plt.Figure:
    """ROC curve for each scorer; legend shows the AUC."""
    kinds = kinds or {}
    styles = _series_styles(list(scores), kinds)
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.plot([0, 1], [0, 1], color=INK["axis"], linewidth=1, linestyle=":", label="Chance")
        for name, score in scores.items():
            fpr, tpr, _ = roc_curve(y_true, score)
            auc = roc_auc_score(y_true, score)
            ax.plot(fpr, tpr, label=f"{display_name(name)}  ({auc:.3f})", **styles[name])
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right", title="Method  (ROC-AUC)", title_fontsize=9)
        ax.set_aspect("equal")
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    scores: Mapping[str, np.ndarray],
    kinds: Mapping[str, str] | None = None,
    title: str = "Precision-recall curves on held-out interactions",
) -> plt.Figure:
    """Precision-recall curve for each scorer; legend shows average precision."""
    kinds = kinds or {}
    styles = _series_styles(list(scores), kinds)
    prevalence = float(np.mean(y_true))
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        ax.axhline(prevalence, color=INK["axis"], linewidth=1, linestyle=":", label=f"Chance ({prevalence:.2f})")
        for name, score in scores.items():
            precision, recall, _ = precision_recall_curve(y_true, score)
            ap = average_precision_score(y_true, score)
            ax.step(recall, precision, where="post", label=f"{display_name(name)}  ({ap:.3f})", **styles[name])
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
        ax.set_title(title)
        ax.legend(loc="lower left", title="Method  (avg. precision)", title_fontsize=9)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion matrix",
    labels: tuple[str, str] = ("No interaction", "Interaction"),
) -> plt.Figure:
    """Annotated confusion matrix (counts and row percentages)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annotations = np.array(
        [[f"{cm[i, j]:,}\n{row_pct[i, j]:.1%}" for j in range(2)] for i in range(2)]
    )
    cmap = LinearSegmentedColormap.from_list("ppi_blues", SEQUENTIAL)
    with plot_style():
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        sns.heatmap(
            row_pct,
            annot=annotations,
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            linewidths=2,
            linecolor=INK["surface"],
            annot_kws={"fontsize": 11},
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        for text, value in zip(ax.texts, row_pct.ravel()):
            text.set_color("white" if value > 0.5 else INK["primary"])
        ax.grid(False)
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.set_title(title)
        ax.tick_params(length=0)
    return fig


def plot_feature_importance(
    importance: pd.DataFrame,
    title: str = "Permutation importance",
    xlabel: str = "Drop in ROC-AUC when shuffled",
) -> plt.Figure:
    """Horizontal bars of ``importance_mean`` with ``importance_std`` whiskers."""
    data = importance.sort_values("importance_mean")
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(data) + 1.2))
        ax.barh(
            data["feature"],
            data["importance_mean"],
            xerr=data.get("importance_std"),
            color=PALETTE[0],
            height=0.7,
            error_kw={"ecolor": INK["muted"], "elinewidth": 1, "capsize": 0},
        )
        ax.axvline(0, color=INK["axis"], linewidth=1)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel=xlabel, ylabel="")
        ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def split_prediction_columns(predictions: pd.DataFrame) -> tuple[dict, dict]:
    """Read ``model__<name>`` / ``heuristic__<name>`` columns of a predictions file."""
    models, heuristics = {}, {}
    for column in predictions.columns:
        kind, _, name = column.partition("__")
        if kind == "model":
            models[name] = predictions[column].to_numpy(dtype=float)
        elif kind == "heuristic":
            heuristics[name] = predictions[column].to_numpy(dtype=float)
    return models, heuristics


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Evaluate saved PPI predictions.")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions.csv from src/model.py")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the predictions folder")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    predictions = pd.read_csv(args.predictions)
    y_true = predictions["label"].to_numpy()
    models, heuristics = split_prediction_columns(predictions)
    if not models and not heuristics:
        raise SystemExit("No model__* or heuristic__* columns found in the predictions file")

    output_dir = args.output_dir or args.predictions.parent
    table = evaluate_scores(y_true, models, heuristics, threshold=args.threshold)
    table.to_csv(output_dir / "metrics.csv")

    kinds = {**{n: "model" for n in models}, **{n: "heuristic" for n in heuristics}}
    scores = {**models, **heuristics}
    figures = output_dir / "figures"
    save_figure(plot_roc_curves(y_true, scores, kinds), figures / "roc_curves.png")
    save_figure(plot_pr_curves(y_true, scores, kinds), figures / "pr_curves.png")
    if models:
        best = table[table["kind"] == "model"]["roc_auc"].idxmax()
        fig = plot_confusion_matrix(
            y_true,
            (models[best] >= args.threshold).astype(int),
            title=f"Confusion matrix - {display_name(best)}",
        )
        save_figure(fig, figures / "confusion_matrix.png")
    print(table.round(3).to_string())
    return table


if __name__ == "__main__":
    main()
