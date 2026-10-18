"""Supervised classifiers for protein-protein interaction prediction.

Each candidate protein pair is represented by the topological feature vector
from :mod:`src.feature_extractor`; a binary classifier learns to separate
hidden true interactions from non-interacting pairs.

Run the pipeline from the repository root::

    python src/model.py
    python src/model.py --input data/external/4932.protein.links.v12.0.txt.gz --min-score 700
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.feature_extractor import TopologicalFeatureExtractor
from src.graph_builder import (
    LinkPredictionSplit,
    build_graph,
    graph_summary,
    load_edge_list,
    make_link_prediction_split,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "sample_ppi_edges.tsv"
DEFAULT_OUTPUT = ROOT / "results"
MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_model(name: str, seed: int = 42) -> ClassifierMixin:
    """Return an unfitted classifier by name.

    * ``logistic_regression`` - linear baseline. Topological features are
      heavy-tailed counts, so they are ``log1p``-transformed and standardised.
    * ``random_forest`` - bagged decision trees; robust to feature scale and
      captures interactions between features.
    * ``gradient_boosting`` - histogram-based gradient-boosted trees.
    """
    if name == "logistic_regression":
        return make_pipeline(
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


@dataclass
class TrainedModel:
    """A fitted classifier together with its cross-validation scores."""

    name: str
    estimator: ClassifierMixin
    cv_scores: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability that each pair interacts."""
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def cross_validate_model(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Stratified k-fold ROC-AUC and average precision on the training pairs."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(clone(model), X, y, cv=cv, scoring=("roc_auc", "average_precision"))
    return {
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "cv_average_precision_mean": float(np.mean(scores["test_average_precision"])),
        "cv_average_precision_std": float(np.std(scores["test_average_precision"])),
    }


def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    model_names: tuple[str, ...] | list[str] = MODEL_NAMES,
    seed: int = 42,
    cv_folds: int = 5,
) -> dict[str, TrainedModel]:
    """Cross-validate (optional) and fit every requested model on all of ``X``."""
    trained: dict[str, TrainedModel] = {}
    for name in model_names:
        start = time.perf_counter()
        estimator = build_model(name, seed=seed)
        cv_scores = cross_validate_model(estimator, X, y, cv_folds, seed) if cv_folds > 1 else {}
        estimator.fit(X, y)
        trained[name] = TrainedModel(name, estimator, cv_scores)
        logger.info(
            "Trained %-20s CV ROC-AUC %s (%.1fs)",
            name,
            f"{cv_scores['cv_roc_auc_mean']:.3f} +/- {cv_scores['cv_roc_auc_std']:.3f}" if cv_scores else "n/a",
            time.perf_counter() - start,
        )
    return trained


def save_model(model: TrainedModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> TrainedModel:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All settings of an end-to-end run (mirrors the command-line options)."""

    input_path: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT
    fmt: str = "auto"
    source_col: str | None = None
    target_col: str | None = None
    weight_col: str | None = None
    min_score: float | None = None
    largest_component: bool = True
    test_fraction: float = 0.1
    train_fraction: float = 0.1
    negative_ratio: float = 1.0
    negative_strategy: str = "uniform"
    models: tuple[str, ...] = MODEL_NAMES
    cv_folds: int = 5
    betweenness_samples: int | None = None
    max_path_length: int = 6
    seed: int = 42
    save_models: bool = True


def build_feature_matrices(
    split: LinkPredictionSplit, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Featurise train pairs on ``train_graph`` and test pairs on ``test_graph``."""
    kwargs = dict(
        max_path_length=config.max_path_length,
        betweenness_samples=config.betweenness_samples,
        seed=config.seed,
    )
    start = time.perf_counter()
    X_train = TopologicalFeatureExtractor(**kwargs).fit_transform(split.train_graph, split.train_pairs)
    X_test = TopologicalFeatureExtractor(**kwargs).fit_transform(split.test_graph, split.test_pairs)
    logger.info(
        "Extracted %d features for %d train / %d test pairs (%.1fs)",
        X_train.shape[1],
        len(X_train),
        len(X_test),
        time.perf_counter() - start,
    )
    return X_train, X_test


def run_pipeline(config: PipelineConfig) -> dict:
    """Load data, split, featurise, train and score every model on the test set."""
    edges = load_edge_list(
        config.input_path,
        fmt=config.fmt,
        source_col=config.source_col,
        target_col=config.target_col,
        weight_col=config.weight_col,
        min_score=config.min_score,
    )
    graph = build_graph(edges, largest_component=config.largest_component)
    summary = graph_summary(graph)
    logger.info("Network: %d proteins, %d interactions", summary["proteins"], summary["interactions"])

    split = make_link_prediction_split(
        graph,
        test_fraction=config.test_fraction,
        train_fraction=config.train_fraction,
        negative_ratio=config.negative_ratio,
        negative_strategy=config.negative_strategy,
        seed=config.seed,
    )
    X_train, X_test = build_feature_matrices(split, config)
    models = train_models(X_train, split.train_labels, config.models, config.seed, config.cv_folds)

    test_auc = {name: roc_auc_score(split.test_labels, m.predict_proba(X_test)) for name, m in models.items()}
    for name, auc in test_auc.items():
        logger.info("Test ROC-AUC %-20s %.3f", name, auc)

    if config.save_models:
        for name, model in models.items():
            save_model(model, Path(config.output_dir) / "models" / f"{name}.joblib")

    return {"graph_summary": summary, "split": split.summary(), "test_roc_auc": test_auc}


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Predict protein-protein interactions from network topology."
    )
    data = parser.add_argument_group("data")
    data.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="PPI edge list (TSV/CSV/STRING)")
    data.add_argument("--format", dest="fmt", choices=("auto", "edgelist", "string"), default="auto")
    data.add_argument("--source-col", help="column of the first interactor")
    data.add_argument("--target-col", help="column of the second interactor")
    data.add_argument("--weight-col", help="numeric confidence column")
    data.add_argument("--min-score", type=float, help="drop interactions below this confidence")
    data.add_argument(
        "--all-components", action="store_true", help="keep every component, not just the largest"
    )

    split = parser.add_argument_group("evaluation protocol")
    split.add_argument("--test-fraction", type=float, default=0.1)
    split.add_argument("--train-fraction", type=float, default=0.1)
    split.add_argument("--negative-ratio", type=float, default=1.0)
    split.add_argument("--negatives", dest="negative_strategy", choices=("uniform", "local"), default="uniform")

    model = parser.add_argument_group("features and models")
    model.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    model.add_argument("--cv-folds", type=int, default=5, help="0 or 1 disables cross-validation")
    model.add_argument("--betweenness-samples", type=int, help="approximate betweenness with k sources")
    model.add_argument("--max-path-length", type=int, default=6)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        fmt=args.fmt,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
        min_score=args.min_score,
        largest_component=not args.all_components,
        test_fraction=args.test_fraction,
        train_fraction=args.train_fraction,
        negative_ratio=args.negative_ratio,
        negative_strategy=args.negative_strategy,
        models=tuple(args.models),
        cv_folds=args.cv_folds,
        betweenness_samples=args.betweenness_samples,
        max_path_length=args.max_path_length,
        seed=args.seed,
        save_models=not args.no_save_models,
    )


def main(argv: list[str] | None = None) -> dict:
    return run_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()
