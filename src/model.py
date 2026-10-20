"""Supervised classifiers for protein-protein interaction prediction.

Each candidate protein pair is represented by the topological feature vector
from :mod:`src.feature_extractor`; a binary classifier learns to separate
hidden true interactions from non-interacting pairs.

Pipeline (``python src/model.py``)
----------------------------------
1. Load the network: by default the preprocessed HI-III human interactome
   (downloaded and cleaned on first use, see :mod:`src.preprocessing`), or
   any edge list passed with ``--input``.
2. Hide interactions and sample non-interacting pairs (leakage-free split).
3. Featurise train/test pairs on their own feature graphs.
4. Cross-validate and fit the classifiers.
5. Score held-out pairs; compare against unsupervised heuristics
   (:mod:`src.evaluate`).
6. Rank currently unobserved protein pairs as candidate novel interactions.
7. Write metrics, predictions, figures and fitted models to ``--output-dir``.

Examples::

    python src/model.py                                   # HI-III human interactome
    python src/model.py --dataset bioplex-2 --negatives local
    python src/model.py --input data/raw/sample_ppi_edges.tsv   # synthetic sample
    python src/model.py --input data/external/9606.protein.links.v12.0.txt.gz \
        --min-score 700 --betweenness-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.evaluate import (
    compute_permutation_importance,
    display_name,
    evaluate_scores,
    heuristic_scores,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    save_figure,
    use_headless_backend,
)
from src.datasets import INTERACTOMES
from src.feature_extractor import TopologicalFeatureExtractor
from src.graph_builder import (
    LinkPredictionSplit,
    build_graph,
    graph_summary,
    load_edge_list,
    make_link_prediction_split,
)
from src.preprocessing import load_symbol_map, processed_path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "hi-iii"
SAMPLE_INPUT = ROOT / "data" / "raw" / "sample_ppi_edges.tsv"
DEFAULT_OUTPUT = ROOT / "results"
#: ``betweenness_samples="auto"``: exact betweenness up to this many proteins,
#: otherwise estimated from ``AUTO_BETWEENNESS_SAMPLES`` random sources.
AUTO_BETWEENNESS_NODES = 3000
AUTO_BETWEENNESS_SAMPLES = 1000
MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting")
#: Heuristics drawn next to the models in ROC / PR figures (all are tabulated).
PLOT_BASELINES: tuple[str, ...] = ("l3", "common_neighbors", "preferential_attachment")


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
# Novel interaction ranking
# ---------------------------------------------------------------------------


def l3_candidate_pairs(
    graph: nx.Graph, n_candidates: int = 2000, chunk_size: int = 512
) -> list[tuple]:
    """Unobserved protein pairs with the highest L3 scores.

    Scoring all ``O(n^2)`` non-interacting pairs is infeasible for real
    interactomes, so candidates are pre-selected by the degree-normalised
    count of length-3 paths (the strongest single predictor). Rows of
    ``A D^-1/2 A D^-1/2 A`` are computed ``chunk_size`` at a time.
    """
    nodes = list(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, dtype=float, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = sparse.diags_array(np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1)), 0.0))
    right = inv_sqrt @ adjacency @ inv_sqrt @ adjacency

    best_scores = np.empty(0)
    best_rows = np.empty(0, dtype=np.int64)
    best_cols = np.empty(0, dtype=np.int64)
    for start in range(0, len(nodes), chunk_size):
        block = sparse.coo_array(adjacency[start : start + chunk_size] @ right)
        rows = block.row.astype(np.int64) + start
        cols = block.col.astype(np.int64)
        keep = rows < cols  # each unordered pair once, no self-pairs
        rows, cols, scores = rows[keep], cols[keep], block.data[keep]
        if rows.size:
            observed = np.asarray(adjacency[rows, cols]).ravel() > 0
            rows, cols, scores = rows[~observed], cols[~observed], scores[~observed]

        best_scores = np.concatenate([best_scores, scores])
        best_rows = np.concatenate([best_rows, rows])
        best_cols = np.concatenate([best_cols, cols])
        if best_scores.size > n_candidates:
            top = np.argpartition(-best_scores, n_candidates)[:n_candidates]
            best_scores, best_rows, best_cols = best_scores[top], best_rows[top], best_cols[top]

    order = np.lexsort((best_cols, best_rows, -best_scores))
    return [(nodes[best_rows[i]], nodes[best_cols[i]]) for i in order]


def rank_novel_interactions(
    graph: nx.Graph,
    model: TrainedModel,
    top_k: int = 100,
    n_candidates: int = 2000,
    extractor: TopologicalFeatureExtractor | None = None,
    symbols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Score unobserved protein pairs and return the ``top_k`` most likely.

    Features are computed on the full observed network, so every currently
    known interaction informs the prediction.
    """
    candidates = l3_candidate_pairs(graph, n_candidates=n_candidates)
    if not candidates:
        return pd.DataFrame(columns=["protein_a", "protein_b", "probability", "l3", "common_neighbors"])
    extractor = (extractor or TopologicalFeatureExtractor()).fit(graph)
    X = extractor.transform(candidates)
    ranked = pd.DataFrame(
        {
            "protein_a": [u for u, _ in candidates],
            "protein_b": [v for _, v in candidates],
            "probability": model.predict_proba(X),
            "l3": X["l3"].to_numpy(),
            "common_neighbors": X["common_neighbors"].to_numpy(),
        }
    )
    ranked = ranked.sort_values(["probability", "l3"], ascending=False, kind="stable")
    ranked = ranked.head(top_k).reset_index(drop=True)
    return add_symbols(ranked, symbols)


def add_symbols(table: pd.DataFrame, symbols: dict[str, str] | None) -> pd.DataFrame:
    """Insert ``symbol_a`` / ``symbol_b`` after the protein columns when symbols are known."""
    if not symbols:
        return table
    table = table.copy()
    table.insert(2, "symbol_a", table["protein_a"].map(symbols))
    table.insert(3, "symbol_b", table["protein_b"].map(symbols))
    return table


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _portable_path(path: str | Path) -> str:
    """Repository-relative POSIX path when possible, so run summaries are portable."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_betweenness_samples(setting: int | str | None, n_proteins: int) -> int | None:
    """Translate ``"auto"`` / ``"exact"`` / ``None`` / an integer into a sample size."""
    if setting is None or setting == "exact":
        return None
    if setting == "auto":
        return AUTO_BETWEENNESS_SAMPLES if n_proteins > AUTO_BETWEENNESS_NODES else None
    return int(setting)


@dataclass
class PipelineConfig:
    """All settings of an end-to-end run (mirrors the command-line options).

    The network is ``input_path`` when given, otherwise the preprocessed
    registry ``dataset``.
    """

    dataset: str | None = DEFAULT_DATASET
    input_path: Path | None = None
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
    betweenness_samples: int | str | None = "auto"
    max_path_length: int = 6
    top_novel: int = 100
    importance_repeats: int = 10
    seed: int = 42
    save_models: bool = True

    def resolve_input(self) -> Path:
        """The edge list to load, preprocessing the registry dataset on first use."""
        if self.input_path is not None:
            return Path(self.input_path)
        if self.dataset is None:
            raise ValueError("Set either input_path or dataset")
        return processed_path(self.dataset)

    def extractor(self, n_proteins: int = 0) -> TopologicalFeatureExtractor:
        return TopologicalFeatureExtractor(
            max_path_length=self.max_path_length,
            betweenness_samples=resolve_betweenness_samples(self.betweenness_samples, n_proteins),
            seed=self.seed,
        )


def build_feature_matrices(
    split: LinkPredictionSplit, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Featurise train pairs on ``train_graph`` and test pairs on ``test_graph``."""
    start = time.perf_counter()
    n_proteins = split.train_graph.number_of_nodes()
    X_train = config.extractor(n_proteins).fit_transform(split.train_graph, split.train_pairs)
    X_test = config.extractor(n_proteins).fit_transform(split.test_graph, split.test_pairs)
    logger.info(
        "Extracted %d features for %d train / %d test pairs (%.1fs)",
        X_train.shape[1],
        len(X_train),
        len(X_test),
        time.perf_counter() - start,
    )
    return X_train, X_test


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full pipeline and write all artifacts to ``config.output_dir``.

    Returns
    -------
    dict with the network summary, split sizes, metric table (as records),
    best model name, artifact paths and runtime.
    """
    t0 = time.perf_counter()
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Network
    input_path = config.resolve_input()
    edges = load_edge_list(
        input_path,
        fmt=config.fmt,
        source_col=config.source_col,
        target_col=config.target_col,
        weight_col=config.weight_col,
        min_score=config.min_score,
    )
    graph = build_graph(edges, largest_component=config.largest_component)
    summary = graph_summary(graph)
    symbols = _symbols_for(input_path)
    logger.info(
        "Network %s: %d proteins, %d interactions",
        input_path.name,
        summary["proteins"],
        summary["interactions"],
    )

    # 2-3. Split and featurise
    split = make_link_prediction_split(
        graph,
        test_fraction=config.test_fraction,
        train_fraction=config.train_fraction,
        negative_ratio=config.negative_ratio,
        negative_strategy=config.negative_strategy,
        seed=config.seed,
    )
    X_train, X_test = build_feature_matrices(split, config)
    y_test = split.test_labels

    # 4. Train
    models = train_models(X_train, split.train_labels, config.models, config.seed, config.cv_folds)

    # 5. Evaluate on held-out interactions
    model_scores = {name: m.predict_proba(X_test) for name, m in models.items()}
    baseline_scores = heuristic_scores(X_test)
    metrics = evaluate_scores(y_test, model_scores, baseline_scores)
    for name, model in models.items():
        for key, value in model.cv_scores.items():
            metrics.loc[name, key] = value
    best = str(metrics[metrics["kind"] == "model"]["roc_auc"].idxmax())
    logger.info(
        "Held-out performance (%s negatives):\n%s",
        config.negative_strategy,
        metrics[["kind", "roc_auc", "average_precision", "r_precision", "f1"]].round(3).to_string(),
    )

    importance = compute_permutation_importance(
        models[best].estimator, X_test, y_test, n_repeats=config.importance_repeats, seed=config.seed
    )

    kinds = {**{n: "model" for n in model_scores}, **{n: "heuristic" for n in baseline_scores}}
    plotted = {**model_scores, **{n: baseline_scores[n] for n in PLOT_BASELINES if n in baseline_scores}}
    figures = {
        "roc_curves": save_figure(plot_roc_curves(y_test, plotted, kinds), figures_dir / "roc_curves.png"),
        "pr_curves": save_figure(plot_pr_curves(y_test, plotted, kinds), figures_dir / "pr_curves.png"),
        "confusion_matrix": save_figure(
            plot_confusion_matrix(
                y_test, models[best].predict(X_test), title=f"Confusion matrix - {display_name(best)}"
            ),
            figures_dir / "confusion_matrix.png",
        ),
        "feature_importance": save_figure(
            plot_feature_importance(importance, title=f"Permutation importance - {display_name(best)}"),
            figures_dir / "feature_importance.png",
        ),
    }

    # 6. Candidate novel interactions
    novel = None
    if config.top_novel > 0:
        novel = rank_novel_interactions(
            graph,
            models[best],
            top_k=config.top_novel,
            extractor=config.extractor(graph.number_of_nodes()),
            symbols=symbols,
        )
        novel.to_csv(output_dir / "novel_predictions.csv", index=False)
        logger.info("Top predicted novel interaction: %s", novel.iloc[0].to_dict() if len(novel) else None)

    # 7. Artifacts
    predictions = pd.DataFrame(
        {
            "protein_a": [u for u, _ in split.test_pairs],
            "protein_b": [v for _, v in split.test_pairs],
            "label": y_test,
            **{f"model__{n}": s for n, s in model_scores.items()},
            **{f"heuristic__{n}": s for n, s in baseline_scores.items()},
        }
    )
    add_symbols(predictions, symbols).to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index_label="method")
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    if config.save_models:
        for name, model in models.items():
            save_model(model, output_dir / "models" / f"{name}.joblib")

    results = {
        "config": {k: (_portable_path(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
        "input": _portable_path(input_path),
        "graph_summary": summary,
        "split": split.summary(),
        "best_model": best,
        "metrics": json.loads(metrics.reset_index().to_json(orient="records")),
        "figures": {k: _portable_path(v) for k, v in figures.items()},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Best model: %s. Artifacts written to %s (%.1fs)", best, output_dir, results["runtime_seconds"])
    results["metrics_table"] = metrics
    results["novel_predictions"] = novel
    return results


def _symbols_for(path: Path) -> dict[str, str]:
    """Gene symbols from a processed edge list; empty for other formats."""
    try:
        return load_symbol_map(path)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
        return {}


def _betweenness_arg(value: str) -> int | str:
    if value in ("auto", "exact"):
        return value
    try:
        samples = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use 'auto', 'exact' or a positive integer") from None
    if samples < 1:
        raise argparse.ArgumentTypeError("sample size must be positive")
    return samples


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Predict protein-protein interactions from network topology."
    )
    data = parser.add_argument_group("data")
    source = data.add_mutually_exclusive_group()
    source.add_argument(
        "--dataset",
        choices=INTERACTOMES,
        default=DEFAULT_DATASET,
        help="preprocessed registry interactome (default: %(default)s)",
    )
    source.add_argument("--input", type=Path, help="any PPI edge list (TSV/CSV/STRING) instead of --dataset")
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
    model.add_argument(
        "--betweenness-samples",
        type=_betweenness_arg,
        default="auto",
        help=f"'exact', an integer k (sample k sources) or 'auto': exact up to "
        f"{AUTO_BETWEENNESS_NODES:,} proteins, else {AUTO_BETWEENNESS_SAMPLES:,} samples",
    )
    model.add_argument("--max-path-length", type=int, default=6)
    model.add_argument("--top-novel", type=int, default=100, help="novel interactions to rank (0 = skip)")
    model.add_argument("--importance-repeats", type=int, default=10, help="permutation importance repeats")

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
        dataset=None if args.input else args.dataset,
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
        top_novel=args.top_novel,
        importance_repeats=args.importance_repeats,
        seed=args.seed,
        save_models=not args.no_save_models,
    )


def main(argv: list[str] | None = None) -> dict:
    config = parse_args(argv)
    use_headless_backend()
    return run_pipeline(config)


if __name__ == "__main__":
    main()
