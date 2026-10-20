"""Benchmark models and heuristics across interactomes and negative-sampling protocols.

Runs the full pipeline (:func:`src.model.run_pipeline`) for every combination
of dataset and negative-sampling strategy and collects held-out ROC-AUC and
average precision into one table and heatmap. The question it answers is
whether a method's performance is a property of the method or of the kind of
interactome it is applied to (binary yeast two-hybrid maps vs literature
curation vs co-complex mass spectrometry).

Usage::

    python src/benchmark.py
    python src/benchmark.py --datasets hi-iii bioplex-2 --negatives uniform
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from src.data_analysis import DEFAULT_DATASETS
from src.datasets import DATASETS, INTERACTOMES
from src.evaluate import INK, SEQUENTIAL, display_name, plot_style, save_figure, use_headless_backend
from src.model import MODEL_NAMES, PipelineConfig, run_pipeline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "benchmark"
RUNS_DIR = ROOT / "results" / "benchmark_runs"
HEATMAP_METHODS = (
    "logistic_regression", "random_forest", "gradient_boosting",
    "l3", "common_neighbors", "adamic_adar", "preferential_attachment", "shortest_path",
)


def run_benchmark(
    datasets: list[str] | tuple[str, ...] = DEFAULT_DATASETS,
    strategies: list[str] | tuple[str, ...] = ("uniform", "local"),
    output_dir: str | Path = DEFAULT_OUTPUT,
    runs_dir: str | Path = RUNS_DIR,
    cv_folds: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Run every dataset x strategy combination and return a long metric table.

    Per-run artifacts (figures, predictions) go to ``runs_dir``; the combined
    table and heatmaps go to ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        for strategy in strategies:
            start = time.perf_counter()
            config = PipelineConfig(
                dataset=dataset,
                output_dir=Path(runs_dir) / f"{dataset}-{strategy}",
                negative_strategy=strategy,
                models=MODEL_NAMES,
                cv_folds=cv_folds,
                top_novel=0,
                importance_repeats=2,
                seed=seed,
                save_models=False,
            )
            results = run_pipeline(config)
            for record in results["metrics"]:
                rows.append(
                    {
                        "dataset": dataset,
                        "network_type": DATASETS[dataset].kind,
                        "negatives": strategy,
                        "method": record["method"],
                        "method_type": record["kind"],
                        "roc_auc": record["roc_auc"],
                        "average_precision": record["average_precision"],
                        "test_pairs": results["split"]["test_positives"] + results["split"]["test_negatives"],
                    }
                )
            logger.info("Benchmark %s / %s done in %.0fs", dataset, strategy, time.perf_counter() - start)

    table = pd.DataFrame(rows)
    table.round(4).to_csv(output_dir / "benchmark_metrics.csv", index=False)
    for metric in ("roc_auc", "average_precision"):
        pivot = benchmark_pivot(table, metric)
        pivot.round(4).to_csv(output_dir / f"{metric}_by_dataset.csv")
        save_figure(plot_benchmark_heatmap(pivot, metric), output_dir / "figures" / f"{metric}_heatmap.png")
    return table


def benchmark_pivot(table: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Rows: dataset / negatives; columns: methods in :data:`HEATMAP_METHODS` order."""
    pivot = table.pivot_table(index=["dataset", "negatives"], columns="method", values=metric)
    order = [m for m in HEATMAP_METHODS if m in pivot.columns]
    datasets = list(dict.fromkeys(table["dataset"]))
    negatives = list(dict.fromkeys(table["negatives"]))
    index = [(d, n) for d in datasets for n in negatives if (d, n) in pivot.index]
    return pivot.loc[index, order]


def plot_benchmark_heatmap(pivot: pd.DataFrame, metric: str = "roc_auc") -> plt.Figure:
    """Annotated heatmap of a benchmark pivot table."""
    cmap = LinearSegmentedColormap.from_list("ppi_blues", SEQUENTIAL)
    labels = [f"{d}  ·  {n}" for d, n in pivot.index]
    with plot_style():
        fig, ax = plt.subplots(figsize=(1.0 * pivot.shape[1] + 2.6, 0.48 * pivot.shape[0] + 1.6))
        sns.heatmap(
            pivot.to_numpy(dtype=float), annot=True, fmt=".2f", cmap=cmap, vmin=0.3, vmax=1.0,
            linewidths=2, linecolor=INK["surface"], cbar_kws={"shrink": 0.8, "label": display_name(metric)},
            xticklabels=[display_name(m) for m in pivot.columns], yticklabels=labels, ax=ax,
        )
        for text in ax.texts:
            text.set_color("white" if float(text.get_text()) > 0.75 else INK["primary"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        ax.set(xlabel="", ylabel="")
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_title(f"{display_name(metric)} on held-out interactions")
    return fig


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Benchmark across interactomes.")
    parser.add_argument("--datasets", nargs="+", choices=INTERACTOMES, default=list(DEFAULT_DATASETS))
    parser.add_argument("--negatives", nargs="+", choices=("uniform", "local"), default=["uniform", "local"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    use_headless_backend()
    table = run_benchmark(args.datasets, args.negatives, args.output_dir, args.runs_dir, args.cv_folds, args.seed)
    print(benchmark_pivot(table).round(3).to_string())
    return table


if __name__ == "__main__":
    main()
