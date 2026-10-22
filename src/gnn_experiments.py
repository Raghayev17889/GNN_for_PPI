"""Graph neural network experiments on a human interactome.

Four questions, each answered on the same leakage-free splits as the
classical pipeline:

1. **Does message passing help?** GraphSAGE and GCN against the identical
   network without message passing (an MLP on the node features), and a
   GraphSAGE hybrid whose decoder also sees pair-level path counts.
2. **What do the node features contribute?** Structural descriptors only,
   biological annotation only (GO, pathways, tissue expression), or both.
3. **How do GNNs compare with the classical pipeline?** Gradient boosting on
   hand-crafted topological features, and the L3 heuristic.
4. **Are novel predictions supported by independent experiments?** Top-ranked
   unobserved HI-III pairs are looked up in three other maps of the human
   interactome (BioPlex 2.0 affinity purification-mass spectrometry,
   Lit-BM-13 literature curation, HI-II-14 yeast two-hybrid).

Every GNN configuration is trained with several random seeds; the reported
numbers are means and standard deviations over seeds on a fixed split.

Usage::

    python src/gnn_experiments.py                       # HI-III -> reports/gnn
    python src/gnn_experiments.py --seeds 1 --epochs 100 --output-dir results/gnn-quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

if __package__ in (None, ""):  # allow `python src/gnn_experiments.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.annotations import load_annotations
from src.evaluate import (
    INK,
    PALETTE,
    compute_metrics,
    display_name,
    plot_pr_curves,
    plot_roc_curves,
    plot_style,
    save_figure,
    use_headless_backend,
)
from src.feature_extractor import TopologicalFeatureExtractor
from src.gnn_model import (
    DISPLAY_NAMES,
    GNN_ARCHITECTURES,
    GNNConfig,
    TrainedGNN,
    rank_novel_interactions,
    train_link_predictor,
)
from src.graph_builder import LinkPredictionSplit, load_graph, make_link_prediction_split
from src.model import build_model, l3_candidate_pairs, resolve_betweenness_samples
from src.model import rank_novel_interactions as rank_classical
from src.model import TrainedModel
from src.node_features import FEATURE_KINDS, NodeFeatures, biological_features, canonical_nodes
from src.preprocessing import load_symbol_map, processed_path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "gnn"
EXTERNAL_MAPS: tuple[str, ...] = ("bioplex-2", "lit-bm-13", "hi-ii-14")
#: Model variants of the ablation: name -> (encoder architecture, pair heuristics in the decoder).
VARIANTS: dict[str, tuple[str, bool]] = {
    "graphsage": ("graphsage", False),
    "gcn": ("gcn", False),
    "mlp": ("mlp", False),
    "graphsage+pairs": ("graphsage", True),
}
VARIANT_NAMES = {**DISPLAY_NAMES, "graphsage+pairs": "GraphSAGE + pair heuristics"}
FEATURE_LABELS = {"structural": "Structural", "biological": "Biological", "both": "Structural + biological"}


@dataclass
class Reference:
    """Scores of the classical reference methods on one split."""

    gradient_boosting: TrainedModel
    scores: dict[str, np.ndarray]


def classical_reference(split: LinkPredictionSplit, seed: int = 42) -> Reference:
    """Gradient boosting on topological features, plus the L3 and preferential-attachment heuristics."""
    samples = resolve_betweenness_samples("auto", split.train_graph.number_of_nodes())
    extractor = lambda: TopologicalFeatureExtractor(betweenness_samples=samples, seed=seed)  # noqa: E731
    X_train = extractor().fit_transform(split.train_graph, split.train_pairs)
    X_test = extractor().fit_transform(split.test_graph, split.test_pairs)
    estimator = build_model("gradient_boosting", seed=seed).fit(X_train, split.train_labels)
    model = TrainedModel("gradient_boosting", estimator)
    return Reference(
        model,
        {
            "gradient_boosting": model.predict_proba(X_test),
            "l3": X_test["l3"].to_numpy(),
            "preferential_attachment": X_test["preferential_attachment"].to_numpy(),
        },
    )


def run_ablation(
    graph: nx.Graph,
    annotations: Mapping,
    strategies: Sequence[str] = ("uniform", "local"),
    variants: Sequence[str] = tuple(VARIANTS),
    feature_kinds: Sequence[str] = FEATURE_KINDS,
    seeds: Sequence[int] = (0, 1, 2),
    epochs: int = 300,
    split_seed: int = 42,
) -> tuple[pd.DataFrame, dict, dict]:
    """Train every model variant x feature set x seed for each negative protocol.

    Returns
    -------
    runs:
        One row per trained model (and per classical reference) with held-out metrics.
    splits:
        ``{strategy: (split, Reference)}``.
    models:
        ``{(strategy, variant, features): TrainedGNN}`` for the first seed.
    """
    nodes = canonical_nodes(graph)
    biological = biological_features(nodes, annotations, seed=split_seed)
    rows, splits, models = [], {}, {}
    for strategy in strategies:
        split = make_link_prediction_split(graph, negative_strategy=strategy, seed=split_seed)
        reference = classical_reference(split, split_seed)
        splits[strategy] = (split, reference)
        for name, score in reference.scores.items():
            metrics = compute_metrics(split.test_labels, score, threshold=None)
            rows.append({"negatives": strategy, "method": name, "architecture": name, "features": "topological",
                         "seed": split_seed, **metrics})
        for kind in feature_kinds:
            for variant in variants:
                architecture, pair_heuristics = VARIANTS[variant]
                for i, seed in enumerate(seeds):
                    features = NodeFeatures(nodes, kind, biological)
                    config = GNNConfig(architecture=architecture, features=kind, epochs=epochs,
                                       negative_strategy=strategy, pair_heuristics=pair_heuristics, seed=seed)
                    trained = train_link_predictor(split, features, config)
                    scores = trained.predict_proba(split.test_graph, split.test_pairs)
                    rows.append({
                        "negatives": strategy, "method": f"{variant}/{kind}", "architecture": variant,
                        "features": kind, "seed": seed,
                        **compute_metrics(split.test_labels, scores, threshold=None),
                        "validation_average_precision": trained.validation_average_precision,
                        "best_epoch": trained.best_epoch,
                        "training_seconds": round(trained.training_seconds, 1),
                    })
                    if i == 0:
                        models[(strategy, variant, kind)] = trained
                    logger.info("%s | %-15s | %-10s | seed %d: AP %.3f", strategy, variant, kind, seed,
                                rows[-1]["average_precision"])
    return pd.DataFrame(rows), splits, models


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every method and protocol."""
    grouped = runs.groupby(["negatives", "method", "architecture", "features"], sort=False)
    summary = grouped[["roc_auc", "average_precision", "r_precision"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = grouped.size()
    return summary.reset_index()


# ---------------------------------------------------------------------------
# External validation of novel predictions
# ---------------------------------------------------------------------------


def load_interaction_sets(names: Iterable[str] = EXTERNAL_MAPS) -> dict[str, set[frozenset]]:
    """Interactions of other interactome maps as sets of unordered pairs."""
    maps = {}
    for name in names:
        edges = pd.read_csv(processed_path(name), sep="\t", dtype=str, usecols=["protein_a", "protein_b"])
        maps[name] = {frozenset(p) for p in zip(edges["protein_a"], edges["protein_b"])}
    return maps


def support_flags(pairs: Sequence[tuple], maps: Mapping[str, set[frozenset]]) -> pd.DataFrame:
    """Boolean table: is each pair reported in each external map (and in any)?"""
    keys = [frozenset((str(u), str(v))) for u, v in pairs]
    flags = pd.DataFrame({f"in_{name}": [k in edges for k in keys] for name, edges in maps.items()})
    flags["in_any"] = flags.any(axis=1)
    return flags


def random_pair_support(
    graph: nx.Graph, maps: Mapping[str, set[frozenset]], n_pairs: int = 200_000, seed: int = 0
) -> pd.Series:
    """Support rate of uniformly random unobserved pairs (the chance level)."""
    rng = np.random.default_rng(seed)
    nodes = canonical_nodes(graph)
    u = rng.integers(len(nodes), size=n_pairs)
    v = rng.integers(len(nodes), size=n_pairs)
    pairs = [(nodes[a], nodes[b]) for a, b in zip(u, v) if a != b and not graph.has_edge(nodes[a], nodes[b])]
    return support_flags(pairs, maps).mean()


def validation_curve(
    rankings: Mapping[str, Sequence[tuple]],
    maps: Mapping[str, set[frozenset]],
    ks: Sequence[int] = (10, 25, 50, 100, 250, 500, 1000),
) -> pd.DataFrame:
    """Fraction of the top-``k`` predictions of each method found in external maps."""
    rows = []
    for method, pairs in rankings.items():
        flags = support_flags(list(pairs), maps)
        for k in ks:
            if k <= len(flags):
                rows.append({"method": method, "k": k, **flags.head(k).mean().to_dict()})
    return pd.DataFrame(rows)


def hub_fraction(pairs: Sequence[tuple], graph: nx.Graph, top_percent: float = 1.0) -> float:
    """Fraction of pairs that involve a protein from the top ``top_percent``% by degree."""
    degrees = pd.Series(dict(graph.degree()))
    hubs = set(degrees[degrees >= degrees.quantile(1 - top_percent / 100)].index)
    return float(np.mean([u in hubs or v in hubs for u, v in pairs])) if len(pairs) else float("nan")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_ablation(summary: pd.DataFrame, metric: str = "average_precision") -> plt.Figure:
    """Grouped bars: metric per feature set and architecture, one panel per negative protocol."""
    strategies = list(dict.fromkeys(summary["negatives"]))
    architectures = [a for a in VARIANTS if a in set(summary["architecture"])]
    kinds = [k for k in FEATURE_KINDS if k in set(summary["features"])]
    references = {"gradient_boosting": "Gradient boosting (topological features)", "l3": "L3 paths"}
    with plot_style():
        fig, axes = plt.subplots(1, len(strategies), figsize=(6.0 * len(strategies), 4.6), sharey=False, squeeze=False)
        for ax, strategy in zip(axes[0], strategies):
            part = summary[summary["negatives"] == strategy].set_index("method")
            width = 0.8 / len(architectures)
            for j, architecture in enumerate(architectures):
                means = [part.loc[f"{architecture}/{k}", f"{metric}_mean"] for k in kinds]
                stds = [part.loc[f"{architecture}/{k}", f"{metric}_std"] for k in kinds]
                x = np.arange(len(kinds)) + (j - (len(architectures) - 1) / 2) * width
                ax.bar(x, means, width * 0.92, yerr=np.nan_to_num(stds), color=PALETTE[j], label=VARIANT_NAMES[architecture],
                       error_kw={"elinewidth": 1, "capsize": 2, "ecolor": INK["secondary"]})
            for (name, label), style in zip(references.items(), ("-", (0, (5, 2)))):
                ax.axhline(part.loc[name, f"{metric}_mean"], color=INK["primary"], linewidth=1.2, linestyle=style,
                           label=label)
            values = summary[summary["negatives"] == strategy][f"{metric}_mean"]
            ax.set_ylim(max(0.0, values.min() - 0.05), min(1.0, values.max() + 0.02))
            ax.set_xticks(np.arange(len(kinds)), [FEATURE_LABELS[k] for k in kinds])
            ax.set_ylabel(display_name(metric))
            ax.set_title(f"{strategy.capitalize()} negatives")
            ax.grid(axis="x", visible=False)
        axes[0][-1].legend(loc="lower right", fontsize=8)
        fig.suptitle("Node features x message passing on HI-III (mean +/- SD over seeds)", x=0.01, ha="left",
                     fontsize=12, fontweight="bold", color=INK["primary"])
        fig.tight_layout()
    return fig


def plot_training_curve(history: pd.DataFrame, title: str) -> plt.Figure:
    """Training loss and validation average precision per epoch."""
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(history["epoch"], history["loss"], color=PALETTE[0], label="Training loss")
        ax.set(xlabel="Epoch", ylabel="Binary cross-entropy")
        right = ax.twinx()
        right.plot(history["epoch"], history["val_average_precision"], color=PALETTE[1], label="Validation AP")
        right.set_ylabel("Validation average precision")
        right.grid(False)
        best = history.loc[history["val_average_precision"].idxmax()]
        right.axvline(best["epoch"], color=INK["muted"], linestyle=":", linewidth=1)
        lines = ax.get_lines() + right.get_lines()[:1]
        ax.legend(lines, [line.get_label() for line in lines], loc="center right")
        ax.set_title(title)
        fig.tight_layout()
    return fig


def plot_validation_curve(curve: pd.DataFrame, chance: float) -> plt.Figure:
    """Share of top-k novel predictions reported by an independent interactome map."""
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for i, (method, part) in enumerate(curve.groupby("method", sort=False)):
            ax.plot(part["k"], 100 * part["in_any"], marker="o", markersize=3.5, color=PALETTE[i], label=method)
        ax.axhline(100 * chance, color=INK["muted"], linestyle=(0, (5, 2)), linewidth=1.2,
                   label=f"Random unobserved pairs ({100 * chance:.2f}%)")
        ax.set_xscale("log")
        ax.set(xlabel="Top-k novel predictions on HI-III", ylabel="Reported in BioPlex / Lit-BM / HI-II-14 (%)")
        ax.set_title("Independent support for predicted interactions")
        ax.legend(fontsize=8)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiments(
    dataset: str = "hi-iii",
    output_dir: str | Path = DEFAULT_OUTPUT,
    seeds: Sequence[int] = (0, 1, 2),
    epochs: int = 300,
    top_k: int = 1000,
    split_seed: int = 42,
) -> dict:
    """Run the ablation, save the reference model and validate novel predictions."""
    start = time.perf_counter()
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    graph = load_graph(processed_path(dataset))
    symbols = load_symbol_map(processed_path(dataset))
    annotations = load_annotations()

    # 1-3. Ablation
    runs, splits, models = run_ablation(graph, annotations, seeds=seeds, epochs=epochs, split_seed=split_seed)
    summary = summarise_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.round(5).to_csv(output_dir / "ablation_runs.csv", index=False)
    summary.round(4).to_csv(output_dir / "ablation_summary.csv", index=False)
    save_figure(plot_ablation(summary), figures / "ablation_average_precision.png")
    save_figure(plot_ablation(summary, "roc_auc"), figures / "ablation_roc_auc.png")

    # Reference GNN: GraphSAGE with structural + biological features, uniform negatives.
    reference: TrainedGNN = models[("uniform", "graphsage", "both")]
    structural: TrainedGNN = models[("uniform", "graphsage", "structural")]
    hybrid: TrainedGNN = models[("uniform", "graphsage+pairs", "both")]
    reference.save(output_dir / "model" / f"graphsage_{dataset}.pt")
    save_figure(plot_training_curve(reference.history, f"GraphSAGE training on {dataset.upper()}"),
                figures / "training_curve.png")
    split, classical = splits["uniform"]
    scores = {
        "graphsage": reference.predict_proba(split.test_graph, split.test_pairs),
        "gradient_boosting": classical.scores["gradient_boosting"],
        "l3": classical.scores["l3"],
    }
    kinds = {"graphsage": "gnn", "gradient_boosting": "model", "l3": "heuristic"}
    save_figure(plot_roc_curves(split.test_labels, scores, kinds), figures / "roc_curves.png")
    save_figure(plot_pr_curves(split.test_labels, scores, kinds), figures / "pr_curves.png")

    # 4. Novel predictions and external validation (messages over the full network)
    maps = load_interaction_sets()
    gnn_novel = rank_novel_interactions(reference, graph, top_k=top_k, symbols=symbols)
    hybrid_novel = rank_novel_interactions(hybrid, graph, top_k=top_k)
    structural_novel = rank_novel_interactions(structural, graph, top_k=top_k)
    samples = resolve_betweenness_samples("auto", graph.number_of_nodes())
    gb_novel = rank_classical(graph, classical.gradient_boosting, top_k=top_k, n_candidates=5 * top_k,
                              extractor=TopologicalFeatureExtractor(betweenness_samples=samples, seed=split_seed))
    l3_pairs = l3_candidate_pairs(graph, n_candidates=top_k)
    rankings = {
        "GraphSAGE (structural + biological)": list(zip(gnn_novel["protein_a"], gnn_novel["protein_b"])),
        "GraphSAGE (structural)": list(zip(structural_novel["protein_a"], structural_novel["protein_b"])),
        "GraphSAGE + pair heuristics": list(zip(hybrid_novel["protein_a"], hybrid_novel["protein_b"])),
        "Gradient boosting": list(zip(gb_novel["protein_a"], gb_novel["protein_b"])),
        "L3 paths": l3_pairs,
    }
    curve = validation_curve(rankings, maps)
    chance = random_pair_support(graph, maps)
    curve.round(4).to_csv(output_dir / "external_validation.csv", index=False)
    save_figure(plot_validation_curve(curve, float(chance["in_any"])), figures / "external_validation.png")

    top = gnn_novel.head(100).reset_index(drop=True)
    top = pd.concat([top, support_flags(list(zip(top["protein_a"], top["protein_b"])), maps)], axis=1)
    top.round(4).to_csv(output_dir / "novel_predictions.csv", index=False)

    results = {
        "dataset": dataset,
        "graph": {"proteins": graph.number_of_nodes(), "interactions": graph.number_of_edges()},
        "seeds": list(seeds),
        "split_seed": split_seed,
        "reference_model": {
            "architecture": "graphsage",
            "node_features": reference.features.kind,
            "n_node_features": len(reference.features.names),
            "best_epoch": reference.best_epoch,
            "validation_average_precision": reference.validation_average_precision,
            "test": compute_metrics(split.test_labels, scores["graphsage"], threshold=None),
        },
        "random_pair_support": chance.to_dict(),
        "hub_fraction_top100": {m: hub_fraction(p[:100], graph) for m, p in rankings.items()},
        "runtime_seconds": round(time.perf_counter() - start, 1),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("GNN experiments written to %s (%.0fs)", output_dir, results["runtime_seconds"])
    results.update(summary=summary, validation=curve, novel=top)
    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="GNN ablation and validation on a human interactome.")
    parser.add_argument("--dataset", default="hi-iii")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=3, help="random seeds per GNN configuration")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=1000, help="novel predictions to validate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    use_headless_backend()
    return run_experiments(args.dataset, args.output_dir, tuple(range(args.seeds)), args.epochs, args.top_k)


if __name__ == "__main__":
    main()
