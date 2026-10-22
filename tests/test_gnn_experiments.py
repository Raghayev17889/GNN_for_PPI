"""Tests for the GNN experiment helpers (no downloads)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.gnn_experiments import (
    VARIANTS,
    hub_fraction,
    plot_ablation,
    plot_validation_curve,
    random_pair_support,
    run_ablation,
    summarise_runs,
    support_flags,
    validation_curve,
)
from src.node_features import canonical_nodes


@pytest.fixture
def maps():
    return {"a": {frozenset(("1", "2")), frozenset(("3", "4"))}, "b": {frozenset(("2", "5"))}}


def test_support_flags(maps):
    flags = support_flags([("2", "1"), ("2", "5"), ("1", "5")], maps)
    assert flags.to_dict(orient="list") == {
        "in_a": [True, False, False],
        "in_b": [False, True, False],
        "in_any": [True, True, False],
    }


def test_validation_curve_counts_top_k(maps):
    rankings = {"m": [("1", "2"), ("9", "8"), ("2", "5"), ("7", "6")]}
    curve = validation_curve(rankings, maps, ks=(1, 2, 4, 10))
    assert curve["k"].tolist() == [1, 2, 4]  # k beyond the ranking is skipped
    assert curve["in_any"].tolist() == [1.0, 0.5, 0.5]


def test_random_pair_support_is_small(maps):
    graph = nx.relabel_nodes(nx.path_graph(50), str)
    support = random_pair_support(graph, maps, n_pairs=5000)
    assert 0 < support["in_any"] < 0.01


def test_hub_fraction():
    graph = nx.star_graph(10)
    graph.add_edge(1, 2)
    assert hub_fraction([(1, 2), (0, 3)], graph, top_percent=5) == pytest.approx(0.5)
    assert np.isnan(hub_fraction([], graph))


def test_ablation_end_to_end(small_ppi_graph):
    nodes = canonical_nodes(small_ppi_graph)
    annotations = {"go_bp": {f"t{i}": frozenset(str(n) for n in nodes[i::4]) for i in range(4)}}
    runs, splits, models = run_ablation(
        small_ppi_graph, annotations, strategies=("uniform",), variants=("graphsage", "graphsage+pairs"),
        feature_kinds=("structural", "both"), seeds=(0, 1), epochs=10,
    )
    assert set(splits) == {"uniform"}
    assert len(runs) == 3 + 2 * 2 * 2  # three references, 2 variants x 2 feature sets x 2 seeds
    assert ("uniform", "graphsage+pairs", "both") in models
    summary = summarise_runs(runs)
    row = summary.set_index("method").loc["graphsage/both"]
    assert row["n_seeds"] == 2 and 0 <= row["roc_auc_mean"] <= 1
    assert plot_ablation(summary).axes
    curve = pd.DataFrame({"method": ["x", "x"], "k": [10, 100], "in_any": [0.2, 0.1]})
    assert plot_validation_curve(curve, 0.001).axes


def test_variants_cover_every_architecture():
    assert {architecture for architecture, _ in VARIANTS.values()} == {"graphsage", "gcn", "mlp"}
    assert any(pairs for _, pairs in VARIANTS.values())
