"""Tests for the interpretation helpers."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.interpretation import (
    CENTRALITY_CLASSES,
    between_module_rate,
    bottleneck_interactions,
    centrality_classes,
    cluster_embeddings,
    cluster_enrichment,
    embedding_map,
    enriched_cluster_fraction,
    functional_similarity_auc,
    interaction_propensity,
)


@pytest.fixture
def modular():
    """Two dense communities joined by one bridge, with embeddings and gene sets matching them."""
    graph = nx.disjoint_union(nx.complete_graph(30), nx.complete_graph(30))
    graph.add_edge(0, 30)
    graph = nx.relabel_nodes(graph, str)
    rng = np.random.default_rng(0)
    centres = np.r_[np.ones(8), np.zeros(8)], np.r_[np.zeros(8), np.ones(8)]
    rows = [centres[int(n) >= 30] + 0.05 * rng.normal(size=16) for n in graph.nodes()]
    embeddings = pd.DataFrame(rows, index=list(graph.nodes()))
    gene_sets = {
        "module A": frozenset(str(i) for i in range(30)),
        "module B": frozenset(str(i) for i in range(30, 60)),
        "mixed": frozenset(str(i) for i in range(0, 60, 7)),
    }
    return graph, embeddings, gene_sets


def test_clusters_recover_modules_and_are_enriched(modular):
    graph, embeddings, gene_sets = modular
    labels = cluster_embeddings(embeddings, n_clusters=2)
    assert labels.value_counts().tolist() == [30, 30]
    assert labels["0"] != labels["30"] and labels["0"] == labels["1"]
    table = cluster_enrichment(labels, gene_sets, max_size=100)
    assert set(table["top_term"]) == {"module A", "module B"}
    assert (table["fdr"] < 1e-10).all()
    observed, null = enriched_cluster_fraction(labels, gene_sets, n_permutations=5)
    assert observed == 1.0 and null.shape == (5,)


def test_functional_similarity_auc(modular):
    _, embeddings, gene_sets = modular
    sets = {k: v for k, v in gene_sets.items() if k != "mixed"}
    assert functional_similarity_auc(embeddings, sets, n_pairs=2000) > 0.95
    shuffled = embeddings.sample(frac=1, random_state=0).set_axis(embeddings.index)
    assert abs(functional_similarity_auc(shuffled, sets, n_pairs=4000) - 0.5) < 0.1


def test_centrality_classes_and_bridges(modular):
    graph, embeddings, _ = modular
    table = centrality_classes(graph, symbols={"0": "BRIDGE"}, betweenness_samples=None)
    assert table.index[0] in {"0", "30"}
    assert set(table["centrality_class"]) <= set(CENTRALITY_CLASSES)
    assert table.loc["0", "symbol"] == "BRIDGE"
    labels = cluster_embeddings(embeddings, n_clusters=2)
    bridges = bottleneck_interactions(graph, labels, top=3, samples=None)
    assert {bridges.iloc[0]["protein_a"], bridges.iloc[0]["protein_b"]} == {"0", "30"}
    assert bool(bridges.iloc[0]["between_modules"])
    assert between_module_rate(graph, labels) == pytest.approx(1 / graph.number_of_edges())


def test_embedding_map_shape(modular):
    _, embeddings, _ = modular
    coordinates = embedding_map(embeddings, perplexity=10)
    assert coordinates.shape == (60, 2) and list(coordinates.index) == list(embeddings.index)


def test_interaction_propensity_one_value_per_protein():
    class Constant:
        nodes = ["a", "b", "c"]

        def predict_proba(self, graph, pairs):
            return np.array([0.5 if "a" in p else 0.1 for p in pairs])

    propensity = interaction_propensity(Constant(), nx.path_graph(["a", "b", "c"]), n_partners=20)
    assert propensity["a"] == pytest.approx(0.5)
    assert propensity["b"] < propensity["a"] and propensity.between(0, 1).all()
