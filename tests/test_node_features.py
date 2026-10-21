"""Tests for GNN node features."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.node_features import (
    STRUCTURAL_FEATURES,
    NodeFeatures,
    adjacency_matrix,
    annotation_matrix,
    biological_features,
    canonical_nodes,
    structural_features,
)


def _toy_annotations(nodes):
    nodes = [str(n) for n in nodes]
    return {
        "go_bp": {f"bp{i}": frozenset(nodes[i::5]) for i in range(5)},
        "tissue": {"brain": frozenset(nodes[:20]), "liver": frozenset(nodes[10:30])},
    }


def test_structural_features_match_networkx(karate_graph):
    nodes = canonical_nodes(karate_graph)
    values = structural_features(adjacency_matrix(karate_graph, nodes))
    assert values.shape == (34, len(STRUCTURAL_FEATURES))
    clustering = nx.clustering(karate_graph)
    neighbour_degree = nx.average_neighbor_degree(karate_graph)
    for i, node in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(karate_graph, node, cutoff=2)
        two_hop = sum(1 for d in lengths.values() if d == 2)
        assert values[i, 0] == pytest.approx(np.log1p(karate_graph.degree(node)))
        assert values[i, 1] == pytest.approx(clustering[node])
        assert values[i, 2] == pytest.approx(np.log1p(neighbour_degree[node]))
        assert values[i, 3] == pytest.approx(np.log1p(two_hop))


def test_annotation_matrix_filters_rare_terms():
    matrix, terms = annotation_matrix(["a", "b", "c"], {"x": {"a", "b"}, "y": {"c"}, "z": {"q"}})
    assert terms == ["x"]
    assert matrix.toarray().tolist() == [[1.0], [1.0], [0.0]]


def test_biological_features_shape_and_indicator(small_ppi_graph):
    nodes = canonical_nodes(small_ppi_graph)
    table = biological_features(nodes, _toy_annotations(nodes), n_components=3)
    assert list(table.index) == nodes
    assert {"has_go_bp", "has_tissue", "go_bp_svd01", "go_bp_svd03", "tissue_svd01"} <= set(table.columns)
    assert "tissue_svd02" not in table.columns  # only two tissue terms -> one component
    assert table["has_tissue"].sum() == 30


def test_node_features_standardised_on_fit_graph(small_ppi_graph):
    features = NodeFeatures.from_graph(small_ppi_graph, "both", _toy_annotations(small_ppi_graph.nodes()))
    adjacency = adjacency_matrix(small_ppi_graph, features.nodes)
    x = features.fit(adjacency).transform(adjacency)
    assert x.dtype == np.float32 and x.shape == (300, len(features.names))
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-5)
    frame = features.frame(adjacency)
    assert isinstance(frame, pd.DataFrame) and list(frame.columns) == features.names


def test_structural_block_follows_the_graph(small_ppi_graph):
    features = NodeFeatures.from_graph(small_ppi_graph, "structural")
    full = adjacency_matrix(small_ppi_graph, features.nodes)
    features.fit(full)
    reduced = small_ppi_graph.copy()
    reduced.remove_edges_from(list(reduced.edges())[:100])
    assert not np.allclose(features.transform(full), features.transform(adjacency_matrix(reduced, features.nodes)))


def test_node_features_validation(small_ppi_graph):
    with pytest.raises(ValueError):
        NodeFeatures.from_graph(small_ppi_graph, "both")  # annotations missing
    with pytest.raises(ValueError):
        NodeFeatures(["a"], "everything")
    features = NodeFeatures.from_graph(small_ppi_graph, "structural")
    with pytest.raises(RuntimeError):
        features.transform(adjacency_matrix(small_ppi_graph, features.nodes))
    with pytest.raises(ValueError):
        features.raw(adjacency_matrix(nx.path_graph(3), [0, 1, 2]))
