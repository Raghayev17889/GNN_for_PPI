from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
import pytest

from src.feature_extractor import (
    NODE_METRICS,
    PAIR_HEURISTICS,
    TopologicalFeatureExtractor,
    compute_node_metrics,
    extract_features,
    shortest_path_lengths,
)


@pytest.fixture
def non_edges(karate_graph):
    nodes = sorted(karate_graph.nodes())
    return [p for p in itertools.combinations(nodes, 2) if not karate_graph.has_edge(*p)]


@pytest.fixture
def features(karate_graph, non_edges):
    return extract_features(karate_graph, non_edges)


def test_feature_columns(features):
    expected = [f"{m}_{agg}" for m in NODE_METRICS for agg in ("min", "max")]
    assert features.columns.tolist() == expected + list(PAIR_HEURISTICS)
    assert TopologicalFeatureExtractor().feature_names == features.columns.tolist()
    assert np.isfinite(features.to_numpy()).all()


@pytest.mark.parametrize(
    "name, reference",
    [
        ("jaccard", nx.jaccard_coefficient),
        ("adamic_adar", nx.adamic_adar_index),
        ("resource_allocation", nx.resource_allocation_index),
        ("preferential_attachment", nx.preferential_attachment),
    ],
)
def test_heuristics_match_networkx(karate_graph, non_edges, features, name, reference):
    expected = [score for *_, score in reference(karate_graph, non_edges)]
    np.testing.assert_allclose(features[name], expected)


def test_common_neighbors(karate_graph, non_edges, features):
    expected = [len(list(nx.common_neighbors(karate_graph, u, v))) for u, v in non_edges]
    np.testing.assert_allclose(features["common_neighbors"], expected)


def test_l3_matches_brute_force(karate_graph, non_edges, features):
    k = dict(karate_graph.degree())
    expected = [
        sum(
            1.0 / math.sqrt(k[a] * k[b])
            for a in karate_graph[u]
            for b in karate_graph[a]
            if karate_graph.has_edge(b, v)
        )
        for u, v in non_edges
    ]
    np.testing.assert_allclose(features["l3"], expected)


def test_node_metrics_match_networkx(karate_graph):
    metrics = compute_node_metrics(karate_graph)
    assert metrics.columns.tolist() == list(NODE_METRICS)
    for name, reference in [
        ("clustering", nx.clustering(karate_graph)),
        ("betweenness", nx.betweenness_centrality(karate_graph)),
        ("closeness", nx.closeness_centrality(karate_graph)),
    ]:
        np.testing.assert_allclose(metrics[name], [reference[n] for n in metrics.index])


def test_features_are_symmetric(karate_graph, non_edges, features):
    swapped = extract_features(karate_graph, [(v, u) for u, v in non_edges])
    np.testing.assert_allclose(features.to_numpy(), swapped.to_numpy())


def test_min_max_node_features(karate_graph, features, non_edges):
    degree = dict(karate_graph.degree())
    u, v = non_edges[0]
    assert features.loc[0, "degree_min"] == min(degree[u], degree[v])
    assert features.loc[0, "degree_max"] == max(degree[u], degree[v])


def test_shortest_path_is_capped_and_handles_components():
    graph = nx.path_graph(6)
    graph.add_edge("a", "b")
    lengths = shortest_path_lengths(graph, [(0, 2), (0, 5), (0, "a")], max_path_length=3)
    np.testing.assert_array_equal(lengths, [2, 4, 4])


def test_isolated_proteins_get_zero_scores():
    graph = nx.Graph([("A", "B"), ("B", "C")])
    graph.add_node("Z")
    X = extract_features(graph, [("A", "Z"), ("A", "C")])
    assert X.loc[0, ["common_neighbors", "adamic_adar", "l3", "jaccard"]].eq(0).all()
    assert X.loc[1, "common_neighbors"] == 1


def test_unknown_protein_raises(karate_graph):
    extractor = TopologicalFeatureExtractor().fit(karate_graph)
    with pytest.raises(KeyError, match="not in the fitted graph"):
        extractor.transform([("P000", "UNKNOWN")])


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        TopologicalFeatureExtractor().transform([("A", "B")])


def test_invalid_parameters():
    with pytest.raises(ValueError):
        TopologicalFeatureExtractor(max_path_length=0)
    with pytest.raises(ValueError):
        TopologicalFeatureExtractor().fit(nx.Graph())
