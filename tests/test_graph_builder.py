from __future__ import annotations

import gzip

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.graph_builder import (
    build_graph,
    clean_edges,
    graph_summary,
    load_edge_list,
    load_graph,
    make_link_prediction_split,
    sample_holdout_edges,
    sample_negative_pairs,
)


def _as_set(pairs):
    return {frozenset(p) for p in pairs}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_clean_edges_removes_self_loops_and_duplicates():
    raw = pd.DataFrame(
        {"a": ["A", "B", "A", "C", " D"], "b": ["B", "A", "A", "D", "C "], "w": [5, 9, 1, 3, 4]}
    )
    edges = clean_edges(raw, "a", "b", "w")
    assert list(edges.columns) == ["protein_a", "protein_b", "weight"]
    assert _as_set(zip(edges.protein_a, edges.protein_b)) == {frozenset("AB"), frozenset("CD")}
    # Duplicate undirected pairs keep the highest confidence.
    assert edges.set_index(["protein_a", "protein_b"]).loc[("A", "B"), "weight"] == 9


def test_clean_edges_min_score_and_missing_columns():
    raw = pd.DataFrame({"a": ["A", "A"], "b": ["B", "C"], "w": [100, 900]})
    assert len(clean_edges(raw, "a", "b", "w", min_score=500)) == 1
    with pytest.raises(KeyError):
        clean_edges(raw, "a", "missing")


def test_load_tsv_edge_list_infers_weight_column(tmp_path):
    path = tmp_path / "edges.tsv"
    path.write_text("protein_a\tprotein_b\tscore\tsource\nA\tB\t800\tx\nB\tC\t300\ty\nB\tA\t200\tx\n")
    edges = load_edge_list(path)
    assert len(edges) == 2
    assert edges["weight"].max() == 800
    assert len(load_edge_list(path, min_score=500)) == 1


def test_load_csv_without_weight_defaults_to_one(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("x,y\nA,B\nB,C\n")
    edges = load_edge_list(path)
    assert edges["weight"].tolist() == [1.0, 1.0]


def test_load_string_gz_is_detected(tmp_path):
    path = tmp_path / "4932.protein.links.v12.0.txt.gz"
    content = (
        "protein1 protein2 combined_score\n"
        "4932.A 4932.B 900\n4932.B 4932.A 900\n4932.A 4932.C 400\n4932.C 4932.A 400\n"
    )
    with gzip.open(path, "wt") as handle:
        handle.write(content)
    edges = load_edge_list(path)
    assert len(edges) == 2
    assert len(load_edge_list(path, min_score=700)) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_edge_list(tmp_path / "nope.tsv")


def test_build_graph_keeps_largest_component():
    edges = pd.DataFrame(
        {"protein_a": ["A", "B", "X"], "protein_b": ["B", "C", "Y"], "weight": [1.0, 2.0, 3.0]}
    )
    assert set(build_graph(edges).nodes()) == {"A", "B", "C"}
    full = build_graph(edges, largest_component=False)
    assert full.number_of_nodes() == 5
    assert full["B"]["C"]["weight"] == 2.0


def test_load_graph_and_summary(tmp_path):
    path = tmp_path / "edges.tsv"
    path.write_text("a\tb\nA\tB\nB\tC\nC\tA\nC\tD\n")
    graph = load_graph(path)
    summary = graph_summary(graph)
    assert summary["proteins"] == 4
    assert summary["interactions"] == 4
    assert summary["connected_components"] == 1
    assert summary["largest_component_fraction"] == 1.0
    assert 0 < summary["average_clustering"] <= 1


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["uniform", "local"])
def test_link_prediction_split_invariants(small_ppi_graph, strategy):
    graph = small_ppi_graph
    split = make_link_prediction_split(
        graph, test_fraction=0.1, train_fraction=0.1, negative_strategy=strategy, seed=3
    )
    m = graph.number_of_edges()
    n_hidden = round(0.1 * m)

    train_pos = _as_set(p for p, y in zip(split.train_pairs, split.train_labels) if y == 1)
    train_neg = _as_set(p for p, y in zip(split.train_pairs, split.train_labels) if y == 0)
    test_pos = _as_set(p for p, y in zip(split.test_pairs, split.test_labels) if y == 1)
    test_neg = _as_set(p for p, y in zip(split.test_pairs, split.test_labels) if y == 0)

    assert len(train_pos) == len(test_pos) == n_hidden
    assert len(train_neg) == len(test_neg) == n_hidden
    assert not train_pos & test_pos
    assert not train_neg & test_neg

    # Positives are real edges, negatives are not.
    assert all(graph.has_edge(*p) for p in train_pos | test_pos)
    assert not any(graph.has_edge(*p) for p in train_neg | test_neg)

    # No leakage: each split's positives are absent from its feature graph,
    # and test positives are absent from the training graph too.
    assert not any(split.test_graph.has_edge(*p) for p in test_pos)
    assert not any(split.train_graph.has_edge(*p) for p in train_pos | test_pos)
    assert split.test_graph.number_of_edges() == m - n_hidden
    assert split.train_graph.number_of_edges() == m - 2 * n_hidden

    # Hiding edges never disconnects the network.
    assert nx.is_connected(split.train_graph)
    assert set(split.train_graph.nodes()) == set(graph.nodes())


def test_split_is_reproducible(small_ppi_graph):
    a = make_link_prediction_split(small_ppi_graph, seed=11)
    b = make_link_prediction_split(small_ppi_graph, seed=11)
    c = make_link_prediction_split(small_ppi_graph, seed=12)
    assert a.test_pairs == b.test_pairs
    np.testing.assert_array_equal(a.train_labels, b.train_labels)
    assert a.test_pairs != c.test_pairs


def test_local_negatives_are_close(small_ppi_graph):
    rng = np.random.default_rng(0)
    pairs = sample_negative_pairs(small_ppi_graph, 50, rng, strategy="local")
    distances = [nx.shortest_path_length(small_ppi_graph, u, v) for u, v in pairs]
    assert max(distances) <= 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_fraction": 0.0},
        {"train_fraction": 1.0},
        {"test_fraction": 0.6, "train_fraction": 0.5},
        {"negative_ratio": 0},
    ],
)
def test_split_rejects_invalid_arguments(karate_graph, kwargs):
    with pytest.raises(ValueError):
        make_link_prediction_split(karate_graph, **kwargs)


def test_holdout_refuses_to_disconnect_a_tree():
    tree = nx.balanced_tree(2, 3)
    with pytest.raises(ValueError):
        sample_holdout_edges(tree, 1, np.random.default_rng(0))


def test_unknown_negative_strategy(karate_graph):
    with pytest.raises(ValueError):
        sample_negative_pairs(karate_graph, 5, np.random.default_rng(0), strategy="nearby")
