"""Tests for the graph neural network link predictor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.gnn_model import (
    Encoder,
    GNNConfig,
    LinkPredictor,
    PairDecoder,
    TrainedGNN,
    adjacency_from_edges,
    edge_index_from_adjacency,
    pair_keys,
    rank_novel_interactions,
    sample_local_negatives,
    sample_uniform_negatives,
    train_link_predictor,
    upper_edges,
)
from src.graph_builder import make_link_prediction_split
from src.node_features import NodeFeatures, adjacency_matrix

FAST = dict(epochs=40, eval_every=5, patience=40, hidden_channels=32, embedding_dim=16, decoder_hidden=16)


@pytest.fixture
def split(small_ppi_graph):
    return make_link_prediction_split(small_ppi_graph, test_fraction=0.1, train_fraction=0.1, seed=3)


@pytest.fixture
def trained(split, small_ppi_graph):
    features = NodeFeatures.from_graph(small_ppi_graph, "structural")
    return train_link_predictor(split, features, GNNConfig(seed=0, **FAST))


@pytest.mark.parametrize("architecture", ["graphsage", "gcn", "mlp"])
def test_encoder_shapes(architecture):
    encoder = Encoder(5, 8, 4, architecture)
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    assert encoder(x, edge_index).shape == (6, 4)


def test_message_passing_uses_the_graph():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    path = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    star = torch.tensor([[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]])
    for architecture, uses_graph in (("graphsage", True), ("gcn", True), ("mlp", False)):
        encoder = Encoder(3, 8, 4, architecture).eval()
        assert torch.allclose(encoder(x, path), encoder(x, star)) != uses_graph


def test_decoder_is_symmetric():
    torch.manual_seed(0)
    decoder = PairDecoder(4, 8).eval()
    h = torch.randn(5, 4)
    pairs = torch.tensor([[0, 1, 2], [3, 4, 0]])
    assert torch.allclose(decoder(h, pairs), decoder(h, pairs.flip(0)))


def test_edge_helpers_round_trip(small_ppi_graph):
    nodes = sorted(small_ppi_graph, key=str)
    adjacency = adjacency_matrix(small_ppi_graph, nodes)
    edges = upper_edges(adjacency)
    assert edges.shape == (2, small_ppi_graph.number_of_edges()) and (edges[0] < edges[1]).all()
    assert (adjacency_from_edges(edges, len(nodes)) != adjacency).nnz == 0
    assert edge_index_from_adjacency(adjacency).shape == (2, 2 * small_ppi_graph.number_of_edges())


def test_negative_samplers_avoid_known_pairs(small_ppi_graph):
    nodes = sorted(small_ppi_graph, key=str)
    adjacency = adjacency_matrix(small_ppi_graph, nodes)
    n = len(nodes)
    known = np.sort(pair_keys(*upper_edges(adjacency), n))
    rng = np.random.default_rng(0)
    for negatives in (
        sample_uniform_negatives(n, 500, known, rng),
        sample_local_negatives(adjacency, 300, known, rng),
    ):
        keys = pair_keys(negatives[0], negatives[1], n)
        assert len(np.unique(keys)) == negatives.shape[1]
        assert not np.isin(keys, known).any()
        assert (negatives[0] != negatives[1]).all()
    # Local negatives are close in the network.
    distance = adjacency @ adjacency
    local = sample_local_negatives(adjacency, 200, known, rng)
    reach3 = distance + distance @ adjacency
    assert all(reach3[u, v] > 0 for u, v in local.T)


def test_training_learns_and_beats_chance(trained, split):
    scores = trained.predict_proba(split.test_graph, split.test_pairs)
    assert scores.shape == (len(split.test_pairs),) and ((0 <= scores) & (scores <= 1)).all()
    assert roc_auc_score(split.test_labels, scores) > 0.65
    assert trained.best_epoch > 0 and not trained.history.empty
    assert set(trained.history.columns) == {"epoch", "loss", "val_average_precision", "val_roc_auc"}


def test_training_is_reproducible(split, small_ppi_graph):
    runs = [
        train_link_predictor(split, NodeFeatures.from_graph(small_ppi_graph, "structural"), GNNConfig(seed=1, **FAST))
        for _ in range(2)
    ]
    np.testing.assert_allclose(
        runs[0].predict_proba(split.test_graph, split.test_pairs),
        runs[1].predict_proba(split.test_graph, split.test_pairs),
    )


def test_local_negative_training_runs(split, small_ppi_graph):
    features = NodeFeatures.from_graph(small_ppi_graph, "structural")
    model = train_link_predictor(split, features, GNNConfig(negative_strategy="local", seed=0, **FAST))
    assert model.predict_proba(split.test_graph, split.test_pairs[:5]).shape == (5,)


def test_save_and_load_round_trip(trained, split, tmp_path):
    path = trained.save(tmp_path / "model.pt")
    loaded = TrainedGNN.load(path)
    assert loaded.config == trained.config and loaded.nodes == [str(n) for n in trained.nodes]
    np.testing.assert_allclose(
        loaded.predict_proba(split.test_graph, split.test_pairs),
        trained.predict_proba(split.test_graph, split.test_pairs),
        rtol=1e-5,
    )
    assert loaded.embed(split.test_graph).shape == (300, 16)


def test_rank_novel_interactions_matches_brute_force(trained, small_ppi_graph):
    table = rank_novel_interactions(trained, small_ppi_graph, top_k=20, block_size=7)
    assert len(table) == 20 and table["probability"].is_monotonic_decreasing
    assert not any(small_ppi_graph.has_edge(u, v) for u, v in zip(table["protein_a"], table["protein_b"]))
    nodes = trained.nodes
    candidates = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))
                  if not small_ppi_graph.has_edge(nodes[i], nodes[j])]
    scores = trained.predict_proba(small_ppi_graph, candidates)
    np.testing.assert_allclose(table["probability"].to_numpy(), np.sort(scores)[::-1][:20], rtol=1e-5)


def test_predict_rejects_foreign_graph(trained, karate_graph):
    with pytest.raises(ValueError):
        trained.embed(karate_graph)


def test_config_validation():
    with pytest.raises(ValueError):
        GNNConfig(architecture="gat")
    with pytest.raises(ValueError):
        GNNConfig(supervision_fraction=1.5)
    model = LinkPredictor(4, GNNConfig(**FAST))
    assert sum(p.numel() for p in model.parameters()) > 0

