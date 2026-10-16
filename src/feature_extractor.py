"""Topological feature extraction for protein pairs.

A candidate interaction ``(u, v)`` is described by the network position of
both proteins. Node-level metrics are computed once per graph and combined
symmetrically (``min`` and ``max`` over the two proteins), so the feature
vector of ``(u, v)`` equals that of ``(v, u)``.

Node-level metrics
------------------
degree
    Number of interaction partners. Hubs interact promiscuously.
clustering
    Fraction of a protein's partners that interact with each other; high in
    stable complexes.
betweenness
    Fraction of shortest paths passing through the protein (bottlenecks).
closeness
    Inverse mean shortest-path distance to all other proteins.
"""

from __future__ import annotations

import logging
from typing import Hashable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NODE_METRICS: tuple[str, ...] = ("degree", "clustering", "betweenness", "closeness")


def compute_node_metrics(graph: nx.Graph) -> pd.DataFrame:
    """Compute node-level topological metrics.

    Returns
    -------
    DataFrame indexed by protein with one column per metric in
    :data:`NODE_METRICS`, rows in ``graph.nodes()`` order.
    """
    nodes = list(graph.nodes())
    logger.debug("Computing node metrics for %d proteins", len(nodes))
    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    closeness = nx.closeness_centrality(graph)
    return pd.DataFrame(
        {
            "degree": [degree[n] for n in nodes],
            "clustering": [clustering[n] for n in nodes],
            "betweenness": [betweenness[n] for n in nodes],
            "closeness": [closeness[n] for n in nodes],
        },
        index=pd.Index(nodes, name="protein"),
    ).astype(float)


def symmetric_pair_features(
    node_metrics: pd.DataFrame, u_idx: np.ndarray, v_idx: np.ndarray
) -> dict[str, np.ndarray]:
    """Combine node metrics of both proteins with order-invariant min/max."""
    values = node_metrics.to_numpy(dtype=float)
    a, b = values[u_idx], values[v_idx]
    features: dict[str, np.ndarray] = {}
    for j, name in enumerate(node_metrics.columns):
        features[f"{name}_min"] = np.minimum(a[:, j], b[:, j])
        features[f"{name}_max"] = np.maximum(a[:, j], b[:, j])
    return features


class TopologicalFeatureExtractor:
    """Fit topological statistics on a graph, then featurise protein pairs.

    Example
    -------
    >>> extractor = TopologicalFeatureExtractor().fit(split.train_graph)
    >>> X_train = extractor.transform(split.train_pairs)

    Fit on the graph the pairs should be scored against (for link
    prediction: the graph *without* the pairs' own edges).
    """

    def fit(self, graph: nx.Graph) -> "TopologicalFeatureExtractor":
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot fit on an empty graph")
        self.graph_ = graph
        self.nodes_ = list(graph.nodes())
        self.index_ = {node: i for i, node in enumerate(self.nodes_)}
        self.node_metrics_ = compute_node_metrics(graph)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "node_metrics_"):
            raise RuntimeError("TopologicalFeatureExtractor must be fitted before transform")

    def _pair_indices(self, pairs: Sequence[tuple[Hashable, Hashable]]):
        try:
            u_idx = np.fromiter((self.index_[u] for u, _ in pairs), dtype=np.int64, count=len(pairs))
            v_idx = np.fromiter((self.index_[v] for _, v in pairs), dtype=np.int64, count=len(pairs))
        except KeyError as exc:
            raise KeyError(f"Protein {exc.args[0]!r} is not in the fitted graph") from None
        return u_idx, v_idx

    def transform(self, pairs: Sequence[tuple[Hashable, Hashable]]) -> pd.DataFrame:
        """Return one row of features per pair (same order as ``pairs``)."""
        self._check_fitted()
        pairs = list(pairs)
        u_idx, v_idx = self._pair_indices(pairs)
        features = symmetric_pair_features(self.node_metrics_, u_idx, v_idx)
        return pd.DataFrame(features)

    def fit_transform(self, graph: nx.Graph, pairs: Sequence[tuple[Hashable, Hashable]]) -> pd.DataFrame:
        return self.fit(graph).transform(pairs)

    @property
    def feature_names(self) -> list[str]:
        return [f"{m}_{agg}" for m in NODE_METRICS for agg in ("min", "max")]
