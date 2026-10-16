"""Topological feature extraction for protein pairs.

A candidate interaction ``(u, v)`` is described by the network position of
both proteins and by how their neighbourhoods overlap. Every feature is
symmetric, so the vector of ``(u, v)`` equals that of ``(v, u)``.

Node-level metrics (combined as ``min`` / ``max`` over the two proteins)
-----------------------------------------------------------------------
degree
    Number of interaction partners. Hubs interact promiscuously.
clustering
    Fraction of a protein's partners that interact with each other; high in
    stable complexes.
betweenness
    Fraction of shortest paths passing through the protein (bottlenecks).
closeness
    Inverse mean shortest-path distance to all other proteins.

Pairwise heuristics
-------------------
common_neighbors
    ``|N(u) & N(v)|`` - triadic closure.
jaccard
    ``|N(u) & N(v)| / |N(u) | N(v)|`` - normalised neighbourhood overlap.
adamic_adar
    Shared partners weighted by ``1 / log(degree)``; rare partners count more.
resource_allocation
    Shared partners weighted by ``1 / degree``.
preferential_attachment
    ``deg(u) * deg(v)``.
l3
    Degree-normalised number of paths of length three,
    ``sum_{a,b} A_ua A_ab A_bv / sqrt(k_a k_b)``. Interacting proteins tend to
    be *similar to each other's partners* rather than share partners, because
    binding relies on complementary interfaces; L3 captures this and
    outperforms triadic-closure scores on PPI networks
    (Kovacs et al., Nature Communications 10, 1240, 2019).
shortest_path
    Hop distance between the proteins, capped at ``max_path_length + 1``
    (also used when no path exists).
"""

from __future__ import annotations

import logging
from typing import Hashable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

NODE_METRICS: tuple[str, ...] = ("degree", "clustering", "betweenness", "closeness")
PAIR_HEURISTICS: tuple[str, ...] = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
    "l3",
    "shortest_path",
)


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


def _rowwise_dot(left: sparse.csr_array, right: sparse.csr_array) -> np.ndarray:
    """``sum_j left[i, j] * right[i, j]`` for every row ``i``."""
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def pairwise_heuristics(
    adjacency: sparse.csr_array,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Neighbourhood-based link-prediction scores for many pairs at once.

    Parameters
    ----------
    adjacency:
        Binary, symmetric CSR adjacency matrix of the graph.
    u_idx, v_idx:
        Row indices of the two proteins of each pair.

    Returns
    -------
    dict mapping heuristic name (all of :data:`PAIR_HEURISTICS` except
    ``shortest_path``) to an array of scores.
    """
    adjacency = sparse.csr_array(adjacency, dtype=float)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    rows_u, rows_v = adjacency[u_idx], adjacency[v_idx]

    with np.errstate(divide="ignore"):
        inv_log_degree = np.where(degree > 1, 1.0 / np.log(np.maximum(degree, 2)), 0.0)
        inv_degree = np.where(degree > 0, 1.0 / np.maximum(degree, 1), 0.0)
        inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1)), 0.0)

    shared = rows_u.multiply(rows_v)  # 1 where a protein is a common partner
    common = np.asarray(shared.sum(axis=1)).ravel()
    deg_u, deg_v = degree[u_idx], degree[v_idx]
    union = deg_u + deg_v - common
    jaccard = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
    adamic_adar = shared @ inv_log_degree
    resource_allocation = shared @ inv_degree

    # L3: (A D^-1/2 A D^-1/2 A)_uv, evaluated only for the requested pairs.
    normalised = sparse.diags_array(inv_sqrt_degree) @ adjacency @ sparse.diags_array(inv_sqrt_degree)
    l3 = _rowwise_dot(sparse.csr_array(rows_u @ normalised), rows_v)

    return {
        "common_neighbors": common,
        "jaccard": jaccard,
        "adamic_adar": np.asarray(adamic_adar).ravel(),
        "resource_allocation": np.asarray(resource_allocation).ravel(),
        "preferential_attachment": deg_u * deg_v,
        "l3": l3,
    }


def shortest_path_lengths(
    graph: nx.Graph,
    pairs: Sequence[tuple[Hashable, Hashable]],
    max_path_length: int = 6,
) -> np.ndarray:
    """Hop distance for each pair, capped at ``max_path_length + 1``."""
    cap = max_path_length + 1
    lengths = np.full(len(pairs), cap, dtype=float)
    for i, (u, v) in enumerate(pairs):
        try:
            lengths[i] = min(nx.shortest_path_length(graph, u, v), cap)
        except nx.NetworkXNoPath:
            pass
    return lengths


class TopologicalFeatureExtractor:
    """Fit topological statistics on a graph, then featurise protein pairs.

    Parameters
    ----------
    max_path_length:
        Distances beyond this are capped (see :func:`shortest_path_lengths`).

    Example
    -------
    >>> extractor = TopologicalFeatureExtractor().fit(split.train_graph)
    >>> X_train = extractor.transform(split.train_pairs)

    Fit on the graph the pairs should be scored against (for link
    prediction: the graph *without* the pairs' own edges).
    """

    def __init__(self, max_path_length: int = 6) -> None:
        if max_path_length < 1:
            raise ValueError("max_path_length must be >= 1")
        self.max_path_length = max_path_length

    def fit(self, graph: nx.Graph) -> "TopologicalFeatureExtractor":
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot fit on an empty graph")
        self.graph_ = graph
        self.nodes_ = list(graph.nodes())
        self.index_ = {node: i for i, node in enumerate(self.nodes_)}
        self.adjacency_ = nx.to_scipy_sparse_array(
            graph, nodelist=self.nodes_, weight=None, dtype=float, format="csr"
        )
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
        features.update(pairwise_heuristics(self.adjacency_, u_idx, v_idx))
        features["shortest_path"] = shortest_path_lengths(self.graph_, pairs, self.max_path_length)
        return pd.DataFrame(features)[self.feature_names]

    def fit_transform(self, graph: nx.Graph, pairs: Sequence[tuple[Hashable, Hashable]]) -> pd.DataFrame:
        return self.fit(graph).transform(pairs)

    @property
    def feature_names(self) -> list[str]:
        node = [f"{m}_{agg}" for m in NODE_METRICS for agg in ("min", "max")]
        return node + list(PAIR_HEURISTICS)


def extract_features(
    graph: nx.Graph,
    pairs: Sequence[tuple[Hashable, Hashable]],
    **kwargs,
) -> pd.DataFrame:
    """Functional shortcut for ``TopologicalFeatureExtractor(**kwargs).fit_transform``."""
    return TopologicalFeatureExtractor(**kwargs).fit_transform(graph, pairs)
