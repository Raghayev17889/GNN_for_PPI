"""Node feature matrices for graph neural networks.

A graph neural network (GNN) combines what is known about each protein (its
node features) with the interaction network (message passing). Two kinds of
node features are provided, so their contributions can be separated:

* **structural** - four cheap descriptors of a protein's network position,
  recomputed on whichever graph the GNN propagates over (so hidden
  interactions never leak into them):

  ``log_degree``
      ``log(1 + k)``, number of interaction partners.
  ``clustering``
      local clustering coefficient.
  ``log_mean_neighbour_degree``
      ``log(1 + mean degree of the partners)``; low for proteins whose
      partners are peripheral, high for proteins attached to hubs.
  ``log_two_hop``
      ``log(1 + number of proteins exactly two steps away)``.

* **biological** - prior knowledge that does not depend on the network:
  Gene Ontology biological process and cellular component, pathway membership
  and tissue expression (:mod:`src.annotations`). Each binary
  protein x term matrix is compressed with truncated SVD into a few dense
  components (a latent-semantic embedding of the annotation profile), plus an
  indicator of whether the protein has any annotation from that source.

Example
-------
>>> features = NodeFeatures.from_graph(graph, kind="both", annotations=load_annotations())
>>> features.fit(train_adjacency)
>>> x = features.transform(train_adjacency)     # (n_proteins, n_features) float32
"""

from __future__ import annotations

import logging
from typing import Hashable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

FeatureKind = Literal["structural", "biological", "both"]
FEATURE_KINDS: tuple[str, ...] = ("structural", "biological", "both")
STRUCTURAL_FEATURES: tuple[str, ...] = (
    "log_degree",
    "clustering",
    "log_mean_neighbour_degree",
    "log_two_hop",
)


def canonical_nodes(graph: nx.Graph) -> list[Hashable]:
    """Deterministic node order shared by every graph of a split."""
    return sorted(graph.nodes(), key=str)


def adjacency_matrix(graph: nx.Graph, nodes: Sequence[Hashable]) -> sparse.csr_array:
    """Binary symmetric CSR adjacency matrix of ``graph`` in ``nodes`` order."""
    return nx.to_scipy_sparse_array(graph, nodelist=list(nodes), weight=None, dtype=np.float64, format="csr")


def structural_features(adjacency: sparse.csr_array) -> np.ndarray:
    """The :data:`STRUCTURAL_FEATURES` for every node, computed with sparse algebra.

    Runs in milliseconds for a 5,000-protein interactome, so it can be
    recomputed for every message-passing graph during training.
    """
    adjacency = sparse.csr_array(adjacency, dtype=np.float64)
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    two_step = sparse.csr_array(adjacency @ adjacency)
    triangles = np.asarray(two_step.multiply(adjacency).sum(axis=1)).ravel() / 2.0
    possible = degree * (degree - 1) / 2.0
    clustering = np.divide(triangles, possible, out=np.zeros_like(degree), where=possible > 0)
    mean_neighbour_degree = np.divide(adjacency @ degree, degree, out=np.zeros_like(degree), where=degree > 0)
    # Proteins reachable in two steps, excluding the protein itself and its partners.
    two_step.setdiag(0)
    two_step.eliminate_zeros()
    reach = (two_step > 0).astype(np.float64)
    within_two = np.asarray(reach.sum(axis=1)).ravel()
    partners_also_two_step = np.asarray(reach.multiply(adjacency).sum(axis=1)).ravel()
    two_hop = within_two - partners_also_two_step
    return np.column_stack(
        [np.log1p(degree), clustering, np.log1p(mean_neighbour_degree), np.log1p(two_hop)]
    )


def annotation_matrix(
    nodes: Sequence[Hashable],
    gene_sets: Mapping[str, frozenset[str] | set[str]],
    min_genes: int = 2,
) -> tuple[sparse.csr_array, list[str]]:
    """Binary protein x term matrix, keeping terms with ``>= min_genes`` of ``nodes``."""
    index = {str(node): i for i, node in enumerate(nodes)}
    rows, cols, terms = [], [], []
    for term in sorted(gene_sets):
        members = [index[g] for g in gene_sets[term] if g in index]
        if len(members) < min_genes:
            continue
        rows.extend(members)
        cols.extend([len(terms)] * len(members))
        terms.append(term)
    matrix = sparse.csr_array(
        (np.ones(len(rows)), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(len(nodes), len(terms)),
    )
    return matrix, terms


def biological_features(
    nodes: Sequence[Hashable],
    annotations: Mapping[str, Mapping[str, frozenset[str]]],
    n_components: int = 32,
    seed: int = 0,
) -> pd.DataFrame:
    """Dense annotation embeddings: ``n_components`` SVD components per source.

    Each protein's annotation profile is L2-normalised before the SVD, so
    heavily annotated proteins do not dominate the components. Proteins
    without annotation get zero components and ``has_<source> = 0``.
    """
    blocks = []
    for source in sorted(annotations):
        matrix, terms = annotation_matrix(nodes, annotations[source])
        has = (np.asarray(matrix.sum(axis=1)).ravel() > 0).astype(float)
        k = min(n_components, max(len(terms) - 1, 0))
        columns = {f"has_{source}": has}
        if k > 0:
            svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7, random_state=seed)
            components = svd.fit_transform(normalize(matrix, norm="l2"))
            for j in range(k):
                columns[f"{source}_svd{j + 1:02d}"] = components[:, j]
            logger.debug(
                "%s: %d terms, %d components explain %.0f%% of the variance",
                source, len(terms), k, 100 * svd.explained_variance_ratio_.sum(),
            )
        blocks.append(pd.DataFrame(columns, index=pd.Index(list(nodes), name="protein")))
    return pd.concat(blocks, axis=1)


class NodeFeatures:
    """Builds standardised node-feature matrices for any graph over a fixed node set.

    Parameters
    ----------
    nodes:
        Node order of every matrix (see :func:`canonical_nodes`).
    kind:
        ``"structural"``, ``"biological"`` or ``"both"``.
    biological:
        Output of :func:`biological_features` (required unless
        ``kind="structural"``).

    The structural block depends on the graph and is recomputed by
    :meth:`transform`; the biological block is fixed. :meth:`fit` stores the
    column means and standard deviations, which should come from the training
    graph so that evaluation graphs are scaled the same way.
    """

    def __init__(
        self,
        nodes: Sequence[Hashable],
        kind: FeatureKind = "both",
        biological: pd.DataFrame | None = None,
    ) -> None:
        if kind not in FEATURE_KINDS:
            raise ValueError(f"kind must be one of {FEATURE_KINDS}, got {kind!r}")
        if kind != "structural" and biological is None:
            raise ValueError("biological features are required for kind='biological' or 'both'")
        self.nodes = list(nodes)
        self.kind = kind
        self.biological = None
        if biological is not None and kind != "structural":
            self.biological = biological.reindex(self.nodes).fillna(0.0)

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        kind: FeatureKind = "both",
        annotations: Mapping[str, Mapping[str, frozenset[str]]] | None = None,
        n_components: int = 32,
        seed: int = 0,
    ) -> "NodeFeatures":
        """Node set of ``graph`` and, unless structural only, biological features."""
        nodes = canonical_nodes(graph)
        biological = None
        if kind != "structural":
            if annotations is None:
                raise ValueError("annotations are required for biological node features")
            biological = biological_features(nodes, annotations, n_components, seed)
        return cls(nodes, kind, biological)

    @property
    def names(self) -> list[str]:
        names = list(STRUCTURAL_FEATURES) if self.kind != "biological" else []
        if self.biological is not None:
            names += list(self.biological.columns)
        return names

    def raw(self, adjacency: sparse.csr_array) -> np.ndarray:
        """Unscaled feature matrix for the graph with this adjacency matrix."""
        if adjacency.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("adjacency matrix does not match the node set")
        blocks = []
        if self.kind != "biological":
            blocks.append(structural_features(adjacency))
        if self.biological is not None:
            blocks.append(self.biological.to_numpy(dtype=np.float64))
        return np.hstack(blocks)

    def fit(self, adjacency: sparse.csr_array) -> "NodeFeatures":
        values = self.raw(adjacency)
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, adjacency: sparse.csr_array) -> np.ndarray:
        """Standardised ``float32`` feature matrix (fit first)."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("NodeFeatures must be fitted before transform")
        return ((self.raw(adjacency) - self.mean_) / self.scale_).astype(np.float32)

    def frame(self, adjacency: sparse.csr_array) -> pd.DataFrame:
        """:meth:`transform` as a labelled DataFrame."""
        return pd.DataFrame(self.transform(adjacency), index=pd.Index(self.nodes, name="protein"), columns=self.names)
