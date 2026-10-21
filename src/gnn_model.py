"""Graph neural networks for protein-protein interaction prediction.

The classical pipeline (:mod:`src.model`) describes a protein pair with
hand-crafted topological scores. A graph neural network instead *learns* a
representation of every protein by repeatedly aggregating information from
its interaction partners (message passing), starting from node features
(:mod:`src.node_features`). A decoder then scores protein pairs from their
two embeddings.

Architecture
------------
**Encoder** - two message-passing layers, so a protein's embedding depends on
its partners and its partners' partners (the two-step neighbourhood that the
L3 principle shows to be informative for PPIs):

* ``graphsage`` (Hamilton et al., NeurIPS 2017) - mean aggregation of the
  neighbours, combined with the protein's own state by a separate weight
  matrix: ``h_v' = W1 h_v + W2 mean_{u in N(v)} h_u``. The separate self term
  keeps hubs from washing out their own identity.
* ``gcn`` (Kipf & Welling, ICLR 2017) - symmetric degree-normalised averaging
  over the neighbours *and* the protein itself:
  ``h_v' = W sum_{u in N(v) + v} h_u / sqrt(d_u d_v)``.
* ``mlp`` - the same two layers without message passing. A control that shows
  how much the network adds beyond the node features alone.

**Decoder** - a symmetric MLP on ``[h_u * h_v, |h_u - h_v|]``. A plain dot
product only rewards *similar* embeddings, which is how triadic closure
(common neighbours) thinks. Interacting proteins are often complementary
rather than similar (Kovacs et al., 2019), and the element-wise features let
the MLP learn either pattern.

**Pair heuristics (optional hybrid)** - with ``pair_heuristics=True`` the
decoder also receives three pair-level path counts, computed on the same
message-passing graph: L3, common neighbours and resource allocation. A GNN
that embeds proteins one at a time cannot count the paths *between* two
specific proteins (two proteins with identical neighbourhoods get identical
embeddings however they are connected; Zhang et al., NeurIPS 2021), and that
count is exactly what separates true interactions from hard negatives a few
steps apart. The hybrid keeps the learned embeddings and adds the missing
pair-level signal.

Training protocol (no leakage)
------------------------------
The GNN is trained and evaluated on the same
:class:`~src.graph_builder.LinkPredictionSplit` as the classical models:

* messages pass over ``train_graph`` during training and over ``test_graph``
  at evaluation, so a protein pair is never scored on a graph that contains
  its own interaction;
* each epoch hides a random ``supervision_fraction`` of the ``train_graph``
  interactions from message passing and uses them as extra positives, next to
  the split's training pairs. Structural node features are recomputed on that
  reduced graph, so a hidden interaction is not visible in its proteins'
  degree or clustering either;
* negatives are re-sampled every epoch (uniform, or half uniform and half
  local hard negatives when ``negative_strategy="local"``), avoiding every
  interaction known at training time;
* a stratified part of the split's training pairs is held out for early
  stopping on average precision; test pairs are never looked at.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Hashable, Literal, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from src.feature_extractor import pairwise_heuristics
from src.graph_builder import LinkPredictionSplit
from src.node_features import FeatureKind, NodeFeatures, adjacency_matrix

logger = logging.getLogger(__name__)

Architecture = Literal["graphsage", "gcn", "mlp"]
GNN_ARCHITECTURES: tuple[str, ...] = ("graphsage", "gcn", "mlp")
DISPLAY_NAMES = {"graphsage": "GraphSAGE", "gcn": "GCN", "mlp": "MLP (no message passing)"}
#: Pair-level scores given to the decoder when ``pair_heuristics=True``.
PAIR_HEURISTICS: tuple[str, ...] = ("l3", "common_neighbors", "resource_allocation")


@dataclass
class GNNConfig:
    """Hyper-parameters of a GNN link predictor.

    The defaults were chosen on the validation pairs of HI-III (never on test
    pairs) and work for all interactomes in the registry.
    """

    architecture: Architecture = "graphsage"
    features: FeatureKind = "both"
    hidden_channels: int = 128
    embedding_dim: int = 64
    decoder_hidden: int = 64
    dropout: float = 0.3
    learning_rate: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 50
    eval_every: int = 5
    supervision_fraction: float = 0.2
    negative_strategy: Literal["uniform", "local"] = "uniform"
    validation_fraction: float = 0.2
    n_components: int = 32
    pair_heuristics: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.architecture not in GNN_ARCHITECTURES:
            raise ValueError(f"architecture must be one of {GNN_ARCHITECTURES}, got {self.architecture!r}")
        if self.negative_strategy not in ("uniform", "local"):
            raise ValueError(f"Unknown negative strategy {self.negative_strategy!r}")
        if not 0 < self.supervision_fraction < 1:
            raise ValueError("supervision_fraction must be in (0, 1)")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Two-layer GraphSAGE / GCN (or MLP) mapping node features to embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        architecture: Architecture = "graphsage",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.dropout = dropout
        if architecture == "graphsage":
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        elif architecture == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif architecture == "mlp":
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, out_channels)
        else:
            raise ValueError(f"Unknown architecture {architecture!r}")

    def _layer(self, layer: nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return layer(x) if self.architecture == "mlp" else layer(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self._layer(self.conv1, x, edge_index).relu()
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return self._layer(self.conv2, h, edge_index)


class PairDecoder(nn.Module):
    """Symmetric MLP scoring a protein pair from its two embeddings."""

    def __init__(self, embedding_dim: int, hidden: int = 64, dropout: float = 0.3, n_pair_features: int = 0) -> None:
        super().__init__()
        self.n_pair_features = n_pair_features
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim + n_pair_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a * b, (a - b).abs()], dim=-1)

    def forward(self, h: torch.Tensor, pairs: torch.Tensor, pair_x: torch.Tensor | None = None) -> torch.Tensor:
        """Interaction logits for ``pairs`` (a ``2 x P`` index tensor).

        ``pair_x`` holds the ``P x n_pair_features`` pair heuristics of the
        hybrid model and is ignored otherwise.
        """
        z = self.pair_features(h[pairs[0]], h[pairs[1]])
        if self.n_pair_features:
            if pair_x is None:
                raise ValueError("this decoder needs pair heuristics")
            z = torch.cat([z, pair_x], dim=-1)
        return self.mlp(z).squeeze(-1)


class LinkPredictor(nn.Module):
    """Encoder + pair decoder."""

    def __init__(self, in_channels: int, config: GNNConfig) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels, config.hidden_channels, config.embedding_dim, config.architecture, config.dropout
        )
        self.decoder = PairDecoder(
            config.embedding_dim,
            config.decoder_hidden,
            config.dropout,
            len(PAIR_HEURISTICS) if config.pair_heuristics else 0,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor, pair_x: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.decoder(self.encoder(x, edge_index), pairs, pair_x)


# ---------------------------------------------------------------------------
# Graph and pair utilities
# ---------------------------------------------------------------------------


def edge_index_from_adjacency(adjacency: sparse.csr_array) -> torch.Tensor:
    """``2 x 2m`` edge index (both directions) of a symmetric adjacency matrix."""
    coo = sparse.coo_array(adjacency)
    return torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))


def upper_edges(adjacency: sparse.csr_array) -> np.ndarray:
    """``2 x m`` array with each undirected edge once (row < col)."""
    coo = sparse.triu(adjacency, k=1, format="coo")
    return np.vstack([coo.row, coo.col]).astype(np.int64)


def adjacency_from_edges(edges: np.ndarray, n: int) -> sparse.csr_array:
    """Symmetric binary adjacency matrix from a ``2 x m`` edge array."""
    rows = np.concatenate([edges[0], edges[1]])
    cols = np.concatenate([edges[1], edges[0]])
    return sparse.csr_array((np.ones(rows.size), (rows, cols)), shape=(n, n))


def pair_keys(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    """Unique integer key of each unordered pair."""
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    return lo.astype(np.int64) * n + hi


def pairs_to_index(pairs: Sequence[tuple[Hashable, Hashable]], index: dict) -> np.ndarray:
    """``2 x P`` integer array for protein pairs."""
    try:
        return np.array([[index[u] for u, _ in pairs], [index[v] for _, v in pairs]], dtype=np.int64).reshape(2, -1)
    except KeyError as exc:
        raise KeyError(f"Protein {exc.args[0]!r} is not in the model's node set") from None


def pair_heuristic_features(adjacency: sparse.csr_array, pairs: np.ndarray) -> np.ndarray:
    """``log1p`` of the :data:`PAIR_HEURISTICS` for a ``2 x P`` pair array."""
    if pairs.shape[1] == 0:
        return np.zeros((0, len(PAIR_HEURISTICS)))
    scores = pairwise_heuristics(adjacency, pairs[0], pairs[1])
    return np.log1p(np.column_stack([scores[name] for name in PAIR_HEURISTICS]))


def block_pair_heuristics(adjacency: sparse.csr_array, rows: np.ndarray) -> np.ndarray:
    """Pair heuristics of ``rows`` against every node: a ``len(rows) x n x 3`` array."""
    adjacency = sparse.csr_array(adjacency, dtype=float)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = sparse.diags_array(np.where(degree > 0, 1.0 / np.maximum(degree, 1), 0.0))
    inv_sqrt = sparse.diags_array(np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1)), 0.0))
    block = adjacency[rows]
    common = (block @ adjacency).toarray()
    resource = (block @ inv_degree @ adjacency).toarray()
    l3 = (block @ (inv_sqrt @ adjacency @ inv_sqrt) @ adjacency).toarray()
    return np.log1p(np.stack([l3, common, resource], axis=-1))


def sample_uniform_negatives(
    n_nodes: int, n_pairs: int, known: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """``n_pairs`` distinct random pairs (``2 x n``) whose keys are not in sorted ``known``."""
    chosen = np.empty(0, dtype=np.int64)
    while chosen.size < n_pairs:
        size = int(1.5 * (n_pairs - chosen.size)) + 16
        u, v = rng.integers(n_nodes, size=size), rng.integers(n_nodes, size=size)
        keys = pair_keys(u[u != v], v[u != v], n_nodes)
        keys = keys[~np.isin(keys, known, assume_unique=False)]
        chosen = np.unique(np.concatenate([chosen, keys]))
    chosen = rng.permutation(chosen)[:n_pairs]
    return np.vstack([chosen // n_nodes, chosen % n_nodes])


def sample_local_negatives(
    adjacency: sparse.csr_array, n_pairs: int, known: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Hard negatives: endpoints of random walks of length 2 or 3 that do not interact."""
    n = adjacency.shape[0]
    indptr, indices = adjacency.indptr, adjacency.indices
    degree = np.diff(indptr)
    chosen = np.empty(0, dtype=np.int64)
    for _ in range(50):
        if chosen.size >= n_pairs:
            break
        size = 2 * (n_pairs - chosen.size) + 16
        start = rng.choice(np.flatnonzero(degree > 0), size=size)
        steps = rng.integers(2, 4, size=size)
        current = start.copy()
        for step in range(3):
            walking = steps > step
            offsets = (rng.random(walking.sum()) * degree[current[walking]]).astype(np.int64)
            current[walking] = indices[indptr[current[walking]] + offsets]
        keep = start != current
        keys = pair_keys(start[keep], current[keep], n)
        keys = keys[~np.isin(keys, known)]
        chosen = np.unique(np.concatenate([chosen, keys]))
    chosen = rng.permutation(chosen)[:n_pairs]
    return np.vstack([chosen // n, chosen % n])


# ---------------------------------------------------------------------------
# Trained model
# ---------------------------------------------------------------------------


@dataclass
class TrainedGNN:
    """A fitted GNN with its node features, training curve and validation score."""

    model: LinkPredictor
    config: GNNConfig
    features: NodeFeatures
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_epoch: int = 0
    validation_average_precision: float = float("nan")
    training_seconds: float = 0.0
    pair_mean: np.ndarray | None = None
    pair_scale: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.config.architecture

    @property
    def nodes(self) -> list[Hashable]:
        return self.features.nodes

    @property
    def index(self) -> dict[Hashable, int]:
        return {node: i for i, node in enumerate(self.nodes)}

    def _inputs(self, graph: nx.Graph) -> tuple[torch.Tensor, torch.Tensor]:
        missing = set(graph.nodes()) ^ set(self.nodes)
        if missing:
            raise ValueError(f"graph and model node sets differ ({len(missing)} proteins)")
        adjacency = adjacency_matrix(graph, self.nodes)
        x = torch.from_numpy(self.features.transform(adjacency))
        return x, edge_index_from_adjacency(adjacency)

    def pair_inputs(self, adjacency: sparse.csr_array, pairs: np.ndarray) -> torch.Tensor | None:
        """Standardised pair heuristics (``None`` unless the model is a hybrid)."""
        if not self.config.pair_heuristics:
            return None
        values = (pair_heuristic_features(adjacency, pairs) - self.pair_mean) / self.pair_scale
        return torch.from_numpy(values.astype(np.float32))

    @torch.no_grad()
    def embed(self, graph: nx.Graph) -> pd.DataFrame:
        """Protein embeddings (one row per protein) with messages passed over ``graph``."""
        self.model.eval()
        h = self.model.encoder(*self._inputs(graph))
        columns = [f"dim_{i + 1:02d}" for i in range(h.shape[1])]
        return pd.DataFrame(h.numpy(), index=pd.Index(self.nodes, name="protein"), columns=columns)

    @torch.no_grad()
    def predict_proba(
        self, graph: nx.Graph, pairs: Sequence[tuple[Hashable, Hashable]], batch_size: int = 65536
    ) -> np.ndarray:
        """Interaction probability of each pair, with messages passed over ``graph``.

        Pass the graph *without* the pairs' own interactions (for evaluation,
        ``split.test_graph``).
        """
        self.model.eval()
        h = self.model.encoder(*self._inputs(graph))
        index = pairs_to_index(list(pairs), self.index)
        adjacency = adjacency_matrix(graph, self.nodes) if self.config.pair_heuristics else None
        logits = []
        for i in range(0, index.shape[1], batch_size):
            chunk = index[:, i : i + batch_size]
            logits.append(self.model.decoder(h, torch.from_numpy(chunk), self.pair_inputs(adjacency, chunk)))
        return torch.sigmoid(torch.cat(logits)).numpy() if logits else np.empty(0)

    def save(self, path: str | Path) -> Path:
        """Self-contained checkpoint: weights, config, node set and feature scaling."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        biological = self.features.biological
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.model.state_dict(),
                "nodes": [str(n) for n in self.nodes],
                "feature_kind": self.features.kind,
                "biological": None if biological is None else torch.from_numpy(biological.to_numpy(np.float32)),
                "biological_columns": [] if biological is None else list(biological.columns),
                "feature_mean": torch.from_numpy(self.features.mean_),
                "feature_scale": torch.from_numpy(self.features.scale_),
                "history": self.history.to_dict(orient="list"),
                "best_epoch": self.best_epoch,
                "validation_average_precision": self.validation_average_precision,
                "training_seconds": self.training_seconds,
                "pair_mean": None if self.pair_mean is None else torch.from_numpy(self.pair_mean),
                "pair_scale": None if self.pair_scale is None else torch.from_numpy(self.pair_scale),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainedGNN":
        """Load a checkpoint written by :meth:`save`."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = GNNConfig(**state["config"])
        nodes = state["nodes"]
        biological = None
        if state["biological"] is not None:
            biological = pd.DataFrame(
                state["biological"].numpy().astype(np.float64), index=nodes, columns=state["biological_columns"]
            )
        features = NodeFeatures(nodes, state["feature_kind"], biological)
        features.mean_ = state["feature_mean"].numpy()
        features.scale_ = state["feature_scale"].numpy()
        model = LinkPredictor(len(features.mean_), config)
        model.load_state_dict(state["state_dict"])
        model.eval()
        return cls(
            model,
            config,
            features,
            pd.DataFrame(state["history"]),
            state["best_epoch"],
            state["validation_average_precision"],
            state["training_seconds"],
            None if state.get("pair_mean") is None else state["pair_mean"].numpy(),
            None if state.get("pair_scale") is None else state["pair_scale"].numpy(),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _stratified_holdout(labels: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask selecting ``fraction`` of each class."""
    mask = np.zeros(labels.size, dtype=bool)
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        n = max(1, int(round(fraction * members.size)))
        mask[rng.choice(members, size=n, replace=False)] = True
    return mask


def train_link_predictor(
    split: LinkPredictionSplit,
    features: NodeFeatures,
    config: GNNConfig | None = None,
) -> TrainedGNN:
    """Train a GNN link predictor on a leakage-free split (see the module docstring).

    Parameters
    ----------
    split:
        Output of :func:`src.graph_builder.make_link_prediction_split`.
    features:
        Node features over the split's node set (fitted here on
        ``split.train_graph``).
    config:
        Hyper-parameters; defaults to :class:`GNNConfig`.
    """
    config = config or GNNConfig()
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    nodes = features.nodes
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    train_adjacency = adjacency_matrix(split.train_graph, nodes)
    known_adjacency = adjacency_matrix(split.test_graph, nodes)  # every interaction known at training time
    known = np.sort(pair_keys(*upper_edges(known_adjacency), n))
    features.fit(train_adjacency)

    pairs = pairs_to_index(split.train_pairs, index)
    labels = np.asarray(split.train_labels, dtype=np.float32)
    validation = _stratified_holdout(labels, config.validation_fraction, rng)
    fit_pos = pairs[:, ~validation & (labels == 1)]
    fit_neg = pairs[:, ~validation & (labels == 0)]
    val_pairs = torch.from_numpy(pairs[:, validation])
    val_labels = labels[validation]

    message_edges = upper_edges(train_adjacency)
    x_train = torch.from_numpy(features.transform(train_adjacency))
    edge_index_train = edge_index_from_adjacency(train_adjacency)

    pair_mean = pair_scale = None
    if config.pair_heuristics:
        reference = pair_heuristic_features(train_adjacency, pairs)
        pair_mean, std = reference.mean(axis=0), reference.std(axis=0)
        pair_scale = np.where(std > 0, std, 1.0)

    def pair_inputs(adjacency: sparse.csr_array, batch: np.ndarray) -> torch.Tensor | None:
        if not config.pair_heuristics:
            return None
        values = (pair_heuristic_features(adjacency, batch) - pair_mean) / pair_scale
        return torch.from_numpy(values.astype(np.float32))

    val_pair_x = pair_inputs(train_adjacency, pairs[:, validation])

    model = LinkPredictor(x_train.shape[1], config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best_state, best_ap, best_epoch, waited = None, -np.inf, 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(message_edges.shape[1])
        n_supervision = int(config.supervision_fraction * order.size)
        supervision = message_edges[:, order[:n_supervision]]
        messages = adjacency_from_edges(message_edges[:, order[n_supervision:]], n)
        x = torch.from_numpy(features.transform(messages))
        edge_index = edge_index_from_adjacency(messages)

        positives = np.hstack([supervision, fit_pos])
        n_new = max(positives.shape[1] - fit_neg.shape[1], 0)
        if config.negative_strategy == "local":
            half = n_new // 2
            sampled = np.hstack(
                [
                    sample_local_negatives(messages, half, known, rng),
                    sample_uniform_negatives(n, n_new - half, known, rng),
                ]
            )
        else:
            sampled = sample_uniform_negatives(n, n_new, known, rng)
        batch = np.hstack([positives, fit_neg, sampled])
        target = torch.cat([torch.ones(positives.shape[1]), torch.zeros(batch.shape[1] - positives.shape[1])])

        optimizer.zero_grad()
        loss = loss_fn(model(x, edge_index, torch.from_numpy(batch), pair_inputs(messages, batch)), target)
        loss.backward()
        optimizer.step()

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(x_train, edge_index_train, val_pairs, val_pair_x)).numpy()
            val_ap = float(average_precision_score(val_labels, scores))
            val_auc = float(roc_auc_score(val_labels, scores))
            history.append({"epoch": epoch, "loss": loss.item(), "val_average_precision": val_ap, "val_roc_auc": val_auc})
            if val_ap > best_ap + 1e-4:
                best_ap, best_epoch, waited = val_ap, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += config.eval_every
                if waited >= config.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    seconds = time.perf_counter() - start
    logger.info(
        "Trained %s (%s features): best validation AP %.3f at epoch %d (%.0fs)",
        DISPLAY_NAMES[config.architecture], features.kind, best_ap, best_epoch, seconds,
    )
    return TrainedGNN(
        model, config, features, pd.DataFrame(history), best_epoch, best_ap, seconds, pair_mean, pair_scale
    )


# ---------------------------------------------------------------------------
# Novel interactions
# ---------------------------------------------------------------------------


@torch.no_grad()
def rank_novel_interactions(
    trained: TrainedGNN,
    graph: nx.Graph,
    top_k: int = 100,
    block_size: int = 48,
    symbols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Score *every* unobserved protein pair on ``graph`` and return the ``top_k``.

    Unlike the classical models, which need candidate pre-selection, the GNN
    scores all ``n(n-1)/2`` pairs: embeddings are computed once and the decoder
    runs over blocks of ``block_size`` proteins at a time.
    """
    trained.model.eval()
    nodes = trained.nodes
    n = len(nodes)
    h = trained.model.encoder(*trained._inputs(graph))
    adjacency = adjacency_matrix(graph, nodes)
    decoder = trained.model.decoder
    best_scores = torch.empty(0)
    best_keys = torch.empty(0, dtype=torch.int64)
    columns = torch.arange(n)
    for start in range(0, n - 1, block_size):
        rows = torch.arange(start, min(start + block_size, n))
        a = h[rows].unsqueeze(1).expand(-1, n, -1)
        b = h.unsqueeze(0).expand(rows.numel(), -1, -1)
        z = decoder.pair_features(a, b)
        if trained.config.pair_heuristics:
            heuristics = (block_pair_heuristics(adjacency, rows.numpy()) - trained.pair_mean) / trained.pair_scale
            z = torch.cat([z, torch.from_numpy(heuristics.astype(np.float32))], dim=-1)
        logits = decoder.mlp(z).squeeze(-1)
        invalid = columns.unsqueeze(0) <= rows.unsqueeze(1)  # each pair once, no self-pairs
        observed = torch.from_numpy(adjacency[start : start + rows.numel()].toarray() > 0)
        logits = logits.masked_fill(invalid | observed, -torch.inf)
        k = min(top_k, logits.numel())
        scores, flat = torch.topk(logits.flatten(), k)
        keys = (rows[flat // n]) * n + columns[flat % n]
        best_scores = torch.cat([best_scores, scores])
        best_keys = torch.cat([best_keys, keys])
        if best_scores.numel() > top_k:
            best_scores, keep = torch.topk(best_scores, top_k)
            best_keys = best_keys[keep]
    order = torch.argsort(best_scores, descending=True, stable=True)
    best_scores, best_keys = best_scores[order], best_keys[order]
    best_scores, best_keys = best_scores[torch.isfinite(best_scores)], best_keys[torch.isfinite(best_scores)]
    table = pd.DataFrame(
        {
            "protein_a": [nodes[int(k) // n] for k in best_keys],
            "protein_b": [nodes[int(k) % n] for k in best_keys],
            "probability": torch.sigmoid(best_scores).numpy(),
        }
    )
    if symbols:
        table.insert(2, "symbol_a", table["protein_a"].map(symbols))
        table.insert(3, "symbol_b", table["protein_b"].map(symbols))
    return table
