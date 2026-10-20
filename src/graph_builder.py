"""Load protein-protein interaction data and build NetworkX graphs.

Supported inputs
----------------
* **Generic edge lists** (``.tsv``, ``.csv``, ``.txt``) with two protein
  columns and an optional numeric confidence column, e.g. BioGRID or IntAct
  exports reduced to interactor pairs, or the bundled synthetic sample.
  Files with or without a header row and with any line-ending convention
  (LF, CRLF or classic-Mac CR) are accepted.
* **STRING** ``protein.links`` files (space separated, columns ``protein1``,
  ``protein2``, ``combined_score``), optionally gzip-compressed.

All loaders return a tidy DataFrame with columns ``protein_a``, ``protein_b``
and ``weight`` in which self-interactions are removed and each undirected
pair appears exactly once (STRING lists every interaction in both directions).

Link-prediction splits
----------------------
:func:`make_link_prediction_split` hides observed interactions and pairs them
with sampled non-interacting protein pairs. Features for each split are
computed on a graph from which that split's positive edges were removed, so a
classifier never sees the interaction it is asked to predict (see the
function docstring for the exact protocol).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Literal, Sequence

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EdgeFormat = Literal["auto", "edgelist", "string"]
NegativeStrategy = Literal["uniform", "local"]
Pair = tuple[Hashable, Hashable]

STANDARD_COLUMNS = ["protein_a", "protein_b", "weight"]
_STRING_COLUMNS = ("protein1", "protein2", "combined_score")


def _detect_format(path: Path) -> EdgeFormat:
    """Guess the file format from the file name and header line."""
    name = path.name.lower()
    if "protein.links" in name:
        return "string"
    opener = _open_text(path)
    with opener as handle:
        header = handle.readline().split()
    if tuple(header[:3]) == _STRING_COLUMNS:
        return "string"
    return "edgelist"


def _open_text(path: Path):
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _first_line(path: Path) -> str:
    """First non-empty line; universal newlines also split on bare CR."""
    with _open_text(path) as handle:
        for line in handle:
            if line.strip():
                return line.strip()
    return ""


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _sniff_header(path: Path) -> bool:
    """Return ``True`` unless every field of the first line is numeric.

    Numeric identifiers (e.g. Entrez gene IDs) in the first row mean the
    file has no header; pass ``header`` explicitly for symbol-based files
    without one.
    """
    line = _first_line(path)
    fields = line.split("\t") if "\t" in line else line.replace(",", " ").split()
    return not (fields and all(_is_number(f.strip()) for f in fields))


def _read_table(path: Path, fmt: EdgeFormat, header: bool = True) -> pd.DataFrame:
    kwargs = {"header": 0 if header else None}
    if fmt == "string":
        return pd.read_csv(path, sep=r"\s+", engine="python", **kwargs)
    lowered = path.name.lower()
    if lowered.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, **kwargs)
    if lowered.endswith((".tsv", ".tsv.gz", ".tab", ".tab.gz")):
        return pd.read_csv(path, sep="\t", **kwargs)
    if "\t" in _first_line(path):
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, sep=r"\s+", engine="python", **kwargs)


def clean_edges(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    weight_col: str | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Standardise an interaction table.

    Steps: rename columns, drop missing identifiers, strip whitespace, apply
    an optional confidence threshold, remove self-interactions and collapse
    duplicate undirected pairs (keeping the highest weight).

    Parameters
    ----------
    df:
        Raw interaction table.
    source_col, target_col:
        Columns holding the two interacting proteins.
    weight_col:
        Optional numeric confidence column. Missing weights default to 1.0.
    min_score:
        Keep only interactions with ``weight >= min_score``.

    Returns
    -------
    DataFrame with columns ``protein_a``, ``protein_b``, ``weight``.
    """
    missing = [c for c in (source_col, target_col, weight_col) if c and c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in edge table: {missing}")

    edges = pd.DataFrame(
        {
            "protein_a": df[source_col],
            "protein_b": df[target_col],
            "weight": pd.to_numeric(df[weight_col], errors="coerce") if weight_col else 1.0,
        }
    )
    edges = edges.dropna(subset=["protein_a", "protein_b"])
    edges["protein_a"] = edges["protein_a"].astype(str).str.strip()
    edges["protein_b"] = edges["protein_b"].astype(str).str.strip()
    edges["weight"] = edges["weight"].fillna(1.0).astype(float)

    if min_score is not None:
        edges = edges[edges["weight"] >= min_score]

    edges = edges[edges["protein_a"] != edges["protein_b"]]

    # Canonical ordering so that (A, B) and (B, A) collapse to one row.
    swap = edges["protein_a"] > edges["protein_b"]
    edges.loc[swap, ["protein_a", "protein_b"]] = edges.loc[swap, ["protein_b", "protein_a"]].values
    edges = (
        edges.groupby(["protein_a", "protein_b"], as_index=False, sort=True)["weight"]
        .max()
        .reset_index(drop=True)
    )
    return edges[STANDARD_COLUMNS]


def read_edge_table(
    path: str | Path,
    fmt: EdgeFormat = "auto",
    header: bool | None = None,
    source_col: str | int | None = None,
    target_col: str | int | None = None,
    weight_col: str | int | None = None,
) -> pd.DataFrame:
    """Read an interaction file *without* cleaning it.

    Returns the raw rows - including self-interactions, duplicates and
    missing values - with standardised columns ``protein_a``, ``protein_b``
    and ``weight`` (1.0 when the file has no confidence column). Use
    :func:`load_edge_list` for a cleaned table, or
    :mod:`src.preprocessing` for an audited cleaning pipeline.

    Parameters
    ----------
    path:
        File to read. ``.gz`` files are decompressed transparently.
    fmt:
        ``"string"`` for STRING ``protein.links`` files, ``"edgelist"`` for
        anything else, or ``"auto"`` to detect from the file name/header.
    header:
        Whether the first line is a header. ``None`` detects it: a first line
        made only of numbers (e.g. Entrez IDs) is treated as data.
    source_col, target_col, weight_col:
        Column names (or positions for header-less files). For edge lists
        they default to the first, second and (if numeric) third columns;
        for STRING to ``protein1``, ``protein2`` and ``combined_score``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")

    if fmt == "auto":
        fmt = _detect_format(path)
    if header is None:
        header = True if fmt == "string" else _sniff_header(path)
    raw = _read_table(path, fmt, header=header)

    if fmt == "string":
        source_col = source_col or "protein1"
        target_col = target_col or "protein2"
        weight_col = weight_col or "combined_score"
    else:
        if raw.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in {path}, found {raw.shape[1]}")
        source_col = raw.columns[0] if source_col is None else source_col
        target_col = raw.columns[1] if target_col is None else target_col
        if weight_col is None and raw.shape[1] >= 3 and pd.api.types.is_numeric_dtype(raw.iloc[:, 2]):
            weight_col = raw.columns[2]

    missing = [c for c in (source_col, target_col, weight_col) if c is not None and c not in raw.columns]
    if missing:
        raise KeyError(f"Columns not found in {path}: {missing}")
    table = pd.DataFrame(
        {
            "protein_a": raw[source_col],
            "protein_b": raw[target_col],
            "weight": pd.to_numeric(raw[weight_col], errors="coerce") if weight_col is not None else 1.0,
        }
    )
    logger.debug("Read %d raw rows from %s (format=%s, header=%s)", len(table), path, fmt, header)
    return table


def load_edge_list(
    path: str | Path,
    fmt: EdgeFormat = "auto",
    source_col: str | int | None = None,
    target_col: str | int | None = None,
    weight_col: str | int | None = None,
    min_score: float | None = None,
    header: bool | None = None,
) -> pd.DataFrame:
    """Read a PPI edge list from disk into a cleaned, standardised DataFrame.

    Equivalent to :func:`read_edge_table` followed by :func:`clean_edges`;
    see those functions for the parameters. ``min_score`` is an optional
    confidence threshold (e.g. 700 for STRING "high confidence").

    Returns
    -------
    DataFrame with columns ``protein_a``, ``protein_b``, ``weight``.
    """
    raw = read_edge_table(path, fmt, header, source_col, target_col, weight_col)
    edges = clean_edges(raw, "protein_a", "protein_b", "weight", min_score)
    logger.info("Loaded %d unique interactions from %s", len(edges), path)
    return edges


def build_graph(edges: pd.DataFrame, largest_component: bool = True) -> nx.Graph:
    """Build an undirected, weighted PPI graph from a standardised edge table.

    Parameters
    ----------
    edges:
        Output of :func:`load_edge_list` / :func:`clean_edges`.
    largest_component:
        Restrict the graph to its largest connected component. Most
        topological features are only comparable within one component.
    """
    graph = nx.from_pandas_edgelist(edges, "protein_a", "protein_b", edge_attr="weight")
    if largest_component and graph.number_of_nodes() > 0:
        giant = max(nx.connected_components(graph), key=len)
        if len(giant) < graph.number_of_nodes():
            logger.info(
                "Keeping largest connected component: %d of %d proteins",
                len(giant),
                graph.number_of_nodes(),
            )
        graph = graph.subgraph(giant).copy()
    return graph


def load_graph(
    path: str | Path,
    fmt: EdgeFormat = "auto",
    min_score: float | None = None,
    largest_component: bool = True,
    **kwargs,
) -> nx.Graph:
    """Convenience wrapper: :func:`load_edge_list` followed by :func:`build_graph`."""
    edges = load_edge_list(path, fmt=fmt, min_score=min_score, **kwargs)
    return build_graph(edges, largest_component=largest_component)


def graph_summary(graph: nx.Graph) -> dict[str, float]:
    """Global descriptive statistics of a PPI network."""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    components = list(nx.connected_components(graph))
    summary: dict[str, float] = {
        "proteins": n,
        "interactions": m,
        "density": nx.density(graph) if n > 1 else 0.0,
        "mean_degree": (2 * m / n) if n else 0.0,
        "max_degree": max(degrees, default=0),
        "average_clustering": nx.average_clustering(graph) if n else 0.0,
        "transitivity": nx.transitivity(graph) if n else 0.0,
        "connected_components": len(components),
        "largest_component_fraction": (max(map(len, components)) / n) if n else 0.0,
    }
    if m > 0 and len(set(degrees)) > 1:
        summary["degree_assortativity"] = nx.degree_assortativity_coefficient(graph)
    return summary


# ---------------------------------------------------------------------------
# Link-prediction splits
# ---------------------------------------------------------------------------


def _canonical(u: Hashable, v: Hashable) -> Pair:
    """Order a node pair deterministically, independent of label type."""
    return (u, v) if str(u) <= str(v) else (v, u)


def _pair_key(pair: Pair) -> tuple[str, str]:
    return str(pair[0]), str(pair[1])


@dataclass
class LinkPredictionSplit:
    """Train/test protein pairs with labels and their feature graphs.

    Attributes
    ----------
    train_graph:
        Observed network minus test *and* training positives. Features for
        ``train_pairs`` must be computed on this graph.
    test_graph:
        Observed network minus test positives. Features for ``test_pairs``
        must be computed on this graph.
    train_pairs, test_pairs:
        Protein pairs; label 1 = hidden interaction, 0 = sampled non-edge.
    train_labels, test_labels:
        Binary labels aligned with the pairs.
    """

    train_graph: nx.Graph
    test_graph: nx.Graph
    train_pairs: list[Pair]
    train_labels: np.ndarray
    test_pairs: list[Pair]
    test_labels: np.ndarray

    def summary(self) -> dict[str, int]:
        return {
            "train_graph_edges": self.train_graph.number_of_edges(),
            "test_graph_edges": self.test_graph.number_of_edges(),
            "train_positives": int(self.train_labels.sum()),
            "train_negatives": int((self.train_labels == 0).sum()),
            "test_positives": int(self.test_labels.sum()),
            "test_negatives": int((self.test_labels == 0).sum()),
        }


def _random_spanning_forest(graph: nx.Graph, rng: np.random.Generator) -> set[frozenset]:
    """Edges of a uniformly-weighted random spanning forest of ``graph``."""
    edges = sorted((_canonical(u, v) for u, v in graph.edges()), key=_pair_key)
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes())
    for (u, v), w in zip(edges, rng.random(len(edges))):
        weighted.add_edge(u, v, w=w)
    return {frozenset(e) for e in nx.minimum_spanning_edges(weighted, weight="w", data=False)}


def sample_holdout_edges(
    graph: nx.Graph,
    n_edges: int,
    rng: np.random.Generator,
    keep_connected: bool = True,
) -> list[Pair]:
    """Sample ``n_edges`` edges to hide from ``graph``.

    With ``keep_connected=True`` the edges of a random spanning forest are
    protected, so removing the sample never disconnects a component or
    isolates a protein (isolated proteins would have undefined features).
    """
    edges = sorted((_canonical(u, v) for u, v in graph.edges()), key=_pair_key)
    if keep_connected:
        protected = _random_spanning_forest(graph, rng)
        candidates = [e for e in edges if frozenset(e) not in protected]
    else:
        candidates = edges
    if n_edges > len(candidates):
        raise ValueError(
            f"Cannot hide {n_edges} edges: only {len(candidates)} are removable "
            "without disconnecting the network. Lower the hold-out fractions."
        )
    chosen = rng.choice(len(candidates), size=n_edges, replace=False)
    return [candidates[i] for i in sorted(chosen)]


def sample_negative_pairs(
    graph: nx.Graph,
    n_pairs: int,
    rng: np.random.Generator,
    strategy: NegativeStrategy = "uniform",
    walk_graph: nx.Graph | None = None,
    forbidden: Sequence[Pair] = (),
    max_attempts_per_pair: int = 200,
) -> list[Pair]:
    """Sample protein pairs that do **not** interact in ``graph``.

    Parameters
    ----------
    graph:
        The full observed network; returned pairs are never edges of it.
    n_pairs:
        Number of negative pairs to draw (without replacement).
    strategy:
        ``"uniform"`` draws pairs uniformly at random - the standard but
        optimistic protocol, as most random pairs are far apart.
        ``"local"`` draws pairs two or three steps apart via short random
        walks on ``walk_graph``; these hard negatives resemble true
        interactions topologically and give a more conservative estimate.
    walk_graph:
        Graph used for the random walks of the ``"local"`` strategy
        (defaults to ``graph``). Pass the split's feature graph.
    forbidden:
        Pairs that must not be returned (e.g. negatives of another split).
    """
    if strategy not in ("uniform", "local"):
        raise ValueError(f"Unknown negative sampling strategy: {strategy!r}")
    walk_graph = graph if walk_graph is None else walk_graph
    nodes = sorted(graph.nodes(), key=str)
    if len(nodes) < 2:
        raise ValueError("Need at least two proteins to sample negative pairs")

    excluded = {frozenset(p) for p in forbidden}
    chosen: list[Pair] = []
    seen: set[frozenset] = set()
    attempts, max_attempts = 0, max(1, n_pairs) * max_attempts_per_pair

    while len(chosen) < n_pairs:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Sampled only {len(chosen)}/{n_pairs} negative pairs after "
                f"{max_attempts} attempts; the network may be too dense."
            )
        u = nodes[int(rng.integers(len(nodes)))]
        if strategy == "uniform":
            v = nodes[int(rng.integers(len(nodes)))]
        else:
            v = u
            for _ in range(int(rng.integers(2, 4))):
                neighbours = sorted(walk_graph.neighbors(v), key=str)
                if not neighbours:
                    break
                v = neighbours[int(rng.integers(len(neighbours)))]
        if u == v or graph.has_edge(u, v):
            continue
        key = frozenset((u, v))
        if key in seen or key in excluded:
            continue
        seen.add(key)
        chosen.append(_canonical(u, v))
    return chosen


def _shuffled(pairs: list[Pair], labels: np.ndarray, rng: np.random.Generator):
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order], labels[order]


def make_link_prediction_split(
    graph: nx.Graph,
    test_fraction: float = 0.1,
    train_fraction: float = 0.1,
    negative_ratio: float = 1.0,
    negative_strategy: NegativeStrategy = "uniform",
    seed: int = 42,
) -> LinkPredictionSplit:
    """Create a leakage-free link-prediction split of a PPI network.

    Protocol
    --------
    1. Hide ``test_fraction`` of the edges -> test positives.
       ``test_graph = G - test positives``.
    2. Hide a further ``train_fraction`` of the original edges from
       ``test_graph`` -> training positives.
       ``train_graph = test_graph - training positives``.
    3. Sample ``negative_ratio`` x as many non-interacting pairs for each
       split (never edges of ``G``; train and test negatives are disjoint).

    A classifier is trained on features computed from ``train_graph`` and
    evaluated on features computed from ``test_graph``. Neither split ever
    sees its own positive edges, which mirrors the real task of predicting
    interactions that have not been observed yet. Spanning-forest edges are
    never hidden, so no protein becomes isolated.

    Parameters
    ----------
    graph:
        Observed PPI network.
    test_fraction, train_fraction:
        Fractions of the original edges hidden as test / training positives.
    negative_ratio:
        Number of negative pairs per positive pair.
    negative_strategy:
        ``"uniform"`` or ``"local"``; see :func:`sample_negative_pairs`.
    seed:
        Random seed controlling every sampling step.
    """
    if not (0 < test_fraction < 1 and 0 < train_fraction < 1):
        raise ValueError("test_fraction and train_fraction must be in (0, 1)")
    if test_fraction + train_fraction >= 1:
        raise ValueError("test_fraction + train_fraction must be < 1")
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")

    rng = np.random.default_rng(seed)
    n_edges = graph.number_of_edges()
    n_test = max(1, int(round(test_fraction * n_edges)))
    n_train = max(1, int(round(train_fraction * n_edges)))

    test_pos = sample_holdout_edges(graph, n_test, rng)
    test_graph = graph.copy()
    test_graph.remove_edges_from(test_pos)

    train_pos = sample_holdout_edges(test_graph, n_train, rng)
    train_graph = test_graph.copy()
    train_graph.remove_edges_from(train_pos)

    test_neg = sample_negative_pairs(
        graph,
        int(round(negative_ratio * n_test)),
        rng,
        strategy=negative_strategy,
        walk_graph=test_graph,
    )
    train_neg = sample_negative_pairs(
        graph,
        int(round(negative_ratio * n_train)),
        rng,
        strategy=negative_strategy,
        walk_graph=train_graph,
        forbidden=test_neg,
    )

    train_pairs, train_labels = _shuffled(
        train_pos + train_neg,
        np.r_[np.ones(len(train_pos), dtype=int), np.zeros(len(train_neg), dtype=int)],
        rng,
    )
    test_pairs, test_labels = _shuffled(
        test_pos + test_neg,
        np.r_[np.ones(len(test_pos), dtype=int), np.zeros(len(test_neg), dtype=int)],
        rng,
    )
    split = LinkPredictionSplit(
        train_graph=train_graph,
        test_graph=test_graph,
        train_pairs=train_pairs,
        train_labels=train_labels,
        test_pairs=test_pairs,
        test_labels=test_labels,
    )
    logger.info("Link-prediction split: %s", split.summary())
    return split
