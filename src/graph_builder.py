"""Load protein-protein interaction data and build NetworkX graphs.

Supported inputs
----------------
* **Generic edge lists** (``.tsv``, ``.csv``, ``.txt``) with two protein
  columns and an optional numeric confidence column, e.g. BioGRID or IntAct
  exports reduced to interactor pairs, or the bundled synthetic sample.
* **STRING** ``protein.links`` files (space separated, columns ``protein1``,
  ``protein2``, ``combined_score``), optionally gzip-compressed.

All loaders return a tidy DataFrame with columns ``protein_a``, ``protein_b``
and ``weight`` in which self-interactions are removed and each undirected
pair appears exactly once (STRING lists every interaction in both directions).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

EdgeFormat = Literal["auto", "edgelist", "string"]

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


def _read_table(path: Path, fmt: EdgeFormat) -> pd.DataFrame:
    if fmt == "string":
        return pd.read_csv(path, sep=r"\s+", engine="python")
    lowered = path.name.lower()
    if lowered.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    if lowered.endswith((".tsv", ".tsv.gz", ".tab", ".tab.gz")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=r"\s+", engine="python")


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


def load_edge_list(
    path: str | Path,
    fmt: EdgeFormat = "auto",
    source_col: str | None = None,
    target_col: str | None = None,
    weight_col: str | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Read a PPI edge list from disk into a standardised DataFrame.

    Parameters
    ----------
    path:
        File to read. ``.gz`` files are decompressed transparently.
    fmt:
        ``"string"`` for STRING ``protein.links`` files, ``"edgelist"`` for
        anything else, or ``"auto"`` to detect from the file name/header.
    source_col, target_col, weight_col:
        Column names. For edge lists they default to the first, second and
        (if numeric) third columns; for STRING to ``protein1``, ``protein2``
        and ``combined_score``.
    min_score:
        Optional confidence threshold (e.g. 700 for STRING "high confidence").

    Returns
    -------
    DataFrame with columns ``protein_a``, ``protein_b``, ``weight``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge list not found: {path}")

    if fmt == "auto":
        fmt = _detect_format(path)
    raw = _read_table(path, fmt)

    if fmt == "string":
        source_col = source_col or "protein1"
        target_col = target_col or "protein2"
        weight_col = weight_col or "combined_score"
    else:
        if raw.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in {path}, found {raw.shape[1]}")
        source_col = source_col or raw.columns[0]
        target_col = target_col or raw.columns[1]
        if weight_col is None and raw.shape[1] >= 3 and pd.api.types.is_numeric_dtype(raw.iloc[:, 2]):
            weight_col = raw.columns[2]

    edges = clean_edges(raw, source_col, target_col, weight_col, min_score)
    logger.info("Loaded %d unique interactions from %s (format=%s)", len(edges), path, fmt)
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
