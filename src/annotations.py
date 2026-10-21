"""Functional annotation of proteins: Gene Ontology, pathways and tissue expression.

The annotations come from Hetionet v1.0 (Himmelstein et al., 2017), which
integrates the sources below under Entrez gene IDs, the identifiers of the
interactome files:

=========  ==========  ====================================================
source     metaedge    meaning
=========  ==========  ====================================================
go_bp      ``GpBP``    Gene Ontology biological process
go_cc      ``GpCC``    Gene Ontology cellular component
pathway    ``GpPW``    Reactome, WikiPathways and Pathway Interaction DB
tissue     ``AeG``     anatomical structure expresses the gene (Bgee/TISSUES)
=========  ==========  ====================================================

GO annotations are propagated up the ontology, and Hetionet keeps terms with
at most 1,000 genes. Gene Ontology *molecular function* is deliberately left out: its most common
term, "protein binding" (GO:0005515), is annotated from interaction
experiments, so it would leak the interactions being predicted into the node
features.

The module serves two purposes: node features for the graph neural network
(:mod:`src.node_features`) and biological interpretation of predictions
(over-representation analysis and shared-annotation rates).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.datasets import EXTERNAL_DIR, dataset_path

logger = logging.getLogger(__name__)

#: Annotation source -> Hetionet metaedge.
SOURCES: dict[str, str] = {
    "go_bp": "GpBP",
    "go_cc": "GpCC",
    "pathway": "GpPW",
    "tissue": "AeG",
}
SOURCE_NAMES = {
    "go_bp": "GO biological process",
    "go_cc": "GO cellular component",
    "pathway": "Pathway",
    "tissue": "Tissue expression",
}

GeneSets = dict[str, frozenset[str]]


def _strip_gene_prefix(values: pd.Series) -> pd.Series:
    return values.str.removeprefix("Gene::")


@lru_cache(maxsize=4)
def _load_hetionet(nodes_path: str, edges_path: str, metaedges: tuple[str, ...]) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_path, sep="\t", usecols=["id", "name"])
    names = dict(zip(nodes["id"], nodes["name"]))
    edges = pd.read_csv(edges_path, sep="\t")
    edges = edges[edges["metaedge"].isin(metaedges)]
    # Gene-to-term metaedges (GpBP, GpCC, GpPW) start at the gene; AeG ends at it.
    gene_first = edges["source"].str.startswith("Gene::")
    gene = np.where(gene_first, edges["source"], edges["target"])
    term = np.where(gene_first, edges["target"], edges["source"])
    table = pd.DataFrame({"metaedge": edges["metaedge"].to_numpy(), "gene": gene, "term_id": term})
    table["gene"] = _strip_gene_prefix(table["gene"])
    table["term"] = table["term_id"].map(names).fillna(table["term_id"])
    return table


def load_annotations(
    sources: Sequence[str] = tuple(SOURCES),
    data_dir: str | Path = EXTERNAL_DIR,
    download_missing: bool = True,
) -> dict[str, GeneSets]:
    """Gene sets per annotation source.

    Returns
    -------
    ``{source: {term name: frozenset of Entrez gene IDs}}``. Term names are
    unique within a source; where two Hetionet terms share a name their
    identifiers are appended.
    """
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise KeyError(f"Unknown annotation sources {unknown}; choose from {tuple(SOURCES)}")
    nodes = dataset_path("hetionet-nodes", data_dir, download_missing)
    edges = dataset_path("hetionet-edges", data_dir, download_missing)
    table = _load_hetionet(str(nodes), str(edges), tuple(sorted(SOURCES[s] for s in sources)))

    annotations: dict[str, GeneSets] = {}
    for source in sources:
        part = table[table["metaedge"] == SOURCES[source]]
        duplicated = part.groupby("term")["term_id"].transform("nunique") > 1
        labels = part["term"].where(~duplicated, part["term"] + " (" + part["term_id"].str.split("::").str[-1] + ")")
        annotations[source] = {
            term: frozenset(genes) for term, genes in part.groupby(labels.to_numpy())["gene"]
        }
        logger.debug("%s: %d terms", source, len(annotations[source]))
    return annotations


def gene_to_terms(gene_sets: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    """Invert ``{term: genes}`` into ``{gene: terms}``."""
    inverted: dict[str, set[str]] = {}
    for term, genes in gene_sets.items():
        for gene in genes:
            inverted.setdefault(gene, set()).add(term)
    return inverted


def coverage(proteins: Iterable[Hashable], annotations: Mapping[str, GeneSets]) -> pd.Series:
    """Fraction of ``proteins`` with at least one term, per source."""
    proteins = {str(p) for p in proteins}
    return pd.Series(
        {
            source: len(proteins & set().union(*sets.values())) / max(len(proteins), 1)
            for source, sets in annotations.items()
        },
        name="coverage",
    )


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def enrichment(
    genes: Iterable[Hashable],
    gene_sets: Mapping[str, Iterable[str]],
    background: Iterable[Hashable],
    min_size: int = 5,
    max_size: int = 500,
) -> pd.DataFrame:
    """Over-representation of gene sets in ``genes`` (one-sided hypergeometric test).

    Gene sets are restricted to ``background`` and kept if their size lies in
    ``[min_size, max_size]``. ``genes`` outside the background are ignored.

    Returns
    -------
    DataFrame sorted by p-value with ``term``, ``overlap``, ``set_size``,
    ``fold_enrichment``, ``p_value``, ``fdr`` (Benjamini-Hochberg) and the
    overlapping ``genes``.
    """
    background = {str(g) for g in background}
    genes = {str(g) for g in genes} & background
    columns = ["term", "overlap", "set_size", "fold_enrichment", "p_value", "fdr", "genes"]
    if not genes:
        return pd.DataFrame(columns=columns)
    n_background, n_genes = len(background), len(genes)
    rows = []
    for term, members in gene_sets.items():
        members = set(members) & background
        if not (min_size <= len(members) <= max_size):
            continue
        hits = genes & members
        if not hits:
            continue
        p = stats.hypergeom.sf(len(hits) - 1, n_background, len(members), n_genes)
        expected = n_genes * len(members) / n_background
        rows.append((term, len(hits), len(members), len(hits) / expected, p, sorted(hits)))
    if not rows:
        return pd.DataFrame(columns=columns)
    table = pd.DataFrame(rows, columns=["term", "overlap", "set_size", "fold_enrichment", "p_value", "genes"])
    table["fdr"] = benjamini_hochberg(table["p_value"].to_numpy())
    return table[columns].sort_values(["p_value", "term"]).reset_index(drop=True)


def shared_annotation_rate(
    pairs: Sequence[tuple[Hashable, Hashable]],
    gene_sets: Mapping[str, Iterable[str]],
    max_size: int | None = 500,
) -> float:
    """Fraction of annotated pairs whose proteins share at least one term.

    Only pairs in which *both* proteins have an annotation count, so the rate
    measures functional similarity rather than annotation coverage. Terms
    larger than ``max_size`` (e.g. "cytoplasm") are ignored because almost any
    two proteins share them.
    """
    if max_size is not None:
        gene_sets = {t: g for t, g in gene_sets.items() if len(g) <= max_size}
    terms = gene_to_terms(gene_sets)
    shared = annotated = 0
    for u, v in pairs:
        tu, tv = terms.get(str(u)), terms.get(str(v))
        if tu and tv:
            annotated += 1
            shared += bool(tu & tv)
    return shared / annotated if annotated else float("nan")
