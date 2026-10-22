"""Biological interpretation of GNN embeddings and predictions.

Helpers used by ``notebooks/04_gnn_biological_interpretation.ipynb``:

* **functional modules** - cluster the learned protein embeddings and test
  each cluster for over-represented pathways / GO terms, against a
  permutation null of the same cluster sizes;
* **functional coherence** - does embedding similarity separate protein pairs
  that share a pathway from pairs that do not?
* **hubs and bottlenecks** - classify proteins by degree and betweenness
  (Yu et al., *PLoS Comput Biol* 2007) and find the interactions that carry
  the most shortest paths (edge betweenness);
* **interaction propensity** - how readily the GNN predicts partners for a
  protein, which exposes the model's degree (hub) bias.
"""

from __future__ import annotations

import logging
from typing import Hashable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from src.annotations import enrichment, gene_to_terms

logger = logging.getLogger(__name__)

CENTRALITY_CLASSES = ("hub-bottleneck", "non-hub bottleneck", "hub non-bottleneck", "peripheral")


# ---------------------------------------------------------------------------
# Embeddings and modules
# ---------------------------------------------------------------------------


def embedding_map(embeddings: pd.DataFrame, seed: int = 0, perplexity: float = 30.0) -> pd.DataFrame:
    """Two-dimensional t-SNE map of the embeddings (cosine geometry)."""
    coordinates = TSNE(
        n_components=2, perplexity=perplexity, init="pca", metric="cosine", random_state=seed
    ).fit_transform(embeddings.to_numpy())
    return pd.DataFrame(coordinates, index=embeddings.index, columns=["x", "y"])


def cluster_embeddings(embeddings: pd.DataFrame, n_clusters: int = 25, seed: int = 0) -> pd.Series:
    """K-means clusters of L2-normalised embeddings, numbered by decreasing size (1 = largest)."""
    values = normalize(embeddings.to_numpy())
    raw = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(values)
    order = pd.Series(raw).value_counts().index  # largest first; ties by label
    relabel = {old: new for new, old in enumerate(order, start=1)}
    return pd.Series([relabel[c] for c in raw], index=embeddings.index, name="cluster")


def _background(gene_sets: Mapping[str, frozenset[str]], proteins) -> set[str]:
    annotated = set().union(*gene_sets.values()) if gene_sets else set()
    return {str(p) for p in proteins} & annotated


def cluster_enrichment(
    labels: pd.Series,
    gene_sets: Mapping[str, frozenset[str]],
    symbols: Mapping[str, str] | None = None,
    fdr: float = 0.05,
    min_size: int = 5,
    max_size: int = 500,
) -> pd.DataFrame:
    """Most enriched term of each cluster.

    The background is every clustered protein with at least one annotation in
    ``gene_sets``. Returns one row per cluster with its size, number of
    significant terms, and the top term with its overlap, fold enrichment and
    FDR (empty when nothing passes ``fdr``).
    """
    background = _background(gene_sets, labels.index)
    rows = []
    for cluster, members in labels.groupby(labels):
        table = enrichment(members.index, gene_sets, background, min_size, max_size)
        significant = table[table["fdr"] < fdr]
        row = {"cluster": cluster, "proteins": len(members), "significant_terms": len(significant)}
        if len(significant):
            top = significant.iloc[0]
            genes = top["genes"]
            if symbols:
                genes = [symbols.get(g, g) for g in genes]
            row.update(
                top_term=top["term"], overlap=int(top["overlap"]), set_size=int(top["set_size"]),
                fold_enrichment=float(top["fold_enrichment"]), fdr=float(top["fdr"]),
                example_genes=", ".join(sorted(genes)[:8]),
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


def enriched_cluster_fraction(
    labels: pd.Series,
    gene_sets: Mapping[str, frozenset[str]],
    fdr: float = 0.05,
    n_permutations: int = 0,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Fraction of clusters with at least one enriched term, and the same for shuffled labels.

    Shuffling keeps the cluster sizes but breaks any relation to the
    embeddings, so the permutation values estimate what chance alone yields.
    """
    def fraction(current: pd.Series) -> float:
        table = cluster_enrichment(current, gene_sets, fdr=fdr)
        return float((table["significant_terms"] > 0).mean())

    observed = fraction(labels)
    rng = np.random.default_rng(seed)
    null = np.array(
        [fraction(pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)) for _ in range(n_permutations)]
    )
    return observed, null


def functional_similarity_auc(
    embeddings: pd.DataFrame,
    gene_sets: Mapping[str, frozenset[str]],
    n_pairs: int = 50_000,
    max_size: int = 500,
    seed: int = 0,
) -> float:
    """ROC-AUC of embedding cosine similarity for "shares a term" vs "shares none".

    Random pairs of annotated proteins are drawn; 0.5 means the embedding
    geometry carries no information about shared function.
    """
    gene_sets = {t: g for t, g in gene_sets.items() if len(g) <= max_size}
    terms = gene_to_terms(gene_sets)
    proteins = [p for p in embeddings.index if terms.get(str(p))]
    if len(proteins) < 2:
        return float("nan")
    values = normalize(embeddings.loc[proteins].to_numpy())
    rng = np.random.default_rng(seed)
    u, v = rng.integers(len(proteins), size=n_pairs), rng.integers(len(proteins), size=n_pairs)
    keep = u != v
    u, v = u[keep], v[keep]
    similarity = np.einsum("ij,ij->i", values[u], values[v])
    shared = np.array([bool(terms[str(proteins[a])] & terms[str(proteins[b])]) for a, b in zip(u, v)])
    if shared.all() or not shared.any():
        return float("nan")
    return float(roc_auc_score(shared, similarity))


# ---------------------------------------------------------------------------
# Hubs, bottlenecks and propensity
# ---------------------------------------------------------------------------


def centrality_classes(
    graph: nx.Graph,
    symbols: Mapping[str, str] | None = None,
    quantile: float = 0.9,
    betweenness_samples: int | None = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Degree, betweenness and Yu et al. hub/bottleneck class of every protein.

    Hubs are the top ``1 - quantile`` by degree, bottlenecks the top
    ``1 - quantile`` by betweenness. Bottlenecks that are not hubs bridge
    modules without having many partners of their own.
    """
    k = betweenness_samples if betweenness_samples and betweenness_samples < graph.number_of_nodes() else None
    betweenness = nx.betweenness_centrality(graph, k=k, seed=seed)
    table = pd.DataFrame(
        {"degree": dict(graph.degree()), "betweenness": betweenness}
    ).rename_axis("protein")
    if symbols:
        table.insert(0, "symbol", [symbols.get(str(p), str(p)) for p in table.index])
    hub = table["degree"] >= table["degree"].quantile(quantile)
    bottleneck = table["betweenness"] >= table["betweenness"].quantile(quantile)
    table["centrality_class"] = np.select(
        [hub & bottleneck, ~hub & bottleneck, hub & ~bottleneck], CENTRALITY_CLASSES[:3], CENTRALITY_CLASSES[3]
    )
    return table.sort_values("betweenness", ascending=False)


def bottleneck_interactions(
    graph: nx.Graph,
    labels: pd.Series | None = None,
    symbols: Mapping[str, str] | None = None,
    top: int = 20,
    samples: int | None = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Interactions with the highest edge betweenness (the network's bridges).

    With ``labels`` (e.g. embedding clusters) each row also says whether the
    interaction joins two different modules.
    """
    k = samples if samples and samples < graph.number_of_nodes() else None
    scores = nx.edge_betweenness_centrality(graph, k=k, seed=seed)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))[:top]
    rows = []
    for (u, v), value in ranked:
        row = {"protein_a": u, "protein_b": v, "edge_betweenness": value}
        if symbols:
            row.update(symbol_a=symbols.get(str(u), str(u)), symbol_b=symbols.get(str(v), str(v)))
        if labels is not None:
            row.update(cluster_a=int(labels[u]), cluster_b=int(labels[v]), between_modules=labels[u] != labels[v])
        rows.append(row)
    return pd.DataFrame(rows)


def between_module_rate(graph: nx.Graph, labels: pd.Series) -> float:
    """Fraction of all interactions that join two different clusters."""
    edges = list(graph.edges())
    return float(np.mean([labels[u] != labels[v] for u, v in edges])) if edges else float("nan")


def interaction_propensity(trained, graph: nx.Graph, n_partners: int = 200, seed: int = 0) -> pd.Series:
    """Mean predicted interaction probability of each protein with random partners.

    A model without degree bias would give every protein a similar
    propensity; a strong correlation with degree means that predictions
    concentrate on hubs.
    """
    nodes = trained.nodes
    rng = np.random.default_rng(seed)
    partners = rng.integers(len(nodes), size=(len(nodes), n_partners))
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in partners[i] if j != i]
    owners = np.array([i for i in range(len(nodes)) for j in partners[i] if j != i])
    probabilities = trained.predict_proba(graph, pairs)
    propensity = np.bincount(owners, weights=probabilities, minlength=len(nodes)) / np.maximum(
        np.bincount(owners, minlength=len(nodes)), 1
    )
    return pd.Series(propensity, index=pd.Index(nodes, name="protein"), name="propensity")


def label_symbols(proteins, symbols: Mapping[str, str]) -> list[str]:
    """Gene symbols for protein identifiers (identifier when unknown)."""
    return [symbols.get(str(p), str(p)) for p in proteins]
