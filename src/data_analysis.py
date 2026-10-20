"""Exploratory analysis and comparison of protein interaction networks.

Functions here answer the questions to ask of an interactome before
modelling it:

* How large, dense and clustered is it, and how heterogeneous are degrees?
  (:func:`network_statistics`, :func:`degree_ccdf`, :func:`clustering_spectrum`)
* How much do different maps of the same interactome agree?
  (:func:`overlap_matrix`)
* Do interacting proteins share partners (triangles) or are they linked by
  paths of length three (squares)? (:func:`closure_statistics`) This decides
  which link-prediction heuristics can work: triadic closure needs
  triangles, L3 needs squares.
* Which proteins are the hubs? (:func:`top_hubs`)

Command line - analyse the processed registry datasets and write tables and
figures::

    python src/data_analysis.py --output-dir reports/data_analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None, ""):  # allow `python src/data_analysis.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.datasets import DATASETS, INTERACTOMES
from src.evaluate import INK, PALETTE, SEQUENTIAL, plot_style, save_figure, use_headless_backend
from src.graph_builder import build_graph, graph_summary, load_edge_list
from src.preprocessing import PROCESSED_DIR, PreprocessingReport, load_symbol_map, processed_path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "data_analysis"
DEFAULT_DATASETS = ("hi-iii", "hi-ii-14", "lit-bm-13", "bioplex-2")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mean_shortest_path(graph: nx.Graph, n_sources: int = 200, seed: int = 0) -> float:
    """Mean hop distance between reachable proteins, estimated from BFS sources."""
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    sources = rng.choice(len(nodes), size=min(n_sources, len(nodes)), replace=False)
    total, count = 0, 0
    for i in sources:
        lengths = nx.single_source_shortest_path_length(graph, nodes[i])
        total += sum(lengths.values())
        count += len(lengths) - 1
    return total / count if count else float("nan")


def network_statistics(graph: nx.Graph, distance_sources: int = 200, seed: int = 0) -> dict[str, float]:
    """Global statistics: size, density, degree, clustering, distance, assortativity."""
    stats = graph_summary(graph)
    degrees = np.array([d for _, d in graph.degree()])
    stats["median_degree"] = float(np.median(degrees)) if degrees.size else 0.0
    stats["mean_shortest_path"] = mean_shortest_path(graph, distance_sources, seed)
    order = [
        "proteins", "interactions", "density", "mean_degree", "median_degree", "max_degree",
        "average_clustering", "transitivity", "degree_assortativity", "mean_shortest_path",
        "connected_components", "largest_component_fraction",
    ]
    return {k: stats[k] for k in order if k in stats}


def compare_networks(graphs: Mapping[str, nx.Graph], **kwargs) -> pd.DataFrame:
    """:func:`network_statistics` for several networks (one row each)."""
    return pd.DataFrame({name: network_statistics(g, **kwargs) for name, g in graphs.items()}).T


def degree_ccdf(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Complementary cumulative degree distribution ``P(K >= k)``."""
    degrees = np.sort(np.array([d for _, d in graph.degree()]))
    values, first = np.unique(degrees, return_index=True)
    return values, 1.0 - first / len(degrees)


def clustering_spectrum(graph: nx.Graph) -> pd.DataFrame:
    """Mean clustering coefficient ``C(k)`` of proteins with degree ``k >= 2``."""
    frame = pd.DataFrame({"degree": dict(graph.degree()), "clustering": nx.clustering(graph)})
    return frame[frame["degree"] >= 2].groupby("degree")["clustering"].agg(["mean", "size"])


def overlap_matrix(graphs: Mapping[str, nx.Graph], level: str = "proteins") -> pd.DataFrame:
    """Pairwise Jaccard similarity of protein sets or interaction sets."""
    if level == "proteins":
        sets = {name: set(g.nodes()) for name, g in graphs.items()}
    elif level == "interactions":
        sets = {name: {frozenset(e) for e in g.edges()} for name, g in graphs.items()}
    else:
        raise ValueError("level must be 'proteins' or 'interactions'")
    names = list(sets)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = len(sets[a] | sets[b])
            matrix.loc[a, b] = len(sets[a] & sets[b]) / union if union else np.nan
    return matrix


def closure_statistics(graph: nx.Graph, chunk_size: int = 2000) -> dict[str, float]:
    """How often an interaction closes a triangle or a square.

    For every edge ``(u, v)`` count common neighbours (paths of length two)
    and simple paths of length three ``u-a-b-v`` that avoid the edge itself
    (``W3_uv - k_u - k_v + 1`` from the walk-count matrix). Returns the
    fraction of interactions with at least one of each, and their means.
    """
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    if not edges:
        return {"triangle_closure": np.nan, "square_closure": np.nan,
                "mean_common_neighbors": np.nan, "mean_l3_paths": np.nan}
    index = {n: i for i, n in enumerate(nodes)}
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, dtype=float, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    u = np.fromiter((index[a] for a, _ in edges), dtype=np.int64, count=len(edges))
    v = np.fromiter((index[b] for _, b in edges), dtype=np.int64, count=len(edges))

    common = np.empty(len(edges))
    paths3 = np.empty(len(edges))
    for start in range(0, len(edges), chunk_size):
        sl = slice(start, start + chunk_size)
        rows_u, rows_v = adjacency[u[sl]], adjacency[v[sl]]
        common[sl] = np.asarray(rows_u.multiply(rows_v).sum(axis=1)).ravel()
        walks3 = np.asarray((rows_u @ adjacency).multiply(rows_v).sum(axis=1)).ravel()
        paths3[sl] = walks3 - degree[u[sl]] - degree[v[sl]] + 1
    return {
        "triangle_closure": float((common > 0).mean()),
        "square_closure": float((paths3 > 0).mean()),
        "mean_common_neighbors": float(common.mean()),
        "mean_l3_paths": float(paths3.mean()),
    }


def randomized_network(graph: nx.Graph, swaps_per_edge: float = 3.0, seed: int = 0) -> nx.Graph:
    """Degree-preserving randomisation (double edge swaps) as a null model."""
    randomized = nx.Graph(graph)
    n_swaps = int(swaps_per_edge * randomized.number_of_edges())
    nx.double_edge_swap(randomized, nswap=n_swaps, max_tries=n_swaps * 20, seed=seed)
    return randomized


def closure_with_null_model(graph: nx.Graph, n_null: int = 3, seed: int = 0) -> dict[str, float]:
    """Closure statistics of a network relative to degree-preserving randomisations.

    Returns the observed :func:`closure_statistics`, their means over
    ``n_null`` randomised networks (``*_random``) and the fold enrichment of
    shared partners and length-3 paths over that null model. Enrichment
    above 1 means interacting proteins are connected that way more often
    than their degrees alone explain.
    """
    observed = closure_statistics(graph)
    nulls = [closure_statistics(randomized_network(graph, seed=seed + i)) for i in range(n_null)]
    result = dict(observed)
    for key in observed:
        result[f"{key}_random"] = float(np.mean([n[key] for n in nulls]))
    result["common_neighbors_enrichment"] = observed["mean_common_neighbors"] / result["mean_common_neighbors_random"]
    result["l3_enrichment"] = observed["mean_l3_paths"] / result["mean_l3_paths_random"]
    return result


def top_hubs(graph: nx.Graph, symbols: Mapping[str, str] | None = None, n: int = 15) -> pd.DataFrame:
    """Highest-degree proteins with gene symbols and clustering coefficients."""
    symbols = symbols or {}
    degree = pd.Series(dict(graph.degree()), name="degree").sort_values(ascending=False).head(n)
    clustering = nx.clustering(graph, nodes=list(degree.index))
    return pd.DataFrame(
        {
            "protein": degree.index,
            "symbol": [symbols.get(p, "") for p in degree.index],
            "degree": degree.to_numpy(),
            "clustering": [clustering[p] for p in degree.index],
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_preprocessing_steps(steps: pd.DataFrame, title: str = "Preprocessing audit") -> plt.Figure:
    """Interactions remaining after each step, annotated with what was removed."""
    labels = ["raw"] + list(steps.index)
    remaining = [steps["interactions_in"].iloc[0]] + steps["interactions_out"].tolist()
    removed = [0] + steps["interactions_removed"].tolist()
    with plot_style():
        fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(labels) + 1.3))
        y = np.arange(len(labels))
        ax.barh(y, remaining, color=PALETTE[0], height=0.62)
        for yi, value, gone in zip(y, remaining, removed):
            note = f"{value:,}" + (f"   (-{gone:,})" if gone else "")
            ax.text(value, yi, "  " + note, va="center", fontsize=9, color=INK["secondary"])
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(remaining) * 1.32)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="Interactions remaining")
        ax.set_title(title)
    return fig


def plot_degree_ccdfs(graphs: Mapping[str, nx.Graph], title: str = "Degree distributions") -> plt.Figure:
    """Overlaid log-log CCDFs, one colour per network in fixed order."""
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for colour, (name, graph) in zip(PALETTE, graphs.items()):
            k, p = degree_ccdf(graph)
            ax.loglog(k, p, marker="o", markersize=3.5, linewidth=1.2, color=colour, label=name)
        ax.set(xlabel="Degree k", ylabel="P(K ≥ k)")
        ax.set_title(title)
        ax.legend()
    return fig


def plot_overlap(matrix: pd.DataFrame, title: str) -> plt.Figure:
    """Annotated heatmap of a Jaccard overlap matrix."""
    import seaborn as sns

    cmap = LinearSegmentedColormap.from_list("ppi_blues", SEQUENTIAL)
    with plot_style():
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        sns.heatmap(
            matrix.astype(float), annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1, square=True,
            linewidths=2, linecolor=INK["surface"], cbar_kws={"shrink": 0.75, "label": "Jaccard"}, ax=ax,
        )
        for text in ax.texts:
            text.set_color("white" if float(text.get_text()) > 0.5 else INK["primary"])
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_title(title)
    return fig


def plot_closure_enrichment(
    closure: pd.DataFrame, title: str = "Shared partners vs paths of length three"
) -> plt.Figure:
    """Fold enrichment of common neighbours and L3 paths over a degree-preserving null model."""
    with plot_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = np.arange(len(closure))
        width = 0.38
        series = [
            ("common_neighbors_enrichment", PALETTE[1], "common neighbours (triangles)"),
            ("l3_enrichment", PALETTE[0], "paths of length 3 (squares)"),
        ]
        for offset, (col, colour, label) in zip((-width / 2, width / 2), series):
            values = closure[col].to_numpy(dtype=float)
            # Bars grow from the random expectation (1x) up for enrichment, down for depletion.
            ax.bar(x + offset, values - 1, bottom=1, width=width * 0.92, color=colour, label=label)
            for xi, value in zip(x + offset, values):
                above = value >= 1
                ax.text(xi, value * (1.06 if above else 0.95), f"{value:.2g}×", ha="center",
                        va="bottom" if above else "top", fontsize=9, color=INK["secondary"])
        ax.axhline(1, color=INK["muted"], linewidth=1, linestyle=":", label="random expectation (1×)")
        ax.set_yscale("log")
        ax.set_ylim(0.5, closure[[c for c, _, _ in series]].to_numpy().max() * 1.8)
        ax.set_xticks(x, closure.index)
        ax.set(ylabel="Observed / degree-preserving random")
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.legend(loc="upper left")
    return fig


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def load_processed_graphs(names: tuple[str, ...] | list[str], processed_dir: Path = PROCESSED_DIR) -> dict:
    """Processed registry networks as graphs (preprocessing them if needed)."""
    return {
        name: build_graph(load_edge_list(processed_path(name, processed_dir)), largest_component=False)
        for name in names
    }


def run_analysis(
    names: tuple[str, ...] | list[str] = DEFAULT_DATASETS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    hubs_dataset: str | None = None,
    processed_dir: str | Path = PROCESSED_DIR,
    n_null: int = 3,
) -> dict[str, pd.DataFrame]:
    """Write comparison tables and figures for the given processed datasets."""
    output_dir, processed_dir = Path(output_dir), Path(processed_dir)
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs = load_processed_graphs(names, processed_dir)

    steps = pd.concat(
        {n: PreprocessingReport.load(processed_dir / f"{n}.report.json").to_frame() for n in names},
        names=["dataset", "step"],
    )
    stats = compare_networks(graphs)
    stats.insert(0, "kind", [DATASETS[n].kind for n in names])
    closure = pd.DataFrame({n: closure_with_null_model(g, n_null=n_null) for n, g in graphs.items()}).T
    proteins = overlap_matrix(graphs, "proteins")
    interactions = overlap_matrix(graphs, "interactions")
    hubs_dataset = hubs_dataset or names[0]
    hubs = top_hubs(graphs[hubs_dataset], load_symbol_map(processed_dir / f"{hubs_dataset}.tsv"))

    tables = {
        "preprocessing_steps": steps.drop(columns="description"),
        "network_statistics": stats,
        "closure": closure,
        "protein_overlap": proteins,
        "interaction_overlap": interactions,
        f"top_hubs_{hubs_dataset}": hubs,
    }
    for name, table in tables.items():
        table.round(4).to_csv(output_dir / f"{name}.csv", index=not name.startswith("top_hubs"))

    save_figure(plot_degree_ccdfs(graphs, "Degree distributions of human interactomes"), figures / "degree_ccdf.png")
    save_figure(plot_overlap(proteins, "Protein overlap (Jaccard)"), figures / "protein_overlap.png")
    save_figure(plot_overlap(interactions, "Interaction overlap (Jaccard)"), figures / "interaction_overlap.png")
    save_figure(plot_closure_enrichment(closure), figures / "closure_enrichment.png")
    first = PreprocessingReport.load(processed_dir / f"{hubs_dataset}.report.json")
    save_figure(plot_preprocessing_steps(first.to_frame(), f"Preprocessing audit - {hubs_dataset}"),
                figures / f"preprocessing_{hubs_dataset}.png")
    logger.info("Data analysis written to %s", output_dir)
    return tables


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description="Compare processed PPI networks.")
    parser.add_argument("--datasets", nargs="+", choices=INTERACTOMES, default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hubs-dataset", choices=INTERACTOMES, help="network for the hub table")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    use_headless_backend()
    tables = run_analysis(args.datasets, args.output_dir, args.hubs_dataset)
    with pd.option_context("display.width", 160, "display.precision", 3):
        for name in ("network_statistics", "closure"):
            print(f"\n{name}\n{tables[name]}")
    return tables


if __name__ == "__main__":
    main()
