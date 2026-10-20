from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.data_analysis import (
    closure_statistics,
    closure_with_null_model,
    clustering_spectrum,
    compare_networks,
    degree_ccdf,
    network_statistics,
    overlap_matrix,
    plot_closure_enrichment,
    plot_degree_ccdfs,
    plot_overlap,
    plot_preprocessing_steps,
    randomized_network,
    run_analysis,
    top_hubs,
)
from src.evaluate import save_figure
from src.preprocessing import PreprocessingConfig, preprocess_interactions


def test_closure_on_triangle_and_square():
    triangle = closure_statistics(nx.complete_graph(3))
    assert triangle["triangle_closure"] == 1.0 and triangle["square_closure"] == 0.0
    square = closure_statistics(nx.cycle_graph(4))
    assert square["triangle_closure"] == 0.0 and square["square_closure"] == 1.0
    assert square["mean_l3_paths"] == 1.0


def test_closure_matches_brute_force(small_ppi_graph):
    stats = closure_statistics(small_ppi_graph, chunk_size=97)
    g = small_ppi_graph
    cn, p3 = [], []
    for u, v in g.edges():
        cn.append(len(set(g[u]) & set(g[v])))
        p3.append(sum(1 for a in g[u] if a != v for b in g[a] if b not in (u, v) and g.has_edge(b, v)))
    assert stats["mean_common_neighbors"] == pytest.approx(np.mean(cn))
    assert stats["mean_l3_paths"] == pytest.approx(np.mean(p3))
    assert stats["square_closure"] == pytest.approx(np.mean(np.array(p3) > 0))


def test_randomization_preserves_degrees(small_ppi_graph):
    randomized = randomized_network(small_ppi_graph, seed=1)
    assert dict(randomized.degree()) == dict(small_ppi_graph.degree())
    assert set(map(frozenset, randomized.edges())) != set(map(frozenset, small_ppi_graph.edges()))


def test_null_model_enrichment(small_ppi_graph):
    result = closure_with_null_model(small_ppi_graph, n_null=2)
    assert result["common_neighbors_enrichment"] == pytest.approx(
        result["mean_common_neighbors"] / result["mean_common_neighbors_random"]
    )
    # powerlaw_cluster_graph adds triangles on purpose.
    assert result["common_neighbors_enrichment"] > 1


def test_statistics_and_overlap(karate_graph, small_ppi_graph):
    stats = network_statistics(karate_graph, distance_sources=34)
    assert stats["proteins"] == 34 and stats["interactions"] == 78
    assert stats["mean_shortest_path"] == pytest.approx(nx.average_shortest_path_length(karate_graph))
    table = compare_networks({"karate": karate_graph, "ppi": small_ppi_graph})
    assert table.index.tolist() == ["karate", "ppi"]

    sub = karate_graph.subgraph(list(karate_graph)[:20]).copy()
    proteins = overlap_matrix({"a": karate_graph, "b": sub}, "proteins")
    assert proteins.loc["a", "a"] == 1.0
    assert proteins.loc["a", "b"] == proteins.loc["b", "a"] == pytest.approx(20 / 34)
    interactions = overlap_matrix({"a": karate_graph, "b": sub}, "interactions")
    assert interactions.loc["a", "b"] == pytest.approx(sub.number_of_edges() / 78)
    with pytest.raises(ValueError):
        overlap_matrix({"a": karate_graph}, "genes")


def test_degree_distribution_helpers(small_ppi_graph):
    k, p = degree_ccdf(small_ppi_graph)
    assert p[0] == 1.0 and np.all(np.diff(p) < 0) and np.all(np.diff(k) > 0)
    spectrum = clustering_spectrum(small_ppi_graph)
    assert spectrum.index.min() >= 2 and spectrum["size"].sum() <= small_ppi_graph.number_of_nodes()


def test_top_hubs(karate_graph):
    hubs = top_hubs(karate_graph, {"P033": "HUB1"}, n=3)
    assert hubs.loc[0, "protein"] == "P033" and hubs.loc[0, "symbol"] == "HUB1"
    assert hubs["degree"].is_monotonic_decreasing


def test_plots(tmp_path, karate_graph, small_ppi_graph):
    raw = pd.DataFrame({"protein_a": ["A", "A", "B"], "protein_b": ["B", "A", "C"], "weight": 1.0})
    _, report = preprocess_interactions(raw, config=PreprocessingConfig())
    closure = pd.DataFrame(
        {"common_neighbors_enrichment": [0.8, 6.0], "l3_enrichment": [1.5, 14.0]}, index=["x", "y"]
    )
    graphs = {"karate": karate_graph, "ppi": small_ppi_graph}
    for fig, name in [
        (plot_preprocessing_steps(report.to_frame()), "steps.png"),
        (plot_degree_ccdfs(graphs), "ccdf.png"),
        (plot_overlap(overlap_matrix(graphs), "overlap"), "overlap.png"),
        (plot_closure_enrichment(closure), "closure.png"),
    ]:
        assert save_figure(fig, tmp_path / name).stat().st_size > 0


def test_run_analysis_on_processed_files(tmp_path, small_ppi_graph):
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, graph in [("hi-iii", small_ppi_graph), ("bioplex-2", nx.karate_club_graph())]:
        raw = pd.DataFrame([(str(u), str(v), 1.0) for u, v in graph.edges()], columns=["protein_a", "protein_b", "weight"])
        edges, report = preprocess_interactions(raw)
        edges.to_csv(processed / f"{name}.tsv", sep="\t", index=False)
        report.save(processed / f"{name}.report.json")

    out = tmp_path / "analysis"
    tables = run_analysis(["hi-iii", "bioplex-2"], out, processed_dir=processed, n_null=1)
    assert tables["network_statistics"].loc["bioplex-2", "kind"] == "co-complex"
    for name in ("network_statistics", "closure", "protein_overlap", "interaction_overlap", "preprocessing_steps"):
        assert (out / f"{name}.csv").exists()
    for name in ("degree_ccdf", "closure_enrichment", "protein_overlap", "preprocessing_hi-iii"):
        assert (out / "figures" / f"{name}.png").exists()
