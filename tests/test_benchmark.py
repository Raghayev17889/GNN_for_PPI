from __future__ import annotations

import pandas as pd

import src.benchmark as benchmark
from src.evaluate import save_figure


def _fake_run_pipeline(config):
    base = 0.9 if config.negative_strategy == "uniform" else 0.7
    methods = [("random_forest", "model", base + 0.05), ("l3", "heuristic", base), ("common_neighbors", "heuristic", 0.5)]
    return {
        "metrics": [
            {"method": m, "kind": k, "roc_auc": auc, "average_precision": auc - 0.01} for m, k, auc in methods
        ],
        "split": {"test_positives": 10, "test_negatives": 10},
    }


def test_run_benchmark_collects_all_combinations(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "run_pipeline", _fake_run_pipeline)
    table = benchmark.run_benchmark(
        ["hi-iii", "bioplex-2"], ["uniform", "local"], tmp_path / "out", tmp_path / "runs"
    )
    assert len(table) == 2 * 2 * 3
    assert set(table["network_type"]) == {"binary", "co-complex"}
    assert (tmp_path / "out" / "benchmark_metrics.csv").exists()
    assert (tmp_path / "out" / "figures" / "roc_auc_heatmap.png").exists()

    pivot = pd.read_csv(tmp_path / "out" / "roc_auc_by_dataset.csv", index_col=[0, 1])
    assert pivot.columns.tolist() == ["random_forest", "l3", "common_neighbors"]  # HEATMAP_METHODS order
    assert pivot.index.tolist()[0] == ("hi-iii", "uniform")
    assert pivot.loc[("bioplex-2", "local"), "random_forest"] == 0.75


def test_heatmap_renders(tmp_path):
    table = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "negatives": ["uniform"] * 4,
            "method": ["l3", "common_neighbors"] * 2,
            "roc_auc": [0.9, 0.6, 0.8, 0.95],
        }
    )
    pivot = benchmark.benchmark_pivot(table)
    assert pivot.shape == (2, 2)
    assert save_figure(benchmark.plot_benchmark_heatmap(pivot), tmp_path / "h.png").stat().st_size > 0
