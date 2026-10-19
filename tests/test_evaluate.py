from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compute_metrics,
    evaluate_scores,
    heuristic_scores,
    main as evaluate_main,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    r_precision,
    save_figure,
)

Y = np.array([0, 0, 0, 1, 1, 1])


def test_perfect_scores():
    metrics = compute_metrics(Y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    for key in ("roc_auc", "average_precision", "r_precision", "precision", "recall", "f1", "accuracy", "mcc"):
        assert metrics[key] == pytest.approx(1.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (3, 0, 0, 3)


def test_threshold_metrics_are_optional():
    metrics = compute_metrics(Y, np.array([5, 1, 0, 9, 7, 2]), threshold=None)
    assert set(metrics) == {"roc_auc", "average_precision", "r_precision"}
    assert metrics["roc_auc"] == pytest.approx(8 / 9)


def test_compute_metrics_needs_both_classes():
    with pytest.raises(ValueError):
        compute_metrics(np.ones(4), np.arange(4))


def test_r_precision():
    assert r_precision(Y, np.array([0.9, 0.1, 0.2, 0.8, 0.7, 0.3])) == pytest.approx(2 / 3)


def test_evaluate_scores_table():
    table = evaluate_scores(
        Y,
        {"good_model": np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])},
        {"weak_heuristic": np.array([3, 1, 2, 2, 1, 3])},
    )
    assert table.index.tolist() == ["good_model", "weak_heuristic"]  # sorted by ROC-AUC
    assert table.loc["good_model", "kind"] == "model"
    assert np.isnan(table.loc["weak_heuristic", "f1"])


def test_heuristic_scores_orient_shortest_path():
    X = pd.DataFrame({"l3": [0.0, 2.0], "shortest_path": [2.0, 5.0], "degree_min": [1, 2]})
    scores = heuristic_scores(X)
    assert set(scores) == {"l3", "shortest_path"}
    assert scores["shortest_path"][0] > scores["shortest_path"][1]  # closer = more likely


def test_plots_are_written(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = {"model": y + rng.normal(0, 0.8, 200), "heuristic": rng.random(200)}
    kinds = {"model": "model", "heuristic": "heuristic"}
    importance = pd.DataFrame({"feature": ["a", "b"], "importance_mean": [0.2, 0.01], "importance_std": [0.02, 0.01]})
    paths = [
        save_figure(plot_roc_curves(y, scores, kinds), tmp_path / "roc.png"),
        save_figure(plot_pr_curves(y, scores, kinds), tmp_path / "pr.png"),
        save_figure(plot_confusion_matrix(y, (scores["model"] > 0.5).astype(int)), tmp_path / "cm.png"),
        save_figure(plot_feature_importance(importance), tmp_path / "sub" / "importance.png"),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_cli_recomputes_metrics(tmp_path):
    predictions = pd.DataFrame(
        {
            "protein_a": list("ABCDEF"),
            "protein_b": list("GHIJKL"),
            "label": Y,
            "model__clf": [0.1, 0.6, 0.3, 0.7, 0.8, 0.9],
            "heuristic__l3": [0, 1, 0, 2, 3, 1],
        }
    )
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    table = evaluate_main(["--predictions", str(path)])
    assert set(table.index) == {"clf", "l3"}
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "figures" / "roc_curves.png").exists()
    assert (tmp_path / "figures" / "confusion_matrix.png").exists()
