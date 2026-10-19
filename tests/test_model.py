from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.model import (
    MODEL_NAMES,
    PipelineConfig,
    TrainedModel,
    build_model,
    l3_candidate_pairs,
    load_model,
    main,
    rank_novel_interactions,
    save_model,
    train_models,
)


@pytest.fixture
def toy_dataset():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((120, 4)), columns=list("abcd"))
    y = (X["a"] + 0.3 * rng.random(120) > 0.65).astype(int).to_numpy()
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_and_train_each_model(toy_dataset, name):
    X, y = toy_dataset
    trained = train_models(X, y, [name], cv_folds=3)[name]
    assert isinstance(trained, TrainedModel)
    proba = trained.predict_proba(X)
    assert proba.shape == (len(X),) and np.all((proba >= 0) & (proba <= 1))
    assert set(trained.predict(X)) <= {0, 1}
    assert 0.5 < trained.cv_scores["cv_roc_auc_mean"] <= 1.0


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("svm")


def test_save_and_load_roundtrip(toy_dataset, tmp_path):
    X, y = toy_dataset
    trained = train_models(X, y, ["logistic_regression"], cv_folds=0)["logistic_regression"]
    assert trained.cv_scores == {}
    path = save_model(trained, tmp_path / "models" / "lr.joblib")
    np.testing.assert_allclose(load_model(path).predict_proba(X), trained.predict_proba(X))


def test_l3_candidates_are_unobserved_and_ranked():
    graph = nx.cycle_graph(6)  # 0-1-2-3-4-5-0
    candidates = l3_candidate_pairs(graph, n_candidates=10)
    assert candidates, "a 6-cycle has length-3 paths between opposite nodes"
    assert all(not graph.has_edge(u, v) and u != v for u, v in candidates)
    assert {frozenset(p) for p in candidates} == {frozenset((0, 3)), frozenset((1, 4)), frozenset((2, 5))}


def test_rank_novel_interactions(small_ppi_graph):
    from src.feature_extractor import TopologicalFeatureExtractor

    names = TopologicalFeatureExtractor().feature_names
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((60, len(names))), columns=names)
    y = (X["l3"] > 0.5).astype(int).to_numpy()
    model = train_models(X, y, ["random_forest"], cv_folds=0)["random_forest"]

    ranked = rank_novel_interactions(small_ppi_graph, model, top_k=15, n_candidates=50)
    assert len(ranked) == 15
    assert ranked["probability"].is_monotonic_decreasing
    assert not any(small_ppi_graph.has_edge(u, v) for u, v in zip(ranked.protein_a, ranked.protein_b))


def test_pipeline_end_to_end(tmp_path):
    graph = nx.powerlaw_cluster_graph(250, 3, 0.3, seed=5)
    edges = pd.DataFrame(
        [(f"P{u:03d}", f"P{v:03d}", 900) for u, v in graph.edges()],
        columns=["protein_a", "protein_b", "score"],
    )
    data = tmp_path / "edges.tsv"
    edges.to_csv(data, sep="\t", index=False)
    out = tmp_path / "out"

    results = main(
        [
            "--input", str(data),
            "--output-dir", str(out),
            "--models", "logistic_regression", "random_forest",
            "--cv-folds", "3",
            "--betweenness-samples", "50",
            "--negatives", "local",
            "--top-novel", "10",
            "--importance-repeats", "2",
        ]
    )

    assert results["best_model"] in {"logistic_regression", "random_forest"}
    for name in ("metrics.csv", "predictions.csv", "feature_importance.csv", "novel_predictions.csv", "run_summary.json"):
        assert (out / name).exists(), name
    for name in ("roc_curves", "pr_curves", "confusion_matrix", "feature_importance"):
        assert (out / "figures" / f"{name}.png").exists(), name
    assert (out / "models" / "random_forest.joblib").exists()

    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["config"]["negative_strategy"] == "local"
    assert summary["split"]["test_positives"] == summary["split"]["test_negatives"]

    metrics = pd.read_csv(out / "metrics.csv", index_col="method")
    assert {"l3", "common_neighbors"} <= set(metrics.index)
    assert metrics.loc[results["best_model"], "roc_auc"] > 0.6

    predictions = pd.read_csv(out / "predictions.csv")
    assert predictions["label"].sum() == summary["split"]["test_positives"]
    assert len(pd.read_csv(out / "novel_predictions.csv")) == 10


def test_config_defaults_point_at_bundled_sample():
    config = PipelineConfig()
    assert config.input_path.name == "sample_ppi_edges.tsv"
    assert config.input_path.exists()
