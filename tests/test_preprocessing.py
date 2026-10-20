from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    PreprocessingConfig,
    PreprocessingReport,
    load_gene_annotation,
    load_symbol_map,
    normalise_identifiers,
    preprocess_dataset,
    preprocess_interactions,
)


@pytest.fixture
def annotation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["TP53", "MDM2", "BRCA1", "BARD1", "XIST", "EGFR", "GRB2"],
            "gene_type": ["protein-coding"] * 4 + ["ncRNA"] + ["protein-coding"] * 2,
            "chromosome": ["17", "12", "17", "2", "X", "7", "17"],
            "description": [""] * 7,
        },
        index=pd.Index(["7157", "4193", "672", "580", "7503", "1956", "2885"], name="entrez_id"),
    )


@pytest.fixture
def raw() -> pd.DataFrame:
    rows = [
        (7157, 4193, 1.0),   # TP53-MDM2
        (4193, 7157, 1.0),   # reversed duplicate
        (7157, 7157, 1.0),   # homodimer
        (7157, 672, 1.0),    # TP53-BRCA1
        (672, 580, 1.0),     # BRCA1-BARD1
        (580, 7503, 1.0),    # BARD1-XIST (non-coding)
        (580, 999999, 1.0),  # unannotated identifier
        (1956, 2885, 1.0),   # EGFR-GRB2: a separate component
    ]
    return pd.DataFrame(rows, columns=["protein_a", "protein_b", "weight"])


def test_full_preprocessing(raw, annotation):
    edges, report = preprocess_interactions(raw, annotation)
    steps = report.to_frame()
    assert steps.index.tolist() == [
        "identifiers", "self-interactions", "duplicates", "annotation", "protein-coding", "largest component",
    ]
    removed = steps["interactions_removed"].to_dict()
    assert removed == {
        "identifiers": 0, "self-interactions": 1, "duplicates": 1,
        "annotation": 1, "protein-coding": 1, "largest component": 1,
    }
    assert {frozenset(p) for p in zip(edges.protein_a, edges.protein_b)} == {
        frozenset(("7157", "4193")), frozenset(("7157", "672")), frozenset(("672", "580")),
    }
    assert set(edges.symbol_a) | set(edges.symbol_b) == {"TP53", "MDM2", "BRCA1", "BARD1"}
    assert report.details["homodimer_proteins"] == 1
    assert report.details["unannotated_ids"] == ["999999"]
    assert report.details["gene_type_counts"]["ncRNA"] == 1
    assert report.details["final_interactions"] == 3
    # Steps chain: each step's input is the previous step's output.
    assert (steps["interactions_in"].iloc[1:].to_numpy() == steps["interactions_out"].iloc[:-1].to_numpy()).all()


def test_switches_and_no_annotation(raw):
    config = PreprocessingConfig(largest_component=False)
    edges, report = preprocess_interactions(raw, None, config)
    assert "annotation" not in report.to_frame().index
    assert "symbol_a" not in edges.columns
    assert len(edges) == 6  # only self-loop and duplicate removed


def test_confidence_threshold(annotation):
    raw = pd.DataFrame({"protein_a": ["7157", "672"], "protein_b": ["4193", "580"], "weight": [900, 200]})
    edges, report = preprocess_interactions(raw, annotation, PreprocessingConfig(min_score=500))
    assert report.to_frame().loc["confidence", "interactions_removed"] == 1
    assert len(edges) == 1


def test_missing_identifiers_are_dropped():
    raw = pd.DataFrame({"protein_a": ["A", None, " C "], "protein_b": ["B", "B", "A"], "weight": [1, 1, np.nan]})
    edges, report = preprocess_interactions(raw, config=PreprocessingConfig(largest_component=False))
    assert report.to_frame().loc["identifiers", "interactions_removed"] == 1
    assert set(edges.protein_a) | set(edges.protein_b) == {"A", "B", "C"}
    assert edges["weight"].notna().all()


def test_normalise_identifiers():
    assert normalise_identifiers(pd.Series([14.0, np.nan, 27.0])).tolist()[::2] == ["14", "27"]
    assert normalise_identifiers(pd.Series([" P1 ", "P2"])).tolist() == ["P1", "P2"]


def test_preprocess_dataset_roundtrip(tmp_path):
    source = tmp_path / "toy.txt"
    source.write_bytes(b"1\t2\t1\r2\t3\t1\r3\t3\t1\r3\t1\t1\r7\t8\t1\r")
    result = preprocess_dataset(str(source), tmp_path / "processed", annotate=False)
    assert result.edges_path.name == "toy.tsv"
    edges = pd.read_csv(result.edges_path, sep="\t", dtype=str)
    assert len(edges) == 3
    loaded = PreprocessingReport.load(result.report_path)
    assert loaded.to_frame().equals(result.report.to_frame())
    assert load_symbol_map(result.edges_path) == {}


def test_symbol_map_and_annotation_loader(tmp_path, raw, annotation):
    edges, _ = preprocess_interactions(raw, annotation)
    path = tmp_path / "edges.tsv"
    edges.to_csv(path, sep="\t", index=False)
    assert load_symbol_map(path) == {"7157": "TP53", "4193": "MDM2", "672": "BRCA1", "580": "BARD1"}

    genes = tmp_path / "genes.tsv"
    genes.write_text(
        "tax_id\tGeneID\tSymbol\tchromosome\tmap_location\ttype_of_gene\tdescription\n"
        "9606\t7157\tTP53\t17\t17p13.1\tprotein-coding\ttumor protein p53\n"
    )
    loaded = load_gene_annotation(genes)
    assert loaded.loc["7157", "symbol"] == "TP53"
    assert loaded.loc["7157", "gene_type"] == "protein-coding"
