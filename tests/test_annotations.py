"""Tests for functional annotation loading and enrichment statistics."""

from __future__ import annotations

import gzip

import numpy as np
import pytest
from scipy import stats

from src.annotations import (
    benjamini_hochberg,
    coverage,
    enrichment,
    gene_to_terms,
    load_annotations,
    shared_annotation_rate,
)


@pytest.fixture
def hetionet_dir(tmp_path):
    """A miniature Hetionet with two GO terms sharing a name, a pathway and a tissue."""
    (tmp_path / "hetionet-v1.0-nodes.tsv").write_text(
        "id\tname\tkind\n"
        "Gene::1\tA\tGene\nGene::2\tB\tGene\nGene::3\tC\tGene\n"
        "Biological Process::GO:1\tsignalling\tBiological Process\n"
        "Biological Process::GO:2\tsignalling\tBiological Process\n"
        "Pathway::PC7_1\tglycolysis\tPathway\n"
        "Anatomy::UBERON:1\tbrain\tAnatomy\n"
        "Molecular Function::GO:5\tprotein binding\tMolecular Function\n"
    )
    rows = [
        ("Gene::1", "GpBP", "Biological Process::GO:1"),
        ("Gene::2", "GpBP", "Biological Process::GO:1"),
        ("Gene::3", "GpBP", "Biological Process::GO:2"),
        ("Gene::1", "GpPW", "Pathway::PC7_1"),
        ("Anatomy::UBERON:1", "AeG", "Gene::2"),
        ("Gene::1", "GpMF", "Molecular Function::GO:5"),
    ]
    with gzip.open(tmp_path / "hetionet-v1.0-edges.sif.gz", "wt") as handle:
        handle.write("source\tmetaedge\ttarget\n")
        handle.writelines("\t".join(r) + "\n" for r in rows)
    return tmp_path


def test_load_annotations_parses_sources(hetionet_dir):
    annotations = load_annotations(data_dir=hetionet_dir, download_missing=False)
    assert set(annotations) == {"go_bp", "go_cc", "pathway", "tissue"}
    # Two terms with the same name are kept apart by their identifiers.
    assert annotations["go_bp"] == {
        "signalling (GO:1)": frozenset({"1", "2"}),
        "signalling (GO:2)": frozenset({"3"}),
    }
    assert annotations["pathway"] == {"glycolysis": frozenset({"1"})}
    # Anatomy -> gene edges are oriented towards the gene.
    assert annotations["tissue"] == {"brain": frozenset({"2"})}
    assert annotations["go_cc"] == {}


def test_load_annotations_rejects_unknown_source(hetionet_dir):
    with pytest.raises(KeyError):
        load_annotations(["go_mf"], data_dir=hetionet_dir, download_missing=False)


def test_coverage_and_inversion():
    annotations = {"a": {"t1": frozenset({"1", "2"})}, "b": {"t2": frozenset({"3"})}}
    assert coverage(["1", "2", "3", "4"], annotations).to_dict() == {"a": 0.5, "b": 0.25}
    assert gene_to_terms(annotations["a"]) == {"1": {"t1"}, "2": {"t1"}}


def test_benjamini_hochberg_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    # Sorted: 0.01, 0.03, 0.04, 0.2 -> 0.04, 0.0533, 0.0533, 0.2
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert benjamini_hochberg(np.array([])).size == 0


def test_enrichment_hypergeometric_p_value():
    background = [str(i) for i in range(100)]
    gene_sets = {"hit": {str(i) for i in range(10)}, "other": {str(i) for i in range(50, 60)}}
    table = enrichment([str(i) for i in range(8)] + ["90", "91"], gene_sets, background, min_size=5)
    top = table.iloc[0]
    assert top["term"] == "hit" and top["overlap"] == 8
    assert top["p_value"] == pytest.approx(stats.hypergeom.sf(7, 100, 10, 10))
    assert top["fold_enrichment"] == pytest.approx(8.0)
    assert "other" not in set(table["term"])  # no overlap


def test_enrichment_respects_size_limits():
    background = [str(i) for i in range(20)]
    table = enrichment(["1"], {"tiny": {"1"}}, background, min_size=2)
    assert table.empty


def test_shared_annotation_rate_ignores_unannotated_pairs():
    gene_sets = {"t": {"1", "2"}, "u": {"3"}}
    pairs = [("1", "2"), ("1", "3"), ("1", "9")]
    assert shared_annotation_rate(pairs, gene_sets) == pytest.approx(0.5)
    assert np.isnan(shared_annotation_rate([("8", "9")], gene_sets))
