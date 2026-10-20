"""Audited preprocessing of raw protein-protein interaction data.

Raw interactome files are rarely analysis-ready: they mix identifier types,
list homodimers as self-loops, repeat interactions in both orientations and
include loci that are not protein-coding genes. This module cleans an edge
table step by step and records how many interactions and proteins each step
removes, so every number downstream can be traced back to the raw file.

Steps (each can be switched off through :class:`PreprocessingConfig`):

1. **identifiers** - drop rows with missing identifiers; normalise IDs to
   clean strings (``14.0`` -> ``14``).
2. **confidence** - optional minimum confidence score.
3. **self-interactions** - remove homodimers (``A-A``). They are real
   biology, but a simple graph cannot represent or predict them.
4. **duplicates** - collapse ``A-B`` / ``B-A`` repeats into one interaction.
5. **annotation** - drop identifiers missing from the gene annotation
   (typically discontinued Entrez IDs).
6. **protein-coding** - keep interactions between protein-coding genes only.
7. **largest component** - keep the largest connected component, on which
   distances and centralities are well defined.

Usage::

    python src/preprocessing.py --dataset hi-iii
    python src/preprocessing.py --all
    python src/preprocessing.py --input my_edges.tsv --name my_network --no-annotation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/preprocessing.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import pandas as pd

from src.datasets import DATASETS, INTERACTOMES, dataset_path
from src.graph_builder import read_edge_table

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_COLUMNS = ["protein_a", "protein_b", "weight", "symbol_a", "symbol_b"]


@dataclass
class PreprocessingConfig:
    """Switches for the cleaning steps."""

    min_score: float | None = None
    drop_unannotated: bool = True
    protein_coding_only: bool = True
    largest_component: bool = True


@dataclass
class StepRecord:
    """Interactions and proteins before and after one cleaning step."""

    step: str
    description: str
    interactions_in: int
    interactions_out: int
    proteins_in: int
    proteins_out: int

    @property
    def interactions_removed(self) -> int:
        return self.interactions_in - self.interactions_out

    @property
    def proteins_removed(self) -> int:
        return self.proteins_in - self.proteins_out


@dataclass
class PreprocessingReport:
    """Audit trail of a preprocessing run."""

    source: str
    config: dict
    steps: list[StepRecord] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """One row per step with in/out/removed counts."""
        rows = [
            {
                **asdict(s),
                "interactions_removed": s.interactions_removed,
                "proteins_removed": s.proteins_removed,
            }
            for s in self.steps
        ]
        columns = [
            "step", "description", "interactions_in", "interactions_out", "interactions_removed",
            "proteins_in", "proteins_out", "proteins_removed",
        ]
        return pd.DataFrame(rows, columns=columns).set_index("step")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "config": self.config,
            "steps": self.to_frame().reset_index().to_dict(orient="records"),
            "details": self.details,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = [
            StepRecord(**{k: s[k] for k in StepRecord.__dataclass_fields__}) for s in data["steps"]
        ]
        return cls(source=data["source"], config=data["config"], steps=steps, details=data["details"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _n_proteins(edges: pd.DataFrame) -> int:
    return int(pd.unique(edges[["protein_a", "protein_b"]].to_numpy().ravel()).size) if len(edges) else 0


def _proteins(edges: pd.DataFrame) -> set[str]:
    return set(edges["protein_a"]) | set(edges["protein_b"])


def normalise_identifiers(ids: pd.Series) -> pd.Series:
    """Clean identifier strings; integer-valued floats lose their ``.0``."""
    if pd.api.types.is_float_dtype(ids):
        numeric = ids.dropna()
        if (numeric == numeric.round()).all():
            ids = ids.astype("Int64")
    return ids.astype("string").str.strip()


def load_gene_annotation(path: str | Path | None = None) -> pd.DataFrame:
    """Gene annotation indexed by Entrez ID (string).

    Columns: ``symbol``, ``gene_type``, ``chromosome``, ``description``.
    Downloads the bundled NCBI snapshot when ``path`` is ``None``.
    """
    path = Path(path) if path is not None else dataset_path("genes-human")
    genes = pd.read_csv(path, sep="\t", dtype={"GeneID": str})
    genes = genes.rename(
        columns={"GeneID": "entrez_id", "Symbol": "symbol", "type_of_gene": "gene_type"}
    )
    return genes.set_index("entrez_id")[["symbol", "gene_type", "chromosome", "description"]]


def load_symbol_map(path: str | Path) -> dict[str, str]:
    """Protein -> gene symbol mapping from a processed edge list (empty if absent)."""
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    if not {"symbol_a", "symbol_b"} <= set(header):
        return {}
    table = pd.read_csv(path, sep="\t", dtype=str, usecols=["protein_a", "protein_b", "symbol_a", "symbol_b"])
    mapping = dict(zip(table["protein_a"], table["symbol_a"]))
    mapping.update(zip(table["protein_b"], table["symbol_b"]))
    return {k: v for k, v in mapping.items() if isinstance(v, str)}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_interactions(
    raw: pd.DataFrame,
    annotation: pd.DataFrame | None = None,
    config: PreprocessingConfig | None = None,
    source: str = "",
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Clean a raw interaction table and record every step.

    Parameters
    ----------
    raw:
        Table with columns ``protein_a``, ``protein_b`` and ``weight``, e.g.
        from :func:`src.graph_builder.read_edge_table`.
    annotation:
        Optional output of :func:`load_gene_annotation`; enables the
        annotation and protein-coding steps and adds gene symbols.
    config:
        Step switches; defaults to :class:`PreprocessingConfig`.
    source:
        Label stored in the report.

    Returns
    -------
    (edges, report) where ``edges`` has columns ``protein_a``,
    ``protein_b``, ``weight`` (+ ``symbol_a``, ``symbol_b`` with an
    annotation), one row per undirected interaction.
    """
    config = config or PreprocessingConfig()
    report = PreprocessingReport(source=source, config=asdict(config))
    details = report.details
    edges = raw[["protein_a", "protein_b", "weight"]].copy()

    def record(step: str, description: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
        report.steps.append(
            StepRecord(step, description, len(before), len(after), _n_proteins(before), _n_proteins(after))
        )
        logger.debug("%-18s %6d -> %6d interactions", step, len(before), len(after))

    # 1. Identifiers
    before = edges
    edges = edges.assign(
        protein_a=normalise_identifiers(edges["protein_a"]),
        protein_b=normalise_identifiers(edges["protein_b"]),
    )
    edges = edges.dropna(subset=["protein_a", "protein_b"])
    edges = edges[(edges["protein_a"] != "") & (edges["protein_b"] != "")]
    edges = edges.astype({"protein_a": str, "protein_b": str})
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0).astype(float)
    record("identifiers", "drop rows with missing protein identifiers", before, edges)
    details["raw_rows"] = len(raw)

    # 2. Confidence threshold
    if config.min_score is not None:
        before = edges
        edges = edges[edges["weight"] >= config.min_score]
        record("confidence", f"keep interactions with score >= {config.min_score:g}", before, edges)

    # 3. Self-interactions
    before = edges
    is_self = edges["protein_a"] == edges["protein_b"]
    details["homodimer_proteins"] = int(edges.loc[is_self, "protein_a"].nunique())
    edges = edges[~is_self]
    record("self-interactions", "remove homodimers (A-A), not representable in a simple graph", before, edges)

    # 4. Duplicates
    before = edges
    swap = edges["protein_a"] > edges["protein_b"]
    edges = edges.assign(
        protein_a=edges["protein_a"].where(~swap, edges["protein_b"]),
        protein_b=edges["protein_b"].where(~swap, edges["protein_a"]),
    )
    edges = edges.groupby(["protein_a", "protein_b"], as_index=False, sort=True)["weight"].max()
    record("duplicates", "collapse repeated and reversed (B-A) interactions", before, edges)

    # 5-6. Annotation and gene types
    if annotation is not None:
        proteins = _proteins(edges)
        annotated = proteins & set(annotation.index)
        unannotated = sorted(proteins - annotated)
        gene_types = annotation["gene_type"].reindex(sorted(proteins)).fillna("unannotated")
        details["gene_type_counts"] = {k: int(v) for k, v in gene_types.value_counts().items()}
        details["unannotated_ids"] = unannotated

        if config.drop_unannotated:
            before = edges
            keep = edges["protein_a"].isin(annotated) & edges["protein_b"].isin(annotated)
            edges = edges[keep]
            record("annotation", "drop identifiers absent from the gene annotation", before, edges)

        if config.protein_coding_only:
            before = edges
            coding = set(annotation.index[annotation["gene_type"] == "protein-coding"])
            keep = edges["protein_a"].isin(coding) & edges["protein_b"].isin(coding)
            edges = edges[keep]
            record("protein-coding", "keep interactions between protein-coding genes", before, edges)

    # 7. Largest connected component
    if config.largest_component and len(edges):
        before = edges
        graph = nx.from_pandas_edgelist(edges, "protein_a", "protein_b")
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        details["components_before_lcc"] = len(components)
        giant = components[0]
        edges = edges[edges["protein_a"].isin(giant)]  # both ends share a component
        record("largest component", "keep the largest connected component", before, edges)

    edges = edges.sort_values(["protein_a", "protein_b"]).reset_index(drop=True)
    if annotation is not None:
        symbols = annotation["symbol"]
        edges["symbol_a"] = edges["protein_a"].map(symbols)
        edges["symbol_b"] = edges["protein_b"].map(symbols)
    details["final_interactions"] = len(edges)
    details["final_proteins"] = _n_proteins(edges)
    return edges, report


@dataclass
class ProcessedDataset:
    """Paths written by :func:`preprocess_dataset`."""

    name: str
    edges_path: Path
    report_path: Path
    report: PreprocessingReport


def preprocess_dataset(
    dataset: str,
    output_dir: str | Path = PROCESSED_DIR,
    config: PreprocessingConfig | None = None,
    annotate: bool = True,
    name: str | None = None,
) -> ProcessedDataset:
    """Download (if needed), preprocess and save a dataset.

    Parameters
    ----------
    dataset:
        A registry name from :mod:`src.datasets` (e.g. ``"hi-iii"``) or a
        path to an edge-list file.
    output_dir:
        Where ``<name>.tsv`` and ``<name>.report.json`` are written.
    annotate:
        Map Entrez IDs to gene symbols and apply the annotation steps.
    name:
        Output name; defaults to the registry name or the file stem.
    """
    if dataset in DATASETS:
        raw_path = dataset_path(dataset)
        name = name or dataset
    else:
        raw_path = Path(dataset)
        name = name or raw_path.name.split(".")[0]

    raw = read_edge_table(raw_path)
    annotation = load_gene_annotation() if annotate else None
    edges, report = preprocess_interactions(raw, annotation, config, source=str(raw_path.name))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / f"{name}.tsv"
    edges.to_csv(edges_path, sep="\t", index=False)
    report_path = report.save(output_dir / f"{name}.report.json")
    logger.info(
        "%s: %d raw rows -> %d interactions among %d proteins (%s)",
        name,
        report.details["raw_rows"],
        report.details["final_interactions"],
        report.details["final_proteins"],
        edges_path,
    )
    return ProcessedDataset(name, edges_path, report_path, report)


def processed_path(name: str, output_dir: str | Path = PROCESSED_DIR, create: bool = True) -> Path:
    """Path of a processed registry dataset, preprocessing it on first use."""
    path = Path(output_dir) / f"{name}.tsv"
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"{path} not found; run `python src/preprocessing.py --dataset {name}`")
        preprocess_dataset(name, output_dir)
    return path


def main(argv: list[str] | None = None) -> list[ProcessedDataset]:
    parser = argparse.ArgumentParser(description="Preprocess raw PPI data with an audit report.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", nargs="+", choices=INTERACTOMES, help="registry dataset(s)")
    source.add_argument("--all", action="store_true", help="every registry interactome")
    source.add_argument("--input", type=Path, help="any edge-list file")
    parser.add_argument("--name", help="output name for --input")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--no-annotation", action="store_true", help="skip gene annotation steps")
    parser.add_argument("--keep-unannotated", action="store_true")
    parser.add_argument("--keep-non-coding", action="store_true")
    parser.add_argument("--all-components", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = PreprocessingConfig(
        min_score=args.min_score,
        drop_unannotated=not args.keep_unannotated,
        protein_coding_only=not args.keep_non_coding,
        largest_component=not args.all_components,
    )
    targets = [str(args.input)] if args.input else (list(INTERACTOMES) if args.all else args.dataset)
    results = []
    for target in targets:
        result = preprocess_dataset(
            target, args.output_dir, config, annotate=not args.no_annotation, name=args.name
        )
        print(f"\n{result.name}  ({result.report.source})")
        print(result.report.to_frame().drop(columns="description").to_string())
        results.append(result)
    return results


if __name__ == "__main__":
    main()
