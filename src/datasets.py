"""Registry and downloader for public human protein interaction datasets.

The interactome files are the Entrez-gene-ID networks distributed with
Kovacs et al. (2019), the paper that introduced the L3 link-prediction
principle; gene annotation comes from an NCBI Entrez Gene snapshot, and
functional annotation (Gene Ontology, pathways, tissue expression) from the
Hetionet v1.0 knowledge graph, which also uses Entrez gene IDs. Every
URL is pinned to a fixed commit and every file is verified against a SHA-256
checksum, so downloads are reproducible.

Usage::

    python src/datasets.py list
    python src/datasets.py download hi-iii genes-human
    python src/datasets.py download hetionet-nodes hetionet-edges
    python src/datasets.py download --all
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/datasets.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "data" / "external"

_L3_COMMIT = "ed6b18f281521759bd70c323b802559fba81e789"
_L3_BASE = f"https://raw.githubusercontent.com/kpisti/L3/{_L3_COMMIT}/"
_GENES_COMMIT = "a7362748a34211e5df6f2d185bb3246279760546"
_GENES_BASE = f"https://raw.githubusercontent.com/dhimmel/entrez-gene/{_GENES_COMMIT}/data/"
_HETIONET_COMMIT = "8a6cc0c5604d0e2908786e4de38af67b8e46ee4a"
_HETIONET_RAW = f"https://raw.githubusercontent.com/hetio/hetionet/{_HETIONET_COMMIT}/hetnet/tsv/"
# The edge file is stored with Git LFS and served from GitHub's media host.
_HETIONET_LFS = f"https://media.githubusercontent.com/media/hetio/hetionet/{_HETIONET_COMMIT}/hetnet/tsv/"
_HETIONET = "Himmelstein et al., eLife 6, e26726 (2017); github.com/hetio/hetionet (CC0)"
_KOVACS_2019 = "Kovacs et al., Nature Communications 10, 1240 (2019); github.com/kpisti/L3"


@dataclass(frozen=True)
class Dataset:
    """A downloadable resource.

    Attributes
    ----------
    name:
        Short identifier used on the command line.
    filename:
        File name under ``data/external/``.
    url:
        Commit-pinned download URL.
    sha256:
        Expected checksum of the downloaded file.
    kind:
        ``binary`` (direct contacts from yeast two-hybrid or curated binary
        evidence), ``co-complex`` (affinity purification-mass spectrometry)
        or ``annotation``.
    description, reference:
        Human-readable provenance.
    """

    name: str
    filename: str
    url: str
    sha256: str
    kind: str
    description: str
    reference: str


DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset(
            "hi-iii",
            "hi-iii.txt",
            _L3_BASE + "hi-iii.txt",
            "1e3dfd31ed1b965e1c0aa66303cdbb712e7926c34040c5069b5f7a59d0839437",
            "binary",
            "HI-III: third-generation CCSB systematic yeast two-hybrid screen of the human "
            "interactome (pre-publication version of HuRI)",
            "Luck et al., Nature 580, 402-408 (2020); file from " + _KOVACS_2019,
        ),
        Dataset(
            "hi-ii-14",
            "hi-ii-14.txt",
            _L3_BASE + "hi-ii-14.txt",
            "904c0e9710017ed916e9cee15143f582b4f1aa04695446bae548017629d0238a",
            "binary",
            "HI-II-14: second-generation CCSB systematic yeast two-hybrid human interactome",
            "Rolland et al., Cell 159, 1212-1226 (2014); file from " + _KOVACS_2019,
        ),
        Dataset(
            "lit-bm-13",
            "lit-bm-13.txt",
            _L3_BASE + "lit-bm-13.txt",
            "56eddaf33e74c42f8429c27938e1c2406b771231604829e5fdc4c51c188b8690",
            "binary",
            "Lit-BM-13: literature-curated binary interactions supported by multiple pieces of evidence",
            "Rolland et al., Cell 159, 1212-1226 (2014); file from " + _KOVACS_2019,
        ),
        Dataset(
            "bioplex-2",
            "bioplex_2.txt",
            _L3_BASE + "bioplex_2.txt",
            "98120a4efbb159b6008d81a403a826867e4a67579da237a77829d7a1510b9395",
            "co-complex",
            "BioPlex 2.0: affinity purification-mass spectrometry co-complex network in HEK293T cells",
            "Huttlin et al., Nature 545, 505-509 (2017); file from " + _KOVACS_2019,
        ),
        Dataset(
            "biogrid",
            "biogrid_all.txt",
            _L3_BASE + "biogrid_all.txt",
            "a47ea7279abd7593935bbf39aeaabde8a3b3bb1db45468352844064531f525a9",
            "mixed",
            "BioGRID human physical interactions (all experimental systems)",
            "BioGRID (thebiogrid.org); file from " + _KOVACS_2019,
        ),
        Dataset(
            "genes-human",
            "genes-human.tsv",
            _GENES_BASE + "genes-human.tsv",
            "90405318c03fab8fa0411e9f5d09ad5763f14b8cc64bb20f10f32850d626e8e6",
            "annotation",
            "NCBI Entrez Gene human gene information (symbol, gene type, chromosome), 2016 snapshot",
            "NCBI Gene; processed by D. Himmelstein, github.com/dhimmel/entrez-gene (CC0)",
        ),
        Dataset(
            "hetionet-nodes",
            "hetionet-v1.0-nodes.tsv",
            _HETIONET_RAW + "hetionet-v1.0-nodes.tsv",
            "3ad9eb86c031df8bc7d5efefa6edc93562182eb9a6a71e5129d3f18b9d42c50e",
            "annotation",
            "Hetionet v1.0 nodes: names of genes, GO terms, pathways and anatomical structures",
            _HETIONET,
        ),
        Dataset(
            "hetionet-edges",
            "hetionet-v1.0-edges.sif.gz",
            _HETIONET_LFS + "hetionet-v1.0-edges.sif.gz",
            "611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947",
            "annotation",
            "Hetionet v1.0 edges: gene-GO, gene-pathway and tissue-expresses-gene relations",
            _HETIONET,
        ),
    )
}

INTERACTOMES = tuple(name for name, d in DATASETS.items() if d.kind != "annotation")


def sha256sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; available: {', '.join(DATASETS)}") from None


def _fetch(url: str, destination: Path, timeout: float, retries: int) -> None:
    """Stream ``url`` into ``destination``, retrying transient network errors."""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(destination, "wb") as out:
                shutil.copyfileobj(response, out)
            return
        except OSError as exc:  # URLError, connection resets, timeouts
            if attempt == retries:
                raise
            wait = 2**attempt
            logger.warning("Download failed (%s); retrying in %ds (%d/%d)", exc, wait, attempt, retries)
            time.sleep(wait)


def download(
    dataset: str | Dataset,
    data_dir: str | Path = EXTERNAL_DIR,
    force: bool = False,
    timeout: float = 60.0,
    retries: int = 3,
) -> Path:
    """Download a dataset (if needed) and verify its checksum.

    An existing file with the right checksum is reused. The download goes to
    a temporary file first, so an interrupted transfer never leaves a corrupt
    file behind.

    Returns
    -------
    Path to the verified local file.
    """
    dataset = get_dataset(dataset) if isinstance(dataset, str) else dataset
    target = Path(data_dir) / dataset.filename
    if target.exists() and not force:
        if sha256sum(target) == dataset.sha256:
            logger.debug("Using cached %s", target)
            return target
        logger.warning("Checksum mismatch for cached %s; downloading again", target)

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", dataset.name, dataset.url)
    fd, name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(fd)
    tmp_path = Path(name)
    try:
        _fetch(dataset.url, tmp_path, timeout, retries)
        actual = sha256sum(tmp_path)
        if actual != dataset.sha256:
            raise ValueError(
                f"Checksum mismatch for {dataset.name}: expected {dataset.sha256}, got {actual}"
            )
        tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)
    target.chmod(0o644)
    return target


def dataset_path(name: str, data_dir: str | Path = EXTERNAL_DIR, download_missing: bool = True) -> Path:
    """Local path of a dataset, downloading it on first use."""
    dataset = get_dataset(name)
    path = Path(data_dir) / dataset.filename
    if download_missing:
        return download(dataset, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python src/datasets.py download {name}`")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List or download PPI datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show available datasets")
    get = sub.add_parser("download", help="download and verify datasets")
    get.add_argument("names", nargs="*", help="dataset names (see `list`)")
    get.add_argument("--all", action="store_true", help="download every dataset")
    get.add_argument("--data-dir", type=Path, default=EXTERNAL_DIR)
    get.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "list":
        for d in DATASETS.values():
            print(f"{d.name:12s} {d.kind:11s} {d.description}\n{'':24s} {d.reference}")
        return
    names = list(DATASETS) if args.all else args.names
    if not names:
        parser.error("give dataset names or --all")
    for name in names:
        path = download(name, args.data_dir, force=args.force)
        print(f"{name:12s} -> {path}")


if __name__ == "__main__":
    main()
