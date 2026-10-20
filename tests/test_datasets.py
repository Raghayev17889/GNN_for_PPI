from __future__ import annotations

import hashlib
import re

import pytest

from src.datasets import DATASETS, INTERACTOMES, Dataset, dataset_path, download, get_dataset, sha256sum


def _local_dataset(tmp_path, content: bytes, checksum: str | None = None) -> Dataset:
    source = tmp_path / "source.txt"
    source.write_bytes(content)
    return Dataset(
        name="toy",
        filename="toy.txt",
        url=source.as_uri(),
        sha256=checksum or hashlib.sha256(content).hexdigest(),
        kind="binary",
        description="toy network",
        reference="test",
    )


def test_registry_is_pinned_and_checksummed():
    assert {"hi-iii", "hi-ii-14", "lit-bm-13", "bioplex-2", "genes-human"} <= set(DATASETS)
    assert "genes-human" not in INTERACTOMES
    for dataset in DATASETS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", dataset.sha256), dataset.name
        assert re.search(r"/[0-9a-f]{40}/", dataset.url), f"{dataset.name} URL is not commit-pinned"
        assert dataset.kind in {"binary", "co-complex", "mixed", "annotation"}


def test_get_dataset_unknown():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("string-v99")


def test_download_verifies_and_caches(tmp_path):
    dataset = _local_dataset(tmp_path, b"1\t2\t1\r3\t4\t1\r")
    out = tmp_path / "external"
    path = download(dataset, out)
    assert path.read_bytes() == b"1\t2\t1\r3\t4\t1\r"
    assert sha256sum(path) == dataset.sha256

    # A cached file with the right checksum is reused without touching the URL.
    (tmp_path / "source.txt").unlink()
    assert download(dataset, out) == path


def test_download_rejects_bad_checksum(tmp_path):
    dataset = _local_dataset(tmp_path, b"1\t2\t1\n", checksum="0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download(dataset, tmp_path / "external")
    assert not list((tmp_path / "external").glob("*"))  # no partial files left behind


def test_dataset_path_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_path("hi-iii", data_dir=tmp_path, download_missing=False)
