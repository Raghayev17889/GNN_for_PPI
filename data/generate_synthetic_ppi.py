"""Generate a synthetic protein-protein interaction (PPI) network.

The network is grown with the *duplication-divergence* model, the standard
mechanistic model of PPI network evolution (Vazquez et al., 2003; Ispolatov
et al., 2005):

1. **Duplication** - a random protein is copied (gene duplication); the
   paralog inherits all of the original's interaction partners.
2. **Divergence** - each inherited interaction is kept only with probability
   ``retention``, mimicking the loss of redundant binding sites.
3. **Dimerisation** - with probability ``dimerisation`` the paralog also
   binds its ancestor (homodimers that become heterodimers).

Real interaction screens also contain spurious hits, so a fraction of random
"noise" edges is added at the end. Every edge receives a confidence score in
``[0, 1000]`` (the scale used by STRING): biological edges are drawn from a
high-confidence Beta distribution and noise edges from a low-confidence one,
so score thresholds behave like they do on real data.

The result is a sparse, heavy-tailed, small-world graph whose paralogs share
partners without necessarily interacting themselves - the property that makes
paths of length three (L3) informative for PPI prediction (Kovacs et al.,
*Nature Communications*, 2019).

Usage
-----
    python data/generate_synthetic_ppi.py
    python data/generate_synthetic_ppi.py --n-proteins 5000 --seed 7 \
        --output data/raw/synthetic_ppi_5k.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "raw" / "sample_ppi_edges.tsv"


def duplication_divergence_edges(
    n_proteins: int = 2000,
    retention: float = 0.4,
    dimerisation: float = 0.15,
    seed_size: int = 8,
    seed: int = 42,
) -> set[tuple[int, int]]:
    """Grow an undirected graph with the duplication-divergence model.

    Parameters
    ----------
    n_proteins:
        Number of proteins (nodes) in the final network.
    retention:
        Probability that an inherited interaction survives divergence.
    dimerisation:
        Probability that a new paralog interacts with its ancestor.
    seed_size:
        Size of the initial fully connected seed complex.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    set of (u, v) integer pairs with ``u < v``.
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must be in (0, 1]")
    if not 0.0 <= dimerisation <= 1.0:
        raise ValueError("dimerisation must be in [0, 1]")
    if seed_size < 2 or n_proteins < seed_size:
        raise ValueError("need 2 <= seed_size <= n_proteins")

    rng = np.random.default_rng(seed)
    neighbours: list[set[int]] = [set() for _ in range(seed_size)]
    for u in range(seed_size):
        for v in range(u + 1, seed_size):
            neighbours[u].add(v)
            neighbours[v].add(u)

    while len(neighbours) < n_proteins:
        ancestor = int(rng.integers(len(neighbours)))
        inherited = [w for w in sorted(neighbours[ancestor]) if rng.random() < retention]
        dimer = rng.random() < dimerisation
        if not inherited and not dimer:
            # A paralog that lost every interaction is invisible in a PPI
            # screen; discard it and try another duplication.
            continue

        new = len(neighbours)
        neighbours.append(set(inherited))
        for w in inherited:
            neighbours[w].add(new)
        if dimer:
            neighbours[new].add(ancestor)
            neighbours[ancestor].add(new)

    return {(u, v) for u, nbrs in enumerate(neighbours) for v in nbrs if u < v}


def add_noise_edges(
    edges: set[tuple[int, int]],
    n_proteins: int,
    noise_fraction: float,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    """Return ``noise_fraction * len(edges)`` random edges absent from ``edges``."""
    n_noise = int(round(noise_fraction * len(edges)))
    noise: set[tuple[int, int]] = set()
    while len(noise) < n_noise:
        u, v = (int(x) for x in rng.integers(n_proteins, size=2))
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair not in edges:
            noise.add(pair)
    return noise


def generate_synthetic_ppi(
    n_proteins: int = 2000,
    retention: float = 0.4,
    dimerisation: float = 0.15,
    noise_fraction: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a scored synthetic PPI edge list.

    Returns
    -------
    DataFrame with columns ``protein_a``, ``protein_b``, ``score`` and
    ``source`` (``"biological"`` or ``"noise"``).
    """
    rng = np.random.default_rng(seed)
    true_edges = duplication_divergence_edges(
        n_proteins=n_proteins, retention=retention, dimerisation=dimerisation, seed=seed
    )
    noise_edges = add_noise_edges(true_edges, n_proteins, noise_fraction, rng)

    # Shuffle protein identifiers so that IDs carry no information about the
    # order in which proteins were added during network growth.
    permutation = rng.permutation(n_proteins)
    width = max(5, len(str(n_proteins)))
    names = np.array([f"P{int(i):0{width}d}" for i in permutation])

    rows = []
    for label, edge_set, (a, b) in (
        ("biological", true_edges, (6.0, 2.0)),
        ("noise", noise_edges, (2.0, 5.0)),
    ):
        ordered = sorted(edge_set)
        scores = np.clip(np.round(rng.beta(a, b, size=len(ordered)) * 1000), 150, 999)
        for (u, v), score in zip(ordered, scores):
            rows.append((names[u], names[v], int(score), label))

    df = pd.DataFrame(rows, columns=["protein_a", "protein_b", "score", "source"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-proteins", type=int, default=2000)
    parser.add_argument("--retention", type=float, default=0.4)
    parser.add_argument("--dimerisation", type=float, default=0.15)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = generate_synthetic_ppi(
        n_proteins=args.n_proteins,
        retention=args.retention,
        dimerisation=args.dimerisation,
        noise_fraction=args.noise_fraction,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    n_nodes = pd.unique(df[["protein_a", "protein_b"]].values.ravel()).size
    print(f"Wrote {len(df):,} interactions among {n_nodes:,} proteins to {args.output}")


if __name__ == "__main__":
    main()
