"""Shared pytest fixtures."""

from __future__ import annotations

import networkx as nx
import pytest


@pytest.fixture
def karate_graph() -> nx.Graph:
    """Zachary's karate club with string protein-like labels."""
    return nx.relabel_nodes(nx.karate_club_graph(), lambda n: f"P{n:03d}")


@pytest.fixture
def small_ppi_graph() -> nx.Graph:
    """A small connected, clustered, heavy-tailed graph (300 proteins)."""
    graph = nx.powerlaw_cluster_graph(300, 3, 0.2, seed=7)
    return nx.relabel_nodes(graph, lambda n: f"P{n:04d}")
