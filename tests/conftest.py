from collections.abc import Callable

import numpy as np
import pytest
import trimesh

from urosim.anatomy.centerlines import generate_centerlines
from urosim.anatomy.mesh_extract import extract_kidney_mesh
from urosim.anatomy.placement import place_nodes_3d
from urosim.anatomy.topology import build_topology

_FAST_VOXEL_SIZE: float = 1.0
_FAST_TARGET_EDGE: float = 2.0
_DEFAULT_SEED: int = 12345


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unity: marks tests that require a running Unity simulator (deselect with -m 'not unity')",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks slow tests (e.g. full-resolution mesh extraction)",
    )


@pytest.fixture(scope="session")
def pelvis_radii() -> tuple[float, float, float]:
    """Canonical pelvis ellipsoid half-axes used across anatomy tests."""
    return (20.0, 15.0, 10.0)


@pytest.fixture
def sample_kidney() -> Callable[..., tuple[object, dict]]:
    """Factory fixture returning a deterministic kidney graph + centerlines.

    Returns a callable with signature
    ``(pelvis_type: str = "A2", seed: int = 12345) -> (graph, centerlines)``
    so individual tests can choose their own topology type and seed
    while sharing the construction boilerplate.
    """

    def _build(
        pelvis_type: str = "A2", seed: int = 12345
    ) -> tuple[object, dict]:
        rng = np.random.default_rng(seed)
        graph = build_topology(pelvis_type, rng)
        place_nodes_3d(graph, rng)
        centerlines = generate_centerlines(graph, rng)
        return graph, centerlines

    return _build


@pytest.fixture(scope="session")
def sample_kidney_graph_fast(
    pelvis_radii: tuple[float, float, float],  # noqa: ARG001 — parameter kept for fixture ordering.
) -> tuple[object, dict]:
    """Session-scoped ``(graph, centerlines)`` matching :func:`kidney_mesh_fast`.

    Exposed via ``conftest.py`` so every anatomy test file can reuse
    the same deterministic build (seed 12345, pelvis type A2) without
    rebuilding the graph per test.
    """
    rng = np.random.default_rng(_DEFAULT_SEED)
    graph = build_topology("A2", rng)
    place_nodes_3d(graph, rng)
    centerlines = generate_centerlines(graph, rng)
    return graph, centerlines


@pytest.fixture(scope="session")
def kidney_mesh_fast(
    pelvis_radii: tuple[float, float, float],
) -> trimesh.Trimesh:
    """Session-scoped kidney mesh extracted at a coarse resolution.

    Extracted once per pytest session at ``voxel_size=1.0`` so the
    expensive SDF evaluation and remeshing pipeline is shared across
    every anatomy test file (normals, texture_coords,
    coverage_points, stones, mesh, …).
    """
    rng = np.random.default_rng(_DEFAULT_SEED)
    graph = build_topology("A2", rng)
    place_nodes_3d(graph, rng)
    centerlines = generate_centerlines(graph, rng)
    return extract_kidney_mesh(
        graph,
        centerlines,
        pelvis_radii,
        voxel_size=_FAST_VOXEL_SIZE,
        target_edge_length=_FAST_TARGET_EDGE,
    )
