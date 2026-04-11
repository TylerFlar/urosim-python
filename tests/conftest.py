from collections.abc import Callable

import numpy as np
import pytest

from urosim.anatomy.centerlines import generate_centerlines
from urosim.anatomy.placement import place_nodes_3d
from urosim.anatomy.topology import build_topology


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
