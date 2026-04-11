"""Tests for ``urosim.anatomy.placement`` and ``urosim.anatomy.centerlines``.

Covers correctness, spatial-anatomy sanity, min-distance enforcement,
centerline smoothness, and determinism of
:func:`~urosim.anatomy.placement.place_nodes_3d` and
:func:`~urosim.anatomy.centerlines.generate_centerlines`. All tests are
deterministic: every rng is created fresh from a known seed via the
local ``_rng`` helper.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from urosim.anatomy.centerlines import generate_centerlines
from urosim.anatomy.placement import place_nodes_3d
from urosim.anatomy.topology import build_topology

SEED = 12345

PELVIS_TYPES: tuple[str, ...] = ("A1", "A2", "A3", "B1", "B2")


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_and_place(
    pelvis_type: str = "A2", seed: int = SEED
) -> nx.DiGraph:
    graph = build_topology(pelvis_type, _rng(seed))
    place_nodes_3d(graph, _rng(seed + 1))
    return graph


# ---------------------------------------------------------------------------
# 1. Every node gets a position attribute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_all_nodes_have_position(pelvis_type: str) -> None:
    graph = _build_and_place(pelvis_type)
    for node in graph.nodes:
        assert "position" in graph.nodes[node], f"{node} is missing position"


# ---------------------------------------------------------------------------
# 2. Positions are shape (3,) float ndarrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_positions_are_3d_float(pelvis_type: str) -> None:
    graph = _build_and_place(pelvis_type)
    for node, data in graph.nodes(data=True):
        pos = data["position"]
        assert isinstance(pos, np.ndarray), f"{node}: {type(pos).__name__}"
        assert pos.shape == (3,), f"{node}: shape {pos.shape}"
        assert np.issubdtype(pos.dtype, np.floating), f"{node}: dtype {pos.dtype}"
        assert np.all(np.isfinite(pos)), f"{node}: non-finite position"


# ---------------------------------------------------------------------------
# 3. Upper-pole minors have higher x than lower-pole minors on average
# ---------------------------------------------------------------------------


def test_upper_pole_higher_than_lower() -> None:
    upper_means: list[float] = []
    lower_means: list[float] = []
    for seed in range(30):
        graph = build_topology("A2", _rng(seed))
        place_nodes_3d(graph, _rng(seed + 1000))

        upper_xs = [
            float(graph.nodes[n]["position"][0])
            for n, d in graph.nodes(data=True)
            if d.get("type") == "minor_calyx" and d.get("pole") == "upper"
        ]
        lower_xs = [
            float(graph.nodes[n]["position"][0])
            for n, d in graph.nodes(data=True)
            if d.get("type") == "minor_calyx" and d.get("pole") == "lower"
        ]
        if upper_xs:
            upper_means.append(float(np.mean(upper_xs)))
        if lower_xs:
            lower_means.append(float(np.mean(lower_xs)))

    mean_upper = float(np.mean(upper_means))
    mean_lower = float(np.mean(lower_means))
    assert mean_upper > mean_lower, (
        f"upper pole mean x ({mean_upper:.2f}) should exceed "
        f"lower pole mean x ({mean_lower:.2f})"
    )


# ---------------------------------------------------------------------------
# 4. No two nodes coincident (< 2 mm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_no_coincident_nodes(pelvis_type: str) -> None:
    for seed in range(20):
        graph = build_topology(pelvis_type, _rng(seed))
        place_nodes_3d(graph, _rng(seed + 1))

        nodes = list(graph.nodes)
        positions = np.stack([graph.nodes[n]["position"] for n in nodes])
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                # Allow a small numerical tolerance around the 2 mm bound.
                assert dist >= 1.99, (
                    f"seed={seed} {pelvis_type}: "
                    f"{nodes[i]} and {nodes[j]} are {dist:.4f} mm apart"
                )


# ---------------------------------------------------------------------------
# 5. Centerlines exist for every edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_centerlines_cover_all_edges(pelvis_type: str) -> None:
    graph = _build_and_place(pelvis_type)
    centerlines = generate_centerlines(graph, _rng(SEED + 2))
    assert set(centerlines.keys()) == set(graph.edges())
    for edge, curve in centerlines.items():
        assert isinstance(curve, np.ndarray), f"{edge}: {type(curve).__name__}"
        assert curve.ndim == 2 and curve.shape[1] == 3, f"{edge}: shape {curve.shape}"
        assert curve.shape[0] >= 20, f"{edge}: only {curve.shape[0]} samples"


# ---------------------------------------------------------------------------
# 6. Centerline endpoints match node positions (within 5 mm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_centerline_endpoints_match(pelvis_type: str) -> None:
    graph = _build_and_place(pelvis_type)
    centerlines = generate_centerlines(graph, _rng(SEED + 2))

    for (u, v), curve in centerlines.items():
        pos_u = graph.nodes[u]["position"]
        pos_v = graph.nodes[v]["position"]
        start_err = float(np.linalg.norm(curve[0] - pos_u))
        end_err = float(np.linalg.norm(curve[-1] - pos_v))
        assert start_err < 5.0, f"({u},{v}): start off by {start_err:.4f} mm"
        assert end_err < 5.0, f"({u},{v}): end off by {end_err:.4f} mm"


# ---------------------------------------------------------------------------
# 7. Centerlines have no large jumps between consecutive samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_no_large_jumps_in_centerlines(pelvis_type: str) -> None:
    for seed in range(10):
        graph = build_topology(pelvis_type, _rng(seed))
        place_nodes_3d(graph, _rng(seed + 1))
        centerlines = generate_centerlines(graph, _rng(seed + 2))

        for edge, curve in centerlines.items():
            steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            max_step = float(steps.max())
            assert max_step < 10.0, (
                f"seed={seed} {pelvis_type} edge={edge}: "
                f"max step {max_step:.4f} mm"
            )


# ---------------------------------------------------------------------------
# 8. Placement is deterministic for the same rng state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_placement_determinism(pelvis_type: str) -> None:
    g1 = build_topology(pelvis_type, _rng())
    g2 = build_topology(pelvis_type, _rng())
    place_nodes_3d(g1, _rng(SEED + 1))
    place_nodes_3d(g2, _rng(SEED + 1))

    assert set(g1.nodes) == set(g2.nodes)
    for node in g1.nodes:
        np.testing.assert_array_equal(
            g1.nodes[node]["position"],
            g2.nodes[node]["position"],
            err_msg=f"position differs at {node}",
        )


# ---------------------------------------------------------------------------
# 9. Centerlines are deterministic for the same rng state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_centerlines_determinism(pelvis_type: str) -> None:
    g1 = _build_and_place(pelvis_type)
    g2 = _build_and_place(pelvis_type)
    c1 = generate_centerlines(g1, _rng(SEED + 2))
    c2 = generate_centerlines(g2, _rng(SEED + 2))

    assert set(c1.keys()) == set(c2.keys())
    for edge in c1:
        np.testing.assert_array_equal(
            c1[edge], c2[edge], err_msg=f"centerline differs at {edge}"
        )


# ---------------------------------------------------------------------------
# 10. Different seeds produce different placements
# ---------------------------------------------------------------------------


def test_different_seeds_differ() -> None:
    g1 = build_topology("A2", _rng(1))
    g2 = build_topology("A2", _rng(1))
    place_nodes_3d(g1, _rng(10))
    place_nodes_3d(g2, _rng(20))

    differed = False
    for node in g1.nodes:
        if not np.allclose(
            g1.nodes[node]["position"], g2.nodes[node]["position"]
        ):
            differed = True
            break
    assert differed, "placement should vary across rng seeds"
