"""Tests for ``urosim.anatomy.topology``.

Covers structural, attribute, and determinism properties of
:func:`~urosim.anatomy.topology.build_topology` across all five Sampaio
pelvis types. All tests are deterministic: every rng is created fresh
from a known seed via the local ``_rng`` helper.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from urosim.anatomy.topology import build_topology

SEED = 12345

PELVIS_TYPES: tuple[str, ...] = ("A1", "A2", "A3", "B1", "B2")
A_TYPES: tuple[str, ...] = ("A1", "A2", "A3")
B_TYPES: tuple[str, ...] = ("B1", "B2")
VALID_POLES: frozenset[str] = frozenset({"upper", "mid", "lower"})


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. Structural: rooted directed tree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_is_rooted_arborescence(pelvis_type: str) -> None:
    graph = build_topology(pelvis_type, _rng())

    assert nx.is_arborescence(graph)
    assert nx.is_weakly_connected(graph)
    assert "pelvis" in graph
    assert graph.in_degree("pelvis") == 0
    for node in graph.nodes:
        if node == "pelvis":
            continue
        assert graph.in_degree(node) == 1, f"{node} has in-degree {graph.in_degree(node)}"


# ---------------------------------------------------------------------------
# 2. Major-calyx counts by pelvis type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", B_TYPES)
def test_b_type_has_exactly_two_majors(pelvis_type: str) -> None:
    for seed in range(50):
        graph = build_topology(pelvis_type, _rng(seed))
        majors = [n for n, d in graph.nodes(data=True) if d.get("type") == "major_calyx"]
        assert len(majors) == 2
        assert {graph.nodes[m]["pole"] for m in majors} == {"upper", "lower"}


@pytest.mark.parametrize("pelvis_type", A_TYPES)
def test_a_type_has_two_or_three_majors(pelvis_type: str) -> None:
    seen_counts: set[int] = set()
    for seed in range(50):
        graph = build_topology(pelvis_type, _rng(seed))
        majors = [n for n, d in graph.nodes(data=True) if d.get("type") == "major_calyx"]
        poles = {graph.nodes[m]["pole"] for m in majors}

        assert len(majors) in {2, 3}
        seen_counts.add(len(majors))
        if len(majors) == 2:
            assert poles == {"upper", "lower"}
        else:
            assert poles == {"upper", "mid", "lower"}

    # Across 50 seeds A-types should realize both 2- and 3-major configs
    # (at a 60/40 split the probability of missing either is ~10^-11).
    assert seen_counts == {2, 3}


# ---------------------------------------------------------------------------
# 3. Minor-calyx total-count invariant (7..13 across all pelvis types)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_minor_count_in_range(pelvis_type: str) -> None:
    for seed in range(200):
        graph = build_topology(pelvis_type, _rng(seed))
        minors = [n for n, d in graph.nodes(data=True) if d.get("type") == "minor_calyx"]
        count = len(minors)
        assert 7 <= count <= 13, f"seed={seed} pelvis_type={pelvis_type} count={count}"


# ---------------------------------------------------------------------------
# 4. Edge attributes: present, float, strictly positive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_edge_attrs_positive_floats(pelvis_type: str) -> None:
    graph = build_topology(pelvis_type, _rng())
    assert graph.number_of_edges() > 0
    for u, v, data in graph.edges(data=True):
        for key in ("length_mm", "width_mm", "angle_deg"):
            assert key in data, f"edge ({u},{v}) missing {key}"
            value = data[key]
            assert isinstance(value, float), f"edge ({u},{v}) {key} is {type(value).__name__}"
            assert value > 0.0, f"edge ({u},{v}) {key}={value}"


# ---------------------------------------------------------------------------
# 5. Minor calyces are leaves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_minors_are_leaves(pelvis_type: str) -> None:
    graph = build_topology(pelvis_type, _rng())
    minors = [n for n, d in graph.nodes(data=True) if d.get("type") == "minor_calyx"]
    assert len(minors) > 0
    for node in minors:
        assert graph.out_degree(node) == 0


# ---------------------------------------------------------------------------
# 6. Node attribute well-formedness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_node_attrs_well_formed(pelvis_type: str) -> None:
    graph = build_topology(pelvis_type, _rng())

    pelvis_data = graph.nodes["pelvis"]
    assert pelvis_data["type"] == "pelvis"
    assert "pole" not in pelvis_data

    for node, data in graph.nodes(data=True):
        if node == "pelvis":
            continue
        assert data["type"] in {"major_calyx", "minor_calyx"}
        assert "pole" in data
        assert data["pole"] in VALID_POLES
        # The pole attribute must match the name prefix.
        if data["type"] == "major_calyx":
            assert node == f"major_{data['pole']}"
        else:
            assert node.startswith(f"minor_{data['pole']}_")


# ---------------------------------------------------------------------------
# 7. Perpendicular calyx: width override lands in [2.0, 4.0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_perpendicular_width_under_4mm(pelvis_type: str) -> None:
    found = False
    for seed in range(2000):
        graph = build_topology(pelvis_type, _rng(seed))
        perp_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("is_perpendicular") is True
        ]
        if not perp_nodes:
            continue

        assert len(perp_nodes) == 1
        node = perp_nodes[0]
        assert graph.nodes[node]["type"] == "minor_calyx"

        parent = next(iter(graph.predecessors(node)))
        width = graph.edges[parent, node]["width_mm"]
        assert isinstance(width, float)
        assert 2.0 <= width < 4.0, f"perpendicular width_mm={width}"
        found = True
        break

    assert found, f"no perpendicular calyx observed in 2000 seeds for {pelvis_type}"


# ---------------------------------------------------------------------------
# 8. Determinism: same seed → identical graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_determinism(pelvis_type: str) -> None:
    g1 = build_topology(pelvis_type, _rng())
    g2 = build_topology(pelvis_type, _rng())

    assert set(g1.nodes) == set(g2.nodes)
    assert set(g1.edges) == set(g2.edges)

    for node in g1.nodes:
        assert g1.nodes[node] == g2.nodes[node], f"node attrs differ at {node}"

    for edge in g1.edges:
        assert g1.edges[edge] == g2.edges[edge], f"edge attrs differ at {edge}"


def test_different_seeds_differ() -> None:
    g1 = build_topology("A2", _rng(1))
    g2 = build_topology("A2", _rng(2))

    edges1 = {(u, v): dict(d) for u, v, d in g1.edges(data=True)}
    edges2 = {(u, v): dict(d) for u, v, d in g2.edges(data=True)}
    # Two different seeds should produce at least one differing edge
    # (structure and/or attributes).
    assert edges1 != edges2


# ---------------------------------------------------------------------------
# 9. Invalid pelvis type
# ---------------------------------------------------------------------------


def test_invalid_pelvis_type_raises() -> None:
    with pytest.raises(ValueError):
        build_topology("Z9", _rng())


def test_invalid_pelvis_type_empty_raises() -> None:
    with pytest.raises(ValueError):
        build_topology("", _rng())
