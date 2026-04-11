"""Tests for :mod:`urosim.anatomy.stones`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import trimesh

from urosim.anatomy.centerlines import generate_centerlines
from urosim.anatomy.placement import place_nodes_3d
from urosim.anatomy.stones import (
    _STONE_COLORS,
    _STONE_HARDNESS,
    StoneSpec,
    place_stones,
)
from urosim.anatomy.topology import build_topology

_VALID_COMPOSITIONS = {"CaOx_mono", "CaOx_di", "uric", "struvite", "cystine"}


def test_returns_list_of_stonespec(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(42)
    stones = place_stones(graph, kidney_mesh_fast, rng)  # type: ignore[arg-type]
    assert isinstance(stones, list)
    for s in stones:
        assert isinstance(s, StoneSpec)


def test_default_count_1_to_4(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    for seed in range(20):
        rng = np.random.default_rng(seed)
        stones = place_stones(graph, kidney_mesh_fast, rng)  # type: ignore[arg-type]
        assert 1 <= len(stones) <= 4


def test_explicit_count_honored(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(7)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=7)  # type: ignore[arg-type]
    assert len(stones) == 7


def test_zero_stones(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(1)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=0)  # type: ignore[arg-type]
    assert stones == []


def test_composition_valid(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(12345)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=20)  # type: ignore[arg-type]
    for s in stones:
        assert s.composition in _VALID_COMPOSITIONS


def test_radius_range(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(12345)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=20)  # type: ignore[arg-type]
    for s in stones:
        # sample_stone_size clamps to [2, 15] mm diameter → [1, 7.5] mm radius.
        assert 1.0 <= s.radius_mm <= 7.5


def test_hardness_matches_table(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(12345)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=20)  # type: ignore[arg-type]
    for s in stones:
        assert s.hardness == _STONE_HARDNESS[s.composition]


def test_color_matches_table(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(12345)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=20)  # type: ignore[arg-type]
    for s in stones:
        assert s.color == _STONE_COLORS[s.composition]


def test_hp_formula(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(12345)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=20)  # type: ignore[arg-type]
    for s in stones:
        expected = (4.0 / 3.0) * np.pi * (s.radius_mm ** 3) * s.hardness
        assert s.hp == pytest.approx(expected, rel=1e-12)


def test_position_near_node(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(12345)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=20)  # type: ignore[arg-type]
    # Gather candidate target positions (pelvis + every minor calyx).
    targets = []
    for _node, attrs in graph.nodes(data=True):  # type: ignore[attr-defined]
        if attrs.get("type") in ("minor_calyx", "pelvis"):
            targets.append(np.asarray(attrs["position"], dtype=np.float64))
    T = np.stack(targets, axis=0)
    for s in stones:
        dists = np.linalg.norm(T - s.position, axis=1)
        assert float(dists.min()) <= 3.0


def test_b1_topology_no_crash(
    kidney_mesh_fast: trimesh.Trimesh,
) -> None:
    """B1/B2 topologies have no mid-pole minor calyces — the function
    must handle a ``location == 'mid'`` draw via its fallback path."""
    rng = np.random.default_rng(12345)
    graph = build_topology("B1", rng)
    place_nodes_3d(graph, rng)
    _ = generate_centerlines(graph, rng)  # state parity with fast fixture
    place_rng = np.random.default_rng(999)
    stones = place_stones(graph, kidney_mesh_fast, place_rng, num_stones=30)  # type: ignore[arg-type]
    assert len(stones) == 30
    # Collect positions of all eligible targets on this graph.
    targets = [
        np.asarray(attrs["position"], dtype=np.float64)
        for _, attrs in graph.nodes(data=True)
        if attrs.get("type") in ("minor_calyx", "pelvis")
    ]
    T = np.stack(targets, axis=0)
    for s in stones:
        assert float(np.linalg.norm(T - s.position, axis=1).min()) <= 3.0


def test_frozen(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(1)
    stones = place_stones(graph, kidney_mesh_fast, rng, num_stones=1)  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        stones[0].radius_mm = 999.0  # type: ignore[misc]


def test_deterministic(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng_a = np.random.default_rng(2024)
    rng_b = np.random.default_rng(2024)
    a = place_stones(graph, kidney_mesh_fast, rng_a, num_stones=5)  # type: ignore[arg-type]
    b = place_stones(graph, kidney_mesh_fast, rng_b, num_stones=5)  # type: ignore[arg-type]
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa.position, sb.position)
        assert sa.radius_mm == sb.radius_mm
        assert sa.composition == sb.composition
        assert sa.hp == sb.hp
        assert sa.hardness == sb.hardness
        assert sa.color == sb.color


def test_negative_num_stones_raises(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        place_stones(graph, kidney_mesh_fast, rng, num_stones=-1)  # type: ignore[arg-type]
