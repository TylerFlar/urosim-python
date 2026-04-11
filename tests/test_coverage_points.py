"""Tests for :mod:`urosim.anatomy.coverage_points`."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from urosim.anatomy.coverage_points import generate_coverage_points


def _minor_calyces(graph) -> set[str]:  # type: ignore[no-untyped-def]
    return {
        node
        for node, attrs in graph.nodes(data=True)
        if attrs.get("type") == "minor_calyx"
    }


def test_returns_all_minor_calyces(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    result = generate_coverage_points(graph, kidney_mesh_fast)  # type: ignore[arg-type]
    assert set(result.keys()) == _minor_calyces(graph)


def test_max_points_per_calyx(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    result = generate_coverage_points(graph, kidney_mesh_fast, points_per_calyx=30)  # type: ignore[arg-type]
    for pts in result.values():
        assert pts.shape[1] == 3
        assert pts.shape[0] <= 30


def test_points_within_radius(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    result = generate_coverage_points(graph, kidney_mesh_fast)  # type: ignore[arg-type]
    for node, pts in result.items():
        if pts.shape[0] == 0:
            continue
        pos = np.asarray(graph.nodes[node]["position"], dtype=np.float64)  # type: ignore[index]
        preds = list(graph.predecessors(node))  # type: ignore[attr-defined]
        width = float(graph.edges[preds[0], node]["width_mm"])  # type: ignore[index]
        # The implementation allows up to 1.5*width on the fallback path.
        dists = np.linalg.norm(pts - pos, axis=1)
        assert float(dists.max()) <= 1.5 * width + 1e-6


def test_points_are_mesh_vertices(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    result = generate_coverage_points(graph, kidney_mesh_fast)  # type: ignore[arg-type]
    verts = np.asarray(kidney_mesh_fast.vertices, dtype=np.float64)
    for pts in result.values():
        for row in pts:
            # Every coverage point must be one of the mesh vertices.
            match = np.all(verts == row, axis=1)
            assert match.any()


def test_custom_count(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    result = generate_coverage_points(graph, kidney_mesh_fast, points_per_calyx=5)  # type: ignore[arg-type]
    for pts in result.values():
        assert pts.shape[0] <= 5


def test_invalid_points_per_calyx_raises(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    with pytest.raises(ValueError):
        generate_coverage_points(graph, kidney_mesh_fast, points_per_calyx=0)  # type: ignore[arg-type]


def test_deterministic(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    graph, _ = sample_kidney_graph_fast
    a = generate_coverage_points(graph, kidney_mesh_fast)  # type: ignore[arg-type]
    b = generate_coverage_points(graph, kidney_mesh_fast)  # type: ignore[arg-type]
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k])
