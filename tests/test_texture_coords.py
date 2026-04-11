"""Tests for :mod:`urosim.anatomy.texture_coords`."""

from __future__ import annotations

import numpy as np
import trimesh

from urosim.anatomy.texture_coords import compute_texture_coords


def _compute(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> np.ndarray:
    graph, centerlines = sample_kidney_graph_fast
    return compute_texture_coords(
        np.asarray(kidney_mesh_fast.vertices, dtype=np.float64),
        graph,  # type: ignore[arg-type]
        centerlines,
    )


def test_shape(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    assert coords.shape == (len(kidney_mesh_fast.vertices), 3)
    assert coords.dtype == np.float64


def test_t_in_unit_interval(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    t = coords[:, 0]
    assert float(t.min()) >= 0.0
    assert float(t.max()) <= 1.0


def test_theta_in_range(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    theta = coords[:, 1]
    assert float(theta.min()) >= 0.0
    assert float(theta.max()) < 2.0 * np.pi


def test_edge_idx_valid(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    _graph, centerlines = sample_kidney_graph_fast
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    edge_idx = coords[:, 2]
    # Integer-valued (stored as float).
    assert np.array_equal(edge_idx, edge_idx.astype(np.int64).astype(np.float64))
    assert float(edge_idx.min()) >= 0.0
    assert float(edge_idx.max()) < float(len(centerlines))


def test_every_edge_claimed(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    _graph, centerlines = sample_kidney_graph_fast
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    claimed = set(coords[:, 2].astype(np.int64).tolist())
    # At fast voxel size (~1.0 mm) the mesh should still have enough
    # vertices near every infundibulum for each edge to be the nearest
    # centerline for at least one vertex.
    assert claimed == set(range(len(centerlines)))


def test_theta_wraps_full_circle(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    """Around any reasonable tube the circumferential coord should span
    a meaningful fraction of 2*pi — mucosal folds depend on it."""
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    theta = coords[:, 1]
    assert float(theta.max() - theta.min()) > np.pi


def test_endpoints(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    """Vertices nearest a centerline endpoint should have t ≈ 0 or ≈ 1."""
    _graph, centerlines = sample_kidney_graph_fast
    verts = np.asarray(kidney_mesh_fast.vertices, dtype=np.float64)
    coords = _compute(kidney_mesh_fast, sample_kidney_graph_fast)

    # Pick the first edge (sorted order) and its endpoints.
    first_edge = sorted(centerlines.items())[0]
    line = np.asarray(first_edge[1], dtype=np.float64)
    start, end = line[0], line[-1]

    start_vert = int(np.argmin(np.linalg.norm(verts - start, axis=1)))
    end_vert = int(np.argmin(np.linalg.norm(verts - end, axis=1)))

    # Relaxed bounds: smoothing and nearby edges may shift which edge
    # owns the endpoint vertex. We just require that when the owning
    # edge IS the first edge, t is near the right endpoint.
    if int(coords[start_vert, 2]) == 0:
        assert coords[start_vert, 0] < 0.3
    if int(coords[end_vert, 2]) == 0:
        assert coords[end_vert, 0] > 0.7


def test_deterministic(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    a = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    b = _compute(kidney_mesh_fast, sample_kidney_graph_fast)
    assert np.array_equal(a, b)
