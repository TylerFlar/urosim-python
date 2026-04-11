"""Tests for :mod:`urosim.anatomy.normals`."""

from __future__ import annotations

import numpy as np
import trimesh

from urosim.anatomy.normals import compute_analytic_normals


def _compute(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
    pelvis_radii: tuple[float, float, float],
) -> np.ndarray:
    graph, centerlines = sample_kidney_graph_fast
    return compute_analytic_normals(
        np.asarray(kidney_mesh_fast.vertices, dtype=np.float64),
        graph,  # type: ignore[arg-type]
        centerlines,
        pelvis_radii,
    )


def test_shape_matches_vertices(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
    pelvis_radii: tuple[float, float, float],
) -> None:
    normals = _compute(kidney_mesh_fast, sample_kidney_graph_fast, pelvis_radii)
    assert normals.shape == (len(kidney_mesh_fast.vertices), 3)
    assert normals.dtype == np.float64


def test_unit_length(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
    pelvis_radii: tuple[float, float, float],
) -> None:
    normals = _compute(kidney_mesh_fast, sample_kidney_graph_fast, pelvis_radii)
    norms = np.linalg.norm(normals, axis=1)
    # Any zero rows (gradient underflow) are allowed but should be rare.
    nonzero = norms > 1e-6
    assert nonzero.mean() > 0.99, f"{(~nonzero).sum()} zero-length normals"
    assert np.allclose(norms[nonzero], 1.0, atol=1e-6)


def test_outward_facing_majority(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
    pelvis_radii: tuple[float, float, float],
) -> None:
    """SDF-gradient normals should agree with the mesh's outward vertex
    normals for the majority of vertices. trimesh's vertex normals are
    known-outward (the fast mesh fixture passes ``is_winding_consistent``
    and has positive volume), so a positive dot product confirms the
    SDF gradient points outward too."""
    normals = _compute(kidney_mesh_fast, sample_kidney_graph_fast, pelvis_radii)
    mesh_normals = np.asarray(kidney_mesh_fast.vertex_normals, dtype=np.float64)
    dots = np.einsum("ij,ij->i", normals, mesh_normals)
    # The SDF is only approximately unit-Lipschitz (Perlin noise is
    # additive), and vertex normals average adjacent face normals which
    # flip on sharp concave junctions, so we allow ~10% disagreement.
    assert (dots > 0).mean() > 0.90


def test_no_nan(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
    pelvis_radii: tuple[float, float, float],
) -> None:
    normals = _compute(kidney_mesh_fast, sample_kidney_graph_fast, pelvis_radii)
    assert np.isfinite(normals).all()


def test_deterministic(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
    pelvis_radii: tuple[float, float, float],
) -> None:
    a = _compute(kidney_mesh_fast, sample_kidney_graph_fast, pelvis_radii)
    b = _compute(kidney_mesh_fast, sample_kidney_graph_fast, pelvis_radii)
    assert np.array_equal(a, b)
