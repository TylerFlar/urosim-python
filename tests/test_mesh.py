"""Tests for :mod:`urosim.anatomy.mesh_extract`.

The fast tier (tests 1-8, 11) reuses a single session-cached mesh
extracted at ``voxel_size=1.0`` so CI cost is bounded. The slow tier
(tests 9, 10) runs full-resolution extractions across multiple seeds
and topologies and is gated behind the ``slow`` marker.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import trimesh

from urosim.anatomy.mesh_extract import extract_kidney_mesh

_FAST_VOXEL_SIZE: float = 1.0
_FAST_TARGET_EDGE: float = 2.0
_DEFAULT_SEED: int = 12345


@pytest.fixture(scope="session")
def kidney_mesh_fast(
    pelvis_radii: tuple[float, float, float],
) -> trimesh.Trimesh:
    """Session-scoped kidney mesh extracted at a coarse resolution.

    Reused across the fast-tier tests so the expensive SDF evaluation
    and remeshing pipeline only runs once per pytest session.
    """
    # Rebuild the sample kidney inline — session-scoped fixtures cannot
    # depend on the function-scoped ``sample_kidney`` fixture.
    from urosim.anatomy.centerlines import generate_centerlines
    from urosim.anatomy.placement import place_nodes_3d
    from urosim.anatomy.topology import build_topology

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


@pytest.fixture(scope="session")
def sample_kidney_graph_fast(
    pelvis_radii: tuple[float, float, float],
) -> tuple[object, dict]:
    """Session-scoped (graph, centerlines) matching ``kidney_mesh_fast``."""
    from urosim.anatomy.centerlines import generate_centerlines
    from urosim.anatomy.placement import place_nodes_3d
    from urosim.anatomy.topology import build_topology

    rng = np.random.default_rng(_DEFAULT_SEED)
    graph = build_topology("A2", rng)
    place_nodes_3d(graph, rng)
    centerlines = generate_centerlines(graph, rng)
    return graph, centerlines


def test_returns_trimesh(kidney_mesh_fast: trimesh.Trimesh) -> None:
    assert isinstance(kidney_mesh_fast, trimesh.Trimesh)
    assert len(kidney_mesh_fast.faces) > 0
    assert len(kidney_mesh_fast.vertices) > 0


def test_is_watertight(kidney_mesh_fast: trimesh.Trimesh) -> None:
    assert kidney_mesh_fast.is_watertight


def test_winding_consistent(kidney_mesh_fast: trimesh.Trimesh) -> None:
    assert kidney_mesh_fast.is_winding_consistent


def test_face_count_in_range(kidney_mesh_fast: trimesh.Trimesh) -> None:
    n_faces = len(kidney_mesh_fast.faces)
    assert 1000 < n_faces <= 50000, f"unexpected face count: {n_faces}"


def test_no_degenerate_faces(kidney_mesh_fast: trimesh.Trimesh) -> None:
    areas = kidney_mesh_fast.area_faces
    assert np.all(areas > 1e-9), "mesh contains zero-area faces"
    assert kidney_mesh_fast.is_volume


def test_bbox_size_reasonable(kidney_mesh_fast: trimesh.Trimesh) -> None:
    extents = kidney_mesh_fast.bounding_box.extents
    assert 40.0 <= float(extents.max()) <= 120.0, f"extents: {extents}"


def test_normals_outward(kidney_mesh_fast: trimesh.Trimesh) -> None:
    # trimesh convention: a watertight mesh with outward-facing normals
    # has positive signed volume.
    assert kidney_mesh_fast.volume > 0.0


def test_encloses_graph_nodes(
    kidney_mesh_fast: trimesh.Trimesh,
    sample_kidney_graph_fast: tuple[object, dict],
) -> None:
    """Every graph node position must lie inside the mesh AABB.

    A stricter interior test (``mesh.contains``) requires an optional
    ``rtree`` backend that is not guaranteed to be installed. The AABB
    check is weak but catches the main failure mode — a mesh extracted
    in the wrong coordinate frame or at the wrong scale.
    """
    graph, _ = sample_kidney_graph_fast
    positions = np.stack(
        [
            np.asarray(attrs["position"], dtype=np.float64)
            for _, attrs in graph.nodes(data=True)  # type: ignore[attr-defined]
        ],
        axis=0,
    )
    lo, hi = kidney_mesh_fast.bounds
    inside = np.all((positions >= lo) & (positions <= hi), axis=1)
    assert inside.all(), (
        f"{(~inside).sum()} of {len(inside)} node positions fall outside "
        f"mesh AABB {lo} -- {hi}"
    )


def test_voxel_size_param(
    sample_kidney, pelvis_radii: tuple[float, float, float]
) -> None:
    graph, centerlines = sample_kidney(pelvis_type="A2", seed=_DEFAULT_SEED)
    mesh = extract_kidney_mesh(
        graph,
        centerlines,
        pelvis_radii,
        voxel_size=1.5,
        target_edge_length=2.5,
    )
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert len(mesh.faces) > 500


@pytest.mark.slow
def test_multiple_seeds(
    sample_kidney, pelvis_radii: tuple[float, float, float]
) -> None:
    """Extract full-resolution kidneys from three distinct seeds."""
    seeds = (12345, 23456, 34567)
    for seed in seeds:
        graph, centerlines = sample_kidney(pelvis_type="A2", seed=seed)
        start = time.perf_counter()
        mesh = extract_kidney_mesh(graph, centerlines, pelvis_radii)
        elapsed = time.perf_counter() - start
        assert elapsed < 60.0, f"seed {seed}: extraction took {elapsed:.1f}s"
        assert mesh.is_watertight, f"seed {seed}: not watertight"
        assert mesh.is_winding_consistent, f"seed {seed}: winding"
        assert 2000 <= len(mesh.faces) <= 50000, (
            f"seed {seed}: face count {len(mesh.faces)} out of range"
        )
        assert np.all(mesh.area_faces > 1e-9), (
            f"seed {seed}: degenerate faces present"
        )
        extents = mesh.bounding_box.extents
        assert 40.0 <= float(extents.max()) <= 120.0, (
            f"seed {seed}: bbox extents {extents}"
        )


@pytest.mark.slow
def test_pelvis_type_b1(
    sample_kidney, pelvis_radii: tuple[float, float, float]
) -> None:
    """Full-resolution smoke test for a different Sampaio topology."""
    graph, centerlines = sample_kidney(pelvis_type="B1", seed=_DEFAULT_SEED)
    mesh = extract_kidney_mesh(graph, centerlines, pelvis_radii)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert 2000 <= len(mesh.faces) <= 50000
