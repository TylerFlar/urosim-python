"""Integration tests for :mod:`urosim.anatomy.generator`.

Exercises the full seed-to-``KidneyModel`` pipeline: attribute
population, determinism under identical seeds, save/load round-trip,
and the invariant on the number of minor calyces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from urosim.anatomy import AnatomyGenerator, KidneyModel

_VALID_PELVIS_TYPES: frozenset[str] = frozenset({"A1", "A2", "A3", "B1", "B2"})


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_generate_all_attributes_populated(seed: int) -> None:
    """Every :class:`KidneyModel` field is populated with sensible shapes."""
    model = AnatomyGenerator(seed=seed).generate()

    # Scalar metadata.
    assert model.seed == seed
    assert model.pelvis_type in _VALID_PELVIS_TYPES
    assert model.metadata["seed"] == seed
    assert model.metadata["pelvis_type"] == model.pelvis_type

    # Mesh is watertight and non-trivial.
    assert model.mesh.is_watertight
    v = model.mesh.vertices.shape[0]
    assert v > 0
    assert model.mesh.faces.shape[0] > 0

    # Per-vertex arrays line up with the mesh.
    assert model.normals.shape == (v, 3)
    norm_lengths = np.linalg.norm(model.normals, axis=1)
    # Normals must be unit length (zero rows are not expected on a
    # well-behaved SDF, but allow the tiny tolerance of finite differences).
    assert np.allclose(norm_lengths, 1.0, atol=1e-3)

    assert model.texture_coords.shape == (v, 3)
    assert np.all(model.texture_coords[:, 0] >= 0.0)
    assert np.all(model.texture_coords[:, 0] <= 1.0)
    assert np.all(model.texture_coords[:, 1] >= 0.0)
    assert np.all(model.texture_coords[:, 1] < 2.0 * np.pi + 1e-9)

    # Graph, centerlines and calyx poses.
    assert model.graph.number_of_nodes() > 0
    assert len(model.centerlines) == model.graph.number_of_edges()
    assert len(model.centerlines) > 0
    calyx_ids = set(model.calyx_ids())
    assert set(model.calyx_poses.keys()) == calyx_ids
    assert set(model.calyx_coverage_points.keys()) <= calyx_ids
    for pos in model.calyx_poses.values():
        assert pos.shape == (3,)

    # Stones.
    assert 1 <= len(model.stones) <= 4
    for stone in model.stones:
        assert stone.position.shape == (3,)
        assert stone.radius_mm > 0.0

    # Scope entry placeholder.
    assert model.ureter_entry.shape == (3,)


def test_determinism_same_seed_produces_identical_output() -> None:
    """Two generations with the same seed produce bit-identical artifacts."""
    a = AnatomyGenerator(seed=42).generate()
    b = AnatomyGenerator(seed=42).generate()

    assert a.pelvis_type == b.pelvis_type
    assert a.metadata == b.metadata

    assert np.array_equal(a.mesh.vertices, b.mesh.vertices)
    assert np.array_equal(a.mesh.faces, b.mesh.faces)
    assert np.array_equal(a.normals, b.normals)
    assert np.array_equal(a.texture_coords, b.texture_coords)

    assert sorted(a.graph.nodes) == sorted(b.graph.nodes)
    assert sorted(a.graph.edges) == sorted(b.graph.edges)
    for node in a.graph.nodes:
        assert np.array_equal(
            a.graph.nodes[node]["position"],
            b.graph.nodes[node]["position"],
        )

    assert len(a.stones) == len(b.stones)
    for s_a, s_b in zip(a.stones, b.stones, strict=True):
        assert s_a.composition == s_b.composition
        assert s_a.radius_mm == s_b.radius_mm
        assert np.array_equal(s_a.position, s_b.position)

    assert sorted(a.centerlines.keys()) == sorted(b.centerlines.keys())
    for key in a.centerlines:
        assert np.array_equal(a.centerlines[key], b.centerlines[key])

    assert np.array_equal(a.ureter_entry, b.ureter_entry)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """A saved :class:`KidneyModel` reloads with identical key attributes."""
    original = AnatomyGenerator(seed=7).generate()

    base_path = tmp_path / "kidney"
    original.save(base_path)
    loaded = KidneyModel.load(base_path)

    # Expected sibling files exist.
    assert base_path.with_suffix(".json").exists()
    assert base_path.with_suffix(".obj").exists()
    assert (tmp_path / "kidney_normals.npy").exists()
    assert (tmp_path / "kidney_texcoords.npy").exists()
    assert (tmp_path / "kidney_coverage.json").exists()
    assert (tmp_path / "kidney_centerlines.npz").exists()

    # Scalars.
    assert loaded.seed == original.seed
    assert loaded.pelvis_type == original.pelvis_type
    assert loaded.metadata["seed"] == original.seed

    # Mesh — vertex and face counts must match; OBJ round-trip is float32
    # through text, so exact equality is not guaranteed. Check counts and
    # approximate positions.
    assert loaded.mesh.vertices.shape == original.mesh.vertices.shape
    assert loaded.mesh.faces.shape == original.mesh.faces.shape
    assert np.allclose(loaded.mesh.vertices, original.mesh.vertices, atol=1e-4)

    # Stones — count, composition, and position (as lists) preserved.
    assert len(loaded.stones) == len(original.stones)
    for s_loaded, s_original in zip(loaded.stones, original.stones, strict=True):
        assert s_loaded.composition == s_original.composition
        assert s_loaded.radius_mm == pytest.approx(s_original.radius_mm)
        assert np.allclose(s_loaded.position, s_original.position)
        assert s_loaded.color == pytest.approx(s_original.color)

    # Graph nodes match; positions preserved through JSON.
    assert sorted(loaded.graph.nodes) == sorted(original.graph.nodes)
    assert sorted(loaded.graph.edges) == sorted(original.graph.edges)
    for node in original.graph.nodes:
        assert np.allclose(
            loaded.graph.nodes[node]["position"],
            original.graph.nodes[node]["position"],
        )

    # Per-vertex arrays round-trip through .npy exactly.
    assert np.array_equal(loaded.normals, original.normals)
    assert np.array_equal(loaded.texture_coords, original.texture_coords)

    # Centerlines keys and values.
    assert set(loaded.centerlines.keys()) == set(original.centerlines.keys())
    for key in original.centerlines:
        assert np.array_equal(loaded.centerlines[key], original.centerlines[key])

    # Coverage points keys and per-calyx shapes match; values via allclose
    # (JSON round-trip through text).
    assert set(loaded.calyx_coverage_points.keys()) == set(
        original.calyx_coverage_points.keys()
    )
    for key in original.calyx_coverage_points:
        assert (
            loaded.calyx_coverage_points[key].shape
            == original.calyx_coverage_points[key].shape
        )
        assert np.allclose(
            loaded.calyx_coverage_points[key],
            original.calyx_coverage_points[key],
        )

    assert np.allclose(loaded.ureter_entry, original.ureter_entry)
    assert loaded.calyx_ids() == original.calyx_ids()


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_calyx_ids_count_in_expected_range(seed: int) -> None:
    """Every kidney has between 7 and 13 minor calyces (pipeline invariant)."""
    model = AnatomyGenerator(seed=seed).generate()
    calyx_ids = model.calyx_ids()
    assert 7 <= len(calyx_ids) <= 13
