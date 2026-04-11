"""Tests for :mod:`urosim.anatomy.sdf`."""

from __future__ import annotations

import numpy as np

from urosim.anatomy.sdf import (
    capsule_sdf,
    ellipsoid_sdf,
    perlin_noise_3d,
    sdf_gradient,
    smooth_min,
    sphere_sdf,
)

SEED = 12345
N_BATCH = 1000


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# sphere_sdf
# ---------------------------------------------------------------------------


def test_sphere_center_returns_negative_radius() -> None:
    center = np.array([1.0, 2.0, 3.0])
    d = sphere_sdf(np.array([[1.0, 2.0, 3.0]]), center, 2.0)
    assert d.shape == (1,)
    assert np.isclose(d[0], -2.0)


def test_sphere_surface_point_returns_zero() -> None:
    d = sphere_sdf(np.array([[2.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]), 2.0)
    assert np.isclose(d[0], 0.0)


def test_sphere_outside_point_is_positive() -> None:
    d = sphere_sdf(np.array([[5.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]), 2.0)
    assert np.isclose(d[0], 3.0)


def test_sphere_inside_point_is_negative() -> None:
    d = sphere_sdf(np.array([[0.5, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0]), 2.0)
    assert d[0] < 0.0
    assert np.isclose(d[0], -1.5)


def test_sphere_batched_signs_consistent() -> None:
    rng = _rng()
    pts = rng.uniform(-5.0, 5.0, size=(N_BATCH, 3))
    center = np.array([0.0, 0.0, 0.0])
    radius = 2.5
    d = sphere_sdf(pts, center, radius)

    assert d.shape == (N_BATCH,)
    assert np.all(np.isfinite(d))

    expected_sign = np.linalg.norm(pts, axis=1) - radius
    assert np.all(np.sign(d) == np.sign(expected_sign))


# ---------------------------------------------------------------------------
# ellipsoid_sdf
# ---------------------------------------------------------------------------


def test_ellipsoid_center_is_well_inside() -> None:
    center = np.array([0.0, 0.0, 0.0])
    half_axes = np.array([3.0, 2.0, 1.0])
    d = ellipsoid_sdf(np.array([[0.0, 0.0, 0.0]]), center, half_axes)
    assert d[0] < 0.0
    # Center should sit ~min(half_axes) below the surface.
    assert np.isclose(d[0], -1.0)


def test_ellipsoid_axis_endpoints_are_surface() -> None:
    center = np.array([0.0, 0.0, 0.0])
    half_axes = np.array([3.0, 2.0, 1.0])
    surface_pts = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [-3.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    d = ellipsoid_sdf(surface_pts, center, half_axes)
    assert np.allclose(d, 0.0, atol=1e-9)


def test_ellipsoid_far_outside_is_positive() -> None:
    center = np.array([0.0, 0.0, 0.0])
    half_axes = np.array([3.0, 2.0, 1.0])
    d = ellipsoid_sdf(np.array([[50.0, 0.0, 0.0]]), center, half_axes)
    assert d[0] > 0.0


def test_ellipsoid_batched_shape() -> None:
    rng = _rng()
    pts = rng.uniform(-5.0, 5.0, size=(N_BATCH, 3))
    d = ellipsoid_sdf(pts, np.zeros(3), np.array([3.0, 2.0, 1.0]))
    assert d.shape == (N_BATCH,)
    assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# capsule_sdf
# ---------------------------------------------------------------------------


def test_capsule_endpoint_a_surface() -> None:
    # Point at distance r_a in a direction perpendicular to the segment
    # should lie exactly on the capsule surface at endpoint a.
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    r_a = 2.0
    r_b = 1.0
    # Perpendicular to ab is anything in the y-z plane.
    p = np.array([[0.0, 2.0, 0.0]])
    d = capsule_sdf(p, a, b, r_a, r_b)
    assert np.isclose(d[0], 0.0)


def test_capsule_endpoint_b_surface() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    r_a = 2.0
    r_b = 1.0
    p = np.array([[10.0, 0.0, 1.0]])
    d = capsule_sdf(p, a, b, r_a, r_b)
    assert np.isclose(d[0], 0.0)


def test_capsule_midpoint_interpolates_radius() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    r_a = 2.0
    r_b = 4.0  # expected radius at midpoint: 3.0
    p = np.array([[5.0, 0.0, 5.0]])  # perpendicular offset 5 from axis
    d = capsule_sdf(p, a, b, r_a, r_b)
    assert np.isclose(d[0], 5.0 - 3.0)


def test_capsule_endpoint_interior_is_negative() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    p = np.array([[0.0, 0.5, 0.0], [10.0, 0.25, 0.0]])
    d = capsule_sdf(p, a, b, 2.0, 1.0)
    assert np.all(d < 0.0)


def test_capsule_constant_radius() -> None:
    # When r_a == r_b, the capsule should match the standard capsule
    # formula (distance to segment minus constant r).
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([4.0, 0.0, 0.0])
    r = 1.5
    # Point directly perpendicular to the midpoint of the segment.
    p = np.array([[2.0, 3.0, 0.0]])
    d = capsule_sdf(p, a, b, r, r)
    assert np.isclose(d[0], 3.0 - r)

    # Beyond endpoint a along negative x: distance is from a.
    p2 = np.array([[-2.0, 0.0, 0.0]])
    d2 = capsule_sdf(p2, a, b, r, r)
    assert np.isclose(d2[0], 2.0 - r)


def test_capsule_batched_shape() -> None:
    rng = _rng()
    pts = rng.uniform(-10.0, 10.0, size=(N_BATCH, 3))
    a = np.array([-2.0, 0.0, 0.0])
    b = np.array([2.0, 1.0, 0.5])
    d = capsule_sdf(pts, a, b, 1.0, 2.0)
    assert d.shape == (N_BATCH,)
    assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# smooth_min
# ---------------------------------------------------------------------------


def test_smooth_min_k_zero_is_exact_min() -> None:
    rng = _rng()
    a = rng.uniform(-5.0, 5.0, size=50)
    b = rng.uniform(-5.0, 5.0, size=50)
    result = smooth_min(a, b, 0.0)
    assert np.array_equal(result, np.minimum(a, b))


def test_smooth_min_equal_values_blends_below() -> None:
    a = np.array([1.0, 2.0, -3.0])
    b = a.copy()
    k = 0.5
    result = smooth_min(a, b, k)
    # With a == b, h = 0.5 and smin = a - k*0.25.
    assert np.all(result < a)
    assert np.allclose(result, a - k * 0.25)


def test_smooth_min_is_symmetric() -> None:
    rng = _rng()
    a = rng.uniform(-5.0, 5.0, size=200)
    b = rng.uniform(-5.0, 5.0, size=200)
    k = 0.75
    ab = smooth_min(a, b, k)
    ba = smooth_min(b, a, k)
    assert np.allclose(ab, ba)


def test_smooth_min_far_apart_recovers_min() -> None:
    # When |a - b| is much larger than k, smooth_min should essentially
    # equal np.minimum.
    a = np.array([0.0, 10.0, -5.0])
    b = np.array([10.0, 0.0, 5.0])
    k = 0.01
    result = smooth_min(a, b, k)
    assert np.allclose(result, np.minimum(a, b), atol=1e-3)


# ---------------------------------------------------------------------------
# perlin_noise_3d
# ---------------------------------------------------------------------------


def test_perlin_is_deterministic() -> None:
    rng = _rng()
    pts = rng.uniform(-5.0, 5.0, size=(200, 3))
    a = perlin_noise_3d(pts, scale=1.0, octaves=3, persistence=0.5)
    b = perlin_noise_3d(pts, scale=1.0, octaves=3, persistence=0.5)
    assert np.array_equal(a, b)


def test_perlin_shape_and_finite() -> None:
    rng = _rng()
    pts = rng.uniform(-5.0, 5.0, size=(N_BATCH, 3))
    n = perlin_noise_3d(pts, scale=0.5, octaves=4, persistence=0.6)
    assert n.shape == (N_BATCH,)
    assert np.all(np.isfinite(n))


def test_perlin_roughly_bounded() -> None:
    rng = _rng()
    pts = rng.uniform(-10.0, 10.0, size=(N_BATCH, 3))
    n = perlin_noise_3d(pts, scale=1.0, octaves=3, persistence=0.5)
    # Value noise with normalized amplitude is bounded in [-1, 1].
    # Allow a small slack for numerical safety.
    assert np.max(n) <= 1.0 + 1e-9
    assert np.min(n) >= -1.0 - 1e-9


# ---------------------------------------------------------------------------
# sdf_gradient
# ---------------------------------------------------------------------------


def test_sdf_gradient_of_sphere_is_radial() -> None:
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    def sphere(p: np.ndarray) -> np.ndarray:
        return sphere_sdf(p, center, radius)

    rng = _rng()
    # Generate random points well away from the origin so finite
    # differences are numerically stable.
    pts = rng.normal(size=(100, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * rng.uniform(2.0, 5.0, size=(100, 1))

    grads = sdf_gradient(sphere, pts, eps=0.01)

    # Unit length.
    assert np.allclose(np.linalg.norm(grads, axis=1), 1.0, atol=1e-4)

    # Radial: each gradient should match p / ||p||.
    expected = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    assert np.allclose(grads, expected, atol=1e-3)


def test_sdf_gradient_batched_shape_and_unit_length() -> None:
    def sphere(p: np.ndarray) -> np.ndarray:
        return sphere_sdf(p, np.zeros(3), 2.0)

    rng = _rng()
    pts = rng.normal(size=(N_BATCH, 3))
    # Avoid the origin where the gradient is undefined.
    pts = pts + np.sign(pts) * 1.0

    grads = sdf_gradient(sphere, pts, eps=0.05)
    assert grads.shape == (N_BATCH, 3)
    assert np.all(np.isfinite(grads))
    assert np.allclose(np.linalg.norm(grads, axis=1), 1.0, atol=1e-3)
