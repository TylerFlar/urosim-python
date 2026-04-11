"""Signed distance field primitives for procedural kidney mesh generation.

This module provides a small toolkit of signed distance functions (SDFs)
and operations used to assemble procedural kidney anatomy — spheres for
calyces, ellipsoids for the pelvis, tapered capsules for infundibular
tubes — plus smooth blending and gradient evaluation helpers.

Convention:
    Every SDF accepts query points as a ``numpy.ndarray`` of shape
    ``(N, 3)`` (float) and returns a ``numpy.ndarray`` of shape ``(N,)``
    (``float64``). The sign follows the standard SDF convention:

    * negative — inside the surface
    * zero — on the surface
    * positive — outside the surface

Units are millimetres throughout, matching the rest of
:mod:`urosim.anatomy`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Guard against division by zero when normalizing gradients or
# projecting onto near-degenerate line segments.
_TINY: float = 1e-12


def sphere_sdf(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Signed distance to a sphere.

    Args:
        points: Query points with shape ``(N, 3)``.
        center: Sphere center with shape ``(3,)``.
        radius: Sphere radius in mm. Must be non-negative.

    Returns:
        ``(N,)`` array of signed distances. Negative inside the sphere,
        zero on the surface, positive outside.
    """
    pts = np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    return np.linalg.norm(pts - c, axis=1) - float(radius)


def ellipsoid_sdf(
    points: np.ndarray,
    center: np.ndarray,
    half_axes: np.ndarray,
) -> np.ndarray:
    """Approximate signed distance to an axis-aligned ellipsoid.

    Uses the standard scale-to-unit-sphere approximation: the query
    points are scaled by ``1 / half_axes`` (relative to ``center``) so
    the ellipsoid maps to a unit sphere, the unit-sphere SDF is taken,
    and the result is re-scaled by ``min(half_axes)``. This is not the
    exact ellipsoid distance (which requires a root-find) but it is
    zero on the surface, has correct sign everywhere, and is a
    Lipschitz under-estimate — sufficient for compositing with other
    SDFs and for marching-cubes extraction.

    Args:
        points: Query points with shape ``(N, 3)``.
        center: Ellipsoid center with shape ``(3,)``.
        half_axes: Semi-axis lengths ``(a, b, c)`` with shape ``(3,)``.
            All entries must be strictly positive.

    Returns:
        ``(N,)`` array of approximate signed distances. Negative well
        inside the ellipsoid, approximately zero on the surface,
        positive outside.
    """
    pts = np.asarray(points, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    h = np.asarray(half_axes, dtype=np.float64).reshape(3)

    q = (pts - c) / h
    unit_dist = np.linalg.norm(q, axis=1) - 1.0
    return unit_dist * float(np.min(h))


def capsule_sdf(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    r_a: float,
    r_b: float,
) -> np.ndarray:
    """Signed distance to a capsule with linearly varying radius.

    The capsule is defined by a line segment from ``a`` to ``b`` with
    radius ``r_a`` at ``a`` and radius ``r_b`` at ``b``. Every query
    point is projected onto the segment to obtain a parameter
    ``h in [0, 1]``; the local radius is ``r_a + (r_b - r_a) * h`` and
    the signed distance is ``||point - closest|| - r``.

    Note that this is the standard *tapered* capsule used for anatomical
    tubes — it does **not** reduce to a constant-radius capsule unless
    ``r_a == r_b``. For very short segments (``||b - a|| < _TINY``) the
    capsule degenerates to a sphere at ``a`` with the mean radius.

    Args:
        points: Query points with shape ``(N, 3)``.
        a: Segment start with shape ``(3,)``.
        b: Segment end with shape ``(3,)``.
        r_a: Radius at ``a`` in mm. Must be non-negative.
        r_b: Radius at ``b`` in mm. Must be non-negative.

    Returns:
        ``(N,)`` array of signed distances.
    """
    pts = np.asarray(points, dtype=np.float64)
    a_vec = np.asarray(a, dtype=np.float64).reshape(3)
    b_vec = np.asarray(b, dtype=np.float64).reshape(3)
    ra = float(r_a)
    rb = float(r_b)

    ab = b_vec - a_vec
    ab_len_sq = float(np.dot(ab, ab))

    if ab_len_sq < _TINY:
        # Degenerate segment — fall back to a sphere at ``a`` with the
        # mean radius so the function stays well-defined.
        mean_r = 0.5 * (ra + rb)
        return np.linalg.norm(pts - a_vec, axis=1) - mean_r

    ap = pts - a_vec
    h = np.clip(ap @ ab / ab_len_sq, 0.0, 1.0)
    closest = a_vec[None, :] + h[:, None] * ab[None, :]
    r = ra + (rb - ra) * h
    return np.linalg.norm(pts - closest, axis=1) - r


def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum of two SDF values.

    Used to blend two SDFs so their union has a rounded fillet instead
    of a hard crease. When ``k == 0`` this reduces to ``np.minimum`` and
    the result is identical to a sharp union. For ``k > 0``, values near
    the intersection are pulled slightly below the hard minimum, giving
    the characteristic smooth-blend "dip".

    The implementation is the symmetric polynomial smin from Inigo
    Quilez::

        h = clamp(0.5 + 0.5 * (b - a) / k, 0, 1)
        smin = b * (1 - h) + a * h - k * h * (1 - h)

    Swapping ``a`` and ``b`` maps ``h`` to ``1 - h`` and leaves the
    expression invariant, so ``smooth_min(a, b, k) == smooth_min(b, a, k)``.

    Args:
        a: First SDF values with shape ``(N,)`` (or scalar).
        b: Second SDF values with shape ``(N,)`` (or scalar).
        k: Blending radius in the same units as ``a`` and ``b``. Must
            be non-negative. ``k == 0`` gives an exact ``min``.

    Returns:
        Array of blended SDF values matching the broadcast shape of
        ``a`` and ``b``.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    k_f = float(k)

    if k_f <= 0.0:
        return np.minimum(a_arr, b_arr)

    h = np.clip(0.5 + 0.5 * (b_arr - a_arr) / k_f, 0.0, 1.0)
    return b_arr * (1.0 - h) + a_arr * h - k_f * h * (1.0 - h)


def perlin_noise_3d(
    points: np.ndarray,
    scale: float = 1.0,
    octaves: int = 3,
    persistence: float = 0.5,
) -> np.ndarray:
    """Hash-based 3D fractal Brownian-motion value noise.

    This is a lightweight, dependency-free value-noise fBm intended for
    adding organic surface detail to SDFs (e.g. bumpy calyx walls). It
    is deterministic — the same inputs always produce the same outputs —
    but makes no cryptographic claims about the hash.

    Each octave samples integer lattice corners through a sine-based
    hash, trilinearly interpolates them with a smoothstep fade, and
    accumulates layers at doubling frequency and decaying amplitude.
    The final result is normalized by the total amplitude and shifted
    to lie approximately in ``[-1, 1]``.

    Args:
        points: Query points with shape ``(N, 3)``.
        scale: Frequency scale. Larger values produce finer detail.
        octaves: Number of fBm layers (``>= 1``).
        persistence: Amplitude decay factor between octaves. Typical
            values lie in ``(0, 1)``.

    Returns:
        ``(N,)`` array of noise values, approximately in ``[-1, 1]``.
    """
    pts = np.asarray(points, dtype=np.float64) * float(scale)

    total = np.zeros(pts.shape[0], dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for _ in range(int(octaves)):
        total += _value_noise_3d(pts * frequency) * amplitude
        max_amp += amplitude
        amplitude *= float(persistence)
        frequency *= 2.0

    if max_amp <= 0.0:
        return total

    # Normalize from [0, 1] into [-1, 1].
    return 2.0 * (total / max_amp) - 1.0


def _value_noise_3d(pts: np.ndarray) -> np.ndarray:
    """Single-octave trilinear value noise in ``[0, 1]``.

    Args:
        pts: Query points with shape ``(N, 3)`` in noise-space.

    Returns:
        ``(N,)`` noise values in ``[0, 1]``.
    """
    floor = np.floor(pts)
    frac = pts - floor
    ix = floor[:, 0]
    iy = floor[:, 1]
    iz = floor[:, 2]
    fx = frac[:, 0]
    fy = frac[:, 1]
    fz = frac[:, 2]

    # Smoothstep fade: 6t^5 - 15t^4 + 10t^3.
    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0)

    c000 = _hash3(ix, iy, iz)
    c100 = _hash3(ix + 1.0, iy, iz)
    c010 = _hash3(ix, iy + 1.0, iz)
    c110 = _hash3(ix + 1.0, iy + 1.0, iz)
    c001 = _hash3(ix, iy, iz + 1.0)
    c101 = _hash3(ix + 1.0, iy, iz + 1.0)
    c011 = _hash3(ix, iy + 1.0, iz + 1.0)
    c111 = _hash3(ix + 1.0, iy + 1.0, iz + 1.0)

    x00 = c000 * (1.0 - ux) + c100 * ux
    x10 = c010 * (1.0 - ux) + c110 * ux
    x01 = c001 * (1.0 - ux) + c101 * ux
    x11 = c011 * (1.0 - ux) + c111 * ux

    y0 = x00 * (1.0 - uy) + x10 * uy
    y1 = x01 * (1.0 - uy) + x11 * uy

    return y0 * (1.0 - uz) + y1 * uz


def _hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pseudo-random hash of three integer lattice coordinates.

    Args:
        x: x-coordinate array.
        y: y-coordinate array.
        z: z-coordinate array.

    Returns:
        Hashed values in ``[0, 1]`` with the same shape as the inputs.
    """
    s = np.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453
    return s - np.floor(s)


def sdf_gradient(
    sdf_func: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    eps: float = 0.1,
) -> np.ndarray:
    """Normalized gradient of an SDF via central finite differences.

    For every query point, ``sdf_func`` is evaluated at ``point ± eps``
    along each axis (six evaluations total) and the resulting gradient
    is normalized to unit length. For a well-formed SDF the gradient
    equals the outward surface normal — this is the standard way to
    recover normals from a purely implicit representation.

    Args:
        sdf_func: Any callable with signature ``(N, 3) -> (N,)``.
        points: Query points with shape ``(N, 3)``.
        eps: Finite-difference step in the same units as ``points``.
            Must be strictly positive.

    Returns:
        ``(N, 3)`` array of unit-length gradient vectors. Points where
        the raw gradient is smaller than ``_TINY`` are returned as the
        zero vector rather than dividing by a near-zero norm.
    """
    pts = np.asarray(points, dtype=np.float64)
    eps_f = float(eps)

    dx = np.array([eps_f, 0.0, 0.0])
    dy = np.array([0.0, eps_f, 0.0])
    dz = np.array([0.0, 0.0, eps_f])

    gx = sdf_func(pts + dx) - sdf_func(pts - dx)
    gy = sdf_func(pts + dy) - sdf_func(pts - dy)
    gz = sdf_func(pts + dz) - sdf_func(pts - dz)

    grad = np.stack([gx, gy, gz], axis=1) / (2.0 * eps_f)
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    safe = np.where(norms < _TINY, 1.0, norms)
    return np.where(norms < _TINY, 0.0, grad / safe)
