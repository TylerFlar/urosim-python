"""Smooth 3D centerline generation for collecting-system edges.

Given a kidney collecting-system graph with 3D node positions (from
:func:`urosim.anatomy.placement.place_nodes_3d`), this module generates a
smooth curved centerline sampled along each infundibular edge. Each
centerline is represented as a ``(M, 3)`` numpy array of sample points
in mm, with the first point pinned to the parent node's position and
the last point pinned to the child node's position.

Determinism: :func:`generate_centerlines` consumes the supplied
``numpy.random.Generator`` in a fixed order (sorted edge order) so that
repeated calls with the same graph and the same initial rng state
produce identical centerline arrays.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.interpolate import CubicSpline

# Minimum number of sample points per centerline.
_MIN_SAMPLES: int = 20

# Target spacing between consecutive sample points (mm). A 60 mm edge
# therefore yields ~31 samples; max step is well under the 10 mm bound.
_TARGET_STEP_MM: float = 2.0

# Curvature control: intermediate control points are offset from the
# straight-line chord by ``length * _CURVATURE_FRACTION * rng.uniform(0.5, 1.5)``.
_CURVATURE_FRACTION: float = 0.10


def generate_centerlines(
    graph: nx.DiGraph,
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate a smooth 3D centerline for every edge in ``graph``.

    For each edge ``(u, v)`` a cubic spline is fit through four control
    points: the parent position, two intermediate points offset slightly
    off-chord for organic curvature, and the child position. The spline
    is sampled at ``M`` equally spaced parameter values, where ``M``
    scales with edge length so consecutive samples are ~2 mm apart (and
    always well under the 10 mm hard bound).

    The first and last samples are pinned exactly to the endpoint node
    positions to protect against any floating-point drift from the
    spline fit.

    Args:
        graph: A collecting-system graph whose nodes carry a ``position``
            attribute (``numpy.ndarray`` of shape ``(3,)``) — typically
            produced by :func:`~urosim.anatomy.placement.place_nodes_3d`.
        rng: numpy ``Generator`` used as the sole randomness source. The
            function iterates edges in sorted order and consumes draws
            in a fixed sequence per edge, so the output is a pure
            function of the graph and the generator's initial state.

    Returns:
        A dict mapping each edge ``(u, v)`` to a ``numpy.ndarray`` of
        shape ``(M, 3)`` representing the sampled centerline in mm.
        ``M >= 20`` for every edge.
    """
    centerlines: dict[tuple[str, str], np.ndarray] = {}

    for u, v in sorted(graph.edges()):
        p0 = np.asarray(graph.nodes[u]["position"], dtype=np.float64)
        p3 = np.asarray(graph.nodes[v]["position"], dtype=np.float64)

        diff = p3 - p0
        length = float(np.linalg.norm(diff))

        # Sample count scales with length; always at least _MIN_SAMPLES.
        m = max(_MIN_SAMPLES, int(np.ceil(length / _TARGET_STEP_MM)) + 1)

        # Degenerate (near-zero-length) edge: skip rng draws, return a
        # constant centerline. This path should not arise in practice
        # because placement enforces a 2 mm minimum node separation.
        if length < 1e-6:
            centerline = np.tile(p0, (m, 1)).astype(np.float64)
            centerline[0] = p0
            centerline[-1] = p3
            centerlines[(u, v)] = centerline
            continue

        direction = diff / length

        # Random perpendicular axis for the curvature offset. Use
        # Gram-Schmidt on a normal draw; fall back deterministically if
        # the draw is parallel to the chord.
        rand_vec = rng.normal(size=3)
        perp = rand_vec - np.dot(rand_vec, direction) * direction
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-9:
            fallback = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(fallback, direction))) > 0.99:
                fallback = np.array([0.0, 0.0, 1.0])
            perp = fallback - np.dot(fallback, direction) * direction
            perp_norm = float(np.linalg.norm(perp))
        perp = perp / perp_norm

        offset_mag = length * _CURVATURE_FRACTION * float(rng.uniform(0.5, 1.5))

        p1 = p0 + 0.33 * diff + perp * offset_mag
        p2 = p0 + 0.67 * diff + perp * (offset_mag * 0.5)

        ts = np.array([0.0, 0.33, 0.67, 1.0], dtype=np.float64)
        pts = np.stack([p0, p1, p2, p3]).astype(np.float64)

        cs = CubicSpline(ts, pts, bc_type="natural")
        sample_params = np.linspace(0.0, 1.0, m)
        centerline = np.asarray(cs(sample_params), dtype=np.float64)

        # Pin endpoints exactly.
        centerline[0] = p0
        centerline[-1] = p3

        centerlines[(u, v)] = centerline

    return centerlines
