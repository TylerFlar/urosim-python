"""Analytic smooth normals from the kidney SDF gradient.

Mesh face normals produced by marching cubes are per-triangle and
visibly faceted; for shading we prefer per-vertex normals that follow
the underlying smooth implicit surface (including the Perlin mucosal
detail baked into :func:`urosim.anatomy.sdf.kidney_sdf`). This module
is a thin wrapper that evaluates the SDF gradient via central finite
differences and returns unit outward-pointing normals.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from urosim.anatomy.sdf import kidney_sdf, sdf_gradient


def compute_analytic_normals(
    vertices: np.ndarray,
    graph: nx.DiGraph,
    centerlines: dict[tuple[str, str], np.ndarray],
    pelvis_radii: tuple[float, float, float],
    eps: float = 0.1,
) -> np.ndarray:
    """Compute outward unit normals from the kidney SDF gradient.

    Evaluates ``kidney_sdf`` via central finite differences at each
    vertex and normalizes the result to unit length. For a well-formed
    SDF the gradient equals the outward surface normal, so this is the
    standard way to recover smooth normals from a purely implicit
    representation. ``kidney_sdf`` is only approximately unit-Lipschitz
    (Perlin detail is additive, see :func:`urosim.anatomy.sdf.kidney_sdf`)
    but the deviation is small and the resulting normals remain
    approximately outward-pointing for vertices on the isosurface.

    Args:
        vertices: Query points, shape ``(V, 3)`` in mm. Typically the
            vertex array of a mesh extracted by
            :func:`urosim.anatomy.mesh_extract.extract_kidney_mesh`, but
            any surface points are valid.
        graph: Collecting-system graph with ``type`` and ``position`` on
            every node and ``width_mm`` on every edge.
        centerlines: Mapping from edge ``(u, v)`` to a ``(M, 3)``
            sampled centerline.
        pelvis_radii: Ellipsoid half-axes ``(a, b, c)`` of the renal
            pelvis.
        eps: Finite-difference step in mm (default 0.1). Passed through
            to :func:`urosim.anatomy.sdf.sdf_gradient`.

    Returns:
        ``(V, 3)`` array of unit-length outward normals. Vertices where
        the raw gradient magnitude underflows are returned as the zero
        vector (inherited from :func:`sdf_gradient`).
    """

    def _sdf(points: np.ndarray) -> np.ndarray:
        return kidney_sdf(points, graph, centerlines, pelvis_radii)

    return sdf_gradient(_sdf, np.asarray(vertices, dtype=np.float64), eps=eps)
