"""Per-minor-calyx surface sample points for coverage tracking.

For each minor calyx node this module returns a small set of points on
the mesh surface near that calyx. The simulator uses these points to
decide whether the scope camera is observing a given calyx — e.g. by
counting how many of a calyx's coverage points fall inside the view
frustum and are unoccluded.

The function is a pure function of its inputs (no rng): a deterministic
stride-based subsampling is used when more candidate vertices than
``points_per_calyx`` are available so repeat calls are bit-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    import trimesh


def generate_coverage_points(
    graph: nx.DiGraph,
    mesh: trimesh.Trimesh,
    points_per_calyx: int = 30,
) -> dict[str, np.ndarray]:
    """Sample mesh vertices near each minor calyx.

    For every minor calyx node, vertices within a ball of radius
    ``width_mm`` (the incoming edge's infundibular width) of the
    calyx's position are collected. If no vertices land in that ball
    the search radius is bumped to ``1.5 * width_mm``. If the candidate
    count exceeds ``points_per_calyx`` the result is subsampled with a
    deterministic stride so output is reproducible.

    Args:
        graph: Collecting-system graph with ``type`` and ``position``
            on every node and ``width_mm`` on every edge. Minor calyces
            must have a unique predecessor (the anatomy pipeline
            guarantees this).
        mesh: Kidney mesh produced by
            :func:`urosim.anatomy.mesh_extract.extract_kidney_mesh`.
            Only ``mesh.vertices`` is read.
        points_per_calyx: Maximum number of points per calyx. Default
            30.

    Returns:
        Dict mapping each minor calyx node id to an ``(M, 3)`` array
        of sampled 3D points with ``M <= points_per_calyx``. Calyces
        with no nearby mesh vertices map to an empty ``(0, 3)`` array.
    """
    if points_per_calyx <= 0:
        raise ValueError(f"points_per_calyx must be positive; got {points_per_calyx}")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    result: dict[str, np.ndarray] = {}

    for node in sorted(graph.nodes()):
        attrs = graph.nodes[node]
        if attrs.get("type") != "minor_calyx":
            continue

        preds = list(graph.predecessors(node))
        if not preds:
            result[node] = np.empty((0, 3), dtype=np.float64)
            continue
        width_mm = float(graph.edges[preds[0], node]["width_mm"])
        pos = np.asarray(attrs["position"], dtype=np.float64)

        dists = np.linalg.norm(verts - pos, axis=1)
        mask = dists <= width_mm
        if not mask.any():
            mask = dists <= 1.5 * width_mm
        candidates = verts[mask]

        count = candidates.shape[0]
        if count == 0:
            result[node] = np.empty((0, 3), dtype=np.float64)
            continue
        if count <= points_per_calyx:
            result[node] = candidates.copy()
            continue

        idx = np.unique(
            np.linspace(0, count - 1, points_per_calyx).astype(np.int64)
        )
        result[node] = candidates[idx]

    return result
