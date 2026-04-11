"""Per-vertex procedural texture coordinates for the kidney mesh.

For each mesh vertex this module computes ``(t, theta, edge_idx)``:

* ``t`` — arc-length fraction along the owning infundibular centerline
  (``0`` at the parent end, ``1`` at the child end).
* ``theta`` — circumferential angle around the tube cross-section, in
  ``[0, 2*pi)``. Measured in a parallel-transported orthonormal frame
  along each centerline so ``theta`` is continuous as the tangent
  rotates.
* ``edge_idx`` — integer identifying the owning edge, assigned via
  ``sorted(centerlines.items())`` so it matches the deterministic
  reduction order used elsewhere in the anatomy pipeline.

These coordinates drive procedural texturing in Unity's Shader Graph —
e.g. veins following the tube axis (modulate by ``t``) or mucosal folds
wrapping circumferentially (modulate by ``theta``). The function is
pure: given the same inputs it returns bit-identical outputs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

_TINY: float = 1e-12


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit length, or zero if below ``_TINY``."""
    n = float(np.linalg.norm(vec))
    if n < _TINY:
        return np.zeros_like(vec)
    return vec / n


def _seed_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal frame ``(U, V)`` perpendicular to ``tangent``.

    Uses world ``+z`` as the reference up vector, falling back to
    ``+x`` when the tangent is nearly parallel to ``+z``.
    """
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = _unit(ref - np.dot(ref, tangent) * tangent)
    v = _unit(np.cross(tangent, u))
    return u, v


def _rotate_onto(vec: np.ndarray, t_from: np.ndarray, t_to: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the minimum rotation taking ``t_from`` onto ``t_to``.

    Uses Rodrigues' rotation formula. When the two tangents are
    effectively parallel the input is returned unchanged.
    """
    axis = np.cross(t_from, t_to)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < _TINY:
        return vec
    axis = axis / axis_norm
    cos_a = float(np.clip(np.dot(t_from, t_to), -1.0, 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
    return (
        vec * cos_a
        + np.cross(axis, vec) * sin_a
        + axis * float(np.dot(axis, vec)) * (1.0 - cos_a)
    )


def _build_edge_frames(
    line: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-sample frames for one centerline.

    Returns:
        Tuple ``(tangents, U, V, local_t)`` each of shape ``(M, 3)``
        (except ``local_t`` which has shape ``(M,)``). ``tangents`` are
        unit vectors using forward differences (last sample copies the
        previous tangent). ``U`` is parallel-transported from sample 0
        via Rodrigues' formula; ``V = tangent x U``. ``local_t`` is the
        cumulative chord length divided by the total chord length,
        clamped to ``[0, 1]``.
    """
    m = line.shape[0]
    tangents = np.zeros_like(line)
    for i in range(m - 1):
        tangents[i] = _unit(line[i + 1] - line[i])
    tangents[m - 1] = tangents[m - 2] if m >= 2 else np.array([1.0, 0.0, 0.0])

    u = np.zeros_like(line)
    v = np.zeros_like(line)
    u0, v0 = _seed_frame(tangents[0])
    u[0] = u0
    v[0] = v0
    for i in range(1, m):
        u_rot = _rotate_onto(u[i - 1], tangents[i - 1], tangents[i])
        # Re-orthogonalize against the new tangent to guard against drift.
        u_rot = _unit(u_rot - np.dot(u_rot, tangents[i]) * tangents[i])
        if np.linalg.norm(u_rot) < _TINY:
            u_rot, _ = _seed_frame(tangents[i])
        u[i] = u_rot
        v[i] = _unit(np.cross(tangents[i], u_rot))

    # Arc length → local_t.
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < _TINY:
        local_t = np.zeros(m)
    else:
        local_t = np.clip(cum / total, 0.0, 1.0)

    return tangents, u, v, local_t


def compute_texture_coords(
    vertices: np.ndarray,
    graph: nx.DiGraph,  # noqa: ARG001 — kept for API stability; graph is not needed yet.
    centerlines: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    """Compute ``[t, theta, edge_idx]`` for every vertex.

    Args:
        vertices: ``(V, 3)`` array of mesh vertex positions in mm.
        graph: Collecting-system graph. Currently accepted for API
            symmetry — all coordinate information comes from the
            centerlines themselves — but retained so future revisions
            can consult edge/node metadata without changing callers.
        centerlines: Mapping from each edge ``(u, v)`` to its sampled
            centerline ``(M, 3)``. Iterated in ``sorted`` order to
            assign reproducible ``edge_idx`` values.

    Returns:
        ``(V, 3)`` float64 array. Columns are ``t`` in ``[0, 1]``,
        ``theta`` in ``[0, 2*pi)``, and ``edge_idx`` in
        ``[0, num_edges)`` (stored as float but integer-valued).

    Raises:
        ValueError: If ``centerlines`` is empty or ``vertices`` does
            not have shape ``(V, 3)``.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3); got {verts.shape}")
    if not centerlines:
        raise ValueError("centerlines is empty")

    sorted_edges = sorted(centerlines.items())

    points_list: list[np.ndarray] = []
    edge_idx_list: list[np.ndarray] = []
    local_t_list: list[np.ndarray] = []
    u_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []

    for idx, (_edge, line) in enumerate(sorted_edges):
        line_arr = np.asarray(line, dtype=np.float64)
        if line_arr.ndim != 2 or line_arr.shape[1] != 3 or line_arr.shape[0] < 2:
            raise ValueError(
                f"centerline for edge {_edge} must have shape (M>=2, 3); got {line_arr.shape}"
            )
        _tangents, u, v, local_t = _build_edge_frames(line_arr)
        m = line_arr.shape[0]
        points_list.append(line_arr)
        edge_idx_list.append(np.full(m, idx, dtype=np.int64))
        local_t_list.append(local_t)
        u_list.append(u)
        v_list.append(v)

    points = np.concatenate(points_list, axis=0)
    edge_idx_all = np.concatenate(edge_idx_list, axis=0)
    local_t_all = np.concatenate(local_t_list, axis=0)
    u_all = np.concatenate(u_list, axis=0)
    v_all = np.concatenate(v_list, axis=0)

    tree = cKDTree(points)
    _dist, nearest = tree.query(verts, k=1)
    nearest = np.asarray(nearest, dtype=np.int64)

    offset = verts - points[nearest]
    u_sel = u_all[nearest]
    v_sel = v_all[nearest]
    theta = np.arctan2(
        np.einsum("ij,ij->i", offset, v_sel),
        np.einsum("ij,ij->i", offset, u_sel),
    )
    theta = np.mod(theta, 2.0 * np.pi)

    return np.column_stack(
        [
            local_t_all[nearest],
            theta,
            edge_idx_all[nearest].astype(np.float64),
        ]
    )
