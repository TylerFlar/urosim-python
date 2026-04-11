"""3D placement of kidney collecting-system nodes.

This module assigns a 3D ``position`` attribute to every node in a
collecting-system graph produced by :func:`urosim.anatomy.topology.build_topology`.
Positions are expressed in millimeters in a kidney-local frame:

- ``x`` — superior/inferior (upper pole is +x, lower pole is -x)
- ``y`` — anterior/posterior
- ``z`` — medial/lateral

The kidney is modeled as a bounding ellipsoid with half-axes
``(55, 25, 15) mm``. The pelvis sits near the medial hilum at the
origin; major calyces fan out toward their pole (upper/mid/lower); minor
calyces branch off their parent major by the edge's
``angle_deg`` at a distance of ``length_mm``. Small jitter is added for
organic irregularity and a simple pairwise relaxation enforces a minimum
inter-node separation.

Determinism: :func:`place_nodes_3d` consumes the supplied
``numpy.random.Generator`` in a fixed order so that repeated calls with
the same graph and the same initial rng state produce identical
positions.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# Kidney ellipsoid half-axes (a, b, c) in mm: x = sup-inf, y = ant-post, z = med-lat.
_KIDNEY_HALF_AXES: tuple[float, float, float] = (55.0, 25.0, 15.0)

# Jitter bounds (uniform ±) in mm.
_PELVIS_JITTER_MM: float = 2.0
_POSITION_JITTER_MM: float = 2.0

# Minimum pairwise separation between any two nodes.
_MIN_NODE_DISTANCE_MM: float = 2.0

# Maximum relaxation passes when resolving near-coincident nodes.
_MAX_RELAX_ITERS: int = 50

# Radial shell (in "ellipsoid-scale" units, where 1.0 is on the surface) in
# which minor calyces should land. Majors are capped below this shell so they
# remain in the renal sinus rather than at the periphery.
_MAJOR_MAX_SCALE: float = 0.70
_MINOR_MIN_SCALE: float = 0.70
_MINOR_MAX_SCALE: float = 0.95

# When a candidate position escapes the ellipsoid (val > 1), the fallback
# clamp lands it at this fraction of the surface.
_ESCAPE_CLAMP_SCALE: float = 0.95

# Fraction of the ray-ellipsoid intersection distance that a branch length is
# allowed to use. Keeps nodes from being pushed to the exact boundary where
# clamping would otherwise be very aggressive.
_MAJOR_LENGTH_FRACTION: float = 0.65
_MINOR_LENGTH_FRACTION: float = 0.90

# Base outward directions for each major-calyx pole (unit vectors).
_POLE_BASE_DIR: dict[str, np.ndarray] = {
    "upper": np.array([1.0, 0.0, 0.0]),
    "lower": np.array([-1.0, 0.0, 0.0]),
    # Mid-pole fans laterally (near x=0) rather than along sup-inf.
    "mid": np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]),
}


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return v / norm


def _perpendicular_unit(direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a random unit vector perpendicular to ``direction``.

    Uses Gram-Schmidt to project a random normal draw onto the plane
    orthogonal to ``direction``. Falls back to a deterministic axis if
    the draw happens to be parallel to ``direction``.
    """
    v = rng.normal(size=3)
    perp = v - np.dot(v, direction) * direction
    norm = float(np.linalg.norm(perp))
    if norm < 1e-9:
        # Extremely unlikely; pick any axis not parallel to direction.
        fallback = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(fallback, direction))) > 0.99:
            fallback = np.array([0.0, 0.0, 1.0])
        perp = fallback - np.dot(fallback, direction) * direction
        norm = float(np.linalg.norm(perp))
    return perp / norm


def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector ``v`` around unit ``axis`` by ``angle_rad`` radians."""
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return (
        v * cos_a
        + np.cross(axis, v) * sin_a
        + axis * float(np.dot(axis, v)) * (1.0 - cos_a)
    )


def _axes_array() -> np.ndarray:
    return np.array(_KIDNEY_HALF_AXES, dtype=np.float64)


def _ellipsoid_val(pos: np.ndarray) -> float:
    """Return ``(x/a)^2 + (y/b)^2 + (z/c)^2`` for the kidney ellipsoid.

    A value of 1.0 means the point is on the ellipsoid surface; smaller
    is strictly inside, larger is outside. The radial scale (``sqrt(val)``)
    is what we use to decide whether a node should be pushed inward or
    outward.
    """
    axes = _axes_array()
    return float(np.sum((pos / axes) ** 2))


def _ray_ellipsoid_t(p0: np.ndarray, direction: np.ndarray) -> float:
    """Distance from ``p0`` along unit ``direction`` to the ellipsoid surface.

    Solves the quadratic obtained by substituting
    ``p0 + t*direction`` into the ellipsoid equation. Returns the larger
    (exit) positive root, which is the distance at which a ray leaves
    the ellipsoid. Returns ``0.0`` if the ray never exits (e.g., when
    called from outside the ellipsoid in a pathological configuration).
    """
    axes = _axes_array()
    u0 = p0 / axes
    ud = direction / axes
    a_coef = float(np.dot(ud, ud))
    b_coef = 2.0 * float(np.dot(u0, ud))
    c_coef = float(np.dot(u0, u0)) - 1.0
    if a_coef < 1e-12:
        return 0.0
    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    if disc < 0.0:
        return 0.0
    sqrt_disc = float(np.sqrt(disc))
    t_exit = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    return max(t_exit, 0.0)


def _clamp_inside(pos: np.ndarray, target_scale: float = _ESCAPE_CLAMP_SCALE) -> np.ndarray:
    """Pull ``pos`` inside the ellipsoid if it has escaped.

    If the point is outside the ellipsoid (``val > 1``), scale it
    radially so it lands at ``target_scale`` of the surface. Otherwise
    leave it untouched.
    """
    val = _ellipsoid_val(pos)
    if val > 1.0:
        return pos * (target_scale / float(np.sqrt(val)))
    return pos


def _project_to_shell(
    pos: np.ndarray, low_scale: float, high_scale: float
) -> np.ndarray:
    """Radially rescale ``pos`` so its ellipsoid scale lies in [low, high].

    Points closer to the origin than ``low_scale`` are pushed outward;
    points outside ``high_scale`` are pulled inward. Pure origins are
    left alone (the caller should have avoided placing a node there).
    """
    val = _ellipsoid_val(pos)
    if val < 1e-12:
        return pos
    current = float(np.sqrt(val))
    if current > high_scale:
        return pos * (high_scale / current)
    if current < low_scale:
        return pos * (low_scale / current)
    return pos


def _reclamp_node(pos: np.ndarray, node_type: str) -> np.ndarray:
    """Apply the type-appropriate ellipsoid clamp to ``pos``.

    Used after each jitter/relaxation push to keep nodes from drifting
    outside the kidney (or, for minors, out of the peripheral shell).
    """
    if node_type == "minor_calyx":
        return _project_to_shell(pos, _MINOR_MIN_SCALE, _MINOR_MAX_SCALE)
    if node_type == "major_calyx":
        return _project_to_shell(pos, 0.0, _MAJOR_MAX_SCALE)
    return _clamp_inside(pos)


def place_nodes_3d(graph: nx.DiGraph, rng: np.random.Generator) -> None:
    """Assign 3D positions to every node in a collecting-system graph.

    Modifies ``graph`` in place, adding a ``position`` attribute
    (``numpy.ndarray`` of shape ``(3,)``, dtype ``float64``, in mm) to
    every node. Placement rules:

    - The pelvis is placed near the origin with ±2 mm uniform jitter,
      representing the medial hilum.
    - Major calyces fan outward from the pelvis along a pole-specific
      base direction (upper = +x, lower = -x, mid = lateral) at a
      distance of the incoming edge's ``length_mm``.
    - Minor calyces branch off their parent major calyx. The branch
      direction is the parent infundibulum direction (pelvis → major)
      rotated by the edge's ``angle_deg`` around a random perpendicular
      axis; the branch distance is capped by the ray-ellipsoid exit
      distance along that direction so a long ``length_mm`` cannot push
      the node far outside the kidney.
    - A ±2 mm uniform jitter is added to every non-pelvis node for
      organic irregularity.
    - Candidate positions are clamped to the kidney ellipsoid: majors
      are kept inside the renal sinus shell (radial scale ≤ 0.70),
      minors are projected into the peripheral shell (radial scale in
      ``[0.70, 0.95]``).
    - A simple pairwise relaxation jitters non-pelvis nodes that land
      within 2 mm of another node, up to ``_MAX_RELAX_ITERS`` passes;
      pushed nodes are re-clamped so relaxation cannot escape the
      ellipsoid.

    Args:
        graph: A directed arborescence from
            :func:`urosim.anatomy.topology.build_topology`. Must contain
            a ``"pelvis"`` root with children of type ``"major_calyx"``
            and grandchildren of type ``"minor_calyx"``. Edges must have
            ``length_mm`` and ``angle_deg`` attributes.
        rng: numpy ``Generator`` used as the sole randomness source.
            Consumed in a fixed order so that the output is a pure
            function of the graph and the generator's initial state.
    """
    positions: dict[str, np.ndarray] = {}

    # --- Step 1: pelvis near origin. ------------------------------------
    pelvis_pos = rng.uniform(-_PELVIS_JITTER_MM, _PELVIS_JITTER_MM, size=3).astype(
        np.float64
    )
    positions["pelvis"] = pelvis_pos

    # --- Step 2: major calyces. -----------------------------------------
    # Iterate in sorted order for deterministic rng consumption.
    major_nodes: list[str] = sorted(graph.successors("pelvis"))
    for major in major_nodes:
        pole = graph.nodes[major]["pole"]
        base_dir = _POLE_BASE_DIR[pole].astype(np.float64)

        # Small perturbation that preserves the pole intent: draw a normal
        # vector, project out its component along base_dir, and add.
        perturb = rng.normal(0.0, 1.0, size=3) * 0.15
        perturb = perturb - np.dot(perturb, base_dir) * base_dir
        direction = _normalize(base_dir + perturb)

        # Cap the branch length using the ray-ellipsoid intersection so
        # a long clinical length cannot push the major out of the sinus.
        clinical_length = float(graph.edges["pelvis", major]["length_mm"])
        t_exit = _ray_ellipsoid_t(pelvis_pos, direction)
        if t_exit > 0.0:
            branch_length = min(clinical_length, _MAJOR_LENGTH_FRACTION * t_exit)
        else:
            branch_length = clinical_length

        jitter = rng.uniform(
            -_POSITION_JITTER_MM, _POSITION_JITTER_MM, size=3
        ).astype(np.float64)
        candidate = pelvis_pos + direction * branch_length + jitter
        positions[major] = _project_to_shell(candidate, 0.0, _MAJOR_MAX_SCALE)

    # --- Step 3: minor calyces. -----------------------------------------
    for major in major_nodes:
        major_pos = positions[major]
        parent_dir = _normalize(major_pos - pelvis_pos)

        minor_nodes: list[str] = sorted(graph.successors(major))
        for minor in minor_nodes:
            # Random perpendicular axis to rotate around.
            axis = _perpendicular_unit(parent_dir, rng)

            angle_deg = float(graph.edges[major, minor]["angle_deg"])
            angle_rad = float(np.deg2rad(angle_deg))
            branch_dir = _normalize(_rodrigues_rotate(parent_dir, axis, angle_rad))

            clinical_length = float(graph.edges[major, minor]["length_mm"])
            t_exit = _ray_ellipsoid_t(major_pos, branch_dir)
            if t_exit > 0.0:
                branch_length = min(
                    clinical_length, _MINOR_LENGTH_FRACTION * t_exit
                )
            else:
                branch_length = clinical_length

            jitter = rng.uniform(
                -_POSITION_JITTER_MM, _POSITION_JITTER_MM, size=3
            ).astype(np.float64)
            candidate = major_pos + branch_dir * branch_length + jitter
            # Project into the peripheral shell so minors land near the
            # kidney surface (radial scale in [0.7, 0.95]).
            positions[minor] = _project_to_shell(
                candidate, _MINOR_MIN_SCALE, _MINOR_MAX_SCALE
            )

    # --- Step 4: relax pairwise min-distance constraint. ----------------
    # Small graph (~≤17 nodes) — naive O(n^2) is fine.
    node_order: list[str] = ["pelvis", *major_nodes]
    for major in major_nodes:
        node_order.extend(sorted(graph.successors(major)))

    node_types: dict[str, str] = {
        node: graph.nodes[node].get("type", "") for node in node_order
    }

    for _iter in range(_MAX_RELAX_ITERS):
        moved = False
        for i, a in enumerate(node_order):
            for b in node_order[i + 1 :]:
                dist = float(np.linalg.norm(positions[a] - positions[b]))
                if dist < _MIN_NODE_DISTANCE_MM:
                    # Push the later (non-pelvis) node; if b is "pelvis"
                    # (can't happen given node_order), push a instead.
                    push_target = b if b != "pelvis" else a
                    push = rng.uniform(-1.0, 1.0, size=3).astype(np.float64) * 2.0
                    pushed = positions[push_target] + push
                    # Re-clamp so relaxation can't kick a node outside
                    # the kidney or out of the peripheral shell.
                    positions[push_target] = _reclamp_node(
                        pushed, node_types[push_target]
                    )
                    moved = True
        if not moved:
            break

    # --- Step 5: write positions back to the graph. ---------------------
    for node, pos in positions.items():
        graph.nodes[node]["position"] = np.asarray(pos, dtype=np.float64)
