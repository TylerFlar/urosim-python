"""Kidney stone placement inside the collecting system.

Produces :class:`StoneSpec` instances describing each stone's position,
size, composition, hit points, hardness, and rendering color. Draws on
the clinical priors in :mod:`urosim.anatomy.distributions` for the
stochastic fields (location, composition, size); hardness and color
are table-driven per composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from urosim.anatomy.distributions import (
    sample_stone_composition,
    sample_stone_location,
    sample_stone_size,
)

if TYPE_CHECKING:
    import trimesh

# Relative laser-fragmentation hardness. 1.0 is the hardest (whewellite
# monohydrate); softer stones have proportionally less hit-point volume
# so they fragment faster for the same stone size. These are design
# knobs for gameplay, not physical material constants.
_STONE_HARDNESS: dict[str, float] = {
    "CaOx_mono": 1.0,
    "CaOx_di": 0.6,
    "uric": 0.4,
    "struvite": 0.3,
    "cystine": 0.9,
}

# Base RGB color per composition (linear, 0-1 range). Matches the
# palette specified in the module spec.
_STONE_COLORS: dict[str, tuple[float, float, float]] = {
    "CaOx_mono": (0.3, 0.2, 0.1),
    "CaOx_di": (0.7, 0.6, 0.3),
    "uric": (0.8, 0.4, 0.2),
    "struvite": (0.9, 0.9, 0.85),
    "cystine": (0.7, 0.7, 0.3),
}

# Standard deviation of the positional jitter applied around each
# target node, in mm. Kept well below the ~4-6 mm calyx cup radius so
# stones land inside the cup.
_OFFSET_STD_MM: float = 0.5

# Tolerance (mm) used when verifying that a placed stone is within the
# collecting system via its distance to the target node. At 6 sigma
# this is effectively always satisfied for the default offset std.
_PLACEMENT_TOLERANCE_MM: float = 3.0


@dataclass(frozen=True)
class StoneSpec:
    """Descriptor for a single placed kidney stone.

    Attributes:
        position: Stone center in mm as a ``(3,)`` float64 array.
        radius_mm: Stone radius in mm (half of the sampled diameter).
        composition: Composition label — one of ``"CaOx_mono"``,
            ``"CaOx_di"``, ``"uric"``, ``"struvite"``, ``"cystine"``.
        hp: Relative hit points. Computed as
            ``(4/3) * pi * radius_mm**3 * hardness`` so large, hard
            stones take more laser time to fragment. The units are
            ``mm^3`` scaled by a dimensionless hardness factor — this
            is a gameplay durability score, not a physical mass.
        hardness: Laser fragmentation time multiplier in ``[0, 1]``
            (see :data:`_STONE_HARDNESS`).
        color: RGB color tuple for rendering, linear 0-1 range.
    """

    position: np.ndarray
    radius_mm: float
    composition: str
    hp: float
    hardness: float
    color: tuple[float, float, float]


def _collect_minor_calyces(graph: nx.DiGraph) -> tuple[list[str], dict[str, list[str]]]:
    """Return (all_minor_calyces_sorted, minors_bucketed_by_pole)."""
    all_minors: list[str] = []
    by_pole: dict[str, list[str]] = {"upper": [], "mid": [], "lower": []}
    for node in sorted(graph.nodes()):
        attrs = graph.nodes[node]
        if attrs.get("type") != "minor_calyx":
            continue
        all_minors.append(node)
        pole = attrs.get("pole")
        if pole in by_pole:
            by_pole[pole].append(node)
    for pole in by_pole:
        by_pole[pole].sort()
    return all_minors, by_pole


def _find_pelvis(graph: nx.DiGraph) -> str:
    for node in sorted(graph.nodes()):
        if graph.nodes[node].get("type") == "pelvis":
            return node
    raise ValueError("graph has no pelvis node")


def place_stones(
    graph: nx.DiGraph,
    mesh: trimesh.Trimesh,  # noqa: ARG001 — accepted for API symmetry; verification uses nodes.
    rng: np.random.Generator,
    num_stones: int | None = None,
) -> list[StoneSpec]:
    """Place kidney stones inside the collecting system.

    Samples 1-4 stone locations by default (or ``num_stones`` if
    provided), then for each stone draws a location label, a matching
    calyx (pelvis or minor calyx by pole), a small positional jitter,
    a composition, and a size. Hardness, HP, and color are looked up
    from per-composition tables.

    Determinism: the rng is consumed in a fixed per-stone order
    (location → target → offset → composition → size) and the number
    of draws per stone is constant — including on the fallback path
    where the sampled location has no matching minor calyces — so
    repeat calls with the same rng state produce identical results.

    Args:
        graph: Collecting-system graph with ``position`` and ``pole``
            populated. Minor calyces must carry a ``pole`` attribute.
        mesh: Kidney mesh. Accepted for API symmetry with other
            anatomy modules; placement verification uses the node
            position instead of ``trimesh.proximity`` so no optional
            rtree dependency is required.
        rng: numpy Generator — the sole source of randomness.
        num_stones: Number of stones to place. ``None`` (default) draws
            a random count in ``[1, 4]``.

    Returns:
        List of :class:`StoneSpec` in placement order.

    Raises:
        ValueError: If ``num_stones`` is negative, if the graph has no
            minor calyces, or if a stone's final position exceeds the
            placement tolerance relative to its target node.
    """
    if num_stones is not None and num_stones < 0:
        raise ValueError(f"num_stones must be >= 0; got {num_stones}")

    if num_stones is None:
        n = int(rng.integers(1, 5))
    else:
        n = int(num_stones)

    all_minors, by_pole = _collect_minor_calyces(graph)
    if not all_minors:
        raise ValueError("graph has no minor calyx nodes to place stones near")
    pelvis_node = _find_pelvis(graph)

    stones: list[StoneSpec] = []
    for _ in range(n):
        location = sample_stone_location(rng)

        if location == "pelvis":
            target = pelvis_node
        else:
            candidates = by_pole.get(location, [])
            if not candidates:
                candidates = all_minors
            # rng.choice always advances state by one draw.
            target = str(rng.choice(candidates))

        target_pos = np.asarray(graph.nodes[target]["position"], dtype=np.float64)
        offset = rng.normal(0.0, _OFFSET_STD_MM, size=3)
        position = target_pos + offset

        composition = sample_stone_composition(rng)
        diameter = sample_stone_size(rng)
        radius_mm = 0.5 * float(diameter)

        hardness = _STONE_HARDNESS[composition]
        hp = (4.0 / 3.0) * float(np.pi) * (radius_mm ** 3) * hardness
        color = _STONE_COLORS[composition]

        dist = float(np.linalg.norm(position - target_pos))
        if dist > _PLACEMENT_TOLERANCE_MM:
            raise ValueError(
                f"stone placement {dist:.2f} mm from target exceeds tolerance "
                f"{_PLACEMENT_TOLERANCE_MM} mm"
            )

        stones.append(
            StoneSpec(
                position=position,
                radius_mm=radius_mm,
                composition=composition,
                hp=hp,
                hardness=hardness,
                color=color,
            )
        )

    return stones
