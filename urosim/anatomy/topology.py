"""Kidney collecting-system topology construction.

This module assembles a directed graph describing the renal collecting
system — pelvis, major calyces, and minor calyces — from the clinical
parameter samplers in :mod:`urosim.anatomy.distributions`. The resulting
graph is a rooted arborescence (pelvis → majors → minors) with
infundibular length, width, and angle stored on every edge.

Determinism: :func:`build_topology` consumes the supplied
``numpy.random.Generator`` in a fixed order so that repeated calls with
the same ``pelvis_type`` and the same initial rng state return
structurally and numerically identical graphs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from urosim.anatomy.distributions import (
    sample_has_perpendicular_calyx,
    sample_infundibular_length,
    sample_infundibular_width,
    sample_infundibulopelvic_angle,
    sample_num_major_calyces,
    sample_num_minor_calyces,
)

_VALID_PELVIS_TYPES: frozenset[str] = frozenset({"A1", "A2", "A3", "B1", "B2"})
_MIN_TOTAL_MINORS: int = 7
_MAX_TOTAL_MINORS: int = 13
_PERPENDICULAR_WIDTH_LOW: float = 2.0
_PERPENDICULAR_WIDTH_HIGH: float = 3.9


def build_topology(pelvis_type: str, rng: np.random.Generator) -> nx.DiGraph:
    """Build a kidney collecting-system topology graph.

    Returns a directed arborescence rooted at a ``"pelvis"`` node. Major
    calyces branch off the pelvis, minor calyces branch off major calyces,
    and every edge carries infundibular morphometry (length, width, angle).

    The graph obeys the following invariants:

    - Root ``"pelvis"`` has attribute ``type="pelvis"`` and no pole.
    - Major-calyx nodes are named ``"major_<pole>"`` with ``type="major_calyx"``
      and a ``pole`` attribute in ``{"upper", "mid", "lower"}``.
    - Minor-calyx nodes are named ``"minor_<pole>_<i>"`` with
      ``type="minor_calyx"`` and a matching ``pole`` attribute. Minor calyces
      are always leaves (out-degree 0).
    - B-type pelvises carry exactly two major calyces (upper, lower).
      A-type pelvises carry two or three, sampled via
      :func:`~urosim.anatomy.distributions.sample_num_major_calyces`.
    - The total number of minor calyces across all majors lies in
      ``[7, 13]``; minor counts are resampled until that constraint holds.
    - Every edge has float attributes ``length_mm``, ``width_mm``,
      ``angle_deg`` that are strictly positive.
    - With probability ~0.114, one randomly selected minor calyx is
      marked ``is_perpendicular=True`` and its incoming-edge ``width_mm``
      is overridden to a draw from ``Uniform[2.0, 3.9)`` mm.

    Args:
        pelvis_type: A Sampaio classification label — one of
            ``"A1"``, ``"A2"``, ``"A3"``, ``"B1"``, ``"B2"``.
        rng: numpy ``Generator`` used as the sole randomness source.
            The function consumes draws in a fixed order, so the output
            is a pure function of ``pelvis_type`` and the generator's
            initial state.

    Returns:
        A :class:`networkx.DiGraph` representing the collecting-system
        topology of a single kidney.

    Raises:
        ValueError: If ``pelvis_type`` is not a recognized Sampaio label.
    """
    if pelvis_type not in _VALID_PELVIS_TYPES:
        raise ValueError(f"unknown pelvis_type: {pelvis_type!r}")

    # --- Step 1: decide major-calyx poles. ------------------------------
    # B-types are bifurcated → fixed upper/lower, no rng draw.
    # A-types draw a count (2 or 3); 2 maps to upper/lower, 3 to
    # upper/mid/lower. major_poles is kept as an ordered list so
    # downstream iteration is deterministic.
    if pelvis_type.startswith("B"):
        major_poles: list[str] = ["upper", "lower"]
    else:
        n_major = sample_num_major_calyces(pelvis_type, rng)
        major_poles = ["upper", "lower"] if n_major == 2 else ["upper", "mid", "lower"]

    # --- Step 2: sample minor-calyx counts in a resample loop. ----------
    # Every iteration draws all per-pole counts in the same fixed order,
    # so the rng state evolves monotonically and the loop is deterministic.
    # In 3-major configs the per-pole ranges sum to exactly [7, 13] and the
    # loop runs once; in 2-major configs it can iterate when the draw
    # lands below 7.
    minor_counts: dict[str, int] = {}
    while True:
        minor_counts = {pole: sample_num_minor_calyces(pole, rng) for pole in major_poles}
        total = sum(minor_counts.values())
        if _MIN_TOTAL_MINORS <= total <= _MAX_TOTAL_MINORS:
            break

    # --- Step 3: build the node set. ------------------------------------
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_node("pelvis", type="pelvis")
    for pole in major_poles:
        graph.add_node(f"major_{pole}", type="major_calyx", pole=pole)
    for pole in major_poles:
        for i in range(minor_counts[pole]):
            graph.add_node(f"minor_{pole}_{i}", type="minor_calyx", pole=pole)

    # --- Step 4: sample edge attributes in a fixed order. ---------------
    # For each pole (in major_poles order): pelvis→major, then
    # major→minor_0..minor_n. Each edge consumes three rng draws in the
    # order length, width, angle.
    for pole in major_poles:
        length_mm = sample_infundibular_length(pole, rng)
        width_mm = sample_infundibular_width(rng)
        angle_deg = sample_infundibulopelvic_angle(rng)
        graph.add_edge(
            "pelvis",
            f"major_{pole}",
            length_mm=length_mm,
            width_mm=width_mm,
            angle_deg=angle_deg,
        )
        for i in range(minor_counts[pole]):
            length_mm = sample_infundibular_length(pole, rng)
            width_mm = sample_infundibular_width(rng)
            angle_deg = sample_infundibulopelvic_angle(rng)
            graph.add_edge(
                f"major_{pole}",
                f"minor_{pole}_{i}",
                length_mm=length_mm,
                width_mm=width_mm,
                angle_deg=angle_deg,
            )

    # --- Step 5: perpendicular calyx. -----------------------------------
    # Always consume the flag draw so downstream rng state does not depend
    # on whether a perpendicular calyx ended up being added.
    has_perpendicular = sample_has_perpendicular_calyx(rng)
    if has_perpendicular:
        minor_nodes = sorted(
            node for node, data in graph.nodes(data=True) if data.get("type") == "minor_calyx"
        )
        chosen = str(rng.choice(minor_nodes))
        graph.nodes[chosen]["is_perpendicular"] = True
        parent = next(iter(graph.predecessors(chosen)))
        graph.edges[parent, chosen]["width_mm"] = float(
            rng.uniform(_PERPENDICULAR_WIDTH_LOW, _PERPENDICULAR_WIDTH_HIGH)
        )

    return graph
