"""One-call kidney anatomy generation from a seed.

:class:`AnatomyGenerator` orchestrates the full anatomy pipeline —
topology, 3D placement, centerlines, SDF + marching-cubes mesh
extraction, analytic normals, texture coordinates, calyx coverage
points, and stone placement — and returns a :class:`KidneyModel` that
bundles every artifact together with the sampled parameters needed to
reproduce it.

Each stochastic step owns its own RNG, derived from the caller's seed
via a distinct integer offset (``np.random.default_rng(seed + offset)``).
This means modifying the RNG consumption of one step does not shift
the randomness of any later step — tweaks to e.g. centerline curvature
will not perturb previously-sampled stone placements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from urosim.anatomy.centerlines import generate_centerlines
from urosim.anatomy.coverage_points import generate_coverage_points
from urosim.anatomy.distributions import sample_pelvis_radius, sample_pelvis_type
from urosim.anatomy.kidney_model import KidneyModel
from urosim.anatomy.mesh_extract import extract_kidney_mesh
from urosim.anatomy.normals import compute_analytic_normals
from urosim.anatomy.placement import place_nodes_3d
from urosim.anatomy.stones import place_stones
from urosim.anatomy.texture_coords import compute_texture_coords
from urosim.anatomy.topology import build_topology

# Per-step RNG offsets. Distinct, non-zero so that changing step N's
# consumption cannot propagate into step N+1, and so that ``seed=0``
# still yields well-defined RNGs for every step.
_OFFSET_PELVIS_TYPE: int = 1
_OFFSET_PELVIS_RADII: int = 2
_OFFSET_TOPOLOGY: int = 3
_OFFSET_PLACEMENT: int = 4
_OFFSET_CENTERLINES: int = 5
_OFFSET_STONES: int = 6

# Placeholder scope-entry offset: 15 mm inferior to the pelvis node
# along the kidney sup-inf (+x) axis. Matches the convention used in
# :mod:`urosim.anatomy.placement`.
_URETER_ENTRY_OFFSET_MM: np.ndarray = np.array([-15.0, 0.0, 0.0], dtype=np.float64)


class AnatomyGenerator:
    """Generate a complete :class:`KidneyModel` from an integer seed.

    Example:
        >>> gen = AnatomyGenerator(seed=42)
        >>> kidney = gen.generate()
        >>> kidney.mesh.is_watertight
        True
    """

    def __init__(self, seed: int) -> None:
        """Initialize the generator.

        Args:
            seed: Integer seed. Every stochastic step derives its own
                RNG from ``seed + <step offset>`` so the generation is
                fully deterministic.
        """
        self.seed: int = int(seed)

    def _rng(self, offset: int) -> np.random.Generator:
        """Return a fresh ``numpy.random.Generator`` for the given step offset."""
        return np.random.default_rng(self.seed + offset)

    def generate(self) -> KidneyModel:
        """Run the full anatomy pipeline and return a :class:`KidneyModel`.

        Pipeline (each stochastic step uses an independent RNG):

        1. Sample pelvis type.
        2. Sample pelvis ellipsoid radii.
        3. Build topology graph.
        4. Place nodes in 3D (in-place).
        5. Generate per-edge centerlines.
        6. Extract mesh via SDF + marching cubes (deterministic).
        7. Compute analytic normals (deterministic).
        8. Compute texture coordinates (deterministic).
        9. Generate calyx coverage points (deterministic).
        10. Place kidney stones.
        11. Assemble :class:`KidneyModel`.

        Returns:
            A fully-populated :class:`KidneyModel`.
        """
        # Step 1: pelvis type.
        pelvis_type = sample_pelvis_type(self._rng(_OFFSET_PELVIS_TYPE))

        # Step 2: pelvis radii.
        pelvis_radii = sample_pelvis_radius(pelvis_type, self._rng(_OFFSET_PELVIS_RADII))

        # Step 3: topology graph.
        graph = build_topology(pelvis_type, self._rng(_OFFSET_TOPOLOGY))

        # Step 4: 3D placement (mutates graph in-place).
        place_nodes_3d(graph, self._rng(_OFFSET_PLACEMENT))

        # Step 5: centerlines.
        centerlines = generate_centerlines(graph, self._rng(_OFFSET_CENTERLINES))

        # Step 6: mesh extraction — deterministic given graph + centerlines + radii.
        mesh = extract_kidney_mesh(graph, centerlines, pelvis_radii)

        # Step 7: analytic normals from SDF gradient — deterministic.
        normals = compute_analytic_normals(
            np.asarray(mesh.vertices), graph, centerlines, pelvis_radii
        )

        # Step 8: procedural texture coordinates — deterministic.
        texture_coords = compute_texture_coords(
            np.asarray(mesh.vertices), graph, centerlines
        )

        # Step 9: calyx coverage points — deterministic.
        calyx_coverage_points = generate_coverage_points(graph, mesh)

        # Step 10: stone placement.
        stones = place_stones(graph, mesh, self._rng(_OFFSET_STONES))

        # Step 11: derived attributes and assembly.
        calyx_poses: dict[str, np.ndarray] = {
            node: np.asarray(graph.nodes[node]["position"], dtype=np.float64).copy()
            for node in graph.nodes
            if graph.nodes[node].get("type") == "minor_calyx"
        }

        pelvis_position = np.asarray(
            graph.nodes["pelvis"]["position"], dtype=np.float64
        )
        ureter_entry = pelvis_position + _URETER_ENTRY_OFFSET_MM

        metadata: dict[str, Any] = {
            "seed": self.seed,
            "pelvis_type": pelvis_type,
            "pelvis_radii": [float(r) for r in pelvis_radii],
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
            "num_stones": int(len(stones)),
            "num_minor_calyces": int(len(calyx_poses)),
        }

        return KidneyModel(
            seed=self.seed,
            pelvis_type=pelvis_type,
            mesh=mesh,
            graph=graph,
            stones=stones,
            ureter_entry=ureter_entry,
            centerlines=centerlines,
            calyx_poses=calyx_poses,
            calyx_coverage_points=calyx_coverage_points,
            texture_coords=texture_coords,
            normals=normals,
            metadata=metadata,
        )
