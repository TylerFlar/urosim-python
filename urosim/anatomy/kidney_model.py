"""Container and persistence for a fully-generated kidney.

This module defines :class:`KidneyModel`, a dataclass that bundles every
artifact produced by :class:`urosim.anatomy.generator.AnatomyGenerator` —
mesh, topology graph, centerlines, per-vertex normals / texture
coordinates, calyx coverage points, and stone placements — together
with the sampled parameters needed to reproduce it.

:meth:`KidneyModel.save` writes the model to a group of sibling files
sharing a common base path, and :meth:`KidneyModel.load` reconstructs
an equivalent :class:`KidneyModel` from those files. The persistence
format uses human-inspectable JSON for metadata and small structures
(graph, stones, calyx poses, coverage points), a Wavefront OBJ for the
mesh, and numpy files for the large per-vertex arrays and centerlines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import trimesh

from urosim.anatomy.stones import StoneSpec

_EDGE_KEY_SEP: str = "|"


def _stone_to_dict(stone: StoneSpec) -> dict[str, Any]:
    """Serialize a :class:`StoneSpec` to a JSON-safe dict."""
    return {
        "position": np.asarray(stone.position, dtype=np.float64).tolist(),
        "radius_mm": float(stone.radius_mm),
        "composition": str(stone.composition),
        "hp": float(stone.hp),
        "hardness": float(stone.hardness),
        "color": [float(c) for c in stone.color],
    }


def _stone_from_dict(data: dict[str, Any]) -> StoneSpec:
    """Reconstruct a :class:`StoneSpec` from a dict produced by :func:`_stone_to_dict`."""
    return StoneSpec(
        position=np.asarray(data["position"], dtype=np.float64),
        radius_mm=float(data["radius_mm"]),
        composition=str(data["composition"]),
        hp=float(data["hp"]),
        hardness=float(data["hardness"]),
        color=tuple(float(c) for c in data["color"]),  # type: ignore[arg-type]
    )


def _graph_to_jsonable(graph: nx.DiGraph) -> dict[str, Any]:
    """Convert ``graph`` to a JSON-safe dict via ``networkx.node_link_data``.

    Node ``position`` attributes (numpy arrays) are converted to lists so
    the result survives a ``json.dumps`` round-trip. All other attributes
    are floats / strings / bools and are already JSON-safe.
    """
    data = nx.node_link_data(graph, edges="edges")
    for node_entry in data["nodes"]:
        pos = node_entry.get("position")
        if pos is not None:
            node_entry["position"] = np.asarray(pos, dtype=np.float64).tolist()
    return data


def _graph_from_jsonable(data: dict[str, Any]) -> nx.DiGraph:
    """Inverse of :func:`_graph_to_jsonable`."""
    graph = nx.node_link_graph(data, directed=True, edges="edges")
    for node in graph.nodes:
        pos = graph.nodes[node].get("position")
        if pos is not None:
            graph.nodes[node]["position"] = np.asarray(pos, dtype=np.float64)
    return graph


def _edge_key_to_str(edge: tuple[str, str]) -> str:
    """Encode a ``(u, v)`` tuple as a single string using ``|`` as separator."""
    u, v = edge
    if _EDGE_KEY_SEP in u or _EDGE_KEY_SEP in v:
        raise ValueError(f"edge node names must not contain {_EDGE_KEY_SEP!r}: {edge!r}")
    return f"{u}{_EDGE_KEY_SEP}{v}"


def _edge_key_from_str(key: str) -> tuple[str, str]:
    """Inverse of :func:`_edge_key_to_str`."""
    u, v = key.split(_EDGE_KEY_SEP, 1)
    return (u, v)


@dataclass
class KidneyModel:
    """Complete kidney anatomy produced by :class:`AnatomyGenerator`.

    Attributes:
        seed: Integer seed that produced this kidney.
        pelvis_type: Sampaio pelvis classification — one of
            ``"A1"``, ``"A2"``, ``"A3"``, ``"B1"``, ``"B2"``.
        mesh: Watertight triangle mesh of the collecting-system surface.
        graph: Topology graph with 3D positions on every node and
            infundibular metrics on every edge.
        stones: Placed kidney stones (1-4 per kidney by default).
        ureter_entry: ``(3,)`` float64 position in mm of the scope entry
            point. Placeholder: the pelvis node position offset 15 mm
            inferiorly along the kidney sup-inf axis. A proper ureter
            model is not yet implemented.
        centerlines: Per-edge sampled centerlines as ``(M, 3)`` float64
            arrays, keyed by ``(parent_node, child_node)`` tuples.
        calyx_poses: Per-minor-calyx 3D positions as ``(3,)`` arrays.
        calyx_coverage_points: Per-minor-calyx ``(M, 3)`` sample points
            lying on the mesh near each calyx cup (``M <= 30``).
        texture_coords: ``(V, 3)`` float64 array of
            ``[t, theta, edge_idx]`` per vertex (see
            :func:`urosim.anatomy.texture_coords.compute_texture_coords`).
        normals: ``(V, 3)`` unit-length outward normals from the SDF
            gradient.
        metadata: Sampled parameters and derived counts for
            reproducibility and inspection.
    """

    seed: int
    pelvis_type: str
    mesh: trimesh.Trimesh
    graph: nx.DiGraph
    stones: list[StoneSpec]
    ureter_entry: np.ndarray
    centerlines: dict[tuple[str, str], np.ndarray]
    calyx_poses: dict[str, np.ndarray]
    calyx_coverage_points: dict[str, np.ndarray]
    texture_coords: np.ndarray
    normals: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def calyx_ids(self) -> list[str]:
        """Return sorted list of all minor-calyx node IDs in the graph."""
        return sorted(
            n for n in self.graph.nodes if self.graph.nodes[n].get("type") == "minor_calyx"
        )

    def save(self, path: Path) -> None:
        """Save the model to ``path`` as a group of sibling files.

        Creates (or overwrites) the following files:

        - ``{path}.json`` — seed, pelvis_type, ureter_entry, graph
          (networkx node-link), stones, calyx_poses, metadata.
        - ``{path}.obj`` — Wavefront OBJ of :attr:`mesh`.
        - ``{path}_normals.npy`` — ``(V, 3)`` analytic normals.
        - ``{path}_texcoords.npy`` — ``(V, 3)`` texture coordinates.
        - ``{path}_coverage.json`` — per-calyx coverage points.
        - ``{path}_centerlines.npz`` — per-edge centerlines keyed by
          ``"parent|child"``.

        The parent directory is created if it does not exist.

        Args:
            path: Base path without extension. Sibling files are written
                next to it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        main_doc: dict[str, Any] = {
            "seed": int(self.seed),
            "pelvis_type": str(self.pelvis_type),
            "ureter_entry": np.asarray(self.ureter_entry, dtype=np.float64).tolist(),
            "graph": _graph_to_jsonable(self.graph),
            "stones": [_stone_to_dict(s) for s in self.stones],
            "calyx_poses": {
                k: np.asarray(v, dtype=np.float64).tolist() for k, v in self.calyx_poses.items()
            },
            "metadata": self.metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(main_doc, indent=2))

        self.mesh.export(str(path.with_suffix(".obj")), file_type="obj")

        np.save(path.parent / f"{path.name}_normals.npy", np.asarray(self.normals))
        np.save(path.parent / f"{path.name}_texcoords.npy", np.asarray(self.texture_coords))

        coverage_doc = {
            k: np.asarray(v, dtype=np.float64).tolist()
            for k, v in self.calyx_coverage_points.items()
        }
        (path.parent / f"{path.name}_coverage.json").write_text(json.dumps(coverage_doc, indent=2))

        centerlines_npz: dict[str, np.ndarray] = {
            _edge_key_to_str(edge): np.asarray(line, dtype=np.float64)
            for edge, line in self.centerlines.items()
        }
        np.savez(
            path.parent / f"{path.name}_centerlines.npz",
            **centerlines_npz,  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: Path) -> KidneyModel:
        """Load a :class:`KidneyModel` previously written by :meth:`save`.

        Args:
            path: Base path passed to :meth:`save` (without extension).

        Returns:
            A :class:`KidneyModel` functionally equivalent to the one
            that was saved.
        """
        path = Path(path)

        main_doc = json.loads(path.with_suffix(".json").read_text())
        seed = int(main_doc["seed"])
        pelvis_type = str(main_doc["pelvis_type"])
        ureter_entry = np.asarray(main_doc["ureter_entry"], dtype=np.float64)
        graph = _graph_from_jsonable(main_doc["graph"])
        stones = [_stone_from_dict(s) for s in main_doc["stones"]]
        calyx_poses = {
            k: np.asarray(v, dtype=np.float64) for k, v in main_doc["calyx_poses"].items()
        }
        metadata = dict(main_doc.get("metadata", {}))

        mesh_obj = trimesh.load(str(path.with_suffix(".obj")), process=False, force="mesh")
        if not isinstance(mesh_obj, trimesh.Trimesh):
            raise TypeError(
                f"loaded mesh is not a trimesh.Trimesh (got {type(mesh_obj).__name__})"
            )
        mesh: trimesh.Trimesh = mesh_obj

        normals = np.load(path.parent / f"{path.name}_normals.npy")
        texture_coords = np.load(path.parent / f"{path.name}_texcoords.npy")

        coverage_doc = json.loads((path.parent / f"{path.name}_coverage.json").read_text())
        calyx_coverage_points = {
            k: np.asarray(v, dtype=np.float64).reshape(-1, 3) for k, v in coverage_doc.items()
        }

        centerlines: dict[tuple[str, str], np.ndarray] = {}
        with np.load(path.parent / f"{path.name}_centerlines.npz") as npz:
            for key in npz.files:
                centerlines[_edge_key_from_str(key)] = np.asarray(npz[key], dtype=np.float64)

        return cls(
            seed=seed,
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
