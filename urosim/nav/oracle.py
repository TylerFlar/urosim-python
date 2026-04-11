"""Navigation oracle — research API for evaluating algorithms on generated kidneys.

:class:`NavigationOracle` wraps a fully generated :class:`KidneyModel` and
exposes a small, stable surface used by nav / planning experiments and by
place-recognition evaluation. It depends only on :mod:`urosim.anatomy`,
:mod:`numpy`, :mod:`scipy`, and :mod:`networkx` — the module imports nothing
from ``client/``, ``grpc``, ``protobuf``, or ``torch``, matching the nav
boundary rules in ``CLAUDE.md``.

All query methods are pure; construction pre-computes the undirected topology
view and per-edge cumulative arc-length arrays so that :meth:`localize` can
vectorize the per-segment projection without repeated work.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from urosim.anatomy.kidney_model import KidneyModel


class NavigationOracle:
    """Ground-truth navigation queries over a generated kidney.

    The oracle pre-computes graph structures and centerline arc lengths
    once, then answers the following queries in O(graph) or O(centerline)
    time:

    - :meth:`coverage` — fraction of minor calyces visited.
    - :meth:`localize` — nearest centerline point to a 3D scope tip.
    - :meth:`optimal_tour` — DFS order visiting every minor calyx.
    - :meth:`path_between` — shortest graph path between two nodes.
    - :meth:`evaluate_place_recognition` — score predicted calyx labels.

    Attributes:
        None are part of the public API. Everything is stored on private
        underscore-prefixed fields; construct a new oracle if the
        underlying :class:`KidneyModel` changes.
    """

    def __init__(self, kidney: KidneyModel) -> None:
        """Pre-compute graph structures for fast queries.

        Args:
            kidney: Fully generated kidney model. Only :attr:`graph` and
                :attr:`centerlines` are consumed; the mesh, normals, and
                stones are ignored.
        """
        self._kidney: KidneyModel = kidney
        self._graph: nx.DiGraph = kidney.graph
        # An undirected view is sufficient for all our queries and avoids
        # copying node / edge attributes.
        self._undirected: nx.Graph = self._graph.to_undirected(as_view=True)
        self._calyx_ids: frozenset[str] = frozenset(kidney.calyx_ids())

        # Per-edge centerlines plus cumulative arc length, so that a
        # segment-local projection parameter can be mapped onto a single
        # edge-global t in [0, 1].
        self._centerlines: dict[tuple[str, str], np.ndarray] = {
            edge: np.asarray(line, dtype=np.float64)
            for edge, line in kidney.centerlines.items()
        }
        self._arclen: dict[tuple[str, str], np.ndarray] = {}
        for edge, line in self._centerlines.items():
            seg_lens = np.linalg.norm(np.diff(line, axis=0), axis=1)
            self._arclen[edge] = np.concatenate([[0.0], np.cumsum(seg_lens)])

    # ------------------------------------------------------------------
    # 1. coverage
    # ------------------------------------------------------------------
    def coverage(self, visited: set[str]) -> float:
        """Fraction of minor calyces visited.

        Args:
            visited: Set of node IDs that have been visited by the scope.
                Entries that are not minor calyces are silently ignored.

        Returns:
            ``len(visited ∩ minor_calyces) / len(minor_calyces)``. Returns
            ``0.0`` for an empty visited set and ``1.0`` when every minor
            calyx has been visited.
        """
        if not self._calyx_ids:
            return 0.0
        hits = len(visited & self._calyx_ids)
        return hits / len(self._calyx_ids)

    # ------------------------------------------------------------------
    # 2. localize
    # ------------------------------------------------------------------
    def localize(self, tip_pos: np.ndarray) -> tuple[tuple[str, str], float]:
        """Map a 3D scope tip position to the nearest centerline point.

        The returned ``edge_key`` is the ``(parent, child)`` tuple that
        keys :attr:`KidneyModel.centerlines`, and ``t`` is the edge-global
        arc-length parameter: ``t == 0`` at the parent end and ``t == 1``
        at the child end, interpolated linearly between sampled points.

        Args:
            tip_pos: Shape ``(3,)`` scope tip position in millimetres.

        Returns:
            ``(edge_key, t)``. ``t`` is clamped to ``[0, 1]`` by
            construction.

        Raises:
            ValueError: If the oracle has no centerlines at all.
        """
        if not self._centerlines:
            raise ValueError("NavigationOracle has no centerlines to localize against")

        pos = np.asarray(tip_pos, dtype=np.float64).reshape(3)

        best_d2: float = float("inf")
        best_edge: tuple[str, str] | None = None
        best_t: float = 0.0

        for edge, line in self._centerlines.items():
            # Vectorised per-segment closest-point projection.
            a = line[:-1]  # (M-1, 3) segment starts
            b = line[1:]  # (M-1, 3) segment ends
            ab = b - a  # (M-1, 3)
            ab_len2 = np.einsum("ij,ij->i", ab, ab)
            # Protect against zero-length segments (shouldn't happen given
            # the 2 mm target spacing, but be safe).
            safe_len2 = np.where(ab_len2 > 0, ab_len2, 1.0)
            s = np.einsum("ij,ij->i", pos - a, ab) / safe_len2
            s = np.clip(s, 0.0, 1.0)
            closest = a + s[:, None] * ab
            diff = closest - pos
            d2 = np.einsum("ij,ij->i", diff, diff)

            idx = int(np.argmin(d2))
            if d2[idx] < best_d2:
                best_d2 = float(d2[idx])
                best_edge = edge
                total = float(self._arclen[edge][-1])
                if total <= 0.0:
                    best_t = 0.0
                else:
                    seg_start = float(self._arclen[edge][idx])
                    seg_len = float(self._arclen[edge][idx + 1] - self._arclen[edge][idx])
                    best_t = (seg_start + s[idx] * seg_len) / total

        assert best_edge is not None  # guaranteed by the empty check above
        # Numerical clamp — the formula above can produce ~1 + eps on the
        # final segment.
        best_t = float(min(max(best_t, 0.0), 1.0))
        return best_edge, best_t

    # ------------------------------------------------------------------
    # 3. optimal_tour
    # ------------------------------------------------------------------
    def optimal_tour(self, start: str = "pelvis") -> list[str]:
        """Visitation order reaching every minor calyx from ``start``.

        The topology is a tree, so this is a pre-order DFS rather than a
        true TSP. Neighbours are visited in sorted order for determinism.

        The returned list contains every node reached by the DFS in
        first-visit order. Minor-calyx IDs are a subset of that list;
        intermediate nodes (pelvis, major calyces) are also included,
        which matches the spec.

        Args:
            start: Starting node ID. Defaults to ``"pelvis"``.

        Returns:
            List of node IDs in the order they are first reached.

        Raises:
            ValueError: If ``start`` is not a node in the topology graph.
        """
        if start not in self._graph.nodes:
            raise ValueError(f"start node {start!r} is not in the topology graph")

        order: list[str] = []
        seen: set[str] = set()
        stack: list[str] = [start]

        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            order.append(node)
            # Sorting in reverse so that the smallest neighbour is visited
            # first after being pushed onto the stack (stack → LIFO).
            neighbours = sorted(self._undirected.neighbors(node), reverse=True)
            for nbr in neighbours:
                if nbr not in seen:
                    stack.append(nbr)

        return order

    # ------------------------------------------------------------------
    # 4. path_between
    # ------------------------------------------------------------------
    def path_between(self, a: str, b: str) -> list[str]:
        """Shortest topology path between two nodes.

        Uses :func:`networkx.shortest_path` on the undirected view of the
        graph, so direction (pelvis → calyx vs. calyx → pelvis) does not
        matter.

        Args:
            a: Source node ID.
            b: Target node ID.

        Returns:
            List of node IDs from ``a`` to ``b`` inclusive. Returns
            ``[a]`` when ``a == b``.

        Raises:
            ValueError: If either endpoint is not in the topology graph.
        """
        if a not in self._graph.nodes:
            raise ValueError(f"node {a!r} is not in the topology graph")
        if b not in self._graph.nodes:
            raise ValueError(f"node {b!r} is not in the topology graph")
        if a == b:
            return [a]
        return list(nx.shortest_path(self._undirected, source=a, target=b))

    # ------------------------------------------------------------------
    # 5. evaluate_place_recognition
    # ------------------------------------------------------------------
    def evaluate_place_recognition(
        self,
        predictions: list[tuple[np.ndarray, str]],
        ground_truth: list[tuple[np.ndarray, str]],
    ) -> dict[str, Any]:
        """Score a place-recognition model on parallel prediction / truth lists.

        Each entry is ``(position, calyx_id)``. The two lists are iterated
        in parallel; positions are metadata (not compared by equality to
        avoid floating-point brittleness).

        For each entry the topological error is ``0`` when the prediction
        matches ground truth, otherwise the number of edges between the
        predicted and true calyx in the undirected graph. Unknown IDs
        (not present in the graph) produce a topological error of ``-1``.

        Args:
            predictions: Model outputs as ``(position, predicted_id)``.
            ground_truth: True labels as ``(position, true_id)``. Must have
                the same length as ``predictions``.

        Returns:
            Dict with keys ``accuracy`` (float in ``[0, 1]``), ``correct``
            (int), ``total`` (int), ``topological_errors`` (list of int
            per entry), and ``mean_topological_error`` (float, mean over
            all entries including correct ones).

        Raises:
            ValueError: If ``predictions`` and ``ground_truth`` have
                different lengths.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"length mismatch: predictions={len(predictions)}, "
                f"ground_truth={len(ground_truth)}"
            )

        total = len(predictions)
        if total == 0:
            return {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "topological_errors": [],
                "mean_topological_error": 0.0,
            }

        correct = 0
        topo_errors: list[int] = []
        for (_pred_pos, pred_id), (_gt_pos, gt_id) in zip(
            predictions, ground_truth, strict=True
        ):
            if pred_id == gt_id:
                correct += 1
                topo_errors.append(0)
                continue
            if pred_id in self._graph.nodes and gt_id in self._graph.nodes:
                try:
                    dist = int(
                        nx.shortest_path_length(self._undirected, pred_id, gt_id)
                    )
                except nx.NetworkXNoPath:
                    dist = -1
            else:
                dist = -1
            topo_errors.append(dist)

        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "topological_errors": topo_errors,
            "mean_topological_error": float(np.mean(topo_errors)),
        }
