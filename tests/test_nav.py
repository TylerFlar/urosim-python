"""Tests for :class:`urosim.nav.oracle.NavigationOracle`.

Every test is parametrized over three seeds (``0``, ``42``, ``99``) so the
oracle is exercised against multiple distinct kidneys. The kidney itself is
generated once per seed per module via a module-scoped fixture, which keeps
the (relatively expensive) mesh extraction out of the per-test loop.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from urosim.anatomy.generator import AnatomyGenerator
from urosim.anatomy.kidney_model import KidneyModel
from urosim.nav.oracle import NavigationOracle

SEEDS: list[int] = [0, 42, 99]


@pytest.fixture(scope="module", params=SEEDS, ids=[f"seed={s}" for s in SEEDS])
def kidney(request: pytest.FixtureRequest) -> KidneyModel:
    """Deterministically generate a kidney for each seed in :data:`SEEDS`."""
    seed: int = request.param
    return AnatomyGenerator(seed=seed).generate()


@pytest.fixture(scope="module")
def oracle(kidney: KidneyModel) -> NavigationOracle:
    """NavigationOracle wrapping the module-scoped kidney."""
    return NavigationOracle(kidney)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def test_coverage_empty_is_zero(oracle: NavigationOracle) -> None:
    assert oracle.coverage(set()) == 0.0


def test_coverage_all_calyces_is_one(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    assert oracle.coverage(set(kidney.calyx_ids())) == 1.0


def test_coverage_partial_matches_expected_fraction(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyces = kidney.calyx_ids()
    assert len(calyces) >= 2, "generator should produce ≥ 2 minor calyces"
    half = set(calyces[: len(calyces) // 2])
    expected = len(half) / len(calyces)
    assert oracle.coverage(half) == pytest.approx(expected)


def test_coverage_ignores_non_calyx_ids(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyces = kidney.calyx_ids()
    # Mix in a pelvis node, a major-calyx node (if present), and a bogus ID.
    major_ids = [
        n
        for n in kidney.graph.nodes
        if kidney.graph.nodes[n].get("type") == "major_calyx"
    ]
    mixed: set[str] = {calyces[0], "pelvis", "definitely_not_a_node"}
    if major_ids:
        mixed.add(major_ids[0])
    expected = 1 / len(calyces)
    assert oracle.coverage(mixed) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# localize
# ---------------------------------------------------------------------------
def test_localize_pelvis_position_returns_pelvis_adjacent_edge(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    pelvis_pos = kidney.graph.nodes["pelvis"]["position"]
    edge, t = oracle.localize(pelvis_pos)
    # The winning edge must touch the pelvis node, because every centerline
    # out of pelvis has its first sample pinned exactly to the pelvis
    # position → squared distance of zero.
    assert "pelvis" in edge
    assert 0.0 <= t <= 1.0
    # The pelvis position lies on the parent end of an outgoing edge, so t
    # should be effectively zero.
    if edge[0] == "pelvis":
        assert t == pytest.approx(0.0, abs=1e-6)


def test_localize_calyx_position_returns_own_edge_at_t_one(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyx_id = kidney.calyx_ids()[0]
    calyx_pos = kidney.graph.nodes[calyx_id]["position"]
    edge, t = oracle.localize(calyx_pos)
    # Minor calyces are leaves → they appear only as the *child* end of
    # exactly one edge, with centerline[-1] pinned to the calyx position.
    assert edge[1] == calyx_id
    assert t == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# optimal_tour
# ---------------------------------------------------------------------------
def test_optimal_tour_visits_every_minor_calyx(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    tour = oracle.optimal_tour()
    calyx_set = set(kidney.calyx_ids())
    assert calyx_set.issubset(set(tour))


def test_optimal_tour_nodes_are_all_valid_graph_nodes(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    tour = oracle.optimal_tour()
    graph_nodes = set(kidney.graph.nodes)
    for node in tour:
        assert node in graph_nodes


def test_optimal_tour_starts_at_start_node(oracle: NavigationOracle) -> None:
    tour = oracle.optimal_tour(start="pelvis")
    assert tour[0] == "pelvis"


def test_optimal_tour_rejects_unknown_start(oracle: NavigationOracle) -> None:
    with pytest.raises(ValueError):
        oracle.optimal_tour(start="not_a_node")


# ---------------------------------------------------------------------------
# path_between
# ---------------------------------------------------------------------------
def test_path_between_self_is_single_node(oracle: NavigationOracle) -> None:
    assert oracle.path_between("pelvis", "pelvis") == ["pelvis"]


def test_path_between_endpoints_match_inputs(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyx = kidney.calyx_ids()[0]
    path = oracle.path_between("pelvis", calyx)
    assert path[0] == "pelvis"
    assert path[-1] == calyx


def test_path_between_consecutive_nodes_are_connected(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyces = kidney.calyx_ids()
    assert len(calyces) >= 2
    a, b = calyces[0], calyces[-1]
    path = oracle.path_between(a, b)
    undirected = kidney.graph.to_undirected(as_view=True)
    for u, v in zip(path[:-1], path[1:], strict=True):
        assert undirected.has_edge(u, v)


def test_path_between_matches_networkx_shortest_path(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyces = kidney.calyx_ids()
    a, b = calyces[0], calyces[-1]
    expected = nx.shortest_path(kidney.graph.to_undirected(as_view=True), a, b)
    assert oracle.path_between(a, b) == list(expected)


def test_path_between_rejects_unknown_nodes(oracle: NavigationOracle) -> None:
    with pytest.raises(ValueError):
        oracle.path_between("pelvis", "not_a_node")
    with pytest.raises(ValueError):
        oracle.path_between("not_a_node", "pelvis")


# ---------------------------------------------------------------------------
# evaluate_place_recognition
# ---------------------------------------------------------------------------
def _gt_entries(kidney: KidneyModel) -> list[tuple[np.ndarray, str]]:
    return [
        (np.asarray(kidney.graph.nodes[c]["position"], dtype=np.float64), c)
        for c in kidney.calyx_ids()
    ]


def test_eval_place_recognition_all_correct(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    gt = _gt_entries(kidney)
    pred = list(gt)
    result = oracle.evaluate_place_recognition(pred, gt)
    assert result["accuracy"] == 1.0
    assert result["correct"] == len(gt)
    assert result["total"] == len(gt)
    assert result["mean_topological_error"] == 0.0


def test_eval_place_recognition_half_correct(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    calyces = kidney.calyx_ids()
    if len(calyces) < 2:
        pytest.skip("need at least 2 calyces to construct half-correct input")
    gt = _gt_entries(kidney)
    pred: list[tuple[np.ndarray, str]] = []
    for i, (pos, true_id) in enumerate(gt):
        if i % 2 == 0:
            pred.append((pos, true_id))  # correct
        else:
            # Swap in a different calyx to force an error.
            other = calyces[(i + 1) % len(calyces)]
            assert other != true_id
            pred.append((pos, other))
    result = oracle.evaluate_place_recognition(pred, gt)
    expected_correct = math.ceil(len(calyces) / 2)
    expected_accuracy = expected_correct / len(calyces)
    assert result["correct"] == expected_correct
    assert result["total"] == len(calyces)
    assert result["accuracy"] == pytest.approx(expected_accuracy)
    # Target is 50% correct; allow the off-by-one for odd counts.
    assert 0.4 <= result["accuracy"] <= 0.6


def test_eval_place_recognition_length_mismatch_raises(
    oracle: NavigationOracle, kidney: KidneyModel
) -> None:
    gt = _gt_entries(kidney)
    with pytest.raises(ValueError):
        oracle.evaluate_place_recognition(gt[:-1], gt)


def test_eval_place_recognition_empty_input(oracle: NavigationOracle) -> None:
    result = oracle.evaluate_place_recognition([], [])
    assert result == {
        "accuracy": 0.0,
        "correct": 0,
        "total": 0,
        "topological_errors": [],
        "mean_topological_error": 0.0,
    }
