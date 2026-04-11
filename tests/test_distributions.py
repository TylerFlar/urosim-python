"""Tests for ``urosim.anatomy.distributions``.

Covers four properties for every sampler:

1. Determinism — same seed, same sequence.
2. Distribution accuracy — sample moments / frequencies match the clinical
   targets within documented tolerances.
3. Bounds — clamped samplers never emit values outside their clamp.
4. Validity — categorical samplers only emit expected labels; invalid
   categorical arguments raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from urosim.anatomy.distributions import (
    sample_has_perpendicular_calyx,
    sample_infundibular_length,
    sample_infundibular_width,
    sample_infundibulopelvic_angle,
    sample_num_major_calyces,
    sample_num_minor_calyces,
    sample_pelvis_radius,
    sample_pelvis_type,
    sample_stone_composition,
    sample_stone_location,
    sample_stone_size,
)

SEED = 12345

PELVIS_TYPES = ("A1", "A2", "A3", "B1", "B2")
POLES = ("upper", "mid", "lower")
SEGMENTS = ("upper", "mid", "lower")
STONE_COMPOSITIONS = {"CaOx_mono", "CaOx_di", "uric", "struvite", "cystine"}
STONE_LOCATIONS = {"lower", "mid", "upper", "pelvis"}


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_determinism_sample_pelvis_type() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_pelvis_type(rng1) for _ in range(100)]
    s2 = [sample_pelvis_type(rng2) for _ in range(100)]
    assert s1 == s2


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_determinism_sample_num_major_calyces(pelvis_type: str) -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_num_major_calyces(pelvis_type, rng1) for _ in range(100)]
    s2 = [sample_num_major_calyces(pelvis_type, rng2) for _ in range(100)]
    assert s1 == s2


@pytest.mark.parametrize("pole", POLES)
def test_determinism_sample_num_minor_calyces(pole: str) -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_num_minor_calyces(pole, rng1) for _ in range(100)]
    s2 = [sample_num_minor_calyces(pole, rng2) for _ in range(100)]
    assert s1 == s2


@pytest.mark.parametrize("segment", SEGMENTS)
def test_determinism_sample_infundibular_length(segment: str) -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_infundibular_length(segment, rng1) for _ in range(100)]
    s2 = [sample_infundibular_length(segment, rng2) for _ in range(100)]
    assert s1 == s2


def test_determinism_sample_infundibular_width() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_infundibular_width(rng1) for _ in range(100)]
    s2 = [sample_infundibular_width(rng2) for _ in range(100)]
    assert s1 == s2


def test_determinism_sample_infundibulopelvic_angle() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_infundibulopelvic_angle(rng1) for _ in range(100)]
    s2 = [sample_infundibulopelvic_angle(rng2) for _ in range(100)]
    assert s1 == s2


def test_determinism_sample_has_perpendicular_calyx() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_has_perpendicular_calyx(rng1) for _ in range(100)]
    s2 = [sample_has_perpendicular_calyx(rng2) for _ in range(100)]
    assert s1 == s2


@pytest.mark.parametrize("pelvis_type", PELVIS_TYPES)
def test_determinism_sample_pelvis_radius(pelvis_type: str) -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_pelvis_radius(pelvis_type, rng1) for _ in range(100)]
    s2 = [sample_pelvis_radius(pelvis_type, rng2) for _ in range(100)]
    assert s1 == s2


def test_determinism_sample_stone_composition() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_stone_composition(rng1) for _ in range(100)]
    s2 = [sample_stone_composition(rng2) for _ in range(100)]
    assert s1 == s2


def test_determinism_sample_stone_size() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_stone_size(rng1) for _ in range(100)]
    s2 = [sample_stone_size(rng2) for _ in range(100)]
    assert s1 == s2


def test_determinism_sample_stone_location() -> None:
    rng1 = _rng()
    rng2 = _rng()
    s1 = [sample_stone_location(rng1) for _ in range(100)]
    s2 = [sample_stone_location(rng2) for _ in range(100)]
    assert s1 == s2


# ---------------------------------------------------------------------------
# 2. Distribution accuracy (10_000 samples)
# ---------------------------------------------------------------------------

N_ACCURACY = 10_000
FREQ_TOL = 0.015  # absolute tolerance for categorical frequencies


def test_pelvis_type_frequencies() -> None:
    rng = _rng()
    draws = [sample_pelvis_type(rng) for _ in range(N_ACCURACY)]
    expected = {"A1": 0.15, "A2": 0.30, "A3": 0.17, "B1": 0.15, "B2": 0.23}
    for label, target in expected.items():
        freq = draws.count(label) / N_ACCURACY
        assert abs(freq - target) < FREQ_TOL, f"{label}: freq={freq}, target={target}"


def test_num_major_calyces_A_split() -> None:
    rng = _rng()
    draws = [sample_num_major_calyces("A2", rng) for _ in range(N_ACCURACY)]
    frac_two = draws.count(2) / N_ACCURACY
    assert 0.57 <= frac_two <= 0.63
    # Only 2 or 3 allowed.
    assert set(draws).issubset({2, 3})


@pytest.mark.parametrize("b_type", ("B1", "B2"))
def test_num_major_calyces_B_always_two(b_type: str) -> None:
    rng = _rng()
    draws = [sample_num_major_calyces(b_type, rng) for _ in range(1_000)]
    assert all(d == 2 for d in draws)


@pytest.mark.parametrize(
    "segment,mean,std",
    [
        ("upper", 30.6, 7.9),
        ("mid", 16.4, 7.7),
        ("lower", 16.0, 6.0),
    ],
)
def test_infundibular_length_moments(segment: str, mean: float, std: float) -> None:
    rng = _rng()
    samples = np.array(
        [sample_infundibular_length(segment, rng) for _ in range(N_ACCURACY)]
    )
    # Mean tolerance: 10% of clinical target (per user spec).
    # Std tolerance is looser (15%) because the clamp at 5 mm bites the
    # mid/lower tails and biases the sample std downward.
    assert abs(samples.mean() - mean) < 0.10 * mean
    assert abs(samples.std() - std) < 0.15 * std


def test_infundibular_width_moments() -> None:
    rng = _rng()
    samples = np.array([sample_infundibular_width(rng) for _ in range(N_ACCURACY)])
    # Mean 10%, std 15% (clamp at 3 mm lightly bites the lower tail).
    assert abs(samples.mean() - 5.5) < 0.10 * 5.5
    assert abs(samples.std() - 1.2) < 0.15 * 1.2


def test_ipa_uniform_moments() -> None:
    rng = _rng()
    samples = np.array(
        [sample_infundibulopelvic_angle(rng) for _ in range(N_ACCURACY)]
    )
    # Uniform[30, 110]: mean 70, spans [30, 110].
    assert abs(samples.mean() - 70.0) < 0.10 * 70.0
    assert samples.min() >= 30.0
    assert samples.max() <= 110.0
    # Sanity: we're actually spanning the range.
    assert samples.min() < 32.0
    assert samples.max() > 108.0


def test_perpendicular_calyx_rate() -> None:
    rng = _rng()
    draws = [sample_has_perpendicular_calyx(rng) for _ in range(N_ACCURACY)]
    rate = sum(1 for d in draws if d) / N_ACCURACY
    assert abs(rate - 0.114) < FREQ_TOL


def test_stone_composition_frequencies() -> None:
    rng = _rng()
    draws = [sample_stone_composition(rng) for _ in range(N_ACCURACY)]
    expected = {
        "CaOx_mono": 0.40,
        "CaOx_di": 0.25,
        "uric": 0.15,
        "struvite": 0.10,
        "cystine": 0.10,
    }
    for label, target in expected.items():
        freq = draws.count(label) / N_ACCURACY
        assert abs(freq - target) < FREQ_TOL, f"{label}: freq={freq}, target={target}"


def test_stone_size_moments() -> None:
    rng = _rng()
    samples = np.array([sample_stone_size(rng) for _ in range(N_ACCURACY)])
    # Target mean ~6 mm per user spec; 10% tolerance.
    assert abs(samples.mean() - 6.0) < 0.10 * 6.0
    assert samples.min() >= 2.0
    assert samples.max() <= 15.0


def test_stone_location_frequencies() -> None:
    rng = _rng()
    draws = [sample_stone_location(rng) for _ in range(N_ACCURACY)]
    expected = {"lower": 0.44, "mid": 0.28, "upper": 0.20, "pelvis": 0.08}
    for label, target in expected.items():
        freq = draws.count(label) / N_ACCURACY
        assert abs(freq - target) < FREQ_TOL, f"{label}: freq={freq}, target={target}"


@pytest.mark.parametrize(
    "pelvis_type,expected_ranges",
    [
        ("A1", ((8.0, 10.0), (5.0, 7.0), (4.0, 5.0))),
        ("A2", ((10.0, 14.0), (6.0, 9.0), (5.0, 7.0))),
        ("A3", ((14.0, 20.0), (9.0, 12.0), (7.0, 9.0))),
        ("B1", ((8.0, 12.0), (5.0, 8.0), (4.0, 6.0))),
        ("B2", ((8.0, 12.0), (5.0, 8.0), (4.0, 6.0))),
    ],
)
def test_pelvis_radius_bounds_by_type(
    pelvis_type: str,
    expected_ranges: tuple[tuple[float, float], ...],
) -> None:
    rng = _rng()
    for _ in range(1_000):
        a, b, c = sample_pelvis_radius(pelvis_type, rng)
        (a_lo, a_hi), (b_lo, b_hi), (c_lo, c_hi) = expected_ranges
        assert a_lo <= a <= a_hi
        assert b_lo <= b <= b_hi
        assert c_lo <= c <= c_hi


# ---------------------------------------------------------------------------
# 3. Bounds / clamp tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segment", SEGMENTS)
def test_infundibular_length_within_clamp(segment: str) -> None:
    rng = _rng()
    samples = np.array(
        [sample_infundibular_length(segment, rng) for _ in range(N_ACCURACY)]
    )
    assert samples.min() >= 5.0
    assert samples.max() <= 60.0


def test_infundibular_width_within_clamp() -> None:
    rng = _rng()
    samples = np.array([sample_infundibular_width(rng) for _ in range(N_ACCURACY)])
    assert samples.min() >= 3.0
    assert samples.max() <= 12.0


def test_stone_size_within_clamp() -> None:
    rng = _rng()
    samples = np.array([sample_stone_size(rng) for _ in range(N_ACCURACY)])
    assert samples.min() >= 2.0
    assert samples.max() <= 15.0


def test_ipa_within_uniform_bounds() -> None:
    rng = _rng()
    samples = np.array(
        [sample_infundibulopelvic_angle(rng) for _ in range(N_ACCURACY)]
    )
    assert samples.min() >= 30.0
    assert samples.max() <= 110.0


# ---------------------------------------------------------------------------
# 4. Validity tests
# ---------------------------------------------------------------------------


def test_pelvis_type_values() -> None:
    rng = _rng()
    valid = set(PELVIS_TYPES)
    for _ in range(1_000):
        assert sample_pelvis_type(rng) in valid


def test_stone_composition_values() -> None:
    rng = _rng()
    for _ in range(1_000):
        assert sample_stone_composition(rng) in STONE_COMPOSITIONS


def test_stone_location_values() -> None:
    rng = _rng()
    for _ in range(1_000):
        assert sample_stone_location(rng) in STONE_LOCATIONS


def test_invalid_pelvis_type_raises_in_num_major_calyces() -> None:
    rng = _rng()
    with pytest.raises(ValueError):
        sample_num_major_calyces("Z9", rng)


def test_invalid_pelvis_type_raises_in_pelvis_radius() -> None:
    rng = _rng()
    with pytest.raises(ValueError):
        sample_pelvis_radius("Z9", rng)


def test_invalid_pole_raises() -> None:
    rng = _rng()
    with pytest.raises(ValueError):
        sample_num_minor_calyces("side", rng)


def test_invalid_segment_raises() -> None:
    rng = _rng()
    with pytest.raises(ValueError):
        sample_infundibular_length("bogus", rng)
