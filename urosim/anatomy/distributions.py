"""Clinical parameter sampling functions for kidney anatomy generation.

Every sampler in this module takes a ``numpy.random.Generator`` as ``rng``
and returns a sampled value. Results are deterministic for a given rng
state; no function reads or mutates numpy's global random state.

The distributions and ranges are drawn from published clinical literature
on renal pelvicalyceal anatomy (Sampaio classification, infundibular
morphometry) and urolithiasis epidemiology.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Pelvis type priors (Sampaio classification).
# ---------------------------------------------------------------------------
_PELVIS_TYPES: tuple[str, ...] = ("A1", "A2", "A3", "B1", "B2")
_PELVIS_TYPE_PROBS: tuple[float, ...] = (0.15, 0.30, 0.17, 0.15, 0.23)

# ---------------------------------------------------------------------------
# Infundibular length priors: Normal(mu, sigma) per segment, clamped [5, 60] mm.
# ---------------------------------------------------------------------------
_INFUND_LEN_PARAMS: dict[str, tuple[float, float]] = {
    "upper": (30.6, 7.9),
    "mid": (16.4, 7.7),
    "lower": (16.0, 6.0),
}
_INFUND_LEN_CLAMP: tuple[float, float] = (5.0, 60.0)

# ---------------------------------------------------------------------------
# Infundibular width: Normal(5.5, 1.2) mm, clamped [3, 12].
# ---------------------------------------------------------------------------
_INFUND_WIDTH_MEAN: float = 5.5
_INFUND_WIDTH_SD: float = 1.2
_INFUND_WIDTH_CLAMP: tuple[float, float] = (3.0, 12.0)

# ---------------------------------------------------------------------------
# Infundibulopelvic angle: Uniform[30, 110] degrees.
# ---------------------------------------------------------------------------
_IPA_LOW: float = 30.0
_IPA_HIGH: float = 110.0

# ---------------------------------------------------------------------------
# Perpendicular calyx probability.
# ---------------------------------------------------------------------------
_PERPENDICULAR_CALYX_PROB: float = 0.114

# ---------------------------------------------------------------------------
# Pelvis ellipsoid half-axis ranges (uniform sampling per axis), mm.
# Keys are Sampaio pelvis types; values are three (low, high) tuples
# for the (a, b, c) half-axes.
# ---------------------------------------------------------------------------
_PELVIS_RADIUS_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "A1": ((8.0, 10.0), (5.0, 7.0), (4.0, 5.0)),
    "A2": ((10.0, 14.0), (6.0, 9.0), (5.0, 7.0)),
    "A3": ((14.0, 20.0), (9.0, 12.0), (7.0, 9.0)),
    "B1": ((8.0, 12.0), (5.0, 8.0), (4.0, 6.0)),
    "B2": ((8.0, 12.0), (5.0, 8.0), (4.0, 6.0)),
}

# ---------------------------------------------------------------------------
# Minor calyces per major calyx, by pole — inclusive integer ranges.
# ---------------------------------------------------------------------------
_MINOR_CALYX_RANGES: dict[str, tuple[int, int]] = {
    "upper": (2, 4),
    "mid": (2, 4),
    "lower": (3, 5),
}

# ---------------------------------------------------------------------------
# Stone composition priors.
# ---------------------------------------------------------------------------
_STONE_COMPOSITIONS: tuple[str, ...] = (
    "CaOx_mono",
    "CaOx_di",
    "uric",
    "struvite",
    "cystine",
)
_STONE_COMPOSITION_PROBS: tuple[float, ...] = (0.40, 0.25, 0.15, 0.10, 0.10)

# ---------------------------------------------------------------------------
# Stone size: log-normal with mu, sigma chosen so E[X] ≈ 6 mm, clamped [2, 15].
# With mu = ln(5.5) and sigma = 0.45, E[X] = exp(mu + sigma^2/2) ≈ 6.08 mm.
# ---------------------------------------------------------------------------
_STONE_SIZE_MU: float = math.log(5.5)
_STONE_SIZE_SIGMA: float = 0.45
_STONE_SIZE_CLAMP: tuple[float, float] = (2.0, 15.0)

# ---------------------------------------------------------------------------
# Stone location priors.
# ---------------------------------------------------------------------------
_STONE_LOCATIONS: tuple[str, ...] = ("lower", "mid", "upper", "pelvis")
_STONE_LOCATION_PROBS: tuple[float, ...] = (0.44, 0.28, 0.20, 0.08)


# ---------------------------------------------------------------------------
# Sanity check: all categorical priors must sum to 1.0. Fail loudly at import.
# ---------------------------------------------------------------------------
def _assert_normalized(name: str, probs: tuple[float, ...]) -> None:
    total = sum(probs)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} probabilities must sum to 1.0, got {total}")


_assert_normalized("_PELVIS_TYPE_PROBS", _PELVIS_TYPE_PROBS)
_assert_normalized("_STONE_COMPOSITION_PROBS", _STONE_COMPOSITION_PROBS)
_assert_normalized("_STONE_LOCATION_PROBS", _STONE_LOCATION_PROBS)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the closed interval ``[low, high]``.

    Args:
        value: The value to clamp.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        ``value`` if within ``[low, high]``, otherwise the nearest bound.
    """
    if value < low:
        return low
    if value > high:
        return high
    return value


def sample_pelvis_type(rng: np.random.Generator) -> str:
    """Sample a Sampaio pelvis classification label.

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        One of ``"A1"``, ``"A2"``, ``"A3"``, ``"B1"``, ``"B2"`` drawn from
        the weighted prior ``(0.15, 0.30, 0.17, 0.15, 0.23)``.
    """
    return str(rng.choice(_PELVIS_TYPES, p=_PELVIS_TYPE_PROBS))


def sample_num_major_calyces(pelvis_type: str, rng: np.random.Generator) -> int:
    """Sample the number of major calyces for a given pelvis type.

    B-type pelvises (bifurcated) are always modeled with exactly 2 major
    calyces. A-type pelvises yield 2 with probability 0.6 and 3 otherwise.

    Args:
        pelvis_type: A Sampaio label (``"A1"``..``"B2"``).
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Number of major calyces (2 or 3).

    Raises:
        ValueError: If ``pelvis_type`` is not a known label.
    """
    if pelvis_type.startswith("B"):
        if pelvis_type not in _PELVIS_TYPES:
            raise ValueError(f"unknown pelvis_type: {pelvis_type!r}")
        return 2
    if pelvis_type.startswith("A"):
        if pelvis_type not in _PELVIS_TYPES:
            raise ValueError(f"unknown pelvis_type: {pelvis_type!r}")
        return 2 if rng.random() < 0.6 else 3
    raise ValueError(f"unknown pelvis_type: {pelvis_type!r}")


def sample_num_minor_calyces(pole: str, rng: np.random.Generator) -> int:
    """Sample the number of minor calyces draining one major calyx.

    Args:
        pole: Which pole the major calyx sits in — ``"upper"``, ``"mid"``,
            or ``"lower"``.
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Integer count in the inclusive range for that pole: upper and mid
        yield 2–4, lower yields 3–5.

    Raises:
        ValueError: If ``pole`` is not one of the three recognized values.
    """
    if pole not in _MINOR_CALYX_RANGES:
        raise ValueError(f"unknown pole: {pole!r}")
    low, high = _MINOR_CALYX_RANGES[pole]
    return int(rng.integers(low, high + 1))


def sample_infundibular_length(segment_type: str, rng: np.random.Generator) -> float:
    """Sample an infundibular length in millimeters for a pole segment.

    The underlying distribution is Normal(mu, sigma) per pole, clamped to
    ``[5, 60]`` mm to exclude non-physical tails:

    - upper: Normal(30.6, 7.9)
    - mid:   Normal(16.4, 7.7)
    - lower: Normal(16.0, 6.0)

    Args:
        segment_type: Pole label — ``"upper"``, ``"mid"``, or ``"lower"``.
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Length in mm, in ``[5.0, 60.0]``.

    Raises:
        ValueError: If ``segment_type`` is not recognized.
    """
    if segment_type not in _INFUND_LEN_PARAMS:
        raise ValueError(f"unknown segment_type: {segment_type!r}")
    mu, sigma = _INFUND_LEN_PARAMS[segment_type]
    low, high = _INFUND_LEN_CLAMP
    return _clamp(float(rng.normal(mu, sigma)), low, high)


def sample_infundibular_width(rng: np.random.Generator) -> float:
    """Sample an infundibular width in millimeters.

    Draws from Normal(5.5, 1.2), clamped to ``[3, 12]`` mm.

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Width in mm, in ``[3.0, 12.0]``.
    """
    low, high = _INFUND_WIDTH_CLAMP
    return _clamp(float(rng.normal(_INFUND_WIDTH_MEAN, _INFUND_WIDTH_SD)), low, high)


def sample_infundibulopelvic_angle(rng: np.random.Generator) -> float:
    """Sample an infundibulopelvic angle in degrees.

    Draws uniformly from ``[30, 110]`` degrees.

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Angle in degrees.
    """
    return float(rng.uniform(_IPA_LOW, _IPA_HIGH))


def sample_has_perpendicular_calyx(rng: np.random.Generator) -> bool:
    """Sample whether a perpendicular calyx is present.

    Returns ``True`` with probability 0.114 (11.4% of kidneys exhibit a
    perpendicular accessory calyx in the cited anatomy studies).

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Boolean flag.
    """
    return bool(rng.random() < _PERPENDICULAR_CALYX_PROB)


def sample_pelvis_radius(
    pelvis_type: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Sample ellipsoid half-axes ``(a, b, c)`` of the renal pelvis in mm.

    Half-axis ranges by type (uniform per axis):

    - A1 (small):      ``(8-10, 5-7, 4-5)``
    - A2 (medium):     ``(10-14, 6-9, 5-7)``
    - A3 (large):      ``(14-20, 9-12, 7-9)``
    - B1/B2 (bifurc.): ``(8-12, 5-8, 4-6)``

    Args:
        pelvis_type: A Sampaio label (``"A1"``..``"B2"``).
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Tuple ``(a, b, c)`` of half-axes in mm.

    Raises:
        ValueError: If ``pelvis_type`` is not a known label.
    """
    if pelvis_type not in _PELVIS_RADIUS_RANGES:
        raise ValueError(f"unknown pelvis_type: {pelvis_type!r}")
    (a_range, b_range, c_range) = _PELVIS_RADIUS_RANGES[pelvis_type]
    a = float(rng.uniform(a_range[0], a_range[1]))
    b = float(rng.uniform(b_range[0], b_range[1]))
    c = float(rng.uniform(c_range[0], c_range[1]))
    return (a, b, c)


def sample_stone_composition(rng: np.random.Generator) -> str:
    """Sample a kidney stone composition label.

    Weighted prior:
    ``CaOx_mono=0.40, CaOx_di=0.25, uric=0.15, struvite=0.10, cystine=0.10``.

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Composition label string.
    """
    return str(rng.choice(_STONE_COMPOSITIONS, p=_STONE_COMPOSITION_PROBS))


def sample_stone_size(rng: np.random.Generator) -> float:
    """Sample a kidney stone size in millimeters.

    Uses a log-normal distribution with ``mu = ln(5.5)`` and ``sigma = 0.45``,
    clamped to ``[2, 15]`` mm. The post-clamp mean is approximately 6 mm.

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Stone diameter in mm, in ``[2.0, 15.0]``.
    """
    raw = float(rng.lognormal(mean=_STONE_SIZE_MU, sigma=_STONE_SIZE_SIGMA))
    low, high = _STONE_SIZE_CLAMP
    return _clamp(raw, low, high)


def sample_stone_location(rng: np.random.Generator) -> str:
    """Sample a kidney stone location label.

    Weighted prior:
    ``lower=0.44, mid=0.28, upper=0.20, pelvis=0.08``.

    Args:
        rng: numpy Generator used as the sole randomness source.

    Returns:
        Location label string.
    """
    return str(rng.choice(_STONE_LOCATIONS, p=_STONE_LOCATION_PROBS))
