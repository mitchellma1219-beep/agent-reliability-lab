"""Statistics used by metrics, regression testing and judge calibration.

Everything here is implemented directly (standard library plus numpy for resampling) and is
cross-checked against scipy and by simulation in ``tests/statistical``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    level: float = 0.95

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it behaves sensibly for small ``n`` and
    for proportions near 0 or 1, which is the usual situation for pass rates. With no
    observations the interval is the whole range [0, 1].
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n < 0 or not 0 <= successes <= max(n, 0):
        raise ValueError("need 0 <= successes <= n")
    if n == 0:
        return Interval(0.0, 1.0, confidence)
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return Interval(low, high, confidence)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail outcomes.

    ``b`` is the number of pairs that passed before and fail now, ``c`` the number that
    failed before and pass now. Under the null hypothesis of no change each discordant pair
    is equally likely to go either way, so ``min(b, c)`` is Binomial(b + c, 1/2). Pairs that
    did not change carry no information and are not used. With no discordant pair the
    p-value is exactly 1: identical runs can never look like a regression.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must not be negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    term = 1
    tail = 1
    for i in range(1, k + 1):
        term = term * (n - i + 1) // i
        tail += term
    p_value: float = (2 * tail) / (2**n)
    return min(1.0, p_value)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    interval: Interval
    n: int
    resamples: int


def bootstrap_mean_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap interval for the mean paired difference ``candidate - baseline``.

    The pairs (one per case) are resampled with replacement. The generator is seeded, so
    the interval is reproducible. If every difference is identical (for example all zero for
    identical runs) the interval collapses to that value.
    """
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same length")
    if not baseline:
        raise ValueError("at least one pair is needed")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    n = differences.size
    estimate = float(differences.mean())
    if np.all(differences == differences[0]):
        return BootstrapResult(estimate, Interval(estimate, estimate, confidence), n, resamples)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    batch = max(1, 1_000_000 // n)
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        indexes = rng.integers(0, n, size=(size, n))
        means[start : start + size] = differences[indexes].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(estimate, Interval(float(low), float(high), confidence), n, resamples)


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q``-th percentile with linear interpolation; ``None`` for an empty sample."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def cohens_kappa(first: Sequence[Hashable], second: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two raters over the same items.

    Returns ``None`` when chance agreement is 1 (both raters used a single identical label
    for everything), where kappa is undefined.
    """
    if len(first) != len(second):
        raise ValueError("both raters must label the same items")
    n = len(first)
    if n == 0:
        return None
    observed = sum(1 for left, right in zip(first, second, strict=True) if left == right) / n
    left_counts, right_counts = Counter(first), Counter(second)
    expected = sum(
        left_counts[label] / n * right_counts[label] / n
        for label in set(left_counts) | set(right_counts)
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)
