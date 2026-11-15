"""Statistics, cross-checked against scipy and against textbook values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from agent_reliability_lab.stats import (
    Interval,
    bootstrap_mean_difference,
    cohens_kappa,
    mcnemar_exact,
    percentile,
    wilson_interval,
)


@pytest.mark.parametrize("n", [1, 2, 5, 17, 60, 250, 1000])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_wilson_matches_scipy_for_every_count(n: int, confidence: float) -> None:
    for successes in sorted({0, 1, n // 3, n // 2, n - 1, n}):
        if not 0 <= successes <= n:
            continue
        ours = wilson_interval(successes, n, confidence)
        reference = scipy_stats.binomtest(successes, n).proportion_ci(confidence, method="wilson")
        assert ours.low == pytest.approx(reference.low, abs=1e-12)
        assert ours.high == pytest.approx(reference.high, abs=1e-12)
        assert 0.0 <= ours.low <= successes / n <= ours.high <= 1.0


def test_wilson_edge_cases() -> None:
    assert wilson_interval(0, 0) == Interval(0.0, 1.0, 0.95)
    assert wilson_interval(0, 10).low == 0.0 and wilson_interval(10, 10).high == 1.0
    assert wilson_interval(60, 60).low == pytest.approx(0.9398, abs=1e-4)
    for bad in [(-1, 5), (6, 5), (1, -1)]:
        with pytest.raises(ValueError):
            wilson_interval(*bad)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence=1.0)


def test_interval_helpers() -> None:
    assert Interval(0.1, 0.4).contains(0.4) and not Interval(0.1, 0.4).contains(0.41)
    assert Interval(0.1, 0.4).excludes_zero() and Interval(-0.4, -0.1).excludes_zero()
    assert not Interval(-0.1, 0.4).excludes_zero() and not Interval(0.0, 0.0).excludes_zero()


@pytest.mark.parametrize(
    "b, c",
    [
        (0, 0),
        (1, 0),
        (5, 0),
        (6, 0),
        (0, 6),
        (3, 1),
        (10, 2),
        (1, 9),
        (7, 7),
        (25, 10),
        (700, 800),
        (3000, 3100),
    ],
)
def test_mcnemar_matches_the_exact_binomial_test(b: int, c: int) -> None:
    ours = mcnemar_exact(b, c)
    reference = scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    assert ours == pytest.approx(reference, rel=1e-9)
    assert ours == mcnemar_exact(c, b), "the test is symmetric"
    assert 0.0 < ours <= 1.0


def test_mcnemar_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert mcnemar_exact(4, 4) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_bootstrap_is_reproducible_and_seed_dependent() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.normal(100, 10, 40).tolist()
    candidate = (np.asarray(baseline) + rng.normal(4, 5, 40)).tolist()
    first = bootstrap_mean_difference(baseline, candidate, seed=11)
    again = bootstrap_mean_difference(baseline, candidate, seed=11)
    other = bootstrap_mean_difference(baseline, candidate, seed=12)
    assert first == again and first.interval != other.interval
    assert first.estimate == pytest.approx(
        float(np.mean(np.asarray(candidate) - np.asarray(baseline)))
    )
    assert first.interval.low < first.estimate < first.interval.high
    assert first.n == 40 and first.resamples == 2000


def test_bootstrap_interval_is_close_to_the_t_interval_for_normal_data() -> None:
    rng = np.random.default_rng(5)
    differences = rng.normal(2.0, 3.0, 200)
    result = bootstrap_mean_difference(
        np.zeros(200).tolist(), differences.tolist(), resamples=5000, seed=1
    )
    standard_error = differences.std(ddof=1) / math.sqrt(200)
    low, high = scipy_stats.t.interval(0.95, 199, loc=differences.mean(), scale=standard_error)
    assert result.interval.low == pytest.approx(low, abs=0.08)
    assert result.interval.high == pytest.approx(high, abs=0.08)


def test_bootstrap_degenerate_and_invalid_input() -> None:
    same = bootstrap_mean_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same.estimate == 0.0 and same.interval == Interval(0.0, 0.0, 0.95)
    assert not same.interval.excludes_zero()
    shifted = bootstrap_mean_difference([1.0, 2.0], [3.0, 4.0])
    assert shifted.interval == Interval(2.0, 2.0, 0.95)
    single = bootstrap_mean_difference([1.0], [4.0])
    assert single.estimate == 3.0 and single.interval.low == single.interval.high == 3.0
    with pytest.raises(ValueError, match="same length"):
        bootstrap_mean_difference([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one pair"):
        bootstrap_mean_difference([], [])
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_mean_difference([1.0], [2.0], resamples=0)


def test_bootstrap_handles_large_samples_in_batches() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 20_000)
    result = bootstrap_mean_difference(baseline.tolist(), (baseline + 0.1).tolist(), resamples=200)
    assert result.estimate == pytest.approx(0.1, abs=1e-9)


def test_percentile_matches_numpy_linear_interpolation() -> None:
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for q in (0, 50, 90, 95, 99, 100):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))
    assert percentile([], 50) is None and percentile([7.0], 99) == 7.0


def test_cohens_kappa_textbook_example() -> None:
    # 2x2 table: both yes 20, A yes / B no 5, A no / B yes 10, both no 15 (n = 50).
    first = ["yes"] * 25 + ["no"] * 25
    second = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    observed = 35 / 50
    expected = (25 / 50) * (30 / 50) + (25 / 50) * (20 / 50)
    assert cohens_kappa(first, second) == pytest.approx((observed - expected) / (1 - expected))
    assert cohens_kappa(first, second) == pytest.approx(0.4)


def test_cohens_kappa_edge_cases() -> None:
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) == pytest.approx(-1.0)
    assert cohens_kappa(["a", "a"], ["a", "a"]) is None, "undefined when chance agreement is 1"
    assert cohens_kappa([], []) is None
    assert cohens_kappa(["a", "a", "a", "a"], ["a", "a", "b", "b"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])
