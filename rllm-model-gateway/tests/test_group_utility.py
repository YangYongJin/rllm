"""Tests for the v3 marginal-gradient value function.

These encode the reference numbers from V3_SPEC.md §10 and, importantly, the
REGRESSION GUARDS for the two ways the v1/v2 rule was wrong.
"""

import math

import pytest

from rllm_model_gateway.group_utility import U, expected_U, marginal_dU


def test_U_is_zero_for_degenerate_groups():
    # No contrast -> no advantage -> no gradient, at either extreme.
    assert U(0, 4) == 0.0
    assert U(4, 4) == 0.0
    assert U(0, 8) == U(8, 8) == 0.0


def test_U_peaks_at_half_and_is_symmetric():
    assert U(2, 4) == 1.0
    assert U(1, 4) == U(3, 4) == 0.75
    for n in (2, 4, 5, 8):
        for k in range(n + 1):
            assert U(k, n) == pytest.approx(U(n - k, n))
        assert U(n // 2, n) == max(U(k, n) for k in range(n + 1))


def test_expected_U_matches_certain_outcomes():
    assert expected_U([1.0, 1.0, 0.0, 0.0]) == pytest.approx(U(2, 4))
    assert expected_U([0.0] * 4) == pytest.approx(0.0)
    assert expected_U([1.0] * 4) == pytest.approx(0.0)
    assert expected_U([0.5]) == 0.0  # a group of one can never be informative


def test_marginal_dU_reference_values():
    # Reference table from V3_SPEC.md §10.
    assert marginal_dU(0, [0.35, 0.03, 0.03, 0.03]) == pytest.approx(0.2547, abs=5e-4)
    assert marginal_dU(0, [0.03, 0.40, 0.40, 0.40]) == pytest.approx(0.1845, abs=5e-4)
    assert marginal_dU(0, [0.97, 0.97, 0.97, 0.97]) == pytest.approx(0.0291, abs=5e-4)
    assert marginal_dU(0, [0.03, 0.03, 0.03, 0.03]) == pytest.approx(0.0291, abs=5e-4)


def test_regression_all_certain_is_not_valuable():
    """v1 bug #1: a rollout certain to succeed in an all-succeed group scored
    MAXIMALLY under V_c = b + p(1-2b), yet contributes zero gradient."""
    all_certain = marginal_dU(0, [0.97] * 4)
    all_doomed = marginal_dU(0, [0.03] * 4)
    assert all_certain == pytest.approx(all_doomed, abs=1e-3), (
        "degenerate groups must be valued equally low regardless of direction"
    )
    informative = marginal_dU(0, [0.35, 0.03, 0.03, 0.03])
    assert informative > 5 * all_certain


def test_regression_doomed_among_promising_is_valuable():
    """v1 bug #2: the likely-failure among likely-successes CREATES the
    contrast, but scored low because the rule only looked at its own p_hat."""
    doomed_among_promising = marginal_dU(0, [0.03, 0.40, 0.40, 0.40])
    doomed_among_doomed = marginal_dU(0, [0.03, 0.03, 0.03, 0.03])
    assert doomed_among_promising > 3 * doomed_among_doomed


def test_marginal_dU_is_nonnegative_and_bounded():
    import random

    rng = random.Random(7)
    for _ in range(300):
        n = rng.choice([2, 3, 4, 5])
        ps = [rng.random() for _ in range(n)]
        d = marginal_dU(rng.randrange(n), ps)
        assert d >= -1e-12, "adding a member can never reduce expected gradient mass"
        assert d <= n  # loose sanity bound


def test_large_group_falls_back_to_sampling():
    import random

    ps = [0.5] * 12
    got = expected_U(ps, rng=random.Random(0))
    exact = sum(
        math.comb(12, k) * 0.5 ** 12 * U(k, 12) for k in range(13)
    )
    assert got == pytest.approx(exact, abs=0.15)


def test_group_of_one_has_no_marginal_value():
    assert marginal_dU(0, [0.5]) == 0.0
