"""Marginal gradient utility for a group of rollouts (controller v3).

For binary rewards, a GRPO/LOO group of size ``n`` with ``k`` successes has
total advantage magnitude

    sum_i |r_i - mean(r)| = 2*k*(n-k)/n

so up to the constant factor 2, the group's *gradient mass* is

    U(k, n) = k*(n-k)/n

which is exactly zero when every rollout fails (k=0) or every rollout succeeds
(k=n), and maximal at k=n/2. A rollout is therefore worth continuing in
proportion to how much it raises its group's EXPECTED U — not in proportion to
how likely it is to succeed.

That distinction is the whole point of v3. The v1/v2 rule
``V_c = b + p*(1-2b)`` depends only on the rollout's own success probability
and is wrong in both directions: it funds groups where every rollout will
succeed (zero gradient), and it kills the likely-failure sitting among
likely-successes (which is precisely what creates the contrast).

Everything here is pure and side-effect free, so the operating point can be
calibrated offline against stored episodes instead of by burning pilot runs.
"""

from __future__ import annotations

import random
from itertools import product
from typing import Sequence

# Exact enumeration is 2^n terms; 8 -> 256, still sub-microsecond. Above that
# we sample, because 2^n stops being free and groups that large are unusual.
EXACT_MAX_N = 8
_MC_SAMPLES = 512


def U(k: int, n: int) -> float:
    """Gradient mass of a group of ``n`` rollouts of which ``k`` succeeded."""
    if n <= 1:
        return 0.0
    return k * (n - k) / n


def expected_U(ps: Sequence[float], *, rng: random.Random | None = None) -> float:
    """E[U] for a group whose members succeed independently w.p. ``ps[j]``.

    Settled members are passed as exactly 0.0 or 1.0; live ones as the head's
    current estimate.
    """
    n = len(ps)
    if n <= 1:
        return 0.0
    if n <= EXACT_MAX_N:
        total = 0.0
        for outcome in product((0, 1), repeat=n):
            prob = 1.0
            for bit, p in zip(outcome, ps):
                prob *= p if bit else (1.0 - p)
                if prob == 0.0:
                    break
            if prob:
                total += prob * U(sum(outcome), n)
        return total
    r = rng or random.Random(0)
    acc = 0.0
    for _ in range(_MC_SAMPLES):
        k = sum(1 for p in ps if r.random() < p)
        acc += U(k, n)
    return acc / _MC_SAMPLES


def marginal_dU(i: int, ps: Sequence[float], *, rng: random.Random | None = None) -> float:
    """Value of CONTINUING member ``i``: E[U | keep i] - E[U | drop i].

    Dropping a member shrinks the group, so this is the rollout's marginal
    contribution to the gradient its group will produce.
    """
    if not 0 <= i < len(ps):
        raise IndexError(f"member {i} out of range for group of {len(ps)}")
    if len(ps) <= 1:
        return 0.0
    without = [p for j, p in enumerate(ps) if j != i]
    return expected_U(ps, rng=rng) - expected_U(without, rng=rng)
