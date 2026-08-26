"""Tests for the controller-only retrieval memory (M8 / brief §4.5)."""

import os
import tempfile

import pytest

from rllm_model_gateway.retrieval import RetrievalMemory, hash_tfidf


def test_hash_tfidf_is_stable_and_normalized():
    a = hash_tfidf("cannot register route with dots", extra_tokens=["aiohttp/web.py"])
    b = hash_tfidf("cannot register route with dots", extra_tokens=["aiohttp/web.py"])
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0, abs=1e-9)
    assert hash_tfidf("") == [0.0] * 256


def test_similar_tasks_score_higher_than_unrelated():
    z1 = hash_tfidf("route registration fails when name contains a dot", extra_tokens=["aiohttp/web_urldispatcher.py"])
    z2 = hash_tfidf("registering a route with a dotted name raises", extra_tokens=["aiohttp/web_urldispatcher.py"])
    z3 = hash_tfidf("dataframe groupby aggregation returns wrong dtype", extra_tokens=["pandas/core/groupby.py"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(z1, z2) > dot(z1, z3)


def test_empty_memory_yields_no_influence():
    m = RetrievalMemory()
    v, rho = m.query([0.1] * 256, [0.2] * 8)
    assert v == 0.0 and rho == 0.0


def test_rho_grows_with_evidence_and_saturates():
    m = RetrievalMemory(max_rho=0.5, k_sim=4.0)
    z = hash_tfidf("null pointer when parsing config", extra_tokens=["pkg/conf.py"])
    e = [1.0, 0.0, 0.0, 0.0]
    _, rho0 = m.query(z, e)
    for _ in range(3):
        m.add(z, e, u=0.8)
    _, rho_few = m.query(z, e)
    for _ in range(60):
        m.add(z, e, u=0.8)
    _, rho_many = m.query(z, e)
    assert rho0 == 0.0
    assert 0.0 < rho_few < rho_many <= 0.5


def test_retrieved_value_tracks_stored_utility():
    m = RetrievalMemory()
    z = hash_tfidf("timeout in async handler", extra_tokens=["aiohttp/client.py"])
    e = [0.5, 0.5, 0.0, 0.0]
    for _ in range(10):
        m.add(z, e, u=0.9)
    v, rho = m.query(z, e)
    assert v == pytest.approx(0.9, abs=1e-6)
    assert rho > 0


def test_dissimilar_query_is_not_influenced():
    """The transfer guard: unrelated experience must not steer a new task."""
    m = RetrievalMemory()
    z_seen = hash_tfidf("http route registration bug", extra_tokens=["aiohttp/web.py"])
    for _ in range(50):
        m.add(z_seen, [1.0, 0.0], u=1.0)
    z_new = hash_tfidf("matrix multiplication overflow in linalg", extra_tokens=["numpy/linalg/core.py"])
    v, rho = m.query(z_new, [0.0, 1.0])
    assert rho < 0.1, "unrelated memory must carry almost no confidence"


def test_policy_version_decay_downweights_stale_experience():
    m = RetrievalMemory(version_decay=0.5)
    z = hash_tfidf("flaky retry logic", extra_tokens=["pkg/retry.py"])
    e = [1.0, 0.0]
    for _ in range(10):
        m.add(z, e, u=1.0, policy_version=0)
    _, rho_same = m.query(z, e, policy_version=0)
    _, rho_stale = m.query(z, e, policy_version=10)
    assert rho_stale < rho_same


def test_time_decay_downweights_old_experience():
    import time as _t
    m = RetrievalMemory(half_life_s=60.0)
    z = hash_tfidf("parser crash on empty input", extra_tokens=["pkg/parse.py"])
    e = [1.0, 0.0]
    now = _t.time()
    for _ in range(10):
        m.add(z, e, u=1.0, ts=now - 600)      # 10 half-lives old
    _, rho_old = m.query(z, e, now=now)
    m2 = RetrievalMemory(half_life_s=60.0)
    for _ in range(10):
        m2.add(z, e, u=1.0, ts=now)
    _, rho_fresh = m2.query(z, e, now=now)
    assert rho_old < rho_fresh


def test_roundtrip_persistence_enables_cross_run_transfer():
    m = RetrievalMemory()
    z = hash_tfidf("segfault on large input", extra_tokens=["pkg/io.py"])
    e = [0.3, 0.7]
    for _ in range(5):
        m.add(z, e, u=0.6, stratum="hard")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mem.jsonl")
        m.save(p)
        m2 = RetrievalMemory.load(p)
        assert len(m2) == len(m)
        v1, r1 = m.query(z, e)
        v2, r2 = m2.query(z, e)
        assert v1 == pytest.approx(v2)
        assert r1 == pytest.approx(r2)


def test_capacity_is_bounded():
    m = RetrievalMemory(capacity=50)
    z = hash_tfidf("x y z", extra_tokens=["a.py"])
    for i in range(200):
        m.add(z, [1.0], u=0.5)
    assert len(m) == 50
