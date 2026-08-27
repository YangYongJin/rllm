"""Retry-classification tests for the EAI sandbox backend.

These exist because misclassification here is catastrophic rather than
cosmetic: on 2026-08-26 a control-plane outage produced `connection refused`
and DNS `server misbehaving`, neither of which was on the transient list, so
sandbox submission raised on the first attempt and killed four concurrent
multi-hour training runs.

The two directions both matter:
  - too narrow  -> an outage kills the run (the 2026-08-26 failure)
  - too wide    -> a remote command that merely PRINTS something like "502"
                   or "connection refused" gets re-executed, corrupting the
                   episode with duplicated side effects.
"""

import pytest

from rllm.sandbox.backends.eai import (
    _has_cli_marker,
    _is_network_error,
    _is_retryable_control_plane,
    _is_transient,
)

# Verbatim stderr from the 2026-08-26 outage that killed all four arms.
DNS_OUTAGE = (
    'Error: Post "https://toolkit-sp.yul201.service-now.com/v1/job?human=1": '
    'Get "https://toolkit-sp.yul201.service-now.com/.well-known/openid-configuration": '
    "dial tcp: lookup toolkit-sp.yul201.service-now.com on 10.150.0.10:53: server misbehaving"
)
REFUSED_OUTAGE = (
    'Error: Post "https://toolkit-sp.yul201.service-now.com/v1/job?human=1": '
    'Get "https://toolkit-sp.yul201.service-now.com/.well-known/openid-configuration": '
    "dial tcp 10.129.136.111:443: connect: connection refused"
)


@pytest.mark.parametrize("err", [DNS_OUTAGE, REFUSED_OUTAGE])
def test_the_outage_that_killed_four_runs_is_now_retried(err):
    """Regression guard for the 2026-08-26 incident."""
    assert _is_network_error(err)
    assert _is_retryable_control_plane(err)


@pytest.mark.parametrize(
    "err",
    [
        'Error: Get "https://host/v1/job": dial tcp 1.2.3.4:443: i/o timeout',
        'Error: Post "https://host/v1/job": net/http: TLS handshake timeout',
        "Error: dial tcp: lookup host on 10.0.0.1:53: no such host",
        'Error: Get "https://host/v1/job": EOF',
        "Error: dial tcp 1.2.3.4:443: connect: network is unreachable",
    ],
)
def test_network_layer_failures_are_retryable(err):
    assert _is_network_error(err)
    assert _is_retryable_control_plane(err)


@pytest.mark.parametrize(
    "err",
    [
        "Error: http: 500 internal error",
        "Error: server-side error (503)",
        "Error: token is invalid",
        "Error: context deadline exceeded",
    ],
)
def test_control_plane_http_failures_stay_retryable(err):
    """The pre-existing transient set must not regress."""
    assert _is_transient(err)
    assert _is_retryable_control_plane(err)


def test_remote_command_output_is_not_treated_as_cli_error():
    """No CLI marker => never retried, however suggestive the text.

    Re-running a verifier or agent command that actually executed would
    duplicate its side effects.
    """
    pytest_output = "FAILED tests/test_net.py::test_refused - connection refused\n1 failed"
    assert not _has_cli_marker(pytest_output)
    assert not _is_network_error(pytest_output)
    assert not _is_transient(pytest_output)
    assert not _is_retryable_control_plane(pytest_output)


def test_bare_status_digits_in_command_output_are_not_retried():
    assert not _is_transient("assert response.status == 502\n  502 != 200")


def test_eof_requires_punctuation_so_it_cannot_hit_unrelated_words():
    """Bare 'eof' as a substring would match ordinary words; it must not."""
    assert not _is_network_error("Error: symbol 'eofill' is undefined")


def test_genuinely_permanent_errors_are_not_retried():
    """A real user error must fail fast rather than burn a 25-minute ladder."""
    for err in [
        "Error: image not found in registry",
        "Error: account quota exceeded",
        "Error: invalid value for --cpu",
    ]:
        assert not _is_retryable_control_plane(err), err


def test_empty_and_none_are_safe():
    for err in ["", None]:
        assert not _is_transient(err)
        assert not _is_network_error(err)
        assert not _is_retryable_control_plane(err)


def test_classification_is_case_insensitive():
    assert _is_network_error('ERROR: Post "https://h/v1/job": DIAL TCP: CONNECTION REFUSED')


# --- _wait_running blind-clock behaviour -----------------------------------
# Third instance of "treat a control-plane failure as a fact": a blind poll
# (eai job get failing -> state "UNKNOWN") used to consume the startup
# deadline, turning an outage into a rollout failure and, with
# raise_on_error=true, into a dead training run.

class _FakeSandbox:
    """Minimal stand-in exercising the real _wait_running against a state script."""

    def __init__(self, states):
        self._states = list(states)
        self.name = "fake"
        self.job_id = "job-1"
        self.polls = 0

    def _state(self):
        self.polls += 1
        return self._states.pop(0) if self._states else "UNKNOWN"


def _run_wait(states, timeout, blind_ext, monkeypatch):
    import rllm.sandbox.backends.eai as eai

    clock = {"t": 1000.0}
    monkeypatch.setattr(eai.time, "time", lambda: clock["t"])
    monkeypatch.setattr(eai.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(eai, "WAIT_BLIND_EXTENSION", blind_ext)
    sb = _FakeSandbox(states)
    eai.EAISandbox._wait_running(sb, timeout)
    return sb


def test_blind_polls_do_not_consume_the_startup_deadline(monkeypatch):
    """20 blind polls (100s) then RUNNING must succeed under a 60s timeout."""
    sb = _run_wait(["UNKNOWN"] * 20 + ["RUNNING"], timeout=60.0,
                   blind_ext=1800.0, monkeypatch=monkeypatch)
    assert sb.polls == 21


def test_blind_extension_is_bounded(monkeypatch):
    """A permanently unreachable API must still fail, not hang forever."""
    import pytest as _p
    with _p.raises(RuntimeError, match="not RUNNING"):
        _run_wait(["UNKNOWN"] * 10000, timeout=60.0, blind_ext=120.0,
                  monkeypatch=monkeypatch)


def test_observable_queuing_still_consumes_the_deadline(monkeypatch):
    """A merely-busy cluster is real evidence; only blindness pauses the clock."""
    import pytest as _p
    with _p.raises(RuntimeError, match="not RUNNING"):
        _run_wait(["QUEUING"] * 10000, timeout=60.0, blind_ext=1800.0,
                  monkeypatch=monkeypatch)


def test_terminal_state_fails_fast_even_when_blind_extension_is_large(monkeypatch):
    import pytest as _p
    with _p.raises(RuntimeError, match="reached"):
        _run_wait(["UNKNOWN", "UNKNOWN", "FAILED"], timeout=60.0,
                  blind_ext=1800.0, monkeypatch=monkeypatch)
