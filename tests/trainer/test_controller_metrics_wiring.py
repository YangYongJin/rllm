"""The controller filter's metrics must actually reach the trainer.

`controller/episodes_dropped_stopped` is the single most important number
about the method -- how many rollouts the controller cut from the update --
and it was computed and thrown away for the entire v1/v2 run history, because
the filter ran inside `_build_trajectory_groups`, whose hook signature returns
only groups.
"""

import os

import pytest

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.trainer.algorithms.config import CompactFilteringConfig, TransformConfig
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.workflows.workflow import TerminationReason


def _Ep(task_id, rollout_idx, correct=False):
    """Minimal real Episode. task_id/rollout_idx are derived from `id`."""
    step = Step(prompt_ids=[1, 2], response_ids=[3, 4], reward=1.0)
    traj = Trajectory(steps=[step], reward=1.0)
    return Episode(
        id=f"{task_id}:{rollout_idx}",
        trajectories=[traj],
        is_correct=correct,
        termination_reason=TerminationReason.ENV_DONE,
    )


@pytest.fixture
def decision_log(tmp_path, monkeypatch):
    d = tmp_path / "decisions"
    d.mkdir()
    monkeypatch.setenv("RLLM_CONTROLLER_ENABLE", "1")
    monkeypatch.setenv("RLLM_CONTROLLER_DECISION_LOG", str(d))
    # Invalidate the module-level mtime cache between tests.
    from rllm.trainer.algorithms import controller_filter
    controller_filter._cache["sig"] = None
    return d


def _write(d, rows):
    import json
    with open(os.path.join(str(d), "controller_decisions_1.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _run(episodes):
    return transform_episodes_to_trajectory_groups(
        episodes, TransformConfig(), CompactFilteringConfig())


def test_dropped_count_reaches_the_metrics_dict(decision_log):
    _write(decision_log, [
        {"session_id": "t1:0", "action": "stop", "eligible": True, "s_t": 0.2, "audit": False},
        {"session_id": "t1:1", "action": "continue", "eligible": True, "s_t": 0.9, "audit": False},
    ])
    eps = [_Ep("t1", 0), _Ep("t1", 1)]
    groups, metrics = _run(eps)
    assert metrics["controller/episodes_dropped_stopped"] == 1.0
    assert metrics["controller/episodes_in"] == 2.0


def test_stopped_episode_is_actually_excluded_from_groups(decision_log):
    """The metric must describe a real drop, not just be a counter."""
    _write(decision_log, [
        {"session_id": "t1:0", "action": "stop", "eligible": True, "s_t": 0.2, "audit": False},
        {"session_id": "t1:1", "action": "continue", "eligible": True, "s_t": 0.9, "audit": False},
    ])
    groups, _ = _run([_Ep("t1", 0), _Ep("t1", 1)])
    kept = sum(len(g.trajectories) for g in groups)
    assert kept == 1, "the stopped rollout still entered a trajectory group"


def test_audits_are_never_dropped_even_when_stopped(decision_log):
    """Audits ran to completion; they are the unbiased anchor and must stay."""
    _write(decision_log, [
        {"session_id": "t1:0", "action": "stop", "eligible": True, "s_t": 0.2, "audit": True},
    ])
    groups, metrics = _run([_Ep("t1", 0)])
    assert metrics["controller/episodes_dropped_stopped"] == 0.0
    assert metrics["controller/episodes_audit"] == 1.0


def test_validation_sessions_are_untouched(decision_log):
    """Val uids carry a ':val' suffix, so they never match a decision record.

    Stopping validation rollouts would corrupt the eval; this is the guard.
    """
    _write(decision_log, [
        {"session_id": "t1:0:val", "action": "stop", "eligible": True, "s_t": 0.1, "audit": False},
    ])
    groups, metrics = _run([_Ep("t1", 0)])
    assert metrics["controller/episodes_dropped_stopped"] == 0.0
    assert sum(len(g.trajectories) for g in groups) == 1


def test_metrics_absent_when_controller_disabled(monkeypatch):
    monkeypatch.delenv("RLLM_CONTROLLER_ENABLE", raising=False)
    monkeypatch.delenv("RLLM_CONTROLLER_DECISION_LOG", raising=False)
    from rllm.trainer.algorithms import controller_filter
    controller_filter._cache["sig"] = None
    groups, metrics = _run([_Ep("t1", 0), _Ep("t1", 1)])
    assert not any(k.startswith("controller/") for k in metrics)
    assert sum(len(g.trajectories) for g in groups) == 2


# --- session-id reuse across training steps --------------------------------
# `task_id:rollout_idx` is not unique per step. Aggregating a session's whole
# history marks it stopped forever, so once the epoch wraps (513 tasks at batch
# 8 -> step ~64) every episode whose uid was ever stopped gets dropped
# regardless of what happened to it. Observed: 29 uids with 10.9h spans, and a
# step reporting 12 drops against only 6 real stops.

def test_stale_stop_from_an_earlier_step_does_not_drop_a_fresh_episode(decision_log):
    """The regression guard for the epoch-wrap corruption."""
    import time
    now = time.time()
    _write(decision_log, [
        # step ~10: this rollout was stopped, hours ago
        {"session_id": "t1:0", "action": "stop", "eligible": True, "s_t": 0.2,
         "audit": False, "ts": now - 40000},
        # same uid, same task recurring in the current step: ran to completion
        {"session_id": "t1:0", "action": "continue", "eligible": True, "s_t": 0.9,
         "audit": False, "ts": now - 60},
        {"session_id": "t1:0", "action": "continue", "eligible": True, "s_t": 0.9,
         "audit": False, "ts": now - 30},
    ])
    groups, metrics = _run([_Ep("t1", 0)])
    assert metrics["controller/episodes_dropped_stopped"] == 0.0, (
        "an episode was dropped because of a decision from a previous step")
    assert sum(len(g.trajectories) for g in groups) == 1


def test_a_stop_in_the_current_episode_still_drops_it(decision_log):
    """The fix must not stop the filter from doing its actual job."""
    import time
    now = time.time()
    _write(decision_log, [
        {"session_id": "t1:0", "action": "continue", "eligible": True, "s_t": 0.9,
         "audit": False, "ts": now - 40000},
        {"session_id": "t1:0", "action": "continue", "eligible": True, "s_t": 0.5,
         "audit": False, "ts": now - 60},
        {"session_id": "t1:0", "action": "stop", "eligible": True, "s_t": 0.2,
         "audit": False, "ts": now - 30},
    ])
    _, metrics = _run([_Ep("t1", 0)])
    assert metrics["controller/episodes_dropped_stopped"] == 1.0


def test_propensity_uses_only_the_current_episode(decision_log):
    """prod_s must not multiply in s_t values from a previous step's rollout."""
    import math, time
    now = time.time()
    _write(decision_log, [
        {"session_id": "t1:0", "action": "continue", "eligible": True, "s_t": 0.1,
         "audit": True, "ts": now - 40000},          # old episode, tiny s_t
        {"session_id": "t1:0", "action": "continue", "eligible": True, "s_t": 0.5,
         "audit": True, "ts": now - 30},
    ])
    eps = [_Ep("t1", 0)]
    _run(eps)
    prod = eps[0].metadata["controller"]["prod_s"]
    assert prod == pytest.approx(0.5, abs=1e-9), (
        f"prod_s={prod} folded in the previous episode's 0.1")
