"""Selection-aware episode filter for the continuation controller (Phase 2 P2.3).

Joins training episodes against the gateway controller's decision log
(``controller_decisions_*.jsonl``, keyed by session id == ``task_id:rollout_idx``)
and DROPS episodes the controller stopped, unless they are audits.

Semantics (PROJECT_BRIEF §4.1/§4.6): a stopped rollout has an UNKNOWN terminal
reward. It must never enter the actor update as a zero-reward sample — it is
excluded entirely. Audit rollouts always ran to completion and stay. Kept
episodes are annotated with ``metadata['controller']`` carrying the audit flag
and the running continue-propensity ``prod_s`` (Π s_t over reached decision
points), from which the completion propensity q(τ) = a + (1-a)·Π s_t is
computed downstream for optional weighting.

Activation mirrors the controller itself: env ``RLLM_CONTROLLER_ENABLE=1`` and
``RLLM_CONTROLLER_DECISION_LOG=<dir>``. Inert otherwise.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {"sig": None, "state": {}}

# Gap that separates two episodes sharing a session id. A single rollout's
# decisions are seconds to minutes apart (one per agent turn, and a rollout runs
# ~10-13 min end to end); the same uid recurring in a later training step is
# hours away. 30 min sits well clear of both.
EPISODE_GAP_S = float(os.environ.get("RLLM_CONTROLLER_EPISODE_GAP_S", "1800"))


def _enabled() -> bool:
    return (
        os.environ.get("RLLM_CONTROLLER_ENABLE", "0") in ("1", "true", "True")
        and bool(os.environ.get("RLLM_CONTROLLER_DECISION_LOG"))
    )


def _load_state() -> dict[str, dict[str, Any]]:
    """session_id -> {stopped, audit, prod_s, decisions}. mtime-cached."""
    log_dir = os.environ.get("RLLM_CONTROLLER_DECISION_LOG", "")
    paths = sorted(glob.glob(os.path.join(log_dir, "controller_decisions_*.jsonl")))
    sig = tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths)
    if sig == _cache["sig"]:
        return _cache["state"]

    # A session id is `task_id:rollout_idx`, which is NOT unique across training
    # steps: the same task reappears in a later step (and every task reappears
    # once the epoch wraps -- 513 tasks at batch 8 means step ~64). Aggregating a
    # session's whole history would mark it stopped forever, so from epoch 2
    # onward every episode whose uid was EVER stopped would be dropped no matter
    # what actually happened to it. With a ~22% stop rate in epoch 1 that would
    # have roughly doubled the drop rate over steps 64-80 -- precisely the range
    # the headline v3-core@80 result comes from.
    #
    # Decisions for one rollout are seconds to minutes apart, while the same uid
    # recurs hours later (observed spans of 10.9h). So split each session's
    # decisions on a large time gap and keep only the most recent run.
    raw: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw.setdefault(r["session_id"], []).append(r)
        except OSError:
            continue

    state: dict[str, dict[str, Any]] = {}
    for sid, recs in raw.items():
        recs.sort(key=lambda r: float(r.get("ts") or 0.0))
        # Walk backwards to the start of the most recent contiguous run.
        start = len(recs) - 1
        while start > 0:
            gap = float(recs[start].get("ts") or 0.0) - float(recs[start - 1].get("ts") or 0.0)
            if gap > EPISODE_GAP_S:
                break
            start -= 1
        st = {"stopped": False, "audit": False, "log_prod_s": 0.0, "decisions": 0}
        for r in recs[start:]:
            st["decisions"] += 1
            st["audit"] = st["audit"] or bool(r.get("audit"))
            s_t = float(r.get("s_t", 1.0))
            # floor_hold turns were overridden by the group floor / contrast
            # guard: survival probability there was 1, not s_t. Folding their
            # s_t into the product understates q and inflates 1/q toward
            # w_max on exactly the floor-protected (rare, high-leverage)
            # trajectories.
            if r.get("eligible") and s_t > 0 and r.get("action") != "floor_hold":
                st["log_prod_s"] += math.log(s_t)
            if r.get("action") == "stop":
                st["stopped"] = True
        state[sid] = st
    _cache["sig"] = sig
    _cache["state"] = state
    return state


def _propensity_enabled() -> bool:
    return os.environ.get("RLLM_CONTROLLER_PROPENSITY", "0") in ("1", "true", "True")


def controller_episode_filter(episodes: list) -> tuple[list, dict[str, float]]:
    """Drop controller-stopped non-audit episodes; annotate the rest.

    With ``RLLM_CONTROLLER_PROPENSITY=1`` (M7), surviving non-audit episodes
    additionally get ``propensity_weight = min(1/Π s_t, w_max)`` stamped on
    each trajectory. Audit membership is DETERMINISTIC per session (blake2b
    hash in the gateway), so a non-audit's completion propensity is exactly
    Π s_t — mixing in the audit fraction (q = a + (1−a)Πs) would bias the
    surviving-sample gradient down precisely where selection pressure is
    strongest. Audits are the always-complete anchor stream at weight 1; the
    pooled estimator is unbiased. The advantage collector multiplies
    advantages by this weight.

    Returns (kept_episodes, metrics). No-op passthrough when disabled.
    """
    if not _enabled():
        return episodes, {}

    propensity_on = _propensity_enabled()
    w_max = float(os.environ.get("RLLM_CONTROLLER_PROPENSITY_WMAX", "10.0"))

    state = _load_state()
    kept, dropped, audits = [], 0, 0
    weights: list[float] = []
    for ep in episodes:
        uid = f"{ep.task_id}:{ep.rollout_idx}"
        st = state.get(uid)
        if st is None:
            kept.append(ep)
            continue
        if st["stopped"] and not st["audit"]:
            dropped += 1
            continue
        if st["audit"]:
            audits += 1
        meta = ep.metadata if isinstance(ep.metadata, dict) else {}
        meta["controller"] = {
            "audit": st["audit"],
            "stopped": st["stopped"],
            "prod_s": math.exp(st["log_prod_s"]),
            "decisions": st["decisions"],
        }
        ep.metadata = meta
        if propensity_on and not st["audit"] and st["decisions"] > 0:
            q = math.exp(st["log_prod_s"])
            w = min(1.0 / max(q, 1e-6), w_max)
            weights.append(w)
            for traj in ep.trajectories:
                tmeta = traj.metadata if isinstance(traj.metadata, dict) else {}
                tmeta["propensity_weight"] = w
                traj.metadata = tmeta
        kept.append(ep)

    metrics = {
        "controller/episodes_in": float(len(episodes)),
        "controller/episodes_dropped_stopped": float(dropped),
        "controller/episodes_audit": float(audits),
    }
    if propensity_on:
        metrics["controller/propensity_weight_mean"] = float(sum(weights) / len(weights)) if weights else 1.0
        metrics["controller/propensity_weight_max"] = float(max(weights)) if weights else 1.0
    if dropped:
        logger.info("[controller-filter] dropped %d stopped non-audit episode(s) of %d", dropped, len(episodes))
    return kept, metrics
