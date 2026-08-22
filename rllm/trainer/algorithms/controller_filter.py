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

    state: dict[str, dict[str, Any]] = {}
    for p in paths:
        try:
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    st = state.setdefault(
                        r["session_id"],
                        {"stopped": False, "audit": bool(r.get("audit")), "log_prod_s": 0.0, "decisions": 0},
                    )
                    st["decisions"] += 1
                    s_t = float(r.get("s_t", 1.0))
                    if r.get("eligible") and s_t > 0:
                        st["log_prod_s"] += math.log(s_t)
                    if r.get("action") == "stop":
                        st["stopped"] = True
        except OSError:
            continue
    _cache["sig"] = sig
    _cache["state"] = state
    return state


def controller_episode_filter(episodes: list) -> tuple[list, dict[str, float]]:
    """Drop controller-stopped non-audit episodes; annotate the rest.

    Returns (kept_episodes, metrics). No-op passthrough when disabled.
    """
    if not _enabled():
        return episodes, {}

    state = _load_state()
    kept, dropped, audits = [], 0, 0
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
        kept.append(ep)

    metrics = {
        "controller/episodes_in": float(len(episodes)),
        "controller/episodes_dropped_stopped": float(dropped),
        "controller/episodes_audit": float(audits),
    }
    if dropped:
        logger.info("[controller-filter] dropped %d stopped non-audit episode(s) of %d", dropped, len(episodes))
    return kept, metrics
