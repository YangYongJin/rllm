"""Stochastic continuation controller (rollout-control research, Phase 2).

Intercepts each session-scoped chat-completions request (= one agent turn) and
decides `continue` vs `stop`. Stopping is done by returning a synthetic
terminal response that speaks the harness's own exit protocol (mini-swe-agent:
a ``bash`` tool call echoing its submit sentinel), so no harness changes are
needed and the flow/sandbox tear down through the normal path.

Semantics per PROJECT_BRIEF §4.6:
- An audit flag is sampled once per session at admission (first turn seen);
  audit sessions always continue but counterfactual ``s_t`` is still logged.
- Every decision is appended as JSONL: session, turn, s_t, action, audit,
  controller version — the propensity/selection data for training-side
  weighting and for the savings report.
- v0 modes: ``random`` (constant continue prob; doubles as the "random
  stopping" ablation arm). The learned head slots into ``_continue_prob``.

Config via environment (v0; the launcher maps hydra keys onto these):
  RLLM_CONTROLLER_ENABLE=1        master switch (absent/0 -> inert)
  RLLM_CONTROLLER_MODE=random
  RLLM_CONTROLLER_P_STOP=0.15     per-turn stop prob in random mode
  RLLM_CONTROLLER_AUDIT_FRACTION=0.1
  RLLM_CONTROLLER_MIN_TURNS=2     never stop before this many completed turns
  RLLM_CONTROLLER_DECISION_LOG=/path/dir   (JSONL per gateway process)
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

CONTROLLER_VERSION = "v0-random-1"
STOP_SENTINEL_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def controller_from_env() -> "ContinuationController | None":
    if os.environ.get("RLLM_CONTROLLER_ENABLE", "0") not in ("1", "true", "True"):
        return None
    return ContinuationController(
        mode=os.environ.get("RLLM_CONTROLLER_MODE", "random"),
        p_stop=float(os.environ.get("RLLM_CONTROLLER_P_STOP", "0.15")),
        audit_fraction=float(os.environ.get("RLLM_CONTROLLER_AUDIT_FRACTION", "0.1")),
        min_turns=int(os.environ.get("RLLM_CONTROLLER_MIN_TURNS", "2")),
        decision_log_dir=os.environ.get("RLLM_CONTROLLER_DECISION_LOG") or None,
        seed=int(os.environ.get("RLLM_CONTROLLER_SEED", "0")) or None,
    )


class ContinuationController:
    def __init__(
        self,
        mode: str = "random",
        p_stop: float = 0.15,
        audit_fraction: float = 0.1,
        min_turns: int = 2,
        decision_log_dir: str | None = None,
        seed: int | None = None,
    ) -> None:
        assert mode in ("random",), f"unsupported controller mode: {mode}"
        self.mode = mode
        self.p_stop = p_stop
        self.audit_fraction = audit_fraction
        self.min_turns = min_turns
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._log_path = None
        if decision_log_dir:
            os.makedirs(decision_log_dir, exist_ok=True)
            self._log_path = os.path.join(decision_log_dir, f"controller_decisions_{os.getpid()}.jsonl")
        logger.info(
            "ContinuationController active: mode=%s p_stop=%.3f audit=%.2f min_turns=%d log=%s",
            mode, p_stop, audit_fraction, min_turns, self._log_path,
        )

    # ------------------------------------------------------------------
    def _session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                st = {"audit": self._rng.random() < self.audit_fraction, "turn": 0, "stopped": False}
                self._sessions[session_id] = st
            return st

    def _continue_prob(self, st: dict[str, Any]) -> float:
        # v0 random mode: constant. The learned head (V_c vs lambda*C through a
        # sigmoid, p_min floor) replaces this method's body.
        return 1.0 - self.p_stop

    # ------------------------------------------------------------------
    def on_turn(self, session_id: str, request_body: dict[str, Any]) -> dict[str, Any] | None:
        """Called per session turn. Returns a synthetic terminal response body
        to STOP the rollout, or None to let the request through."""
        st = self._session(session_id)
        st["turn"] += 1
        turn = st["turn"]

        s_t = self._continue_prob(st)
        eligible = turn > self.min_turns and not st["stopped"]
        sampled_continue = True
        if eligible:
            sampled_continue = self._rng.random() < s_t
        # Audits always complete; the counterfactual decision is still logged.
        action = "continue" if (sampled_continue or st["audit"] or not eligible) else "stop"

        self._log(
            {
                "session_id": session_id,
                "turn": turn,
                "s_t": s_t,
                "eligible": eligible,
                "sampled_continue": sampled_continue,
                "audit": st["audit"],
                "action": action,
                "controller_version": CONTROLLER_VERSION,
                "ts": time.time(),
            }
        )

        if action != "stop":
            return None
        st["stopped"] = True
        logger.info("[controller] stopping session %s at turn %d (s_t=%.3f)", session_id, turn, s_t)
        return self._terminal_response(request_body)

    # ------------------------------------------------------------------
    def _terminal_response(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """OpenAI-format response carrying the harness's own exit action."""
        return {
            "id": f"chatcmpl-ctrlstop-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.get("model", "unknown"),
            "rllm_controller": "stopped",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Stopping this attempt now.",
                        "tool_calls": [
                            {
                                "id": f"call_ctrl_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": STOP_SENTINEL_CMD}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _log(self, record: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        line = json.dumps(record)
        with self._lock, open(self._log_path, "a") as f:
            f.write(line + "\n")
