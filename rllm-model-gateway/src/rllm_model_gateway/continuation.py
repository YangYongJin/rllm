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

import hashlib
import json
import logging
import os
import random
import math
import threading
import time
import uuid
from typing import Any

from rllm_model_gateway.group_utility import expected_U, marginal_dU

logger = logging.getLogger(__name__)

CONTROLLER_VERSION = "v0-random-1"  # legacy default; see _version_string()
RELOAD_STAT_EVERY = 50


def _version_string(mode: str, rule: str, head: dict | None) -> str:
    """Identify the controller that produced a decision record.

    The old constant was stamped on every record regardless of mode, so logs
    from a learned run were indistinguishable from a random one and could not
    be traced to a head.
    """
    if head is None:
        return f"{mode}-{rule}-nohead"
    payload = json.dumps(
        {"w": head.get("weights"), "b": head.get("bias"), "f": head.get("feature_names")},
        sort_keys=True,
    ).encode()
    return f"{mode}-{rule}-{hashlib.blake2b(payload, digest_size=4).hexdigest()}"
STOP_SENTINEL_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def _load_json_or_none(path: str | None, what: str) -> dict | None:
    if not path:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        logger.exception("could not load %s from %s; falling back", what, path)
        return None


def controller_from_env() -> "ContinuationController | None":
    if os.environ.get("RLLM_CONTROLLER_ENABLE", "0") not in ("1", "true", "True"):
        return None
    return ContinuationController(
        cost_head=_load_json_or_none(os.environ.get("RLLM_CONTROLLER_COST_HEAD"), "cost head"),
        mode=os.environ.get("RLLM_CONTROLLER_MODE", "random"),
        head_path=os.environ.get("RLLM_CONTROLLER_HEAD") or None,
        lam=float(os.environ.get("RLLM_CONTROLLER_LAMBDA", "0.3")),
        beta=float(os.environ.get("RLLM_CONTROLLER_BETA", "0.0")),
        online_stats=os.environ.get("RLLM_CONTROLLER_ONLINE_STATS", "0") in ("1", "true", "True"),
        group_floor=int(os.environ.get("RLLM_CONTROLLER_GROUP_FLOOR", "0")),
        rule=os.environ.get("RLLM_CONTROLLER_RULE", "sigmoid_diff"),
        tau=float(os.environ.get("RLLM_CONTROLLER_TAU", "0.02")),
        max_contrast_loss=float(os.environ.get("RLLM_CONTROLLER_MAX_CONTRAST_LOSS", "0.0")),
        hot_reload=os.environ.get("RLLM_CONTROLLER_HOT_RELOAD", "0") in ("1", "true", "True"),
        gamma_dup=float(os.environ.get("RLLM_CONTROLLER_GAMMA_DUP", "0.0")),
        task_rates_path=os.environ.get("RLLM_CONTROLLER_TASK_RATES") or None,
        temperature=float(os.environ.get("RLLM_CONTROLLER_TEMPERATURE", "0.15")),
        p_min=float(os.environ.get("RLLM_CONTROLLER_P_MIN", "0.05")),
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
        head_path: str | None = None,
        lam: float = 0.3,
        beta: float = 0.0,
        online_stats: bool = False,
        group_floor: int = 0,
        task_rates_path: str | None = None,
        rule: str = "sigmoid_diff",
        tau: float = 0.02,
        max_contrast_loss: float = 0.0,
        hot_reload: bool = False,
        cost_head: dict | None = None,
        gamma_dup: float = 0.0,
        temperature: float = 0.15,
        p_min: float = 0.05,
        p_stop: float = 0.15,
        audit_fraction: float = 0.1,
        min_turns: int = 2,
        decision_log_dir: str | None = None,
        seed: int | None = None,
    ) -> None:
        assert mode in ("random", "learned"), f"unsupported controller mode: {mode}"
        self.mode = mode
        self.lam, self.beta, self.temperature, self.p_min = lam, beta, temperature, p_min
        # OFF by default: observe_outcome() had no caller until 2026-08-25, so
        # every run to date used the frozen priors. Keep that reproducible.
        self.online_stats = online_stats
        # v3-M2: never let a task's group of rollouts fall below this many
        # survivors. A GRPO group needs >=2 non-identical members to yield any
        # advantage; stopping siblings can otherwise discard the one rare
        # success on a hard task (see V3_DESIGN.md D1). 0 disables the floor.
        self.group_floor = group_floor
        self._group_alive: dict[str, int] = {}
        # v3: rule='ratio' scores marginal group gradient per remaining
        # token against a shadow price tau. 'sigmoid_diff' keeps v1/v2.
        self.rule = rule
        self.tau = tau
        # Relative-contrast guard: refuse any stop that would destroy more than
        # this fraction of the group's remaining expected gradient. Adapts to
        # the group's actual state, unlike the fixed count floor. 0 disables.
        self.max_contrast_loss = max_contrast_loss
        self.hot_reload = hot_reload
        self.cost_head = cost_head
        self.gamma_dup = gamma_dup
        # group_key -> {session_id: p_hat (or settled 0.0/1.0)}
        self._group_p: dict[str, dict[str, float]] = {}
        # v3-M3: per-task solve rates beat per-repo ones (within-repo variance
        # is what decides a group's fate). Falls back repo -> global prior.
        self._task_rates: dict[str, float] = {}
        if task_rates_path:
            try:
                with open(task_rates_path) as fh:
                    raw = json.load(fh)
                for k, v in raw.items():
                    n = float(v.get("attempts", 0) or 0)
                    if n > 0:
                        self._task_rates[k] = float(v.get("solves", 0) or 0) / n
                logger.info("loaded %d task solve rates from %s", len(self._task_rates), task_rates_path)
            except Exception:
                logger.exception("could not load task rates from %s", task_rates_path)
        self._head = None
        self._head_path = head_path
        self._head_mtime = 0.0
        self._decisions_since_stat = 0
        if mode == "learned":
            with open(head_path) as f:
                self._head = json.load(f)
            try:
                self._head_mtime = os.path.getmtime(head_path)
            except OSError:
                pass
            self._b_x: dict[str, list[float]] = {}  # repo -> [solves, total] success EMA basis
            self._turn_costs: list[int] = []  # completed-session turn counts (C estimate)
        self.p_stop = p_stop
        self.audit_fraction = audit_fraction
        self.min_turns = min_turns
        self._rng = random.Random(seed)
        self._audit_salt = str(seed if seed is not None else 0)
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._log_path = None
        if decision_log_dir:
            os.makedirs(decision_log_dir, exist_ok=True)
            self._log_path = os.path.join(decision_log_dir, f"controller_decisions_{os.getpid()}.jsonl")
        self.version = _version_string(mode, rule, self._head)
        logger.info(
            "ContinuationController active: version=%s mode=%s rule=%s tau=%.4f p_stop=%.3f "
            "audit=%.2f min_turns=%d floor=%d max_contrast_loss=%.2f online_stats=%s log=%s",
            self.version, mode, rule, tau, p_stop, audit_fraction, min_turns,
            group_floor, max_contrast_loss, online_stats, self._log_path,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _group_key(session_id: str) -> str:
        """Group = the task; sessions are '<task_id>:<rollout_idx>[:val]'."""
        return session_id.rsplit(":", 1)[0] if session_id.count(":") >= 1 else session_id

    def _base_rate(self, session_id: str, repo: str) -> float:
        """b(x): task rate if known, else per-repo online rate, else prior."""
        task = self._group_key(session_id).split("/")[-1]
        if task in self._task_rates:
            return self._task_rates[task]
        solved, total = self._b_x.get(repo, [0.0, 0.0])
        return (solved + 1.0) / (total + 8.0)

    def _is_audit(self, session_id: str) -> bool:
        """Deterministic audit membership, derived from the session id.

        Previously this was sampled once and kept in mutable per-session state.
        Any state loss (gateway restart — method_9b_v1 had 12) re-sampled it, so
        a session admitted as an audit could be re-admitted as non-audit and then
        STOPPED: 59 sessions changed audit flag mid-stream, contaminating the very
        stream the false-stop rate is estimated from. Hashing is stable across
        restarts and processes, keeps the expected fraction, and makes audit
        membership reproducible from the session id alone.
        """
        h = hashlib.blake2b(f"{self._audit_salt}:{session_id}".encode(), digest_size=8).digest()
        return (int.from_bytes(h, "big") % 1_000_000) < int(self.audit_fraction * 1_000_000)

    def _session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            st = self._sessions.get(session_id)
            if st is None:
                st = {"audit": self._is_audit(session_id), "turn": 0, "stopped": False,
                      "session_id": session_id}
                self._sessions[session_id] = st
                gk = self._group_key(session_id)
                self._group_alive[gk] = self._group_alive.get(gk, 0) + 1
            return st

    def _maybe_reload_head(self) -> None:
        """Pick up a head refit by controller/online_refresh.py (M4).

        The refresher writes atomically (tmp + os.replace), so a changed mtime
        means a complete file. Statting every decision would be wasteful, so we
        check every RELOAD_STAT_EVERY decisions -- at ~30 rollouts/step that is
        several times per optimizer step, far finer than the refresh interval.
        """
        if not self.hot_reload or self.mode != "learned" or not self._head_path:
            return
        self._decisions_since_stat += 1
        if self._decisions_since_stat < RELOAD_STAT_EVERY:
            return
        self._decisions_since_stat = 0
        try:
            mt = os.path.getmtime(self._head_path)
        except OSError:
            return
        if mt <= self._head_mtime:
            return
        try:
            with open(self._head_path) as fh:
                new_head = json.load(fh)
            if not new_head.get("feature_names"):
                raise ValueError("head has no feature_names")
        except Exception:
            logger.exception("head reload failed; keeping the current head")
            return
        with self._lock:
            self._head = new_head
            self._head_mtime = mt
            self.version = _version_string(self.mode, self.rule, new_head)
        logger.info("controller head reloaded -> version=%s", self.version)

    def _p_hat(self, st: dict[str, Any]) -> float:
        """Head estimate that this rollout eventually solves its task."""
        h = self._head
        feats = st.get("features") or {}
        x = [(float(feats.get(k, 0.0)) - m) / s_ for k, m, s_ in zip(h["feature_names"], h["mu"], h["sd"])]
        z = sum(wi * xi for wi, xi in zip(h["weights"], x)) + h["bias"]
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def _remaining_cost(self, st: dict[str, Any]) -> float:
        """Expected remaining cost of this rollout, in turns (>=1).

        v3 uses a cost head when one is loaded; otherwise the v1/v2 fallback of
        (mean_turns - t), which is a single global constant for every rollout.
        """
        if self.cost_head:
            h = self.cost_head
            feats = st.get("features") or {}
            x = [(float(feats.get(k, 0.0)) - m) / s_ for k, m, s_ in zip(h["feature_names"], h["mu"], h["sd"])]
            pred = sum(wi * xi for wi, xi in zip(h["weights"], x)) + h["bias"]
            return max(1.0, float(pred))
        mean_turns = (sum(self._turn_costs) / len(self._turn_costs)) if self._turn_costs else 12.0
        return max(1.0, mean_turns - st["turn"])

    def _continue_prob_ratio(self, st: dict[str, Any], session_id: str) -> float:
        """v3: marginal group gradient per remaining token, vs shadow price tau.

        Values a rollout by what it adds to its GROUP's expected advantage mass
        rather than by its own success probability -- see group_utility.py.
        """
        gk = self._group_key(session_id)
        with self._lock:
            members = dict(self._group_p.get(gk) or {})
        members[session_id] = self._p_hat(st)
        ids = sorted(members)
        ps = [members[k] for k in ids]
        try:
            idx = ids.index(session_id)
        except ValueError:  # pragma: no cover - defensive
            return 1.0 - self.p_stop
        d_u = marginal_dU(idx, ps)
        if self.max_contrast_loss > 0.0:
            u_now = expected_U(ps)
            st["_contrast_frac"] = (d_u / u_now) if u_now > 0 else 0.0
        if self.gamma_dup > 0.0:
            d_u *= max(0.0, 1.0 - self.gamma_dup * float(st.get("dup_sim", 0.0)))
        score = d_u / max(self._remaining_cost(st), 1e-6)
        z = max(-30.0, min(30.0, (score - self.tau) / max(self.temperature, 1e-6)))
        return self.p_min + (1.0 - self.p_min) * (1.0 / (1.0 + math.exp(-z)))

    def _continue_prob(self, st: dict[str, Any]) -> float:
        if self.mode == "random" or self._head is None:
            return 1.0 - self.p_stop
        # Learned mode (PROJECT_BRIEF §4.2/§4.6):
        #   p_hat = sigmoid(head(features));  V_c = p_hat*(1-b) + (1-p_hat)*b
        #   C     = expected remaining turns / cap (EWMA over completed sessions)
        #   s_t   = p_min + (1-p_min) * sigmoid((V_c - lam*C) / T)
        p_hat = self._p_hat(st)
        repo = st.get("repo", "unknown")
        b = self._base_rate(st.get("session_id", ""), repo)
        v_c = p_hat * (1.0 - b) + (1.0 - p_hat) * b
        mean_turns = (sum(self._turn_costs) / len(self._turn_costs)) if self._turn_costs else 12.0
        remaining = max(0.0, mean_turns - st["turn"]) / max(mean_turns, 1.0)
        # beta re-centers the per-turn hazard: s_t compounds across turns, so
        # typical prefixes must sit near s_t~0.9+, not ~0.5, for sane trajectory-
        # level stop rates. beta>0 shifts the operating point accordingly.
        z2 = (v_c - self.lam * remaining + self.beta) / self.temperature
        z2 = max(-30.0, min(30.0, z2))
        return self.p_min + (1.0 - self.p_min) * (1.0 / (1.0 + math.exp(-z2)))

    # ------------------------------------------------------------------
    def on_turn(self, session_id: str, request_body: dict[str, Any]) -> dict[str, Any] | None:
        """Called per session turn. Returns a synthetic terminal response body
        to STOP the rollout, or None to let the request through."""
        # Validation sessions (":val"-suffixed by AgentFlowEngine) are NEVER
        # stopped: the controller is a training-cost intervention; val must
        # measure the policy exactly like an uncontrolled baseline.
        if session_id.endswith(":val"):
            return None
        self._maybe_reload_head()
        st = self._session(session_id)
        st["turn"] += 1
        turn = st["turn"]
        if self.mode == "learned":
            self._update_features(st, session_id, request_body)

        if self.rule == "ratio" and self.mode == "learned" and self._head is not None:
            s_t = self._continue_prob_ratio(st, session_id)
            gk_reg = self._group_key(session_id)
            with self._lock:
                self._group_p.setdefault(gk_reg, {})[session_id] = self._p_hat(st)
        else:
            s_t = self._continue_prob(st)
        eligible = turn > self.min_turns and not st["stopped"]
        sampled_continue = True
        if eligible:
            sampled_continue = self._rng.random() < s_t
        # v3-M2 group floor: a GRPO group needs >=2 members with differing
        # rewards to produce any advantage, so stopping the last survivors
        # destroys the very signal the rollouts were paid for (worst case: the
        # one rare success on a hard task). Hold, and log it distinctly so the
        # propensity accounting stays exact.
        gk = self._group_key(session_id)
        floor_hold = False
        if eligible and not sampled_continue:
            if self.group_floor > 0:
                with self._lock:
                    if self._group_alive.get(gk, 0) <= self.group_floor:
                        floor_hold = True
            if (not floor_hold and self.max_contrast_loss > 0.0
                    and st.get("_contrast_frac", 0.0) > self.max_contrast_loss):
                floor_hold = True   # would destroy too much of the group's gradient
        # Audits always complete; the counterfactual decision is still logged.
        stopping = eligible and not sampled_continue and not st["audit"] and not floor_hold
        action = "stop" if stopping else ("floor_hold" if floor_hold else "continue")
        if stopping:
            with self._lock:
                self._group_alive[gk] = max(0, self._group_alive.get(gk, 0) - 1)
                # A stopped rollout leaves the group entirely: it is excluded
                # from the actor batch, so it can no longer contribute contrast.
                self._group_p.get(gk, {}).pop(session_id, None)

        self._log(
            {
                "session_id": session_id,
                "turn": turn,
                "s_t": s_t,
                "eligible": eligible,
                "sampled_continue": sampled_continue,
                "audit": st["audit"],
                "action": action,
                "group_alive": self._group_alive.get(gk, 0),
                "floor_hold": floor_hold,
                "controller_version": self.version,
                "ts": time.time(),
            }
        )

        if action != "stop":
            return None
        st["stopped"] = True
        logger.info("[controller] stopping session %s at turn %d (s_t=%.3f)", session_id, turn, s_t)
        return self._terminal_response(request_body)

    # ------------------------------------------------------------------
    _REPOS = ("aiohttp", "coveragepy", "datalad", "numpy", "orange3", "pandas", "pillow", "pyramid", "scrapy", "tornado")

    def _update_features(self, st: dict[str, Any], session_id: str, request_body: dict[str, Any]) -> None:
        """Cheap prefix features mirroring controller/featurize.py (char-based)."""
        msgs = request_body.get("messages") or []
        gen = sum(len(str(m.get("content") or "")) for m in msgs if m.get("role") == "assistant")
        obs = sum(len(str(m.get("content") or "")) for m in msgs if m.get("role") in ("tool", "user"))
        turn = st["turn"]
        if "repo" not in st:
            st["repo"] = next((r for r in self._REPOS if r in session_id), "unknown")
        last_gen = gen - st.get("_prev_gen", 0)
        last_obs = obs - st.get("_prev_obs", 0)
        st["_prev_gen"], st["_prev_obs"] = gen, obs
        feats = {
            "turn": float(turn),
            "frac_of_cap": turn / 40.0,
            "cum_gen_chars": float(gen),
            "cum_obs_chars": float(obs),
            "last_gen_chars": float(max(0, last_gen)),
            "last_obs_chars": float(max(0, last_obs)),
            "gen_rate": gen / max(1, turn),
        }
        for r in self._REPOS:
            feats[f"repo_{r}"] = 1.0 if r == st["repo"] else 0.0
        st["features"] = feats

    def observe_outcome(self, session_id: str, solved: bool, turns: int) -> None:
        """Online feedback: update b(x) and the remaining-turn cost estimate.

        Fed by the engine via POST /controller/outcome after each episode is
        graded. Gated on ``online_stats`` so an existing run's operating point
        stays reproducible: with it OFF, b(x) stays at its 0.125 prior and
        mean_turns at 12.0 (that is what method_9b_v1 ran).
        """
        if self.mode != "learned" or not self.online_stats:
            return
        if session_id.endswith(":val"):
            return  # validation must not steer the training-time controller
        with self._lock:
            st = self._sessions.get(session_id) or {}
            # Derive the repo from the session id rather than session state:
            # state may already have been evicted by the time grading finishes.
            repo = st.get("repo") or next((r for r in self._REPOS if r in session_id), "unknown")
            rec = self._b_x.setdefault(repo, [0.0, 0.0])
            rec[0] += 1.0 if solved else 0.0
            rec[1] += 1.0
            # Settle this member for the group's utility calculation, so live
            # siblings score against a known outcome rather than an estimate.
            gk = self._group_key(session_id)
            if gk in self._group_p and session_id in self._group_p[gk]:
                self._group_p[gk][session_id] = 1.0 if solved else 0.0
            self._turn_costs.append(turns)
            if len(self._turn_costs) > 500:
                self._turn_costs = self._turn_costs[-500:]

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
