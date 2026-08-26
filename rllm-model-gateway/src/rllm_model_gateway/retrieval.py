"""Controller-only retrieval memory (brief §4.5, spec M8).

The point of this component is the paper's second contribution: a controller
that **transfers across related tasks without transferring the policy**. The
learned head cannot do that alone — its repo one-hot features are identically
zero on a held-out repository, and its task-level base rates have no entry for
an unseen task. Retrieval fills exactly that gap by answering "how did rollouts
that *looked like this*, on tasks *like this one*, turn out?"

    M = {(z_x, e(h_t), u, policy_version)}

`z_x` is a task representation, `e(h_t)` the prefix embedding the head already
computes, `u` the realized utility. Retrieval is a similarity-weighted mean over
the K nearest, blended into the local estimate only where neighbours are dense
and similar:

    V_tr = (1 - rho) * V_c + rho * V_ret

Deliberately no embedding model: `z_x` is hashed TF-IDF over the problem
statement and touched file paths, which keeps this inside the gateway's request
path (brief §4.4 constraint). Memory is controller-only — it never feeds
off-policy actor updates.

Drift: prefix embeddings move as the policy changes, so entries carry a policy
version and are down-weighted with distance from the current one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import deque
from typing import Iterable, Sequence

_TOKEN = re.compile(r"[a-z_][a-z0-9_]{2,}")
_STOP = {"the", "and", "for", "that", "this", "with", "from", "when", "should", "issue",
         "error", "expected", "actual", "have", "not", "but", "are", "was", "were"}


def hash_tfidf(text: str, dims: int = 256, extra_tokens: Iterable[str] = ()) -> list[float]:
    """L2-normalized hashed bag of words. Cheap, dependency-free, stable."""
    counts: dict[int, float] = {}
    toks = [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]
    toks.extend(str(t).lower() for t in extra_tokens)
    if not toks:
        return [0.0] * dims
    for t in toks:
        # NOT builtin hash(): Python randomizes string hashing per process, so
        # vectors written by one run would be meaningless to the next -- which
        # would silently destroy exactly the cross-run transfer this exists for.
        h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=4).digest(), "big") % dims
        counts[h] = counts.get(h, 0.0) + 1.0
    vec = [0.0] * dims
    for h, c in counts.items():
        vec[h] = 1.0 + math.log(c)          # sublinear tf
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


class RetrievalMemory:
    """Time-decayed, stratum-balanced store of controller experience."""

    def __init__(self, capacity: int = 20000, k: int = 16, half_life_s: float = 6 * 3600,
                 version_decay: float = 0.15, max_rho: float = 0.5, k_sim: float = 4.0) -> None:
        self.capacity = capacity
        self.k = k
        self.half_life_s = half_life_s
        self.version_decay = version_decay
        self.max_rho = max_rho
        self.k_sim = k_sim
        self._items: deque = deque(maxlen=capacity)
        self._per_stratum: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, z_x: Sequence[float], e_h: Sequence[float], u: float,
            policy_version: int = 0, stratum: str = "unknown", ts: float | None = None) -> None:
        if len(self._items) == self.capacity:
            old = self._items[0]
            self._per_stratum[old["stratum"]] = max(0, self._per_stratum.get(old["stratum"], 1) - 1)
        self._items.append({"z": list(z_x), "e": list(e_h), "u": float(u),
                            "v": int(policy_version), "stratum": stratum,
                            "ts": float(ts if ts is not None else time.time())})
        self._per_stratum[stratum] = self._per_stratum.get(stratum, 0) + 1

    def query(self, z_x: Sequence[float], e_h: Sequence[float], policy_version: int = 0,
              now: float | None = None) -> tuple[float, float]:
        """Return (V_ret, rho). rho is 0 when evidence is thin or dissimilar."""
        if not self._items:
            return 0.0, 0.0
        now = now if now is not None else time.time()
        scored = []
        for it in self._items:
            sim = 0.6 * _cos(z_x, it["z"]) + 0.4 * _cos(e_h, it["e"])
            if sim <= 0.0:
                continue
            age = max(0.0, now - it["ts"])
            w = sim * (0.5 ** (age / self.half_life_s))
            w *= math.exp(-self.version_decay * abs(policy_version - it["v"]))
            if w > 0:
                scored.append((w, it["u"]))
        if not scored:
            return 0.0, 0.0
        scored.sort(key=lambda t: -t[0])
        top = scored[: self.k]
        mass = sum(w for w, _ in top)
        if mass <= 0:
            return 0.0, 0.0
        v_ret = sum(w * u for w, u in top) / mass
        # Confidence grows with neighbour mass and saturates at max_rho, so a
        # thin or dissimilar memory cannot override the local estimate.
        rho = self.max_rho * (mass / (mass + self.k_sim))
        return v_ret, rho

    # -- persistence: what makes transfer ACROSS runs possible ---------------
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            for it in self._items:
                fh.write(json.dumps(it) + "\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, **kw) -> "RetrievalMemory":
        m = cls(**kw)
        if not os.path.exists(path):
            return m
        with open(path) as fh:
            for line in fh:
                try:
                    it = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m.add(it["z"], it["e"], it["u"], it.get("v", 0), it.get("stratum", "unknown"), it.get("ts"))
        return m
