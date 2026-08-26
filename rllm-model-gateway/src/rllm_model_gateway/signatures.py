"""Action signatures and the duplicate discount (brief §4.4, spec M9).

Two rollouts on the same task can be following near-identical paths — same
commands, same files, same order. The fourth such trajectory adds little even
when each looks individually promising. M1's marginal utility asks "does this
rollout add *outcome* contrast?"; this asks "does it add *behavioural*
diversity?".

The discount is `1 - gamma*D(h_t)`, applied once after the learned/retrieved
value estimates are combined, where `D` is the maximum similarity to a live
sibling. It lowers a redundant rollout's value without declaring it worthless.

Deliberately cheap: normalized command strings, edited-file sets, and command
3-grams compared by Jaccard. No embedding model in the critical path — the
brief rules that out, and this runs inside the gateway's request path.
"""

from __future__ import annotations

import re
from typing import Iterable

_WS = re.compile(r"\s+")
_VOLATILE = re.compile(r"0x[0-9a-f]+|\b\d{4,}\b|/tmp/[^\s]+")
# Paths that look like files the agent touched.
_PATHISH = re.compile(r"[\w./-]+\.(?:py|txt|cfg|toml|ini|json|yaml|yml|rst|md)")
_WRITEISH = re.compile(r">>?\s*(\S+)|sed\s+-i[^\s]*\s+\S*\s*(\S+)|tee\s+(\S+)")


def normalize_cmd(text: str) -> str:
    """Collapse whitespace and mask volatile tokens so reruns compare equal."""
    return _VOLATILE.sub("#", _WS.sub(" ", (text or "").strip()))[:200]


def ngrams(tokens: Iterable[str], n: int = 3) -> set[tuple[str, ...]]:
    toks = list(tokens)
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


class RolloutSignature:
    """Compact behavioural fingerprint of one rollout, updated per turn."""

    __slots__ = ("cmds", "files", "grams", "seq")

    def __init__(self) -> None:
        self.cmds: set[str] = set()
        self.files: set[str] = set()
        self.grams: set[tuple[str, ...]] = set()
        self.seq: list[str] = []

    def update(self, action: str) -> None:
        norm = normalize_cmd(action)
        if not norm:
            return
        self.cmds.add(norm)
        self.seq.append(norm.split(" ")[0])
        if len(self.seq) > 64:          # bounded: only recent order matters
            self.seq = self.seq[-64:]
        self.grams = ngrams(self.seq, 3)
        for m in _PATHISH.findall(action or ""):
            self.files.add(m)
        for groups in _WRITEISH.findall(action or ""):
            for g in groups:
                if g:
                    self.files.add(g)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def similarity(x: RolloutSignature, y: RolloutSignature) -> float:
    """Blend of command, edited-file, and command-order similarity in [0,1].

    Files are weighted highest: two rollouts editing the same file are far more
    redundant than two that merely both ran `ls`.
    """
    return min(1.0, 0.35 * _jaccard(x.cmds, y.cmds)
               + 0.45 * _jaccard(x.files, y.files)
               + 0.20 * _jaccard(x.grams, y.grams))


def max_similarity(target: RolloutSignature, others: Iterable[RolloutSignature]) -> float:
    """D(h_t): similarity to the most similar live sibling."""
    best = 0.0
    for o in others:
        if o is target:
            continue
        best = max(best, similarity(target, o))
        if best >= 1.0:
            break
    return best
