"""
runtime.py - the playing side of the bot (numpy only, no numba).

Flow per decision:
    1. walk the public betting tree with this hand's action history -> node
    2. bucket our hand: preflop from the precomputed equity table, postflop
       from a quick Monte Carlo equity vs a random hand -> percentile bucket
    3. look up the mixed strategy of the active model at (node, bucket) and
       sample an action

Opponent exploitation: every opponent action (node, action) is scored under
each opponent model's action frequencies (equilibrium and 7 archetypes).
Once one archetype explains the opponent clearly better than equilibrium,
we switch to that archetype's restricted-Nash-response strategy, which
exploits it while staying hard to counter-exploit.
"""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np

from gto import buckets as B
from gto import fasteval as fe
from gto.tree import FOLD, CALL, RHALF, RPOT, NONTERM, Tree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---- tuning knobs ----------------------------------------------------------
# Time bank: the judge gives 8 s per half; plumbing overhead is charged too,
# so our own feature computation aims to use COMPUTE_BUDGET of it.
COMPUTE_BUDGET = 5.0        # seconds per half for hand-strength estimation
EST_PER_HAND = 2.2          # expected feature computations per remaining hand
# Sample ladders (cheap -> precise): (runouts, opponents) for flop/turn,
# opponent hands for the river. The controller picks the most precise level
# whose measured cost fits the per-computation allowance.
LADDER_FT = [(8, 8), (16, 12), (24, 12), (32, 16), (48, 16), (64, 20)]
LADDER_R = [96, 192, 320, 512, 768, 1024]
MAX_MS = 12.0               # never spend more than this on one estimate
MIN_OPP_ACTIONS = 50        # observations before we may switch model
SWITCH_MARGIN = 10.0        # log-likelihood lead (nats) over eq needed to switch
STAY_MARGIN = 4.0           # hysteresis: lead needed to keep exploiting
EQ_ONLY = bool(os.environ.get("GTO_EQ_ONLY"))  # testing: disable exploitation
LOW_VISIT = 2.0             # infosets visited less than this use the fallback

# ---- loaded once at import (startup time is free) ---------------------------
TREE = Tree()
_NODES = TREE.nodes
_STORE = np.load(os.path.join(DATA, "strategy.npz"))
EDGES = B.unpack(_STORE)
QMID = _STORE["qmid"].astype(np.float64)          # (4, KMAX) percentile per bucket
KS = B.KS
MODELS = [str(m) for m in _STORE["models"]]
STRAT = {m: _STORE[f"strat_{m}"].astype(np.float32) for m in MODELS}
VISITS = {m: _STORE[f"visits_{m}"].astype(np.float32) for m in MODELS}
PREFLOP_EQ = np.load(os.path.join(DATA, "preflop_eq.npy"))

OFFSET = np.zeros(len(_NODES), dtype=np.int64)
_off = 0
for _n in _NODES:
    if _n.kind == NONTERM:
        OFFSET[_n.id] = _off
        _off += KS[_n.street] * len(_n.actions)


def _marginal_logp(counts: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Per node log P(action code) for one opponent model, smoothed toward
    the eq model's frequencies, then mixed with uniform for robustness."""
    out = np.full(counts.shape, -50.0, dtype=np.float64)
    for n in _NODES:
        if n.kind != NONTERM:
            continue
        acts = n.actions
        c = counts[n.id, acts]
        pr = prior[n.id, acts]
        pr = pr / pr.sum() if pr.sum() > 0 else np.full(len(acts), 1.0 / len(acts))
        p = (c + 3.0 * pr) / (c.sum() + 3.0)
        p = 0.85 * p + 0.15 / len(acts)
        out[n.id, acts] = np.log(p)
    return out


_eq_counts = _STORE["marg_eq"].astype(np.float64) + 0.25
LOGP = {m: _marginal_logp(_STORE[f"marg_{m}"].astype(np.float64), _eq_counts) for m in MODELS}

_RNG = np.random.default_rng()


def warmup():
    fe.equity([0, 5, 10, 15, 20], [25, 30, 35], 32, _RNG)
    fe.hs_features([0, 5, 10, 15, 20], [25, 30, 35], 8, 8, _RNG)


# ---- helpers ---------------------------------------------------------------

def _bucket_of(hole, board, level: int) -> int:
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    if street == 0:
        return int(B.assign(EDGES, 0, float(PREFLOP_EQ[fe.hand_index(sorted(hole))])))
    if street == 3:
        return int(B.assign(EDGES, 3, fe.equity(hole, board, LADDER_R[level], _RNG)))
    r, m = LADDER_FT[level]
    e, v = fe.hs_features(hole, board, r, m, _RNG)
    return int(B.assign(EDGES, street, e, v))


def _fallback(node, q: float, legal) -> int:
    """Simple strength-based play for infosets the solver rarely visited."""
    facing = node.to_call > 0
    canraise = RPOT in legal
    if facing:
        need = node.to_call / (node.pot + node.to_call)
        if q >= 0.96 and canraise:
            return RPOT
        if q >= max(0.55, 0.3 + need):
            return CALL
        return FOLD
    if q >= 0.9 and canraise:
        return RPOT
    return CALL


class Brain:
    """All per-half state lives in one of these, stored in the memory dict."""

    def __init__(self, num_hands: int = 1000):
        self.rng = random.Random()
        self.num_hands = num_hands
        self.time_used = 0.0
        self.feat_time = 0.0
        # measured ms per estimate at each ladder level (start with rough guesses)
        self.cost = {"ft": [0.3, 0.7, 1.1, 1.6, 2.4, 3.6], "r": [0.5, 0.9, 1.4, 2.2, 3.2, 4.2]}
        self.cache_key = None
        self.cache_bucket = 0
        self.hand_log = {}          # hand_index -> (history codes, my seat, my last action)
        self.ll = {m: 0.0 for m in MODELS}
        self.n_obs = 0
        self.active = "eq"
        self.mode_hands = {m: 0 for m in MODELS}

    # -- decisions ------------------------------------------------------------
    def act(self, obs) -> int:
        t0 = time.perf_counter()
        try:
            return self._act(obs)
        finally:
            self.time_used += time.perf_counter() - t0

    def _act(self, obs) -> int:
        legal = [int(a) for a in obs.legal_actions]
        hist = [int(a.action) for a in obs.action_history]
        seat = 0 if obs.is_button else 1
        try:
            n = TREE.walk(hist)
            node = _NODES[n]
        except (ValueError, IndexError):
            node = None
        street = int(obs.street)

        key = (obs.hand_index, street)
        if key != self.cache_key:
            kind = "r" if street == 3 else "ft"
            level = self._pick_level(kind, obs.hand_index) if street else 0
            t1 = time.perf_counter()
            b = _bucket_of(list(obs.hole_cards), list(obs.board), level)
            dt = time.perf_counter() - t1
            if street:
                self.feat_time += dt
                c = self.cost[kind]
                c[level] = 0.8 * c[level] + 0.2 * dt * 1000.0
            self.cache_key, self.cache_bucket = key, b
        b = self.cache_bucket

        if node is None or node.kind != NONTERM or node.actor != seat or node.actions != legal:
            a = _fallback_obs(obs, QMID[street, b], legal)
        else:
            i = OFFSET[n] + b * len(legal)
            model = self.active
            if VISITS[model][i] < LOW_VISIT:
                model = "eq"
            if VISITS[model][i] < LOW_VISIT:
                a = _fallback(node, QMID[street, b], legal)
            else:
                p = STRAT[model][i:i + len(legal)].astype(np.float64)
                s = p.sum()
                if not np.isfinite(s) or s <= 0:
                    p = STRAT["eq"][i:i + len(legal)].astype(np.float64)
                    s = p.sum()
                r = self.rng.random() * s
                a = legal[-1]
                c = 0.0
                for act, pa in zip(legal, p):
                    c += pa
                    if r < c:
                        a = act
                        break
        self.hand_log[obs.hand_index] = (hist, seat, a)
        return a

    def _pick_level(self, kind: str, hand_index: int) -> int:
        """Most precise sample level whose cost fits the remaining budget."""
        if self.time_used > 6.0:      # safety: total own time near the 8 s bank
            return 0
        left_hands = max(1, self.num_hands - hand_index)
        allowance_ms = 1000.0 * (COMPUTE_BUDGET - self.feat_time) / (left_hands * EST_PER_HAND)
        allowance_ms = min(allowance_ms, MAX_MS)
        cost = self.cost[kind]
        level = 0
        for i, c in enumerate(cost):
            if c <= allowance_ms:
                level = i
        return level

    # -- learning -----------------------------------------------------------------
    def hand_end(self, result) -> None:
        t0 = time.perf_counter()
        try:
            self._hand_end(result)
        finally:
            self.time_used += time.perf_counter() - t0

    def _opp_actions(self, result):
        """(node id, action code) for every opponent decision in the hand."""
        log = self.hand_log.pop(result.hand_index, None)
        if log is None:
            # We never acted: we were the big blind and the button folded at once.
            return [(TREE.root, FOLD)] if result.my_net > 0 and not result.went_to_showdown else []
        hist, seat, mine = log
        out = []
        n = TREE.root
        for a in hist + [mine]:
            node = _NODES[n]
            if node.kind != NONTERM or a not in node.actions:
                return out
            if node.actor != seat:
                out.append((n, a))
            n = node.children[node.actions.index(a)]
        node = _NODES[n]
        if node.kind == NONTERM and node.actor != seat:
            # the opponent's final, unseen action ended the hand
            if result.went_to_showdown and CALL in node.actions:
                out.append((n, CALL))
            elif not result.went_to_showdown and result.my_net > 0 and FOLD in node.actions:
                out.append((n, FOLD))
        return out

    def _hand_end(self, result) -> None:
        for n, a in self._opp_actions(result):
            self.n_obs += 1
            for m in MODELS:
                self.ll[m] += LOGP[m][n, a]
        self.mode_hands[self.active] += 1
        self._choose_model()
        # forget stale logs (hands we never got a result for)
        if len(self.hand_log) > 8:
            for k in sorted(self.hand_log)[:-4]:
                del self.hand_log[k]

    def _choose_model(self) -> None:
        if self.n_obs < MIN_OPP_ACTIONS or EQ_ONLY:
            self.active = "eq"
            return
        best = max(MODELS, key=lambda m: self.ll[m])
        lead = self.ll[best] - self.ll["eq"]
        if best == "eq":
            self.active = "eq"
        elif best == self.active:
            if lead < STAY_MARGIN:
                self.active = "eq"
        elif lead > SWITCH_MARGIN:
            self.active = best
        elif self.active != "eq" and self.ll[self.active] - self.ll["eq"] < STAY_MARGIN:
            self.active = "eq"


def _fallback_obs(obs, q, legal) -> int:
    """Fallback when the history does not map onto the tree (should not happen)."""
    class _N:
        pass
    n = _N()
    n.to_call, n.pot = obs.to_call, obs.pot
    a = _fallback(n, q, legal)
    return a if a in legal else (CALL if CALL in legal else legal[0])
