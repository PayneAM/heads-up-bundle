"""
gen_data.py - OFFLINE: build the preflop equity table and the training deals.

Run from the bundle folder (needs numba, which is NOT needed by the bot):
    python -m gto.offline.gen_data

Writes:
    gto/data/preflop_eq.npy   equity of every 5-card hand vs a random hand,
                              indexed by fasteval.hand_index (shipped with bot)
    gto/offline/deals.npz     training deals: per-street equities of both
                              seats and the showdown result (offline only)
"""

from __future__ import annotations

import os
import sys
import time
from itertools import combinations

import numpy as np
from numba import njit, prange

from gto.fasteval import FL_TABLE, NF_TABLE, RANK_W, SUIT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)
TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.int64)


@njit(cache=True)
def score(hole, board, nf, fl, rw, su, pairs, triples):
    best = 0
    for i in range(10):
        a = hole[pairs[i, 0]]
        b = hole[pairs[i, 1]]
        pk = rw[a] + rw[b]
        ps = su[a] if su[a] == su[b] else -1
        for j in range(10):
            c = board[triples[j, 0]]
            d = board[triples[j, 1]]
            e = board[triples[j, 2]]
            k = pk + rw[c] + rw[d] + rw[e]
            if ps >= 0 and su[c] == ps and su[d] == ps and su[e] == ps:
                s = fl[k]
            else:
                s = nf[k]
            if s > best:
                best = s
    return best


@njit(cache=True)
def equity_mc(hole, board, nb, n, nf, fl, rw, su, pairs, triples):
    """Equity of hole vs one random hand, nb known board cards (board[:nb])."""
    used = np.zeros(36, dtype=np.bool_)
    for i in range(5):
        used[hole[i]] = True
    for i in range(nb):
        used[board[i]] = True
    deck = np.empty(36, dtype=np.int64)
    m = 0
    for c in range(36):
        if not used[c]:
            deck[m] = c
            m += 1
    need = 5 + (5 - nb)
    opp = np.empty(5, dtype=np.int64)
    fb = np.empty(5, dtype=np.int64)
    for i in range(nb):
        fb[i] = board[i]
    tot = 0.0
    for _ in range(n):
        # partial Fisher-Yates
        for i in range(need):
            j = i + np.random.randint(m - i)
            t = deck[i]
            deck[i] = deck[j]
            deck[j] = t
        for i in range(5):
            opp[i] = deck[i]
        for i in range(5 - nb):
            fb[nb + i] = deck[5 + i]
        s1 = score(hole, fb, nf, fl, rw, su, pairs, triples)
        s2 = score(opp, fb, nf, fl, rw, su, pairs, triples)
        if s1 > s2:
            tot += 1.0
        elif s1 == s2:
            tot += 0.5
    return tot / n


@njit(parallel=True, cache=True)
def preflop_table(hands, n, nf, fl, rw, su, pairs, triples):
    out = np.empty(hands.shape[0], dtype=np.float32)
    empty = np.zeros(5, dtype=np.int64)
    for i in prange(hands.shape[0]):
        out[i] = equity_mc(hands[i], empty, 0, n, nf, fl, rw, su, pairs, triples)
    return out


@njit(cache=True)
def hs_mc(hole, board, nb, R, M, nf, fl, rw, su, pairs, triples):
    """(E, V): mean and variance over R runouts of equity vs M random hands."""
    used = np.zeros(36, dtype=np.bool_)
    for i in range(5):
        used[hole[i]] = True
    for i in range(nb):
        used[board[i]] = True
    deck = np.empty(36, dtype=np.int64)
    m = 0
    for c in range(36):
        if not used[c]:
            deck[m] = c
            m += 1
    nr = 5 - nb
    fb = np.empty(5, dtype=np.int64)
    for i in range(nb):
        fb[i] = board[i]
    opp = np.empty(5, dtype=np.int64)
    s1 = 0.0
    s2 = 0.0
    for _ in range(R):
        for i in range(nr):
            j = i + np.random.randint(m - i)
            t = deck[i]; deck[i] = deck[j]; deck[j] = t
            fb[nb + i] = deck[i]
        mine = score(hole, fb, nf, fl, rw, su, pairs, triples)
        w = 0.0
        for _k in range(M):
            for i in range(5):
                j = nr + i + np.random.randint(m - nr - i)
                t = deck[nr + i]; deck[nr + i] = deck[j]; deck[j] = t
                opp[i] = deck[nr + i]
            th = score(opp, fb, nf, fl, rw, su, pairs, triples)
            if mine > th:
                w += 1.0
            elif mine == th:
                w += 0.5
        hs = w / M
        s1 += hs
        s2 += hs * hs
    e = s1 / R
    return e, max(0.0, s2 / R - e * e)


@njit(parallel=True, cache=True)
def make_deals(seed, count, nf, fl, rw, su, pairs, triples):
    """feat[d, seat, :] = flopE, flopV, turnE, turnV, riverE; res = +1/-1/0 for seat 0."""
    feat = np.empty((count, 2, 5), dtype=np.float32)
    holes = np.empty((count, 2, 5), dtype=np.int64)
    res = np.empty(count, dtype=np.int8)
    for d in prange(count):
        np.random.seed(seed + d)
        perm = np.random.permutation(36)[:15]
        board = perm[10:15].copy()
        for s in range(2):
            holes[d, s] = np.sort(perm[5 * s:5 * s + 5])
            e, v = hs_mc(holes[d, s], board, 3, 40, 20, nf, fl, rw, su, pairs, triples)
            feat[d, s, 0] = e
            feat[d, s, 1] = v
            e, v = hs_mc(holes[d, s], board, 4, 26, 24, nf, fl, rw, su, pairs, triples)
            feat[d, s, 2] = e
            feat[d, s, 3] = v
            feat[d, s, 4] = equity_mc(holes[d, s], board, 5, 500, nf, fl, rw, su, pairs, triples)
        a = score(holes[d, 0], board, nf, fl, rw, su, pairs, triples)
        b = score(holes[d, 1], board, nf, fl, rw, su, pairs, triples)
        res[d] = 1 if a > b else (-1 if b > a else 0)
    return holes, feat, res


def main():
    args = (NF_TABLE.astype(np.int32), FL_TABLE.astype(np.int32), RANK_W, SUIT, PAIRS, TRIPLES)
    os.makedirs(DATA, exist_ok=True)
    pf_path = os.path.join(DATA, "preflop_eq.npy")
    if not os.path.exists(pf_path) or "--force" in sys.argv:
        hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)
        t = time.time()
        table = preflop_table(hands, 3000, *args)
        np.save(pf_path, table)
        print(f"preflop table: {len(table)} hands, {time.time() - t:.1f}s, "
              f"range {table.min():.3f}..{table.max():.3f}")
    count = int(os.environ.get("DEALS", "1000000"))
    t = time.time()
    holes, feat, res = make_deals(777, count, *args)
    np.savez(os.path.join(HERE, "deals.npz"), holes=holes, feat=feat, res=res)
    print(f"deals: {count}, {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
