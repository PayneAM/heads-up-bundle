"""
fasteval.py - vectorised Short Deck PLO5 hand evaluation and equity (numpy only).

Trick: a five-card hand's score depends only on its rank multiset and on
whether it is a flush. Give each card the weight 5**rank; the sum of five
weights is a unique key for the rank multiset (no rank appears more than 4
times), and keys are ADDITIVE, so a 2-hole + 3-board combination's key is
just pair_key + triple_key. Two flat tables (non-flush / flush) indexed by
that key give the evaluator's exact score.
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from math import comb

import numpy as np

from evaluator import _score_ranks  # reference scoring logic (exact same scores)

NUM_CARDS = 36
KEY_SIZE = 5 ** 9

RANK_W = np.array([5 ** (c >> 2) for c in range(NUM_CARDS)], dtype=np.int64)
SUIT = np.array([c & 3 for c in range(NUM_CARDS)], dtype=np.int64)

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)


def _build_tables() -> tuple[np.ndarray, np.ndarray]:
    nf = np.zeros(KEY_SIZE, dtype=np.int32)
    fl = np.zeros(KEY_SIZE, dtype=np.int32)
    for ranks in combinations_with_replacement(range(9), 5):
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue
        key = sum(5 ** r for r in ranks)
        desc = tuple(sorted(ranks, reverse=True))
        nf[key] = _score_ranks(desc, False)
        if len(set(ranks)) == 5:
            fl[key] = _score_ranks(desc, True)
    return nf, fl


NF_TABLE, FL_TABLE = _build_tables()

# Combination index of a sorted 5-card hand (combinatorial number system),
# used for the preflop equity table.
_COMB = np.array([[comb(n, k) for k in range(6)] for n in range(NUM_CARDS + 1)], dtype=np.int64)


def hand_index(sorted_hole) -> int:
    a, b, c, d, e = sorted_hole
    return int(_COMB[a, 1] + _COMB[b, 2] + _COMB[c, 3] + _COMB[d, 4] + _COMB[e, 5])


def _pair_parts(hole: np.ndarray):
    """hole (..., 5) -> pair keys (..., 10) and pair suit codes (-1 = offsuit)."""
    a = hole[..., PAIRS[:, 0]]
    b = hole[..., PAIRS[:, 1]]
    key = RANK_W[a] + RANK_W[b]
    sa, sb = SUIT[a], SUIT[b]
    return key, np.where(sa == sb, sa, -1)


def _triple_parts(board: np.ndarray):
    """board (..., 5) -> triple keys (..., 10) and suit codes (-2 = not monotone)."""
    a = board[..., TRIPLES[:, 0]]
    b = board[..., TRIPLES[:, 1]]
    c = board[..., TRIPLES[:, 2]]
    key = RANK_W[a] + RANK_W[b] + RANK_W[c]
    sa, sb, sc = SUIT[a], SUIT[b], SUIT[c]
    return key, np.where((sa == sb) & (sb == sc), sa, -2)


def _best(pk, ps, tk, ts) -> np.ndarray:
    key = pk[..., :, None] + tk[..., None, :]
    flush = ps[..., :, None] == ts[..., None, :]
    score = np.where(flush, FL_TABLE[key], NF_TABLE[key])
    return score.reshape(score.shape[:-2] + (100,)).max(axis=-1)


def scores(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Omaha scores for holes (N, 5) on boards (N, 5)."""
    pk, ps = _pair_parts(holes)
    tk, ts = _triple_parts(boards)
    return _best(pk, ps, tk, ts)


def equity(hole, board, n: int, rng: np.random.Generator) -> float:
    """Monte Carlo equity of `hole` against ONE uniformly random hand,
    rolling out the rest of the board. hole: 5 ints, board: 0/3/4/5 ints."""
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    used = np.zeros(NUM_CARDS, dtype=bool)
    used[hole] = True
    used[board] = True
    deck = np.flatnonzero(~used)
    need = 5 + (5 - len(board))
    picks = deck[np.argpartition(rng.random((n, len(deck))), need - 1, axis=1)[:, :need]]
    opp = picks[:, :5]
    full_board = np.concatenate([np.broadcast_to(board, (n, len(board))), picks[:, 5:]], axis=1)
    tk, ts = _triple_parts(full_board)
    pk, ps = _pair_parts(hole)
    mine = _best(np.broadcast_to(pk, (n, 10)), np.broadcast_to(ps, (n, 10)), tk, ts)
    pk2, ps2 = _pair_parts(opp)
    theirs = _best(pk2, ps2, tk, ts)
    return float(np.mean((mine > theirs) + 0.5 * (mine == theirs)))


def hs_features(hole, board, runouts: int, opps: int, rng: np.random.Generator):
    """Hand-strength features for flop/turn: (E, V).

    E = expected equity vs a random hand over random runouts, V = variance
    of the river hand strength across runouts. Same E with high V is a draw;
    low V is a made hand. That distinction is what plain equity buckets miss.
    On the river (or if runouts == 0) V is 0 and E is plain equity.
    """
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    nb = len(board)
    if nb == 5 or runouts <= 0:
        return equity(hole, board, max(opps * max(runouts, 1), 64), rng), 0.0
    used = np.zeros(NUM_CARDS, dtype=bool)
    used[hole] = True
    used[board] = True
    deck = np.flatnonzero(~used)                       # (D,)
    need_b = 5 - nb
    R, M = runouts, opps
    perm = np.argsort(rng.random((R, len(deck))), axis=1)
    run = deck[perm[:, :need_b]]                        # (R, need_b)
    rest = deck[perm[:, need_b:]]                       # (R, D - need_b)
    pick = np.argpartition(rng.random((R, M, rest.shape[1])), 5, axis=2)[:, :, :5]
    opp = np.take_along_axis(np.broadcast_to(rest[:, None, :], (R, M, rest.shape[1])), pick, axis=2)
    full = np.concatenate([np.broadcast_to(board, (R, nb)), run], axis=1)   # (R, 5)
    tk, ts = _triple_parts(full)                                            # (R, 10)
    pk, ps = _pair_parts(hole)
    mine = _best(np.broadcast_to(pk, (R, 10)), np.broadcast_to(ps, (R, 10)), tk, ts)  # (R,)
    pk2, ps2 = _pair_parts(opp)                                             # (R, M, 10)
    theirs = _best(pk2, ps2, tk[:, None, :], ts[:, None, :])                # (R, M)
    hs = np.mean((mine[:, None] > theirs) + 0.5 * (mine[:, None] == theirs), axis=1)
    return float(hs.mean()), float(hs.var())
