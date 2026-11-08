"""
buckets.py - v2 card abstraction, shared by the trainer and the bot.

    preflop : 20 buckets on preflop equity (table)
    flop    : 10 equity buckets x 3 potential buckets = 30
    turn    : 10 equity buckets x 3 potential buckets = 30
    river   : 20 buckets on equity

"Potential" is the variance of the river hand strength across runouts:
for the same equity, high variance = a draw, low variance = a made hand.
Equity cut points are quantiles, finer near the top where nut-ness matters.
"""

from __future__ import annotations

import numpy as np

KS = np.array([20, 30, 30, 20], dtype=np.int64)
KMAX = int(KS.max())
NV = 3  # potential buckets per equity bucket on flop/turn

Q20 = np.array([0.10, 0.20, 0.29, 0.37, 0.44, 0.51, 0.57, 0.63, 0.68, 0.73,
                0.77, 0.81, 0.845, 0.875, 0.90, 0.925, 0.945, 0.963, 0.98])
Q10 = np.array([0.15, 0.30, 0.45, 0.58, 0.70, 0.80, 0.88, 0.94, 0.975])


def _mid(q):
    lo = np.r_[0, q]
    hi = np.r_[q, 1]
    return (lo + hi) / 2


def fit(pf_eq, feat):
    """Fit cut points from training data.
    pf_eq: (N,) preflop equities; feat: (N, 5) flopE, flopV, turnE, turnV, riverE."""
    edges = {"pf": np.quantile(pf_eq, Q20), "river": np.quantile(feat[:, 4], Q20)}
    for st, (ce, cv) in ((1, (0, 1)), (2, (2, 3))):
        e_edges = np.quantile(feat[:, ce], Q10)
        eb = np.searchsorted(e_edges, feat[:, ce], side="right")
        v_edges = np.zeros((len(Q10) + 1, NV - 1))
        for b in range(len(Q10) + 1):
            v = feat[eb == b, cv]
            v_edges[b] = np.quantile(v, [1 / 3, 2 / 3]) if len(v) else [0.0, 0.0]
        edges[f"e{st}"] = e_edges
        edges[f"v{st}"] = v_edges
    qmid = np.zeros((4, KMAX))
    qmid[0, :20] = _mid(Q20)
    qmid[3, :20] = _mid(Q20)
    for st in (1, 2):
        qmid[st, :30] = np.repeat(_mid(Q10), NV)  # bucket = e * NV + v
    return edges, qmid


def assign(edges, street: int, e, v=None):
    """Bucket(s) for street given equity e (and potential v on flop/turn). Vectorised."""
    if street == 0:
        return np.searchsorted(edges["pf"], e, side="right")
    if street == 3:
        return np.searchsorted(edges["river"], e, side="right")
    eb = np.searchsorted(edges[f"e{street}"], e, side="right")
    ve = edges[f"v{street}"][eb]                      # (..., 2)
    vb = (np.asarray(v)[..., None] > ve).sum(axis=-1)
    return eb * NV + vb


def pack(edges):
    return {f"edges_{k}": v for k, v in edges.items()}


def unpack(store):
    return {k[len("edges_"):]: np.asarray(store[k], dtype=np.float64)
            for k in store.files if k.startswith("edges_")}
