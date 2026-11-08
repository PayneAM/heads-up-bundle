"""
train.py - OFFLINE: solve the abstracted game with MCCFR and build the
exploitative "restricted Nash response" strategies.

    python -m gto.offline.train                 # all models
    ITERS=2000000 python -m gto.offline.train   # more iterations

Abstraction (v2): the exact public betting tree (gto/tree.py) x the card
buckets of gto/buckets.py (20 preflop, 30 flop, 30 turn, 20 river; flop and
turn split by equity AND draw potential). Infoset = (node, own bucket).
The equilibrium is trained as several independent seeds whose average
strategies are combined (the maximin set of a zero-sum game is convex).

Models:
    eq        self-play equilibrium of the abstract game (the GTO core)
    <arch>    restricted Nash response vs an archetype: the opponent plays the
              archetype with prob p_lock and a best-responding strategy
              otherwise, so the counter-strategy exploits but stays robust.

Output: gto/data/strategy.npz (shipped with the bot).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from numba import njit

from gto import buckets as B
from gto.tree import FOLDED, NONTERM, SHOWDOWN, Tree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

KS = B.KS

ARCHETYPES = ["station", "maniac", "nit", "random", "half", "pot", "honest"]
# How much each counter-strategy assumes the opponent IS the archetype. Lower =
# less exploitative but much harder to counter-exploit (v2.1: all 0.5; was 0.6-0.8).
P_LOCK = {"station": 0.5, "maniac": 0.5, "nit": 0.5, "random": 0.5,
          "half": 0.5, "pot": 0.5, "honest": 0.5}


def flatten(tree: Tree):
    n = len(tree.nodes)
    kind = np.zeros(n, np.int8)
    street = np.zeros(n, np.int8)
    actor = np.zeros(n, np.int8)
    nact = np.zeros(n, np.int8)
    child = np.full((n, 4), -1, np.int32)
    acts = np.full((n, 4), -1, np.int8)
    c0 = np.zeros(n, np.float64)
    c1 = np.zeros(n, np.float64)
    folder = np.full(n, -1, np.int8)
    pot = np.zeros(n, np.float64)
    tocall = np.zeros(n, np.float64)
    offset = np.zeros(n, np.int64)
    off = 0
    for nd in tree.nodes:
        i = nd.id
        kind[i], street[i], actor[i] = nd.kind, nd.street, nd.actor
        c0[i], c1[i], folder[i], pot[i], tocall[i] = nd.contrib[0], nd.contrib[1], nd.folder, nd.pot, nd.to_call
        if nd.kind == NONTERM:
            nact[i] = len(nd.actions)
            for j, (a, c) in enumerate(zip(nd.actions, nd.children)):
                acts[i, j] = a
                child[i, j] = c
            offset[i] = off
            off += KS[nd.street] * len(nd.actions)
    return dict(kind=kind, street=street, actor=actor, nact=nact, child=child, acts=acts,
                c0=c0, c1=c1, folder=folder, pot=pot, tocall=tocall, offset=offset), off


# --------------------------------------------------------------------------
# archetype policies (probabilities over the node's action slots)
# --------------------------------------------------------------------------

@njit(cache=True)
def arch_policy(arch, q, facing, canraise, pot, tocall, street, out):
    # slots: facing -> [F, C, RH, RP], not facing -> [C, RH, RP]
    pf = np.zeros(4)  # by action code F, C, RH, RP
    if arch == 0:  # station
        if facing:
            if q < 0.1:
                pf[0] = 0.03
            if q > 0.93 and canraise:
                pf[3] = 0.3
        else:
            if q > 0.85 and canraise:
                pf[2] = 0.5
        pf[1] = 1.0 - pf.sum()
    elif arch == 1:  # maniac
        if canraise:
            pf[3] = 0.65
            pf[2] = 0.25
        if facing and q < 0.1:
            pf[0] = 0.05
        pf[1] = 1.0 - pf.sum()
    elif arch == 2:  # nit / over-folder
        if facing:
            if q < 0.55:
                pf[0] = 1.0
            elif q < 0.88 or not canraise:
                pf[1] = 1.0
            else:
                pf[3] = 0.6
                pf[1] = 0.4
        else:
            if q > 0.8 and canraise:
                pf[3] = 0.7
            pf[1] = 1.0 - pf.sum()
    elif arch == 3:  # uniform random
        pf[1] = 1.0
        if facing:
            pf[0] = 1.0
        if canraise:
            pf[2] = 1.0
            pf[3] = 1.0
        pf /= pf.sum()
    elif arch == 4:  # always half pot
        if canraise:
            pf[2] = 1.0
        else:
            pf[1] = 1.0
    elif arch == 5:  # always pot
        if canraise:
            pf[3] = 1.0
        else:
            pf[1] = 1.0
    else:  # honest: bets strength, calls with "pot odds" on percentile
        if facing:
            need = tocall / (pot + tocall)
            if street == 0:
                need = need * 0.6
            if q > 0.88 and canraise:
                pf[3] = 0.8
                pf[1] = 0.2
            elif q > need + 0.25:
                pf[1] = 1.0
            else:
                pf[0] = 1.0
        else:
            if q > 0.75 and canraise:
                pf[3] = 1.0
            elif q > 0.6 and canraise:
                pf[2] = 1.0
            else:
                pf[1] = 1.0
    if facing:
        out[0], out[1], out[2], out[3] = pf[0], pf[1], pf[2], pf[3]
    else:
        out[0], out[1], out[2], out[3] = pf[1], pf[2], pf[3], 0.0


@njit(cache=True)
def regret_strategy(R, tab, idx, na, out):
    s = 0.0
    for a in range(na):
        v = R[tab, idx + a]
        v = v if v > 0 else 0.0
        out[a] = v
        s += v
    if s > 0:
        for a in range(na):
            out[a] /= s
    else:
        for a in range(na):
            out[a] = 1.0 / na


@njit(cache=True)
def sample(p, na):
    r = np.random.random()
    c = 0.0
    for a in range(na):
        c += p[a]
        if r < c:
            return a
    return na - 1


@njit(cache=True)
def traverse(n, p, d, tabs, arch, R, S, kind, street, actor, nact, child, acts, c0, c1, folder,
             pot, tocall, offset, bucket, res, qmid):
    k = kind[n]
    if k != 0:
        if k == 1:
            u0 = -c0[n] if folder[n] == 0 else c1[n]
        else:
            u0 = res[d] * c0[n]
        return u0 if p == 0 else -u0
    a_ = actor[n]
    st = street[n]
    b = bucket[d, a_, st]
    na = nact[n]
    idx = offset[n] + b * na
    sigma = np.empty(4)
    tab = tabs[a_]
    if tab == 2:
        arch_policy(arch, qmid[st, b], tocall[n] > 0, na == (4 if tocall[n] > 0 else 3),
                    pot[n], tocall[n], st, sigma)
    else:
        regret_strategy(R, tab, idx, na, sigma)
    if a_ == p:
        vals = np.empty(4)
        v = 0.0
        for a in range(na):
            vals[a] = traverse(child[n, a], p, d, tabs, arch, R, S, kind, street, actor, nact, child,
                               acts, c0, c1, folder, pot, tocall, offset, bucket, res, qmid)
            v += sigma[a] * vals[a]
        for a in range(na):
            R[tab, idx + a] += vals[a] - v
        return v
    if tab != 2:
        for a in range(na):
            S[tab, idx + a] += sigma[a]
    a = sample(sigma, na)
    return traverse(child[n, a], p, d, tabs, arch, R, S, kind, street, actor, nact, child,
                    acts, c0, c1, folder, pot, tocall, offset, bucket, res, qmid)


@njit(cache=True)
def run_chunk(iters, seed, mode, arch, plock, R, S, kind, street, actor, nact, child, acts, c0, c1,
              folder, pot, tocall, offset, bucket, res, qmid):
    """mode 0: self-play equilibrium (one table). mode 1: restricted Nash response."""
    np.random.seed(seed)
    nd = bucket.shape[0]
    tabs = np.zeros(2, np.int64)
    for _ in range(iters):
        d = np.random.randint(nd)
        if mode == 0:
            tabs[0] = 0
            tabs[1] = 0
            for p in range(2):
                traverse(0, p, d, tabs, arch, R, S, kind, street, actor, nact, child, acts, c0, c1,
                         folder, pot, tocall, offset, bucket, res, qmid)
        else:
            h = np.random.randint(2)
            locked = np.random.random() < plock
            tabs[h] = 0
            tabs[1 - h] = 2 if locked else 1
            traverse(0, h, d, tabs, arch, R, S, kind, street, actor, nact, child, acts, c0, c1,
                     folder, pot, tocall, offset, bucket, res, qmid)
            if not locked:
                traverse(0, 1 - h, d, tabs, arch, R, S, kind, street, actor, nact, child, acts, c0,
                         c1, folder, pot, tocall, offset, bucket, res, qmid)


@njit(cache=True)
def simulate(games, seed, hero_strat, vill_strat, varch, kind, street, actor, nact, child, acts,
             c0, c1, folder, pot, tocall, offset, bucket, res, qmid, counts):
    """Play hero (avg strategy) vs villain (avg strategy or archetype if varch>=0).
    Adds villain action counts per node into counts; returns hero mean chips/hand."""
    np.random.seed(seed)
    nd = bucket.shape[0]
    sigma = np.empty(4)
    total = 0.0
    for _ in range(games):
        d = np.random.randint(nd)
        h = np.random.randint(2)
        n = 0
        while kind[n] == 0:
            a_ = actor[n]
            st = street[n]
            b = bucket[d, a_, st]
            na = nact[n]
            idx = offset[n] + b * na
            if a_ != h and varch >= 0:
                arch_policy(varch, qmid[st, b], tocall[n] > 0, na == (4 if tocall[n] > 0 else 3),
                            pot[n], tocall[n], st, sigma)
            else:
                src = hero_strat if a_ == h else vill_strat
                for a in range(na):
                    sigma[a] = src[idx + a]
            a = sample(sigma, na)
            if a_ != h:
                counts[n, acts[n, a]] += 1
            n = child[n, a]
        if kind[n] == 1:
            u0 = -c0[n] if folder[n] == 0 else c1[n]
        else:
            u0 = res[d] * c0[n]
        total += u0 if h == 0 else -u0
    return total / games


def normalize(S, flat, size):
    """Turn strategy sums into probabilities per infoset (uniform if unvisited)."""
    out = np.zeros(size, np.float64)
    vis = np.zeros(size // 1, np.float32)
    nact, offset, kind = flat["nact"], flat["offset"], flat["kind"]
    for n in np.flatnonzero(kind == NONTERM):
        na = int(nact[n])
        K = int(KS[flat["street"][n]])
        blk = S[offset[n]:offset[n] + K * na].reshape(K, na)
        tot = blk.sum(axis=1, keepdims=True)
        p = np.where(tot > 0, blk / np.maximum(tot, 1e-12), 1.0 / na)
        out[offset[n]:offset[n] + K * na] = p.ravel()
        vis[offset[n]:offset[n] + K * na] = np.repeat(tot.ravel(), na)
    return out, vis


def load_buckets():
    """Per-deal buckets (cached in offline/buckets.npz to keep memory low)."""
    cache = os.path.join(HERE, "buckets.npz")
    if not os.path.exists(cache):
        deals = np.load(os.path.join(HERE, "deals.npz"))
        pf = np.load(os.path.join(DATA, "preflop_eq.npy"))
        from gto.fasteval import _COMB
        holes = deals["holes"]
        idx = (_COMB[holes[..., 0], 1] + _COMB[holes[..., 1], 2] + _COMB[holes[..., 2], 3]
               + _COMB[holes[..., 3], 4] + _COMB[holes[..., 4], 5])
        del holes
        pfe = pf[idx].astype(np.float64)                 # (D, 2)
        feat = deals["feat"].astype(np.float64)           # (D, 2, 5)
        edges, qmid = B.fit(pfe.ravel(), feat.reshape(-1, 5))
        bucket = np.stack([B.assign(edges, 0, pfe),
                           B.assign(edges, 1, feat[..., 0], feat[..., 1]),
                           B.assign(edges, 2, feat[..., 2], feat[..., 3]),
                           B.assign(edges, 3, feat[..., 4])], axis=2).astype(np.int8)
        np.savez(cache, bucket=bucket, res=deals["res"], qmid=qmid, **B.pack(edges))
    c = np.load(cache)
    return c["bucket"], c["res"].astype(np.float64), B.unpack(c), c["qmid"]


def train(mode, arch, plock, iters, flat, size, bucket, res, seed, qmid):
    R = np.zeros((2, size))
    S = np.zeros((2, size))
    chunk = max(1, iters // 40)
    args = [flat[k] for k in ("kind", "street", "actor", "nact", "child", "acts", "c0", "c1",
                              "folder", "pot", "tocall", "offset")]
    t = time.time()
    done = 0
    k = 0
    while done < iters:
        m = min(chunk, iters - done)
        run_chunk(m, seed * 1000 + k, mode, arch, plock, R, S, *args, bucket, res, qmid)
        done += m
        k += 1
        R *= k / (k + 1.0)  # linear CFR discounting
        S *= k / (k + 1.0)
    print(f"  {iters} iters in {time.time() - t:.0f}s")
    return R, S


PARTS = os.path.join(HERE, "parts")


EQ_SEEDS = int(os.environ.get("EQ_SEEDS", "6"))


def main():
    """python -m gto.offline.train eq<k>|<arch>   trains one model into offline/parts/
    python -m gto.offline.train merge          builds gto/data/strategy.npz"""
    tree = Tree()
    flat, size = flatten(tree)
    bucket, res, edges, qmid = load_buckets()
    iters = int(os.environ.get("ITERS", "3000000"))
    os.makedirs(PARTS, exist_ok=True)
    name = sys.argv[1]
    if name != "merge":
        if name.startswith("eq"):
            R, S = train(0, 0, 0.0, iters, flat, size, bucket, res, 1 + int(name[2:] or 0), qmid)
        else:
            ai = ARCHETYPES.index(name)
            R, S = train(1, ai, P_LOCK[name], iters, flat, size, bucket, res, 100 + ai, qmid)
        st, vis = normalize(S[0], flat, size)
        np.savez(os.path.join(PARTS, f"{name}.npz"), strat=st.astype(np.float32),
                 visits=vis.astype(np.float32))
        print("trained", name)
        return
    args = [flat[k] for k in ("kind", "street", "actor", "nact", "child", "acts", "c0", "c1",
                              "folder", "pot", "tocall", "offset")]
    out_path = os.path.join(DATA, "strategy.npz")
    store = {"qmid": qmid, "models": np.array(["eq"] + ARCHETYPES), **B.pack(edges)}
    # equilibrium: visit-weighted average of the independent seeds
    num = np.zeros(size)
    den = np.zeros(size)
    for k in range(EQ_SEEDS):
        part = np.load(os.path.join(PARTS, f"eq{k}.npz"))
        num += part["strat"] * part["visits"]
        den += part["visits"]
    eq = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    # unvisited infosets: uniform (normalize per node block)
    for n in np.flatnonzero(flat["kind"] == NONTERM):
        na = int(flat["nact"][n])
        K = int(KS[flat["street"][n]])
        o = flat["offset"][n]
        blk = eq[o:o + K * na].reshape(K, na)
        tot = blk.sum(axis=1, keepdims=True)
        eq[o:o + K * na] = np.where(tot > 0, blk / np.maximum(tot, 1e-12), 1.0 / na).ravel()
    store["strat_eq"] = eq.astype(np.float16)
    store["visits_eq"] = np.minimum(den, 60000).astype(np.float16)
    for nm in ARCHETYPES:
        part = np.load(os.path.join(PARTS, f"{nm}.npz"))
        store[f"strat_{nm}"] = part["strat"].astype(np.float16)
        store[f"visits_{nm}"] = np.minimum(part["visits"], 60000).astype(np.float16)
    eq = store["strat_eq"].astype(np.float64)
    # opponent action marginals (villain model vs our eq hero) + evaluation
    nnodes = len(tree.nodes)
    for mi, name in enumerate(["eq"] + ARCHETYPES):
        counts = np.zeros((nnodes, 4))
        varch = -1 if name == "eq" else mi - 1
        ev_eq = simulate(400000, 7, eq, eq, varch, *args, bucket, res, qmid, counts)
        store[f"marg_{name}"] = counts.astype(np.float32)
        line = f"  vs {name:8s}: eq-core {ev_eq / 2:+.3f} bb/hand"
        if name != "eq":
            ex = store[f"strat_{name}"].astype(np.float64)
            ev_ex = simulate(400000, 7, ex, eq, varch, *args, bucket, res, qmid, np.zeros((nnodes, 4)))
            ev_ex_vs_eq = simulate(400000, 7, ex, eq, -1, *args, bucket, res, qmid, np.zeros((nnodes, 4)))
            line += f" | exploit {ev_ex / 2:+.3f} bb/hand | exploit vs eq {ev_ex_vs_eq / 2:+.3f}"
        print(line)
    np.savez_compressed(out_path, **store)
    print("saved", out_path)


if __name__ == "__main__":
    main()
