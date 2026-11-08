"""
tree.py - the public betting tree of heads-up Short Deck PLO5.

The game has only four actions, a raise cap and 40bb stacks, so the whole
betting tree (independent of cards) is small enough to enumerate. Its
rules mirror game.Game exactly, so any real action history maps to one node.

Seats: seat 0 is the BUTTON (small blind, acts first preflop), seat 1 is
the big blind (acts first postflop).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FOLD, CALL, RHALF, RPOT = 0, 1, 2, 3
SB, BB, STACK, CAP = 1, 2, 80, 4

# terminal kinds
NONTERM, FOLDED, SHOWDOWN = 0, 1, 2


@dataclass
class Node:
    id: int
    kind: int
    street: int = 0
    actor: int = -1
    actions: list = field(default_factory=list)   # legal action codes, ascending
    children: list = field(default_factory=list)  # node ids, aligned with actions
    # terminal payoff info
    folder: int = -1
    contrib: tuple = (0, 0)
    # state needed for features / exploitation
    pot: int = 0
    to_call: int = 0
    raises: int = 0


class _State:
    __slots__ = ("street", "to_act", "contrib", "bets", "stacks", "acted", "raises", "last_inc")

    def copy(self):
        s = _State()
        s.street, s.to_act, s.raises, s.last_inc = self.street, self.to_act, self.raises, self.last_inc
        s.contrib, s.bets, s.stacks, s.acted = list(self.contrib), list(self.bets), list(self.stacks), list(self.acted)
        return s


def _root() -> _State:
    s = _State()
    s.street, s.to_act, s.raises, s.last_inc = 0, 0, 0, BB
    s.contrib, s.bets, s.stacks, s.acted = [SB, BB], [SB, BB], [STACK - SB, STACK - BB], [False, False]
    return s


def _to_call(s, p):
    return max(0, s.bets[1 - p] - s.bets[p])


def _legal(s, p):
    acts = []
    if _to_call(s, p) > 0:
        acts.append(FOLD)
    acts.append(CALL)
    if s.raises < CAP and s.stacks[p] > _to_call(s, p) and s.stacks[1 - p] > 0:
        acts += [RHALF, RPOT]
    return acts


def _raise_to(s, p, a):
    my_bet, opp_bet = s.bets[p], s.bets[1 - p]
    tc = _to_call(s, p)
    pot = s.contrib[0] + s.contrib[1]
    all_in_to = my_bet + s.stacks[p]
    pot_to = opp_bet + pot + tc
    half_to = opp_bet + (pot + tc) // 2
    min_to = opp_bet + max(s.last_inc, BB)
    target = half_to if a == RHALF else pot_to
    return min(max(target, min_to), pot_to, all_in_to)


def _put(s, p, amt):
    s.stacks[p] -= amt
    s.bets[p] += amt
    s.contrib[p] += amt


class Tree:
    def __init__(self):
        self.nodes: list[Node] = []
        self.root = self._build(_root())

    def _new(self, **kw) -> Node:
        n = Node(id=len(self.nodes), **kw)
        self.nodes.append(n)
        return n

    def _build(self, s: _State) -> int:
        p = s.to_act
        node = self._new(kind=NONTERM, street=s.street, actor=p,
                         pot=s.contrib[0] + s.contrib[1], to_call=_to_call(s, p), raises=s.raises)
        node.actions = _legal(s, p)
        for a in node.actions:
            node.children.append(self._apply(s, p, a))
        return node.id

    def _apply(self, s: _State, p: int, a: int) -> int:
        if a == FOLD:
            t = self._new(kind=FOLDED, street=s.street, folder=p, contrib=tuple(s.contrib),
                          pot=s.contrib[0] + s.contrib[1])
            return t.id
        s = s.copy()
        o = 1 - p
        if a == CALL:
            _put(s, p, min(_to_call(s, p), s.stacks[p]))
        else:
            new_total = _raise_to(s, p, a)
            amt = new_total - s.bets[p]
            s.last_inc = new_total - s.bets[o]
            s.raises += 1
            s.acted[o] = False
            _put(s, p, amt)
        s.acted[p] = True
        if not (s.acted[0] and s.acted[1] and s.bets[0] == s.bets[1]):
            s.to_act = o
            return self._build(s)
        if s.street == 3 or 0 in s.stacks:
            t = self._new(kind=SHOWDOWN, street=s.street, contrib=tuple(s.contrib),
                          pot=s.contrib[0] + s.contrib[1])
            return t.id
        s.street += 1
        s.bets = [0, 0]
        s.acted = [False, False]
        s.raises = 0
        s.last_inc = BB
        s.to_act = 1  # non-button acts first postflop
        return self._build(s)

    def walk(self, history) -> int:
        """Follow a list of action codes from the root; returns the node id."""
        n = self.root
        for a in history:
            node = self.nodes[n]
            n = node.children[node.actions.index(a)]
        return n


if __name__ == "__main__":
    t = Tree()
    kinds = [n.kind for n in t.nodes]
    print("nodes", len(t.nodes), "decision", kinds.count(NONTERM),
          "fold", kinds.count(FOLDED), "showdown", kinds.count(SHOWDOWN))
    for st in range(4):
        print("street", st, sum(1 for n in t.nodes if n.kind == NONTERM and n.street == st))
