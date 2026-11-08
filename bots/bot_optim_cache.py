"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Fish Dragon"
AGENT_NAME = "bot_optim_cache"


# ===== YOUR CODE HERE =====

from itertools import combinations
import random
import time

import numpy as np

from evaluator import get_table


# =============================================================================
# Non-GTO Monte Carlo equity bot
# Base strategy + raw equity cache only
# =============================================================================

PARAMS = {
    "base_call_margin": 0.013315,
    "committed_margin_multiplier": 0.262490,
    "fallback_call_price": 0.559557,
    "flop_uncertainty_margin": 0.042052,
    "fold_rate_half_discount": 0.043377,
    "fold_rate_pot_discount": 0.018939,
    "fold_rate_reference": 0.111627,
    "half_raise_extra_edge": 0.073882,
    "loose_raise_bonus": 0.043906,
    "postflop_button_bonus": 0.015994,
    "postflop_half_threshold": 0.630354,
    "postflop_oop_penalty": 0.002320,
    "postflop_pot_threshold": 0.357228,
    "pot_raise_extra_margin": 0.043055,
    "preflop_half_threshold": 0.448523,
    "preflop_pot_threshold": 0.620579,
    "preflop_uncertainty_margin": 0.010656,
    "raise_count_margin": 0.006768,
    "randomization": 0.002588,
    "river_uncertainty_margin": 0.001409,
    "short_stack_aggression_bonus": 0.024264,
    "tight_raise_penalty": 0.410112,
    "tight_raise_reference": 0.758479,
    "turn_uncertainty_margin": 0.001584,
}

# =============================================================================
# Time/sample/cache constants
# =============================================================================

BUDGET_SOFT = 4.80
BUDGET_HARD = 6.20

MIN_SAMPLES = 64
MAX_SAMPLES_PREFLOP = 384
MAX_SAMPLES_FLOP = 768
MAX_SAMPLES_TURN = 1024
MAX_SAMPLES_RIVER = 1536

CACHE_MIN_SAMPLES = 96
MAX_CACHE_ENTRIES = 4000

MAX_SHOWDOWNS_KEPT = 60

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


# =============================================================================
# Fast vectorized evaluator table
# =============================================================================

_TABLE = get_table()
_MASKS = np.fromiter(_TABLE.keys(), dtype=np.int64, count=len(_TABLE))
_SCORES = np.fromiter(_TABLE.values(), dtype=np.int32, count=len(_TABLE))
_ORDER = np.argsort(_MASKS)
_MASKS = _MASKS[_ORDER]
_SCORES = _SCORES[_ORDER]
del _ORDER

_ALL_CARDS = np.arange(36, dtype=np.int64)
_ONE = np.int64(1)

_PAIR_I = np.array([i for i, j in combinations(range(5), 2)], dtype=np.int64)
_PAIR_J = np.array([j for i, j in combinations(range(5), 2)], dtype=np.int64)

_TRIPLE_A = np.array([i for i, j, k in combinations(range(5), 3)], dtype=np.int64)
_TRIPLE_B = np.array([j for i, j, k in combinations(range(5), 3)], dtype=np.int64)
_TRIPLE_C = np.array([k for i, j, k in combinations(range(5), 3)], dtype=np.int64)


def estimate_equity(hole, board, n, rng) -> float:
    """
    Estimate win probability against one uniformly random opponent hand.

    Ties count as half a win. Missing board cards are sampled randomly.
    """
    if n <= 0:
        return 0.5

    known = list(hole) + list(board)

    keep = np.ones(36, dtype=bool)
    keep[known] = False
    pool = _ALL_CARDS[keep]

    # 5 opponent cards + enough board cards to complete a 5-card board.
    need = 10 - len(board)

    keys = rng.random((n, pool.size))
    sampled = pool[np.argpartition(keys, need - 1, axis=1)[:, :need]]

    opp = sampled[:, :5]

    if len(board):
        fixed_board = np.broadcast_to(
            np.array(board, dtype=np.int64),
            (n, len(board)),
        )
        full_board = np.concatenate([fixed_board, sampled[:, 5:]], axis=1)
    else:
        full_board = sampled[:, 5:]

    opp_bits = _ONE << opp
    board_bits = _ONE << full_board

    opp_pairs = opp_bits[:, _PAIR_I] | opp_bits[:, _PAIR_J]

    board_triples = (
        board_bits[:, _TRIPLE_A]
        | board_bits[:, _TRIPLE_B]
        | board_bits[:, _TRIPLE_C]
    )

    my_bits = _ONE << np.array(hole, dtype=np.int64)
    my_pairs = my_bits[_PAIR_I] | my_bits[_PAIR_J]

    my_masks = (my_pairs[None, :, None] | board_triples[:, None, :]).reshape(n, 100)
    opp_masks = (opp_pairs[:, :, None] | board_triples[:, None, :]).reshape(n, 100)

    all_masks = np.concatenate([my_masks, opp_masks], axis=0)
    scores = _SCORES[np.searchsorted(_MASKS, all_masks)]

    best = scores.max(axis=1)
    my_best = best[:n]
    opp_best = best[n:]

    wins = (my_best > opp_best).sum()
    ties = (my_best == opp_best).sum()

    return float((wins + 0.5 * ties) / n)


# =============================================================================
# Lifecycle
# =============================================================================

def new_half(config: AgentConfig, memory: dict) -> None:
    """
    Called once at the start of each half-match.
    """
    memory.clear()

    memory["rng"] = random.Random(12345)
    memory["np_rng"] = np.random.default_rng(12345)

    memory["num_hands"] = int(config.num_hands)
    memory["hands_played"] = 0
    memory["decisions"] = 0

    memory["t_used"] = 0.0

    # Keep base bot's more aggressive cost estimate.
    memory["cost_ps"] = 8.0e-6

    memory["cur_hand"] = -1
    memory["seen"] = 0

    memory["my_raises"] = 0
    memory["my_raise_opportunities"] = 0

    # Raw equity cache.
    memory["eq_cache"] = {}

    memory["opp"] = {
        "raises": 0,
        "calls": 0,
        "folds_seen": 0,
        "showdowns": 0,
        "showdown_hands": [],
    }


# =============================================================================
# Opponent tracking
# =============================================================================

def _pot_price(obs: Observation) -> float:
    if obs.to_call <= 0:
        return 0.0
    return obs.to_call / max(1.0, obs.pot + obs.to_call)


def _raise_rate(memory: dict) -> float:
    """
    Smoothed opponent raise frequency.

    Prior is about 30%.
    """
    opp = memory["opp"]
    return (opp["raises"] + 1.5) / max(1.0, opp["raises"] + opp["calls"] + 5.0)


def _fold_rate(memory: dict) -> float:
    """
    Smoothed estimate of how often opponent folds after we raise.

    Prior is about 30%.
    """
    return (memory["opp"]["folds_seen"] + 2.0) / max(1.0, memory["my_raises"] + 6.5)


def _update_from_history(obs: Observation, memory: dict) -> None:
    """
    Count each action once.

    The observation sends the full action history every decision, so we track
    how many actions we have already processed this hand.
    """
    if obs.hand_index != memory["cur_hand"]:
        memory["cur_hand"] = obs.hand_index
        memory["seen"] = 0

    opp = memory["opp"]

    for a in obs.action_history[memory["seen"]:]:
        if a.is_me:
            if a.action in RAISES:
                memory["my_raises"] += 1
        else:
            if a.action in RAISES:
                opp["raises"] += 1
            elif a.action == Action.CHECK_CALL:
                opp["calls"] += 1

    memory["seen"] = len(obs.action_history)


# =============================================================================
# Sample budget and equity cache
# =============================================================================

def _street_max_samples(street: Street) -> int:
    if street == Street.PREFLOP:
        return MAX_SAMPLES_PREFLOP
    if street == Street.FLOP:
        return MAX_SAMPLES_FLOP
    if street == Street.TURN:
        return MAX_SAMPLES_TURN
    return MAX_SAMPLES_RIVER


def _sample_budget(obs: Observation, memory: dict) -> int:
    """
    Base bot sample budget.

    This intentionally keeps bot_base_optim's behavior of forcing at least
    MIN_SAMPLES when time remains.
    """
    if memory["t_used"] >= BUDGET_HARD:
        return 0

    remaining_time = BUDGET_SOFT - memory["t_used"]
    if remaining_time <= 0:
        return 0

    hands_left = max(1, memory["num_hands"] - memory["hands_played"])
    played = max(1, memory["hands_played"])

    if played >= 30:
        decisions_per_hand = memory["decisions"] / played
        decisions_per_hand = max(1.7, min(4.5, decisions_per_hand))
    else:
        decisions_per_hand = 3.0

    decisions_left = max(1.0, hands_left * decisions_per_hand)
    seconds_for_this_decision = remaining_time / decisions_left

    n = int(seconds_for_this_decision / max(1.0e-7, memory["cost_ps"]))

    # Spend a bit more on expensive call decisions.
    price = _pot_price(obs)
    if obs.to_call > 0 and price > 0.30:
        n = int(n * 1.20)

    # Spend a bit less when checking is free.
    if obs.to_call == 0:
        n = int(n * 0.85)

    return max(MIN_SAMPLES, min(_street_max_samples(obs.street), n))


def _equity_cache_key(obs: Observation):
    """
    Raw random-hand equity depends on hand/street/hole/board.

    Betting history changes how we interpret equity, but not the raw equity
    against a random hand on the same board.
    """
    return (
        int(obs.hand_index),
        int(obs.street),
        tuple(obs.hole_cards),
        tuple(obs.board),
    )


def _get_cached_equity(obs: Observation, memory: dict, n: int):
    """
    Return:
        equity, from_cache
    """
    cache = memory.setdefault("eq_cache", {})
    key = _equity_cache_key(obs)

    if key in cache:
        return cache[key], True

    eq = estimate_equity(obs.hole_cards, obs.board, n, memory["np_rng"])

    if n >= CACHE_MIN_SAMPLES:
        if len(cache) >= MAX_CACHE_ENTRIES:
            cache.clear()
        cache[key] = eq

    return eq, False


# =============================================================================
# Hand feature helpers
# =============================================================================

def _rank_counts(cards) -> dict:
    counts = {}
    for c in cards:
        r = card_rank(c)
        counts[r] = counts.get(r, 0) + 1
    return counts


def _suit_counts(cards) -> dict:
    counts = {}
    for c in cards:
        s = card_suit(c)
        counts[s] = counts.get(s, 0) + 1
    return counts


def _preflop_quality_hint(hole) -> float:
    """
    Small heuristic adjustment to stabilize preflop Monte Carlo.

    Short Deck uses ranks 6 through A, encoded as 0..8.
    """
    ranks = sorted((card_rank(c) for c in hole), reverse=True)
    rank_counts = _rank_counts(hole)
    suit_counts = _suit_counts(hole)

    bonus = 0.0

    high_count = sum(1 for r in ranks if r >= 6)
    ace_count = sum(1 for r in ranks if r == 8)

    bonus += 0.004 * high_count
    bonus += 0.004 * ace_count

    pairs = sum(1 for v in rank_counts.values() if v == 2)
    trips = sum(1 for v in rank_counts.values() if v == 3)
    quads = sum(1 for v in rank_counts.values() if v == 4)

    bonus += 0.007 * pairs
    bonus += 0.012 * trips
    bonus += 0.004 * quads

    max_suit = max(suit_counts.values())
    if max_suit >= 2:
        bonus += 0.006
    if max_suit >= 3:
        bonus += 0.004

    unique = sorted(set(ranks))
    longest = 1
    current = 1

    for a, b in zip(unique, unique[1:]):
        if b == a + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    bonus += 0.004 * max(0, longest - 2)

    low_count = sum(1 for r in ranks if r <= 2)
    bonus -= 0.004 * max(0, low_count - 2)

    return max(-0.025, min(0.035, bonus))


# =============================================================================
# Decision helpers
# =============================================================================

def _pick_raise(obs: Observation, big: bool) -> Action:
    """
    Prefer pot or half-pot, but always return a legal fallback.
    """
    order = (
        (Action.RAISE_POT, Action.RAISE_HALF)
        if big
        else (Action.RAISE_HALF, Action.RAISE_POT)
    )

    for action in order:
        if action in obs.legal_actions:
            return action

    return Action.CHECK_CALL


def _facing_opp_raise_this_street(obs: Observation) -> bool:
    return any(
        (not a.is_me) and a.street == obs.street and a.action in RAISES
        for a in obs.action_history
    )


def _facing_opp_pot_raise_this_street(obs: Observation) -> bool:
    return any(
        (not a.is_me) and a.street == obs.street and a.action == Action.RAISE_POT
        for a in obs.action_history
    )


def _street_uncertainty_margin(street: Street) -> float:
    if street == Street.PREFLOP:
        return PARAMS["preflop_uncertainty_margin"]
    if street == Street.FLOP:
        return PARAMS["flop_uncertainty_margin"]
    if street == Street.TURN:
        return PARAMS["turn_uncertainty_margin"]
    return PARAMS["river_uncertainty_margin"]


def _adjust_equity(obs: Observation, memory: dict, raw_eq: float) -> float:
    """
    Convert raw random-hand equity into decision equity.
    """
    eq = raw_eq
    rr = _raise_rate(memory)

    if obs.street == Street.PREFLOP:
        eq += _preflop_quality_hint(obs.hole_cards)

    if _facing_opp_raise_this_street(obs):
        tight_gap = max(0.0, PARAMS["tight_raise_reference"] - rr)
        loose_gap = max(0.0, rr - PARAMS["tight_raise_reference"])

        eq -= PARAMS["tight_raise_penalty"] * tight_gap
        eq += PARAMS["loose_raise_bonus"] * loose_gap

    if obs.street != Street.PREFLOP:
        if obs.is_button:
            eq += PARAMS["postflop_button_bonus"]
        else:
            eq -= PARAMS["postflop_oop_penalty"]

    effective_stack = min(obs.my_stack, obs.opp_stack)
    if effective_stack <= 18:
        eq += PARAMS["short_stack_aggression_bonus"]

    eq += memory["rng"].uniform(-PARAMS["randomization"], PARAMS["randomization"])

    return max(0.0, min(1.0, eq))


def _raise_thresholds(obs: Observation, memory: dict) -> tuple[float, float]:
    if obs.street == Street.PREFLOP:
        pot_thr = PARAMS["preflop_pot_threshold"]
        half_thr = PARAMS["preflop_half_threshold"]
    else:
        pot_thr = PARAMS["postflop_pot_threshold"]
        half_thr = PARAMS["postflop_half_threshold"]

    fr = _fold_rate(memory)
    fold_excess = max(0.0, fr - PARAMS["fold_rate_reference"])

    pot_thr -= PARAMS["fold_rate_pot_discount"] * fold_excess
    half_thr -= PARAMS["fold_rate_half_discount"] * fold_excess

    if obs.to_call == 0:
        half_thr += 0.008

    if obs.raises_this_street >= 2:
        pot_thr += 0.025
        half_thr += 0.015

    return pot_thr, half_thr


def _decide(obs: Observation, memory: dict, raw_eq: float) -> Action:
    """
    Main decision rule.
    """
    legal = obs.legal_actions

    if len(legal) == 1:
        return legal[0]

    can_raise = Action.RAISE_HALF in legal or Action.RAISE_POT in legal

    price = _pot_price(obs)
    eq_eff = _adjust_equity(obs, memory, raw_eq)

    pot_thr, half_thr = _raise_thresholds(obs, memory)

    committed = obs.to_call >= obs.my_stack

    margin = (
        PARAMS["base_call_margin"]
        + PARAMS["raise_count_margin"] * obs.raises_this_street
        + _street_uncertainty_margin(obs.street)
    )

    if _facing_opp_pot_raise_this_street(obs):
        margin += PARAMS["pot_raise_extra_margin"]

    if committed:
        margin *= PARAMS["committed_margin_multiplier"]

    if can_raise:
        memory["my_raise_opportunities"] += 1

        if eq_eff >= pot_thr:
            return _pick_raise(obs, big=True)

        if eq_eff >= half_thr and eq_eff > price + PARAMS["half_raise_extra_edge"]:
            return _pick_raise(obs, big=False)

        if (
            obs.to_call == 0
            and obs.street != Street.PREFLOP
            and obs.street != Street.RIVER
            and obs.is_button
            and _fold_rate(memory) > 0.42
            and eq_eff > 0.47
        ):
            return _pick_raise(obs, big=False)

    if eq_eff > price + margin:
        return Action.CHECK_CALL

    if obs.to_call == 0:
        return Action.CHECK_CALL

    if Action.FOLD in legal:
        return Action.FOLD

    return Action.CHECK_CALL


def _no_sim_fallback(obs: Observation) -> Action:
    """
    Cheap fallback when simulation budget is exhausted.
    """
    if obs.to_call == 0:
        return Action.CHECK_CALL

    price = _pot_price(obs)

    if obs.to_call >= obs.my_stack:
        return (
            Action.CHECK_CALL
            if price <= PARAMS["fallback_call_price"] + 0.08
            else Action.FOLD
        )

    return Action.CHECK_CALL if price <= PARAMS["fallback_call_price"] else Action.FOLD


# =============================================================================
# Main action function
# =============================================================================

def choose_action(obs: Observation, memory: dict) -> Action:
    """
    Called every time it is our turn.

    Must return one of obs.legal_actions.
    """
    start = time.perf_counter()

    try:
        _update_from_history(obs, memory)
        memory["decisions"] += 1

        legal = obs.legal_actions

        if len(legal) == 1:
            return legal[0]

        n = _sample_budget(obs, memory)

        if n <= 0:
            action = _no_sim_fallback(obs)
        else:
            t0 = time.perf_counter()

            eq, from_cache = _get_cached_equity(obs, memory, n)

            elapsed = time.perf_counter() - t0

            if not from_cache and elapsed > 0 and n > 0:
                sample_cost = elapsed / n
                memory["cost_ps"] = 0.90 * memory["cost_ps"] + 0.10 * sample_cost

            action = _decide(obs, memory, eq)

        if action in legal:
            return action

    except Exception:
        traceback.print_exc()

    finally:
        memory["t_used"] += time.perf_counter() - start

    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


# =============================================================================
# Hand-end learning
# =============================================================================

def on_hand_end(result: HandResult, memory: dict) -> None:
    """
    Called once after each finished hand.
    """
    try:
        memory["hands_played"] += 1

        opp = memory["opp"]

        if not result.went_to_showdown and result.winner == "me":
            opp["folds_seen"] += 1

        if result.opp_hole is not None:
            opp["showdowns"] += 1
            opp["showdown_hands"].append(
                (tuple(result.opp_hole), tuple(result.board))
            )

            if len(opp["showdown_hands"]) > MAX_SHOWDOWNS_KEPT:
                del opp["showdown_hands"][0]

        cache = memory.get("eq_cache")
        if isinstance(cache, dict) and len(cache) > MAX_CACHE_ENTRIES:
            cache.clear()

    except Exception:
        traceback.print_exc()


# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)