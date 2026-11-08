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
AGENT_NAME = "bot_base"


# ===== YOUR CODE HERE =====
#
# Strategy: estimate equity by vectorized Monte Carlo (numpy), compare it with
# the price of continuing (pot odds), and adapt to how the opponent behaves.
#   * preflop / flop / turn / river all use the same simulator
#   * a time guard spends the 8 s bank evenly over the whole half
#   * every decision is wrapped so a bug can never cost an error

import random
import time
from itertools import combinations

import numpy as np

from evaluator import get_table

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# A half has 1000 hands, so 1000 showdowns is the most we could ever store:
# this cap is effectively "keep everything" and memory stays tiny (<1 MB).
MAX_SHOWDOWNS_KEPT = 1000

# ---- time budget (seconds of compute spent inside choose_action per half) ----
# The judge's bank is 8 s and also charges HTTP overhead (~0.3 ms per request),
# so we plan for 4.0 s and never go past the hard cap of 5.5 s.
BUDGET_SOFT = 4.0
BUDGET_HARD = 5.5
MIN_SAMPLES = 30
MAX_SAMPLES = 500

# ---- fast lookup: sorted 36-bit masks -> scores (built once at startup) ----
_table = get_table()
_MASKS = np.fromiter(_table.keys(), dtype=np.int64, count=len(_table))
_SCORES = np.fromiter(_table.values(), dtype=np.int32, count=len(_table))
_order = np.argsort(_MASKS)
_MASKS, _SCORES = _MASKS[_order], _SCORES[_order]
del _order

_ALL = np.arange(36, dtype=np.int64)
_ONE = np.int64(1)
_PI = np.array([i for i, j in combinations(range(5), 2)])
_PJ = np.array([j for i, j in combinations(range(5), 2)])
_TA = np.array([i for i, j, k in combinations(range(5), 3)])
_TB = np.array([j for i, j, k in combinations(range(5), 3)])
_TC = np.array([k for i, j, k in combinations(range(5), 3)])


def estimate_equity(hole, board, n, rng) -> float:
    """Win probability (ties count half) against a uniformly random opponent
    hand, averaged over n random run-outs of the missing board cards."""
    known = list(hole) + list(board)
    keep = np.ones(36, dtype=bool)
    keep[known] = False
    pool = _ALL[keep]
    need = 10 - len(board)                      # 5 opponent cards + missing board
    r = rng.random((n, pool.size))
    cards = pool[np.argpartition(r, need - 1, axis=1)[:, :need]]
    opp = cards[:, :5]
    if len(board):
        known_board = np.broadcast_to(np.array(board, dtype=np.int64), (n, len(board)))
        full_board = np.concatenate([known_board, cards[:, 5:]], axis=1)
    else:
        full_board = cards[:, 5:]
    opp_bits = _ONE << opp
    board_bits = _ONE << full_board
    opp_pairs = opp_bits[:, _PI] | opp_bits[:, _PJ]                       # (n, 10)
    triples = board_bits[:, _TA] | board_bits[:, _TB] | board_bits[:, _TC]  # (n, 10)
    my_bits = _ONE << np.array(hole, dtype=np.int64)
    my_pairs = my_bits[_PI] | my_bits[_PJ]                                # (10,)
    mine = (my_pairs[None, :, None] | triples[:, None, :]).reshape(n, 100)
    theirs = (opp_pairs[:, :, None] | triples[:, None, :]).reshape(n, 100)
    scores = _SCORES[np.searchsorted(_MASKS, np.concatenate([mine, theirs], axis=0))]
    best = scores.max(axis=1)
    a, b = best[:n], best[n:]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / n)


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    Initialize EVERY key here: on_hand_end crashing with a KeyError counts
    as an error, and 5 errors forfeits the half.
    """
    memory["rng"] = random.Random(12345)             # for mixed-strategy coin flips
    memory["np_rng"] = np.random.default_rng(12345)  # for the simulator
    memory["num_hands"] = int(config.num_hands)
    memory["hands_played"] = 0
    memory["decisions"] = 0
    memory["t_used"] = 0.0                           # compute spent in choose_action
    memory["cost_ps"] = 8e-6                         # seconds per simulated sample (EMA)
    memory["cur_hand"] = -1
    memory["seen"] = 0
    memory["my_raises"] = 0                          # raises we made (for fold-rate stats)
    memory["opp"] = {
        "raises": 0,                                 # opponent raise actions observed
        "calls": 0,                                  # opponent check/call actions observed
        "folds_seen": 0,                             # hands the opponent folded to us
        "showdowns": 0,
        "showdown_hands": [],                        # (opp_hole, board), capped
    }


def _raise_rate(memory: dict) -> float:
    """Smoothed share of opponent actions that are raises (prior 30%)."""
    o = memory["opp"]
    return (o["raises"] + 1.5) / (o["raises"] + o["calls"] + 5)


def _fold_rate(memory: dict) -> float:
    """Smoothed chance the opponent folds when we raise (prior 30%)."""
    return (memory["opp"]["folds_seen"] + 2) / (memory["my_raises"] + 6.5)


def _update_from_history(obs: Observation, memory: dict) -> None:
    """Count each action exactly once, even though the full history is
    re-sent on every decision within a hand."""
    if obs.hand_index != memory["cur_hand"]:
        memory["cur_hand"] = obs.hand_index
        memory["seen"] = 0
    opp = memory["opp"]
    for a in obs.action_history[memory["seen"]:]:
        if a.is_me:
            if a.action in RAISES:
                memory["my_raises"] += 1
        elif a.action in RAISES:
            opp["raises"] += 1
        elif a.action == Action.CHECK_CALL:
            opp["calls"] += 1
    memory["seen"] = len(obs.action_history)


def _sample_budget(memory: dict) -> int:
    """How many simulation samples we can afford right now."""
    if memory["t_used"] >= BUDGET_HARD:
        return 0
    hands_left = max(1, memory["num_hands"] - memory["hands_played"])
    played = memory["hands_played"]
    per_hand = max(2.0, memory["decisions"] / played) if played >= 20 else 3.0
    per_decision = max(0.0, BUDGET_SOFT - memory["t_used"]) / (hands_left * per_hand)
    n = int(per_decision / memory["cost_ps"])
    return max(MIN_SAMPLES, min(MAX_SAMPLES, n))


def _pick_raise(obs: Observation, big: bool) -> Action:
    """Prefer the requested raise size, fall back to the other, then to call."""
    legal = obs.legal_actions
    order = (Action.RAISE_POT, Action.RAISE_HALF) if big else (Action.RAISE_HALF, Action.RAISE_POT)
    for a in order:
        if a in legal:
            return a
    return Action.CHECK_CALL


def _decide(obs: Observation, memory: dict, eq: float) -> Action:
    """Turn an equity estimate into an action (equity vs. price of continuing)."""
    legal = obs.legal_actions
    can_raise = Action.RAISE_HALF in legal or Action.RAISE_POT in legal
    preflop = obs.street == Street.PREFLOP
    rr = _raise_rate(memory)
    fr = _fold_rate(memory)
    rng = memory["rng"]

    # Preflop equities are squeezed into ~0.38-0.66 (vs. a random hand), so
    # the raise thresholds are lower there. Values come from calibrating
    # 400 random hands: 0.585 is ~p92, 0.545 is ~p78.
    pot_thr, half_thr = (0.585, 0.545) if preflop else (0.70, 0.58)

    # Against a calling station (rarely folds) value-bet a little thinner.
    if memory["my_raises"] >= 15 and fr < 0.15:
        pot_thr -= 0.02
        half_thr -= 0.03

    # ---------- nothing to call: check, bet for value, or bluff ----------
    if obs.to_call == 0:
        if not can_raise:
            return Action.CHECK_CALL
        if eq >= pot_thr:
            return _pick_raise(obs, big=True)
        if eq >= half_thr:
            return _pick_raise(obs, big=False)
        # Semi-bluff only on flop/turn and only against opponents that fold.
        bluff_freq = min(0.30, max(0.0, (fr - 0.25) * 0.8))
        if (not preflop and obs.street != Street.RIVER
                and 0.30 <= eq < half_thr and rng.random() < bluff_freq):
            return _pick_raise(obs, big=False)
        return Action.CHECK_CALL

    # ---------- facing a bet: equity against the price ----------
    price = obs.to_call / (obs.pot + obs.to_call)

    # A tight opponent's raise means more than a random hand: shrink our equity.
    facing_raise = any((not a.is_me) and a.action in RAISES and a.street == obs.street
                       for a in obs.action_history)
    eq_eff = eq - (0.25 * max(0.0, 0.55 - rr) if facing_raise else 0.0)

    committed = obs.to_call >= obs.my_stack            # calling puts us all-in
    margin = 0.0 if committed else 0.025 + 0.015 * obs.raises_this_street

    if can_raise and eq_eff >= pot_thr:
        return _pick_raise(obs, big=True)
    if can_raise and eq_eff >= half_thr and eq_eff > price + 0.10:
        return _pick_raise(obs, big=False)
    if eq_eff > price + margin:
        return Action.CHECK_CALL
    return Action.FOLD


def _no_sim_fallback(obs: Observation) -> Action:
    """Used only when the time budget is exhausted: cheap and safe."""
    if obs.to_call == 0:
        return Action.CHECK_CALL
    price = obs.to_call / (obs.pot + obs.to_call)
    return Action.CHECK_CALL if price <= 0.2 else Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    Any bug falls back to a legal safe action instead of costing an error.
    """
    start = time.perf_counter()
    try:
        _update_from_history(obs, memory)
        memory["decisions"] += 1
        legal = obs.legal_actions
        if len(legal) == 1:                               # nothing to decide
            return legal[0]
        n = _sample_budget(memory)
        if n == 0:
            action = _no_sim_fallback(obs)
        else:
            t0 = time.perf_counter()
            eq = estimate_equity(obs.hole_cards, obs.board, n, memory["np_rng"])
            cost = (time.perf_counter() - t0) / n
            memory["cost_ps"] = 0.9 * memory["cost_ps"] + 0.1 * cost
            action = _decide(obs, memory, eq)
        if action in legal:
            return action
    except Exception:
        traceback.print_exc()
    finally:
        memory["t_used"] += time.perf_counter() - start
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Guarded so it can never crash."""
    try:
        memory["hands_played"] += 1
        opp = memory["opp"]
        # A fold never shows up in action_history (we are not asked again),
        # so the only place to learn about it is here.
        if not result.went_to_showdown and result.winner == "me":
            opp["folds_seen"] += 1
        if result.opp_hole is not None:
            opp["showdowns"] += 1
            opp["showdown_hands"].append((result.opp_hole, result.board))
            if len(opp["showdown_hands"]) > MAX_SHOWDOWNS_KEPT:
                del opp["showdown_hands"][0]
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