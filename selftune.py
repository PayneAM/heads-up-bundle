"""
selftune_v2.py

PARAMS tuner for agent.py: baseline-relative, noise-aware, with confirmation.
Same idea as selftune_play.py (mutate PARAMS, play duplicate matches against the
champion and fixed opponents) but fixes the things that made its promotions
unreliable:

  1. Scores are DELTAS vs the current champion on the same deals.
     (Old: absolute scores. A candidate identical to the champion scored +5.5
     because CallingStation alone gives +15, so the 0.20 threshold always passed.)
  2. Two stages: screen many candidates cheaply, then CONFIRM the top few on
     fresh seeds. Only the confirm result can promote (kills the winner's curse).
  3. Promotion needs a significance level (z) as well as a size.
  4. New seeds every generation (the champion can't overfit to a fixed card set).
  5. Self-play pool: champion + the original "anchor" + older champions, so the
     bot can't win by exploiting only the last champion, and can't drift to a
     worse place than where it started.
  6. Constraints (pot threshold >= half threshold, ...) and bounds for any param.
  7. Fewer mutated params per candidate (default 25%, was 75%) so a change
     can be attributed to something.
  8. Module cache: no 5 MB/load leak, no repeated 40 ms loads.
  9. --fixed-samples: the bot uses a fixed number of Monte Carlo samples, like
     the ~250 it gets in a real 1000-hand half (the old harness gave it 400-1500,
     so tuned values fit a much less noisy bot). Also makes runs CPU-load independent.
 10. A hold-out check at the end: final champion vs the ORIGINAL params on
     unseen seeds, with a confidence interval.

Smoke test (about a minute):
    python selftune_v2.py --agent bots/bot_base.py --mode mixed \\
        --opponents baseline-bots/baseline_always_pot.py \\
        --generations 1 --candidates 4 --deals 20 --seeds 1 --confirm-seeds 1 \\
        --finalists 2 --holdout-seeds 0 --jobs 1

Real run:
    python selftune_v2.py --agent agent.py --mode mixed \\
        --opponents bots/bot_base.py baseline-bots/baseline_always_pot.py \\
                    baseline-bots/baseline_calling_station.py \\
        --generations 15 --candidates 16 --deals 100 --seeds 2 \\
        --confirm-seeds 4 --finalists 3 --jobs 12 --anneal

Afterwards: copy the printed PARAMS into the agent, run `python -m pytest -q`,
then play real judge matches over several seeds.
"""

from __future__ import annotations

import os

# The tournament limits numpy to one thread; do the same so parallel jobs don't
# fight over cores (must be set before numpy is imported by any bot).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from game import (
    Action,
    AgentConfig,
    BIG_BLIND,
    Game,
    button_for_deal,
    default_action,
    make_deal,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# =============================================================================
# Loading agent modules (cached: one module object per file/slot)
# =============================================================================

_LOAD_COUNTER = 0
_CACHE: dict[tuple, Any] = {}
_DEFAULTS: dict[tuple, dict] = {}


def _exec_module(path: str):
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    for p in (HERE, os.path.dirname(path)):
        if p not in sys.path:
            sys.path.insert(0, p)
    name = f"loaded_agent_{os.getpid()}_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _constant(n: int):
    def fixed(*_a, **_k):
        return n
    return fixed


def get_module(path: str, slot: int, params: dict | None, fixed_samples: int, fresh: bool):
    """Return an agent module ready to play.

    slot 0/1 are the two seats of a match, so champion-vs-candidate (same file)
    get two independent modules with independent PARAMS dicts. PARAMS is reset to
    the file's defaults before `params` are applied, so nothing leaks between
    candidates. fresh=True reloads the file every time (old behaviour; use it if
    an opponent keeps module-level state).
    """
    path = abs_path(path)
    key = (path, slot, fixed_samples)
    module = None if fresh else _CACHE.get(key)

    if module is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent file not found: {path}")
        module = _exec_module(path)
        for fn in ("new_half", "choose_action", "on_hand_end"):
            if not hasattr(module, fn):
                raise RuntimeError(f"{path} is missing required function {fn}")
        if hasattr(module, "PARAMS"):
            _DEFAULTS[key] = dict(module.PARAMS)
        if fixed_samples > 0 and hasattr(module, "_sample_budget"):
            module._sample_budget = _constant(fixed_samples)
        if not fresh:
            _CACHE[key] = module
    elif key in _DEFAULTS:
        module.PARAMS.clear()
        module.PARAMS.update(_DEFAULTS[key])

    if params is not None:
        if not hasattr(module, "PARAMS"):
            raise RuntimeError(f"{path} does not expose PARAMS but params were provided")
        module.PARAMS.update(params)
    return module


def release_module(module, fresh: bool) -> None:
    if fresh:
        sys.modules.pop(module.__name__, None)


def load_base_params(agent_path: str) -> dict[str, float]:
    module = get_module(agent_path, slot=9, params=None, fixed_samples=0, fresh=False)
    if not hasattr(module, "PARAMS"):
        raise RuntimeError(f"{agent_path} does not expose PARAMS")
    params = {k: float(v) for k, v in module.PARAMS.items() if isinstance(v, (int, float))}
    if not params:
        raise RuntimeError("PARAMS has no numeric values")
    return params


# =============================================================================
# Playing games in-process
# =============================================================================

@dataclass
class BotSpec:
    path: str
    params: dict[str, float] | None
    label: str


@dataclass
class BotState:
    module: Any
    memory: dict
    label: str
    errors: int = 0
    choose_seconds: float = 0.0


def safe_choose(bot: BotState, obs) -> Action:
    start = time.perf_counter()
    try:
        action = Action(bot.module.choose_action(obs, bot.memory))
        if action not in obs.legal_actions:
            bot.errors += 1
            return default_action(obs.to_call)
        return action
    except Exception:
        bot.errors += 1
        print(f"\n[{bot.label}] choose_action crashed", flush=True)
        traceback.print_exc()
        return default_action(obs.to_call)
    finally:
        bot.choose_seconds += time.perf_counter() - start


def play_one_half(spec_a, spec_b, seed, half, num_deals, run_cfg) -> dict[str, Any]:
    """One duplicate half. half=1: A in seat 0; half=2: A in seat 1."""
    fresh, fixed = run_cfg["fresh"], run_cfg["fixed_samples"]
    config = AgentConfig(num_hands=run_cfg["cfg_hands"])
    seat_a = 0 if half == 1 else 1

    bots = []
    for spec, slot in ((spec_a, 0), (spec_b, 1)):
        module = get_module(spec.path, slot, spec.params, fixed, fresh)
        bots.append(BotState(module, {}, f"{spec.label}_half{half}"))
    bot_a, bot_b = bots

    nets_a: list[int] = []
    try:
        for bot in bots:
            try:
                bot.memory.clear()
                bot.module.new_half(config, bot.memory)
            except Exception:
                bot.errors += 1
                print(f"\n[{bot.label}] new_half crashed", flush=True)
                traceback.print_exc()

        for h in range(num_deals):
            game = Game(make_deal(seed, h), button_seat=button_for_deal(h), hand_index=h)
            while not game.is_terminal:
                seat = game.to_act
                bot = bot_a if seat == seat_a else bot_b
                obs = game.observation(seat)
                action = safe_choose(bot, obs)
                try:
                    game.apply(action)
                except Exception:
                    bot.errors += 1
                    traceback.print_exc()
                    game.apply(default_action(obs.to_call))
            for bot, seat in ((bot_a, seat_a), (bot_b, 1 - seat_a)):
                try:
                    bot.module.on_hand_end(game.hand_result(seat), bot.memory)
                except Exception:
                    bot.errors += 1
                    traceback.print_exc()
            nets_a.append(int(game.hand_result(seat_a).my_net))
    finally:
        release_module(bot_a.module, fresh)
        release_module(bot_b.module, fresh)

    return {"nets_a": nets_a, "errors_a": bot_a.errors, "errors_b": bot_b.errors,
            "secs_a": bot_a.choose_seconds}


def match_task(task):
    """One unit of parallel work: A vs B over several seeds (both halves each).
    Returns paired chips per seed from A's point of view."""
    key, spec_a, spec_b, seeds, deals, run_cfg = task
    paired: dict[str, list[int]] = {}
    err_a = err_b = 0
    secs = 0.0
    for seed in seeds:
        h1 = play_one_half(spec_a, spec_b, seed, 1, deals, run_cfg)
        h2 = play_one_half(spec_a, spec_b, seed, 2, deals, run_cfg)
        paired[seed] = [a + b for a, b in zip(h1["nets_a"], h2["nets_a"])]
        err_a += h1["errors_a"] + h2["errors_a"]
        err_b += h1["errors_b"] + h2["errors_b"]
        secs += h1["secs_a"] + h2["secs_a"]
    return key, {"paired": paired, "errors_a": err_a, "errors_b": err_b, "secs_a": secs}


def run_tasks(pool, tasks) -> dict:
    results = {}
    if pool is None:
        for t in tasks:
            k, r = match_task(t)
            results[k] = r
    else:
        futures = [pool.submit(match_task, t) for t in tasks]
        for f in as_completed(futures):
            k, r = f.result()
            results[k] = r
    return results


# =============================================================================
# Bounds, constraints, mutation
# =============================================================================

_KNOWN_BOUNDS: dict[str, tuple[float, float]] = {}
for _n in ("preflop_half_threshold", "preflop_pot_threshold",
           "postflop_half_threshold", "postflop_pot_threshold",
           "pre_pot_thr", "pre_half_thr", "post_pot_thr", "post_half_thr"):
    _KNOWN_BOUNDS[_n] = (0.20, 0.95)
for _n in ("base_call_margin", "raise_count_margin", "pot_raise_extra_margin",
           "preflop_uncertainty_margin", "flop_uncertainty_margin",
           "turn_uncertainty_margin", "river_uncertainty_margin", "half_raise_extra_edge",
           "call_margin", "margin_per_raise", "raise_over_price"):
    _KNOWN_BOUNDS[_n] = (-0.08, 0.30)
for _n in ("fold_rate_half_discount", "fold_rate_pot_discount", "loose_raise_bonus",
           "postflop_button_bonus", "postflop_oop_penalty", "short_stack_aggression_bonus",
           "station_pot_adj", "station_half_adj", "oop_disc"):
    _KNOWN_BOUNDS[_n] = (-0.15, 0.35)
_KNOWN_BOUNDS.update({
    "committed_margin_multiplier": (0.0, 1.0),
    "fallback_call_price": (0.0, 0.70),
    "fold_rate_reference": (0.0, 1.0),
    "tight_raise_reference": (0.0, 1.0),
    "tight_raise_penalty": (0.0, 0.90),
    "shrink": (0.0, 0.90),
    "short_stack_threshold": (2.0, 80.0),
    "randomization": (0.0, 0.04),
    "bluff_scale": (0.0, 3.0),
    "bluff_min_eq": (0.0, 0.6),
})

# (hi, lo, gap): require params[hi] >= params[lo] + gap
DEFAULT_CONSTRAINTS = [
    ("preflop_pot_threshold", "preflop_half_threshold", 0.02),
    ("postflop_pot_threshold", "postflop_half_threshold", 0.02),
    ("pre_pot_thr", "pre_half_thr", 0.02),
    ("post_pot_thr", "post_half_thr", 0.02),
]


def clamp_param(name: str, value: float, ref: float, overrides: dict) -> float:
    v = float(value)
    if name in overrides:
        lo, hi = overrides[name]
    elif name in _KNOWN_BOUNDS:
        lo, hi = _KNOWN_BOUNDS[name]
    else:  # unknown parameter: stay within [-ref, 3*ref] of the starting value
        r = abs(ref) if abs(ref) > 1e-9 else 0.05
        lo, hi = ref - 2 * r, ref + 2 * r
    return max(lo, min(hi, v))


def satisfied(p: dict, c) -> bool:
    hi, lo, gap = c
    return hi not in p or lo not in p or p[hi] >= p[lo] + gap - 1e-12


def repair(p: dict, constraints, ref: dict, overrides: dict) -> dict:
    for hi, lo, gap in constraints:
        if hi in p and lo in p and p[hi] < p[lo] + gap:
            p[hi] = clamp_param(hi, p[lo] + gap, ref.get(hi, p[hi]), overrides)
    return p


def mutate(parent, scale, rng, rate, ref, overrides, constraints):
    child = dict(parent)
    keys = [k for k, v in child.items() if isinstance(v, (int, float))]
    chosen = [k for k in keys if rng.random() <= rate] or [rng.choice(keys)]
    for k in chosen:
        v = float(child[k])
        new = v * rng.uniform(1 - scale, 1 + scale) if abs(v) > 1e-9 else rng.uniform(-scale, scale) * 0.02
        child[k] = clamp_param(k, new, ref.get(k, v), overrides)
    return repair(child, constraints, ref, overrides)


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) if k in a and k in b else a.get(k, b.get(k))
            for k in sorted(set(a) | set(b))}


def make_candidates(champion, hof, args, rng, ref, overrides, constraints):
    out = [mutate(champion, args.scale, rng, args.mutation_rate, ref, overrides, constraints)
           for _ in range(args.candidates)]
    if len(hof) >= 2 and args.candidates >= 4:
        for _ in range(max(1, args.candidates // 4)):
            child = crossover(rng.choice(hof), rng.choice(hof), rng)
            out.append(mutate(child, args.scale * 0.75, rng, args.mutation_rate,
                              ref, overrides, constraints))
    return out


# =============================================================================
# Evaluation: deltas vs the champion on identical deals
# =============================================================================

def params_key(p: dict) -> tuple:
    return tuple(sorted(p.items()))


def build_targets(agent_path, champion, anchor, hof, fixed_paths, args, rng, self_name="champion"):
    """Opponents a candidate is measured against."""
    targets = []
    if args.mode in ("self", "mixed"):
        members = [(self_name, champion)]
        seen = {params_key(champion)}
        extra = [("anchor", anchor)] + [("hof", h) for h in rng.sample(hof, len(hof))]
        for name, p in extra:
            if len(members) >= args.pool_size:
                break
            if params_key(p) in seen:
                continue
            seen.add(params_key(p))
            members.append((name, p))
        for i, (name, p) in enumerate(members):
            label = name if i == 0 else f"{name}{i}"
            targets.append({
                "name": label,
                "spec": BotSpec(agent_path, p, label),
                "weight": args.self_weight / len(members),
                "zero_base": params_key(p) == params_key(champion),
            })
    if args.mode in ("fixed", "mixed"):
        for path in fixed_paths:
            label = os.path.basename(path)
            targets.append({"name": label, "spec": BotSpec(path, None, label),
                            "weight": args.fixed_weight, "zero_base": False})
    return targets


def evaluate(pool, args, cands, base_params, targets, seeds) -> list[dict]:
    """Score each candidate as its improvement over `base_params` (bb/deal),
    measured per deal against every target on the same cards."""
    agent_path = abs_path(args.agent)
    run_cfg = {"fresh": args.fresh_modules, "fixed_samples": args.fixed_samples,
               "cfg_hands": args.cfg_hands}
    base_spec = BotSpec(agent_path, base_params, "base")
    tasks = []
    for t in targets:
        if not t["zero_base"]:
            tasks.append((("base", t["name"]), base_spec, t["spec"], seeds, args.deals, run_cfg))
    for ci, cp in enumerate(cands):
        cspec = BotSpec(agent_path, cp, f"cand{ci}")
        for t in targets:
            tasks.append((("cand", ci, t["name"]), cspec, t["spec"], seeds, args.deals, run_cfg))
    results = run_tasks(pool, tasks)

    out = []
    total_w = sum(t["weight"] for t in targets)
    if total_w <= 0:
        raise RuntimeError("all target weights are zero")
    for ci in range(len(cands)):
        trs, errors = [], 0
        for t in targets:
            c = results[("cand", ci, t["name"])]
            errors += c["errors_a"]
            deltas, absolute = [], []
            for s in seeds:
                cp = c["paired"][s]
                bp = [0] * len(cp) if t["zero_base"] else results[("base", t["name"])]["paired"][s]
                deltas += [(x - y) / BIG_BLIND for x, y in zip(cp, bp)]
                absolute += [x / BIG_BLIND for x in cp]
            n = len(deltas)
            sd = statistics.stdev(deltas) if n > 1 else 0.0
            trs.append({"name": t["name"], "weight": t["weight"], "n": n,
                        "delta": sum(deltas) / n, "se": sd / math.sqrt(n) if n > 1 else 0.0,
                        "cand_abs": sum(absolute) / n})
        score = sum(tr["weight"] * tr["delta"] for tr in trs) / total_w
        se = math.sqrt(sum((tr["weight"] * tr["se"]) ** 2 for tr in trs)) / total_w
        out.append({"score": score, "se": se, "z": score / se if se > 0 else 0.0,
                    "min_target": min(tr["delta"] for tr in trs),
                    "errors": errors, "targets": trs})
    return out


def fmt_eval(ev: dict) -> str:
    parts = " ".join(f"{t['name']}:{t['delta']:+.2f}±{t['se']:.2f}" for t in ev["targets"])
    return (f"score {ev['score']:+.3f}±{ev['se']:.3f} (z {ev['z']:+.1f}) "
            f"min {ev['min_target']:+.2f} err {ev['errors']} | {parts}")


# =============================================================================
# Output helpers
# =============================================================================

def pretty_params(params: dict) -> str:
    return "PARAMS = {\n" + "\n".join(f'    "{k}": {params[k]:.6f},' for k in sorted(params)) + "\n}"


def save_params(path: str, params: dict, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "meta": meta}, f, indent=2, sort_keys=True, default=float)
    print(f"Saved params to {path}", flush=True)


def parse_bounds(items) -> dict:
    out = {}
    for item in items or []:
        name, rng_ = item.split("=")
        lo, hi = (float(x) for x in rng_.split(","))
        out[name] = (lo, hi)
    return out


def parse_constraints(items) -> list:
    out = list(DEFAULT_CONSTRAINTS)
    for item in items or []:
        hi, lo, gap = item.split(",")
        out.append((hi, lo, float(gap)))
    return out


# =============================================================================
# Main loop
# =============================================================================

def run_tuning(args, pool) -> None:
    rng = random.Random(args.rng_seed)
    agent_path = abs_path(args.agent)
    fixed_paths = [abs_path(p) for p in args.opponents]
    overrides = parse_bounds(args.bound)

    champion = load_base_params(agent_path)
    anchor = dict(champion)

    constraints = []
    for c in parse_constraints(args.constraint):
        if satisfied(champion, c):
            constraints.append(c)
        elif args.repair_champion:
            champion = repair(champion, [c], anchor, overrides)
            constraints.append(c)
            print(f"NOTE: repaired starting params for constraint {c}")
        else:
            print(f"WARNING: starting params violate {c[0]} >= {c[1]} + {c[2]}; "
                  f"constraint NOT enforced (use --repair-champion to fix it)")

    print("=" * 80)
    print(f"Base PARAMS from {agent_path}")
    print(pretty_params(champion))
    print("=" * 80)
    print(f"mode={args.mode} opponents={[os.path.basename(p) for p in fixed_paths]}\n"
          f"generations={args.generations} candidates={args.candidates} deals/half={args.deals}\n"
          f"screen seeds={args.seeds} confirm seeds={args.confirm_seeds} finalists={args.finalists}\n"
          f"promote: delta>={args.promote_threshold:+.2f} bb, z>={args.promote_z}, "
          f"no target worse than {args.min_target_floor:+.2f}\n"
          f"fixed samples={args.fixed_samples or 'bot default'} cfg_hands={args.cfg_hands} "
          f"jobs={args.jobs}\n", flush=True)

    seed_counter = 0

    def draw_seeds(n):
        nonlocal seed_counter
        s = [str(args.seed_start + seed_counter + i) for i in range(n)]
        seed_counter += n
        return s

    hof = [dict(champion)]
    history = []
    t_start = time.perf_counter()

    # Sanity check: the champion must score ~0 against itself.
    if args.mode in ("self", "mixed"):
        targets = build_targets(agent_path, champion, anchor, [], [], args, rng)
        old_deals, args.deals = args.deals, max(10, min(args.deals, 40))
        ev = evaluate(pool, args, [champion], champion, targets, draw_seeds(2))[0]
        args.deals = old_deals
        flag = "OK" if abs(ev["z"]) < 3 else "SUSPICIOUS (should be ~0)"
        print(f"Sanity: champion vs itself {ev['score']:+.3f}±{ev['se']:.3f} bb/deal -> {flag}\n",
              flush=True)

    for gen in range(1, args.generations + 1):
        g0 = time.perf_counter()
        print("=" * 80)
        print(f"GENERATION {gen}/{args.generations}  (scale {args.scale:.3f})")
        print("=" * 80, flush=True)

        screen_seeds = draw_seeds(args.seeds)
        confirm_seeds = draw_seeds(args.confirm_seeds)
        targets = build_targets(agent_path, champion, anchor, hof, fixed_paths, args, rng)
        cands = make_candidates(champion, hof, args, rng, anchor, overrides, constraints)
        print(f"targets: {[t['name'] for t in targets]} | candidates: {len(cands)} | "
              f"screen seeds {screen_seeds} confirm seeds {confirm_seeds}", flush=True)

        # ---- stage 1: screen everyone ----
        screen = evaluate(pool, args, cands, champion, targets, screen_seeds)
        order = sorted(range(len(cands)), key=lambda i: screen[i]["score"], reverse=True)
        print("\nscreen (top 6):")
        for i in order[:6]:
            print(f"  cand {i:02d}: {fmt_eval(screen[i])}")
        finalists = [i for i in order if screen[i]["errors"] == 0][:args.finalists]

        # ---- stage 2: confirm finalists on fresh seeds ----
        promoted = None
        if finalists:
            confirm = evaluate(pool, args, [cands[i] for i in finalists], champion,
                               targets, confirm_seeds)
            print("\nconfirm (fresh seeds):")
            best = None
            for fi, ev in zip(finalists, confirm):
                ok = (ev["errors"] == 0 and ev["score"] >= args.promote_threshold
                      and ev["z"] >= args.promote_z
                      and ev["min_target"] >= args.min_target_floor)
                print(f"  cand {fi:02d}: {fmt_eval(ev)} -> {'PROMOTABLE' if ok else 'rejected'}")
                if ok and (best is None or ev["score"] > best[1]["score"]):
                    best = (fi, ev)
            if best:
                promoted = best
        else:
            print("\nno error-free finalists")

        if promoted:
            fi, ev = promoted
            champion = dict(cands[fi])
            hof.append(dict(champion))
            if len(hof) > args.hof_size:
                hof.pop(1 if len(hof) > 2 else 0)   # keep hof[0] (the original) as an anchor
            print(f"\nPROMOTED cand {fi:02d} (+{ev['score']:.3f} bb/deal vs old champion)")
            print(pretty_params(champion), flush=True)
            save_params(args.out, champion, {"generation": gen, "confirm": ev,
                                             "confirm_seeds": confirm_seeds,
                                             "mode": args.mode, "history": history})
        else:
            print("\nNo promotion this generation.")

        history.append({"generation": gen, "promoted": bool(promoted),
                        "best_screen": screen[order[0]]["score"],
                        "seconds": time.perf_counter() - g0})
        print(f"generation time {time.perf_counter() - g0:.1f}s", flush=True)
        if args.anneal:
            args.scale *= 0.92

    print("\n" + "=" * 80)
    print(f"TUNING COMPLETE in {time.perf_counter() - t_start:.1f}s | "
          f"promotions: {sum(h['promoted'] for h in history)}/{len(history)}")
    print("=" * 80)

    # ---- hold-out: final champion vs the ORIGINAL params on unseen seeds ----
    holdout = None
    if args.holdout_seeds > 0 and params_key(champion) != params_key(anchor):
        seeds = draw_seeds(args.holdout_seeds)
        saved_pool = args.pool_size
        args.pool_size = 1
        targets = build_targets(agent_path, anchor, anchor, [], fixed_paths, args, rng,
                                self_name="original")
        args.pool_size = saved_pool
        holdout = evaluate(pool, args, [champion], anchor, targets, seeds)[0]
        print(f"\nHOLD-OUT on unseen seeds {seeds} (final champion minus original, bb/deal):")
        print(f"  {fmt_eval(holdout)}")
        print("  delta vs 'original' = head-to-head; delta vs a fixed bot = improvement over "
              "the original's result against it.")
    elif params_key(champion) == params_key(anchor):
        print("\nNo promotions, so the PARAMS are unchanged. That is a valid result: "
              "nothing beat the champion with confidence.")

    print("\nFinal PARAMS:")
    print(pretty_params(champion))
    save_params(args.out, champion, {"final": True, "history": history, "holdout": holdout,
                                     "mode": args.mode, "args": vars(args)})


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Noise-aware PARAMS tuner for agent.py")
    p.add_argument("--agent", default="agent.py", help="tunable agent (must expose PARAMS)")
    p.add_argument("--mode", choices=["self", "fixed", "mixed"], default="mixed")
    p.add_argument("--opponents", nargs="*", default=[], help="fixed opponent bot files")

    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--deals", type=int, default=80, help="deals per half per seed")
    p.add_argument("--seeds", type=int, default=2, help="screening seeds per generation")
    p.add_argument("--confirm-seeds", type=int, default=4, help="fresh seeds to confirm finalists")
    p.add_argument("--finalists", type=int, default=3)
    p.add_argument("--holdout-seeds", type=int, default=6, help="0 disables the final check")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--rng-seed", type=int, default=123)

    p.add_argument("--scale", type=float, default=0.20, help="mutation size (+-fraction)")
    p.add_argument("--mutation-rate", type=float, default=0.25, help="share of params changed")
    p.add_argument("--anneal", action="store_true", help="shrink the mutation scale each generation")

    p.add_argument("--promote-threshold", type=float, default=0.10,
                   help="min confirmed gain over the champion (bb/deal)")
    p.add_argument("--promote-z", type=float, default=1.5, help="min z of the confirmed gain")
    p.add_argument("--min-target-floor", type=float, default=-0.50,
                   help="reject if the gain against ANY single target is below this")
    p.add_argument("--self-weight", type=float, default=1.0, help="total weight of the self pool")
    p.add_argument("--fixed-weight", type=float, default=1.0, help="weight of EACH fixed opponent")
    p.add_argument("--pool-size", type=int, default=3, help="self-play pool: champion+anchor+older")
    p.add_argument("--hof-size", type=int, default=5)

    p.add_argument("--fixed-samples", type=int, default=250,
                   help="Monte Carlo samples per decision (0 = use the bot's own time budget)")
    p.add_argument("--cfg-hands", type=int, default=1000, help="num_hands told to the bots")
    p.add_argument("--fresh-modules", action="store_true", help="reload agent files every half")

    p.add_argument("--constraint", action="append", metavar="HI,LO,GAP",
                   help="require params[HI] >= params[LO] + GAP (repeatable)")
    p.add_argument("--repair-champion", action="store_true",
                   help="fix starting params that violate a constraint")
    p.add_argument("--bound", action="append", metavar="NAME=LO,HI", help="override a bound")

    p.add_argument("--out", default="tuned_params.json")
    p.add_argument("--jobs", type=int, default=1)
    return p


def validate(args) -> None:
    for name in ("generations", "candidates", "deals", "seeds", "confirm_seeds", "finalists", "jobs"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise ValueError("--mutation-rate must be in [0, 1]")
    if not os.path.exists(abs_path(args.agent)):
        raise FileNotFoundError(abs_path(args.agent))
    if args.mode in ("fixed", "mixed") and not args.opponents:
        raise ValueError("--mode fixed/mixed requires --opponents")
    for o in args.opponents:
        if not os.path.exists(abs_path(o)):
            raise FileNotFoundError(abs_path(o))


def main() -> int:
    args = build_parser().parse_args()
    pool = None
    try:
        validate(args)
        pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
        run_tuning(args, pool)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())