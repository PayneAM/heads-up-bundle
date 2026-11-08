"""
selftune_play.py

Mixed self-play tuner for agent.py.

Evaluates every candidate against:
  1. current champion params from agent.py
  2. one or more fixed opponent bot files, e.g. bots/bot_mc_gto.py

Quick smoke test:
    python selftune_play.py --mode mixed --opponents bots/bot_mc_gto.py --generations 2 --candidates 4 --deals 40 --seeds 1 --jobs 4

Longer run:
    python selftune_play.py --mode mixed --opponents bots/bot_mc_gto.py --generations 10 --candidates 16 --deals 100 --seeds 3 --jobs 8

After tuning:
  1. Copy printed PARAMS into agent.py.
  2. Run python -m pytest -q.
  3. Run real judge matches.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
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
DEFAULT_AGENT_PATH = os.path.join(HERE, "agent.py")

_LOAD_COUNTER = 0


# =============================================================================
# Loading agent modules
# =============================================================================

def abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(HERE, path)


def load_agent_module(
    path: str,
    params: dict[str, float] | None = None,
    require_params: bool = False,
):
    """
    Load a fresh independent copy of an agent file.

    If params is not None, the loaded module must expose global PARAMS.
    Fixed opponents are loaded with params=None.
    """
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1

    path = abs_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent file not found: {path}")

    # Help imports from repo root and opponent's own folder.
    root = HERE
    folder = os.path.dirname(path)
    if root not in sys.path:
        sys.path.insert(0, root)
    if folder not in sys.path:
        sys.path.insert(0, folder)

    module_name = (
        f"loaded_agent_{os.getpid()}_"
        f"{_LOAD_COUNTER}_{random.randrange(10**9)}"
    )

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if require_params and not hasattr(module, "PARAMS"):
        raise RuntimeError(f"{path} does not expose PARAMS")

    if params is not None:
        if not hasattr(module, "PARAMS"):
            raise RuntimeError(f"{path} does not expose PARAMS but params were provided")
        module.PARAMS.update(params)

    for fn in ("new_half", "choose_action", "on_hand_end"):
        if not hasattr(module, fn):
            raise RuntimeError(f"{path} is missing required function {fn}")

    return module


def load_base_params(agent_path: str) -> dict[str, float]:
    module = load_agent_module(agent_path, require_params=True)

    params: dict[str, float] = {}
    for k, v in module.PARAMS.items():
        if isinstance(v, (int, float)):
            params[k] = float(v)

    if not params:
        raise RuntimeError("agent.py PARAMS has no numeric values")

    return params


# =============================================================================
# Bot/game runner
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
    seat: int
    label: str
    errors: int = 0
    invalid_actions: int = 0
    choose_seconds: float = 0.0


def make_bot_state(spec: BotSpec, seat: int, suffix: str) -> BotState:
    module = load_agent_module(spec.path, spec.params)
    return BotState(
        module=module,
        memory={},
        seat=seat,
        label=f"{spec.label}_{suffix}",
    )


def safe_new_half(bot: BotState, config: AgentConfig) -> None:
    try:
        bot.memory.clear()
        bot.module.new_half(config, bot.memory)
    except Exception:
        bot.errors += 1
        print(f"\n[{bot.label}] new_half crashed", flush=True)
        traceback.print_exc()


def safe_choose(bot: BotState, obs) -> Action:
    start = time.perf_counter()

    try:
        action = bot.module.choose_action(obs, bot.memory)
        action = Action(action)

        if action not in obs.legal_actions:
            bot.invalid_actions += 1
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


def safe_hand_end(bot: BotState, result) -> None:
    try:
        bot.module.on_hand_end(result, bot.memory)
    except Exception:
        bot.errors += 1
        print(f"\n[{bot.label}] on_hand_end crashed", flush=True)
        traceback.print_exc()


def play_one_half(
    spec_a: BotSpec,
    spec_b: BotSpec,
    seed: str,
    half: int,
    num_deals: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Play one duplicate half.

    half=1: A seat 0, B seat 1.
    half=2: A seat 1, B seat 0.
    """
    if half not in (1, 2):
        raise ValueError("half must be 1 or 2")

    config = AgentConfig(num_hands=num_deals)

    seat_a = 0 if half == 1 else 1
    seat_b = 1 - seat_a

    bot_a = make_bot_state(spec_a, seat_a, f"half{half}")
    bot_b = make_bot_state(spec_b, seat_b, f"half{half}")

    safe_new_half(bot_a, config)
    safe_new_half(bot_b, config)

    nets_a: list[int] = []
    nets_b: list[int] = []

    for hand_index in range(num_deals):
        deal = make_deal(seed, hand_index)
        button = button_for_deal(hand_index)
        game = Game(deal, button_seat=button, hand_index=hand_index)

        while not game.is_terminal:
            seat = game.to_act
            bot = bot_a if seat == seat_a else bot_b

            obs = game.observation(seat)
            action = safe_choose(bot, obs)

            try:
                game.apply(action)
            except Exception:
                bot.errors += 1
                print(f"\n[{bot.label}] game.apply failed with action={action}", flush=True)
                traceback.print_exc()
                game.apply(default_action(obs.to_call))

        result_a = game.hand_result(seat_a)
        result_b = game.hand_result(seat_b)

        safe_hand_end(bot_a, result_a)
        safe_hand_end(bot_b, result_b)

        nets_a.append(int(result_a.my_net))
        nets_b.append(int(result_b.my_net))

    if verbose:
        print(
            f"half {half} seed={seed}: "
            f"A {sum(nets_a):+d} chips, B {sum(nets_b):+d} chips, "
            f"A errors={bot_a.errors}, B errors={bot_b.errors}",
            flush=True,
        )

    return {
        "nets_a": nets_a,
        "nets_b": nets_b,
        "bot_a_errors": bot_a.errors,
        "bot_b_errors": bot_b.errors,
        "bot_a_invalid_actions": bot_a.invalid_actions,
        "bot_b_invalid_actions": bot_b.invalid_actions,
        "bot_a_choose_seconds": bot_a.choose_seconds,
        "bot_b_choose_seconds": bot_b.choose_seconds,
    }


def play_duplicate_match(
    spec_a: BotSpec,
    spec_b: BotSpec,
    seed: str,
    num_deals: int,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Play both duplicate halves and return score from A's point of view.
    """
    h1 = play_one_half(spec_a, spec_b, seed, half=1, num_deals=num_deals, verbose=verbose)
    h2 = play_one_half(spec_a, spec_b, seed, half=2, num_deals=num_deals, verbose=verbose)

    paired_chips = [
        h1["nets_a"][i] + h2["nets_a"][i]
        for i in range(num_deals)
    ]

    total_chips_a = sum(paired_chips)
    bb_per_deal_a = total_chips_a / BIG_BLIND / num_deals

    if num_deals >= 2:
        paired_bb = [x / BIG_BLIND for x in paired_chips]
        std_bb = statistics.stdev(paired_bb)
        stderr_bb = std_bb / math.sqrt(num_deals)
    else:
        std_bb = 0.0
        stderr_bb = 0.0

    errors_a = h1["bot_a_errors"] + h2["bot_a_errors"]
    errors_b = h1["bot_b_errors"] + h2["bot_b_errors"]

    return {
        "seed": seed,
        "num_deals": num_deals,
        "total_chips_a": total_chips_a,
        "bb_per_deal_a": bb_per_deal_a,
        "paired_chips": paired_chips,
        "std_bb": std_bb,
        "stderr_bb": stderr_bb,
        "errors_a": errors_a,
        "errors_b": errors_b,
        "half1_a_chips": sum(h1["nets_a"]),
        "half2_a_chips": sum(h2["nets_a"]),
        "a_choose_seconds": h1["bot_a_choose_seconds"] + h2["bot_a_choose_seconds"],
        "b_choose_seconds": h1["bot_b_choose_seconds"] + h2["bot_b_choose_seconds"],
    }


# =============================================================================
# Mutation
# =============================================================================

def clamp_param(name: str, value: float) -> float:
    if "threshold" in name:
        return max(0.20, min(0.95, value))
    if "margin" in name:
        return max(-0.08, min(0.30, value))
    if "penalty" in name:
        return max(0.00, min(0.90, value))
    if "bonus" in name:
        return max(-0.15, min(0.25, value))
    if "discount" in name:
        return max(0.00, min(0.35, value))
    if "reference" in name:
        return max(0.00, min(1.00, value))
    if "price" in name:
        return max(0.00, min(0.70, value))
    if "randomization" in name:
        return max(0.00, min(0.04, value))
    if "multiplier" in name:
        return max(0.00, min(1.00, value))
    if "scale" in name:
        return max(0.00, min(3.00, value))
    if "var" in name:
        return max(0.00, min(0.20, value))
    if "station" in name and "min" in name:
        return max(1.00, min(100.00, value))
    if "short_stack_threshold" in name:
        return max(2.00, min(80.00, value))
    if name.startswith("use_"):
        return 1.0 if value >= 0.5 else 0.0
    return value


def mutate_params(
    parent: dict[str, float],
    scale: float,
    rng: random.Random,
    mutation_rate: float,
) -> dict[str, float]:
    child = dict(parent)

    for k, v in list(child.items()):
        if not isinstance(v, (int, float)):
            continue

        if rng.random() > mutation_rate:
            continue

        if k.startswith("use_"):
            if rng.random() < 0.08:
                child[k] = 0.0 if float(v) >= 0.5 else 1.0
            continue

        if abs(v) > 1.0e-9:
            factor = rng.uniform(1.0 - scale, 1.0 + scale)
            new_v = float(v) * factor
        else:
            new_v = rng.uniform(-scale, scale) * 0.02

        child[k] = clamp_param(k, new_v)

    return child


def crossover_params(
    a: dict[str, float],
    b: dict[str, float],
    rng: random.Random,
) -> dict[str, float]:
    keys = sorted(set(a) | set(b))
    child: dict[str, float] = {}

    for k in keys:
        if k in a and k in b:
            child[k] = a[k] if rng.random() < 0.5 else b[k]
        elif k in a:
            child[k] = a[k]
        else:
            child[k] = b[k]

    return child


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_candidate_against_spec(
    candidate_params: dict[str, float],
    opponent_spec: BotSpec,
    agent_path: str,
    seeds: list[str],
    num_deals: int,
    verbose: bool = False,
) -> dict[str, Any]:
    candidate_spec = BotSpec(
        path=agent_path,
        params=candidate_params,
        label="candidate",
    )

    seed_results = []
    errors_candidate = 0
    errors_opponent = 0

    for seed in seeds:
        r = play_duplicate_match(
            spec_a=candidate_spec,
            spec_b=opponent_spec,
            seed=seed,
            num_deals=num_deals,
            verbose=verbose,
        )
        seed_results.append(r)
        errors_candidate += r["errors_a"]
        errors_opponent += r["errors_b"]

    scores = [r["bb_per_deal_a"] for r in seed_results]
    mean_score = statistics.mean(scores)

    return {
        "opponent": opponent_spec.label,
        "mean_bb_per_deal": mean_score,
        "scores": scores,
        "score_std": statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        "seed_results": seed_results,
        "errors_candidate": errors_candidate,
        "errors_opponent": errors_opponent,
    }


def evaluate_candidate_multi(
    candidate_params: dict[str, float],
    champion_params: dict[str, float],
    agent_path: str,
    fixed_opponents: list[str],
    mode: str,
    seeds: list[str],
    num_deals: int,
    verbose: bool = False,
    self_weight: float = 1.0,
    fixed_weight: float = 1.0,
) -> dict[str, Any]:
    target_results = []
    weighted_scores = []
    total_candidate_errors = 0

    if mode in ("self", "mixed"):
        champion_spec = BotSpec(
            path=agent_path,
            params=champion_params,
            label="champion",
        )
        r = evaluate_candidate_against_spec(
            candidate_params=candidate_params,
            opponent_spec=champion_spec,
            agent_path=agent_path,
            seeds=seeds,
            num_deals=num_deals,
            verbose=verbose,
        )
        target_results.append(r)
        weighted_scores.append((self_weight, r["mean_bb_per_deal"]))
        total_candidate_errors += r["errors_candidate"]

    if mode in ("fixed", "mixed"):
        for opp_path in fixed_opponents:
            label = os.path.basename(opp_path)
            opp_spec = BotSpec(
                path=opp_path,
                params=None,
                label=label,
            )
            r = evaluate_candidate_against_spec(
                candidate_params=candidate_params,
                opponent_spec=opp_spec,
                agent_path=agent_path,
                seeds=seeds,
                num_deals=num_deals,
                verbose=verbose,
            )
            target_results.append(r)
            weighted_scores.append((fixed_weight, r["mean_bb_per_deal"]))
            total_candidate_errors += r["errors_candidate"]

    if not weighted_scores:
        raise RuntimeError("No evaluation targets selected")

    total_w = sum(w for w, _ in weighted_scores)
    mean_score = sum(w * s for w, s in weighted_scores) / total_w

    min_target_score = min(r["mean_bb_per_deal"] for r in target_results)

    return {
        "mean_bb_per_deal": mean_score,
        "min_target_score": min_target_score,
        "target_results": target_results,
        "errors_candidate": total_candidate_errors,
    }


def evaluate_candidate_worker(payload):
    (
        idx,
        candidate,
        champion,
        agent_path,
        fixed_opponents,
        mode,
        seeds,
        num_deals,
        verbose,
        self_weight,
        fixed_weight,
    ) = payload

    t0 = time.perf_counter()

    result = evaluate_candidate_multi(
        candidate_params=candidate,
        champion_params=champion,
        agent_path=agent_path,
        fixed_opponents=fixed_opponents,
        mode=mode,
        seeds=seeds,
        num_deals=num_deals,
        verbose=verbose,
        self_weight=self_weight,
        fixed_weight=fixed_weight,
    )

    elapsed = time.perf_counter() - t0
    return idx, candidate, result, elapsed


# =============================================================================
# Formatting/saving
# =============================================================================

def pretty_params(params: dict[str, float]) -> str:
    lines = ["PARAMS = {"]
    for k in sorted(params):
        lines.append(f'    "{k}": {params[k]:.6f},')
    lines.append("}")
    return "\n".join(lines)


def save_params(path: str, params: dict[str, float], meta: dict[str, Any]) -> None:
    payload = {
        "params": params,
        "meta": meta,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"\nSaved params to {path}", flush=True)


def summarize_eval_result(result: dict[str, Any]) -> str:
    parts = []
    for tr in result["target_results"]:
        scores = " ".join(f"{x:+.3f}" for x in tr["scores"])
        parts.append(
            f"{tr['opponent']}={tr['mean_bb_per_deal']:+.3f}"
            f"[{scores}]"
            f"/err{tr['errors_candidate']}"
        )
    return " | ".join(parts)


def print_eval_line(
    gen: int,
    idx: int,
    total: int,
    score: float,
    eval_result: dict[str, Any],
    elapsed: float,
) -> None:
    print(
        f"gen {gen:02d} cand {idx:02d}/{total:02d}: "
        f"score={score:+.3f} "
        f"min_target={eval_result['min_target_score']:+.3f} "
        f"{summarize_eval_result(eval_result)} "
        f"errors={eval_result['errors_candidate']} "
        f"time={elapsed:.1f}s",
        flush=True,
    )


# =============================================================================
# Generation runner
# =============================================================================

def run_generation_evals(
    gen: int,
    candidates: list[dict[str, float]],
    champion: dict[str, float],
    agent_path: str,
    fixed_opponents: list[str],
    mode: str,
    seeds: list[str],
    num_deals: int,
    jobs: int,
    verbose: bool,
    self_weight: float,
    fixed_weight: float,
) -> list[tuple[float, dict[str, float], dict[str, Any], float]]:
    payloads = [
        (
            i,
            cand,
            champion,
            agent_path,
            fixed_opponents,
            mode,
            seeds,
            num_deals,
            verbose,
            self_weight,
            fixed_weight,
        )
        for i, cand in enumerate(candidates, start=1)
    ]

    results: list[tuple[float, dict[str, float], dict[str, Any], float]] = []

    if jobs <= 1:
        for payload in payloads:
            idx, cand, eval_result, elapsed = evaluate_candidate_worker(payload)
            score = eval_result["mean_bb_per_deal"]
            results.append((score, cand, eval_result, elapsed))
            print_eval_line(gen, idx, len(candidates), score, eval_result, elapsed)
        return results

    jobs = min(jobs, len(candidates))

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(evaluate_candidate_worker, p) for p in payloads]

        for future in as_completed(futures):
            idx, cand, eval_result, elapsed = future.result()
            score = eval_result["mean_bb_per_deal"]
            results.append((score, cand, eval_result, elapsed))
            print_eval_line(gen, idx, len(candidates), score, eval_result, elapsed)

    return results


# =============================================================================
# Main tuning loop
# =============================================================================

def run_tuning(args) -> None:
    rng = random.Random(args.rng_seed)

    agent_path = abs_path(args.agent)
    fixed_opponents = [abs_path(p) for p in args.opponents]

    champion = load_base_params(agent_path)

    print("=" * 80)
    print(f"Loaded base PARAMS from {agent_path}")
    print(pretty_params(champion))
    print("=" * 80)

    seeds = [str(args.seed_start + i) for i in range(args.seeds)]

    print(
        f"\nTuning settings:"
        f"\n  mode: {args.mode}"
        f"\n  agent: {agent_path}"
        f"\n  opponents: {fixed_opponents if fixed_opponents else 'none'}"
        f"\n  generations: {args.generations}"
        f"\n  candidates/generation: {args.candidates}"
        f"\n  deals/half/seed: {args.deals}"
        f"\n  seeds: {seeds}"
        f"\n  jobs: {args.jobs}"
        f"\n  mutation scale: {args.scale}"
        f"\n  mutation rate: {args.mutation_rate}"
        f"\n  promote threshold: {args.promote_threshold:+.3f}"
        f"\n  min target floor: {args.min_target_floor:+.3f}"
        f"\n  self weight: {args.self_weight}"
        f"\n  fixed weight: {args.fixed_weight}"
        f"\n",
        flush=True,
    )

    if args.mode in ("self", "mixed"):
        print("Sanity check: champion vs champion...", flush=True)
        sanity = evaluate_candidate_multi(
            candidate_params=champion,
            champion_params=champion,
            agent_path=agent_path,
            fixed_opponents=[],
            mode="self",
            seeds=seeds[: max(1, min(2, len(seeds)))],
            num_deals=max(10, min(args.deals, 40)),
            verbose=False,
            self_weight=1.0,
            fixed_weight=0.0,
        )
        print(f"  champion self score: {sanity['mean_bb_per_deal']:+.3f}", flush=True)

    best_ever = dict(champion)
    best_ever_score = 0.0
    hall_of_fame: list[dict[str, float]] = [dict(champion)]

    start_all = time.perf_counter()

    for gen in range(1, args.generations + 1):
        print("\n" + "=" * 80)
        print(f"GENERATION {gen}/{args.generations}")
        print("=" * 80)

        candidates: list[dict[str, float]] = []

        for _ in range(args.candidates):
            candidates.append(
                mutate_params(
                    champion,
                    scale=args.scale,
                    rng=rng,
                    mutation_rate=args.mutation_rate,
                )
            )

        if len(hall_of_fame) >= 2 and args.candidates >= 4:
            for _ in range(max(1, args.candidates // 4)):
                p1 = rng.choice(hall_of_fame)
                p2 = rng.choice(hall_of_fame)
                child = crossover_params(p1, p2, rng)
                child = mutate_params(
                    child,
                    scale=args.scale * 0.75,
                    rng=rng,
                    mutation_rate=args.mutation_rate,
                )
                candidates.append(child)

        results = run_generation_evals(
            gen=gen,
            candidates=candidates,
            champion=champion,
            agent_path=agent_path,
            fixed_opponents=fixed_opponents,
            mode=args.mode,
            seeds=seeds,
            num_deals=args.deals,
            jobs=args.jobs,
            verbose=args.verbose,
            self_weight=args.self_weight,
            fixed_weight=args.fixed_weight,
        )

        results.sort(key=lambda x: x[0], reverse=True)
        top_score, top_params, top_eval, _elapsed = results[0]

        print("\nBest candidate this generation:")
        print(f"  score: {top_score:+.3f}")
        print(f"  min target: {top_eval['min_target_score']:+.3f}")
        print(f"  errors: {top_eval['errors_candidate']}")
        print(f"  details: {summarize_eval_result(top_eval)}")

        can_promote = (
            top_eval["errors_candidate"] == 0
            and top_score >= args.promote_threshold
            and top_eval["min_target_score"] >= args.min_target_floor
        )

        if can_promote:
            print("\nPROMOTING candidate to champion.", flush=True)

            champion = dict(top_params)
            hall_of_fame.append(dict(champion))
            if len(hall_of_fame) > args.hof_size:
                hall_of_fame.pop(0)

            if top_score > best_ever_score:
                best_ever_score = top_score
                best_ever = dict(top_params)

            save_params(
                args.out,
                champion,
                meta={
                    "generation": gen,
                    "score": top_score,
                    "min_target_score": top_eval["min_target_score"],
                    "mode": args.mode,
                    "deals": args.deals,
                    "seeds": seeds,
                    "jobs": args.jobs,
                    "opponents": fixed_opponents,
                    "details": top_eval,
                },
            )

            print("\nNew champion PARAMS:")
            print(pretty_params(champion))

        else:
            print("\nNot promoted.")
            if top_eval["errors_candidate"] != 0:
                print("  reason: candidate had errors")
            if top_score < args.promote_threshold:
                print(f"  reason: score {top_score:+.3f} < threshold {args.promote_threshold:+.3f}")
            if top_eval["min_target_score"] < args.min_target_floor:
                print(
                    f"  reason: min target {top_eval['min_target_score']:+.3f} "
                    f"< floor {args.min_target_floor:+.3f}"
                )

        if args.anneal:
            args.scale *= 0.92
            print(f"Annealed mutation scale to {args.scale:.4f}")

    elapsed_all = time.perf_counter() - start_all

    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)
    print(f"Elapsed: {elapsed_all:.1f}s")
    print(f"Best observed promoted score: {best_ever_score:+.3f}")

    print("\nFinal champion PARAMS:")
    print(pretty_params(champion))

    save_params(
        args.out,
        champion,
        meta={
            "final": True,
            "elapsed_seconds": elapsed_all,
            "generations": args.generations,
            "candidates": args.candidates,
            "deals": args.deals,
            "seeds": seeds,
            "mode": args.mode,
            "jobs": args.jobs,
            "opponents": fixed_opponents,
        },
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mixed self-play/fixed-opponent PARAMS tuner for agent.py"
    )

    p.add_argument("--agent", default="agent.py", help="Tunable agent path. Must expose PARAMS.")
    p.add_argument(
        "--mode",
        choices=["self", "fixed", "mixed"],
        default="mixed",
        help="self: candidate vs champion; fixed: candidate vs opponents; mixed: both.",
    )
    p.add_argument("--opponents", nargs="*", default=[], help="Fixed opponent bot files.")

    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--candidates", type=int, default=6)
    p.add_argument("--deals", type=int, default=80)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--rng-seed", type=int, default=123)

    p.add_argument("--scale", type=float, default=0.20)
    p.add_argument("--mutation-rate", type=float, default=0.75)

    p.add_argument("--promote-threshold", type=float, default=0.20)
    p.add_argument(
        "--min-target-floor",
        type=float,
        default=-0.35,
        help="Do not promote if candidate is worse than this against any target.",
    )

    p.add_argument("--self-weight", type=float, default=1.0)
    p.add_argument("--fixed-weight", type=float, default=1.0)

    p.add_argument("--hof-size", type=int, default=5)
    p.add_argument("--out", default="tuned_params.json")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--anneal", action="store_true")
    p.add_argument("--verbose", action="store_true")

    return p


def validate_args(args) -> None:
    if args.generations < 1:
        raise ValueError("--generations must be >= 1")
    if args.candidates < 1:
        raise ValueError("--candidates must be >= 1")
    if args.deals < 1:
        raise ValueError("--deals must be >= 1")
    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if not (0.0 <= args.mutation_rate <= 1.0):
        raise ValueError("--mutation-rate must be in [0, 1]")
    if args.self_weight < 0:
        raise ValueError("--self-weight must be >= 0")
    if args.fixed_weight < 0:
        raise ValueError("--fixed-weight must be >= 0")

    agent_path = abs_path(args.agent)
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Could not find tunable agent: {agent_path}")

    if args.mode in ("fixed", "mixed") and not args.opponents:
        raise ValueError("--mode fixed/mixed requires --opponents")

    for opp in args.opponents:
        opp_path = abs_path(opp)
        if not os.path.exists(opp_path):
            raise FileNotFoundError(f"Could not find opponent: {opp_path}")


def main() -> int:
    args = build_parser().parse_args()

    try:
        validate_args(args)
        run_tuning(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())