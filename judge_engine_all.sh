#!/usr/bin/env bash

source .venv/bin/activate
trap 'kill 0' EXIT

test_bots=(
    # "baseline-bots/baseline_random.py"
    # "baseline-bots/baseline_always_pot.py"
    # "baseline-bots/baseline_calling_station.py"
    # "baseline-bots/baseline_always_half.py"
    "bots/bot_base.py"
    "bots/bot_base_optim.py"
    "bots/bot_optim_cache.py"
    "agent_gto.py"
)

for bot in "${test_bots[@]}"; do
    PORT=5008 python "$bot" > /tmp/bot1.log 2>&1 &
    pid1=$!
    PORT=5009 python agent.py > /tmp/bot2.log 2>&1 &
    pid2=$!

    sleep 2

    python judge_engine.py --deals 1000 --seed 7
    python judge_engine.py --deals 1000 --seed 11
    python judge_engine.py --deals 1000 --seed 23

    kill "$pid1" "$pid2" 2>/dev/null
    wait "$pid1" "$pid2" 2>/dev/null
done

# PORT=5008 python bots/bot_base_optim.py > /tmp/bot1.log 2>&1 &
# PORT=5008 python bots/bot_optim_cache.py > /tmp/bot1.log 2>&1 &


# PORT=5008 python agent_gto.py > /tmp/bot2.log 2>&1 &
# pid1=$!
# PORT=5009 python agent_fast.py > /tmp/bot2.log 2>&1 &
# pid2=$!
# for i in 7 9 13; do

#     sleep 2

#     python judge_engine.py --deals 1000 --seed $((7 + i * 4))

# done

# kill "$pid1" "$pid2" 2>/dev/null
# wait "$pid1" "$pid2" 2>/dev/null