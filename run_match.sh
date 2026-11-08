#!/usr/bin/env bash

source .venv/bin/activate

test_bots=(
    "baseline-bots/baseline_random.py"
    "baseline-bots/baseline_always_pot.py"
    "baseline-bots/baseline_calling_station.py"
    "baseline-bots/baseline_always_half.py"
)

trap 'kill 0' EXIT

# for i in {0..3}; do
#     PORT=5008 python "${test_bots[$i]}" > /tmp/bot1.log 2>&1 &
#     PID1=$!
#     PORT=5009 python agent.py > /tmp/bot2.log 2>&1 &
#     PID2=$!

#     sleep 3

#     python judge_engine.py --deals 100 --seed 7 --json "out_$i.json" --logs

#     kill "$PID1" "$PID2" 2>/dev/null
#     wait "$PID1" "$PID2" 2>/dev/null
# done


my_bots=(
    "bots/bot_base.py"
    "bots/bot_base_optim.py"
    "agent_dumbass.py"
)

for i in {0..2}; do
    PORT=5008 python "${my_bots[$i]}" > /tmp/bot1.log 2>&1 &
    PID1=$!
    PORT=5009 python agent.py > /tmp/bot2.log 2>&1 &
    PID2=$!

    sleep 3

    python judge_engine.py --deals 200 --seed 7 --json "out_$i.json" --logs

    kill "$PID1" "$PID2" 2>/dev/null
    wait "$PID1" "$PID2" 2>/dev/null
done