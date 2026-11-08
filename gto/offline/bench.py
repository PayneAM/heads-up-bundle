"""Benchmark agent.py against opponents with the real judge.

    python -m gto.offline.bench                       # all baselines + sparring bot
    python -m gto.offline.bench baseline_random --deals 1000 --seed 3
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OPPONENTS = {
    "baseline_random": os.path.join(ROOT, "baseline-bots", "baseline_random.py"),
    "baseline_calling_station": os.path.join(ROOT, "baseline-bots", "baseline_calling_station.py"),
    "baseline_always_half": os.path.join(ROOT, "baseline-bots", "baseline_always_half.py"),
    "baseline_always_pot": os.path.join(ROOT, "baseline-bots", "baseline_always_pot.py"),
    "sparring_equity": os.path.join(HERE, "sparring_equity_bot.py"),
}


def start(script, port):
    env = dict(os.environ, PORT=str(port))
    p = subprocess.Popen([sys.executable, script], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            return p
        except Exception:
            time.sleep(0.25)
    p.kill()
    raise RuntimeError(f"{script} did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("opponents", nargs="*")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", default="11")
    ap.add_argument("--me", default=os.path.join(ROOT, "agent.py"))
    args = ap.parse_args()
    for name in args.opponents or list(OPPONENTS):
        script = OPPONENTS.get(name, name)
        a = start(args.me, 5108)
        b = start(script, 5109)
        try:
            out = os.path.join(tempfile.gettempdir(), f"bench_{os.path.basename(name)}.json")
            r = subprocess.run([sys.executable, "judge_engine.py", "--agent1", "http://127.0.0.1:5108", "--agent2", "http://127.0.0.1:5109",
                                "--deals", str(args.deals), "--seed", args.seed, "--json", out],
                               cwd=ROOT, capture_output=True, text=True)
            lines = [l for l in r.stdout.splitlines() if l.strip()]
            print(f"=== {name}")
            print("\n".join(lines[-14:]))
            sys.stdout.flush()
        finally:
            a.kill()
            b.kill()


if __name__ == "__main__":
    main()
