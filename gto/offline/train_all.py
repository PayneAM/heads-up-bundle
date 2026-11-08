"""Train every model in parallel processes, then merge.

    python -m gto.offline.train_all
"""
import os
import subprocess
import sys

from gto.offline.train import ARCHETYPES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    logs = os.path.join(HERE, "logs")
    os.makedirs(logs, exist_ok=True)
    jobs = []
    running = []
    limit = int(os.environ.get("PARALLEL", "4"))
    from gto.offline.train import EQ_SEEDS
    for name in (sys.argv[1:] or [f"eq{k}" for k in range(EQ_SEEDS)] + ARCHETYPES):
        while len([p for p in running if p.poll() is None]) >= limit:
            running[0].wait()
            running = [p for p in running if p.poll() is None]
        env = dict(os.environ)
        is_eq = name.startswith("eq")
        env["ITERS"] = os.environ.get("ITERS_EQ" if is_eq else "ITERS_EX",
                                      "8000000" if is_eq else "6000000")
        env["NUMBA_CACHE_DIR"] = os.path.join(HERE, "numba_cache", name)  # no shared-cache races
        log = open(os.path.join(logs, f"{name}.log"), "w")
        jobs.append((name, subprocess.Popen([sys.executable, "-m", "gto.offline.train", name],
                                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
        running.append(jobs[-1][1])
    failed = [n for n, p in jobs if p.wait() != 0]
    if failed:
        sys.exit(f"failed: {failed}")
    parts = os.path.join(HERE, "parts")
    needed = [f"eq{k}" for k in range(EQ_SEEDS)] + ARCHETYPES
    if all(os.path.exists(os.path.join(parts, f"{n}.npz")) for n in needed):
        subprocess.check_call([sys.executable, "-m", "gto.offline.train", "merge"], cwd=ROOT,
                              env=dict(os.environ, NUMBA_CACHE_DIR=os.path.join(HERE, "numba_cache", "merge")))


if __name__ == "__main__":
    main()
