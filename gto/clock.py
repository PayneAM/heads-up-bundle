"""
clock.py - time-bank control that sees what the judge charges.

The judge charges the WHOLE request (web server, JSON parsing, network)
minus 1 ms, not just our strategy code. On a slow or busy machine that
overhead alone can be ~1 ms per request x ~3,700 requests per half, so a
controller that only times its own Monte Carlo runs the bank dry.

We therefore measure the process's total CPU time (time.process_time:
every thread, user + system, including uvicorn/FastAPI work) since the
half started, add a small allowance per request for non-CPU latency, and
size each hand-strength estimate so the half ends near TARGET seconds.
The per-hand overhead is measured, not assumed, so this self-calibrates to
whatever machine the tournament runs on.
"""

from __future__ import annotations

import os
import sys
import time

TARGET = float(os.environ.get("GTO_TIME_TARGET", "4.0"))   # charged seconds per half we aim for
PANIC = 5.5                  # beyond this: cheapest estimates, skip optional work
EXTRA_PER_CALL = 0.0005      # s per request the judge sees that is not our CPU time
EST_PER_HAND = 2.2           # hand-strength estimates per hand (one per street reached)
PRIOR_OVERHEAD_MS = 2.5      # non-estimate cost per hand, before we have measured it
MAX_MS = 10.0                # cap for a single estimate
FALLBACK_PER_CALL = 0.0003  # coarse-timer fallback: plumbing cost per request we can't time
LOG = bool(os.environ.get("GTO_CLOCK_LOG"))


def _cpu_granularity() -> float:
    """Smallest step of time.process_time() (Linux: ~1e-6 s, Windows: ~15.6 ms)."""
    t0 = time.process_time()
    w0 = time.perf_counter()
    t1 = t0
    while t1 == t0 and time.perf_counter() - w0 < 0.05:
        t1 = time.process_time()
    return (t1 - t0) if t1 != t0 else 0.05


# A coarse CPU clock rounds every short request up or down by ~15 ms, which
# wildly misestimates the bank; then we time our own code by wall clock instead.
FINE_CPU_CLOCK = _cpu_granularity() < 0.002


class Clock:
    def __init__(self, num_hands: int = 1000):
        self.num_hands = max(1, int(num_hands))
        self.cpu0 = time.process_time()
        self.calls = 0
        self.est_time = 0.0      # seconds spent inside hand-strength estimates
        self.logged = False
        # wall vs CPU time inside our own code: on a busy shared machine the
        # judge's wall clock runs faster than our CPU clock (waiting for a core)
        self.wall_in = 0.0
        self.cpu_in = 0.0
        self._w = self._c = 0.0

    def begin(self) -> None:
        """Call at the start of each decision request."""
        self.calls += 1
        self._w, self._c = time.perf_counter(), time.process_time()

    def end(self) -> None:
        self.wall_in += time.perf_counter() - self._w
        self.cpu_in += time.process_time() - self._c

    def tick(self) -> None:  # kept for compatibility
        self.calls += 1

    def add_wall(self, seconds: float) -> None:
        """Time spent in our code outside a decision (e.g. on_hand_end)."""
        self.wall_in += seconds

    def contention(self) -> float:
        """How much slower wall time runs than CPU time (>= 1)."""
        if not FINE_CPU_CLOCK or self.cpu_in < 0.05:
            return 1.0
        return min(4.0, max(1.0, self.wall_in / self.cpu_in))

    def charged(self) -> float:
        """Estimated time-bank usage so far this half (seconds)."""
        if not FINE_CPU_CLOCK:
            return self.wall_in + self.calls * FALLBACK_PER_CALL
        cpu = time.process_time() - self.cpu0
        return cpu * self.contention() + self.calls * EXTRA_PER_CALL

    def panic(self) -> bool:
        return self.charged() > PANIC

    def add_estimate(self, seconds: float) -> None:
        self.est_time += seconds

    def allowance_ms(self, hand_index: int) -> float:
        """Milliseconds one hand-strength estimate may take right now."""
        used = self.charged()
        if used > PANIC:
            return 0.0
        left = max(1, self.num_hands - hand_index)
        per_hand_ms = 1000.0 * (TARGET - used) / left
        done = max(1, hand_index)
        measured = 1000.0 * max(0.0, used - self.est_time) / done
        w = min(1.0, hand_index / 50.0)                       # trust measurements after ~50 hands
        overhead_ms = (1 - w) * PRIOR_OVERHEAD_MS + w * measured
        allow = (per_hand_ms - overhead_ms) / EST_PER_HAND
        if LOG and not self.logged and hand_index >= self.num_hands - 1:
            self.logged = True
            print(f"[clock] charged~{used:.2f}s calls={self.calls} est={self.est_time:.2f}s "
                  f"overhead/hand={measured:.2f}ms contention={self.contention():.2f} "
                  f"fine_cpu_clock={FINE_CPU_CLOCK}", file=sys.stderr, flush=True)
        return max(0.0, min(MAX_MS, allow))


def pick_level(costs_ms, allowance_ms: float) -> int:
    """Most precise ladder level whose measured cost fits the allowance (0 if none)."""
    level = 0
    for i, c in enumerate(costs_ms):
        if c <= allowance_ms:
            level = i
    return level
