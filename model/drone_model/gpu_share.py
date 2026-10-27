"""Duty-cycle limit on GPU work, adjustable while a job runs.

A share file holds one number from 1 to 100: the percent of its normal pace a job may run at. After each unit
of GPU work (a training batch, an inference frame) the job sleeps busy * (100 / share - 1) seconds, so at 50 it
works half the time and keeps the GPU busy at most about half as much as at 100. The file is reread at most
every 5 s, so editing it changes a running job. Sleeping only spreads the same computation out in time, so
results do not depend on the share.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

REREAD_SECONDS = 5.0


class GpuShare:
    def __init__(self, path: Path | None):
        self.path = path
        self.share = 100.0
        self.read_at = -math.inf
        self.last = time.perf_counter()
        self.slept_s = 0.0

    def current(self) -> float:
        if self.path is None:
            return 100.0
        now = time.monotonic()
        if now - self.read_at >= REREAD_SECONDS:
            self.read_at = now
            try:
                value = float(self.path.read_text().split()[0])
                if not 1 <= value <= 100:
                    raise ValueError(value)
            except (OSError, ValueError, IndexError) as error:
                # An unreadable or out-of-range file keeps the last valid share, so a typo never stops a job.
                print(f'gpu_share: ignoring {self.path} ({error!r}), keeping {self.share:g}', file=sys.stderr, flush=True)
            else:
                if value != self.share:
                    print(f'gpu_share: {self.share:g} -> {value:g}', file=sys.stderr, flush=True)
                self.share = value
        return self.share

    def wait(self, busy_s: float) -> None:
        """Sleep after busy_s seconds of work so that the work fills `share` percent of the time."""
        share = self.current()
        if share < 100 and busy_s > 0:
            pause = busy_s * (100 / share - 1)
            time.sleep(pause)
            self.slept_s += pause

    def callback(self, _trainer=None) -> None:
        """Ultralytics batch-end callback: the work is the wall time since the previous call or restart."""
        self.wait(time.perf_counter() - self.last)
        self.last = time.perf_counter()

    def restart(self, _trainer=None) -> None:
        """Ultralytics epoch-start callback, so time between runs or epochs is not counted as work."""
        self.last = time.perf_counter()
