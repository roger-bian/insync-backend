import time


class Profiler:
    """
    Opt-in per-stage timer for the frame loop.

    Stages are timed by lapping: every `lap` closes the interval opened by the
    previous `lap` or by `reset`, so the whole loop body is accounted for
    without a timer object per stage. Disabled it costs one attribute test.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.totals = {}
        self.order = []
        self._mark = 0.0

    def reset(self):
        if self.enabled:
            self._mark = time.perf_counter()

    def lap(self, stage: str):
        if not self.enabled:
            return
        now = time.perf_counter()
        if stage not in self.totals:
            self.totals[stage] = 0.0
            self.order.append(stage)
        self.totals[stage] += now - self._mark
        self._mark = now

    def report(self, frames: int, wall: float):
        if not self.enabled or not frames or wall <= 0:
            return

        print(f"\n{'stage':<14}{'total s':>9}{'ms/frame':>10}{'% wall':>9}")
        print("-" * 42)
        for stage in self.order:
            total = self.totals[stage]
            print(f"{stage:<14}{total:>9.2f}{total / frames * 1000:>10.2f}"
                  f"{total / wall * 100:>8.1f}%")

        # Whatever the laps did not cover: thread hand-off, tqdm, interpreter.
        other = wall - sum(self.totals.values())
        print(f"{'other':<14}{other:>9.2f}{other / frames * 1000:>10.2f}"
              f"{other / wall * 100:>8.1f}%")
        print("-" * 42)
        print(f"{'total':<14}{wall:>9.2f}{wall / frames * 1000:>10.2f}"
              f"{100.0:>8.1f}%")
