"""P2 低开销时序采集器（内存缓存，运行后落盘）。

实时循环里只做 `time.perf_counter()` 取值 + list/Counter 追加；不写文件、不打日志。
运行结束再 `summary()` / `dump()`。

可挂在 PikachuBridge 上（`bridge.timing = Timing()`），也可由 runner 持有。
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict


def _pctl(xs: list, p: float):
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


class Timing:
    def __init__(self) -> None:
        self._marks: dict = {}
        self._durs: dict = defaultdict(list)      # name -> [ms, ...]
        self._counts: Counter = Counter()
        self._events: list = []                    # (monotonic, kind, payload)
        self._ints: dict = defaultdict(list)       # interval name -> [ms, ...]
        self._last_t: dict = {}

    # ---- marks（绝对单调时间） ----
    def __setitem__(self, k, v):
        self._marks[k] = v

    def __getitem__(self, k):
        return self._marks[k]

    def __contains__(self, k):
        return k in self._marks

    def __iter__(self):
        return iter(self._marks)

    def get(self, k, default=None):
        return self._marks.get(k, default)

    def update(self, d):
        self._marks.update(d)

    def counts(self):
        return dict(self._counts)

    def mark(self, k):
        t = time.monotonic()
        self._marks[k] = t
        return t

    # ---- counts / durations / intervals ----
    def count(self, k, n=1):
        self._counts[k] += n
        return self._counts[k]

    def add(self, k, ms):
        self._durs[k].append(ms)

    def span(self, k, t0):
        self._durs[k].append((time.monotonic() - t0) * 1e3)

    def interval(self, k):
        """记录 k 相邻两次调用之间的间隔 (ms)。"""
        now = time.monotonic()
        prev = self._last_t.get(k)
        self._last_t[k] = now
        if prev is not None:
            self._durs[k].append((now - prev) * 1e3)

    def event(self, kind, payload=None):
        self._events.append((time.monotonic(), kind, payload))
        return self._events[-1][0]

    def first_event(self, kind):
        for t, k, p in self._events:
            if k == kind:
                return t, p
        return None

    def stats(self, k):
        xs = self._durs.get(k) or []
        return {"n": len(xs), "p50": _pctl(xs, 0.5), "p95": _pctl(xs, 0.95),
                "max": (max(xs) if xs else None),
                "mean": (sum(xs) / len(xs) if xs else None)}

    def summary(self) -> dict:
        return {
            "marks": {k: round(v, 4) for k, v in self._marks.items() if not k.startswith("_")},
            "counts": dict(self._counts),
            "stats_ms": {k: self.stats(k) for k in sorted(self._durs)},
            "events": [(round(t, 4), k, p) for t, k, p in self._events],
        }

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2, ensure_ascii=False)
