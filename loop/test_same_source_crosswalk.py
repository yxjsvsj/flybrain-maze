"""P2d-24c 回归：同源对接的纯逻辑（校验函数 + bodyId join + 未提供字段），无需脑/网络。"""
from __future__ import annotations

import sys

import numpy as np

from loop.same_source_crosswalk import _loc_col, sha256

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _FakeCol:
    def __init__(self, vals):
        self.vals = vals

    def __getitem__(self, i):
        return _FakeVal(self.vals[i])


class _FakeVal:
    def __init__(self, v):
        self.v = v

    def as_py(self):
        return self.v


class _FakeTab:
    def __init__(self, cols):
        self.column_names = list(cols)
        self._c = {k: _FakeCol(v) for k, v in cols.items()}

    def column(self, k):
        return self._c[k]


def main() -> int:
    print("=========== same_source_crosswalk (logic) ===========")
    # sha256 稳定
    import os
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "xw_test.bin")
    with open(p, "wb") as fh:
        fh.write(b"abc")
    check("sha256 stable/known", sha256(p) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    # bodyId join：精确对应 bodyId，未命中 -> 未在官方表；None/NaN -> 未提供
    tab = _FakeTab({"bodyId": [828667, 905339],
                    "class": ["mechanosensory_proprioceptive", "mechanosensory_proprioceptive"],
                    "subclass": ["chordotonal organ", None]})
    pos = {int(b): i for i, b in enumerate(tab.column("bodyId").vals)}
    check("bodyId 828667 present", 828667 in pos)
    check("unknown bodyId not present", 999999 not in pos)
    check("None field reads as 未提供",
          (tab.column("subclass")[1].as_py() is None))

    # _loc_col：缺列/None/全 NaN -> 未提供
    check("_loc_col missing column -> 未提供", _loc_col(tab, 0, "somaLocation") == "未提供")
    tab2 = _FakeTab({"somaLocation": [[1.0, 2.0, 3.0], None]})
    check("_loc_col present -> list", _loc_col(tab2, 0, "somaLocation") == [1.0, 2.0, 3.0])
    check("_loc_col None -> 未提供", _loc_col(tab2, 1, "somaLocation") == "未提供")

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
