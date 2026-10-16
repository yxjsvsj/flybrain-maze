"""P2d-26f 回归：ring_validate 的验证逻辑（合成：真带状→显著；打乱→不显著）。用稠密假 W。"""
from __future__ import annotations

import sys

import numpy as np

from loop.ring_validate import weighted_circ

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _DenseW:
    """最小 W 替身：__getitem__ 返回带 .todense() 的对象。"""
    def __init__(self, M):
        self.M = M

    def __getitem__(self, idx):
        r, c = idx
        sub = self.M[np.ix_(np.atleast_1d(r), np.atleast_1d(c))]

        class _S:
            def __init__(s, a):
                s.a = a
            def todense(s):
                return s.a
        return _S(sub)


def _run_validate(epg_ids, ang_deg, pen_ids, W, idx_of, B=300, seed=0):
    from loop.ring_validate import validate
    return validate(epg_ids, ang_deg, {"PEN_a(PEN1)": pen_ids}, W, idx_of, B=B, seed=seed)


def main() -> int:
    print("=========== ring_validate (synthetic) ===========")
    # 8 个 EPG 均匀铺满圆
    NEP = 24
    epg = list(range(100, 100 + NEP))
    ang = [i * 15.0 for i in range(NEP)]
    # idx_of 与真实口径一致：bodyId -> 全局行/列索引；EPG=0..23，PEN=24,25
    idx_of = {100 + i: i for i in range(NEP)}
    idx_of[200], idx_of[201] = 24, 25

    # 真带状：PEN 只连 75..120° 的 4 个相邻 EPG（行=PEN 全局索引 24,25）
    M = np.zeros((26, NEP))
    M[24, [5, 6, 7, 8]] = 1.0
    M[25, [14, 15, 16, 17]] = 1.0
    W = _DenseW(M)
    r = _run_validate(epg, ang, [200], W, idx_of)
    rec = r["PEN_a(PEN1)"]["cells"][0]
    check("banded PEN: high R_w", rec["R_w"] > 0.9)
    check("banded PEN: significant (p small)", rec["p"] < 0.05)
    check("banded PEN: below null median", rec["R_w"] > rec["null_med_Rw"])

    # 对照：PEN 连全部 EPG（环上均匀）-> R_w 低，不显著
    Ma = np.zeros((26, NEP))
    Ma[24, :] = 1.0
    Ma[25, :] = 1.0
    Wa = _DenseW(Ma)
    ra = _run_validate(epg, ang, [200], Wa, idx_of)
    reca = ra["PEN_a(PEN1)"]["cells"][0]
    check("uniform-input PEN: low R_w", reca["R_w"] < 0.4)
    check("uniform-input PEN: not significant", reca["p"] > 0.05)

    # weighted_circ 基本正确性
    R, m = weighted_circ(np.radians([10.0, 10.0]), np.array([1.0, 3.0]))
    check("weighted_circ mean correct", abs(np.degrees(m) - 10) < 1e-6 and abs(R - 1) < 1e-9)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
