"""P2d-15a 回归：dn_dynamics_probe 的段边界、启动延迟仅限刺激段、群体差分。纯 numpy。"""
from __future__ import annotations

import sys

import numpy as np

from loop.dn_dynamics_probe import PLANS, per_dn_metrics, plan_segments, pop_diff, seg_means

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== dn_dynamics_probe (P2d-15a) ===========")
    dt = 0.02
    plan = PLANS["left"]                                    # A1.2 / C2.0 / A1.8
    segs = plan_segments(plan, dt)
    _an, a0, a1 = segs[0]
    _sn, s0, s1 = segs[1]
    _pn, p0, p1 = segs[2]
    n = segs[-1][2]

    check("plan_segments: contiguous & cover", segs[0][1] == 0 and segs[-1][2] == n
          and all(segs[i][2] == segs[i + 1][1] for i in range(len(segs) - 1)))

    # 神经元1：刺激段内起反应（应在刺激段内检出 onset）
    dn_stim = np.zeros(n)
    dn_stim[s0:s1] = 1.0
    # 神经元2：**撤除后**才开始反应（刺激段内无反应 -> onset 必须为 -1，不能被撤除后数据误导）
    dn_post = np.zeros(n)
    dn_post[p0:] = 1.0
    act = np.stack([dn_stim, dn_post], 0)
    m = per_dn_metrics(act, plan, dt)
    assert m is not None
    check("onset in-stimulus neuron detected (>=0)", m["onset"][0] >= 0)
    check("onset latency small for in-stimulus neuron", m["onset"][0] < 0.5)
    check("onset NOT detected for post-only neuron (search bounded to stim segment)",
          m["onset"][1] == -1.0)
    check("dresp>0 for in-stimulus neuron", m["dresp"][0] > 0)

    # 回正：撤除后回基线的神经元 ret 小；不回的 ret 大
    dn_stuck = np.zeros(n)
    dn_stuck[s0:] = 1.0                                     # 撤除后仍保持
    m2 = per_dn_metrics(np.stack([dn_stim, dn_stuck], 0), plan, dt)
    assert m2 is not None
    check("returns-to-baseline -> ret small", m2["ret"][0] < 0.5)
    check("stuck-after-removal -> ret large", m2["ret"][1] > 0.5)

    # 群体差分 D = mean(L组) - mean(R组)
    act2 = np.zeros((4, n))
    act2[0:2, s0:s1] = 1.0                                  # 左组刺激段升高；右组全 0
    D = pop_diff(act2, [0, 1], [2, 3])
    check("pop_diff: D>0 during stimulus", float(np.mean(D[s0:s1])) > 0)
    check("pop_diff: D~0 before stimulus", abs(float(np.mean(D[a0:a1]))) < 1e-9)
    sm = seg_means(D, plan, dt)
    check("seg_means: one mean per segment", len(sm) == 3)
    check("time-matched net effect decays after removal", abs(sm[2]) < abs(sm[1]))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
