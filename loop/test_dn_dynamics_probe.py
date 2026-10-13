"""P2d-15 回归：dn_dynamics_probe 的窗口/度量/回正计算（纯 numpy，无需脑）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.dn_dynamics_probe import PLANS, per_dn_metrics, plan_segments

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== dn_dynamics_probe metrics ===========")
    dt = 0.02
    plan = PLANS["left"]                                    # A1.2 / C2.0 / A1.8
    labels = []
    for name, dur in plan:
        labels += [name] * int(round(dur / dt))
    n = len(labels)

    segs = plan_segments(plan, dt)
    check("plan_segments: 3 segments", len(segs) == 3)
    check("plan_segments: contiguous & cover all steps",
          segs[0][1] == 0 and segs[-1][2] == n and all(segs[i][2] == segs[i + 1][1]
                                                      for i in range(len(segs) - 1)))
    check("plan_segments: stim segment labels are C",
          all(labels[i] == "C" for i in range(segs[1][1], segs[1][2])))

    # 合成：一个"启动后持续、撤除后回基线"的神经元；一个"不回来"的神经元
    base_dn = np.zeros(n)
    t = np.arange(n) * dt
    sw = labels.index("C")
    good = np.where(t >= sw * dt, 0.5, 0.0)                 # 启动后一直 0.5
    post = [i for i, x in enumerate(labels) if x == "A"]
    post = [i for i in post if i > sw]                      # 撤除后
    good[post[0]:] = 0.0                                    # 撤除立刻回 0（回正）
    stuck = good.copy()
    stuck[post[0]:] = 0.5                                   # 撤除后**不**回（持续）
    act = np.stack([good, stuck], 0)

    m = per_dn_metrics(act, labels, plan, dt)
    assert m is not None
    check("onset latency detected (>=0) for both", m["onset"][0] >= 0 and m["onset"][1] >= 0)
    check("onset latency ~ switch (small)", m["onset"][0] < 0.5)
    check("recovery ratio small for returning neuron", m["ret"][0] < 0.5)
    check("recovery ratio large for stuck neuron", m["ret"][1] > 0.5)
    check("dresp positive for both", m["dresp"][0] > 0 and m["dresp"][1] > 0)

    # 左右反号检查：右刺激下合成一个负响应
    plan_r = PLANS["right"]
    labels_r = []
    for name, dur in plan_r:
        labels_r += [name] * int(round(dur / dt))
    nr = len(labels_r)
    swr = labels_r.index("B")
    neg = np.where(np.arange(nr) * dt >= swr * dt, -0.4, 0.0)
    mr = per_dn_metrics(np.stack([neg], 0), labels_r, plan_r, dt)
    assert mr is not None
    check("sign(left)+ vs sign(right)- are opposite",
          np.sign(m["dresp"][0]) != np.sign(mr["dresp"][0]))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
