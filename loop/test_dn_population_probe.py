"""P2d-13a 回归：修复后的群体探针 —— 独立试次、感觉缓存不失效、顺序无关。

用假脑（CPU、快、确定性），验证：
  - 每次 run_trial 取到足量**新** ToF 帧（frames_consumed ≈ seconds*tof_hz，不是 ~1）；
  - FR-only 真的改变右路 size、FL-only 真的改变左路 size（A/B/C 不复用失效缓存）；
  - 改变刺激执行顺序不改变各独立试次的结果；
  - 感觉有效性门槛在输入不变化时**报错**而不是继续。
"""
from __future__ import annotations

import sys

import numpy as np

from loop.dn_population_probe import SIZE_KEYS, TOL, run_trial, sensory_gate
from loop.run import make_setup

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== dn_population_probe (fixed) ===========")
    from flybrain import Trace
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", dt=None, device="cpu", seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))

    kw = dict(seed=64, seconds=1.5, settle=0.8, tof_hz=10.0)
    a = run_trial(cfg, brain, groups, dn_idx, Trace, {}, **kw)
    b = run_trial(cfg, brain, groups, dn_idx, Trace, {"FR": 400.0}, **kw)
    c = run_trial(cfg, brain, groups, dn_idx, Trace, {"FL": 400.0}, **kw)

    check(f"trial consumes ~expected NEW frames ({a['frames_consumed']} vs "
          f"{a['expected_frames']})", a["frames_consumed"] >= a["expected_frames"] - 1)
    check("FR-only changes right size (size_R B>A)",
          b["info"]["size_R"] - a["info"]["size_R"] > TOL)
    check("FL-only changes left size (size_L C>A)",
          c["info"]["size_L"] - a["info"]["size_L"] > TOL)
    check("A~symmetric right size (no stale OBSTACLE cache)",
          abs(a["info"]["size_R"] - a["info"]["size_L"]) < 0.3)

    # 顺序无关：先 A 后 B，对比先 B 后 A
    a2 = run_trial(cfg, brain, groups, dn_idx, Trace, {}, **kw)
    b2 = run_trial(cfg, brain, groups, dn_idx, Trace, {"FR": 400.0}, **kw)
    check("A independent trial reproducible (order-free)",
          np.allclose(a["act"], a2["act"], atol=1e-5) and a["info"]["size_R"] == a2["info"]["size_R"])
    check("B independent trial reproducible (order-free)",
          np.allclose(b["act"], b2["act"], atol=1e-5) and b["info"]["size_R"] == b2["info"]["size_R"])

    # 门槛：把 B 换成与 A 相同的输入 -> 必须报错
    bad = {(64, "A_none"): a, (64, "B_FR"): a, (64, "C_FL"): c}
    raised = False
    try:
        sensory_gate(bad, [64])
    except SystemExit:
        raised = True
    check("sensory_gate raises when FR input does not change size", raised)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
