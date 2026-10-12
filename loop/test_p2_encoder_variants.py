"""P2d-13d 回归：P2 感觉分组适配层（BASE vs TEST）单变量有效性。纯 numpy + 假脑。

验证：
  - F 比 FL/FR 都近且 FL≠FR 时，BASE 左右被 F 淹没（|sizeL-sizeR| 小），TEST 保留左右差（更大）；
  - FR-only / FL-only：BASE≈TEST（F 远，本就不主导）；
  - F-only：TEST 不再产生侧向神经注入（chase_L/chase_R≈0），BASE 有；且 front_safety 不变；
  - 传感器/掩码 front 与 dmin_F 在 BASE/TEST 一致（front 只用 F）。
"""
from __future__ import annotations

import sys

import numpy as np

from config import apply_preset
from hardware.tof5 import ORDER
from hardware.tof5_client import TofFrame
from loop.p2_encoder_variants import make_encoder
from loop.run import make_setup

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def probe(groups, cfg, variant, ranges):
    enc = make_encoder(variant, groups, cfg, 0.02)
    rng = {n: ranges.get(n) for n in ORDER}
    st = {n: ("VALID" if ranges.get(n) is not None else "NO_TARGET") for n in ORDER}
    f = TofFrame(session_id="t", seq=1, t=1.0, healthy=True, ranges=rng, status=st,
                 ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)
    d = enc.sense(f)
    enc.update(f, d)
    _, info = enc.inject()
    return {"size_L": info["size_L"], "size_R": info["size_R"],
            "chase_L": enc._cache["L"][1], "chase_R": enc._cache["R"][1],
            "dmin_F": info["dmin_F"], "front": tuple(enc.front.tolist())}


def main() -> int:
    print("=========== P2 encoder variants (BASE vs TEST) ===========")
    cfg, _m, _c, _b, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", dt=None, device="cpu", seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    apply_preset(cfg, "real")               # 用真脑预设的编码增益（chase_size_gain=2.0）

    # 1) F 更近、FL≠FR：BASE 被 F 淹没，TEST 保留左右差
    fr = {"F": 200.0, "FL": 400.0, "FR": 600.0}
    b = probe(groups, cfg, "BASE", fr)
    t = probe(groups, cfg, "TEST", fr)
    check("BASE: F-near washes out L/R (|sizeL-sizeR| small)",
          abs(b["size_L"] - b["size_R"]) < 0.2)
    check("TEST: L/R difference preserved (larger than BASE)",
          abs(t["size_L"] - t["size_R"]) > abs(b["size_L"] - b["size_R"]) + 0.05)
    check("TEST: FL(400) nearer than FR(600) -> size_L > size_R", t["size_L"] > t["size_R"])

    # 2) FR-only / FL-only：BASE≈TEST（F 远，不主导）
    fr_only = {"FR": 400.0}
    fl_only = {"FL": 400.0}
    b2, t2 = probe(groups, cfg, "BASE", fr_only), probe(groups, cfg, "TEST", fr_only)
    check("FR-only: size_R BASE≈TEST", abs(b2["size_R"] - t2["size_R"]) < 0.05)
    b3, t3 = probe(groups, cfg, "BASE", fl_only), probe(groups, cfg, "TEST", fl_only)
    check("FL-only: size_L BASE≈TEST", abs(b3["size_L"] - t3["size_L"]) < 0.05)

    # 3) F-only：TEST 无侧向注入，front_safety 不变
    fonly = {"F": 300.0}
    bf, tf = probe(groups, cfg, "BASE", fonly), probe(groups, cfg, "TEST", fonly)
    check("F-only: BASE has lateral injection (chase_L/R > 0)",
          bf["chase_L"] > 0 and bf["chase_R"] > 0)
    check("F-only: TEST lateral injection << BASE (near-zero, residual from NO_TARGET)",
          tf["chase_L"] < 0.25 and (bf["chase_L"] - tf["chase_L"]) > 0.5)
    check("F-only: front mask identical BASE/TEST", bf["front"] == tf["front"])
    check("F-only: dmin_F identical BASE/TEST (front_safety unchanged)",
          abs(bf["dmin_F"] - tf["dmin_F"]) < 1e-9)

    # 4) 对称：两者都应左右对称
    sym = {"FL": 400.0, "FR": 400.0}
    bs, ts = probe(groups, cfg, "BASE", sym), probe(groups, cfg, "TEST", sym)
    check("symmetric: BASE L~R", abs(bs["size_L"] - bs["size_R"]) < 1e-6)
    check("symmetric: TEST L~R", abs(ts["size_L"] - ts["size_R"]) < 1e-6)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
