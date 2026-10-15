"""P2d-23 回归：接口审计的代码事实（感觉掩码/分组/动力学/包络），用假脑+纯常量。"""
from __future__ import annotations

import sys

import numpy as np

from brain.neurons import GROUPS
from config import Config, DecoderConfig, EncoderConfig
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.encode_tof import PhysicalToFEncoder
from loop.population_turn import GAIN, INDEX_CHECKSUM, L_IDX, R_IDX, SIGN, TAU
from loop.run import make_setup

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== interface_audit facts ===========")
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", dt=None, device="cpu", seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=0.40)
    left = [n for n, m in zip(enc.names, enc.left) if m]
    right = [n for n, m in zip(enc.names, enc.right) if m]
    check("left pool = [FL,F]", left == ["FL", "F"])
    check("right pool = [F,FR]", right == ["F", "FR"])
    check("center F enters BOTH L/R pools", "F" in left and "F" in right)
    check("L/R (70deg) NOT in the neural pools", "L" not in left and "R" not in right)

    check("EncoderConfig fov/cone", EncoderConfig().avoidance_fov_deg == 120.0
          and EncoderConfig().front_cone_deg == 40.0)
    check("GROUPS: turn=DNa02, fwd=DNg100, back=MDN, escape=DNp01",
          GROUPS["turn_L"][0] == ["DNa02"] and GROUPS["fwd"][0] == ["DNg100"]
          and GROUPS["back"][0] == ["MDN"] and GROUPS["escape"][0] == ["DNp01"])
    check("GROUPS: loom=LPLC2, threat=LC4, chase=LC10a",
          GROUPS["loom_L"][0] == ["LPLC2"] and GROUPS["threat_L"][0] == ["LC4"]
          and GROUPS["chase_L"][0] == ["LC10a"])

    # flybrain 类级动力学常量（实际安装版本）
    from flybrain import FlyBrain
    check("flybrain tonic/noise/gain/tau", FlyBrain.tonic == 0.14 and FlyBrain.noise_hz == 1.2
          and FlyBrain.gain == 3.0 and FlyBrain.tau == 0.100 and FlyBrain.noise_amp == 0.22)

    # P2Control 人工包络默认值
    d = DecoderConfig()
    check("P2Control envelope defaults", d.readout_fwd_frac == 0.75 and d.turn_slowdown == 0.5
          and d.smooth == 0.35 and d.front_safety is True and d.front_stop_dist == 0.25
          and d.front_slow_dist == 1.0)

    check("population frozen: |L|=|R|=50, ck matches", len(L_IDX) == 50 and len(R_IDX) == 50
          and GAIN == 8.0 and SIGN == -1.0 and TAU == 0.2
          and __import__("loop.population_turn", fromlist=["checksum"]).checksum(L_IDX, R_IDX)
          == INDEX_CHECKSUM)

    # 车状态不回喂：brain.step 参数只有 eye_drive / inject（无位姿）
    import inspect
    params = list(inspect.signature(brain.step).parameters)
    check("brain.step takes only (eye_drive, inject) — no pose/heading", params == ["eye_drive", "inject"])
    check("Config default DecoderConfig gain", Config().decoder.turn_gain == 0.8)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
