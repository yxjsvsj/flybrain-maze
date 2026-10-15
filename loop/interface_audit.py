"""P2d-23：MaleCNS 感觉—神经—运动接口审计（**只读**，不改任何东西、不控车）。

打印：感觉→编码→注入→网络的路径、神经元分组与 fallback、flybrain 实际动力学参数、
P2Control 人工包络、群体读出与 motor map，以及"车状态是否回喂神经网络"。
"""
from __future__ import annotations

import sys

import numpy as np

from brain.neurons import FALLBACK, GROUPS
from config import PRESETS
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.encode_tof import PhysicalToFEncoder
from loop.population_turn import GAIN, INDEX_CHECKSUM, L_IDX, R_IDX, SIGN, TAU
from loop.run import make_setup


def _fmt(mask, names):
    return [n for n, m in zip(names, mask) if m]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device="cuda", seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=0.40)
    print("[P2d-23] MaleCNS 接口审计（只读）  大脑:", type(brain._b).__module__ + "."
          + type(brain._b).__name__)
    print("\n== 一、感觉路径 ==")
    print("  ToF 顺序:", ORDER, " 角度:", {n: DEFAULT_ANGLES_DEG[n] for n in ORDER})
    print("  avoidance_fov_deg:", cfg.encoder.avoidance_fov_deg,
          " front_cone_deg:", cfg.encoder.front_cone_deg)
    print("  神经避障左池(left):", _fmt(enc.left, enc.names))
    print("  神经避障右池(right):", _fmt(enc.right, enc.names))
    print("  正前方(front, 仅 front_safety):", _fmt(enc.front, enc.names))
    print("  → 中心 F 同时进左右两池:", "F" in _fmt(enc.left, enc.names)
          and "F" in _fmt(enc.right, enc.names))
    print("  → L/R(±70°) 是否进神经池:", ("L" in _fmt(enc.left, enc.names))
          or ("R" in _fmt(enc.right, enc.names)), "(只用于建图 valid_mask)")
    print("  NO_TARGET -> sense():", "dist=max_range", " -> size=wall_size/max_range >0 -> 仍有非零注入")
    print("  TOO_NEAR  -> sense():", "dist=MIN_DIST_M -> 触发 front_safety")
    print("  注入公式: loom/chase/threat = cache d*per_step; fwd = forward_drive_dv*per_step")
    print("  per_step=1-exp(-dt/tau):", round(1 - np.exp(-brain.dt / 0.100), 5),
          " forward_drive_dv:", cfg.encoder.forward_drive_dv)
    print("\n== 二、神经元分组与 fallback ==")
    for name, (types, side) in GROUPS.items():
        idx = groups[name]
        fell = len(brain.cells(types, side=side)) == 0
        print(f"  {name:9s} <- {types[0]:8s} side={side}  n={len(idx):5d}  "
              f"{'** FALLBACK **' if fell else 'ok'}")
    print("  真实预设 chase_size_gain(LC10a静态proximity):", PRESETS['real']['enc']['chase_size_gain'],
          " loom_size_gain(LPLC2):", PRESETS['real']['enc']['loom_size_gain'],
          " forward_drive_dv:", PRESETS['real']['enc']['forward_drive_dv'])
    print("\n== 三、flybrain 实际动力学 ==")
    b = brain._b
    print(f"  dt={b.dt} tau={b.tau} gain={b.gain} tonic={b.tonic} noise_hz={b.noise_hz} "
          f"noise_amp={b.noise_amp} decay={b.decay} refractory_steps={b.refractory_steps} "
          f"sensory_input={b.sensory_input}")
    print("  每步: v <- decay*v + gain*synaptic(fired_prev) + tonic + noise + eye + inject; v>=1 -> spike")
    if b.superclass is not None:
        sens = int(np.sum(np.char.find(b.superclass.astype(str), 'sensory') >= 0))
        print("  sensory_input=False: 关掉" , sens, "个 sensory 神经元的**入突触**（W 行置0）；"
              "tonic/noise/recurrent 不变")
    print("\n== 四、运动输出路径 ==")
    print("  Decoder(dn 模式): turn = turn_gain*turn_sign*(r[turn_L]-r[turn_R]); v = fwd_gain*r[fwd]-...")
    print("  群体读出: D=mean(trace[L50])-mean(trace[R50]); turn = clip(sign*gain*(D-b0)), "
          f"gain={GAIN} sign={SIGN} tau={TAU}; |L|+|R|={len(L_IDX)+len(R_IDX)} ck={INDEX_CHECKSUM}")
    d = cfg.decoder
    print(f"  P2Control 人工包络: fwd_frac={d.readout_fwd_frac} turn_slowdown={d.turn_slowdown} "
          f"smooth={d.smooth} front_safety={d.front_safety} stop={d.front_stop_dist} "
          f"slow={d.front_slow_dist} max_omega={cfg.car.max_omega}")
    print("  → 神经只接 omega 分量; v 由固定前进速度+front_safety 决定; 平滑/限幅/降速均为人工")
    print("\n== 五、车状态是否回喂神经网络 ==")
    print("  brain.step(eye_drive=None, inject=...) 只吃外部注入; 位姿/朝向/转动状态**不进入**网络")
    print("  神经可见：5路ToF(经编码); 不可见：车位置、朝向、是否在转、地图、终点")
    print("  结论：无神经主导的运动记忆/方向保持/探索状态/目标选择 (至多是 trace 的短时残留)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
