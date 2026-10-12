"""P2d-13a：修复后的下行神经元群体探针（独立试次，静态受控 5 路 ToF）。

修复 P2d-13 原探针（见 dn_population_probe_p2d13_buggy.py）的三个错误：
  1. **独立试次**：每个 (brain_seed, stimulus) 都 brain.reset(seed) 并重建 Encoder/Decoder/Trace，
     所有条件拥有相同初始化、时长、分析窗口（不再跨条件复用状态）。
  2. **新采样帧**：ToF 以 ~10Hz 递增 seq / 单调时间戳呈现（静态障碍距离恒定），脑跑 50Hz，
     帧间 ZOH；`frames_consumed ≈ seconds*tof_hz`。
  3. **有效性门槛**：FR-only 必须改变右侧 size/chase，FL-only 改变左侧，A 与二者可分；
     触发不稳时**报错**而非继续下结论。不把 DNg100 的恒定注入当作"FR/FL 输入成功"的证据。

分析：每 seed 做**配对** B−A / C−A / C−B（不只比绝对 turn_L−turn_R），并用独立 A→A 试次
（A1/A2）测时间漂移作为显著性基线。报告零响应比例、差分幅度、跨 seed 变异。

注释：DNa02(转向)/DNg100(前进)/MDN(后退)/DNp01(逃逸) 为**生物学注释**；其余差分/符号均为
**实验性推断**，不据此命名"左转/右转神经元"，不冒充果蝇天然运动解码。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup

STIMULI = {
    "A_none": {},
    "B_FR":   {"FR": 400.0},
    "C_FL":   {"FL": 400.0},
    "D_F":    {"F": 400.0},
    "E_sym":  {"FL": 400.0, "FR": 400.0},
    "A2_none": {},                        # 独立 A 重复 -> 漂移对照
}
GROUPS = ("turn_L", "turn_R", "fwd", "back", "escape")
SIZE_KEYS = ("size_L", "size_R", "growth_L", "growth_R", "dmin_L", "dmin_R", "dmin_F",
             "prox_l", "prox_r")
TOL = 0.02


def make_frame(stim, seq, t):
    ranges = {n: None for n in ORDER}
    status = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        ranges[n], status[n] = float(mm), "VALID"
    return TofFrame(session_id="probe", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 1000.0 / 10.0 for n in ORDER}, init_count=0, read_errors=0,
                    reinit_count=0)


def run_trial(cfg, brain, groups, dn_idx, trace_cls, stim, *, seed, seconds=3.0, settle=1.5,
              tof_hz=10.0, mpc=0.40, enc_cls=PhysicalToFEncoder):
    """一次独立试次：全新 Encoder/Decoder/Trace + brain.reset(seed)。"""
    enc = enc_cls(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                  sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                  car_radius=cfg.car.radius, meters_per_cell=mpc)
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    trace = trace_cls(brain, idx=dn_idx, tau=0.2)
    brain.reset(seed)

    dt = brain.dt
    n = int(round(seconds / dt))
    k0 = int(round(settle / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    rec = {g: [] for g in GROUPS}
    acts, infos = [], []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = make_frame(stim, seq, t)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, info = enc.inject()
        fired = brain.step(inject=inj)
        dec.update(fired, info)
        act = trace.observe(fired)
        if k >= k0:
            for g in GROUPS:
                rec[g].append(float(dec.rates[dec.index[g]]))
            acts.append(np.asarray(act, np.float32))
            infos.append(info)
    rates = {g: float(np.mean(v)) if v else 0.0 for g, v in rec.items()}
    act = np.mean(np.asarray(acts, np.float32), 0) if acts else np.zeros(0, np.float32)
    info_ss = {k: float(np.mean([i[k] for i in infos])) for k in SIZE_KEYS} if infos else {}
    return {"rates": rates, "act": act, "info": info_ss,
            "frames_consumed": int(enc.frames_consumed),
            "expected_frames": max(1, int(round(seconds * tof_hz)))}


def sensory_gate(trials, seeds):
    """有效性门槛：FR 改右、FL 改左、A 可分、独立试次取到足量新帧。失败即 raise。"""
    msgs = []
    for sd in seeds:
        a = trials[(sd, "A_none")]
        if a["frames_consumed"] < a["expected_frames"] - 2:
            msgs.append(f"seed{sd} A 仅消费 {a['frames_consumed']} 帧 < {a['expected_frames']}")
        b = trials[(sd, "B_FR")]
        c = trials[(sd, "C_FL")]
        if not (b["info"]["size_R"] - a["info"]["size_R"] > TOL):
            msgs.append(f"seed{sd} FR-only 未改变 size_R "
                        f"({a['info']['size_R']:.3f}->{b['info']['size_R']:.3f})")
        if not (c["info"]["size_L"] - a["info"]["size_L"] > TOL):
            msgs.append(f"seed{sd} FL-only 未改变 size_L "
                        f"({a['info']['size_L']:.3f}->{c['info']['size_L']:.3f})")
        if abs(b["info"]["size_R"] - b["info"]["size_L"]) < TOL:
            msgs.append(f"seed{sd} FR-only 左右不可分 (size_R≈size_L)")
    if msgs:
        raise SystemExit("[probe] 感觉有效性门槛失败（探针无效，拒绝下结论）:\n  "
                         + "\n  ".join(msgs))
    return True


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="64,101,202,303")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--tof-hz", type=float, default=10.0)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    from flybrain import Trace
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    print(f"[probe] DNs={len(dn_idx)} seeds={seeds} seconds={args.seconds} "
          f"tof={args.tof_hz}Hz（独立试次）")
    print("[annot] 注释: turn=DNa02(L/R) fwd=DNg100 back=MDN escape=DNp01；"
          "diff/符号均为实验性推断\n")

    trials = {}
    for sd in seeds:
        for name, stim in STIMULI.items():
            trials[(sd, name)] = run_trial(
                cfg, brain, groups, dn_idx, Trace, stim, seed=sd,
                seconds=args.seconds, settle=args.settle, tof_hz=args.tof_hz)
    sensory_gate(trials, seeds)
    print("[gate] 感觉有效性通过：FR->右、FL->左、A 可分；新帧数达标")
    print("[gate] frames_consumed:", " ".join(
        f"{sd}:{trials[(sd, 'A_none')]['frames_consumed']}/{trials[(sd, 'A_none')]['expected_frames']}"
        for sd in seeds), "\n")

    # 表1：注释组稳态发放率
    print("表1 注释下行组稳态发放率(Hz)  [独立试次]")
    print(f"{'seed':>5s} {'stim':7s} {'turn_L':>7s} {'turn_R':>7s} {'diff(L-R)':>9s} "
          f"{'fwd':>6s} {'back':>6s} {'escape':>7s}")
    for sd in seeds:
        for name in STIMULI:
            r = trials[(sd, name)]["rates"]
            print(f"{sd:5d} {name:7s} {r['turn_L']:7.3f} {r['turn_R']:7.3f} "
                  f"{r['turn_L'] - r['turn_R']:9.3f} {r['fwd']:6.3f} {r['back']:6.3f} "
                  f"{r['escape']:7.3f}")

    # 表2：配对差分（DNa02 的 turn_L / turn_R / diff），含 A->A 漂移对照
    def dlr(sd, a, b):
        ra, rb = trials[(sd, a)]["rates"], trials[(sd, b)]["rates"]
        return (rb["turn_L"] - ra["turn_L"], rb["turn_R"] - ra["turn_R"],
                (rb["turn_L"] - rb["turn_R"]) - (ra["turn_L"] - ra["turn_R"]))
    print("\n表2 配对差分 [目标-参照]（DNa02）")
    print(f"{'pair':9s} " + " ".join(f"{'dL':>7s}{'dR':>7s}{'ddiff':>8s}" for _ in seeds))
    for label, ref, tgt in (("B-A", "A_none", "B_FR"), ("C-A", "A_none", "C_FL"),
                            ("C-B", "B_FR", "C_FL"), ("A-A(drift)", "A_none", "A2_none")):
        cells = []
        for sd in seeds:
            cells.append("%+7.3f%+7.3f%+8.3f" % dlr(sd, ref, tgt))
        print(f"{label:9s} " + " ".join(cells))
    print("\n表2b 配对 diff(dturn_L-turn_R) 跨 seed 一致性")
    for label, ref, tgt in (("B-A", "A_none", "B_FR"), ("C-A", "A_none", "C_FL"),
                            ("C-B", "B_FR", "C_FL"), ("A-A(drift)", "A_none", "A2_none")):
        ds = [dlr(sd, ref, tgt)[2] for sd in seeds]
        signs = [np.sign(d) for d in ds]
        cons = all(s == signs[0] for s in signs) and signs[0] != 0
        print(f"  {label:9s} " + " ".join(f"{d:+.3f}" for d in ds)
              + f"   mean{np.mean(ds):+.3f}  std{np.std(ds):.3f}  同号={cons}")

    # 表3：群体配对差分 C-B 的显著性（对照 A-A 漂移）
    print("\n表3 群体（全部下行神经元）配对差分 C_FL-B_FR 显著性")
    dmat = np.stack([trials[(sd, "C_FL")]["act"] - trials[(sd, "B_FR")]["act"] for sd in seeds], 0)
    drift = np.stack([np.abs(trials[(sd, "A2_none")]["act"] - trials[(sd, "A_none")]["act"])
                      for sd in seeds], 0)
    thr = float(np.percentile(drift, 99)) if drift.size else 0.0
    mean_cb, std_cb = dmat.mean(0), dmat.std(0)
    mag = np.abs(mean_cb)
    zero_ratio = float(np.mean(mag < thr)) if mag.size else 1.0
    consistent = np.all(np.sign(dmat) == np.sign(dmat[0]), axis=0) & (mag > thr) & (std_cb < mag)
    print(f"  显著阈值(来自 A->A 漂移的 99 分位) thr={thr:.4f}")
    print(f"  零响应比例 |mean|<thr: {zero_ratio * 100:.1f}%   最大|mean|={mag.max():.4f}  "
          f"最大跨seed std={std_cb.max():.4f}")
    print(f"  跨 seed 同号且过阈值且 std<|mean| 的神经元: {int(consistent.sum())}/{len(dn_idx)}")
    if consistent.any():
        order = np.argsort(-mag)
        print(f"  {'dn_i':>6s} {'mean(C-B)':>10s} {'std':>8s}")
        for i in order:
            if consistent[i]:
                print(f"  {int(dn_idx[i]):6d} {mean_cb[i]:+10.4f} {std_cb[i]:8.4f}")
                if sum(consistent[order[:list(order).index(i) + 1]]) >= 10:
                    break

    # 表4：对称/正前是否有可重复方向证据（对照漂移）
    print("\n表4 对称 E / 正前 D 的方向证据（用 A->A 漂移判显著）")
    for name in ("D_F", "E_sym"):
        ds = [dlr(sd, "A_none", name)[2] for sd in seeds]
        print(f"  {name:7s} DNa02 diff: " + " ".join(f"{d:+.3f}" for d in ds)
              + f"  mean{np.mean(ds):+.3f}±{np.std(ds):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
