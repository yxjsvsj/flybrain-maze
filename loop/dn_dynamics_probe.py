"""P2d-15b：固定群体的时间匹配对照复核（离线，不控车）。

在 P2d-15a 之上修正**统计口径**，不换群体、不改参数：
  - 群体**锁定** P2d-15a 在 screen seeds(64/101/202/303) 选出的 L50/R50，不重选、不改 K、
    不在 confirm seed 上调整。
  - 群体判据用**时间匹配对照**：净刺激效应 = D_stim(刺激) − D_stim(同时刻 AAA)；
    净残余 = D_post(刺激) − D_post(同时刻 AAA)。以"净效应是否衰减/接近无遮挡基线"为准，
    **不**要求原始 |D_post|<|D_stim|；不凭总体中位数下结论。
  - 单神经元恢复**分侧**统计：左/右各只用各自确有响应的神经元，报告样本数与幅度。
  - 实际运行 `alt` 的 A→B→A→C→A 及同分段全 A 对照，看能否换向、撤除后净效应是否减弱。

严格边界：不控车、不训练、不调参、不改 P1/MaleCNS 权重/正式 readout/感觉编码；
无 Dijkstra/老师/实体。DNa02/DNg100/MDN/DNp01 为生物学注释，其余为实验性推断。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.encode_tof import PhysicalToFEncoder
from loop.run import make_setup

STIMS = {"A": {}, "B": {"FR": 400.0}, "C": {"FL": 400.0}}
PLANS = {
    "left":  [("A", 1.2), ("C", 2.0), ("A", 1.8)],
    "right": [("A", 1.2), ("B", 2.0), ("A", 1.8)],
    "alt":   [("A", 1.0), ("B", 1.4), ("A", 1.2), ("C", 1.4), ("A", 1.4)],
    "alt_ctrl": [("A", 1.0), ("A", 1.4), ("A", 1.2), ("A", 1.4), ("A", 1.4)],
    "aaa":   [("A", 1.2), ("A", 2.0), ("A", 1.8)],
}
AGG = "descending_neuron"
K_GROUP = 50


def plan_segments(plan, dt):
    segs, acc = [], 0
    for name, dur in plan:
        m = int(round(dur / dt))
        segs.append((name, acc, acc + m))
        acc += m
    return segs


def _w(i0, i1, dt, win_s):
    w = max(1, int(round(win_s / dt)))
    return list(range(max(i0, i1 - w), i1))


def seg_means(D, plan, dt, win_s=0.5):
    return [float(np.mean(D[_w(i0, i1, dt, win_s)])) for _n, i0, i1 in plan_segments(plan, dt)]


def make_frame(stim, seq, t):
    ranges = {n: None for n in ORDER}
    status = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        ranges[n], status[n] = float(mm), "VALID"
    return TofFrame(session_id="dyn", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def run_plan(cfg, brain, groups, dn_idx, trace_cls, seed, plan, *, tof_hz=10.0, mpc=0.40):
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    trace = trace_cls(brain, idx=dn_idx, tau=0.2)
    brain.reset(seed)
    dt = brain.dt
    period = 1.0 / tof_hz
    frame, seq, last_t, total = None, 0, -1e9, 0
    rows = []
    for name, dur in plan:
        for _ in range(int(round(dur / dt))):
            tt = total * dt
            if frame is None or (tt - last_t) >= period - 1e-9:
                seq += 1
                frame = make_frame(STIMS[name], seq, tt)
                last_t = tt
            d = enc.sense(frame)
            enc.update(frame, d)
            inj, _info = enc.inject()
            fired = brain.step(inject=inj)
            rows.append(np.asarray(trace.observe(fired), np.float32))
            total += 1
    return np.stack(rows, 1) if rows else np.zeros((len(dn_idx), 0), np.float32)


def per_dn_metrics(act, plan, dt, *, k=3.0, win_s=0.5):
    """基线(pre-A)/刺激响应/撤除恢复/启动延迟(**仅刺激段内**)/回基线比。"""
    segs = plan_segments(plan, dt)
    if len(segs) < 3:
        return None
    _an, a0, a1 = segs[0]
    _sn, s0, s1 = segs[1]
    _pn, p0, p1 = segs[2]
    pre, stw, post = _w(a0, a1, dt, win_s), _w(s0, s1, dt, win_s), _w(p0, p1, dt, win_s)
    base = act[:, pre].mean(1)
    bs = act[:, pre].std(1) + 1e-6
    resp = act[:, stw].mean(1)
    rec = act[:, post].mean(1)
    onset = np.full(act.shape[0], -1.0)
    thr = k * bs
    for j in range(act.shape[0]):
        dev = np.abs(act[j] - base[j])
        for i in range(s0, s1):
            if dev[i] > thr[j]:
                onset[j] = (i - s0) * dt
                break
    dresp = resp - base
    return {"base": base, "resp": resp, "rec": rec, "dresp": dresp,
            "ret": np.abs(rec - base) / (np.abs(dresp) + 1e-6), "onset": onset}


def pop_diff(act, L, R):
    return act[L].mean(0) - act[R].mean(0)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-seeds", default="64,101,202,303")
    ap.add_argument("--confirm-seeds", default="512,777,999")
    ap.add_argument("--brain", default="real")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    confirm = [int(s) for s in args.confirm_seeds.split(",")]

    from flybrain import Trace
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells([AGG]))
    dt = brain.dt
    n = len(dn_idx)
    print(f"[P2d-15b] DNs={n}  10Hz ToF / {1/dt:.0f}Hz  screen={screen} confirm={confirm}")

    def metrics(seeds, key):
        return {sd: per_dn_metrics(run_plan(cfg, brain, groups, dn_idx, Trace, sd, PLANS[key]),
                                   PLANS[key], dt) for sd in seeds}
    ml, mr = metrics(screen, "left"), metrics(screen, "right")

    # ---- 1) 单细胞：分侧响应集合 + 各侧恢复 ----
    def ok_dir_onset(sd):
        a, b = ml[sd], mr[sd]
        return ((np.abs(a["dresp"]) > 0.05) & (np.abs(b["dresp"]) > 0.05)
                & (a["onset"] >= 0) & (b["onset"] >= 0)
                & (np.sign(a["dresp"]) == -np.sign(b["dresp"])))
    md = np.all([ok_dir_onset(sd) for sd in screen], 0)
    rl = np.mean([np.abs(ml[sd]["dresp"]) for sd in screen], 0) > 0.1
    rr = np.mean([np.abs(mr[sd]["dresp"]) for sd in screen], 0) > 0.1
    print(f"\n[单细胞] 方向+启动(全screen): {int(md.sum())}/{n}")

    def side_report(tag, mask, M):
        rets = np.array([M[sd]["ret"][mask] for sd in screen])
        print(f"[单细胞·{tag}] 自身响应神经元 {int(mask.sum())} 个；"
              f"基线 {np.mean([M[sd]['base'][mask].mean() for sd in screen]):.3f}  "
              f"响应 {np.mean([M[sd]['resp'][mask].mean() for sd in screen]):.3f}  "
              f"撤除后 {np.mean([M[sd]['rec'][mask].mean() for sd in screen]):.3f}  "
              f"回基线比中位 {np.median(rets):.2f} (<0.5 占 {np.mean(rets.mean(0) < 0.5) * 100:.0f}%)")
    side_report("左(FL)", rl, ml)
    side_report("右(FR)", rr, mr)

    # ---- 2) 群体（锁定 L50/R50）+ 时间匹配净效应 ----
    idx = np.mean([ml[sd]["dresp"] - mr[sd]["dresp"] for sd in screen], 0)
    order = np.argsort(-np.abs(idx))
    L = [i for i in order if idx[i] > 0][:K_GROUP]
    R = [i for i in order if idx[i] < 0][:K_GROUP]
    print(f"\n[群体] 锁定 15a 规则(不重选): screen 平均(左-右) top-{K_GROUP} 按符号 → "
          f"L组 {len(L)}, R组 {len(R)}")

    def dseg(seed, key):
        return seg_means(pop_diff(run_plan(cfg, brain, groups, dn_idx, Trace, seed, PLANS[key]), L, R),
                         PLANS[key], dt)

    print(f"\n{'seed':>6s} {'侧':>4s} {'pre':>8s} {'stim':>8s} {'post':>8s} | "
          f"{'netStim':>8s} {'netPost':>8s} {'衰减':>5s}")
    recs = []
    for sd in confirm:
        da = dseg(sd, "aaa")
        for tag, key in (("左", "left"), ("右", "right")):
            d = dseg(sd, key)
            ns, npo = d[1] - da[1], d[2] - da[2]
            dec = abs(npo) < abs(ns)
            recs.append((sd, tag, ns, npo, dec))
            print(f"{sd:6d} {tag:>4s} {d[0]:+8.3f} {d[1]:+8.3f} {d[2]:+8.3f} | "
                  f"{ns:+8.3f} {npo:+8.3f} {str(dec):>5s}")
    nL = [r for r in recs if r[1] == "左"]
    nR = [r for r in recs if r[1] == "右"]
    a_ok = all(r[2] > 0 for r in nL) and all(r[2] < 0 for r in nR)
    b_ok = all(r[4] for r in recs)
    print(f"[群体·净] 左净刺激全正: {all(r[2] > 0 for r in nL)}；右净刺激全负: "
          f"{all(r[2] < 0 for r in nR)}；撤除后净残余|·|下降 {sum(r[4] for r in recs)}/{len(recs)}")

    # ---- 3) alt 连续序列 + 同分段全 A 对照 ----
    print("\n[群体·alt] A→B→A→C→A（同分段全 A 对照）")
    print(f"{'seed':>6s} {'段':>5s} {'刺激':>8s} {'对照':>8s} {'净':>8s}")
    alt_ok = []
    for sd in confirm:
        da = dseg(sd, "alt_ctrl")
        d = dseg(sd, "alt")
        names = ["A0", "B", "A1", "C", "A2"]
        net = [d[i] - da[i] for i in range(5)]
        for i, nm in enumerate(names):
            print(f"{sd:6d} {nm:>5s} {d[i]:+8.3f} {da[i]:+8.3f} {net[i]:+8.3f}")
        ok = (net[1] < 0 < net[3]) and abs(net[2]) < abs(net[1]) and abs(net[4]) < abs(net[3])
        alt_ok.append(ok)
    print(f"[群体·alt] 可由右(B) 换向左(C) 且撤除后净效应减弱 全seed: {all(alt_ok)}")

    # ---- 结论边界 ----
    print("\n[结论]")
    print(f"  (a) 独立 confirm 可重复的左右方向信号: {'支持' if a_ok else '不支持'}")
    print(f"  (b) 该信号撤除后相对无遮挡对照衰减: {'支持' if b_ok else '不支持'}")
    print(f"  (c) 是否足以完成小车转向闭环: 本轮不控车，**不能证明**")
    v = "PASS" if (a_ok and b_ok) else "FAIL"
    print(f"[GATE1b {v}] " + ("存在值得验证的群体运动候选（等待批准最小闭环试验，不自动进入）"
          if v == "PASS" else "停止群体搜索，转向 MaleCNS 神经动力学/感觉—运动接口建模审查"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
