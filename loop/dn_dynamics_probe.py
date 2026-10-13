"""P2d-15a：群体动态信号判据复核（离线，不控车；不改 MaleCNS/感觉编码/P1）。

修复并复核 P2d-15 第一关的判据：
  1. per_dn_metrics 的"启动延迟"**只在刺激存在的时间段内搜索**（不扫撤除后的数据）。
  2. 单细胞结果与**群体级**结果**分开报告**，不用单细胞否定群体。
  3. 检查左右两个不同的下行神经元群体能否共同形成**稳定方向差分**，并看**撤除后差分是否下降**。
  4. 运行 `alt` 连续序列，并加等长的 **A→A→A** 无障碍对照识别自发漂移。
  5. 回基线统计**只针对确有刺激响应的神经元**，逐 seed 展示响应幅度与撤除后变化。
  6. screen seed 64/101/202/303；confirm seed 512/777/999（不同批）。群体构造只用
     **事先声明的规则**（按 screen 平均方向指数取 top-K 分左右组），不在确认集上改规则。

披露：DNa02/DNg100/MDN/DNp01 为生物学注释；Trace(τ=0.2) 读出；其余为实验性推断，
不冒充天然运动解码。本脚本**不控制车**。
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
    "aaa":   [("A", 1.2), ("A", 2.0), ("A", 1.8)],      # 无障碍对照（自发漂移）
}
AGG = "descending_neuron"
K_GROUP = 50          # 群体构造：按 screen 方向指数取 |idx| 前 K 个，按符号分左右组（预先声明）


def make_frame(stim, seq, t):
    ranges = {n: None for n in ORDER}
    status = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        ranges[n], status[n] = float(mm), "VALID"
    return TofFrame(session_id="dyn", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def plan_segments(plan, dt):
    segs, acc = [], 0
    for name, dur in plan:
        m = int(round(dur / dt))
        segs.append((name, acc, acc + m))
        acc += m
    return segs


def run_plan(cfg, brain, groups, dn_idx, trace_cls, seed, plan, *, tof_hz=10.0, mpc=0.40):
    """连续序列；返回每个下行神经元的痕迹活动 (n_dn, n_steps)。"""
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


def _w(i0, i1, dt, win_s):
    w = max(1, int(round(win_s / dt)))
    return list(range(max(i0, i1 - w), i1))


def per_dn_metrics(act, plan, dt, *, k=3.0, win_s=0.5):
    """基线(pre-A)/刺激响应/撤除恢复/启动延迟(**只在刺激段内搜索**)/回基线比。"""
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
        for i in range(s0, s1):                        # 仅刺激段内
            if dev[i] > thr[j]:
                onset[j] = (i - s0) * dt
                break
    dresp = resp - base
    return {"base": base, "resp": resp, "rec": rec, "dresp": dresp,
            "ret": np.abs(rec - base) / (np.abs(dresp) + 1e-6), "onset": onset,
            "segs": (a0, a1, s0, s1, p0, p1)}


def pop_diff(act, L, R, plan, dt):
    """群体方向差分 D(t)=mean(L组)-mean(R组)。返回 (t, D, segs, dt)。"""
    segs = plan_segments(plan, dt)
    d = act[L].mean(0) - act[R].mean(0)
    return np.arange(act.shape[1]) * dt, d, segs


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
    print(f"[P2d-15a] DNs={n}  10Hz ToF / {1/dt:.0f}Hz  screen={screen} confirm={confirm}")

    def metrics(seeds, key):
        return {sd: per_dn_metrics(run_plan(cfg, brain, groups, dn_idx, Trace, sd, PLANS[key]),
                                   PLANS[key], dt) for sd in seeds}
    ml, mr = metrics(screen, "left"), metrics(screen, "right")

    # ---- 1) 单细胞（分开报告）----
    def ok_dir_onset(sd):
        a, b = ml[sd], mr[sd]
        return ((np.abs(a["dresp"]) > 0.05) & (np.abs(b["dresp"]) > 0.05)
                & (a["onset"] >= 0) & (b["onset"] >= 0)
                & (np.sign(a["dresp"]) == -np.sign(b["dresp"])))
    dir_onset = np.all([ok_dir_onset(sd) for sd in screen], 0)
    ret_only = np.all([(ml[sd]["ret"] < 0.5) & (mr[sd]["ret"] < 0.5) for sd in screen], 0)
    print(f"\n[单细胞] 方向+启动(全 screen): {int(dir_onset.sum())}/{n}  "
          f"仅撤除回基线(全 screen): {int(ret_only.sum())}/{n}  "
          f"两者皆有: {int((dir_onset & ret_only).sum())}/{n}")

    # ---- 2) 响应神经元的回基线（只挑真有响应的）----
    mean_resp = np.mean([np.abs(ml[sd]["dresp"]) for sd in screen], 0)
    resp_dn = mean_resp > 0.1
    rets = np.array([ml[sd]["ret"][resp_dn] for sd in screen])       # (nseed, n_resp)
    rets_r = np.array([mr[sd]["ret"][resp_dn] for sd in screen])
    print(f"[响应神经元 {int(resp_dn.sum())} 个] 撤除回基线比 |rec-base|/|resp-base|："
          f"左 中位 {np.median(rets):.2f}, <0.5 占 {np.mean(rets.mean(0) < 0.5) * 100:.0f}%；"
          f"右 中位 {np.median(rets_r):.2f}, <0.5 占 {np.mean(rets_r.mean(0) < 0.5) * 100:.0f}%")
    for sd in screen:
        print(f"    seed {sd:3d}: 左响应均值 {np.mean(np.abs(ml[sd]['dresp'][resp_dn])):.3f} "
              f"撤除后|rec-base|均值 {np.mean(np.abs(ml[sd]['rec'][resp_dn] - ml[sd]['base'][resp_dn])):.3f}")

    # ---- 3) 群体方向差分（预先声明的规则：screen 平均方向指数 top-K 分左右组）----
    idx = np.mean([ml[sd]["dresp"] - mr[sd]["dresp"] for sd in screen], 0)
    order = np.argsort(-np.abs(idx))
    L = [i for i in order if idx[i] > 0][:K_GROUP]
    R = [i for i in order if idx[i] < 0][:K_GROUP]
    print(f"\n[群体] 规则=screen 平均(左响应-右响应) top-{K_GROUP} 按符号分组 → "
          f"L组 {len(L)} 个, R组 {len(R)} 个 (idx L均值 {idx[L].mean():+.3f}, R均值 {idx[R].mean():+.3f})")

    def D_at(seed, key):
        act = run_plan(cfg, brain, groups, dn_idx, Trace, seed, PLANS[key])
        _t, d, segs = pop_diff(act, L, R, PLANS[key], dt)
        _an, a0, a1 = segs[0]
        _sn, s0, s1 = segs[1]
        _pn, p0, p1 = segs[2]
        g = lambda win: float(np.mean(win))                          # noqa: E731
        return {"pre": g(d[_w(a0, a1, dt, 0.5)]), "stim": g(d[_w(s0, s1, dt, 0.5)]),
                "post": g(d[_w(p0, p1, dt, 0.5)])}

    print(f"\n{'seed':>6s} {'plan':6s} {'D_pre':>8s} {'D_stim':>8s} {'D_post':>8s} {'drop?':>6s}")
    for sd in confirm:
        dl, dr, da = D_at(sd, "left"), D_at(sd, "right"), D_at(sd, "aaa")
        drop_l = abs(dl["post"]) < abs(dl["stim"]) - 1e-9
        drop_r = abs(dr["post"]) < abs(dr["stim"]) - 1e-9
        print(f"{sd:6d} {'left':6s} {dl['pre']:+8.3f} {dl['stim']:+8.3f} {dl['post']:+8.3f} "
              f"{str(drop_l):>6s}")
        print(f"{sd:6d} {'right':6s} {dr['pre']:+8.3f} {dr['stim']:+8.3f} {dr['post']:+8.3f} "
              f"{str(drop_r):>6s}")
        print(f"{sd:6d} {'AAA':6s} {da['pre']:+8.3f} {da['stim']:+8.3f} {da['post']:+8.3f} "
              f"{'(drift)':>6s}")
    # 跨 seed 方向一致性 + 差分下降
    aL = [D_at(sd, "left") for sd in confirm]
    aR = [D_at(sd, "right") for sd in confirm]
    aA = [D_at(sd, "aaa") for sd in confirm]
    sL = [np.sign(x["stim"]) for x in aL]
    sR = [np.sign(x["stim"]) for x in aR]
    drift = np.mean([abs(x["stim"]) for x in aA])
    okL = all(s > 0 for s in sL)
    okR = all(s < 0 for s in sR)
    drop_all = all(abs(x["post"]) < abs(x["stim"]) for x in aL + aR)
    print(f"\n[群体·confirm] 左刺激 D>0 全 seed: {okL}；右刺激 D<0 全 seed: {okR}；"
          f"撤除后|D|下降 全seed: {drop_all}")
    print(f"[群体·confirm] 自发漂移 |D_stim(AAA)| 均值 {drift:.3f} vs 刺激 |D_stim| 均值 "
          f"{np.mean([abs(x['stim']) for x in aL + aR]):.3f}")
    pop_pass = okL and okR and drop_all and (np.mean([abs(x["stim"]) for x in aL + aR]) > 2 * drift)
    print(f"\n[GATE1a {'PASS' if pop_pass else 'FAIL'}] "
          + ("群体级方向差分稳定、移除后下降且高于自发漂移 → 可讨论最小运动解码"
             if pop_pass else
             "群体级方向差分未通过独立确认 → 报告结果并停止，转向神经动力学/接口建模讨论"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
