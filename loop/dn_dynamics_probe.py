"""P2d-15 第一关：MaleCNS 多下行神经元的转向启动/回正动态信号研究（离线，不控车）。

保持 BASE 五路感觉编码、MaleCNS 权重、10Hz ToF / 50Hz 脑步不变。连续时间序列
 A(无遮挡) -> 左/右前障碍 -> A(无遮挡)，脑连续运行、切换不重置；记录**每个下行神经元**的
脉冲痕迹活动，分析：启动延迟、方向差异、持续时间、撤除后回基线能力、跨 seed 可重复性。
重点寻找同时提供"开始转弯"(onset) 与"结束转弯/回正"(offset/return) 的真实神经活动，
不只按静态左右分类准确率选神经元。

候选筛选用 screen seeds；**确认用不同的 confirm seeds**（不得同批 seed 既筛又验）。
披露：DNa02(转向)/DNg100(前进)/MDN(后退)/DNp01(逃逸) 是**生物学注释**；其余差分/符号均为
**实验性推断**，不冒充果蝇天然运动解码；本关不控制车。

用法： python -m loop.dn_dynamics_probe --screen-seeds 64,101,202,303 --confirm-seeds 512,777,999
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
# 每条计划：(说明, 切换刺激, [(段名, 时长s), ...])
PLANS = {
    "left":  [("A", 1.2), ("C", 2.0), ("A", 1.8)],
    "right": [("A", 1.2), ("B", 2.0), ("A", 1.8)],
    "alt":   [("A", 1.0), ("B", 1.4), ("A", 1.2), ("C", 1.4), ("A", 1.4)],
}
AGG = "descending_neuron"


def make_frame(stim, seq, t):
    ranges = {n: None for n in ORDER}
    status = {n: "NO_TARGET" for n in ORDER}
    for n, mm in stim.items():
        ranges[n], status[n] = float(mm), "VALID"
    return TofFrame(session_id="dyn", seq=seq, t=t, healthy=True, ranges=ranges, status=status,
                    ages_ms={n: 100.0 for n in ORDER}, init_count=0, read_errors=0, reinit_count=0)


def run_plan(cfg, brain, groups, dn_idx, trace_cls, seed, plan, *, tof_hz=10.0, mpc=0.40):
    """连续 A->stim->A 时间序列；返回每个下行神经元的痕迹活动 (n_dn, n_steps) 与段边界。"""
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    trace = trace_cls(brain, idx=dn_idx, tau=0.2)
    brain.reset(seed)
    dt = brain.dt
    period = 1.0 / tof_hz
    frame, seq, last_t, total = None, 0, -1e9, 0
    rows, labels = [], []
    for name, dur in plan:
        m = int(round(dur / dt))
        for _ in range(m):
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
            labels.append(name)
            total += 1
    return np.stack(rows, 1) if rows else np.zeros((len(dn_idx), 0), np.float32), labels


def plan_segments(plan, dt):
    """返回 [(name, i0, i1), ...]（按计划时长，确定边界）。"""
    segs, acc = [], 0
    for name, dur in plan:
        m = int(round(dur / dt))
        segs.append((name, acc, acc + m))
        acc += m
    return segs


def per_dn_metrics(act, labels, plan, dt, *, k=3.0, win_s=0.5):
    """每个下行神经元：基线(pre-A) / 刺激响应 / 撤除后(post-A) / 启动延迟 / 回基线比。"""
    segs = plan_segments(plan, dt)
    if len(segs) < 3:
        return None
    _an, a0, a1 = segs[0]
    _sn, s0, s1 = segs[1]
    _pn, p0, p1 = segs[2]
    w = max(1, int(round(win_s / dt)))
    pre = list(range(max(a0, a1 - w), a1))
    stw = list(range(max(s0, s1 - w), s1))
    post = list(range(max(p0, p1 - w), p1))
    base = act[:, pre].mean(1)
    bs = act[:, pre].std(1) + 1e-6
    resp = act[:, stw].mean(1)
    rec = act[:, post].mean(1)
    sw = s0
    onset = np.full(act.shape[0], -1.0)
    thr = k * bs
    for j in range(act.shape[0]):
        dev = np.abs(act[j] - base[j])
        for i in range(sw, len(labels)):
            if dev[i] > thr[j]:
                onset[j] = (i - sw) * dt
                break
    dresp = resp - base
    ret = np.abs(rec - base) / (np.abs(dresp) + 1e-6)
    return {"base": base, "resp": resp, "rec": rec, "dresp": dresp, "ret": ret, "onset": onset}


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
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args(argv)
    screen = [int(s) for s in args.screen_seeds.split(",")]
    confirm = [int(s) for s in args.confirm_seeds.split(",")]

    from flybrain import Trace
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind=args.brain, map_name="gen", dt=None, device=args.device, seed=screen[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells([AGG]))
    dt = brain.dt
    print(f"[P2d-15 gate1] DNs={len(dn_idx)}  10Hz ToF / {1/dt:.0f}Hz brain  "
          f"screen={screen} confirm={confirm}")
    print("[annot] DNa02/DNg100/MDN/DNp01 为生物学注释；其余为实验性推断。本关不控车。\n")

    def run_seeds(seeds):
        out = {}
        for sd in seeds:
            for key in ("left", "right"):
                act, labels = run_plan(cfg, brain, groups, dn_idx, Trace, sd, PLANS[key])
                out[(sd, key)] = per_dn_metrics(act, labels, PLANS[key], dt)
        return out

    sc = run_seeds(screen)
    # 筛选：左/右都显著、方向相反、撤除后回基线、启动延迟有界
    n_dn = len(dn_idx)
    def sel(key, m):
        return (np.abs(m["dresp"]) > 0.05) & (m["onset"] >= 0) & (m["onset"] < 1.5)
    ok = np.ones(n_dn, bool)
    for sd in screen:
        ml, mr = sc[(sd, "left")], sc[(sd, "right")]
        ok &= sel("left", ml) & sel("right", mr)
        ok &= np.sign(ml["dresp"]) == -np.sign(mr["dresp"])     # 左右反号
        ok &= ml["ret"] < 0.5                                    # 撤除后回基线
    cand = np.nonzero(ok)[0]
    print(f"[screen] 通过筛选的下行神经元: {len(cand)}/{n_dn}  "
          f"(左右反号 + 启动<1.5s + 撤除回基线, 全部 screen seed)")
    # 分项拆解：定位瓶颈
    ok_do = np.ones(n_dn, bool)          # 仅 方向 + 启动
    ok_ret = np.ones(n_dn, bool)         # 仅 撤除回基线
    for sd in screen:
        ml, mr = sc[(sd, "left")], sc[(sd, "right")]
        ok_do &= sel("left", ml) & sel("right", mr) & (np.sign(ml["dresp"]) == -np.sign(mr["dresp"]))
        ok_ret &= (ml["ret"] < 0.5) & (mr["ret"] < 0.5)
    print(f"[screen] 仅方向+启动(不看回正): {int(ok_do.sum())}/{n_dn}  "
          f"仅撤除回基线(不看方向): {int(ok_ret.sum())}/{n_dn}")
    # 泄漏自查：仅左右反应强烈但未过筛的
    strong = np.nonzero(np.abs(sc[(screen[0], 'left')]["dresp"]) > 0.2)[0]
    print(f"[screen] |左响应|>0.2 的神经元 {len(strong)}；其中通过定向+回基线{len(np.intersect1d(cand, strong))}")

    # 确认：在**不同** seed 上复算同一批候选
    cf = run_seeds(confirm)
    def conf_ok(i):
        good = 0
        for sd in confirm:
            ml, mr = cf[(sd, "left")], cf[(sd, "right")]
            if (abs(ml["dresp"][i]) > 0.05 and ml["onset"][i] >= 0 and ml["onset"][i] < 1.5
                    and np.sign(ml["dresp"][i]) == -np.sign(mr["dresp"][i]) and ml["ret"][i] < 0.5):
                good += 1
        return good
    print(f"\n[confirm] 候选在 confirm seed 上逐条复算（需每个 seed 都满足）：")
    confirmed = []
    for i in cand[:40]:
        g = conf_ok(i)
        flag = "OK" if g == len(confirm) else f"{g}/{len(confirm)}"
        confirmed.append(g == len(confirm))
        print(f"  dn_idx={int(dn_idx[i]):7d}  confirm {flag}")
    n_conf = int(np.sum(confirmed))
    print(f"\n[gate1] 筛选候选 {len(cand)}，在**独立 confirm seed** 上全通过的：{n_conf}")

    # 回正信号：撤除后"回落到基线"的能力分布（全群）
    rets = np.array([sc[(sd, "left")]["ret"] for sd in screen])       # (nseed, n_dn)
    print(f"[gate1] 撤除后回基线比例 |rec-base|/|resp-base|：中位 {np.median(rets):.2f}，"
          f"<0.5 的神经元占 {np.mean(rets.mean(0) < 0.5) * 100:.0f}%")
    verdict = "PASS" if n_conf >= 1 else "FAIL"
    print(f"\n[GATE1 {verdict}] " + ("找到可重复的转向启动+回正神经群体（可进入第二关）"
          if n_conf >= 1 else
          "未找到可重复的神经群体动态信号（启动/回正）→ 报告瓶颈并停止，不进入第二关"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
