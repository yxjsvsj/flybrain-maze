"""P2d-29：**唯一**新候选读出 NS（Neural Slow-integration）——固定参数，离线先验证。

来自 P2d-28 证据的**预注册**参数：
  - 输入：与 POP **完全相同**的 L50/R50 Trace(τ=0.2) 差 D(t)（只用当前与过去的神经活动）。
  - 背景实测相关时间 τ_corr≈0.1 s；障碍诱发的 Dmb 位移是**持续**的（数秒）。
  - **因果时间整合**：D_f ← D_f + (dt/τ_f)(D − D_f)，**τ_f=1.0 s**（= 10×τ_corr，固定，未搜索）；
    状态初始化 D_f(0)=b0（沿用既有 estimate_b0，运行前固定）。
  - 输出：`turn = clip(sign · gain · (D_f − b0), −1, 1)`，**gain=8、sign=−1 不变**。
  - **禁止**：ToF 原始距离、地图、目标、真实车身角、老师信号、预定转弯时长、手写避障方向。

离线验证门槛（冻结）：无遮拦背景误转向 **mean|turn|_bg 相对 POP 下降 ≥30%**，且侧障出现时**方向正确**
（左近 FL<CLEAR → turn<0；右近 FR<CLEAR → turn>0）的保留率 ≥0.6。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.02
TAU_F = 1.0          # 冻结：= 10 × 实测背景相关时间(0.1 s)
GAIN, SIGN = 8.0, -1.0
D_FAR = 1.2
SIDE_CLOSE_MM = 400.0
REDUCE_MIN = 0.30    # 背景误转向须相对 POP 下降 ≥30%
DIR_KEEP_MIN = 0.60  # 侧障方向正确保留率下限


def causal_lp(x, dt=DT, tau=TAU_F, x0=0.0):
    """一阶因果低通（= 时间整合）。x: 序列；x0: 初值（固定为 b0）。"""
    a = dt / tau
    out = np.empty(len(x), np.float64)
    f = float(x0)
    for i, v in enumerate(x):
        f += a * (v - f)
        out[i] = f
    return out


def ns_turn(Dmb, dt=DT, tau=TAU_F, gain=GAIN, sign=SIGN):
    """NS 读出：对 Dmb 做因果整合后按既有 gain/sign 映射（无其它输入）。"""
    F = causal_lp(Dmb, dt, tau, x0=0.0)          # Dmb = D - b0，故初值 0
    return np.clip(sign * gain * F, -1.0, 1.0)


def _arrays(rows):
    g = lambda k, d=0.0: np.array([float(r[k]) if r[k] not in ("", "None") else d for r in rows])
    return {"Dmb": g("Dmb"), "turn_pop": g("turn"), "dmin_F": g("dmin_F"),
            "FL": g("tof_FL", 9999.0), "FR": g("tof_FR", 9999.0), "v": g("v")}


def evaluate(csv_dir):
    trials = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "trace_*.csv"))):
        name = os.path.basename(f)
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        A = _arrays(rows)
        turn_ns = ns_turn(A["Dmb"])
        bg = A["dmin_F"] >= D_FAR
        left = A["FL"] < SIDE_CLOSE_MM
        right = A["FR"] < SIDE_CLOSE_MM
        def dir_keep(turn):
            ok_l = (turn[left] < 0) if left.sum() else np.array([True])
            ok_r = (turn[right] > 0) if right.sum() else np.array([True])
            return float(np.mean(np.r_[ok_l, ok_r])) if (left.sum() + right.sum()) else np.nan
        def amp(turn, m):
            return float(np.mean(np.abs(turn[m]))) if m.sum() else np.nan
        trials.append({
            "name": name,
            "bg_turn_pop": round(amp(A["turn_pop"], bg), 4),
            "bg_turn_ns": round(amp(turn_ns, bg), 4),
            "dir_keep_pop": round(dir_keep(A["turn_pop"]), 3),
            "dir_keep_ns": round(dir_keep(turn_ns), 3),
            "obst_amp_pop": round(amp(A["turn_pop"], left | right), 4),
            "obst_amp_ns": round(amp(turn_ns, left | right), 4),
            "flip_pop": int(np.sum(np.diff(np.sign(A["turn_pop"])) != 0)),
            "flip_ns": int(np.sum(np.diff(np.sign(turn_ns)) != 0)),
        })
    return trials


def open_loop(seeds, *, device="cuda", tof_hz=10.0, mpc=0.40):
    """A→侧障→A 开环（无车）记录 D；比较 POP 与 NS 的诱发响应方向/幅度。"""
    from flybrain.data import ensure_data
    from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
    from loop.dn_dynamics_probe import PLANS, STIMS, make_frame
    from loop.encode_tof import PhysicalToFEncoder
    from loop.population_turn import PopulationTurn, estimate_b0
    from loop.run import make_setup
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    dt = brain.dt
    res = []
    for sd in seeds:
        b0 = estimate_b0(cfg, brain, groups, dn, sd)
        for side, plan_key in (("left", "left"), ("right", "right")):
            enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                     sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                     car_radius=cfg.car.radius, meters_per_cell=mpc)
            pt = PopulationTurn(brain, brain.dt, b0=b0, dn=dn)
            brain.reset(sd)
            period = 1.0 / tof_hz
            frame, seq, last_t, total = None, 0, -1e9, 0
            Ds, labels = [], []
            for name, dur in PLANS[plan_key]:
                for _ in range(int(round(dur / dt))):
                    tt = total * dt
                    if frame is None or (tt - last_t) >= period - 1e-9:
                        seq += 1; frame = make_frame(STIMS[name], seq, tt); last_t = tt
                    d = enc.sense(frame); enc.update(frame, d); inj, _ = enc.inject()
                    fired = brain.step(inject=inj)
                    Ds.append(pt.D(fired)); labels.append(name); total += 1
            Ds = np.array(Ds); labels = np.array(labels)
            Dmb = Ds - b0
            turn_pop = np.clip(SIGN * GAIN * Dmb, -1, 1)
            turn_ns = ns_turn(Dmb)
            base = np.isin(labels, ["A"]); stim = ~base
            # 段级诱发幅度（stim 段均值）与假转向（A 段 |turn| 均值）、漏检、撤除后衰减
            def seg_metrics(turn):
                t_stim = float(np.mean(turn[stim])) if stim.sum() else np.nan
                t_base = float(np.mean(np.abs(turn[base]))) if base.sum() else np.nan
                miss = bool(stim.sum() and abs(t_stim) < 0.3)
                # 撤除后衰减：最后一个 stim 段之后的前 0.3s 的 |turn|
                idx = np.flatnonzero(stim)
                dec = np.nan
                if len(idx):
                    end = idx[-1] + 1
                    after = turn[end:end + int(round(0.3 / DT))]
                    dec = float(np.mean(np.abs(after))) if len(after) else np.nan
                return t_stim, t_base, miss, dec
            ps, pb, pm, pd = seg_metrics(turn_pop)
            ns_, nb, nm, nd = seg_metrics(turn_ns)
            res.append({"seed": sd, "side": side, "expect_sign": -1 if side == "left" else 1,
                        "pop_stim": round(ps, 4), "pop_false": round(pb, 4), "pop_miss": pm, "pop_decay": round(pd, 4),
                        "ns_stim": round(ns_, 4), "ns_false": round(nb, 4), "ns_miss": nm, "ns_decay": round(nd, 4)})
    return res


# ---- Part D: frozen seeds for the closed loop ----
SCREEN_SEEDS = [64, 101, 202, 303]
CONFIRM_SEEDS = [7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108]   # 全新、事前冻结
MODES = ("NS", "POP", "ZERO", "FLIP")
MIN_HEADING_DEG = 45.0     # 真正完成转弯的最小净转角（防"直线冒充"）
GATE_COMPLETE = 12         # ≥12/16
GATE_COLLISION = 0


def run_closedloop(cfg, brain, groups, dn, seed, rows, *, b0, mode="NS", seconds=40.0,
                   tof_hz=10.0, mpc=0.40, rig=None):
    from loop.decode import Decoder
    from loop.dn_steering import DNReadout
    from loop.dynamic_tof_sim import frame_from_pose, load_rig
    from loop.encode_tof import PhysicalToFEncoder
    from loop.local_avoidance_bench import CAR_R, wall_clearance
    from loop.p2_control import P2Control
    from loop.population_turn import PopulationTurn
    from world.car import DiffDriveCar
    from world.maze import Maze
    maze = Maze(rows); sx, sy, sth = maze.start
    cfg.decoder.mode = "dn"; cfg.decoder.escape_thresh = 1e9
    fe = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    rd = DNReadout(brain.dt, b0=b0)
    obj = PopulationTurn(brain, brain.dt, b0=b0, dn=dn)
    st = {"Df": 0.0}
    dt = brain.dt
    def src(c):
        D = obj.D(c.fired); Dmb = D - b0
        if mode == "POP":
            return float(np.clip(SIGN * GAIN * Dmb, -1, 1))
        if mode == "NS":
            st["Df"] += (dt / TAU_F) * (Dmb - st["Df"])
            return float(np.clip(SIGN * GAIN * st["Df"], -1, 1))
        if mode == "ZERO":
            return 0.0
        if mode == "FLIP":
            return float(-np.clip(SIGN * GAIN * Dmb, -1, 1))
        raise KeyError(mode)
    ctl = P2Control(cfg.decoder, fe, src, name=mode)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt,
                             __import__("hardware.tof5", fromlist=["DEFAULT_ANGLES_DEG"]).DEFAULT_ANGLES_DEG,
                             sensor_order=__import__("hardware.tof5", fromlist=["ORDER"]).ORDER,
                             car_max_speed=cfg.car.max_speed, car_radius=cfg.car.radius, meters_per_cell=mpc)
    car = DiffDriveCar(sx, sy, sth, cfg.car); brain.reset(seed); rig = rig or load_rig()
    n = int(round(seconds / dt)); period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    gx, gy = (maze.goal[0], maze.goal[1]) if maze.goal else (None, None)
    recs = []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = frame_from_pose(maze, car.x, car.y, car.theta, t, seq, rig=rig, meters_per_cell=mpc)
            last_t = t
        d = enc.sense(frame); enc.update(frame, d); inj, info = enc.inject()
        fired = brain.step(inject=inj)
        v, om = ctl.update(fired, info); car.set_command(v, om); car.step(maze, dt)
        recs.append({"t": t, "x": car.x, "y": car.y, "theta": car.theta,
                     "gd": maze.dist_to_goal(car.x, car.y), "turn": float(ctl.last_turn_cmd),
                     "v": float(ctl.v), "om": float(ctl.omega), "front": bool(ctl.front_blocked),
                     "cev": int(car.collision_events)})
        if recs[-1]["gd"] == 0:
            break
    th = np.unwrap([r["theta"] for r in recs]) if recs else np.array([0.0])
    heading = float(np.degrees(th[-1] - th[0])) if len(th) else 0.0
    reach = bool(recs and recs[-1]["gd"] == 0)
    cev = recs[-1]["cev"] if recs else 0
    summ = {"reach": reach, "cev": int(cev), "T": round(recs[-1]["t"], 2) if recs else 0.0,
            "heading_deg": round(heading, 1),
            "real_complete": bool(reach and cev == 0 and abs(heading) >= MIN_HEADING_DEG),
            "minGap": round(min((wall_clearance(maze, r["x"], r["y"], CAR_R) for r in recs), default=float("nan")), 4)}
    return summ, recs


def _closed_main(seeds, device, seconds):
    import json as _json
    from loop.dynamic_tof_sim import load_rig
    from loop.local_avoidance_bench import SCENARIOS
    from loop.population_turn import estimate_b0
    from loop.run import make_setup
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"])); rig = load_rig()
    b0 = {sd: estimate_b0(cfg, brain, groups, dn, sd) for sd in seeds}
    out = []
    for sd in seeds:
        for sc in ("must_left", "must_right"):
            for mode in MODES:
                s, _ = run_closedloop(cfg, brain, groups, dn, sd, SCENARIOS[sc],
                                      b0=b0[sd], mode=mode, seconds=seconds, rig=rig)
                out.append({"seed": sd, "scenario": sc, "mode": mode, **s})
                print(f"  {sd} {sc:10s} {mode:4s} reach={str(s['reach'])[:1]} cev={s['cev']} "
                      f"headΔ={s['heading_deg']:+7.1f} real={str(s['real_complete'])[:1]} T={s['T']}")
    ns = [o for o in out if o["mode"] == "NS"]
    comp = sum(o["real_complete"] for o in ns)
    coll = sum(o["cev"] for o in ns)
    zf = [o for o in out if o["mode"] in ("ZERO", "FLIP")]
    zf_comp = sum(o["real_complete"] for o in zf)
    gate = (comp >= GATE_COMPLETE) and (coll == 0) and (zf_comp == 0)
    verdict = "PASS" if gate else "FAIL"
    print(f"[Part D] NS real_complete={comp}/16 collisions={coll} | ZERO/FLIP real_complete={zf_comp} -> {verdict}")
    _json.dump({"verdict": verdict, "ns_complete": comp, "collisions": coll, "zf_complete": zf_comp,
                "gate": {"GATE_COMPLETE": GATE_COMPLETE, "GATE_COLLISION": GATE_COLLISION},
                "per": out}, open(os.path.join(HERE, "p2d29_closedloop.json"), "w", encoding="utf-8"),
               ensure_ascii=False, indent=1)
    return 0


def base_only(seeds, *, seconds=15.0, device="cuda", tof_hz=10.0, mpc=0.40):
    """持续**无遮拦(A)** 背景（真·无遮挡）下 POP 与 NS 的误转向。"""
    from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
    from loop.dn_dynamics_probe import STIMS, make_frame
    from loop.encode_tof import PhysicalToFEncoder
    from loop.population_turn import PopulationTurn, estimate_b0
    from loop.run import make_setup
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=device, seed=seeds[0],
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    dt = brain.dt
    out = []
    for sd in seeds:
        b0 = estimate_b0(cfg, brain, groups, dn, sd)
        enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                 sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                 car_radius=cfg.car.radius, meters_per_cell=mpc)
        pt = PopulationTurn(brain, brain.dt, b0=b0, dn=dn)
        brain.reset(sd)
        period = 1.0 / tof_hz; frame, seq, last_t = None, 0, -1e9
        Ds = []
        for k in range(int(round(seconds / dt))):
            tt = k * dt
            if frame is None or (tt - last_t) >= period - 1e-9:
                seq += 1; frame = make_frame(STIMS["A"], seq, tt); last_t = tt
            d = enc.sense(frame); enc.update(frame, d); inj, _ = enc.inject()
            fired = brain.step(inject=inj)
            Ds.append(pt.D(fired))
        Dmb = np.array(Ds) - b0
        tp = np.clip(SIGN * GAIN * Dmb, -1, 1); tn = ns_turn(Dmb)
        out.append({"seed": sd, "pop_false": round(float(np.mean(np.abs(tp))), 4),
                    "ns_false": round(float(np.mean(np.abs(tn))), 4),
                    "pop_flips": int(np.sum(np.diff(np.sign(tp)) != 0)),
                    "ns_flips": int(np.sum(np.diff(np.sign(tn)) != 0))})
    return out


def _base_main(seeds, device):
    import json as _json
    res = base_only(seeds, device=device)
    pf = float(np.mean([r["pop_false"] for r in res])); nf = float(np.mean([r["ns_false"] for r in res]))
    red = (pf - nf) / pf if pf else 0.0
    print(f"[long no-obstacle {len(res)} seeds] POP false={pf:.3f} NS false={nf:.3f} reduction={red:.2%}")
    for r in res:
        print(f"  seed{r['seed']} POP|{r['pop_false']}| flips{r['pop_flips']} | NS|{r['ns_false']}| flips{r['ns_flips']}")
    _json.dump({"pop_false_mean": round(pf, 4), "ns_false_mean": round(nf, 4), "reduction": round(red, 3),
                "per": res}, open(os.path.join(HERE, "p2d29_base.json"), "w", encoding="utf-8"),
               ensure_ascii=False, indent=1)
    return 0


def _open_loop_main(seeds, device):
    import json as _json
    res = open_loop(seeds, device=device)
    pop_dir = sum(1 for r in res if np.sign(r["pop_stim"]) == r["expect_sign"] and abs(r["pop_stim"]) > 0.05)
    ns_dir = sum(1 for r in res if np.sign(r["ns_stim"]) == r["expect_sign"] and abs(r["ns_stim"]) > 0.05)
    pop_false = float(np.mean([r["pop_false"] for r in res]))
    ns_false = float(np.mean([r["ns_false"] for r in res]))
    red = (pop_false - ns_false) / pop_false if pop_false else 0.0
    ns_miss = sum(1 for r in res if r["ns_miss"])
    # 冻结门：背景误转向↓≥30% 且 方向正确（NS）
    gate_ok = (red >= REDUCE_MIN) and (ns_dir == len(res))
    verdict = "PASS_OFFLINE" if gate_ok else "FAIL_OFFLINE"
    print(f"[open-loop] POP dir {pop_dir}/{len(res)} false={pop_false:.3f} | "
          f"NS dir {ns_dir}/{len(res)} false={ns_false:.3f} miss={ns_miss} reduction={red:.2%} -> {verdict}")
    for r in res:
        print(f"  seed{r['seed']} {r['side']:5s} exp{'+' if r['expect_sign']>0 else '-'} "
              f"POP stim{r['pop_stim']:+.3f} false{r['pop_false']:.3f} | "
              f"NS stim{r['ns_stim']:+.3f} false{r['ns_false']:.3f} miss={r['ns_miss']} decay{r['ns_decay']}")
    _json.dump({"verdict": verdict, "pop_dir": pop_dir, "ns_dir": ns_dir, "n": len(res),
                "pop_false_mean": round(pop_false, 4), "ns_false_mean": round(ns_false, 4),
                "reduction": round(red, 3), "ns_miss": ns_miss,
                "gate": {"REDUCE_MIN": REDUCE_MIN}, "per": res},
               open(os.path.join(HERE, "p2d29_openloop.json"), "w", encoding="utf-8"),
               ensure_ascii=False, indent=1)
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                     # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=os.path.join(HERE, "p2d28_out"))
    ap.add_argument("--open-loop", default="")
    ap.add_argument("--base-only", default="")
    ap.add_argument("--closed-loop", default="")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.closed_loop:
        return _closed_main([int(s) for s in a.closed_loop.split(",")], a.device, a.seconds)
    if a.base_only:
        return _base_main([int(s) for s in a.base_only.split(",")], a.device)
    if a.open_loop:
        return _open_loop_main([int(s) for s in a.open_loop.split(",")], a.device)
    tr = evaluate(a.csv_dir)
    bgp = np.nanmean([t["bg_turn_pop"] for t in tr]); bgn = np.nanmean([t["bg_turn_ns"] for t in tr])
    dkp = np.nanmean([t["dir_keep_pop"] for t in tr]); dkn = np.nanmean([t["dir_keep_ns"] for t in tr])
    red = (bgp - bgn) / bgp if bgp else 0.0
    verdict = ("PASS_OFFLINE" if red >= REDUCE_MIN and dkn >= DIR_KEEP_MIN else "FAIL_OFFLINE")
    out = {"params": {"TAU_F": TAU_F, "GAIN": GAIN, "SIGN": SIGN, "D_FAR": D_FAR,
                      "SIDE_CLOSE_MM": SIDE_CLOSE_MM},
           "bg_turn_pop_mean": round(float(bgp), 4), "bg_turn_ns_mean": round(float(bgn), 4),
           "bg_reduction": round(float(red), 3),
           "dir_keep_pop_mean": round(float(dkp), 3), "dir_keep_ns_mean": round(float(dkn), 3),
           "gate": {"REDUCE_MIN": REDUCE_MIN, "DIR_KEEP_MIN": DIR_KEEP_MIN}, "verdict": verdict,
           "per_trial": tr}
    json.dump(out, open(os.path.join(HERE, "p2d29_offline.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("bg_turn_pop_mean", "bg_turn_ns_mean", "bg_reduction",
                                          "dir_keep_pop_mean", "dir_keep_ns_mean", "verdict")},
                     ensure_ascii=False, indent=1))
    for t in tr:
        print(f"  {t['name']}: bg pop={t['bg_turn_pop']} ns={t['bg_turn_ns']} "
              f"dirkeep pop={t['dir_keep_pop']} ns={t['dir_keep_ns']} flips pop={t['flip_pop']} ns={t['flip_ns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
