"""P2d-28：MaleCNS 脑控小车"转弯无法结束"**因果诊断**（只定位，不调参、不开发新控制器）。

- **原样复现** 6 个 POP 闭环试次：confirm seeds 606/808/1001 × must_left/must_right（不重选）。
- 固定：L50/R50（L_IDX/R_IDX）、Trace τ=0.2、gain=8、sign=−1、estimate_b0 规则、BASE 5-ToF 编码器、
  sensory_input=False、脑/运动学/安全 STOP —— **均不改**。
- 逐时间点**同步**记录（sim 20 ms；ToF 10 Hz 保持）：5 路原始 ToF、编码后侧别输入、L50/R50 群体 Trace 均值、
  D(t)、b0、剪裁前/后神经转向命令、限幅、P2Control 的 v/ω、车辆朝向、front_safety、碰撞与前进进展。
- 本模块的 `run_instrumented` 是 `loop.dn_steering_diag.run_trace` **循环体的忠实拷贝**，仅**增加记录**，
  不改任何控制逻辑/参数（用于在不改行为的前提下取全通道时序）。

事件与归因仅用于**离线评价**，**绝不回流控制器**。
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.decode import Decoder
from loop.dn_steering import DNReadout, calibrate_baseline
from loop.dynamic_tof_sim import frame_from_pose, load_rig
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import CAR_R, SCENARIOS, wall_clearance
from loop.population_turn import (GAIN, SIGN, TAU, PopulationTurn, estimate_b0)
from loop.run import make_setup
from world.car import DiffDriveCar
from world.maze import Maze

HERE = os.path.dirname(os.path.abspath(__file__))
D_FAR = 1.2          # dmin_F < 此值 = 决策区（接近前方）
STRONG = 0.30        # |turn|>= 此值 = 明显转向
SAT = 0.999          # 限幅判定
SEEDS = [606, 808, 1001]
SUCCESS = (1001, "must_right")
COLS = ["t", "seq", "x", "y", "theta", "gd", "need_deg", "dmin_F",
        "tof_F", "tof_FL", "tof_FR", "tof_L", "tof_R", "st_FL", "st_FR",
        "size_L", "size_R", "chase_L", "chase_R", "loom_L", "loom_R", "threat_L", "threat_R",
        "nL", "nR", "D", "Dmb", "turn_pre", "turn", "clipped", "v", "om", "front", "cev"]


def run_instrumented(cfg, brain, groups, dn, seed, rows, *, b0_dec, b0_pop, seconds=40.0,
                     tof_hz=10.0, mpc=0.40, rig=None):
    """run_trace 循环体的忠实拷贝 + 全通道记录。turn 用 L50/R50 PopulationTurn(b0_pop)。"""
    from loop.p2_control import P2Control
    maze = Maze(rows)
    sx, sy, sth = maze.start
    cfg.decoder.mode = "dn"
    cfg.decoder.escape_thresh = 1e9
    fe = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)
    idx = fe.index
    rd = DNReadout(brain.dt, b0=b0_dec)
    diag = DNReadout(brain.dt, b0=b0_dec)
    obj = PopulationTurn(brain, brain.dt, b0=b0_pop, dn=dn)      # 固化 L50/R50, gain=8, sign=-1, tau=0.2
    state = {"nL": 0.0, "nR": 0.0}

    def rec_turn(c):
        turn = obj(c.fired)                                      # 内部调用 trace.observe 一次
        a = obj.trace.features()
        state["nL"] = float(a[:obj.nL].mean()); state["nR"] = float(a[obj.nL:].mean())
        return turn                                 # 与 POP 完全一致（未改）

    ctl = P2Control(cfg.decoder, fe, rec_turn, name="POP")
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    car = DiffDriveCar(sx, sy, sth, cfg.car)
    brain.reset(seed)
    rig = rig or load_rig()
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    gx, gy = (maze.goal[0], maze.goal[1]) if maze.goal else (None, None)
    recs = []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = frame_from_pose(maze, car.x, car.y, car.theta, t, seq, rig=rig, meters_per_cell=mpc)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, info = enc.inject()
        fired = brain.step(inject=inj)
        v, omega = ctl.update(fired, info)
        car.set_command(v, omega)
        car.step(maze, dt)
        corr = diag(float(ctl.dec.last_brain_turn))
        gd = maze.dist_to_goal(car.x, car.y)
        cl = enc._cache.get("L", (0, 0, 0, 0, 0))
        cr = enc._cache.get("R", (0, 0, 0, 0, 0))
        D = state["nL"] - state["nR"]
        turn = float(ctl.last_turn_cmd)
        need = (math.atan2(gy - car.y, gx - car.x) - car.theta + math.pi) % (2 * math.pi) - math.pi \
            if gx is not None else 0.0
        recs.append({
            "t": t, "seq": seq, "x": car.x, "y": car.y, "theta": car.theta, "gd": gd,
            "need_deg": math.degrees(need),
            "dmin_F": info.get("dmin_F", 99.0),
            "tof_F": frame.ranges.get("F"), "tof_FL": frame.ranges.get("FL"),
            "tof_FR": frame.ranges.get("FR"), "tof_L": frame.ranges.get("L"),
            "tof_R": frame.ranges.get("R"),
            "st_FL": frame.status.get("FL"), "st_FR": frame.status.get("FR"),
            "size_L": cl[3], "size_R": cr[3], "chase_L": cl[1], "chase_R": cr[1],
            "loom_L": cl[0], "loom_R": cr[0], "threat_L": cl[2], "threat_R": cr[2],
            "nL": state["nL"], "nR": state["nR"], "D": D, "Dmb": D - b0_pop,
            "turn_pre": float(SIGN * GAIN * (D - b0_pop)), "turn": turn,
            "clipped": bool(abs(turn) >= SAT), "v": float(ctl.v), "om": float(ctl.omega),
            "front": bool(ctl.front_blocked), "cev": int(car.collision_events),
        })
        if gd == 0:
            break
    best, last_imp = float("inf"), 0.0
    for r in recs:
        if r["gd"] >= 0 and r["gd"] < best:
            best, last_imp = r["gd"], r["t"]
    T = recs[-1]["t"] if recs else 0.0
    summary = {"reach": bool(recs and recs[-1]["gd"] == 0), "bestD": int(best),
               "cev": recs[-1]["cev"] if recs else 0, "T": T,
               "stuck": (best > 0) and (T - last_imp) > 8.0,
               "sat%": ctl.saturation_ratio() * 100, "stop%": ctl.stop_ratio() * 100,
               "minGap": min((wall_clearance(maze, r["x"], r["y"], CAR_R) for r in recs), default=float("nan"))}
    return summary, recs


# ---------------- offline event + metrics (pure) ----------------
def idx_first(recs, pred):
    return next((i for i, r in enumerate(recs) if pred(r)), None)


def events(recs, dt):
    ev = {}
    ev["approach"] = idx_first(recs, lambda r: r["dmin_F"] < D_FAR)
    ev["turn_start"] = idx_first(recs, lambda r: abs(r["turn"]) >= STRONG)
    ts = ev["turn_start"]
    if ts is not None:
        ev["lateral_clear"] = next((i for i in range(ts, len(recs))
                                    if abs(recs[i]["size_L"] - recs[i]["size_R"]) < 0.05), None)
        ev["forward_resume"] = next((i for i in range(ts, len(recs))
                                     if abs(recs[i]["turn"]) < 0.1 and recs[i]["v"] > 0.02), None)
    else:
        ev["lateral_clear"] = ev["forward_resume"] = None
    return ev


def metrics(recs, dt):
    th = np.array([r["theta"] for r in recs])
    turn = np.array([r["turn"] for r in recs])
    Dmb = np.array([r["Dmb"] for r in recs])
    sizeLR = np.array([r["size_L"] - r["size_R"] for r in recs])
    v = np.array([r["v"] for r in recs])
    dminF = np.array([r["dmin_F"] for r in recs])
    ev = events(recs, dt)
    ts = ev["turn_start"]
    out = {"reach": bool(recs and recs[-1]["gd"] == 0),
           "T": round(recs[-1]["t"], 2) if recs else 0.0,
           "bestD": int(min((r["gd"] for r in recs), default=0)),
           "cev": int(recs[-1]["cev"] if recs else 0),
           "total_heading_deg": round(float(np.degrees(np.unwrap(th)[-1] - np.unwrap(th)[0])), 1),
           "approach_t": round(recs[ev["approach"]]["t"], 2) if ev["approach"] is not None else None,
           "turn_start_t": round(recs[ts]["t"], 2) if ts is not None else None,
           "lateral_clear_t": round(recs[ev["lateral_clear"]]["t"], 2) if ev["lateral_clear"] is not None else None,
           "forward_resume_t": round(recs[ev["forward_resume"]]["t"], 2) if ev["forward_resume"] is not None else None}
    if ts is not None:
        seg = slice(ts, len(recs))
        out["frac_sat_after_turn"] = round(float(np.mean(np.abs(turn[seg]) >= SAT)), 3)
        out["sign_turn"] = float(np.sign(np.mean(turn[seg]))) if len(turn[seg]) else 0.0
        out["Dmb_min_after_turn"] = round(float(np.min(np.abs(Dmb[seg]))), 4)
        out["Dmb_end"] = round(float(Dmb[-1]), 4)
        # 转向后"命令已趋零但仍旋转"：turn 首次 <0.1 之后的累计转向
        z = next((i for i in range(ts, len(recs)) if abs(recs[i]["turn"]) < 0.1), None)
        if z is not None:
            thz = np.unwrap(th[z:])
            out["heading_after_turn_zero_deg"] = round(float(np.degrees(thz[-1] - thz[0])), 1) if len(thz) else 0.0
        else:
            out["heading_after_turn_zero_deg"] = None
        out["v_end"] = round(float(v[-1]), 3)
    # 无遮挡背景（dmin_F>=D_FAR）中的自发 turn 偏置
    bg = np.array([r["turn"] for r in recs if r["dmin_F"] >= D_FAR])
    out["bg_turn_mean"] = round(float(np.mean(bg)), 4) if len(bg) else None
    out["bg_turn_absmean"] = round(float(np.mean(np.abs(bg))), 4) if len(bg) else None
    # 侧别感觉 vs 神经输出相关（转向后）
    if ts is not None and len(recs) - ts > 5:
        s = sizeLR[ts:]; u = turn[ts:]
        out["corr_sizeLR_turn"] = round(float(np.corrcoef(s, u)[0, 1]), 3) if s.std() > 0 and u.std() > 0 else None
    else:
        out["corr_sizeLR_turn"] = None
    # 振荡证据：神经转向命令 / 侧别感觉 的换号率；剪裁前幅度；D 波动
    T = recs[-1]["t"] if recs else 1.0
    out["turn_sign_changes"] = int(np.sum(np.diff(np.sign(turn)) != 0))
    out["turn_osc_hz"] = round(out["turn_sign_changes"] / T, 2)
    out["sizeLR_sign_changes"] = int(np.sum(np.diff(np.sign(sizeLR)) != 0))
    out["frac_sat_any"] = round(float(np.mean(np.abs(turn) >= SAT)), 3)
    pre = np.array([r["turn_pre"] for r in recs])
    out["frac_preclip_over1"] = round(float(np.mean(np.abs(pre) > 1.0)), 3)   # gain=8 放大到饱和的比例
    out["D_std"] = round(float(np.std([r["D"] for r in recs])), 4)
    out["net_omega"] = round(float(np.mean([r["om"] for r in recs])), 3)
    out["net_turn"] = round(float(np.mean(turn)), 4)
    return out


ATTRIB = {"S": "感觉输入持续/反复触发", "N": "神经信号持续或背景偏置(含振荡)",
          "M": "人工映射/限幅造成放大", "V": "车辆运动响应", "U": "无法确定"}


def attribute(met, recs):
    """基于证据归因（可多标签；无证据给 U）。**只用离线记录**。"""
    tags = []
    # N：神经转向命令快速振荡（背景/无稳定状态）或 D-b0 未回背景
    if met.get("turn_osc_hz", 0.0) >= 2.0:
        tags.append("N")
    elif met.get("Dmb_min_after_turn") is not None and met["Dmb_min_after_turn"] > 0.05:
        tags.append("N")
    # M：gain=8 把小幅 D（std~0.08）放大到剪裁饱和——剪裁前 |turn|>1 出现在可观比例
    if met.get("frac_preclip_over1", 0.0) > 0.05:
        tags.append("M")
    # S：侧向障碍始终未减（感觉持续）。注意：sizeLR 换号在车辆自旋时会**混入**，故不作为
    #    S 的独立证据（证据见 bg 段：无遮挡时振荡仍存在 → 非感觉驱动）。
    if met.get("lateral_clear_t") is None and met.get("approach_t") is not None:
        tags.append("S")
    # V：车辆累计转角巨大（持续旋转）
    if abs(met.get("total_heading_deg", 0.0)) > 360.0:
        tags.append("V")
    if not tags:
        tags.append("U")
    return tags


_FLOAT = {"t", "x", "y", "theta", "gd", "need_deg", "dmin_F", "tof_F", "tof_FL", "tof_FR",
          "tof_L", "tof_R", "size_L", "size_R", "chase_L", "chase_R", "loom_L", "loom_R",
          "threat_L", "threat_R", "nL", "nR", "D", "Dmb", "turn_pre", "turn", "v", "om"}
_BOOL = {"front", "clipped"}
_INT = {"seq", "cev"}


def _cast(row):
    out = {}
    for k, v in row.items():
        if k in _FLOAT:
            out[k] = float("nan") if v in ("", "None") else float(v)
        elif k in _BOOL:
            out[k] = (v == "True")
        elif k in _INT:
            out[k] = int(float(v))
        else:
            out[k] = v
    return out


def reanalyze(outdir, dt=0.02):
    """从已保存的 trace CSV 重算指标（不重跑模拟）。"""
    res = []
    for sd in SEEDS:
        for sc in ("must_left", "must_right"):
            p = os.path.join(outdir, f"trace_{sd}_{sc}.csv")
            recs = [_cast(r) for r in csv.DictReader(open(p, encoding="utf-8"))]
            met = metrics(recs, dt)
            met.update({"seed": sd, "scenario": sc})
            met["attrib"] = "".join(sorted(set(attribute(met, recs))))
            res.append(met)
    json.dump(res, open(os.path.join(HERE, "p2d28_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # 逐事件证据表
    ev_cols = ["seed", "scenario", "reach", "T", "bestD", "approach_t", "turn_start_t",
               "lateral_clear_t", "forward_resume_t", "total_heading_deg", "net_omega",
               "net_turn", "turn_osc_hz", "frac_sat_any", "frac_preclip_over1", "D_std",
               "bg_turn_absmean", "corr_sizeLR_turn", "attrib"]
    with open(os.path.join(HERE, "p2d28_events.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ev_cols); w.writeheader()
        for m in res:
            w.writerow({k: m.get(k, "") for k in ev_cols})
    return res


def plot_compare(outdir, key_fail=(606, "must_left"), key_ok=(1001, "must_right")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for col, (sd, sc) in enumerate((key_fail, key_ok)):
        recs = [_cast(r) for r in csv.DictReader(open(os.path.join(outdir, f"trace_{sd}_{sc}.csv"),
                                                     encoding="utf-8"))]
        t = [r["t"] for r in recs]
        ax[0, col].plot(t, [r["turn"] for r in recs], lw=0.7, label="turn")
        ax[0, col].plot(t, [r["Dmb"] for r in recs], lw=0.8, label="D-b0")
        ax[0, col].axhline(0, c="0.7"); ax[0, col].set_title(f"seed{sd} {sc}: turn/D (net_turn={np.mean([r['turn'] for r in recs]):+.3f})")
        ax[0, col].legend(fontsize=7)
        ax[1, col].plot(t, [r["v"] for r in recs], lw=0.8, label="v")
        ax[1, col].plot(t, [r["theta"] for r in recs], lw=0.8, label="theta")
        ax[1, col].plot(t, [r["gd"] for r in recs], lw=0.8, label="goal_dist")
        ax[1, col].legend(fontsize=7); ax[1, col].set_xlabel("t (s)")
    ax[0, 2].plot([r["tof_FL"] for r in [_cast(x) for x in csv.DictReader(open(os.path.join(outdir, f"trace_{key_fail[0]}_{key_fail[1]}.csv"), encoding="utf-8"))]], lw=0.8, label="FL fail")
    ax[0, 2].plot([r["tof_FR"] for r in [_cast(x) for x in csv.DictReader(open(os.path.join(outdir, f"trace_{key_fail[0]}_{key_fail[1]}.csv"), encoding="utf-8"))]], lw=0.8, label="FR fail")
    ax[0, 2].legend(fontsize=7); ax[0, 2].set_title("front ToF (failure)")
    ax[1, 2].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "p2d28_compare.png"), dpi=110, bbox_inches="tight")
    print("plot -> loop/p2d28_compare.png")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                        # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--outdir", default=os.path.join(HERE, "p2d28_out"))
    ap.add_argument("--from-csv", action="store_true", help="只从已保存 CSV 重算（不重跑）")
    a = ap.parse_args(argv)
    if a.from_csv:
        res = reanalyze(a.outdir)
        for m in res:
            print(f"{m['seed']:5d} {m['scenario']:10s} reach={m['reach']!s:1s} "
                  f"headΔ={m['total_heading_deg']:+7.1f} osc={m['turn_osc_hz']}Hz "
                  f"sat={m['frac_sat_any']} preclip>1={m['frac_preclip_over1']} "
                  f"netΩ={m['net_omega']} D_std={m['D_std']} attrib={m['attrib']}")
        plot_compare(a.outdir)
        return 0
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=a.device, seed=64,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()
    os.makedirs(a.outdir, exist_ok=True)
    b0pop = {sd: estimate_b0(cfg, brain, groups, dn, sd) for sd in SEEDS}
    b0dec = {sd: calibrate_baseline(cfg, brain, groups, dn, sd) for sd in SEEDS}
    all_met = []
    for sd in SEEDS:
        for sc in ("must_left", "must_right"):
            s, recs = run_instrumented(cfg, brain, groups, dn, sd, SCENARIOS[sc],
                                       b0_dec=b0dec[sd], b0_pop=b0pop[sd], seconds=a.seconds, rig=rig)
            met = metrics(recs, brain.dt)
            met.update({"seed": sd, "scenario": sc, "b0_pop": round(b0pop[sd], 4)})
            met["attrib"] = "".join(sorted(set(attribute(met, recs))))
            all_met.append(met)
            with open(os.path.join(a.outdir, f"trace_{sd}_{sc}.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
                for r in recs:
                    w.writerow({k: r.get(k) for k in COLS})
            print(f"{sd:5d} {sc:10s} reach={str(s['reach'])[:1]} bestD={s['bestD']:3d} "
                  f"T={met['T']:5.2f} headΔ={met['total_heading_deg']:+7.1f} "
                  f"sat={met.get('frac_sat_after_turn')} Dmb_min={met.get('Dmb_min_after_turn')} "
                  f"lateral_clear={met['lateral_clear_t']} attrib={met['attrib']}")
    json.dump(all_met, open(os.path.join(HERE, "p2d28_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p2d28_metrics.json + p2d28_out/trace_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
