"""P2d-19：背景神经注入 / 基线漂移 / 持续转向的因果拆解（只诊断，不改正式控制器）。

在 P2d-18 之上把"无障碍"拆成三种**神经输入**不同的状态（Frame 与 Injection 分开控制）：
  orig             回放真实 ToF + BASE 注入
  no_target        五路 NO_TARGET，但 **BASE 原样执行**（仍含左右基础 chase + 固定 fwd）
  fwd_only         五路 NO_TARGET，**只保留 fwd** 注入（移除其余感觉注入）
  zero_injection   五路 NO_TARGET，**不施加任何外部注入**（含关闭 fwd）
C/D 只用于诊断性神经消融，不进正式小车控制器，不据此刻意设计人工转向规则。
同 seed、同初始化、分叉前回放相同历史（校验 D/turn/注入一致），分叉后只改 Frame/Injection。
分叉规则：|turn|>=0.5（诊断 assertive 阈值）持续 >=0.5s 的最长段起点（成功/失败同规则）。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import SCENARIOS
from loop.p2_diag import run_diag
from loop.population_turn import GAIN, L_IDX, R_IDX, SIGN, TAU, estimate_b0
from loop.run import make_setup

_CODE2ST = {2.0: "VALID", 1.0: "TOO_NEAR", 0.0: "NO_TARGET"}
# 条件 -> (Frame 模式, 注入模式)
CONDS = {"orig": ("orig", "enc"), "no_target": ("none", "enc"),
         "fwd_only": ("none", "fwd"), "zero_injection": ("none", "none")}
INJ_GROUPS = ("loom_L", "loom_R", "chase_L", "chase_R", "threat_L", "threat_R", "fwd")


def pick_fork(turn, t, thr=0.5, min_s=0.5):
    """事先固定规则：|turn|>=thr 的最长连续段起点；无则退化到 |turn| 最大步。"""
    m = np.abs(turn) >= thr
    best, blen, i, n = -1, 0, 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if t[j] - t[i] >= min_s and (j - i) > blen:
                best, blen = i, j - i
            i = j + 1
        else:
            i += 1
    if best < 0 and len(turn):
        best = int(np.argmax(np.abs(turn)))
    return best


def frame_at(rec, k, mode="orig"):
    rng, st = {}, {}
    for n in ORDER:
        code, val = float(rec["st" + n][k]), float(rec["tof" + n][k])
        if mode == "none" or (mode == "noF" and n == "F") or (mode == "noLR" and n in ("FL", "FR")):
            rng[n], st[n] = None, "NO_TARGET"
        elif code <= 0.0:
            rng[n], st[n] = None, "NO_TARGET"
        else:
            rng[n], st[n] = val, _CODE2ST.get(code, "NO_TARGET")
    return TofFrame(session_id="cf", seq=int(rec["seq"][k]), t=float(rec["t"][k]), healthy=True,
                    ranges=rng, status=st, ages_ms={n: 100.0 for n in ORDER},
                    init_count=0, read_errors=0, reinit_count=0)


def _inject_amounts(groups, inj):
    amt = {g: 0.0 for g in INJ_GROUPS}
    for idx, dv in inj:
        for g in INJ_GROUPS:
            if np.array_equal(np.asarray(idx), np.asarray(groups[g])):
                amt[g] += float(dv)
                break
    return amt


def run_cf(cfg, brain, groups, dn_idx, seed, rec, fork_i, frame_mode, inj_mode, *,
           b0, gain=GAIN, sign=SIGN, tau=TAU, n_total=400, mpc=0.40):
    """分叉后重跑 n_total 步（默认 8s）；返回 D/turn 序列与分叉后各群注入均值。"""
    from flybrain import Trace
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    ps = float(enc.per_step)
    fwd_once = float(enc.cfg.forward_drive_dv * ps) if enc.cfg.forward_drive_dv > 0 else 0.0
    n = min(n_total, len(rec["t"]))
    Ds, turns = [], []
    inj_post = {g: 0.0 for g in INJ_GROUPS}
    inj_pre = {g: 0.0 for g in INJ_GROUPS}
    kk = kp = 0
    for k in range(n):
        fm = "orig" if k < fork_i else frame_mode
        im = "enc" if k < fork_i else inj_mode
        f = frame_at(rec, k, fm)
        d = enc.sense(f)
        enc.update(f, d)
        inj, _info = enc.inject()
        if im == "fwd":
            inj = [(groups["fwd"], fwd_once)] if fwd_once > 0 else []
        elif im == "none":
            inj = []
        fired = brain.step(inject=inj)
        amt = _inject_amounts(groups, inj)
        if k >= fork_i:
            for g in INJ_GROUPS:
                inj_post[g] += amt[g]
            kk += 1
        else:
            for g in INJ_GROUPS:
                inj_pre[g] += amt[g]
            kp += 1
        a = np.asarray(tr.observe(fired), np.float32)
        Dv = float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean())
        Ds.append(Dv)
        turns.append(float(np.clip(sign * gain * (Dv - b0), -1.0, 1.0)))
    if kk:
        inj_post = {g: v / kk for g, v in inj_post.items()}
    if kp:
        inj_pre = {g: v / kp for g, v in inj_pre.items()}
    return np.array(Ds, float), np.array(turns, float), inj_post, inj_pre


def run_A_drift(cfg, brain, groups, dn_idx, seed, b0, *, seconds=40.0, gain=GAIN, sign=SIGN,
                tau=TAU, tof_hz=10.0, mpc=0.40):
    """连续无遮挡 A（五路 NO_TARGET，BASE 注入照旧），固定 b0，看 D-b0/turn 是否漂移。"""
    from flybrain import Trace
    from loop.dn_dynamics_probe import STIMS, make_frame
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    dt = brain.dt
    n = int(round(seconds / dt))
    period = 1.0 / tof_hz
    frame, seq, last_t = None, 0, -1e9
    Ds, turns = [], []
    for k in range(n):
        t = k * dt
        if frame is None or (t - last_t) >= period - 1e-9:
            seq += 1
            frame = make_frame(STIMS["A"], seq, t)
            last_t = t
        d = enc.sense(frame)
        enc.update(frame, d)
        inj, _info = enc.inject()
        fired = brain.step(inject=inj)
        a = np.asarray(tr.observe(fired), np.float32)
        Dv = float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean())
        Ds.append(Dv)
        turns.append(float(np.clip(sign * gain * (Dv - b0), -1.0, 1.0)))
    return np.array(Ds, float), np.array(turns, float)


def metrics(Ds, turns, fork_i, b0, dt, win_s=2.0):
    na = int(round(win_s / dt))
    post_D = Ds[fork_i:fork_i + na] - b0
    post_t = turns[fork_i:fork_i + na]
    pre_t = turns[max(0, fork_i - na):fork_i]
    pre_abs = float(np.mean(np.abs(pre_t))) if len(pre_t) else 0.0
    half = 0.5 * pre_abs
    tfall = -1.0
    for j in range(len(post_t)):
        if abs(post_t[j]) <= half:
            tfall = round(j * dt, 2)
            break
    return {"signed_Dmb": float(np.mean(post_D)) if len(post_D) else 0.0,
            "abs_Dmb": float(np.mean(np.abs(post_D))) if len(post_D) else 0.0,
            "turn_signed": float(np.mean(post_t)) if len(post_t) else 0.0,
            "turn_abs": float(np.mean(np.abs(post_t))) if len(post_t) else 0.0,
            "turn_dur": float(np.mean(np.abs(post_t) >= 0.5)) * win_s if len(post_t) else 0.0,
            "sat": float(np.mean(np.abs(post_t) >= 0.999)) if len(post_t) else 0.0,
            "pre_turn_abs": pre_abs, "t_fall_half": tfall}


def _fig(rec, fork, conds, drift, seed, sc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, (Ds, turns) in conds.items():
        ax[0].plot(np.arange(len(Ds)) * 0.02, Ds, lw=1.1, label=name)
        ax[1].plot(np.arange(len(turns)) * 0.02, turns, lw=1.1, label=name)
    for a, ttl in zip(ax[:2], ("D(t)  [virtual sim]", "turn_cmd  [virtual sim]")):
        a.axvline(fork * 0.02, color="k", ls=":", lw=1)
        a.set_title(ttl + f"\nseed {seed}/{sc}, fork={fork * 0.02:.1f}s")
        a.set_xlabel("t (s)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax[2].plot(np.arange(len(drift)) * 0.02, drift, lw=1.0)
    ax[2].set_title("continuous A: D(t)-b0 (b0 drift)  [virtual sim]")
    ax[2].set_xlabel("t (s)")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1001:must_right,808:must_right,606:must_right")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=1001,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    from loop.dynamic_tof_sim import load_rig
    rig = load_rig()
    dt = brain.dt
    b0s = {}
    print("[P2d-19] 分叉规则=|turn|>=0.5 持续>=0.5s 最长段起点  分叉后 2s  只诊断")
    print("         Frame×Injection 严格配对；B/C/D 仅诊断性消融，不进正式控制器\n")
    for case in args.cases.split(","):
        sd_s, sc = case.split(":")
        seed = int(sd_s)
        if seed not in b0s:
            b0s[seed] = estimate_b0(cfg, brain, groups, dn_idx, seed)
        b0 = b0s[seed]
        rec = run_diag(cfg, brain, groups, dn_idx, seed, SCENARIOS[sc], seconds=args.seconds,
                       rig=rig, b0=b0)
        fork = pick_fork(rec["turn"], rec["t"])
        n_total = min(fork + int(round(5.0 / dt)), len(rec["t"]))
        conds, inj_print, checks = {}, {}, []
        for name, (fm, im) in CONDS.items():
            Ds, turns, inj_post, inj_pre = run_cf(cfg, brain, groups, dn_idx, seed, rec, fork,
                                                  fm, im, b0=b0, n_total=n_total)
            conds[name] = (Ds, turns)
            inj_print[name] = inj_post
            # 分叉前 D/turn 一致；分叉前注入和一致；orig 全程 D/turn 一致
            pre = slice(0, fork)
            dj = float(sum(np.sum(rec["inj_" + g][:fork]) for g in
                           ("loomL", "loomR", "chaseL", "chaseR", "threatL", "threatR", "fwd")))
            dp = float(sum(inj_pre.values())) * max(1, fork)
            ok_dt = (len(Ds) > fork and np.allclose(Ds[pre], rec["D"][pre], atol=1e-4)
                     and np.allclose(turns[pre], rec["turn"][pre], atol=1e-4))
            ok_inj = abs(dp - dj) <= 1e-3 * max(1.0, dj)
            if name == "orig":
                ok_inj = ok_inj and np.allclose(Ds, rec["D"][:len(Ds)], atol=1e-4)
            checks.append((name, bool(ok_dt), bool(ok_inj)))
        tag = "成功" if rec["reach"] else "失败"
        print(f"=== seed {seed} / {sc} [{tag}]  b0={b0:+.3f}  fork_t={rec['t'][fork]:.1f}s ===")
        print(f"{'cond':14s} {'signed(D-b0)':>12s} {'abs(D-b0)':>10s} {'turn_signed':>11s} "
              f"{'|turn|':>7s} {'sat%':>5s} {'dur':>4s} {'t_half':>6s}")
        for name in CONDS:
            m = metrics(conds[name][0], conds[name][1], fork, b0, dt, win_s=args.window_s)
            print(f"{name:14s} {m['signed_Dmb']:+12.3f} {m['abs_Dmb']:10.3f} {m['turn_signed']:+11.3f} "
                  f"{m['turn_abs']:7.3f} {m['sat'] * 100:5.0f} {m['turn_dur']:4.1f} "
                  f"{m['t_fall_half']:6.2f}")
        to = metrics(conds["orig"][0], conds["orig"][1], fork, b0, dt, win_s=args.window_s)["turn_abs"] + 1e-9
        for name in ("no_target", "fwd_only", "zero_injection"):
            ta = metrics(conds[name][0], conds[name][1], fork, b0, dt, win_s=args.window_s)["turn_abs"]
            print(f"  Δ({name} vs orig): |turn| {ta / to * 100:.0f}%")
        print("  各条件分叉后实际注入(每步均值): " + "  ".join(
            f"{name}={{chaseL:{v['chase_L']:.4f},chaseR:{v['chase_R']:.4f},"
            f"loomL:{v['loom_L']:.4f},loomR:{v['loom_R']:.4f},fwd:{v['fwd']:.4f}}}"
            for name, v in inj_print.items()))
        print("  一致性检查(分叉前 D/turn, 分叉前注入和, orig 全程):", checks)
        # b0 漂移：连续 A 40s
        Dd, td = run_A_drift(cfg, brain, groups, dn_idx, seed, b0, seconds=40.0)
        Dm = Dd - b0
        wins = {f"{s}s": float(np.mean(Dm[int(s / dt) - int(5 / dt):int(s / dt)])) for s in (5, 10, 20, 40)}
        print(f"  连续A D-b0 漂移: 初始 {float(np.mean(Dm[:int(1 / dt)])):+.3f}  " +
              " ".join(f"{k}:{v:+.3f}" for k, v in wins.items()) +
              f"  turn|均|={float(np.mean(np.abs(td))):.3f}")
        if args.out_dir:
            import os
            os.makedirs(args.out_dir, exist_ok=True)
            _fig(rec, fork, conds, Dm, seed, sc, os.path.join(args.out_dir, f"p2d19_{seed}_{sc}.png"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
