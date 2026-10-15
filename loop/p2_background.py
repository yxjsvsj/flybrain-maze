"""P2d-20：持续转向 = 刺激历史记忆 还是 自主背景活动？（收尾诊断，不改控制）

审计本机 flybrain v0.1.0 的 step()：v <- decay*v + gain*synaptic(fired_prev) + tonic
  + noise(Bernoulli noise_hz*dt, amp noise_amp) + eye + inject; v>=1 -> spike。
**inject=[] 只关外部注入，不关 tonic/noise/recurrent**，故"零注入"≠"无神经输入"。

三组同 seed/同脑年龄配对（步数、dt、reset 相同；只有分叉前输入不同）：
  A stim_history_zero  真实闭环感觉历史到分叉点，之后 inject=[]
  B A_history_zero     一直无遮挡 A（NO_TARGET）历史到分叉点，之后 inject=[]
  C zero_from_start    始终 inject=[]
A 分叉前必须与 P2d-19 原轨迹逐时间步一致；B/C 为不同历史，不要求分叉前活动与 A 一致。
只研究神经活动（不做运动、不喂控制器）。记录真实 spike 数/发放率（不只平滑转向）。
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.encode_tof import PhysicalToFEncoder
from loop.local_avoidance_bench import SCENARIOS
from loop.p2_counterfactual import frame_at, pick_fork, run_A_drift
from loop.p2_diag import run_diag
from loop.population_turn import GAIN, L_IDX, R_IDX, SIGN, TAU, estimate_b0
from loop.run import make_setup

MODES = ("stim_history_zero", "A_history_zero", "zero_from_start")
WINDOWS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0))


def audit_brain(brain) -> dict:
    """审计实际安装版本的动力学来源（来自实例，不靠别的源码）。"""
    b = getattr(brain, "_b", brain)              # RealFlyBrain 包装了 flybrain.FlyBrain
    return {"class": type(b).__module__ + "." + type(b).__name__,
            "dt": float(b.dt), "tau": float(b.tau), "gain": float(b.gain),
            "tonic": float(b.tonic), "noise_hz": float(b.noise_hz),
            "noise_amp": float(b.noise_amp), "decay": float(b.decay),
            "refractory_steps": int(b.refractory_steps), "sensory_input": bool(b.sensory_input),
            "order": "synaptic(prev fired)*gain -> v*=decay -> +tonic -> +noise -> +eye -> +inject -> spike"}


def run_bg(cfg, brain, groups, dn_idx, seed, rec, fork, mode, *, b0, gain=GAIN, sign=SIGN, tau=TAU,
           n_total=600, mpc=0.40):
    """分叉后 10s（n_total 步）；返回每步记录。"""
    from flybrain import Trace
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=mpc)
    tr = Trace(brain, idx=np.asarray(dn_idx)[np.array(list(L_IDX) + list(R_IDX), int)], tau=tau)
    brain.reset(seed)
    dt = brain.dt
    n = min(n_total, len(rec["t"]))
    win = max(1, int(round(0.25 / dt)))
    bl, br = np.asarray(dn_idx)[np.array(list(L_IDX), int)], np.asarray(dn_idx)[np.array(list(R_IDX), int)]
    sl = set(bl.tolist())
    sr = set(br.tolist())
    spikeL, spikeR = [], []
    Ds, raw, turn = [], [], []
    for k in range(n):
        if k < fork:
            f = frame_at(rec, k, "orig") if mode == "stim_history_zero" else frame_at(rec, k, "none")
            d = enc.sense(f)
            enc.update(f, d)
            inj = [] if mode == "zero_from_start" else enc.inject()[0]
        else:
            f = frame_at(rec, k, "none")
            d = enc.sense(f)
            enc.update(f, d)
            inj = []
        fired = brain.step(inject=inj)
        fl = np.isin(fired, bl).sum() if len(fired) else 0
        fr = np.isin(fired, br).sum() if len(fired) else 0
        spikeL.append(int(fl))
        spikeR.append(int(fr))
        a = np.asarray(tr.observe(fired), np.float32)
        Dv = float(a[:len(L_IDX)].mean() - a[len(L_IDX):].mean())
        Ds.append(Dv)
        raw.append(sign * gain * (Dv - b0))
        turn.append(float(np.clip(sign * gain * (Dv - b0), -1.0, 1.0)))
    out = {"t": np.arange(n) * dt, "D": np.array(Ds), "Dmb": np.array(Ds) - b0,
           "raw": np.array(raw), "turn": np.array(turn),
           "spikeL": np.array(spikeL), "spikeR": np.array(spikeR), "fork": fork, "dt": dt,
           "win": win}
    return out


def win_stats(x, fork, dt, lo, hi):
    i0, i1 = fork + int(round(lo / dt)), fork + int(round(hi / dt))
    seg = x[i0:i1]
    return float(np.mean(seg)) if len(seg) else 0.0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1001:must_right,808:must_right,606:must_right")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="real", map_name="gen", dt=None, device=args.device, seed=1001,
        preset=None, sensory_input=False, make_decoder=False, verbose=False)
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    from loop.dynamic_tof_sim import load_rig
    rig = load_rig()
    dt = brain.dt
    a = audit_brain(brain)
    print("[P2d-20] flybrain 审计:", a["class"], f"| dt={a['dt']} tau={a['tau']} gain={a['gain']} "
          f"tonic={a['tonic']:.3f} noise_hz={a['noise_hz']} noise_amp={a['noise_amp']} "
          f"refractory_steps={a['refractory_steps']} sensory_input={a['sensory_input']}")
    print("         运算顺序:", a["order"])
    print("         → inject=[] 仍保留 recurrent+tonic+noise；'零注入'≠'无神经输入'\n")
    b0s = {}
    for case in args.cases.split(","):
        sd_s, sc = case.split(":")
        seed = int(sd_s)
        if seed not in b0s:
            b0s[seed] = estimate_b0(cfg, brain, groups, dn_idx, seed)
        b0 = b0s[seed]
        rec = run_diag(cfg, brain, groups, dn_idx, seed, SCENARIOS[sc], seconds=args.seconds,
                       rig=rig, b0=b0)
        fork = pick_fork(rec["turn"], rec["t"])
        n_total = min(fork + int(round(10.0 / dt)), len(rec["t"]))
        tag = "成功" if rec["reach"] else "失败"
        print(f"=== seed {seed} / {sc} [{tag}] b0={b0:+.3f} fork_t={rec['t'][fork]:.1f}s "
              f"（观察分叉后 10s）===")
        res = {}
        okA = False
        for mode in MODES:
            res[mode] = run_bg(cfg, brain, groups, dn_idx, seed, rec, fork, mode, b0=b0,
                               n_total=n_total)
            if mode == "stim_history_zero":
                okA = (np.allclose(res[mode]["D"][:fork], rec["D"][:fork], atol=1e-4)
                       and np.allclose(res[mode]["turn"][:fork], rec["turn"][:fork], atol=1e-4))
        print(f"  A 分叉前一致: {okA}")
        print(f"  {'mode':18s} {'win':8s} {'D-b0(sgn)':>10s} {'|D-b0|':>8s} {'|turn|':>7s} "
              f"{'sat%':>5s} {'spkL/步':>8s} {'spkR/步':>8s}")
        for mode in MODES:
            r = res[mode]
            for lo, hi in WINDOWS:
                dm = win_stats(r["Dmb"], fork, dt, lo, hi)
                ad = win_stats(np.abs(r["Dmb"]), fork, dt, lo, hi)
                ta = win_stats(np.abs(r["turn"]), fork, dt, lo, hi)
                sat = win_stats((np.abs(r["turn"]) >= 0.999).astype(float), fork, dt, lo, hi) * 100
                skl = win_stats(r["spikeL"].astype(float), fork, dt, lo, hi)
                skr = win_stats(r["spikeR"].astype(float), fork, dt, lo, hi)
                print(f"  {mode:18s} {f'{lo:.0f}-{hi:.0f}s':8s} {dm:+10.3f} {ad:8.3f} "
                      f"{ta:7.3f} {sat:5.0f} {skl:8.2f} {skr:8.2f}")
        # A 相对 B/C 的额外活动（绝对 D-b0），按窗口
        print("  A 相对 B/C 的额外 |D-b0|（同窗口）:")
        for lo, hi in WINDOWS:
            da = win_stats(np.abs(res["stim_history_zero"]["Dmb"]), fork, dt, lo, hi)
            db = win_stats(np.abs(res["A_history_zero"]["Dmb"]), fork, dt, lo, hi)
            dcz = win_stats(np.abs(res["zero_from_start"]["Dmb"]), fork, dt, lo, hi)
            print(f"    {lo:.0f}-{hi:.0f}s: A={da:.3f} B={db:.3f} C={dcz:.3f}  "
                  f"A-B={da - db:+.3f} A-C={da - dcz:+.3f}")
        # 40s 无遮挡基线漂移经 gain=8 对应的转向偏差
        Dd, _td = run_A_drift(cfg, brain, groups, dn_idx, seed, b0, seconds=40.0)
        Dm = Dd - b0
        drift = float(np.mean(Dm[int(35 / dt):]) - np.mean(Dm[:int(1 / dt)]))
        print(f"  40s A 基线漂移 Δ(D-b0)={drift:+.3f} → 经 gain=8 对应转向偏差 "
              f"{SIGN * 8 * drift:+.2f}（±1 限幅前）")
        if args.out_dir:
            import os
            os.makedirs(args.out_dir, exist_ok=True)
            _fig(res, fork, dt, seed, sc, os.path.join(args.out_dir, f"p2d20_{seed}_{sc}.png"))
        print()
    return 0


def _fig(res, fork, dt, seed, sc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for mode, r in res.items():
        t = r["t"]
        ax[0].plot(t, r["Dmb"], lw=1.0, label=mode)
        ax[1].plot(t, r["turn"], lw=1.0, label=mode)
        ax[2].plot(t, r["spikeL"] + r["spikeR"], lw=0.8, label=mode)
    ax[0].axvline(fork * dt, color="k", ls=":", lw=1)
    for bx, ttl in zip(ax, ("D(t)-b0  [virtual sim]", "turn_cmd  [virtual sim]",
                            "L50+R50 spikes/step  [virtual sim]")):
        bx.set_title(ttl + f"\nseed {seed}/{sc}, fork={fork * dt:.1f}s")
        bx.set_xlabel("t (s)")
        bx.legend(fontsize=8)
        bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
