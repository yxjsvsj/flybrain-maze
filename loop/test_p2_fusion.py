"""P2d-12 任务二回归：P2Fusion 语义一致性（与冻结 readout 组合同式）。"""
from __future__ import annotations

import sys

import numpy as np

from config import Config
from loop.p2_fusion import P2Fusion

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def t_sequence_vs_frozen():
    """多步序列级：P2Fusion 必须与冻结 Decoder(readout) 逐步一致（含 front-block 与恢复）。"""
    import numpy as np
    from loop.decode import Decoder
    from loop.run import make_setup
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", device="cpu", make_decoder=False, verbose=False)
    cfg.decoder.mode = "readout"
    cfg.decoder.brain_gain = 0.15
    cfg.decoder.memory_gain = 1.0
    cfg.decoder.readout_path = "readout_real.npz"
    dn = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn)

    seq = [(0.3, 0.1, 5.0), (0.5, -0.2, 5.0), (-0.4, 0.3, 5.0),
           (0.0, 0.0, cfg.decoder.front_stop_dist - 1e-3),      # front hard stop
           (0.2, 0.0, 5.0), (0.1, 0.1, 0.9), (0.0, 0.0, 5.0)]   # recovery

    class Stub:
        def __init__(self, vals):
            self.vals, self.i = vals, 0

        def predict(self, feats):
            v = self.vals[min(self.i, len(self.vals) - 1)]
            self.i += 1
            return v

    dec.readout = Stub([s[0] for s in seq])
    fuse = P2Fusion(cfg.decoder, cfg.car)
    empty = np.empty(0, int)
    ok, bad = True, None
    for bt, pu, df in seq:
        vd, od = dec.update(empty, info={"pursuit_turn": pu, "dmin_F": df})
        vf, of = fuse.step(brain_turn=bt, pursuit_turn=pu, dmin_F=df)
        if abs(vd - vf) > 1e-9 or abs(od - of) > 1e-9:
            ok, bad = False, (bt, pu, df, (vd, od), (vf, of))
    check("sequence: P2Fusion == frozen Decoder (v,omega) incl front-block+recovery", ok)
    if bad:
        print("    mismatch:", bad)

    # bias/sham/gate 经 P2Fusion 输出应有限且符号合理
    fuse.reset()
    fb = P2Fusion(cfg.decoder, cfg.car)
    _, om_bias = fb.step(brain_turn=-0.267, pursuit_turn=0.0, mode="bias", bias=-0.267, bg=0.15, mg=0.0)
    check("bias mode at bt==bias -> omega ~0", abs(om_bias) < 1e-9)


def _fuse():
    cfg = Config()
    cfg.decoder.mode = "readout"
    cfg.decoder.brain_gain = 0.15
    cfg.decoder.memory_gain = 1.0
    return P2Fusion(cfg.decoder, cfg.car), cfg


def _smooth_steady(turns, a, dt_target_fn):
    v = 0.0
    for _ in range(400):
        v = (1 - a) * v + a * dt_target_fn()
    return v


def main() -> int:
    print("=========== P2Fusion semantics ===========")
    f, cfg = _fuse()
    a = cfg.decoder.smooth

    # 1) 零增益：bg=mg=0 -> turn=0 -> omega 稳态 0
    for _ in range(400):
        f.step(brain_turn=0.9, pursuit_turn=0.5, bg=0.0, mg=0.0)
    check("zero gain -> omega ~0", abs(f.omega) < 1e-9)

    # 2) 平滑/裁剪一致性：常量 brain_turn=0.5, bg=0.15 -> turn=0.075 -> 稳态 omega
    f.reset()
    for _ in range(400):
        f.step(brain_turn=0.5, pursuit_turn=0.0, bg=0.15, mg=0.0)
    expect = cfg.car.max_omega * 0.15 * 0.5
    check(f"plain steady omega == max_omega*bg*bt ({f.omega:.5f} vs {expect:.5f})",
          abs(f.omega - expect) < 1e-6)

    # 3) bias 校正：bt 常量 c，bias=c -> turn=0 -> omega 稳态 0（去常驻）
    f.reset()
    for _ in range(400):
        f.step(brain_turn=-0.267, pursuit_turn=0.0, mode="bias", bias=-0.267, bg=0.15, mg=0.0)
    check("bias-corrected constant == bias -> omega ~0", abs(f.omega) < 1e-6)

    # 4) bias 校正：常值 bt=0.5, bias=-0.267 -> turn=0.15*(0.767) -> 稳态
    f.reset()
    for _ in range(400):
        f.step(brain_turn=0.5, pursuit_turn=0.0, mode="bias", bias=-0.267, bg=0.15, mg=0.0)
    exp4 = cfg.car.max_omega * float(np.clip(0.15 * (0.5 + 0.267), -1, 1))
    check(f"bias steady == max_omega*clip(bg*(bt-bias)) ({f.omega:.5f} vs {exp4:.5f})",
          abs(f.omega - exp4) < 1e-6)

    # 5) SHAM：时变 bt 被替换为常数 bias -> 稳态只由 bias 决定
    f.reset()
    for k in range(400):
        f.step(brain_turn=(0.9 if k % 2 else -0.9), pursuit_turn=0.0, mode="sham",
               bias=-0.267, bg=0.15, mg=0.0)
    exp5 = cfg.car.max_omega * float(np.clip(0.15 * (-0.267), -1, 1))
    check(f"sham steady uses constant bias ({f.omega:.5f} vs {exp5:.5f})",
          abs(f.omega - exp5) < 1e-6)

    # 6) 条件门控：神经与 pursuit 反向 -> bg 降为 0.25x
    f.reset()
    _, om_opp = f.step(brain_turn=+0.5, pursuit_turn=-0.5, mode="cond", bg=0.15, mg=1.0)
    turn_opp = float(np.clip(0.15 * 0.25 * 0.5 + 1.0 * (-0.5), -1, 1))
    check("cond gate reduces neural when opposing", abs(om_opp - cfg.car.max_omega * a * turn_opp) < 1e-6)

    # 7) 前向安全：dmin_F <= stop_dist -> v=0
    f.reset()
    v7, _ = f.step(brain_turn=0.0, pursuit_turn=0.0, dmin_F=cfg.decoder.front_stop_dist - 0.01)
    check("front_stop -> v=0", abs(v7) < 1e-9)

    # 8) 首次步与冻结平滑公式一致
    f.reset()
    v8, om8 = f.step(brain_turn=0.4, pursuit_turn=0.2, bg=0.15, mg=1.0)
    turn8 = float(np.clip(0.15 * 0.4 + 1.0 * 0.2, -1, 1))
    check("first step == smooth*a*omega", abs(om8 - cfg.car.max_omega * a * turn8) < 1e-12)

    t_sequence_vs_frozen()

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
