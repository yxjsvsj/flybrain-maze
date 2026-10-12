"""P2d-12a 任务A回归：train_p2_readout 时间正确性（~10Hz ToF + 50Hz 脑 + ZOH）。"""
from __future__ import annotations

import sys

import numpy as np

from loop.dynamic_tof_sim import load_rig
from loop.run import make_setup
from loop.train_p2_readout import rollout, teacher_turn

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== train_p2_readout time semantics ===========")
    cfg, _m, _c, brain, groups, _e, _ = make_setup(
        brain_kind="fake", map_name="gen", device="cpu", make_decoder=False, verbose=False)
    cfg.decoder.mode = "readout"
    dn = np.asarray(brain.cells(["descending_neuron"]))
    rig = load_rig()

    def drive(info, feats, turn_t, cfg):
        return cfg.car.max_speed * cfg.decoder.readout_fwd_frac, 0.0

    seconds, tof_hz = 2.0, 10.0
    X, Y, meta = rollout(cfg, brain, groups, dn, None, [999], seconds, brain.dt,
                         rig, 0.40, drive, tof_hz=tof_hz)
    exp_frames = int(round(seconds * tof_hz))
    exp_steps = int(round(seconds / brain.dt))
    check(f"tof_frames ≈ seconds*tof_hz ({meta['tof_frames']} vs {exp_frames})",
          abs(meta["tof_frames"] - exp_frames) <= 1)
    check(f"brain_steps ≈ seconds/dt ({meta['brain_steps']} vs {exp_steps})",
          abs(meta["brain_steps"] - exp_steps) <= 1)
    check("samples == brain_steps (Trace every brain step)", X.shape[0] == meta["brain_steps"])
    check("feats dim == dn count", X.shape[1] == len(dn))
    check("labels finite", bool(np.all(np.isfinite(Y))))
    # 老师：无障碍≈0；FR 近 -> 左(>0)；FL 近 -> 右(<0)
    check("teacher FR near -> left (prox_r>prox_l)",
          teacher_turn({"prox_r": 0.6, "prox_l": 0.1}) > 0)
    check("teacher FL near -> right", teacher_turn({"prox_r": 0.1, "prox_l": 0.6}) < 0)
    check("teacher no obstacle ~0", abs(teacher_turn({"prox_r": 0.0, "prox_l": 0.0})) < 1e-9)
    check("teacher F-only (prox_r==prox_l) -> 0", abs(teacher_turn({"prox_r": 0.5, "prox_l": 0.5})) < 1e-9)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
