"""P2 预热离线验证（无硬件、无运动）。

A. FakeBrain：`reset(seed)` 后与"从未预热"的初态逐位一致（预热不得改变正式初态）。
B. helper 机制 + 失败语义：预热/复位失败必须抛 WarmupError（调用方据此拒绝启动 Bridge）。
C. RealFlyBrain（若可用）：预热吸收 ~0.75–1s 冷启动后，**首个正式 brain.step** 应为热态。

运行：  python -m loop.test_warmup
"""
from __future__ import annotations

import sys

import numpy as np

from brain.wrap import FakeBrain
from loop.p2_warmup import WarmupError, reset_after_warmup, run_warmup

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _Enc:
    def sense(self, frame):
        return np.zeros(5, np.float32)

    def update(self, frame, dists):
        return True

    def inject(self):
        return [], {}


class _Dec:
    def update(self, fired, info=None, stalled=False):
        return 0.0, 0.0


class _BadStep:
    dt = 0.02

    def step(self, *a, **k):
        raise RuntimeError("boom")


class _BadReset:
    def reset(self, seed=None):
        raise RuntimeError("nope")


def t_fake_reset_parity():
    print("\n[A] FakeBrain: warmup+reset 后与原始初态一致")
    K, DT = 40, 0.02
    inj = [(np.arange(3), 0.5)]
    base_b = FakeBrain(dt=DT, seed=64)
    base = [base_b.step(inject=inj).tolist() for _ in range(K)]

    warm_b = FakeBrain(dt=DT, seed=64)
    for _ in range(7):                                  # 预热（异注入）
        warm_b.step(inject=[(np.arange(1), 0.3)])
    warm_b.reset(64)
    after = [warm_b.step(inject=inj).tolist() for _ in range(K)]

    check("fired sequences identical after reset", base == after)
    check(f"seed preserved (seed={warm_b.seed})", warm_b.seed == 64)


def t_helper_and_failures():
    print("\n[B] helper 机制 + 失败 -> WarmupError")
    wt = run_warmup(brain=FakeBrain(dt=0.02, seed=64), enc=_Enc(), dec=_Dec(), steps=3)
    check("warmup runs N steps", wt["warmup_steps"] == 3)
    check("warmup records first-step ms", wt["warmup_first_step_ms"] is not None)

    zero = run_warmup(brain=FakeBrain(dt=0.02, seed=64), enc=_Enc(), dec=_Dec(), steps=0)
    check("steps=0 is a no-op", zero["warmup_steps"] == 0)

    raised = False
    try:
        run_warmup(brain=_BadStep(), enc=_Enc(), dec=_Dec(), steps=1)
    except WarmupError:
        raised = True
    check("brain.step failure -> WarmupError", raised)

    raised = False
    try:
        reset_after_warmup(brain=_BadReset(), seed=64)
    except WarmupError:
        raised = True
    check("brain.reset failure -> WarmupError", raised)


def t_real_cold_absorbed():
    print("\n[C] RealFlyBrain: 预热吸收冷启动，首个正式 step 为热态（若 CUDA 可用）")
    try:
        from loop.run import make_setup
        from loop.decode import Decoder
        from loop.encode_tof import PhysicalToFEncoder
        from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER

        cfg, maze, car, brain, groups, _e0, _ = make_setup(
            brain_kind="real", map_name="gen", dt=None, device="cuda", seed=64,
            preset=None, sensory_input=False, maze_seed=1000, maze_cells=(6, 4),
            maze_scale=2, maze_loop=0.08, make_decoder=False, verbose=False)
        cfg.decoder.mode = "readout"
        cfg.decoder.memory_gain = 0.0
        cfg.decoder.brain_gain = 0.0
        cfg.decoder.readout_path = "readout_real.npz"
        dn_idx = np.asarray(brain.cells(["descending_neuron"]))

        def build():
            e = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                                   sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                                   car_radius=cfg.car.radius, meters_per_cell=0.40)
            d = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
            return e, d

        enc, dec = build()
        wt = run_warmup(brain=brain, enc=enc, dec=dec, steps=1)
        reset_after_warmup(brain=brain, seed=64)
        enc, dec = build()
        import time
        t = time.perf_counter()
        brain.step(inject=[])
        formal_ms = (time.perf_counter() - t) * 1e3
        check(f"warmup absorbed cold (warmup first step {wt['warmup_first_step_ms']:.0f} ms)",
              (wt["warmup_first_step_ms"] or 0) > 50)
        check(f"first FORMAL brain.step is warm ({formal_ms:.1f} ms < 50)",
              formal_ms < 50)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [SKIP] RealFlyBrain section unavailable: {type(exc).__name__}: {exc}")


def main() -> int:
    print("============== P2 warmup offline validation ==============")
    t_fake_reset_parity()
    t_helper_and_failures()
    t_real_cold_absorbed()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
