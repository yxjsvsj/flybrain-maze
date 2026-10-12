"""P2 预热 helper：实体电机控制前预热 RealFlyBrain/CUDA 计算链。

背景
----
fresh 进程里 `brain.step` 首次调用要 ~0.75–1.0 s（CUDA 首次 kernel/module 初始化），
其后 ~1.6 ms。实体运行器把它放在控制循环的 step 0，会顶住实时节拍、延迟首个 DRIVE 帧。

做什么
------
`run_warmup()`：用**代表性合成 ToF 帧**跑完整链
    PhysicalToFEncoder.sense/update/inject -> brain.step -> Decoder.update(Trace/readout)
**不发 HTTP、不碰电机、不改占用地图**。

`reset_after_warmup()`：`brain.reset(seed)` 回到原始初态；调用方随后必须**重建**
encoder 与 Decoder（清掉预热产生的缓存/Trace/历史）。

失败即抛 `WarmupError`——调用方必须据此**拒绝启动 Bridge**，绝不继续实体运动。
"""
from __future__ import annotations

import time

from hardware.tof5 import ORDER
from hardware.tof5_client import TofFrame


class WarmupError(RuntimeError):
    """预热或复位失败。调用方必须拒绝启动，不得继续实体运动。"""


def _warm_frame(seq: int, t: float, mm: float = 900.0) -> TofFrame:
    st = {n: "VALID" for n in ORDER}
    return TofFrame(session_id="warmup", seq=seq, t=t, healthy=True,
                    ranges={n: mm for n in ORDER}, status=st,
                    ages_ms={n: 1.0 for n in ORDER},
                    init_count=0, read_errors=0, reinit_count=0)


def run_warmup(*, brain, enc, dec, steps: int, frames=None) -> dict:
    """跑 steps 步代表性计算链。返回 timing dict；异常 -> WarmupError。

    enc/dec 与正式运行同一套构造（预热后会被重建，见 reset_after_warmup）。
    frames 可选：覆盖每步的合成帧（例如带上真实感觉输入路径）。
    """
    if steps <= 0:
        return {"warmup_steps": 0, "warmup_first_step_ms": None, "warmup_total_s": 0.0}
    t0 = time.monotonic()
    first_ms = None
    try:
        for i in range(steps):
            frame = frames[i] if frames is not None else _warm_frame(i, i * float(brain.dt))
            dists = enc.sense(frame)
            enc.update(frame, dists)
            inject, info = enc.inject()
            tb = time.monotonic()
            fired = brain.step(inject=inject)
            if first_ms is None:
                first_ms = (time.monotonic() - tb) * 1e3
            dec.update(fired, info=info, stalled=False)
    except Exception as exc:                                   # noqa: BLE001
        raise WarmupError(f"warmup step failed: {type(exc).__name__}: {exc}") from exc
    return {"warmup_steps": steps, "warmup_first_step_ms": first_ms,
            "warmup_total_s": time.monotonic() - t0}


def reset_after_warmup(*, brain, seed) -> None:
    """brain.reset(seed) 回到原始初态。失败 -> WarmupError（调用方必须拒绝启动）。"""
    try:
        brain.reset(seed)
    except Exception as exc:                                   # noqa: BLE001
        raise WarmupError(f"brain.reset failed: {type(exc).__name__}: {exc}") from exc
