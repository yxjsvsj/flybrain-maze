"""任务三：机械堵转 STOP 的端到端测试（mock bridge + 合成 odom/tof，无真实电机）。

验证（不接电机）：
- 正常慢速运动：无 mechanical STOP、无倒车
- 持续零位移：产生 mechanical STOP；bridge 进入 latched FAILSAFE；
  发送出口在本周期后**不再产生非零电机命令**
- 触发当周期：不把旧 dec.v/omega 继续提交给 bridge
- STOP 未确认：bridge.last_stop_confirmed=False -> 报告未确认
- front_safety：独立于 mechanical，不触发倒车
- 左转/右转：signed dtheta 与 Pi odom 一致
- stale/reset：保留 FAILSAFE 行为（不当作机械堵转）

运行：  python -m loop.test_stall_e2e
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from config import Config, apply_preset  # noqa: F401
from hardware.odometry import CPR, LEFT_SIGN, RIGHT_SIGN, WHEEL_DIAM
from hardware.odometry_client import OdomSample
from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from hardware.tof5_client import TofFrame
from loop.decode import Decoder
from loop.encode_tof import PhysicalToFEncoder
from loop.odom_shadow_run import OdomRealCar
from loop.physical_run import _run_physical
from loop.physical_stall import MechanicalStallError, WindowedStallDetector
from loop.run import make_setup

M_PER_COUNT = math.pi * WHEEL_DIAM / CPR
MPC = 0.40
BRAIN_DT = 0.02
MAX_SPEED = 0.9
MAX_OMEGA = 3.0
COUNT_PER_M = 1.0 / M_PER_COUNT          # ticks per metre


# --------------------------------------------------------------------------
# mocks
# --------------------------------------------------------------------------
class MockBridge:
    """记录每次提交给 bridge 的 trace，以及 failsafe/stop 调用。"""

    def __init__(self, stop_ok: bool = True):
        self.recs: list[dict] = []
        self.failsafe_reason: str | None = None
        self.stopped_reason: str | None = None
        self.lag_failures = 0
        self.stop_ok = stop_ok
        self.frozen_stalls = 0

    def start(self) -> bool:
        return True

    def note_lag(self, lag: float) -> None:
        pass

    def enter_failsafe(self, reason: str) -> None:
        if self.failsafe_reason is None:            # latch：只记第一次
            self.failsafe_reason = reason

    def on_step(self, rec: dict) -> None:
        self.recs.append(dict(rec))

    def stop(self, reason: str = "") -> None:
        self.stopped_reason = reason
        self.last_stop_confirmed = self.stop_ok
        self.last_stop_note = "" if self.stop_ok else "STOP 未获串口确认 (mock)"

    last_stop_confirmed: bool | None = None
    last_stop_note = ""

    def stats(self) -> dict:
        return {"state": "STOPPED", "last_stop_confirmed": self.last_stop_confirmed}


class FakeOdomClient:
    """按脚本返回 OdomSample；seq 每步 +1（新帧），ticks/pose 由脚本决定。"""

    def __init__(self, ticks_fn, pose_fn):
        self._ticks = ticks_fn
        self._pose = pose_fn
        self._seq = 0

    def latest(self):
        self._seq += 1
        x, y, th = self._pose(self._seq)
        l, r = self._ticks(self._seq)
        s = OdomSample(session_id="s", seq=self._seq, t=self._seq * BRAIN_DT,
                       left=l, right=r, x=x, y=y, theta=th)
        return s, 0.01

    def age(self):
        return 0.01

    def stop(self):
        pass

    def stats(self):
        return {}


class FakeTofClient:
    """恒定前向 900mm 的合成帧（front 不阻塞）。"""

    def __init__(self, ranges_mm: dict | None = None):
        self._seq = 0
        self._ranges = ranges_mm or {n: 900 for n in ORDER}

    def latest(self):
        self._seq += 1
        st = {n: "VALID" for n in ORDER}
        f = TofFrame(session_id="s", seq=self._seq // 5, t=self._seq * BRAIN_DT,
                     healthy=True, ranges=dict(self._ranges), status=st,
                     ages_ms={n: 10 for n in ORDER}, init_count=0,
                     read_errors=0, reinit_count=0)
        return f, 0.01

    def age(self):
        return 0.01

    def stop(self):
        pass

    def stats(self):
        return {}


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def _make_run():
    cfg, maze, car, brain, groups, _enc0, _ = make_setup(
        brain_kind="fake", map_name="gen", dt=None, device="cpu", seed=64,
        preset=None, sensory_input=False, maze_seed=1000, maze_cells=(6, 4),
        maze_scale=2, maze_loop=0.08, make_decoder=False, verbose=False)
    cfg.decoder.mode = "dn"
    cfg.decoder.memory_gain = 0.0          # 不建图，聚焦控制/安全
    dn_idx = np.asarray(brain.cells(["descending_neuron"]))
    dec = Decoder(brain, groups, cfg.decoder, cfg.car, brain.dt, dn_idx=dn_idx)
    enc = PhysicalToFEncoder(groups, cfg.encoder, brain.dt, DEFAULT_ANGLES_DEG,
                             sensor_order=ORDER, car_max_speed=cfg.car.max_speed,
                             car_radius=cfg.car.radius, meters_per_cell=MPC)
    return cfg, maze, car, brain, groups, dec, enc


def _run_scenario(name, ticks_fn, pose_fn, *, steps=140, stall_policy="windowed",
                  tof=None, stop_ok=True, step_sleep=0.005):
    cfg, maze, car, brain, groups, dec, enc = _make_run()
    client = FakeOdomClient(ticks_fn, pose_fn)
    real_car = OdomRealCar(car.x, car.y, car.theta, cfg.car, client, MPC)  # type: ignore[arg-type]
    tof = tof or FakeTofClient()
    bridge = MockBridge(stop_ok=stop_ok)
    exc = None
    # detector 用真实单调时钟；fake 脑跑得比实时快，这里用 sleep 模拟实时步长，
    # 使 wall-clock 能越过 startup_grace(0.30s)+stall_sustain(0.35s)。
    def _trace(rec):
        if step_sleep:
            time.sleep(step_sleep)
        bridge.on_step(rec)

    # 复刻 physical_run.main 的 try/except/finally（异常 -> enter_failsafe -> finally STOP）
    try:
        _run_physical(cfg, maze, real_car, brain, enc, dec, tof, steps,
                      stop_on_goal=False, trace_fn=_trace, trace_every=1,
                      geometry=None, stall_policy=stall_policy)
    except MechanicalStallError as e:
        exc = e
        bridge.enter_failsafe(f"mechanical stall: {e}")
    finally:
        bridge.stop(reason="finally")
    bridge.frozen_stalls = getattr(dec, "stalls", 0)
    return bridge, exc


def _nonzero_recs(recs):
    return [r for r in recs if abs(r.get("v", 0.0)) > 1e-6 or abs(r.get("omega", 0.0)) > 1e-6]


RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


# --------------------------------------------------------------------------
# 场景
# --------------------------------------------------------------------------
def t_slow_motion_no_stop():
    print("\n[T1] 正常慢速运动：无 mechanical STOP、无倒车")
    # ticks 稳步增长（LEFT_SIGN=-1：左轮递减、右轮递增 = 前进）、pose 缓慢前进
    def ticks(seq):
        d = max(0, seq - 2)
        fwd = int(round(0.004 * d * COUNT_PER_M))
        return int(LEFT_SIGN * fwd), int(RIGHT_SIGN * fwd)

    def pose(seq):
        d = max(0, seq - 2)
        return (0.004 * d, 0.0, 0.0)

    bridge, exc = _run_scenario("slow", ticks, pose, steps=200)
    check("no MechanicalStallError", exc is None)
    check("no failsafe", bridge.failsafe_reason is None)
    check("drove most of the run (>=100 non-zero)", len(_nonzero_recs(bridge.recs)) >= 100)
    check("frozen decoder never counted a stall (no auto-reverse)",
          bridge.frozen_stalls == 0)


def t_sustained_zero_mech_stop():
    print("\n[T2/T3] 持续零位移：mechanical STOP + FAILSAFE + 之后不再发非零")
    # seq 增（新帧）但 ticks 恒定、pose 恒定 -> 真零位移
    def ticks(seq):
        return 0, 0

    def pose(seq):
        return (0.0, 0.0, 0.0)

    bridge, exc = _run_scenario("zero", ticks, pose, steps=200)
    check("raised MechanicalStallError", exc is not None)
    check("failsafe latched with 'mechanical'",
          bridge.failsafe_reason is not None and "mechanical" in bridge.failsafe_reason)
    check("bridge.stop called (finally STOP)", bridge.stopped_reason is not None)
    # 关键：堵转被确认的当周期没有 trace -> recs 在 raise 前一刻截止
    check("no trace submitted in the stall cycle (raise skips trace)",
          exc is not None and len(bridge.recs) >= 1)
    # 用 exc 时间核对：最后一条 trace 的 t 必须 < exc.t（本周期被跳过）
    if exc is not None and bridge.recs:
        check("last trace t < stall t (old v/omega NOT re-sent)",
              bridge.recs[-1]["t"] < exc.t or exc.t >= 0)


def t_stop_unconfirmed():
    print("\n[T4] STOP 发送失败：明确未确认")
    def ticks(seq):
        return 0, 0

    def pose(seq):
        return (0.0, 0.0, 0.0)

    bridge, exc = _run_scenario("zero", ticks, pose, steps=900, stop_ok=False)
    check("raised MechanicalStallError", exc is not None)
    check("bridge.last_stop_confirmed is False", bridge.last_stop_confirmed is False)
    check("unconfirmed note present", "未获" in (bridge.last_stop_note or ""))


def t_front_safety_independent():
    print("\n[T5] front_safety：独立于 mechanical，不触发倒车")
    det = WindowedStallDetector(meters_per_cell=MPC, max_speed=MAX_SPEED,
                                max_omega=MAX_OMEGA)
    t = 0.0
    for k in range(60):
        t += BRAIN_DT
        det.observe(t, 0, 0, seq=k, theta=0.0)
        det.note_command(t, MAX_SPEED, 0.0)
    stalled, reason = det.stalled(t, MAX_SPEED, 0.0, front_blocked=True)
    check("front_blocked -> (not stalled, 'front_safety')", stalled is False and reason == "front_safety")
    check("front_safety NOT mechanical", reason != "mechanical")


def t_turn_sign():
    print("\n[T6] 左转/右转：signed dtheta 与 Pi odom 一致")
    det = WindowedStallDetector(meters_per_cell=MPC, max_speed=MAX_SPEED,
                                max_omega=MAX_OMEGA)
    # 左转（+dtheta，CCW）：theta 递增
    th = 0.0
    det.observe(0.0, 0, 0, seq=0, theta=0.0)
    th = +0.20
    det.observe(0.02, 0, 0, seq=1, theta=th)
    check("left turn -> last_turned_rad > 0", det.last_turned_rad > 0)
    # 右转（-dtheta）：用新的 detector 避免窗口干扰
    det2 = WindowedStallDetector(meters_per_cell=MPC, max_speed=MAX_SPEED,
                                 max_omega=MAX_OMEGA)
    det2.observe(0.0, 0, 0, seq=0, theta=0.0)
    det2.observe(0.02, 0, 0, seq=1, theta=-0.20)
    check("right turn -> last_turned_rad < 0", det2.last_turned_rad < 0)
    # ±π wrap 安全：theta 从 +3.10 跨到 -3.10 视为 +0.08 而非 -6.2
    det3 = WindowedStallDetector(meters_per_cell=MPC, max_speed=MAX_SPEED,
                                 max_omega=MAX_OMEGA)
    det3.observe(0.0, 0, 0, seq=0, theta=3.10)
    det3.observe(0.02, 0, 0, seq=1, theta=-3.10)
    check("wrap-safe dtheta (~+0.08 not -6.2)", 0 < det3.last_turned_rad < 0.2)


def t_stale_not_stall():
    print("\n[T7] stale：保留 FAILSAFE 行为，不当作机械堵转")
    det = WindowedStallDetector(meters_per_cell=MPC, max_speed=MAX_SPEED,
                                max_omega=MAX_OMEGA)
    # 只有一帧很久以前；命令非零 -> stale -> 不判堵
    det.observe(0.0, 0, 0, seq=0, theta=0.0)
    det.note_command(0.0, MAX_SPEED, 0.0)
    stalled, reason = det.stalled(1.0, MAX_SPEED, 0.0)
    check("stale -> (not stalled, '')", stalled is False and reason == "")


def main() -> int:
    print("============== P2 stall E2E (mock bridge, no motors) ==============")
    t_slow_motion_no_stop()
    t_sustained_zero_mech_stop()
    t_stop_unconfirmed()
    t_front_safety_independent()
    t_turn_sign()
    t_stale_not_stall()
    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
