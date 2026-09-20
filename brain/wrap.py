"""两种"脑"，同一个接口：

    brain.cells(types, side=None) -> 索引数组
    brain.step(inject=[(idx, 电压), ...]) -> 本步发放的神经元索引
    brain.reset(seed=None)
    brain.n, brain.dt

FakeBrain 用来在下载 260MB 连接组之前把闭环跑通；RealFlyBrain 是真货。
两者动力学公式一致（v <- decay*v + gain*W@spikes + tonic + noise，过阈值发放并清零），
所以解码器的增益在两者之间大体可以平移。
"""
from __future__ import annotations

import numpy as np


class FakeBrain:
    """~380 个 LIF 神经元 + 手工接线。

    存在的唯一理由：让"传感器 -> 注入 -> 发放 -> 解码 -> 车轮"这条管道先跑通，
    这样调试时你能分清是管道坏了还是连接组的问题。

    接线故意模仿我们期望真脑表现出来的反射：
        loom_L -> turn_R   左侧有东西逼近 -> 右转
        loom_R -> turn_L
        threat_L/R -> escape
    真脑不需要这段代码，它的接线来自电镜。

    所有强度都用"稳态电压增量"表述（DRIVE / SYNAPSES 的单位），这样和 dt、decay
    解耦——按"每步注入电压"写的话，连续注入会把池子打到饱和。
    """

    tau = 0.100
    gain = 3.0
    tonic = 0.14
    noise_hz = 1.2
    noise_amp = 0.22

    POOLS = {
        "loom_L": 20, "loom_R": 20,
        "threat_L": 20, "threat_R": 20,
        "turn_L": 20, "turn_R": 20,
        "fwd": 20, "back": 20, "escape": 20,
        "bg": 200,
    }

    # MaleCNS 细胞类型名 -> (左侧池, 右侧池)；None 表示无侧别
    TYPE_MAP = {
        "LPLC2": ("loom_L", "loom_R"),
        "LC4": ("threat_L", "threat_R"),
        "LC10a": ("turn_L", "turn_R"),   # 真脑里 LC10a 通 DNa02，假脑直接接转向池
        "DNa02": ("turn_L", "turn_R"),
        "DNg100": ("fwd", None),
        "MDN": ("back", None),
        "DNp01": ("escape", None),
    }
    OUTPUT_POOLS = ["loom_L", "loom_R", "threat_L", "threat_R",
                    "turn_L", "turn_R", "fwd", "back", "escape"]

    # 常驻驱动：稳态电压增量。fwd 0.78 -> V_ss~1.55 -> ~10 Hz 前进张力
    DRIVE = {"fwd": 0.78, "turn_L": 0.05, "turn_R": 0.05}

    # (突触前池, 突触后池, 突触前每 100 Hz 发放带来的稳态电压增量)
    SYNAPSES = [
        ("loom_L", "turn_R", 4.0),      # 对侧避障
        ("loom_R", "turn_L", 4.0),
        # loom -> escape 必须很弱：真蝇里 LPLC2 确实通 DNp01，但让"贴墙"就触发逃逸
        # 会让车一路倒车。逃逸应该只由 threat（真正高速逼近）触发。
        ("loom_L", "escape", 0.2),
        ("loom_R", "escape", 0.2),
        ("threat_L", "escape", 8.0),    # 快速逼近 -> 逃逸
        ("threat_R", "escape", 8.0),
        ("escape", "turn_L", 2.0),
        ("escape", "turn_R", 2.0),
    ]

    def __init__(self, dt: float = 0.020, seed: int = 64):
        self.dt = float(dt)
        self.n = sum(self.POOLS.values())
        self.seed = seed
        self.batch = 1          # flybrain.Trace 会读这个属性
        self.decay = float(np.exp(-self.dt / self.tau))
        self.per_step = 1.0 - self.decay

        self._slices, start = {}, 0
        for name, count in self.POOLS.items():
            self._slices[name] = slice(start, start + count)
            start += count

        # 和 FlyBrain 对 tonic 的处理一致：保证稳态电压与 20ms 标定值相同
        scale = self.per_step / (1 - np.exp(-0.020 / self.tau))
        self._tonic = np.full(self.n, self.tonic * scale, np.float32)
        for pool, dv in self.DRIVE.items():
            self._tonic[self._slices[pool]] += dv * self.per_step

        self._W = self._wire()
        self.reset(seed)

    def _wire(self) -> np.ndarray:
        W = np.zeros((self.n, self.n), np.float32)
        S = self._slices
        for pre, post, dv in self.SYNAPSES:
            n_pre = S[pre].stop - S[pre].start
            # 前池以 f Hz 发放时每步电流 = n_pre * f*dt * w；稳态电压 = 电流/(1-decay)
            w = dv * self.per_step / (self.gain * n_pre * 100.0 * self.dt)
            W[S[post], S[pre]] += w
        # 一点背景耦合，别让网络死得太干净
        rng = np.random.default_rng(1234)
        bg = S["bg"]
        m = rng.random((bg.stop - bg.start, bg.stop - bg.start)) < 0.02
        W[bg, bg] += (m * 0.02).astype(np.float32)
        return W

    # ---- 接口 ----------------------------------------------------------------
    def cells(self, types, side=None) -> np.ndarray:
        out = []
        for t in types:
            if t == "descending_neuron":
                out += [self._slices[p] for p in self.OUTPUT_POOLS]
                continue
            entry = self.TYPE_MAP.get(t)
            if entry is None:
                continue
            left, right = entry
            if side == "L" and left:
                out.append(self._slices[left])
            elif side == "R" and right:
                out.append(self._slices[right])
            elif side is None:
                out += [self._slices[p] for p in (left, right) if p]
        if not out:
            return np.empty(0, np.int64)
        return np.concatenate([np.arange(s.start, s.stop) for s in out])

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.v = np.zeros(self.n, np.float32)
        self.spikes = np.zeros(self.n, np.float32)
        self.steps = 0

    def step(self, eye_drive=None, inject=()) -> np.ndarray:
        current = (self._W @ self.spikes) * self.gain
        self.v = self.v * self.decay + current + self._tonic
        self.v += (self.rng.random(self.n) < self.noise_hz * self.dt) * np.float32(self.noise_amp)
        for idx, amount in inject:
            idx = np.asarray(idx)
            if len(idx):
                self.v[idx] += amount
        fired = np.flatnonzero(self.v >= 1.0)
        self.v[fired] = 0.0
        self.spikes[:] = 0.0
        self.spikes[fired] = 1.0
        self.steps += 1
        return fired


class RealFlyBrain:
    """flybrain.FlyBrain 的薄封装：补齐 cells/step/reset 的一致签名，并打印信息。"""

    def __init__(self, cfg, data=None, verbose: bool = True):
        from flybrain import FlyBrain

        self.cfg = cfg
        self._b = FlyBrain(data=data, seed=cfg.seed, device=cfg.device,
                           dt=cfg.dt, sensory_input=cfg.sensory_input)
        self.n = self._b.n
        self.dt = self._b.dt
        self.batch = self._b.batch
        if verbose:
            print(f"[brain] MaleCNS {self.n} neurons, device={self._b.device}, "
                  f"dt={self.dt * 1000:.1f} ms, sensory_input={cfg.sensory_input}")

    def cells(self, types, side=None) -> np.ndarray:
        return self._b.cells(types, side=side)

    def reset(self, seed=None) -> None:
        self._b.reset(seed)

    def step(self, eye_drive=None, inject=()) -> np.ndarray:
        return self._b.step(eye_drive=eye_drive, inject=inject)

    @property
    def raw(self):
        return self._b


def make_brain(cfg, verbose: bool = True):
    """cfg: config.BrainConfig"""
    if cfg.kind == "real":
        return RealFlyBrain(cfg, verbose=verbose)
    return FakeBrain(dt=cfg.dt, seed=cfg.seed)
