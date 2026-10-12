"""P2d-13d：正前方 F 感觉分组的单变量消融 —— **P2 实验适配层**。

⚠ 披露：本模块只改变 **P2 侧的感觉输入分组**（哪些 ToF 通道进入左/右神经池），
   **不触碰** `loop/encode_tof.py`、不改冻结 P1、**不改 MaleCNS 自身的连接结构/权重**、
   不改神经元类型、不改传感器外参、不改 front_safety。

BASE：现有 `PhysicalToFEncoder` 行为，原样保留 —— 左=FL+F，右=FR+F，正面安全=F。
TEST：只把左右神经掩码改为 左=FL、右=FR（中心 F 不再进左右池）；正面安全仍=F；
       L/R 仍可用于建图(valid_mask 不变)，不加入局部避障。
"""
from __future__ import annotations

from hardware.tof5 import DEFAULT_ANGLES_DEG, ORDER
from loop.encode_tof import PhysicalToFEncoder

VARIANTS = ("BASE", "TEST")


class SideOnlyToFEncoder(PhysicalToFEncoder):
    """TEST：左右神经感觉只用 FL / FR；中心 F 仅留给 front_safety 与建图。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.left 原本 = {FL, F}，self.right = {FR, F}；这里剔掉 center(F)
        self.left = self.left & (self.angles > 0)
        self.right = self.right & (self.angles < 0)


def encoder_cls(variant: str):
    if variant == "BASE":
        return PhysicalToFEncoder
    if variant == "TEST":
        return SideOnlyToFEncoder
    raise KeyError(f"未知 variant {variant!r}，可选 {VARIANTS}")


def make_encoder(variant, groups, cfg, dt, mpc=0.40):
    cls = encoder_cls(variant)
    return cls(groups, cfg.encoder, dt, DEFAULT_ANGLES_DEG, sensor_order=ORDER,
               car_max_speed=cfg.car.max_speed, car_radius=cfg.car.radius, meters_per_cell=mpc)
