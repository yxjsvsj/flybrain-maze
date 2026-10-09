"""P2d ToF 外参 / 几何（第一阶段：mapping only）。

坐标系统一：**原点 = 后驱动轮轴中点；+x = 车头；+y = 左侧**。单位 **米**。
Windows 的 P2 runner 与静态诊断都从 `hardware/tof_geometry.json` 读；Pi 的 UDP
原始测距协议不变。

规矩
----
- 严格区分 **米 / cells**；本模块一切距离/位置都是**米**，转 cells 由调用方乘/除
  `meters_per_cell` 完成。
- 传感器光学窗口 **不是** 机器人访问位置。
- 命中点 = `sensor_origin(世界, 米) + range(米) · 世界射线方向`。
- 不要用一个标量去"统一 - car_radius"套在实体 ToF 上。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GEOMETRY_PATH = os.path.join(_REPO_ROOT, "hardware", "tof_geometry.json")


@dataclass(frozen=True)
class SensorExtrinsic:
    name: str
    x_m: float       # 相对后轴中点，+x 前
    y_m: float       # +y 左
    yaw_deg: float   # 相对车头，逆时针为正

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)


def load_geometry(path: str | None = None) -> dict[str, SensorExtrinsic]:
    """读硬件几何 JSON，返回 {name: SensorExtrinsic}。"""
    path = path or DEFAULT_GEOMETRY_PATH
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    sensors = doc.get("sensors")
    if not isinstance(sensors, dict) or not sensors:
        raise ValueError(f"几何文件缺少 sensors: {path}")
    return {
        n: SensorExtrinsic(str(n), float(s["x_m"]), float(s["y_m"]), float(s["yaw_deg"]))
        for n, s in sensors.items()
    }


def rotate(px: float, py: float, theta_rad: float) -> tuple[float, float]:
    """把车体坐标 (px, py) 按车头角 theta_rad 旋转到世界坐标（不含平移）。"""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return px * c - py * s, px * s + py * c


def sensor_world_ray(car_x_m: float, car_y_m: float, car_theta_rad: float,
                     ext: SensorExtrinsic, range_m: float):
    """返回 (ox, oy, world_angle_rad, hit_x, hit_y)，全部米、世界坐标。"""
    dx, dy = rotate(ext.x_m, ext.y_m, car_theta_rad)
    ox, oy = car_x_m + dx, car_y_m + dy
    wa = car_theta_rad + ext.yaw_rad
    hx = ox + range_m * math.cos(wa)
    hy = oy + range_m * math.sin(wa)
    return ox, oy, wa, hx, hy


def frame_to_world_rays(frame, car_x_m: float, car_y_m: float, car_theta_rad: float,
                        geometry: dict[str, SensorExtrinsic],
                        min_dist_m: float = 0.05,
                        range_bias_m: float = 0.0):
    """ToF 帧 -> 每条射线的世界几何（米）。

    返回 list[(name, ox, oy, world_angle, range_m, kind)]：
      VALID    -> kind="hit"（精确命中，进建图，带 range_bias_m 修正）
      TOO_NEAR -> kind="near"（保留危险状态，range=min_dist_m，**不作精确墙面命中**）
      NO_TARGET / IO_ERROR -> 跳过（绝不写虚假墙）
    """
    out = []
    for name, ext in geometry.items():
        st = frame.status.get(name)
        mm = frame.ranges.get(name)
        if st == "VALID" and mm is not None:
            r = max(min_dist_m, float(mm) / 1000.0 - float(range_bias_m))
            ox, oy, wa, _, _ = sensor_world_ray(car_x_m, car_y_m, car_theta_rad, ext, r)
            out.append((name, ox, oy, wa, r, "hit"))
        elif st == "TOO_NEAR":
            ox, oy, wa, _, _ = sensor_world_ray(car_x_m, car_y_m, car_theta_rad, ext, min_dist_m)
            out.append((name, ox, oy, wa, min_dist_m, "near"))
    return out


def frame_to_cell_rays(frame, car_x_cell: float, car_y_cell: float, car_theta_rad: float,
                       geometry: dict[str, SensorExtrinsic], meters_per_cell: float,
                       min_dist_m: float = 0.05, range_bias_m: float = 0.0):
    """同上，但输出给栅格建图用的 **cells** 射线：
    (ox_cell, oy_cell, world_angle, dist_cell, kind)。车体位姿以 cells 给出。"""
    mpc = float(meters_per_cell)
    out = []
    for name, ext in geometry.items():
        st = frame.status.get(name)
        mm = frame.ranges.get(name)
        if st == "VALID" and mm is not None:
            r = max(min_dist_m, float(mm) / 1000.0 - float(range_bias_m))
            kind = "hit"
        elif st == "TOO_NEAR":
            r, kind = min_dist_m, "near"
        else:
            continue
        dx, dy = rotate(ext.x_m, ext.y_m, car_theta_rad)
        ox = car_x_cell + dx / mpc
        oy = car_y_cell + dy / mpc
        wa = car_theta_rad + ext.yaw_rad
        out.append((ox, oy, wa, max(min_dist_m / mpc, r / mpc), kind))
    return out
