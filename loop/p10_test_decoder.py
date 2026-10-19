"""P10-C1b 专用测试解码器：**只**由冻结、可核验的 DN 活动产生运动输出。

不修改 P1 冻结的 `loop/decode.py`。禁止：Dijkstra、memory_gain、pursuit_turn、外部路径方向、
scripted、线性训练 readout、人工转弯、计时脱困/回正、缺失组回退到全部 DN。
安全层**只**限速/急停，不决定方向。

**重要**：转向映射基于 DNa02 L/R，但 P7-0 已证明 `LC10a→DNa02` 直接边=0、DNa02 由 cb_intrinsic 主导
→ **神经依据不足**，故 `decoder_ready=False`（不伪造通过）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FROZEN_GROUPS = {
    "turn_L": {"types": ["DNa02"], "side": "L"},
    "turn_R": {"types": ["DNa02"], "side": "R"},
    "fwd": {"types": ["DNg100"], "side": None},
    "back": {"types": ["MDN"], "side": None},
}
FROZEN_GAINS = {"turn_sign": 1.0, "turn_gain": 1.0, "fwd_gain": 1.0, "back_gain": 1.0}
ALLOWED_RATE_KEYS = set(FROZEN_GROUPS)
FORBIDDEN_KEYS = {
    "map", "occupancy", "pose", "global_pose", "dijkstra", "teacher_action", "teacher_label",
    "goal", "goal_coord", "goal_direction", "nav_memory", "planner", "path_plan",
    "turn_command", "manual_turn", "pursuit_turn", "memory_gain", "bias", "escape",
    "scripted", "readout",
}

NEURAL_BASIS = {
    "turn_mapping": "DNa02 L/R -> sign(turn_L - turn_R)",
    "P7_0_LC10a_DNa02": "0 direct edges (P7-0 read-only audit)",
    "DNa02_input": "dominated by cb_intrinsic (L abs 0.804 / R 0.808); visual input weak/near-zero",
    "verdict": "insufficient neural basis for left/right steering -> decoder_ready=False",
}


class DecoderValidationError(RuntimeError):
    pass


def resolve_frozen_groups(D):
    """从真实 data 解析冻结 bodyId；**绝不**回退到全部 descending neurons。"""
    ct = D["cell_type"]; side = D["side"]; ids = D["ids"]
    out = {}
    for name, spec in FROZEN_GROUPS.items():
        mask = np.isin(ct, spec["types"])
        if spec["side"] is not None:
            mask &= side == spec["side"]
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise DecoderValidationError(f"group {name} empty; NO fallback allowed")
        bids = [int(ids[i]) for i in idx]
        if len(set(bids)) != len(bids):
            raise DecoderValidationError(f"group {name} has duplicate bodyIds")
        out[name] = {"bodyIds": bids, "types": list(np.unique(ct[idx])), "side": spec["side"]}
    return out


def validate_groups(groups, ids, cell_type, side):
    """校验 bodyId 存在、类型/侧别匹配、无重复、无回退。"""
    id_map = {int(b): i for i, b in enumerate(ids)}
    for name, spec in FROZEN_GROUPS.items():
        g = groups.get(name)
        if not g or not g.get("bodyIds"):
            raise DecoderValidationError(f"group {name} missing/empty")
        bids = list(g["bodyIds"])
        if len(set(bids)) != len(bids):
            raise DecoderValidationError(f"group {name} duplicate bodyIds")
        for b in bids:
            if int(b) not in id_map:
                raise DecoderValidationError(f"group {name}: bodyId {b} not found")
            i = id_map[int(b)]
            if cell_type[i] not in spec["types"]:
                raise DecoderValidationError(f"group {name}: bodyId {b} type {cell_type[i]} != {spec['types']}")
            if spec["side"] is not None and side[i] != spec["side"]:
                raise DecoderValidationError(f"group {name}: bodyId {b} side {side[i]} != {spec['side']}")
    return True


class P10TestDecoder:
    """冻结的 DN→(v, omega) 映射；无可训练参数、无外部方向。"""

    def __init__(self, groups, gains=None, max_omega=1.0, max_speed=1.0):
        for name in FROZEN_GROUPS:
            g = groups.get(name)
            if not g or not g.get("bodyIds"):
                raise DecoderValidationError(f"group {name} missing/empty; NO fallback allowed")
            bids = list(g["bodyIds"])
            if len(set(bids)) != len(bids):
                raise DecoderValidationError(f"group {name} duplicate bodyIds")
        self.groups = groups
        self.gains = dict(FROZEN_GAINS if gains is None else gains)
        self.max_omega = float(max_omega)
        self.max_speed = float(max_speed)

    def step(self, rates: dict, safety: dict | None = None):
        bad = set(rates) & FORBIDDEN_KEYS
        if bad:
            raise DecoderValidationError(f"forbidden inputs: {sorted(bad)}")
        unknown = set(rates) - ALLOWED_RATE_KEYS
        if unknown:
            raise DecoderValidationError(f"unknown inputs: {sorted(unknown)}")
        for k, v in rates.items():
            if not math.isfinite(float(v)):
                raise DecoderValidationError(f"non-finite rate: {k}={v}")
        g = self.gains
        turn = float(g["turn_sign"]) * (float(rates.get("turn_L", 0.0)) - float(rates.get("turn_R", 0.0)))
        v = float(g["fwd_gain"]) * float(rates.get("fwd", 0.0)) - float(g["back_gain"]) * float(rates.get("back", 0.0))
        v = self.max_speed * float(np.clip(v, -1.0, 1.0))
        omega = self.max_omega * float(np.clip(g["turn_gain"] * turn, -1.0, 1.0))
        if safety:
            if safety.get("emergency_stop"):
                v, omega = 0.0, 0.0
            elif safety.get("speed_limit") is not None:
                lim = max(0.0, float(safety["speed_limit"]))
                v = math.copysign(min(abs(v), lim), v) if v != 0 else 0.0
        return {"v": float(v), "omega": float(omega), "turn": float(turn), "source": "dn_only"}

    def audit(self):
        return {"formula": "turn = turn_sign*(r_turn_L - r_turn_R); v = fwd_gain*r_fwd - back_gain*r_back",
                "gains": self.gains, "groups": self.groups, "external_direction": False,
                "fallback_to_all_DN": False, "safety": "speed limit + emergency stop only"}


def build_freeze(D):
    groups = resolve_frozen_groups(D)
    validate_groups(groups, D["ids"], D["cell_type"], D["side"])
    with open(os.path.join(HERE, "p10_test_decoder.py"), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "module": "loop/p10_test_decoder.py", "sha256": sha,
        "groups": groups, "gains": FROZEN_GAINS,
        "allowed_rate_keys": sorted(ALLOWED_RATE_KEYS),
        "forbidden_keys": sorted(FORBIDDEN_KEYS),
        "neural_basis": NEURAL_BASIS,
        "decoder_ready": False,
        "requirements": [
            "independent neural evidence for the DNa02 left/right steering mapping",
            "frozen, neurally justified gains (not arbitrarily chosen)",
            "proof that test-phase has no external/artificial direction source",
        ],
    }
