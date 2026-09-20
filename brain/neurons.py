"""神经元组：把"车的传感器/执行器"映射到果蝇的细胞类型。

细胞类型名取自 MaleCNS / FlyWire 注释，选型参考 flybrain/eyes.py 的 CHANNELS
和 README 里点名的指令神经元：
    LPLC2   looming（物体逼近）        -> 避障
    LC4     快速 looming / 逃逸        -> 紧急避障
    DNa02   转向                       -> omega
    DNg100  前进                       -> v
    MDN     后退                       -> -v
    DNp01   巨纤维，逃逸起飞            -> 急退急转
"""
import numpy as np

GROUPS = {
    "loom_L":   (["LPLC2"], "L"),
    "loom_R":   (["LPLC2"], "R"),
    "threat_L": (["LC4"],   "L"),
    "threat_R": (["LC4"],   "R"),
    # LC10a -> DNa02 是已知的转向通路（LPLC2/LC4 主要通 DNp01 逃逸）。
    # 真脑靠这一路转向，假脑把它接到转向池上。
    "chase_L":  (["LC10a"], "L"),
    "chase_R":  (["LC10a"], "R"),
    "turn_L":   (["DNa02"], "L"),
    "turn_R":   (["DNa02"], "R"),
    "fwd":      (["DNg100"], None),
    "back":     (["MDN"],    None),
    "escape":   (["DNp01"],  None),
}

FALLBACK = "descending_neuron"


def resolve(brain, verbose: bool = True) -> dict:
    """返回 {组名: 神经元索引数组}。

    某个类型在连接组里找不到时，退回整个下行神经元类并大声警告——这几乎肯定会让
    解码器失去侧向特异性，属于必须人工确认的情况，不要当成正常路径。
    """
    out, missing = {}, []
    for name, (types, side) in GROUPS.items():
        idx = brain.cells(types, side=side)
        if len(idx) == 0:
            missing.append(f"{name}({types[0]})")
            idx = brain.cells([FALLBACK], side=side)
        out[name] = np.asarray(idx)

    if missing and verbose:
        print(f"[neurons] WARNING: 连接组里没有 {missing}")
        print(f"[neurons] WARNING: 已退回 '{FALLBACK}'，左右可能不再有区分度，请检查！")
    if verbose:
        for name, idx in out.items():
            print(f"[neurons] {name:9s} -> {len(idx):5d} 个神经元")
    return out
