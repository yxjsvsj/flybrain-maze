"""P2d-26f：**预注册**的 EB 空间环坐标重建（**只用解剖空间数据**；无动力学/无注入/无控车）。

方法与阈值见 `loop/RING_PREREG.md`（**运行前冻结**）。本模块只做：
  1) 由 EB 网格顶点定 PCA 平面/中心/极角；
  2) 逐细胞（严格 token `EB` 的突触点）算圆均角、圆集中度、峰数、bootstrap 不确定性、分半差；
  3) 依**冻结规则**判有效/无效（无效者给原因），前后合并/前/后三种分布**并列报告**。

**零点/正向是人为解剖参考，不是真实朝向。**
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from loop.coord_frame import obj_vertices

GROUPS = ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)")
HERE = os.path.dirname(os.path.abspath(__file__))

# ------- 冻结阈值（RING_PREREG.md §3） -------
N_MIN = 50
RATIO_MIN = 0.20
R_MIN = 0.50
MODE_MAX = 1
UNC_MAX_DEG = 30.0
BOOT_B = 500
BOOT_SEED = 0
SPLIT_SEED = 0
NBINS = 24
MODE_FRAC = 0.25

CELL_COLS = ["bodyId", "group", "n_total", "n_EB", "EB_ratio", "n_pre_EB", "n_post_EB",
             "circ_mean_deg", "R", "raw_circ_std_deg", "n_modes", "boot_circ_std_deg",
             "ci95_lo_deg", "ci95_hi_deg", "ci95_half_deg", "split_half_diff_deg",
             "pre_mean_deg", "post_mean_deg", "valid", "reason"]


# ---------------- geometry / circular stats ----------------
def eb_frame(V):
    """EB 网格顶点 → 中心 c 与 PCA 平面基 e1,e2（及法向）。"""
    V = np.asarray(V, float).reshape(-1, 3)
    c = V.mean(0)
    U, S, Vt = np.linalg.svd(V - c, full_matrices=False)
    return {"center": c, "e1": Vt[0], "e2": Vt[1], "normal": Vt[2],
            "explained": (S ** 2 / np.sum(S ** 2)).tolist()}


def angles(P, frame):
    P = np.asarray(P, float).reshape(-1, 3)
    if not len(P):
        return np.array([])
    d = P - frame["center"]
    return np.arctan2(d @ frame["e2"], d @ frame["e1"])        # (-pi, pi]


def circ_stats(theta):
    theta = np.asarray(theta, float)
    n = len(theta)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "R": 0.0, "circ_std": float("inf")}
    C = float(np.cos(theta).mean()); S = float(np.sin(theta).mean())
    R = float(np.hypot(C, S))
    mean = float(np.arctan2(S, C))
    csd = float(np.sqrt(-2 * np.log(R))) if R > 1e-12 else float("inf")
    return {"n": n, "mean": mean, "R": R, "circ_std": csd}


def shortest_arc_ci(means_rad, level=0.95):
    """圆上最短包含 `level` 比例的弧；(lo, hi) 弧度，宽 = 最短弧长。"""
    a = np.sort(np.mod(np.asarray(means_rad, float), 2 * np.pi))
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    k = min(n - 1, max(1, int(np.floor(level * n))))
    best = (float("inf"), a[0], a[0])
    for i in range(n):
        j = (i + k) % n
        span = (a[j] - a[i]) % (2 * np.pi)
        if best is None or span < best[0]:
            best = (span, a[i], a[j])
    return float(best[1]), float(best[2])


def bootstrap_mean_angle(theta, B=BOOT_B, seed=BOOT_SEED, level=0.95):
    """**bootstrap 均值角的不确定性**（与原始突触角度分布的圆标准差**区分**）。

    实现：B 次有放回重采样，每次算该样本的**圆均值角** `m_b`；再对这 B 个 `m_b`：
      - `boot_circ_std_deg` = 其圆标准差（= 阈值 `UNC_MAX_DEG` 所门的"bootstrap 不确定性"）；
      - `ci95_lo/hi/half_deg` = 这 B 个 `m_b` 上**最短 95% 弧**的端点与半宽（明确的 CI 定义）。
    **上一版把 `boot_circ_std_deg` 误命名为 `ci_half_deg`**；本版分开命名，不再混用。
    """
    theta = np.asarray(theta, float)
    n = len(theta)
    if n < 2:
        return {"mean": float("nan"), "boot_circ_std_deg": float("inf"),
                "ci95_lo_deg": float("nan"), "ci95_hi_deg": float("nan"),
                "ci95_half_deg": float("inf"), "B": B, "level": level}
    rng = np.random.default_rng(seed)
    st = theta[rng.integers(0, n, size=(B, n))]                # (B, n)
    means = np.arctan2(np.sin(st).mean(1), np.cos(st).mean(1))  # 逐次 bootstrap 圆均值角
    Cb = float(np.cos(means).mean()); Sb = float(np.sin(means).mean())
    R = float(np.hypot(Cb, Sb))
    boot_std = float(np.degrees(np.sqrt(-2 * np.log(R)))) if R > 1e-12 else float("inf")
    lo, hi = shortest_arc_ci(means, level)
    half = float(np.degrees(((hi - lo) % (2 * np.pi)) / 2))
    return {"mean": float(np.arctan2(Sb, Cb)), "boot_circ_std_deg": boot_std,
            "ci95_lo_deg": float(np.degrees(lo)), "ci95_hi_deg": float(np.degrees(hi)),
            "ci95_half_deg": half, "B": B, "level": level}


def n_modes(theta, nbins=NBINS, frac=MODE_FRAC, merge_deg=60.0):
    theta = np.asarray(theta, float)
    if len(theta) < 2:
        return 0
    h, _ = np.histogram(theta, bins=nbins, range=(-np.pi, np.pi))
    h = np.convolve(np.r_[h[-1], h, h[0]], np.ones(3) / 3, mode="valid")
    if h.max() <= 0:
        return 0
    peaks = [i for i in range(nbins)
             if h[i] >= h[(i - 1) % nbins] and h[i] > h[(i + 1) % nbins] and h[i] >= frac * h.max()]
    if not peaks:
        return 0
    step = 360.0 / nbins
    # 环形合并相近峰
    peaks.sort()
    merged = [peaks[0]]
    for p in peaks[1:]:
        if (p - merged[-1]) * step < merge_deg:
            continue
        merged.append(p)
    if len(merged) > 1 and (360 - (merged[-1] - merged[0]) * step) < merge_deg:
        merged.pop()
    return len(merged)


def circ_diff_deg(a, b):
    d = np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b)))
    return abs(float(d))


def split_half(theta, seed=SPLIT_SEED):
    theta = np.asarray(theta, float)
    if len(theta) < 4:
        return float("inf")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(theta))
    h1, h2 = theta[perm[:len(theta) // 2]], theta[perm[len(theta) // 2:]]
    return circ_diff_deg(circ_stats(h1)["mean"], circ_stats(h2)["mean"])


def cell_quality(theta_eb, n_total):
    st = circ_stats(theta_eb)                                  # 原始突触角度分布
    raw_std_deg = float(np.degrees(st["circ_std"]))            # 分布圆标准差（宽窄）
    ratio = (st["n"] / n_total) if n_total else 0.0
    bo = bootstrap_mean_angle(theta_eb)                        # 均值角估计的不确定性
    md = n_modes(theta_eb)
    sh = split_half(theta_eb)
    reasons = []
    if st["n"] < N_MIN:
        reasons.append(f"n_EB<{N_MIN}")
    if ratio < RATIO_MIN:
        reasons.append(f"EB_ratio<{RATIO_MIN}")
    if st["R"] < R_MIN:
        reasons.append(f"R<{R_MIN}")
    if md > MODE_MAX:
        reasons.append(f"multimodal({md})")
    if bo["boot_circ_std_deg"] > UNC_MAX_DEG:                  # 原门槛，门的是 bootstrap 不确定性
        reasons.append(f"boot_std>{UNC_MAX_DEG:g}")
    return {"n_EB": st["n"], "EB_ratio": round(ratio, 4), "circ_mean": st["mean"],
            "R": round(st["R"], 4), "raw_circ_std_deg": round(raw_std_deg, 2),
            "n_modes": md, "boot_circ_std_deg": round(bo["boot_circ_std_deg"], 2),
            "ci95_lo_deg": round(bo["ci95_lo_deg"], 2), "ci95_hi_deg": round(bo["ci95_hi_deg"], 2),
            "ci95_half_deg": round(bo["ci95_half_deg"], 2),
            "split_half_diff_deg": round(sh, 2), "valid": not reasons,
            "reason": ";".join(reasons)}


# ---------------- input / build ----------------
def load_coords_npz(path):
    d = np.load(path)
    return {k: d[k] for k in ("bodyId", "kind", "xyz", "roi_code", "roi_names")}


def eb_mask(coords, token="EB"):
    """严格 token：roi 值按 ';' 分词后含恰好 token。"""
    names = np.array([str(x) for x in coords["roi_names"]])
    tok = np.array([[t.strip() for t in n.split(";")] for n in names], dtype=object)
    has = np.array([token in set(t) for t in tok])
    codes_of_token = np.flatnonzero(has)
    return np.isin(coords["roi_code"], codes_of_token)


def load_targets_groups(path):
    d = json.load(open(path, encoding="utf-8"))
    g = {}
    for k in GROUPS:
        for b in d[k]:
            g[int(b)] = k
    return g


def build(coords, frame, group_of):
    bid = coords["bodyId"]; kind = coords["kind"]; xyz = coords["xyz"].astype(float)
    is_eb = eb_mask(coords)
    rows = []
    for b in sorted(set(bid.tolist())):
        m = bid == b
        m_eb = m & is_eb
        n_total = int(m.sum())
        th_eb = angles(xyz[m_eb], frame)
        q = cell_quality(th_eb, n_total)
        th_pre = angles(xyz[m_eb & (kind == 0)], frame)
        th_post = angles(xyz[m_eb & (kind == 1)], frame)
        rows.append({
            "bodyId": int(b), "group": group_of.get(int(b), ""), "n_total": n_total,
            "n_EB": q["n_EB"], "EB_ratio": q["EB_ratio"],
            "n_pre_EB": int((m_eb & (kind == 0)).sum()), "n_post_EB": int((m_eb & (kind == 1)).sum()),
            "circ_mean_deg": round(float(np.degrees(q["circ_mean"])), 2) if q["n_EB"] else "",
            "R": q["R"], "raw_circ_std_deg": q["raw_circ_std_deg"], "n_modes": q["n_modes"],
            "boot_circ_std_deg": q["boot_circ_std_deg"], "ci95_lo_deg": q["ci95_lo_deg"],
            "ci95_hi_deg": q["ci95_hi_deg"], "ci95_half_deg": q["ci95_half_deg"],
            "split_half_diff_deg": q["split_half_diff_deg"],
            "pre_mean_deg": round(float(np.degrees(circ_stats(th_pre)["mean"])), 2) if len(th_pre) else "",
            "post_mean_deg": round(float(np.degrees(circ_stats(th_post)["mean"])), 2) if len(th_post) else "",
            "valid": q["valid"], "reason": q["reason"]})
    return rows, is_eb


def _fmt(v):
    return "" if v is None else v


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default=os.path.join(HERE, "p2d26e_coords.npz"))
    ap.add_argument("--mesh", default=os.path.join(HERE, "assets", "EB.obj"))
    ap.add_argument("--targets", default=os.path.join(HERE, "p2d26d_targets.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "p2d26f_cells.csv"))
    a = ap.parse_args(argv)
    frame = eb_frame(obj_vertices(open(a.mesh, encoding="utf-8", errors="replace")))
    coords = load_coords_npz(a.coords)
    group_of = load_targets_groups(a.targets)
    rows, is_eb = build(coords, frame, group_of)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CELL_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    nv = sum(1 for r in rows if r["valid"])
    print(f"[P2d-26f] mesh verts frame explained={frame['explained']}")
    print(f"  cells={len(rows)} valid={nv} EB_rows={int(is_eb.sum())} -> {a.out}")
    for g in GROUPS:
        gr = [r for r in rows if r["group"] == g]
        gv = [r for r in gr if r["valid"]]
        print(f"  {g}: cells={len(gr)} valid={len(gv)}"
              + (f" mean_R={np.mean([r['R'] for r in gv]):.3f}" if gv else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
