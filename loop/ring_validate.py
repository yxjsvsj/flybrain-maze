"""P2d-26f 任务4：**独立验证**（连接带状 vs 打乱空模型 + 分半稳定）。

- 连接**只用于验证**，**绝不**回馈调整轴/角/阈值/细胞集合（预注册冻结）。
- EPG→PEN 强度取自 flybrain `weights.npz`（与 `compass_ring.py` 同一子矩阵口径）。
- 对每个 PEN 细胞：加权圆集中度 R_w = |Σ w_i e^{iθ_i}| / Σ w_i（θ 来自**空间**角）。
- 空模型：把 EPG 的角度标签**随机置换** B 次 → R_w 的零分布 → p 值。
- 分别报告 PEN_a / PEN_b。
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

GROUPS = ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)")
HERE = os.path.dirname(os.path.abspath(__file__))
NULL_B = 1000
NULL_SEED = 0


def weighted_circ(angles_rad, w):
    """加权圆集中度与加权圆均（弧度）。"""
    if len(angles_rad) == 0 or w.sum() <= 0:
        return 0.0, float("nan")
    C = float((w * np.cos(angles_rad)).sum() / w.sum())
    S = float((w * np.sin(angles_rad)).sum() / w.sum())
    return float(np.hypot(C, S)), float(np.arctan2(S, C))


def validate(epg_ids, epg_ang_deg, pen_by_group, W, idx_of, B=NULL_B, seed=NULL_SEED):
    """返回每 PEN 细胞的 R_w、p，及每组汇总。"""
    rng = np.random.default_rng(seed)
    ang = np.radians(np.asarray(epg_ang_deg, float))            # EPG 空间角（弧度）
    epg_ix = [idx_of[b] for b in epg_ids]
    out = {}
    for g, pen_ids in pen_by_group.items():
        recs = []
        for b in pen_ids:
            pj = idx_of[b]
            col = np.asarray(W[pj, epg_ix].todense()).ravel()   # EPG→PEN(=W[pen,epg])
            m = col > 0
            if m.sum() < 2:
                recs.append({"bodyId": int(b), "n_epg_in": int(m.sum()),
                             "R_w": float("nan"), "p": float("nan"), "wmean_deg": float("nan")})
                continue
            obs, wm = weighted_circ(ang[m], col[m])
            null = np.empty(B)
            cols = col[m]
            for i in range(B):
                perm = rng.permutation(ang)                     # 打乱 EPG↔角度
                null[i] = weighted_circ(perm[m], cols)[0]
            # 加一估计（Phipson-Smyth）：即使无置换 ≥ 观测，p 也不为 0
            p = (1.0 + float((null >= obs).sum())) / (1.0 + B)
            recs.append({"bodyId": int(b), "n_epg_in": int(m.sum()), "R_w": round(obs, 4),
                         "p": round(p, 4), "wmean_deg": round(float(np.degrees(wm)), 2),
                         "null_med_Rw": round(float(np.median(null)), 4)})
        rw = np.array([r["R_w"] for r in recs if r["R_w"] == r["R_w"]])
        ps = np.array([r["p"] for r in recs if r["p"] == r["p"]])
        out[g] = {"cells": recs,
                  "median_R_w": round(float(np.median(rw)), 4) if len(rw) else float("nan"),
                  "frac_p_lt_05": round(float((ps < 0.05).mean()), 4) if len(ps) else float("nan"),
                  "n": len(recs)}
    return out


def load_cells(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    epg = [(int(r["bodyId"]), float(r["circ_mean_deg"]), float(r["split_half_diff_deg"]))
           for r in rows if r["group"] == "EPG" and r["valid"] == "True"]
    pen = {g: [int(r["bodyId"]) for r in rows if r["group"] == g] for g in GROUPS[1:]}
    return epg, pen


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(HERE, "p2d26f_cells.csv"))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    from flybrain.data import ensure_data
    from scipy import sparse
    data = ensure_data()
    meta = np.load(str(data / "brain.npz"))
    ids = np.asarray(meta["ids"])
    W = sparse.load_npz(str(data / "weights.npz")).tocsr()
    idx_of = {int(b): i for i, b in enumerate(ids.tolist())}

    epg, pen = load_cells(a.cells)
    epg_ids = [e[0] for e in epg]
    ang = [e[1] for e in epg]
    sh = [e[2] for e in epg]
    print(f"[P2d-26f] valid EPG for validation: {len(epg_ids)} / 46")
    print(f"  split-half diff: median={np.median(sh):.1f} deg, frac<20deg={np.mean(np.array(sh)<20):.2f}")
    res = validate(epg_ids, ang, {g: pen[g] for g in pen}, W, idx_of)
    import json
    summ = {"n_epg_valid": len(epg_ids),
            "split_half_median_deg": round(float(np.median(sh)), 2),
            "split_half_frac_lt20": round(float(np.mean(np.array(sh) < 20)), 3)}
    for g in ("PEN_a(PEN1)", "PEN_b(PEN2)"):
        r = res[g]
        nulls = [c.get("null_med_Rw") for c in r["cells"] if c.get("null_med_Rw") is not None]
        summ[g] = {"n": r["n"], "median_R_w": r["median_R_w"],
                   "frac_p_lt_05": r["frac_p_lt_05"],
                   "null_median_R_w": round(float(np.median(nulls)), 4) if nulls else None}
        print(f"  {g}: n={r['n']} median_R_w={r['median_R_w']} "
              f"null_median_R_w={summ[g]['null_median_R_w']} frac(p<0.05)={r['frac_p_lt_05']}")
    json.dump({"summary": summ, "per_pen": res}, open(os.path.join(HERE, "p2d26f_validation.json"),
              "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, polar=True)
            eph = np.radians([e[1] for e in epg])
            ax.scatter(eph, np.ones(len(eph)), s=18, c="tab:blue", label=f"EPG (n={len(eph)})")
            for g, col, mk, rr in (("PEN_a(PEN1)", "tab:red", "^", 0.82),
                                   ("PEN_b(PEN2)", "tab:green", "s", 0.66)):
                wm = [c["wmean_deg"] for c in res[g]["cells"] if c["wmean_deg"] == c["wmean_deg"]]
                ax.scatter(np.radians(wm), np.full(len(wm), rr), s=40, c=col, marker=mk,
                           label=f"{g} wmean (n={len(wm)})")
            ax.set_title("P2d-26f EB spatial ring (anatomical; 0-dir arbitrary)")
            ax.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.25, -0.15))
            fig.savefig(os.path.join(HERE, "p2d26f_ring.png"), dpi=110, bbox_inches="tight")
            print("  plot -> loop/p2d26f_ring.png")
        except Exception as e:                                 # noqa: BLE001
            print("  plot skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
