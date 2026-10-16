"""P3-1：真实 MaleCNS v1.0 的**视觉输入→下行运动神经**通路可行性审计（只用类型/连接/身份）。

严格区分三类证据，不混用、不据名臆造天然转向符号：
  (A) 解剖连接证据（本模块，来自官方 weights.npz + brain.npz 类型/侧别/bodyId）；
  (B) 当前人工 LIF 神经活动（本轮**默认不测**；仅当解剖清楚且能事前固定检测方案时才做一次短时离线实验）；
  (C) 人工电机映射（**不**在本轮，属已关闭的 P2d 读出路线）。

W 口径：`weights.npz` 为 CSR，`W[post, pre]`（`brain.py::_propagate` 按 fired=pre 收集其出边）。
故 **视觉→下行** = `W[dn_idx][:, vis_idx]`。**不**用任何随机 seed 导航成功率来筛神经元。
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    from flybrain.data import ensure_data
    data = ensure_data()
    meta = np.load(str(data / "brain.npz"))
    return {
        "ids": np.asarray(meta["ids"]),
        "cell_type": np.array([str(x) for x in meta["cell_type"]]),
        "side": np.array([str(x) for x in meta["side"]]),
        "superclass": np.array([str(x) for x in meta["superclass"]]),
        "visual": np.asarray(meta["visual"]),
        "azimuth": np.asarray(meta["azimuth"]),
        "W": sparse.load_npz(str(data / "weights.npz")).tocsr(),
    }


def group_idx(D, sup):
    return np.flatnonzero(D["superclass"] == sup)


def vpn_dn_edges(D, min_w=0.0):
    """直接 视觉(VPN)→下行(DN) 边（可追溯 bodyId/类型/侧别/权重）。"""
    vis = group_idx(D, "visual_projection")
    dn = group_idx(D, "descending_neuron")
    M = D["W"][dn][:, vis].tocoo()
    ids, ct, side = D["ids"], D["cell_type"], D["side"]
    edges = []
    for i, j, w in zip(M.row, M.col, M.data):
        if w <= min_w:
            continue
        di, vi = dn[i], vis[j]
        edges.append({"dn": int(ids[di]), "dn_type": ct[di], "dn_side": side[di],
                      "vpn": int(ids[vi]), "vpn_type": ct[vi], "vpn_side": side[vi],
                      "w": float(w)})
    return edges


def by_type(edges):
    agg = collections.Counter()
    for e in edges:
        agg[(e["dn_type"], e["vpn_type"])] += e["w"]
    return agg


def ipsi_contra(edges):
    """按 VPN 类型统计到**同侧/对侧** DN 的权重占比（左右同源可追溯性）。"""
    agg = collections.defaultdict(lambda: [0.0, 0.0])   # type -> [ipsi, contra]
    for e in edges:
        if e["vpn_side"] not in ("L", "R") or e["dn_side"] not in ("L", "R"):
            continue
        ipsi = e["vpn_side"] == e["dn_side"]
        agg[e["vpn_type"]][0 if ipsi else 1] += e["w"]
    out = {}
    for t, (a, b) in agg.items():
        s = a + b
        out[t] = {"ipsi": round(a / s, 3) if s else None, "contra": round(b / s, 3) if s else None,
                  "total": round(s, 2)}
    return out


def indirect_paths(D, vpn_types, dn_types, max_hops=2, top=20):
    """视觉→(中间)→下行 的**间接**路径（仅拓扑）。**严格按 W[post,pre]**：

      - `x` 从视觉来：`in_w = Σ_{v} W[x, v]`（x 的**入**权，源=视觉）→ 用 `W[:, vis_sel]` 的行 x；
      - `x` 去下行：`out_w = Σ_{d} W[d, x]`（x 的**出**权，靶=下行）→ 用 `W[dn_sel, :]` 的**列** x。
    权重**可正可负**（抑制为负）：`out_w` 同时给出**带符号和**与**绝对值和**，避免正负相消误解。
    """
    ct = D["cell_type"]; W = D["W"].tocsr()
    vis = group_idx(D, "visual_projection"); dn = group_idx(D, "descending_neuron")
    vis_sel = vis[np.isin(ct[vis], list(vpn_types))]
    dn_sel = dn[np.isin(ct[dn], list(dn_types))]
    M_in = W.tocsc()[:, vis_sel].tocsr()          # (n, n_vis) 入权：M_in[x,v]=W[x,v]
    M_out = W[dn_sel, :].tocsc()                  # (n_dn, n) 出权：M_out[:,x]=W[dn,x]
    rows = np.unique(M_in.tocoo().row)
    out = []
    vis_set = set(vis_sel.tolist()); dn_set = set(dn_sel.tolist())
    for x in rows:
        if x in vis_set or x in dn_set:
            continue
        w_in = float(M_in.getrow(x).sum())
        col = M_out.getcol(x)                     # 出权（x -> 下行）
        if col.nnz == 0:
            continue
        w_out_signed = float(col.data.sum())
        w_out_abs = float(np.abs(col.data).sum())
        out.append({"via_type": ct[x], "in_w": round(w_in, 4),
                    "out_w_signed": round(w_out_signed, 4), "out_w_abs": round(w_out_abs, 4),
                    "n_dn_targets": int(col.nnz)})
    out.sort(key=lambda r: -r["out_w_abs"])
    return out[:top]


# ------- 冻结的离线左右视觉响应检测（仅当解剖清楚才做一次） -------
STA_VPN = ("LC4", "LPLC4", "LPLC2")      # 解剖证据最强的 ipsi 视觉 VPN
STIM_A = 0.5
BASE_STEPS, STIM_STEPS, POST_STEPS = 25, 50, 25     # 0.5s / 1.0s / 0.5s @20ms
RESP_SEEDS = [11, 12, 13, 14, 15]                    # 冻结
RESP_MIN = 1.0            # DN 响应最小增量(spikes/step, stim−base)
STABLE_MIN = 4            # ≥4/5 seed 上 ipsi>contra 且 ipsi>base
FALSE_MAX = 1.0          # 背景误触发上限(spikes/step)


def dn_target_set(D, vpn_types=STA_VPN, min_w=0.0):
    """(旧) 以 **dn_type** 为键聚合的靶集合——会**丢失同型多细胞**。保留供对比。"""
    ed = vpn_dn_edges(D, min_w=min_w)
    tgt = {}
    for e in ed:
        if e["vpn_type"] in vpn_types:
            key = e["dn_type"]
            tgt.setdefault(key, {"dn_type": key, "side": e["dn_side"], "w": 0.0, "body": e["dn"]})
            tgt[key]["w"] += e["w"]
    return tgt


def dn_target_set_by_body(D, vpn_types=STA_VPN, min_w=0.0):
    """**修正版**：以**原始 bodyId** 为唯一键，保留**左右全部实际相连**的靶细胞（同型多细胞不合并）。
    **不按响应大小**筛。返回 {bodyId: {dn_type, side, w}}。"""
    ed = vpn_dn_edges(D, min_w=min_w)
    tgt = {}
    for e in ed:
        if e["vpn_type"] in vpn_types:
            b = int(e["dn"])
            tgt.setdefault(b, {"dn_type": e["dn_type"], "side": e["dn_side"], "w": 0.0, "body": b})
            tgt[b]["w"] += e["w"]
    return tgt


GROUP_KEYS = ("photo", "LC4", "LPLC4", "LPLC2", "other_VPN", "DN_L", "DN_R")


def _groups(D, dn_by_body):
    ct = D["cell_type"]
    vis = np.asarray(D["visual"])
    vpn = group_idx(D, "visual_projection")
    sta = np.isin(ct[vpn], list(STA_VPN))
    idx_of = {int(b): i for i, b in enumerate(D["ids"].tolist())}
    dnL = np.array([idx_of[v["body"]] for v in dn_by_body.values() if v["side"] == "L"], int)
    dnR = np.array([idx_of[v["body"]] for v in dn_by_body.values() if v["side"] == "R"], int)
    return {"photo": np.asarray(vis, int), "LC4": np.flatnonzero(ct == "LC4"),
            "LPLC4": np.flatnonzero(ct == "LPLC4"), "LPLC2": np.flatnonzero(ct == "LPLC2"),
            "other_VPN": vpn[~sta], "DN_L": dnL, "DN_R": dnR}


def _run_segments(brain, groups, drive, pre=BASE_STEPS, stim=STIM_STEPS, post=POST_STEPS):
    """**单次连续**跑 PRE(无驱动)→STIM(指定驱动)→POST(撤除)，段间**不 reset**；
    同一 `brain.step()` 发放同时统计**全部组**（不先测一组再推进）。"""
    total = pre + stim + post
    counts = {g: np.zeros(total, np.int64) for g in groups}
    for k in range(total):
        dv = drive if (pre <= k < pre + stim) else None
        f = np.asarray(brain.step(eye_drive=dv))
        for g, idx in groups.items():
            if len(idx):
                counts[g][k] = int(np.isin(f, idx).sum())
    return counts


def _seg_stats(c, pre, stim):
    pre_r = float(c[:pre].mean()); st_r = float(c[pre:pre + stim].mean()); po_r = float(c[pre + stim:].mean())
    return pre_r, st_r, po_r


def visual_response(D, *, seeds=RESP_SEEDS, device="cuda"):
    """修正版：bodyId 键的左右全靶；连续 PRE/STIM/POST（不 reset）；同步多层计数（群体+每细胞）。"""
    from flybrain.brain import FlyBrain
    from flybrain.data import ensure_data
    data = ensure_data()
    meta = np.load(str(data / "brain.npz"))
    az = meta["azimuth"]; vis = meta["visual"]
    Lm = az < -0.3; Rm = az > 0.3
    tgt_body = dn_target_set_by_body(D)
    tgt_type = dn_target_set(D)
    groups = _groups(D, tgt_body)
    brain = FlyBrain(data=data, device=device, seed=seeds[0], batch=1, sensory_input=False)
    L = len(vis)
    def drive(mask):
        d = np.zeros(L, np.float32); d[mask] = STIM_A; return brain.xp.asarray(d)
    drv = {"control": None, "left": drive(Lm), "right": drive(Rm)}
    per = []
    raw = {}
    for sd in seeds:
        row = {"seed": sd, "seg": {}}
        for cond in ("control", "left", "right"):
            brain.reset(sd)                                  # 每条件仅一次 reset（段间不 reset）
            c = _run_segments(brain, groups, drv[cond])
            raw[f"{sd}_{cond}"] = np.array([c[g] for g in GROUP_KEYS])   # 原始逐步计数
            res = {}
            for g in GROUP_KEYS:
                idx = groups[g]; n = max(1, len(idx))
                pre_r, st_r, po_r = _seg_stats(c[g], BASE_STEPS, STIM_STEPS)
                res[g] = {"n": int(len(idx)),
                          "pop_pre": round(pre_r, 3), "pop_stim": round(st_r, 3), "pop_post": round(po_r, 3),
                          "evoked_pop": round(st_r - pre_r, 3), "post_pop": round(po_r - st_r, 3),
                          "cell_pre": round(pre_r / n, 4), "cell_stim": round(st_r / n, 4),
                          "cell_post": round(po_r / n, 4),
                          "evoked_cell": round((st_r - pre_r) / n, 4)}
            row["seg"][cond] = res
        # 判据（per-cell，避免左右细胞数不等造成假差异）
        li = row["seg"]["left"]["DN_L"]["evoked_cell"]; lc = row["seg"]["left"]["DN_R"]["evoked_cell"]
        ri = row["seg"]["right"]["DN_R"]["evoked_cell"]; rc = row["seg"]["right"]["DN_L"]["evoked_cell"]
        row["left_ipsi_minus_contra_cell"] = round(li - lc, 4)
        row["right_ipsi_minus_contra_cell"] = round(ri - rc, 4)
        row["ipsi_gt_contra"] = bool((li - lc > 0) and (ri - rc > 0))
        row["dnL_evoked_pop"] = row["seg"]["left"]["DN_L"]["evoked_pop"]
        row["dnR_evoked_pop"] = row["seg"]["right"]["DN_R"]["evoked_pop"]
        per.append(row)
    n_ok = sum(r["ipsi_gt_contra"] for r in per)
    # 原判据（RESP_MIN 按群体 DN 激发增量；FALSE_MAX 按对照段 DN 背景群体率——解释限制见报告）
    resp_ok = all((r["dnL_evoked_pop"] >= RESP_MIN and r["dnR_evoked_pop"] >= RESP_MIN) for r in per)
    false_bg = float(np.mean([max(r["seg"]["control"]["DN_L"]["pop_pre"], r["seg"]["control"]["DN_R"]["pop_pre"])
                              for r in per]))
    go = resp_ok and (n_ok >= STABLE_MIN) and (false_bg <= FALSE_MAX)
    np.savez_compressed(os.path.join(HERE, "p31_raw.npz"), **raw)
    return {"dn_targets_by_type": len(tgt_type), "dn_targets_by_body": len(tgt_body),
            "n_dn_L": len(groups["DN_L"]), "n_dn_R": len(groups["DN_R"]),
            "params": {"STA_VPN": list(STA_VPN), "STIM_A": STIM_A, "seeds": seeds,
                       "PRE/STIM/POST": [BASE_STEPS, STIM_STEPS, POST_STEPS],
                       "RESP_MIN": RESP_MIN, "STABLE_MIN": STABLE_MIN, "FALSE_MAX": FALSE_MAX,
                       "sensory_input": False, "reset": "once per condition; none between segments"},
            "per_seed": per, "ipsi_gt_contra_seeds": n_ok, "resp_min_ok": resp_ok,
            "false_bg_pop": round(false_bg, 4), "verdict": "GO" if go else "NO-GO"}


def _response_main(seeds, device):
    D = load_data()
    r = visual_response(D, seeds=seeds, device=device)
    print(f"[P3-1a] DN targets by_type={r['dn_targets_by_type']} by_body={r['dn_targets_by_body']} "
          f"(L={r['n_dn_L']} R={r['n_dn_R']})")
    for row in r["per_seed"]:
        seg = row["seg"]["left"]
        print(f"  seed{row['seed']} left: photo cell {seg['photo']['cell_pre']}->{seg['photo']['cell_stim']} "
              f"({seg['photo']['evoked_cell']:+}) | otherVPN {seg['other_VPN']['cell_pre']}->{seg['other_VPN']['cell_stim']} "
              f"| DN_L pop {seg['DN_L']['pop_pre']}->{seg['DN_L']['pop_stim']} DN_R pop {seg['DN_R']['pop_pre']}->{seg['DN_R']['pop_stim']}"
              f" | ipsi-contra_cell={row['left_ipsi_minus_contra_cell']}/{row['right_ipsi_minus_contra_cell']} "
              f"ipsi>contra={row['ipsi_gt_contra']}")
    print(f"  resp_min_ok={r['resp_min_ok']} ipsi>contra={r['ipsi_gt_contra_seeds']}/{len(seeds)} "
          f"false_bg_pop={r['false_bg_pop']} -> {r['verdict']}")
    json.dump(r, open(os.path.join(HERE, "p31_response.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p31_response.json + p31_raw.npz")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.response:
        return _response_main([int(s) for s in a.response.split(",")], a.device)
    D = load_data()
    ed = vpn_dn_edges(D, min_w=0.0)
    agg = by_type(ed)
    ic = ipsi_contra(ed)
    report = {"n_vpn": int((D["superclass"] == "visual_projection").sum()),
              "n_dn": int((D["superclass"] == "descending_neuron").sum()),
              "n_edges": len(ed), "total_weight": round(sum(e["w"] for e in ed), 3),
              "top_pairs": [{"dn_type": d, "vpn_type": v, "w": round(w, 3)} for (d, v), w in agg.most_common(30)],
              "vpn_ipsi_contra": {k: ic[k] for k in sorted(ic, key=lambda t: -ic[t]["total"])[:20]}}
    print(f"[P3-1] visual_projection={report['n_vpn']} descending_neuron={report['n_dn']} "
          f"direct edges={report['n_edges']} total W={report['total_weight']}")
    print("top VPN->DN pairs:")
    for r in report["top_pairs"][:20]:
        print(f"  {r['dn_type']:12s} <- {r['vpn_type']:16s} w={r['w']}")
    print("ipsi vs contra (VPN type -> DN side):")
    for t, v in list(report["vpn_ipsi_contra"].items())[:12]:
        print(f"  {t:16s} ipsi={v['ipsi']} contra={v['contra']} total={v['total']}")
    json.dump(report, open(os.path.join(HERE, "p31_visual_motor.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p31_visual_motor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
