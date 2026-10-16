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
    """视觉→(中间)→下行 的**间接**路径（仅拓扑，不测活动）。"""
    ct = D["cell_type"]; W = D["W"]
    vis = group_idx(D, "visual_projection"); dn = group_idx(D, "descending_neuron")
    vis_sel = vis[np.isin(ct[vis], list(vpn_types))]
    dn_sel = dn[np.isin(ct[dn], list(dn_types))]
    # 一跳：vis->X（对全部神经元）
    out = []
    M1 = W.tocsc()[:, vis_sel]                        # (n, n_vis) 每个中间神经元从 vis 收的权重
    # 取收到 vis 的神经元（col 非零）
    rows = np.unique(M1.tocoo().row)
    # 二跳：这些中间 -> dn
    Wc = W  # csr
    for x in rows:
        if x in set(vis_sel.tolist()) or x in set(dn_sel.tolist()):
            continue
        w_in = M1.getrow(x).sum()
        if w_in <= 0:
            continue
        row = Wc.getrow(x)
        hit = row.indices[np.isin(row.indices, dn_sel)]
        if len(hit):
            wt = row.data[np.isin(row.indices, dn_sel)].sum()
            out.append({"via_type": ct[x], "in_w": round(float(w_in), 3), "out_w": round(float(wt), 3)})
    out.sort(key=lambda r: -r["out_w"])
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
    """解剖确定的下行靶集合（直接来自 STA_VPN，任一非零边即入），带侧别；**不用活动结果筛**。"""
    ed = vpn_dn_edges(D, min_w=min_w)
    tgt = {}
    for e in ed:
        if e["vpn_type"] in vpn_types:
            key = e["dn_type"]
            tgt.setdefault(key, {"dn_type": key, "side": e["dn_side"], "w": 0.0, "body": e["dn"]})
            tgt[key]["w"] += e["w"]
    return tgt


def _dn_rate(brain, dn_idx, drive, n_steps, dt):
    from flybrain import Trace
    tr = Trace(brain, idx=dn_idx, tau=0.1)
    cnt = 0
    for _ in range(n_steps):
        f = brain.step(eye_drive=drive)
        cnt += int(np.isin(np.asarray(f), dn_idx).sum())
        tr.observe(f)
    return cnt / n_steps, tr.features()


def visual_response(D, *, seeds=RESP_SEEDS, device="cuda"):
    """冻结检测：背景 / 左眼刺激 / 右眼刺激，测解剖确定的 DN 靶集合 ipsi vs contra。"""
    from flybrain.brain import FlyBrain
    from flybrain.data import ensure_data
    data = ensure_data()
    meta = np.load(str(data / "brain.npz"))
    az = meta["azimuth"]; vis = meta["visual"]
    left_mask = az < -0.3; right_mask = az > 0.3
    tgt = dn_target_set(D)
    dn_bodies = {t: v["body"] for t, v in tgt.items()}
    # bodyId -> 全脑索引
    allids = D["ids"]; idx_of = {int(b): i for i, b in enumerate(allids.tolist())}
    dn_L = [idx_of[v["body"]] for v in tgt.values() if v["side"] == "L"]
    dn_R = [idx_of[v["body"]] for v in tgt.values() if v["side"] == "R"]
    brain = FlyBrain(data=data, device=device, seed=seeds[0], batch=1, sensory_input=False)
    L = len(vis)
    def drive(mask, amount):
        d = np.zeros(L, np.float32); d[mask] = amount; return brain.xp.asarray(d)
    drvL = drive(left_mask, STIM_A); drvR = drive(right_mask, STIM_A); drv0 = None
    out = []
    for sd in seeds:
        row = {"seed": sd}
        for cond, drv, side in (("base", drv0, None), ("left_eye", drvL, "L"), ("right_eye", drvR, "R")):
            brain.reset(sd)
            _dn_rate(brain, np.array(dn_L + dn_R), drv, BASE_STEPS, brain.dt)   # warm BASE
            rate, _ = _dn_rate(brain, np.array(dn_L + dn_R), drv, STIM_STEPS, brain.dt)
            # 分侧：粗测 L/R 靶
            brain.reset(sd)
            _dn_rate(brain, np.array(dn_L + dn_R), drv, BASE_STEPS, brain.dt)
            rL, _ = _dn_rate(brain, np.array(dn_L) if dn_L else np.array([0]), drv, STIM_STEPS, brain.dt)
            rR, _ = _dn_rate(brain, np.array(dn_R) if dn_R else np.array([0]), drv, STIM_STEPS, brain.dt)
            row[cond] = {"dn_rate": round(float(rate), 4), "L": round(float(rL), 4), "R": round(float(rR), 4)}
        # ipsi(左眼→左DN) 与 contra(左眼→右DN)
        row["left_ipsi"] = row["left_eye"]["L"]; row["left_contra"] = row["left_eye"]["R"]
        row["right_ipsi"] = row["right_eye"]["R"]; row["right_contra"] = row["right_eye"]["L"]
        row["baseL"] = row["base"]["L"]; row["baseR"] = row["base"]["R"]
        row["ipsi_gt_contra"] = bool((row["left_ipsi"] - row["left_contra"] > 0)
                                     and (row["right_ipsi"] - row["right_contra"] > 0))
        out.append(row)
    n_ok = sum(r["ipsi_gt_contra"] for r in out)
    false_bg = float(np.mean([max(r["baseL"], r["baseR"]) for r in out]))
    go = (n_ok >= STABLE_MIN) and (false_bg <= FALSE_MAX)
    return {"dn_set": {t: tgt[t] for t in tgt}, "n_dn_L": len(dn_L), "n_dn_R": len(dn_R),
            "params": {"STA_VPN": list(STA_VPN), "STIM_A": STIM_A, "seeds": seeds,
                       "BASE/STIM/POST": [BASE_STEPS, STIM_STEPS, POST_STEPS],
                       "RESP_MIN": RESP_MIN, "STABLE_MIN": STABLE_MIN, "FALSE_MAX": FALSE_MAX},
            "per_seed": out, "ipsi_gt_contra_seeds": n_ok, "false_bg": round(false_bg, 4),
            "verdict": "GO" if go else "NO-GO"}


def _response_main(seeds, device):
    D = load_data()
    r = visual_response(D, seeds=seeds, device=device)
    print(f"[P3-1 response] DN set={list(r['dn_set'])} L={r['n_dn_L']} R={r['n_dn_R']}")
    for row in r["per_seed"]:
        print(f"  seed{row['seed']} base(L/R)={row['baseL']}/{row['baseR']} "
              f"left_eye(L/R)={row['left_eye']['L']}/{row['left_eye']['R']} "
              f"right_eye(L/R)={row['right_eye']['L']}/{row['right_eye']['R']} "
              f"ipsi>contra={row['ipsi_gt_contra']}")
    print(f"  ipsi>contra seeds={r['ipsi_gt_contra_seeds']}/{len(seeds)} false_bg={r['false_bg']} -> {r['verdict']}")
    json.dump(r, open(os.path.join(HERE, "p31_response.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote p31_response.json")
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
