"""P2d-12 任务三门槛：当前 1314 维下行神经元特征对 A/FR/FL/D 场景的**线性可分辨性**。

离线读 `physical_static_diag` 的 `*_feats.npz`（每帧 1314 维真实脑 Trace）+ 对应 CSV 的
五路 mm；用**留一运行交叉验证**的线性 ridge 探针比较：
  - DN 探针（1314 维神经痕迹） -> 场景类别；
  - 传感器探针（FR_mm, FL_mm 2 维） -> 场景类别（对照基线）。
若 DN 明显低于传感器基线，说明当前神经特征对 FR/FL 信息不足，不应靠标签/参数强造能力。

用法：  python -m loop.p2_readout_recoverability --dir <cal17或cal18目录>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np


def _label(name):
    base = os.path.basename(name).replace("static_", "")
    c = base[0].upper()
    return {"A": "A", "B": "FR", "C": "FL", "D": "D"}.get(c)


def _load_run(feats_path):
    d = np.load(feats_path, allow_pickle=True)
    X = np.asarray(d["feats"], np.float32)
    seq = np.asarray(d["tof_seq"]).ravel()
    csv_path = feats_path.replace("_feats.npz", ".csv")
    S = np.full((len(X), 2), np.nan, np.float32)
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        by_seq = {int(r["tof_seq"]): r for r in rows if r.get("tof_seq") not in (None, "")}
        for i, sq in enumerate(seq):
            r = by_seq.get(int(sq))
            if r is not None:
                fr = r.get("FR_mm")
                fl = r.get("FL_mm")
                S[i, 0] = float(fr) if fr not in (None, "") else np.nan
                S[i, 1] = float(fl) if fl not in (None, "") else np.nan
    return X, S


def _ridge_fit(X, Y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1), np.float32)])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xb.T @ Y)


def _probe_loro(runs, keys, dims, ncls):
    """留一运行：train on other runs -> predict held-out; 返回准确率 + 混淆。"""
    conf = np.zeros((ncls, ncls), int)
    for k, (f, (X, S, y, lab)) in runs.items():
        tr_X = np.vstack([runs[j][1][0] for j in runs if j != k])
        tr_S = np.vstack([runs[j][1][1] for j in runs if j != k])
        tr_y = np.concatenate([runs[j][1][2] for j in runs if j != k])
        Ytr = np.eye(ncls, dtype=np.float32)[tr_y]
        W = _ridge_fit(tr_X if keys == "dn" else tr_S, Ytr, lam=1.0)
        Xte = X if keys == "dn" else S
        Xte = np.nan_to_num(Xte, nan=0.0)
        pred = np.argmax(np.hstack([Xte, np.ones((len(Xte), 1), np.float32)]) @ W, axis=1)
        for t, p in zip(y, pred):
            conf[t, p] += 1
    acc = np.trace(conf) / conf.sum()
    return acc, conf


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                           # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "static_*_feats.npz")))
    classes = ["A", "FR", "FL", "D"]
    cidx = {c: i for i, c in enumerate(classes)}
    runs = {}
    for f in files:
        lab = _label(f)
        if lab is None:
            continue
        X, S = _load_run(f)
        y = np.full(len(X), cidx[lab], int)
        runs[os.path.basename(f)] = (f, (X, np.nan_to_num(S, nan=0.0), y, lab))
    print(f"[recoverability] {len(runs)} 运行, 特征维 {next(iter(runs.values()))[1][0].shape[1]}, "
          f"类别 {classes}")
    ncls = len(classes)
    acc_dn, conf_dn = _probe_loro(runs, "dn", None, ncls)
    acc_s, conf_s = _probe_loro(runs, "sensor", 2, ncls)
    print(f"  DN(1314)     留一准确率 = {acc_dn:.3f}")
    print(f"  sensor(FR/FL) 留一准确率 = {acc_s:.3f}")
    print(f"  DN 混淆(行=真, 列=预测, 序 {classes}):")
    for r, name in enumerate(classes):
        print(f"    {name:3s} {conf_dn[r]}")
    print("  注：sensor 探针用 FR/FL mm（本次平板场景），是**信息上界**参照。")

    # 逐类二分类更容易看：FR-only vs 其余
    for pos, neg, tag in ((["FR"], ["A", "FL", "D"], "FR-only"),
                          (["FL"], ["A", "FR", "D"], "FL-only")):
        accs = []
        for i, k in enumerate(runs):
            tr = [runs[j][1] for j in runs if j != k]
            Xtr = np.vstack([t[0] for t in tr])
            ytr = np.concatenate([np.isin(t[2], [cidx[c] for c in pos]).astype(int) for t in tr])
            W = _ridge_fit(Xtr, np.eye(2, dtype=np.float32)[ytr], 1.0)
            Xte = runs[k][1][0]
            p = np.argmax(np.hstack([Xte, np.ones((len(Xte), 1), np.float32)]) @ W, axis=1)
            yte = np.isin(runs[k][1][2], [cidx[c] for c in pos]).astype(int)
            accs.append(float((p == yte).mean()))
        print(f"  DN 二分类 {tag} 留一准确率均值 = {np.mean(accs):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
