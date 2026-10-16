"""P2d-24c：**同源注释对接**（只读）。按 brain.npz.ids ↔ 官方 v1.0 bodyId 精确连接。

- 校验本机 brain.npz/weights.npz 的 sha256 是否等于 flybrain 预构建值（证明同源于 MaleCNS v1.0）。
- 用官方 `body-annotations-male-cns-v1.0-minconf-0.5.feather` 的 **bodyId** 精确 join（不按名称强配）。
- 报告 SNpp59 全部 6 个：本机索引 / bodyId / 侧别 / 官方 class/subclass/flywireType/type/instance/somaLocation 等真实字段。
- **不注入、不改 K、不接 P2Control、不重建/覆盖脑文件。**

用法： python -m loop.same_source_crosswalk --ann <path.feather>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--ann", required=True, help="官方 body-annotations-...v1.0...feather")
    args = ap.parse_args(argv)
    from flybrain.data import FILES, ensure_data
    data = ensure_data(args.data)
    print(f"[P2d-24c] 数据目录 = {data}")

    # 1) 校验值
    print("\n== 文件校验（本机 vs flybrain 预构建）==")
    ok = True
    for name, exp in FILES.items():
        got = sha256(os.path.join(data, name))
        m = got == exp
        ok = ok and m
        print(f"  {name:12s} {'MATCH' if m else 'MISMATCH'}  {got}")
    print(f"  → 同源于官方 brain-v1（MaleCNS v1.0 构建）: {ok}")

    meta = np.load(os.path.join(data, "brain.npz"))
    ids = np.asarray(meta["ids"])
    ct = np.array([str(x) for x in meta["cell_type"]])
    side = np.array([str(x) for x in meta["side"]])
    sup = np.array([str(x) for x in meta["superclass"]])

    # 2) 官方注释表（按 bodyId 精确 join；纯 pyarrow，不依赖 pandas）
    import pyarrow.feather as feather
    tab = feather.read_table(args.ann)
    cols = tab.column_names
    bid = np.asarray(tab.column("bodyId").to_numpy(), dtype=np.int64)
    pos = {int(b): i for i, b in enumerate(bid)}
    print("\n== 官方 v1.0 注释表 ==")
    print(f"  行数={tab.num_rows}  实际字段={cols}")
    print("  本机 brain.npz 未保存、仅官方表有的字段: class/subclass/supertype/entryNerve/exitNerve/"
          "receptorType/somaNeuromere/instance/statusLabel 等")

    def gv(i, k):
        if k not in cols:
            return "未提供"
        v = tab.column(k)[i].as_py()
        return "未提供" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    def show(name, limit=None):
        idx = np.flatnonzero(ct == name)
        print(f"\n== {name}: 本机命中 {len(idx)} ==")
        hdr = ("本机idx", "bodyId", "侧别", "superclass", "class", "subclass", "type",
               "entryNerve", "exitNerve", "receptorType", "somaNeuromere")
        print("  " + " ".join(f"{h:>12s}" for h in hdr))
        sel = idx[:limit] if limit else idx
        for i in sel:
            b = int(ids[i])
            j = pos.get(b)
            row = [str(int(i)), str(b), str(side[i]), str(sup[i]),
                   gv(j, "class") if j is not None else "未在官方表",
                   gv(j, "subclass") if j is not None else "未在官方表",
                   gv(j, "type") if j is not None else "未在官方表",
                   gv(j, "entryNerve") if j is not None else "未在官方表",
                   gv(j, "exitNerve") if j is not None else "未在官方表",
                   gv(j, "receptorType") if j is not None else "未在官方表",
                   gv(j, "somaNeuromere") if j is not None else "未在官方表"]
            print("  " + " ".join(f"{v:>12s}" for v in row))
            if j is not None:
                print(f"      instance={gv(j, 'instance')} flywireType={gv(j, 'flywireType')} "
                      f"somaSide={gv(j, 'somaSide')} rootSide={gv(j, 'rootSide')} "
                      f"somaLocation={_loc_col(tab, j, 'somaLocation')} "
                      f"tosomaLocation={_loc_col(tab, j, 'tosomaLocation')}")

    show("SNpp59")                 # 全部 6 个
    show("SNta29", limit=3)        # 仅类别核对
    show("LgLG3", limit=3)         # 仅类别核对
    print("\n[边界] 只做同源注释对接；不注入/不调 K/不接 P2Control/不重建脑文件。")
    return 0


def _loc_col(tab, j, k):
    if k not in tab.column_names:
        return "未提供"
    v = tab.column(k)[j].as_py()
    if v is None:
        return "未提供"
    try:
        a = np.asarray(v, float)
        return "未提供" if np.all(np.isnan(a)) else a.tolist()
    except Exception:                       # noqa: BLE001
        return str(v)


def _na(x):
    try:
        return x is None or bool(np.isscalar(x) and not isinstance(x, (str, bytes)) and np.isnan(x))
    except Exception:                       # noqa: BLE001
        return x is None


def _loc(r, k):
    if r is None or k not in r.index or _na(r[k]):
        return "未提供"
    v = r[k]
    try:
        return np.asarray(v, float).tolist()
    except Exception:                       # noqa: BLE001
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
