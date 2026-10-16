"""P2d-26d：**官方 neuPrint** 目标细胞突触坐标提取（优先路线）。

- 固定数据集 `male-cns:v1.0`；只用 `p2d26d_targets.json` 里**已确认的 88 个原始 bodyId**，
  不按类型名重挑。
- 认证：只读**执行机上合法配置**的 token（环境变量或 neuprint 客户端配置）。
  **没有认证 → 明确报告"缺少认证"**，不反复试匿名，不把访问失败说成数据不存在。
- 先验小批（少数字段+身份），通过后再取全部 88；区分 **请求失败 / 查询为空 / 确实返回数据**。
- 前置/后置分别计数；同一突触的多重 ROI 行按 `key_fn` 去重但**保留 ROI 归属**。

纯逻辑 `aggregate()` 可离线测试；网络部分（`fetch`）需要 token。
"""
from __future__ import annotations

import json
import os
import sys

TARGETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2d26d_targets.json")
DATASET = "male-cns:v1.0"


def load_targets(path: str = TARGETS) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    ids = []
    for k in ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)"):
        ids += list(d[k])
    return {"dataset": d["dataset"], "groups": {k: d[k] for k in
            ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)")}, "all_ids": ids}


def load_token(env: str = "NEUPRINT_TOKEN") -> str | None:
    """只取执行机上合法配置的凭据；取不到返回 None（不申请、不猜）。"""
    tok = os.environ.get(env) or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if tok:
        tok = tok.strip()
        # 兼容写进 env 的整段 JSON {"token": "..."}
        if tok.startswith("{"):
            try:
                tok = json.loads(tok).get("token", "")
            except Exception:                       # noqa: BLE001
                pass
        return tok or None
    for p in (os.path.join(os.path.expanduser("~"), ".neuprint"),
              os.path.join(os.path.expanduser("~"), ".config", "neuprint"),
              os.path.join(os.path.expanduser("~"), ".cloudvolume", "secrets",
                           "neuprint-secret.json")):
        if os.path.isfile(p):
            try:
                d = json.load(open(p, "r", encoding="utf-8"))
                if isinstance(d, dict):
                    return d.get("token") or d.get(env) or None
            except Exception:                       # noqa: BLE001
                pass
    return None


def _norm_kind(v) -> str | None:
    s = str(v).lower()
    if "pre" in s:
        return "pre"
    if "post" in s:
        return "post"
    return None


def aggregate(rows, key_fn=None):
    """把原始行聚合成每 bodyId 的统计。rows: 可迭代 dict，需含
    `bodyId`（或 `body`）、`kind`（PreSyn/PostSyn 或 pre/post）、`x/y/z`、
    可选 `roi`/`conf`/`synId`。

    去重：同一突触的**多重 ROI 行**按 key_fn（默认 kind+x+y+z）合并为 1，
    但累计其 ROI 集合；前置/后置分别计数。
    """
    def default_key(r):
        return (_norm_kind(r.get("kind")), int(r["x"]), int(r["y"]), int(r["z"]))
    kf = key_fn or default_key
    out: dict = {}

    def rec(bid):
        return out.setdefault(bid, {"bodyId": int(bid), "raw_rows": 0, "unique_syn": 0,
                                    "pre": 0, "post": 0, "rois": set(), "conf": set(),
                                    "_seen": set(), "_bbox": None, "status": "ok"})
    for r in rows:
        bid = r.get("bodyId", r.get("body"))
        if bid is None:
            continue
        e = rec(bid)
        e["raw_rows"] += 1
        k = kf(r)
        if k not in e["_seen"]:
            e["_seen"].add(k)
            e["unique_syn"] += 1
            kind = _norm_kind(r.get("kind"))
            if kind == "pre":
                e["pre"] += 1
            elif kind == "post":
                e["post"] += 1
            x, y, z = int(r["x"]), int(r["y"]), int(r["z"])
            bb = e["_bbox"]
            e["_bbox"] = (x, y, z, x, y, z) if bb is None else (
                min(bb[0], x), min(bb[1], y), min(bb[2], z), max(bb[3], x), max(bb[4], y), max(bb[5], z))
        # ROI 归属：即使该点被去重，也要保留其多重 ROI 信息
        if r.get("roi"):
            e["rois"].add(str(r["roi"]))
        if r.get("conf") is not None:
            e["conf"].add(float(r["conf"]))
    for e in out.values():
        e["rois"] = sorted(e["rois"])
        e["conf"] = sorted(e["conf"])
        e["bbox"] = None if e["_bbox"] is None else {
            "x": [e["_bbox"][0], e["_bbox"][3]], "y": [e["_bbox"][1], e["_bbox"][4]],
            "z": [e["_bbox"][2], e["_bbox"][5]]}
        del e["_seen"], e["_bbox"]
        e["status"] = "returned_data" if e["unique_syn"] else "empty"
    return out


def fetch(body_ids, token, dataset=DATASET, server="https://neuprint.janelia.org",
          batch=500):
    """官方查询：优先 neuprint-python 客户端；否则抛明确错误。需 token。"""
    if not token:
        raise PermissionError("缺少认证：未找到 neuPrint token（不会尝试匿名请求）")
    try:
        from neuprint import Client, fetch_synapses     # type: ignore
    except Exception as e:                              # noqa: BLE001
        raise RuntimeError("需要官方 neuprint 客户端：pip install neuprint-python") from e
    c = Client(server, dataset=dataset, token=token)
    rows = []
    for i in range(0, len(body_ids), batch):
        chunk = list(body_ids[i:i + batch])
        # fetch_synapses 返回逐突触点（pre/post，3-D 坐标，置信度，ROI），bodyId 由调用方补
        for bid in chunk:
            try:
                s = fetch_synapses([bid], client=c)
            except Exception as e:                      # noqa: BLE001
                rows.append({"bodyId": bid, "status": f"failed:{type(e).__name__}"})
                continue
            if s is None or len(s) == 0:
                rows.append({"bodyId": bid, "status": "empty"})
                continue
            for _, r in s.iterrows():
                rows.append({"bodyId": bid, "kind": r.get("type"), "x": r.get("x"),
                             "y": r.get("y"), "z": r.get("z"), "conf": r.get("confidence"),
                             "roi": r.get("roi")})
    return rows


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                   # noqa: BLE001
        pass
    t = load_targets()
    print(f"[P2d-26d] dataset={t['dataset']}  targets={len(t['all_ids'])} "
          f"(EPG={len(t['groups']['EPG'])} PEN_a={len(t['groups']['PEN_a(PEN1)'])} "
          f"PEN_b={len(t['groups']['PEN_b(PEN2)'])})")
    tok = load_token()
    if not tok:
        print("缺少认证：执行机未配置 neuPrint token（NEUPRINT_TOKEN / ~/.neuprint）。")
        print("不尝试匿名请求；不判定为“数据不存在”。")
        print("后备：见 loop/P2D26D_REPORT.md 的 syn-partners 流式筛选方案（需预算确认）。")
        return 2
    print("已找到 token，开始小批核验……")
    sample = t["all_ids"][:3]
    rows = fetch(sample, tok)
    agg = aggregate(rows)
    print("小批返回：", {b: (e["unique_syn"], e["pre"], e["post"], e["status"])
                        for b, e in agg.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
