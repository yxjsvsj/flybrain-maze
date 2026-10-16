"""P2d-26e：**官方 neuPrint 定向查询**（方案 A）——目标细胞突触坐标提取（完整执行流程）。

- 固定数据集 `male-cns:v1.0`；只用 `p2d26d_targets.json` 里**已确认的 88 个原始 bodyId**，
  **不按类型名重挑/替换**。
- token 只从**执行机安全配置**读取（环境变量或 neuprint 客户端配置）；**不输出/不上传/不写仓库**。
  未配置 → 明确提示 `缺少认证` 并停止（退出码 2），**不尝试匿名重复查询**。
- 正式查询前**先核对实际返回字段**（小批）：逐列探测，缺必需列则中止。
- 全程区分 **`returned_data` / `empty` / `request_failed`**；**缺失坐标的行绝不当作正常突触**。
- 前置/后置分别计数；同一突触的**多重 ROI 行**去重但**保留 ROI 归属**。
- 分批落盘（**中断后保留已完成进度**），结果小体量；原始大文件与 token **不入库**。

纯逻辑（`aggregate` / `run` / 持久化）可离线测试（注入 `fetch_one`）；网络部分需要 token。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback

TARGETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2d26d_targets.json")
DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"
GROUPS = ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)")

OK, EMPTY, FAILED, INTERRUPTED, NOTQ = (
    "returned_data", "empty", "request_failed", "interrupted", "not_queried")
_MARKERS = (FAILED, EMPTY)

COVERAGE_COLS = ["bodyId", "group", "status", "raw_rows", "unique_syn", "pre", "post",
                 "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "rois"]


# ---------------------------------------------------------------- targets / token
def load_targets(path: str = TARGETS) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    ids, group_of = [], {}
    for k in GROUPS:
        for b in d[k]:
            ids.append(int(b))
            group_of[int(b)] = k
    return {"dataset": d["dataset"], "groups": {k: list(d[k]) for k in GROUPS},
            "all_ids": ids, "group_of": group_of}


TOKEN_FILES = (os.path.join(os.path.expanduser("~"), ".neuprint"),
               os.path.join(os.path.expanduser("~"), ".config", "neuprint"),
               os.path.join(os.path.expanduser("~"), ".cloudvolume", "secrets",
                            "neuprint-secret.json"))


def load_token(env: str = "NEUPRINT_TOKEN") -> str | None:
    """只取执行机上合法配置的凭据；取不到返回 None（不申请、不猜、不落盘）。"""
    tok = os.environ.get(env) or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if tok:
        tok = tok.strip()
        if tok.startswith("{"):                     # env 里可能是整段 JSON
            try:
                tok = json.loads(tok).get("token", "")
            except Exception:                       # noqa: BLE001
                pass
        return tok or None
    for p in TOKEN_FILES:
        if os.path.isfile(p):
            try:
                d = json.load(open(p, "r", encoding="utf-8"))
                if isinstance(d, dict):
                    v = d.get("token") or d.get(env)
                    if v:
                        return str(v).strip()
            except Exception:                       # noqa: BLE001
                pass
    return None


def fetch_guard(token) -> None:
    """统一入口守卫：无 token 一律抛 PermissionError（绝不尝试匿名）。"""
    if not token:
        raise PermissionError("缺少认证：未找到 neuPrint token（不会尝试匿名请求）")


# ---------------------------------------------------------------- aggregation
def _norm_kind(v) -> str | None:
    s = str(v).lower()
    if "pre" in s and "post" not in s:
        return "pre"
    if "post" in s:
        return "post"
    return None


def _has_xyz(r) -> bool:
    return all(r.get(k) is not None for k in ("x", "y", "z"))


def aggregate(rows, key_fn=None):
    """每 bodyId 统计。rows 为**突触行**（须含 x/y/z）或**标记行**（含 `status`，无坐标）。

    - 只有带坐标的突触行才计数；**标记行（request_failed/empty）绝不当作突触**。
    - 同一突触的多重 ROI 行按 key_fn（默认 kind+x+y+z）去重，但**累计全部 ROI**。
    - 最终 status：有唯一突触 → returned_data；否则按标记 request_failed > empty。
    """
    def default_key(r):
        return (_norm_kind(r.get("kind")), int(r["x"]), int(r["y"]), int(r["z"]))
    kf = key_fn or default_key
    out: dict = {}

    def rec(bid):
        return out.setdefault(int(bid), {
            "bodyId": int(bid), "raw_rows": 0, "unique_syn": 0, "pre": 0, "post": 0,
            "rois": set(), "conf": set(), "_seen": set(), "_bbox": None, "_markers": set()})

    for r in rows:
        bid = r.get("bodyId", r.get("body"))
        if bid is None:
            continue
        e = rec(bid)
        if _has_xyz(r):
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
                    min(bb[0], x), min(bb[1], y), min(bb[2], z),
                    max(bb[3], x), max(bb[4], y), max(bb[5], z))
            if r.get("roi"):                        # 去重也要保 ROI 归属
                e["rois"].add(str(r["roi"]))
            if r.get("conf") is not None:
                e["conf"].add(float(r["conf"]))
        elif r.get("status") in _MARKERS:
            e["_markers"].add(r["status"])
    for e in out.values():
        e["rois"] = sorted(e["rois"])
        e["conf"] = sorted(e["conf"])
        bx = e.pop("_bbox")
        e["bbox"] = None if bx is None else {"x": [bx[0], bx[3]], "y": [bx[1], bx[4]],
                                             "z": [bx[2], bx[5]]}
        del e["_seen"]
        m = e.pop("_markers")
        if e["unique_syn"] > 0:
            e["status"] = OK
        elif FAILED in m:
            e["status"] = FAILED
        elif EMPTY in m:
            e["status"] = EMPTY
        else:
            e["status"] = NOTQ
    return out


# ---------------------------------------------------------------- neuPrint adapter
def describe_columns(cols) -> dict:
    """把实际列名映射到我们需要的键；返回 {key: real_col|None} + missing 列表。"""
    low = {str(c).lower(): c for c in cols}
    def pick(*names):
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None
    m = {"x": pick("x"), "y": pick("y"), "z": pick("z"),
         "kind": pick("type", "kind", "prepost", "pre_post"),
         "conf": pick("confidence", "conf"), "roi": pick("roi", "region")}
    m["missing_required"] = [k for k in ("x", "y", "z") if m[k] is None]
    m["missing_optional"] = [k for k in ("kind", "conf", "roi") if m[k] is None]
    return m


def frame_to_rows(bid, df, mapping) -> list:
    """DataFrame → 突触行。缺必需列抛错（不静默丢坐标）。"""
    if df is None or len(df) == 0:
        return []
    if mapping["missing_required"]:
        raise KeyError(f"缺少必需列 {mapping['missing_required']}；"
                       f"实际列={list(getattr(df, 'columns', []))}")
    d = df
    if mapping["x"] not in list(getattr(df, "columns", [])):
        idx_names = list(getattr(df.index, "names", []) or [])
        if any(n in idx_names for n in ("bodyId", "type", "roi")):
            d = df.reset_index()
    out = []
    for _, r in d.iterrows():
        out.append({"bodyId": int(bid),
                    "kind": r[mapping["kind"]] if mapping["kind"] else None,
                    "x": r[mapping["x"]], "y": r[mapping["y"]], "z": r[mapping["z"]],
                    "conf": r[mapping["conf"]] if mapping["conf"] else None,
                    "roi": r[mapping["roi"]] if mapping["roi"] else None})
    return out


def make_neuprint_client(token, dataset=DATASET, server=SERVER):
    if not token:
        raise PermissionError("缺少认证：未找到 neuPrint token（不会尝试匿名请求）")
    try:
        from neuprint import Client                      # type: ignore
    except Exception as e:                               # noqa: BLE001
        raise RuntimeError("需要官方 neuprint 客户端：pip install neuprint-python") from e
    return Client(server, dataset=dataset, token=token)


def make_fetch_one(client, mapping_holder):
    """返回 fetch_one(bid) -> list[突触行]；并支持先做 describe（字段核对）。"""
    from neuprint import fetch_synapses                  # type: ignore

    def fetch_one(bid):
        df = fetch_synapses([int(bid)], client=client)
        return frame_to_rows(bid, df, mapping_holder["mapping"])
    return fetch_one


def describe(client, sample_ids):
    """正式查询前的字段核对：拿小批返回的实际列名。"""
    from neuprint import fetch_synapses                  # type: ignore
    frame = None
    for bid in sample_ids:
        df = fetch_synapses([int(bid)], client=client)
        if df is not None and len(df) > 0:
            frame = df
            break
    if frame is None:
        return None, []                                  # 无可核对样本（不阻断，但记录）
    return describe_columns(list(frame.columns)), list(frame.columns)


# ---------------------------------------------------------------- orchestration
def _empty_out():
    return {"coverage": {}, "syn_rows": [], "status_by_body": {}}


def run(targets, fetch_one, outdir, *, describe_fn=None, sample_n=3, checkpoint=20,
        write=True, log=print):
    """完整流程：字段核对 → 小批验证 → 全部 88 → 落盘（含中断保留）。

    返回 {coverage, syn_rows, status_by_body, summary}。覆盖率 = 有 returned_data 的 body 数。
    """
    ids = list(targets["all_ids"])
    group_of = targets["group_of"]
    status_by_body = {b: NOTQ for b in ids}
    syn_rows = []

    def handle(bid):
        """-> (status, contributed_rows, note). 缺坐标的返回行绝不作为突触。"""
        try:
            rows = fetch_one(bid)
        except KeyboardInterrupt:
            raise
        except Exception as e:                           # noqa: BLE001
            return FAILED, [{"bodyId": bid, "status": FAILED, "error": type(e).__name__}], \
                f"request_failed({type(e).__name__})"
        if not rows:
            return EMPTY, [{"bodyId": bid, "status": EMPTY}], "empty"
        valid = [r for r in rows if _has_xyz(r)]
        if not valid:
            return FAILED, [{"bodyId": bid, "status": FAILED, "error": "no_coords"}], \
                f"no_coords(rows={len(rows)})"
        return OK, valid, f"n={len(valid)}"

    if describe_fn is not None:
        mapping, cols = describe_fn()
        log(f"[字段核对] 实际返回列 = {cols}")
        if mapping is not None:
            log(f"[字段核对] 映射 = {mapping}")
            if mapping.get("missing_required"):
                raise KeyError(f"字段核对失败：缺少必需列 {mapping['missing_required']}")
        else:
            log("[字段核对] 小批无返回样本 → 记录但继续（不据此判定数据不存在）")

    # 小批验证
    for bid in ids[:sample_n]:
        st, contrib, note = handle(bid)
        status_by_body[bid] = st
        syn_rows.extend(contrib)
        log(f"  小批 {bid}: {st} {note}")

    # 全部 88（跳过已验的小批，避免重复计数；小批行已入 syn_rows）
    interrupted = False
    cur_idx = sample_n
    try:
        for i in range(sample_n, len(ids)):
            bid = ids[i]
            cur_idx = i
            st, contrib, note = handle(bid)
            status_by_body[bid] = st
            syn_rows.extend(contrib)
            if st != OK:
                log(f"  {bid}: {st} {note}")
            if checkpoint and (i + 1) % checkpoint == 0:
                log(f"  ... {i + 1}/{len(ids)} 已查询（checkpoint 落盘）")
                if write:
                    _persist(outdir, targets, syn_rows, status_by_body)
    except KeyboardInterrupt:
        interrupted = True
        for b in ids[cur_idx:]:
            if status_by_body[b] == NOTQ:
                status_by_body[b] = INTERRUPTED
        log(f"  已中断：完成 {cur_idx}/{len(ids)}；其余标为 interrupted（保留进度）")

    cov = aggregate(syn_rows)
    if write:
        _persist(outdir, targets, syn_rows, status_by_body)
    n_data = sum(1 for b in ids if status_by_body[b] == OK)
    summary = {"total": len(ids), "returned_data": n_data,
               "empty": sum(1 for s in status_by_body.values() if s == EMPTY),
               "request_failed": sum(1 for s in status_by_body.values() if s == FAILED),
               "interrupted": sum(1 for s in status_by_body.values() if s == INTERRUPTED),
               "not_queried": sum(1 for s in status_by_body.values() if s == NOTQ),
               "coverage_frac": round(n_data / len(ids), 4), "interrupted_flag": interrupted}
    log(f"[汇总] {summary}")
    return {"coverage": cov, "syn_rows": syn_rows, "status_by_body": status_by_body,
            "summary": summary}


def _persist(outdir, targets, syn_rows, status_by_body):
    os.makedirs(outdir, exist_ok=True)
    group_of = targets["group_of"]
    cov = aggregate(syn_rows)
    # 覆盖表（88 行，状态优先取 status_by_body）
    cov_path = os.path.join(outdir, "p2d26d_coverage.csv")
    with open(cov_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COVERAGE_COLS)
        w.writeheader()
        for bid in targets["all_ids"]:
            e = cov.get(int(bid), {})
            st = status_by_body.get(bid, e.get("status", NOTQ))
            if st == NOTQ and e.get("status"):
                st = e["status"]
            bb = e.get("bbox") or {}
            def rng(ax):
                return (bb.get(ax, [None, None]) or [None, None])
            w.writerow({"bodyId": bid, "group": group_of.get(bid, ""), "status": st,
                        "raw_rows": e.get("raw_rows", 0), "unique_syn": e.get("unique_syn", 0),
                        "pre": e.get("pre", 0), "post": e.get("post", 0),
                        "x_min": rng("x")[0], "x_max": rng("x")[1],
                        "y_min": rng("y")[0], "y_max": rng("y")[1],
                        "z_min": rng("z")[0], "z_max": rng("z")[1],
                        "rois": ";".join(e.get("rois", []))})
    # 突触子集（小体量）
    syn_path = os.path.join(outdir, "p2d26e_coords.csv")
    with open(syn_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["bodyId", "kind", "x", "y", "z", "conf", "roi"])
        w.writeheader()
        for r in syn_rows:
            if _has_xyz(r):
                w.writerow({k: r.get(k) for k in ("bodyId", "kind", "x", "y", "z", "conf", "roi")})
    # 状态（中断/失败可恢复）
    st_path = os.path.join(outdir, "p2d26e_status.json")
    with open(st_path, "w", encoding="utf-8") as fh:
        json.dump({"dataset": targets["dataset"], "status_by_body": status_by_body},
                  fh, ensure_ascii=False, indent=0)
    return cov_path, syn_path, st_path


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    import argparse
    ap = argparse.ArgumentParser(description="P2d-26e official neuPrint targeted fetch")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--sample", type=int, default=3)
    ap.add_argument("--checkpoint", type=int, default=20)
    a = ap.parse_args(argv)

    t = load_targets()
    log = print
    log(f"[P2d-26e] dataset={t['dataset']}  targets={len(t['all_ids'])} "
        f"(EPG={len(t['groups']['EPG'])} PEN_a={len(t['groups']['PEN_a(PEN1)'])} "
        f"PEN_b={len(t['groups']['PEN_b(PEN2)'])})")
    tok = load_token()
    if not tok:
        log("缺少认证：执行机未配置 neuPrint token（NEUPRINT_TOKEN / ~/.neuprint）。")
        log("不尝试匿名请求；不判定为“数据不存在”。")
        log("请在 https://neuprint.janelia.org/account 取得 token 后配置到本机安全配置再运行。")
        return 2

    client = make_neuprint_client(tok, dataset=a.dataset)
    holder = {"mapping": None}

    def describe_fn():
        m, cols = describe(client, t["all_ids"][:a.sample])
        holder["mapping"] = m or {"x": None, "y": None, "z": None, "kind": None,
                                  "conf": None, "roi": None, "missing_required": ["x", "y", "z"]}
        return holder["mapping"], cols

    fetch_one = make_fetch_one(client, holder)
    res = run(t, fetch_one, a.outdir, describe_fn=describe_fn, sample_n=a.sample,
              checkpoint=a.checkpoint, log=log)
    return 0 if res["summary"]["coverage_frac"] > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
