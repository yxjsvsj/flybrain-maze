"""P2d-26e 回归（离线端到端）：完整执行流程——部分失败 / 部分为空 / 多 ROI 重复 /
全部 88 目标保留 / 坐标持久化 / 中断后状态记录。**无网络**（注入伪 fetch_one）。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

from loop.neuprint_fetch import (EMPTY, FAILED, INTERRUPTED, OK, aggregate,
                                 load_targets, run)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def make_faker(ids, interrupt_at=None):
    """按索引决定 失败/空/数据；首个数据细胞含一条多重 ROI 重复行。"""
    def fetch_one(bid):
        i = ids.index(bid)
        if interrupt_at is not None and i == interrupt_at:
            raise KeyboardInterrupt
        if i % 7 == 3:
            raise RuntimeError("simulated request failure")
        if i % 5 == 2:
            return []                                    # 查询为空
        rows = [{"bodyId": bid, "kind": "PreSyn", "x": 100 + i, "y": 200 + i, "z": 300 + i,
                 "conf": 0.9, "roi": "EB"},
                {"bodyId": bid, "kind": "PostSyn", "x": 101 + i, "y": 201 + i, "z": 301 + i,
                 "conf": 0.8, "roi": "EB"}]
        if i == 0:                                       # 同一突触的多重 ROI 重复行
            rows.append({"bodyId": bid, "kind": "PreSyn", "x": 100 + i, "y": 200 + i, "z": 300 + i,
                         "conf": 0.9, "roi": "EBr1"})
        return rows
    return fetch_one


def read_coverage(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    print("=========== neuprint_fetch (e2e offline) ===========")
    t = load_targets()
    ids = t["all_ids"]
    silent = lambda *a, **k: None

    # ---- 1) 完整 88 ----
    d = tempfile.mkdtemp()
    res = run(t, make_faker(ids), d, checkpoint=20, log=silent)
    s = res["summary"]
    exp_fail = sum(1 for i in range(len(ids)) if i % 7 == 3)
    exp_empty = sum(1 for i in range(len(ids)) if i % 5 == 2 and i % 7 != 3)
    exp_data = len(ids) - exp_fail - exp_empty
    check("all 88 have a status", set(res["status_by_body"]) == set(ids))
    check("no not_queried after full run", s["not_queried"] == 0)
    check("returned_data count", s["returned_data"] == exp_data)
    check("empty count", s["empty"] == exp_empty)
    check("request_failed count", s["request_failed"] == exp_fail)
    check("counts sum to 88", s["returned_data"] + s["empty"] + s["request_failed"] == 88)
    check("coverage_frac correct", abs(s["coverage_frac"] - exp_data / 88) < 5e-5)

    cov = read_coverage(os.path.join(d, "p2d26d_coverage.csv"))
    check("coverage CSV has 88 rows", len(cov) == 88)
    check("coverage CSV preserves all bodyIds", {int(r["bodyId"]) for r in cov} == set(ids))
    check("groups assigned", all(r["group"] for r in cov))
    c0 = [r for r in cov if int(r["bodyId"]) == ids[0]][0]
    check("body0 unique_syn==2 (dup merged)", c0["unique_syn"] == "2")
    check("body0 raw_rows==3", c0["raw_rows"] == "3")
    check("body0 rois keep both", "EB" in c0["rois"] and "EBr1" in c0["rois"])
    check("failed row has no coords", all(r["x_min"] == "" for r in cov if r["status"] == FAILED))
    check("empty row status empty", any(r["status"] == EMPTY for r in cov))

    coords = list(csv.DictReader(open(os.path.join(d, "p2d26e_coords.csv"), newline="", encoding="utf-8")))
    check("coords persisted (all coord rows)", len(coords) == 2 * exp_data + 1)
    check("coords carry ROI", all(r["roi"] for r in coords))
    st = json.load(open(os.path.join(d, "p2d26e_status.json"), encoding="utf-8"))
    check("status json has 88", len(st["status_by_body"]) == 88)

    # ---- 2) 中断：保留已完成进度 ----
    d2 = tempfile.mkdtemp()
    res2 = run(t, make_faker(ids, interrupt_at=10), d2, checkpoint=20, log=silent)
    st2 = res2["status_by_body"]
    check("interrupt recorded", res2["summary"]["interrupted_flag"] is True)
    check("completed before interrupt kept", st2[ids[5]] in (OK, EMPTY, FAILED))
    check("current body marked interrupted", st2[ids[10]] == INTERRUPTED)
    check("rest marked interrupted", all(st2[b] == INTERRUPTED for b in ids[10:]))
    check("interrupted persisted to CSV",
          sum(1 for r in read_coverage(os.path.join(d2, "p2d26d_coverage.csv"))
              if r["status"] == INTERRUPTED) == len(ids) - 10)

    # ---- 3) 字段核对失败 -> 中止 ----
    def bad_describe():
        return {"x": None, "y": "y", "z": "z", "kind": "type", "conf": None, "roi": None,
                "missing_required": ["x"]}, ["bodyId", "type", "y", "z"]
    d3 = tempfile.mkdtemp()
    try:
        run(t, make_faker(ids), d3, describe_fn=bad_describe, log=silent)
        check("describe missing-required aborts", False)
    except KeyError:
        check("describe missing-required aborts", True)

    # ---- 4) 返回行缺坐标 -> 不算突触 ----
    def no_coord_fetch(bid):
        return [{"bodyId": bid, "kind": "PreSyn", "roi": "EB"}]     # 无 x/y/z
    d4 = tempfile.mkdtemp()
    res4 = run(t, no_coord_fetch, d4, sample_n=1, log=silent)
    check("rows without coords -> request_failed",
          all(v == FAILED for v in res4["status_by_body"].values()))
    check("no phantom synapses", all(e["unique_syn"] == 0 and e["bbox"] is None
                                     for e in aggregate(res4["syn_rows"]).values()))

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
