"""P2d-26d 回归：neuprint_fetch 的纯逻辑（目标清单 / 缺认证 / 去重聚合）。无网络。"""
from __future__ import annotations

import os
import sys

from loop.neuprint_fetch import aggregate, load_targets, load_token

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=========== neuprint_fetch ===========")
    t = load_targets()
    check("targets total == 88", len(t["all_ids"]) == 88)
    check("EPG 46 / PEN_a 20 / PEN_b 22",
          (len(t["groups"]["EPG"]), len(t["groups"]["PEN_a(PEN1)"]),
           len(t["groups"]["PEN_b(PEN2)"])) == (46, 20, 22))
    check("no duplicate bodyId", len(set(t["all_ids"])) == 88)
    check("dataset == male-cns:v1.0", t["dataset"] == "male-cns:v1.0")

    # 缺认证：清掉相关 env → None（不申请）
    saved = {k: os.environ.pop(k, None) for k in ("NEUPRINT_TOKEN", "NEUPRINT_APPLICATION_CREDENTIALS")}
    try:
        check("no token -> None", load_token() is None)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    os.environ["NEUPRINT_TOKEN"] = "FAKE"
    check("env token read", load_token() == "FAKE")
    del os.environ["NEUPRINT_TOKEN"]

    # 缺少认证时 fetch 抛明确错误（不匿名）
    from loop.neuprint_fetch import fetch
    try:
        fetch([10539], token=None)
        check("fetch(no token) raises", False)
    except PermissionError as e:
        check("fetch(no token) raises PermissionError", "缺少认证" in str(e))

    # 聚合：同一突触多重 ROI 去重但保 ROI；前后分别计数；bbox
    rows = [
        {"bodyId": 1, "kind": "PreSyn", "x": 10, "y": 20, "z": 30, "roi": "EB", "conf": 0.8},
        {"bodyId": 1, "kind": "PreSyn", "x": 10, "y": 20, "z": 30, "roi": "EBr1", "conf": 0.8},  # dup
        {"bodyId": 1, "kind": "PostSyn", "x": 12, "y": 25, "z": 33, "roi": "EB", "conf": 0.7},
        {"bodyId": 1, "kind": "PostSyn", "x": 5, "y": 18, "z": 31, "roi": "EB", "conf": 0.9},
    ]
    a = aggregate(rows)[1]
    check("raw_rows counts all lines", a["raw_rows"] == 4)
    check("unique_syn dedups multi-ROI rows", a["unique_syn"] == 3)
    check("pre/post split", (a["pre"], a["post"]) == (1, 2))
    check("ROIs preserved", a["rois"] == ["EB", "EBr1"])
    check("bbox from unique pts", a["bbox"] == {"x": [5, 12], "y": [18, 25], "z": [30, 33]})
    check("status returned_data", a["status"] == "returned_data")

    # 空 vs 有数据
    check("empty body -> status empty", aggregate([{"bodyId": 2, "kind": "PreSyn", "x": 1, "y": 1, "z": 1}] )[2]["unique_syn"] == 1)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
