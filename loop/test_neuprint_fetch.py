"""P2d-26e 回归（单元）：neuprint_fetch 的纯逻辑（目标清单 / 缺认证 / 去重聚合 / 字段映射）。无网络。"""
from __future__ import annotations

import os
import sys

from loop.neuprint_fetch import (EMPTY, FAILED, OK, aggregate, describe_columns,
                                 frame_to_rows, load_targets, load_token)

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _Idx:
    names = ["bodyId", "type", "roi"]


class _Frame:
    """最小 DataFrame 替身：只用 columns / index.names / len / iterrows。"""
    def __init__(self, cols, rows):
        self.columns = cols
        self._rows = rows
        self.index = _Idx()

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


def main() -> int:
    print("=========== neuprint_fetch (unit) ===========")
    t = load_targets()
    check("targets total == 88", len(t["all_ids"]) == 88)
    check("EPG 46 / PEN_a 20 / PEN_b 22",
          (len(t["groups"]["EPG"]), len(t["groups"]["PEN_a(PEN1)"]),
           len(t["groups"]["PEN_b(PEN2)"])) == (46, 20, 22))
    check("no duplicate bodyId", len(set(t["all_ids"])) == 88)
    check("group_of covers all", len(t["group_of"]) == 88)
    check("dataset == male-cns:v1.0", t["dataset"] == "male-cns:v1.0")

    import loop.neuprint_fetch as nf
    saved = {k: os.environ.pop(k, None) for k in ("NEUPRINT_TOKEN", "NEUPRINT_APPLICATION_CREDENTIALS")}
    saved_files = nf.TOKEN_FILES
    nf.TOKEN_FILES = ()                              # 隔离本机 ~/.neuprint（否则会读到真实 token）
    try:
        check("no token -> None", load_token() is None)
    finally:
        nf.TOKEN_FILES = saved_files
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    os.environ["NEUPRINT_TOKEN"] = "FAKE"
    check("env token read", load_token() == "FAKE")
    del os.environ["NEUPRINT_TOKEN"]

    from loop.neuprint_fetch import fetch_guard
    try:
        fetch_guard(None)
        check("no-token guard raises", False)
    except PermissionError as e:
        check("no-token guard -> PermissionError(缺少认证)", "缺少认证" in str(e))

    # 聚合：多 ROI 去重保 ROI；前/后分离；bbox
    rows = [
        {"bodyId": 1, "kind": "PreSyn", "x": 10, "y": 20, "z": 30, "roi": "EB", "conf": 0.8},
        {"bodyId": 1, "kind": "PreSyn", "x": 10, "y": 20, "z": 30, "roi": "EBr1", "conf": 0.8},
        {"bodyId": 1, "kind": "PostSyn", "x": 12, "y": 25, "z": 33, "roi": "EB", "conf": 0.7},
        {"bodyId": 1, "kind": "PostSyn", "x": 5, "y": 18, "z": 31, "roi": "EB", "conf": 0.9},
    ]
    a = aggregate(rows)[1]
    check("raw_rows counts coord rows", a["raw_rows"] == 4)
    check("unique_syn dedups multi-ROI", a["unique_syn"] == 3)
    check("pre/post split", (a["pre"], a["post"]) == (1, 2))
    check("ROI preserved after dedup", a["rois"] == ["EB", "EBr1"])
    check("bbox from unique pts", a["bbox"] == {"x": [5, 12], "y": [18, 25], "z": [30, 33]})
    check("status returned_data", a["status"] == OK)

    # 标记行绝不当作突触
    f = aggregate([{"bodyId": 2, "status": FAILED}])[2]
    check("failed marker -> request_failed, 0 syn", f["status"] == FAILED and f["unique_syn"] == 0
          and f["raw_rows"] == 0 and f["bbox"] is None)
    e = aggregate([{"bodyId": 3, "status": EMPTY}])[3]
    check("empty marker -> empty, 0 syn", e["status"] == EMPTY and e["unique_syn"] == 0)
    mixed = aggregate([{"bodyId": 4, "status": FAILED},
                       {"bodyId": 4, "kind": "PreSyn", "x": 1, "y": 1, "z": 1}])[4]
    check("data wins over marker", mixed["status"] == OK and mixed["unique_syn"] == 1)

    # 字段映射
    m = describe_columns(["bodyId", "type", "x", "y", "z", "confidence", "roi"])
    check("describe maps required", m["missing_required"] == [] and m["x"] == "x"
          and m["kind"] == "type" and m["conf"] == "confidence")
    m2 = describe_columns(["bodyId", "y", "z"])           # 缺 x
    check("describe flags missing x", m2["missing_required"] == ["x"])

    # frame_to_rows：缺必需列要抛错，不静默丢坐标
    fr = _Frame(["bodyId", "type", "x", "y", "z", "confidence", "roi"],
                [{"bodyId": 9, "type": "PRE", "x": 1, "y": 2, "z": 3, "confidence": 0.5, "roi": "EB"}])
    out = frame_to_rows(9, fr, describe_columns(fr.columns))
    check("frame_to_rows parses a row", out and out[0]["x"] == 1 and out[0]["roi"] == "EB")
    try:
        frame_to_rows(9, _Frame(["bodyId", "y", "z"], [{"bodyId": 9, "y": 2, "z": 3}]),
                      describe_columns(["bodyId", "y", "z"]))
        check("frame_to_rows raises on missing x", False)
    except KeyError:
        check("frame_to_rows raises on missing x", True)

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
