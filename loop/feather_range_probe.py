"""P2d-26c：**惰性 HTTP-range 读** Arrow/feather 的元数据与小样（不整包下载）。

只读取：文件尾部的 footer/schema（`pa.ipc.open_file` 只 seek 需要处）+ 指定 record batch 的字节。
用于核实 syn-points / syn-partners / body-stats 的**真实字段**，避免把"未读到的列名"写成事实。
GCS 支持 `Range`（Accept-Ranges: bytes）。

用法： python -m loop.feather_range_probe --url <https .feather> --size <bytes> [--batch 0] [--only a,b]
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request


class RangeReader(io.RawIOBase):
    """可 seek 的只读流：read(n) -> HTTP Range GET [pos, pos+n)。"""

    def __init__(self, url: str, size: int, timeout: float = 60.0):
        self.url, self.size, self.timeout = url, int(size), timeout
        self.pos = 0
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1
                                               else self.size + offset)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = 64 * 1024
        if self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=self.timeout).read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def probe(url: str, size: int, batch: int = 0, only=None):
    import pyarrow as pa
    rr = RangeReader(url, size)
    reader = pa.ipc.open_file(pa.PythonFile(rr, mode="r"))
    names = [f.name for f in reader.schema]
    if only:
        keep = set(only)
        names = [n for n in names if n in keep] or names
        b = reader.get_batch(batch).select(names)
    else:
        b = reader.get_batch(batch)
    head = {n: b.column(n)[:3].to_pylist() for n in b.schema.names}
    return {"fields": [f.name for f in reader.schema], "num_batches": reader.num_record_batches,
            "batch_rows": b.num_rows, "batch_head": head, "range_bytes": rr.bytes_read}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    r = probe(a.url, a.size, a.batch, a.only.split(",") if a.only else None)
    print(f"num_batches={r['num_batches']}  batch_rows={r['batch_rows']}  "
          f"range_read≈{r['range_bytes'] / 1e6:.1f} MB（未整包下载）")
    print("fields:", r["fields"])
    print("batch_head:", r["batch_head"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
