"""P2d-26c 回归：range 读的流语义（seek/read 契约 + Range 头）——用本地 HTTP 服务模拟。无需脑。"""
from __future__ import annotations

import functools
import http.server
import io
import os
import socketserver
import sys
import tempfile
import threading

import pyarrow as pa

from loop.feather_range_probe import RangeReader, probe

RESULTS = []


def check(label, ok):
    RESULTS.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):              # 静默
        pass


class _RangeHandler(_Handler):
    def send_head(self):
        path = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().send_head()
        lo, hi = rng[6:].split("-")
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            lo = int(lo) if lo else 0
            hi = int(hi) if hi else size - 1
            hi = min(hi, size - 1)
            fh.seek(lo)
            data = fh.read(hi - lo + 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        return io.BytesIO(data)


def main() -> int:
    print("=========== feather_range_probe ===========")
    d = tempfile.mkdtemp()
    tbl = pa.table({"x": pa.array(list(range(1000)), pa.int32()),
                    "body": pa.array(list(range(2000, 3000)), pa.int64())})
    path = os.path.join(d, "t.feather")
    with pa.OSFile(path, "wb") as sink:
        with pa.ipc.new_file(sink, tbl.schema) as w:
            w.write_table(tbl)
    size = os.path.getsize(path)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(_RangeHandler, directory=d))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/t.feather"

    # 流语义
    rr = RangeReader(url, size)
    rr.seek(0)
    check("RangeReader.read from 0", rr.read(8) == open(path, "rb").read(8))
    rr.seek(4)
    check("RangeReader.read honors seek", rr.read(4) == open(path, "rb").read()[4:8])
    rr.seek(-8, 2)
    check("RangeReader.seek(whence=2)", rr.read(8) == open(path, "rb").read()[-8:])

    # 完整探测（schema + batch），且只读很小一段
    r = probe(url, size, 0)
    check("probe fields parsed", r["fields"] == ["x", "body"])
    check("probe batch_rows == 1000", r["batch_rows"] == 1000)
    check("probe sample values", r["batch_head"]["body"][:3] == [2000, 2001, 2002])
    check("probe range-read < full file", r["range_bytes"] < size)
    httpd.shutdown()

    npass = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{npass}/{len(RESULTS)} " + ("ALL PASS" if npass == len(RESULTS) else "FAILED"))
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
