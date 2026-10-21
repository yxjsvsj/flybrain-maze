"""D2 launcher that reuses the D1 read-only REPLAY dashboard.

The page remains D1's GET-only replay. This wrapper only prints the D2
feasibility status and refuses non-loopback binding; it has no motor path.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser

from demo.brain_turn_7d.replay import ReplayData, create_server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D2 read-only dashboard (reuses D1 REPLAY)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args(argv)
    data = ReplayData()
    srv = create_server(args.host, args.port, data)
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print(f"[d2-dashboard] {url}")
    print("[d2-dashboard] D1=NO_GO D2=INSUFFICIENT_EVIDENCE")
    print("[d2-dashboard] REPLAY ONLY; no new brain, no training, no motor, no POST drive")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
