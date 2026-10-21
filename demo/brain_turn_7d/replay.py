"""Local, read-only replay of archived DN readout turns.

No FlyBrain construction, no dynamics, no hardware imports, no motor output.
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


from . import audit

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "index.html"
MODES = audit.MODES
DEFAULT_SCENE = "A_OPEN"


def _clip(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _preview(v: float, omega: float) -> dict[str, Any]:
    left_mps = v - omega * audit.WHEEL_TRACK_M / 2.0
    right_mps = v + omega * audit.WHEEL_TRACK_M / 2.0
    return {
        "fixed_forward_mps": round(v, 4),
        "left_mps": round(left_mps, 4),
        "right_mps": round(right_mps, 4),
        "left_cmd": round(_clip(left_mps / audit.MAX_WHEEL_MPS, -1.0, 1.0), 4),
        "right_cmd": round(_clip(right_mps / audit.MAX_WHEEL_MPS, -1.0, 1.0), 4),
        "sent": False,
        "note": "engineering preview only; no /api/drive, no Arduino, no motor",
    }


def _frame_out(scene: dict[str, Any], frame: dict[str, Any], mode: str) -> dict[str, Any]:
    raw = float(frame["neural_turn_raw"])
    if mode == "BRAIN":
        turn = raw
        ablation = False
    elif mode == "ZERO":
        turn = 0.0
        ablation = True
    elif mode == "FLIP":
        turn = -raw
        ablation = True
    else:
        raise ValueError(f"unknown mode {mode}")
    omega = float(audit.MAX_OMEGA * turn)
    return {
        "scene": scene["id"],
        "scene_label": scene["label"],
        "mode": mode,
        "i": frame["i"],
        "n": scene["frames_active"],
        "t": frame["t"],
        "tof_mm": frame["tof_mm"],
        "tof_status": frame["tof_status"],
        "neural": {
            "features": "1314 DN Trace values (archived)",
            "feature_bins": frame["feature_bins"],
            "feature_l2": frame["feature_l2"],
            "brain_fired_count": scene["brain_fired_count"]["median"],
            "turn_raw": round(raw, 6),
            "turn_source": "frozen readout_real.npz",
        },
        "brain_turn_raw": round(raw, 6),
        "brain_turn": round(turn, 6),
        "turn_source": "archived DN readout" if mode == "BRAIN" else f"ablation:{mode}",
        "turn_ablation": ablation,
        "memory_gain": 0.0,
        "pursuit_turn": 0.0,
        "omega_rad_s": round(omega, 6),
        "motor_preview": _preview(audit.FIXED_FORWARD_MPS, omega),
        "safety": {
            "state": "REPLAY_ONLY",
            "actuators": "DISCONNECTED",
            "drive_api": "NOT_PRESENT",
            "motor_output": "PREVIEW_ONLY",
            "web_post": "FORBIDDEN",
        },
        "source": {
            "csv": scene["source_csv"],
            "feats": scene["source_feats"],
            "csv_sha256": scene["csv_sha256"],
            "feats_sha256": scene["feats_sha256"],
        },
    }


class ReplayData:
    def __init__(self) -> None:
        self.report = audit.build_report()
        self.readout = self.report["readout"]
        self.scenes = {
            sid: s for sid, s in self.report["scenes"].items()
            if s.get("status") != "MISSING"
        }

    def scene_ids(self) -> list[str]:
        return list(self.scenes)

    def scene(self, sid: str) -> dict[str, Any]:
        if sid not in self.scenes:
            raise KeyError(sid)
        return self.scenes[sid]

    def replay(self, sid: str, mode: str) -> list[dict[str, Any]]:
        if mode not in MODES:
            raise KeyError(f"mode {mode}")
        scene = self.scene(sid)
        return [_frame_out(scene, f, mode) for f in scene["_frames"]]

    def meta(self) -> dict[str, Any]:
        return {
            "task": self.report["task"],
            "baseline": self.report["baseline"],
            "verdict": self.report["verdict"],
            "reason_codes": self.report["reason_codes"],
            "readout": self.readout,
            "modes": list(MODES),
            "default_scene": DEFAULT_SCENE,
            "scenes": [
                {"id": sid, "label": s["label"], "class": s["class"], "frames": s["frames_active"]}
                for sid, s in self.scenes.items()
            ] + [{"id": "B_LEFT", "label": "B 左侧障碍 — NO ARCHIVE", "class": "B", "frames": 0,
                  "missing": True}],
            "safety": {
                "state": "REPLAY_ONLY",
                "actuators": "DISCONNECTED",
                "drive_api": "NOT_PRESENT",
                "motor_output": "PREVIEW_ONLY",
                "web_post": "FORBIDDEN",
            },
            "constants": {
                "fixed_forward_mps": audit.FIXED_FORWARD_MPS,
                "wheel_track_m": audit.WHEEL_TRACK_M,
                "max_omega_rad_s": audit.MAX_OMEGA,
                "memory_gain": 0.0,
            },
        }

    def sources(self) -> list[dict[str, Any]]:
        return self.report["archive_manifest"]


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  data: ReplayData | None = None) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("replay is local-only; host must be loopback")
    dataset = data or ReplayData()
    handler = make_handler(dataset)
    return ThreadingHTTPServer((host, port), handler)


def make_handler(dataset: ReplayData):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BrainTurnReplay/1.0"

        def log_message(self, format, *args):  # noqa: A002
            return

        def _json(self, obj: Any, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self) -> None:
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._html()
                return
            if u.path == "/api/meta":
                self._json(dataset.meta())
                return
            if u.path == "/api/sources":
                self._json({"sources": dataset.sources()})
                return
            if u.path == "/api/frame":
                q = parse_qs(u.query)
                sid = (q.get("scene") or [DEFAULT_SCENE])[0]
                mode = (q.get("mode") or ["BRAIN"])[0]
                try:
                    scene = dataset.scene(sid)
                    i = int((q.get("i") or [0])[0])
                    if i < 0 or i >= scene["frames_active"]:
                        raise ValueError("frame index out of range")
                    self._json(_frame_out(scene, scene["_frames"][i], mode))
                except (KeyError, ValueError) as exc:
                    self._json({"error": str(exc)}, 400)
                return
            if u.path == "/api/replay":
                q = parse_qs(u.query)
                sid = (q.get("scene") or [DEFAULT_SCENE])[0]
                mode = (q.get("mode") or ["BRAIN"])[0]
                try:
                    self._json({"frames": dataset.replay(sid, mode)})
                except KeyError as exc:
                    self._json({"error": str(exc)}, 400)
                return
            if u.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            self._json({"error": "POST forbidden: replay is read-only"}, 405)

        def do_PUT(self):  # noqa: N802
            self._json({"error": "PUT forbidden: replay is read-only"}, 405)

        def do_DELETE(self):  # noqa: N802
            self._json({"error": "DELETE forbidden: replay is read-only"}, 405)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D1 local replay: archived neural turn, no actuators")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args(argv)
    data = ReplayData()
    srv = create_server(args.host, args.port, data)
    url = f"http://{args.host}:{srv.server_address[1]}/"
    print(f"[d1-replay] {url}")
    print(f"[d1-replay] verdict={data.report['verdict']} readout={data.readout['sha256']}")
    print("[d1-replay] REPLAY ONLY: no brain, no motor, no POST drive route")
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
