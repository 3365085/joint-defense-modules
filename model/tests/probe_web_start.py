"""Probe the web console by hitting /api/start with a real sample clip,
then polling /api/status until it sees progress, then /api/stop.

Standalone (not pytest-collected) because it requires the web process to be
already running on 127.0.0.1:7861. Run manually while the server is up.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7861"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as rsp:
        return json.loads(rsp.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=15) as rsp:
        return json.loads(rsp.read().decode("utf-8"))


def main() -> int:
    sample = Path(__file__).resolve().parents[1] / "samples" / "glare_attacked.mp4"
    if not sample.exists():
        print(f"sample not found: {sample}")
        return 2
    print(f"[1] /api/test-source for {sample}")
    probe = post(
        "/api/test-source", {"source_type": "file", "source": str(sample)}
    )
    print(json.dumps(probe, ensure_ascii=False, indent=2))

    print("[2] /api/start")
    start = post(
        "/api/start",
        {
            "source_type": "file",
            "source": str(sample),
            "profile": "full_gpu",
            "realtime": False,
            "feature_options": {"static_image_enabled": True},
        },
    )
    print(json.dumps(start["status"].get("running"), ensure_ascii=False))

    last_idx = -1
    # Poll until the clip finishes or we see alert_confirmed.
    for i in range(80):
        time.sleep(0.5)
        status = get("/api/status")["status"]
        idx = int(status.get("frame_idx", 0) or 0)
        if idx != last_idx:
            print(
                f"frame={idx} alert={status.get('alert_confirmed')} "
                f"p_adv={status.get('p_adv')} reason={status.get('reason')[:60]!r}"
            )
            last_idx = idx
        if not status.get("running"):
            print("[server] run ended")
            break
    else:
        print("[warn] poll window exhausted without completion")

    print("[3] /api/stop")
    stop = post("/api/stop", {})
    print(json.dumps(stop.get("ok"), ensure_ascii=False))

    # Fetch final status for summary.
    status = get("/api/status")["status"]
    print(
        json.dumps(
            {
                "alert_event_count": status.get("alert_event_count"),
                "recent_events": status.get("recent_events", [])[:3],
                "error": status.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
