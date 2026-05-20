from __future__ import annotations

import argparse
import os
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

from defense.runtime.config import DEFAULT_CONFIG_PATH


NO_REUSE_ENV = "MODULE_A_WEB_NO_REUSE"


def open_browser_later(url: str) -> None:
    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Timer(0.8, _open).start()


def warn_if_public_host(host: str) -> None:
    if str(host).strip() in {"0.0.0.0", "::"} and not os.environ.get("MODULE_A_WEB_TOKEN"):
        print("WARNING: binding to a public host without MODULE_A_WEB_TOKEN exposes control APIs.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module A web monitor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--auto-port", action="store_true")
    parser.add_argument("--reuse-port", action="store_true", help="Do not stop an existing local monitor on the requested port.")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--access-log", action="store_true", help="Enable per-request HTTP access logs for debugging.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args(argv)


def _owning_pids_for_port(port: int) -> set[int]:
    if os.name != "nt":
        return set()
    command = (
        "Get-NetTCPConnection -LocalPort "
        + str(int(port))
        + " -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def _existing_monitor_pids() -> set[int]:
    if os.name != "nt":
        return set()
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*-m defense.web.server*' } | "
        "Select-Object -ExpandProperty ProcessId -Unique"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def stop_existing_local_monitor(port: int, *, timeout_s: float = 8.0) -> list[int]:
    current_pid = os.getpid()
    stopped: list[int] = []
    pids = _owning_pids_for_port(port) | _existing_monitor_pids()
    for pid in sorted(pids):
        if pid == current_pid:
            continue
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)
            stopped.append(pid)
        except Exception:
            pass
    if not stopped:
        return stopped
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        remaining = (_owning_pids_for_port(port) | _existing_monitor_pids()) - {current_pid}
        if not remaining:
            break
        time.sleep(0.2)
    return stopped


def select_port(host: str, port: int, auto_port: bool) -> int:
    if not auto_port:
        return int(port)
    for candidate in range(int(port), int(port) + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((str(host), int(candidate)))
            except OSError:
                continue
            return int(candidate)
    return int(port)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from .fastapi_app import create_app
    import uvicorn

    if not args.reuse_port and not os.environ.get(NO_REUSE_ENV):
        stopped = stop_existing_local_monitor(int(args.port))
        if stopped:
            print(f"Stopped existing Module A web process tree on port {args.port}: {', '.join(str(pid) for pid in stopped)}")
    port = select_port(str(args.host), int(args.port), bool(args.auto_port))
    warn_if_public_host(str(args.host))
    app = create_app(config_path=Path(args.config), bind_host=str(args.host))
    url = f"http://{args.host}:{port}/"
    print(f"Module A monitor running at {url}")
    if args.open_browser:
        open_browser_later(url)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=str(args.host),
            port=port,
            log_level="warning" if args.quiet else "info",
            access_log=bool(args.access_log),
        )
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.state.engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
