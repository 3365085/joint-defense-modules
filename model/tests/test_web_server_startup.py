from __future__ import annotations

from defense.web import server


def test_stop_existing_local_monitor_kills_port_owners(monkeypatch) -> None:
    calls: list[list[str]] = []
    seen = {"count": 0}

    def fake_owners(port: int) -> set[int]:
        seen["count"] += 1
        return {111, 222} if seen["count"] == 1 else set()

    def fake_run(args, **kwargs):
        calls.append(list(args))

        class Result:
            stdout = ""

        return Result()

    monkeypatch.setattr(server, "_owning_pids_for_port", fake_owners)
    monkeypatch.setattr(server, "_existing_monitor_pids", lambda: set())
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.os, "getpid", lambda: 999)

    stopped = server.stop_existing_local_monitor(7860, timeout_s=0.1)

    assert stopped == [111, 222]
    assert calls == [
        ["taskkill", "/PID", "111", "/T", "/F"],
        ["taskkill", "/PID", "222", "/T", "/F"],
    ]


def test_stop_existing_local_monitor_skips_current_process(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(server, "_owning_pids_for_port", lambda port: {111, 222})
    monkeypatch.setattr(server, "_existing_monitor_pids", lambda: set())
    monkeypatch.setattr(server.subprocess, "run", lambda args, **kwargs: calls.append(list(args)))
    monkeypatch.setattr(server.os, "getpid", lambda: 111)

    stopped = server.stop_existing_local_monitor(7860, timeout_s=0.1)

    assert stopped == [222]
    assert calls == [["taskkill", "/PID", "222", "/T", "/F"]]


def test_stop_existing_local_monitor_kills_auto_port_monitor_process(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(server, "_owning_pids_for_port", lambda port: set())
    monkeypatch.setattr(server, "_existing_monitor_pids", lambda: {333})
    monkeypatch.setattr(server.subprocess, "run", lambda args, **kwargs: calls.append(list(args)))
    monkeypatch.setattr(server.os, "getpid", lambda: 999)

    stopped = server.stop_existing_local_monitor(7860, timeout_s=0.1)

    assert stopped == [333]
    assert calls == [["taskkill", "/PID", "333", "/T", "/F"]]
