from __future__ import annotations

from defense.web.server import parse_args


def test_web_server_disables_access_log_by_default() -> None:
    args = parse_args([])

    assert args.access_log is False


def test_web_server_can_enable_access_log_explicitly() -> None:
    args = parse_args(["--access-log"])

    assert args.access_log is True
