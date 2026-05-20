# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Local-only artifact helpers for the bundled YOLOv5 base."""

import urllib
from pathlib import Path

OFFLINE_ONLY_MESSAGE = "Bundled YOLOv5 base is offline-only; provide local files explicitly."


def is_url(url, check=True):
    """Determines if a string looks like a URL without touching the network."""
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return False if check else True
    except AssertionError:
        return False


def gsutil_getsize(url=""):
    """Reject cloud storage lookups in the bundled runtime base."""
    raise RuntimeError(OFFLINE_ONLY_MESSAGE)


def url_getsize(url="https://ultralytics.com/images/bus.jpg"):
    """Reject remote size checks in the bundled runtime base."""
    raise RuntimeError(OFFLINE_ONLY_MESSAGE)


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Reject curl downloads in the bundled runtime base."""
    raise RuntimeError(OFFLINE_ONLY_MESSAGE)


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """Reject all download attempts in the bundled runtime base."""
    raise RuntimeError(OFFLINE_ONLY_MESSAGE)


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    """Return a local artifact path or fail clearly; never fetch from the network."""
    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        raise FileNotFoundError(f"{OFFLINE_ONLY_MESSAGE} Missing local weight file: {file}")

    return str(file)
