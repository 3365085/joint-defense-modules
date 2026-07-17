from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from defense.diagnostics.authoritative_manifest import (  # noqa: E402
    load_authoritative_manifest,
)
from defense.pipelines.derived_video_cache import (  # noqa: E402
    canonical_payload_sha256,
    default_derived_video_cache_root,
    resolve_derived_video_source,
    sha256_file,
)
from defense.pipelines.video_decoder_factory import (  # noqa: E402
    create_video_decoder,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs"
    / "acceptance"
    / "module_a_authoritative_manifest_v1.json"
)
PROFILE_ID = "h264_nvenc_lossless_yuv420p_v1"
PROFILE_PAYLOAD = {
    "container": "mp4",
    "video_codec": "h264",
    "pixel_format": "yuv420p",
    "geometry_policy": "preserve",
    "frame_rate_policy": "passthrough",
    "timestamp_policy": "passthrough",
    "audio_policy": "drop",
    "encoder": "h264_nvenc",
    "preset": "p4",
    "tune": "lossless",
}


def _run(
    command: list[str],
    *,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        command,
        capture_output=True,
        check=True,
        **(
            {}
            if binary
            else {
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
        ),
    )


def _probe(ffprobe: str, path: Path) -> dict[str, Any]:
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=codec_name,codec_long_name,profile,width,height,"
                "r_frame_rate,avg_frame_rate,nb_frames,duration,pix_fmt"
            ),
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(completed.stdout)


def _decoded_framemd5_sha256(ffmpeg: str, path: Path) -> tuple[str, int]:
    completed = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "framemd5",
            "-",
        ],
        binary=True,
    )
    payload = completed.stdout
    return hashlib.sha256(payload).hexdigest().upper(), len(payload)


def _first_stream(probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise RuntimeError("video_stream_missing")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError("video_stream_metadata_invalid")
    return stream


def _duration_within_one_frame(
    source_stream: dict[str, Any],
    derived_stream: dict[str, Any],
) -> tuple[float, bool]:
    numerator_text, denominator_text = str(
        source_stream["avg_frame_rate"]
    ).split("/", 1)
    fps = float(numerator_text) / max(1.0, float(denominator_text))
    delta = abs(
        float(derived_stream["duration"])
        - float(source_stream["duration"])
    )
    return delta, delta <= (1.0 / max(1.0, fps)) + 1e-6


def _verify_nvdec(path: Path, expected_frames: int) -> dict[str, Any]:
    decoder = create_video_decoder(
        path,
        preference="nvdec",
        allow_cpu_fallback=False,
    )
    frames = 0
    started = time.perf_counter()
    try:
        while True:
            lease = decoder.read()
            if lease is None:
                break
            lease.release()
            frames += 1
        status = dict(decoder.status_snapshot())
    finally:
        decoder.close()
    wall_time_s = time.perf_counter() - started
    if frames != expected_frames:
        raise RuntimeError(
            f"nvdec_frame_count_mismatch:{frames}:{expected_frames}"
        )
    if str(status.get("effective_backend") or "") != "nvdec":
        raise RuntimeError(
            "nvdec_effective_backend_mismatch:"
            f"{status.get('effective_backend')}"
        )
    if int(status.get("fallback_count") or 0) != 0:
        raise RuntimeError(
            f"nvdec_fallback_not_zero:{status.get('fallback_count')}"
        )
    return {
        "frames_read": frames,
        "wall_time_s": wall_time_s,
        "throughput_fps": frames / max(wall_time_s, 1e-9),
        "status": status,
    }


def build_nvdec_derived_video(
    *,
    manifest_path: Path,
    asset_id: str,
    cache_root: Path,
    ffmpeg: str,
    ffprobe: str,
    force: bool,
) -> dict[str, Any]:
    manifest = load_authoritative_manifest(
        manifest_path,
        verify_files=True,
        strict_counts=True,
    )
    asset = manifest.asset_by_id(asset_id)
    if asset.category == "model":
        raise ValueError("asset_id must identify a video")
    source_path = Path(asset.canonical_path).resolve(strict=True)
    if sha256_file(source_path) != asset.sha256.upper():
        raise RuntimeError("authoritative_source_sha256_mismatch")

    if not force:
        existing = resolve_derived_video_source(
            source_path,
            cache_root=cache_root,
        )
        if existing is not None:
            return {
                "ok": True,
                "reused": True,
                "asset_id": asset.asset_id,
                "source_path": str(existing.source_path),
                "source_sha256": existing.source_sha256,
                "decode_source": str(existing.decode_path),
                "decode_source_sha256": existing.derived_sha256,
                "metadata_path": str(existing.metadata_path),
            }

    source_sha256 = asset.sha256.upper()
    cache_dir = cache_root / source_sha256.lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / "source_h264_nvenc_lossless.mp4"
    temp_path = cache_dir / "source_h264_nvenc_lossless.tmp.mp4"
    metadata_path = cache_dir / "metadata.json"
    metadata_temp_path = cache_dir / "metadata.tmp.json"
    temp_path.unlink(missing_ok=True)
    metadata_temp_path.unlink(missing_ok=True)

    source_probe = _probe(ffprobe, source_path)
    source_stream = _first_stream(source_probe)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "lossless",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        transcode_wall_time_s = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg_transcode_failed:"
                f"{completed.returncode}:{completed.stderr[-4000:]}"
            )

        derived_probe = _probe(ffprobe, temp_path)
        derived_stream = _first_stream(derived_probe)
        duration_delta_s, duration_ok = _duration_within_one_frame(
            source_stream,
            derived_stream,
        )
        verification: dict[str, Any] = {
            "source_manifest_match": True,
            "source_sha256_verified": True,
            "width_match": int(source_stream["width"])
            == int(derived_stream["width"]),
            "height_match": int(source_stream["height"])
            == int(derived_stream["height"]),
            "r_frame_rate_match": source_stream["r_frame_rate"]
            == derived_stream["r_frame_rate"],
            "avg_frame_rate_match": source_stream["avg_frame_rate"]
            == derived_stream["avg_frame_rate"],
            "frame_count_match": int(source_stream["nb_frames"])
            == int(derived_stream["nb_frames"]),
            "duration_delta_s": duration_delta_s,
            "duration_within_one_frame": duration_ok,
            "derived_codec_h264": derived_stream["codec_name"] == "h264",
            "derived_pix_fmt_yuv420p": (
                derived_stream["pix_fmt"] == "yuv420p"
            ),
        }
        required_probe_checks = (
            "width_match",
            "height_match",
            "r_frame_rate_match",
            "avg_frame_rate_match",
            "frame_count_match",
            "duration_within_one_frame",
            "derived_codec_h264",
            "derived_pix_fmt_yuv420p",
        )
        if not all(verification[key] is True for key in required_probe_checks):
            raise RuntimeError(
                "derived_video_probe_parity_failed:"
                f"{json.dumps(verification, sort_keys=True)}"
            )

        source_framemd5, source_framemd5_bytes = (
            _decoded_framemd5_sha256(ffmpeg, source_path)
        )
        derived_framemd5, derived_framemd5_bytes = (
            _decoded_framemd5_sha256(ffmpeg, temp_path)
        )
        verification.update(
            {
                "source_framemd5_sha256": source_framemd5,
                "derived_framemd5_sha256": derived_framemd5,
                "source_framemd5_bytes": source_framemd5_bytes,
                "derived_framemd5_bytes": derived_framemd5_bytes,
                "decoded_framemd5_match": (
                    source_framemd5 == derived_framemd5
                ),
            }
        )
        if verification["decoded_framemd5_match"] is not True:
            raise RuntimeError("derived_video_decoded_framemd5_mismatch")

        os.replace(temp_path, final_path)
        nvdec = _verify_nvdec(
            final_path,
            expected_frames=int(source_stream["nb_frames"]),
        )
        verification.update(
            {
                "nvdec_frame_count_match": (
                    nvdec["frames_read"]
                    == int(source_stream["nb_frames"])
                ),
                "nvdec_effective_backend": (
                    nvdec["status"].get("effective_backend") == "nvdec"
                ),
                "nvdec_fallback_zero": (
                    int(nvdec["status"].get("fallback_count") or 0) == 0
                ),
                "nvdec_frames_read": nvdec["frames_read"],
                "nvdec_wall_time_s": nvdec["wall_time_s"],
                "nvdec_throughput_fps": nvdec["throughput_fps"],
                "nvdec_status": nvdec["status"],
            }
        )
        derived_sha256 = sha256_file(final_path)
        ffmpeg_version = _run([ffmpeg, "-version"]).stdout.splitlines()[0]
        ffmpeg_path = Path(ffmpeg).resolve(strict=True)
        builder_path = Path(__file__).resolve(strict=True)
        metadata = {
            "schema_version": 1,
            "artifact_type": "nvdec_derived_video",
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "source": {
                "asset_id": asset.asset_id,
                "path": str(source_path),
                "sha256": source_sha256,
                "size_bytes": source_path.stat().st_size,
                "role": asset.role,
                "label": asset.label,
                "attack_type": asset.attack_type,
                "codec": source_stream["codec_name"],
                "width": int(source_stream["width"]),
                "height": int(source_stream["height"]),
                "fps": source_stream["avg_frame_rate"],
                "frame_count": int(source_stream["nb_frames"]),
                "duration_s": float(source_stream["duration"]),
            },
            "derived": {
                "path": str(final_path.resolve()),
                "relative_path": final_path.name,
                "sha256": derived_sha256,
                "size_bytes": final_path.stat().st_size,
                "codec": derived_stream["codec_name"],
                "profile": derived_stream.get("profile"),
                "pixel_format": derived_stream["pix_fmt"],
                "width": int(derived_stream["width"]),
                "height": int(derived_stream["height"]),
                "fps": derived_stream["avg_frame_rate"],
                "frame_count": int(derived_stream["nb_frames"]),
                "duration_s": float(derived_stream["duration"]),
            },
            "profile": {
                "id": PROFILE_ID,
                "sha256": canonical_payload_sha256(PROFILE_PAYLOAD),
                "payload": dict(PROFILE_PAYLOAD),
            },
            "transcode": {
                "tool": "ffmpeg",
                "version": ffmpeg_version,
                "decode_backend": (
                    "ffmpeg_software_"
                    + str(source_stream["codec_name"]).lower()
                ),
                "encode_backend": "h264_nvenc",
                "preset": "p4",
                "tune": "lossless",
                "pixel_format": "yuv420p",
                "command": command,
                "wall_time_s": transcode_wall_time_s,
                "stderr_tail": completed.stderr[-4000:],
            },
            "toolchain": {
                "ffmpeg_path": str(ffmpeg_path),
                "ffmpeg_version": ffmpeg_version,
                "ffmpeg_sha256": sha256_file(ffmpeg_path),
                "builder_path": str(builder_path),
                "builder_sha256": sha256_file(builder_path),
            },
            "verification": verification,
        }
        metadata_temp_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(metadata_temp_path, metadata_path)
    finally:
        temp_path.unlink(missing_ok=True)
        metadata_temp_path.unlink(missing_ok=True)

    resolution = resolve_derived_video_source(
        source_path,
        cache_root=cache_root,
    )
    if resolution is None:
        raise RuntimeError("derived_video_resolution_missing_after_build")
    return {
        "ok": True,
        "reused": False,
        "asset_id": asset.asset_id,
        "source_path": str(resolution.source_path),
        "source_sha256": resolution.source_sha256,
        "decode_source": str(resolution.decode_path),
        "decode_source_sha256": resolution.derived_sha256,
        "metadata_path": str(resolution.metadata_path),
        "metadata": resolution.metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lossless H.264 NVENC derivative bound to an authoritative "
            "source SHA, verify decoded-frame parity, then verify full NVDEC."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=default_derived_video_cache_root(),
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.ffmpeg or not args.ffprobe:
        parser.error("ffmpeg and ffprobe must be available")

    try:
        payload = build_nvdec_derived_video(
            manifest_path=args.manifest,
            asset_id=args.asset_id,
            cache_root=args.cache_root.resolve(),
            ffmpeg=str(args.ffmpeg),
            ffprobe=str(args.ffprobe),
            force=bool(args.force),
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
