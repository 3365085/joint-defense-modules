from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from defense.diagnostics.a4_dataset import (
    ADV_PATCH_TRAJECTORY_MODES,
    REQUIRED_ATTACK_TYPES,
    A4VariantSpec,
    A4DatasetValidationError,
    CenterAnchorProvider,
    _position_capture_for_bounded_window,
    _render_attack_frame,
    _trajectory_parameters,
    _trajectory_state,
    build_a4_training_dataset,
    load_a4_dataset_manifest,
    load_clean_sources,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_MANIFEST = (
    ROOT / "configs/acceptance/module_a_authoritative_manifest_v1.json"
)


class _FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        *,
        fail_first_read_after_seek: bool = False,
    ) -> None:
        self.frames = frames
        self.position = 0
        self.fail_first_read_after_seek = fail_first_read_after_seek
        self.seek_attempted = False
        self.seek_failure_consumed = False
        self.released = False
        self.read_indices: list[int] = []

    def isOpened(self) -> bool:
        return not self.released

    def set(self, prop: int, value: float) -> bool:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.seek_attempted = True
        return True

    def get(self, prop: int) -> float:
        assert prop == cv2.CAP_PROP_POS_FRAMES
        return float(self.position)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if (
            self.seek_attempted
            and self.fail_first_read_after_seek
            and not self.seek_failure_consumed
        ):
            self.seek_failure_consumed = True
            return False, None
        if self.position >= len(self.frames):
            return False, None
        index = self.position
        self.position += 1
        self.read_indices.append(index)
        return True, self.frames[index].copy()

    def release(self) -> None:
        self.released = True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(
    path: Path,
    *,
    frames: int = 20,
    fps: float = 10.0,
    color_offset: int = 0,
) -> None:
    import numpy as np

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (96, 64),
    )
    assert writer.isOpened()
    try:
        for frame_idx in range(frames):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            frame[..., 0] = 30 + frame_idx * 3 + color_offset
            frame[..., 1] = 70 + color_offset
            frame[..., 2] = 110 + color_offset
            cv2.rectangle(
                frame,
                (20 + frame_idx % 8, 10),
                (62 + frame_idx % 8, 56),
                (80, 180, 220),
                thickness=-1,
            )
            writer.write(frame)
    finally:
        writer.release()


def _write_clean_manifest(path: Path, video: Path, *, split: str = "heldout") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip_id",
                "path",
                "scene_id",
                "split",
                "base_source_sha256",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "fixture_base",
                "path": str(video.resolve()),
                "scene_id": "fixture_scene",
                "split": split,
                "base_source_sha256": _sha256(video),
            }
        )


def _write_multi_clean_manifest(
    path: Path,
    videos: list[Path],
    *,
    split: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip_id",
                "path",
                "scene_id",
                "split",
                "base_source_sha256",
            ),
        )
        writer.writeheader()
        for index, video in enumerate(videos, start=1):
            writer.writerow(
                {
                    "clip_id": f"fixture_base_{index}",
                    "path": str(video.resolve()),
                    "scene_id": f"fixture_scene_{index}",
                    "split": split,
                    "base_source_sha256": _sha256(video),
                }
            )


def test_clean_source_validation_rejects_rebuilt_demo_path(tmp_path: Path) -> None:
    forbidden = tmp_path / "rebuilt_demo"
    forbidden.mkdir()
    video = forbidden / "source.mp4"
    video.write_bytes(b"fixture")
    manifest = tmp_path / "clean.csv"
    _write_clean_manifest(manifest, video)

    with pytest.raises(A4DatasetValidationError, match="rebuilt_demo_path_forbidden"):
        load_clean_sources(
            manifest,
            authoritative_video_hashes=(),
            verify_source_hashes=False,
        )


def test_clean_source_validation_rejects_authoritative_hash_overlap(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"authoritative-overlap-fixture")
    manifest = tmp_path / "clean.csv"
    _write_clean_manifest(manifest, video)

    with pytest.raises(A4DatasetValidationError, match="authoritative_video_overlap"):
        load_clean_sources(
            manifest,
            authoritative_video_hashes=(_sha256(video),),
            verify_source_hashes=True,
        )


def test_adv_patch_trajectory_grid_is_seeded_and_not_per_frame_untracked() -> None:
    seed = 123456
    first = _trajectory_parameters("discrete_jump/jitter", seed=seed)
    second = _trajectory_parameters("discrete_jump/jitter", seed=seed)
    assert first == second
    assert first["jump_interval_frames"] in {4, 6, 8, 12, 16}

    anchor = (20.0, 10.0, 70.0, 58.0)
    state_a = _trajectory_state(
        "discrete_jump/jitter",
        frame_idx=9,
        anchor=anchor,
        seed=seed,
    )
    state_b = _trajectory_state(
        "discrete_jump/jitter",
        frame_idx=9,
        anchor=anchor,
        seed=seed,
    )
    assert state_a == state_b


@pytest.mark.parametrize(
    ("anchor", "expected_bound"),
    [
        ((498.0, 498.0, 502.0, 502.0), "min"),
        ((100.0, 100.0, 900.0, 900.0), "max"),
    ],
)
def test_adv_patch_render_clamps_final_frame_area_ratio(
    anchor: tuple[float, float, float, float],
    expected_bound: str,
) -> None:
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    rendered = _render_attack_frame(
        frame,
        spec=A4VariantSpec(
            attack_type="adv_patch",
            trajectory_mode="target_anchored_static",
            label=1,
            suffix="adv_patch",
        ),
        anchor=anchor,
        frame_idx=20,
        seed=42,
        texture=np.full((96, 96, 3), 255, dtype=np.uint8),
        attack_start_frame=0,
        attack_ramp_frames=0,
    )
    changed_ratio = float(np.mean(np.any(rendered != frame, axis=2)))
    if expected_bound == "min":
        assert 0.0018 <= changed_ratio <= 0.0022
    else:
        assert 0.0195 <= changed_ratio <= 0.0205


def test_failed_random_seek_reopens_and_sequentially_decodes_exact_offset() -> None:
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(8)]
    random_seek_capture = _FakeCapture(
        frames,
        fail_first_read_after_seek=True,
    )
    reopened: list[_FakeCapture] = []

    def capture_factory(path: str) -> _FakeCapture:
        assert path.endswith("fixture.mp4")
        capture = _FakeCapture(frames)
        reopened.append(capture)
        return capture

    capture, first_frame, positioning = _position_capture_for_bounded_window(
        Path("fixture.mp4"),
        random_seek_capture,
        source_start_frame=3,
        capture_factory=capture_factory,
    )

    assert random_seek_capture.released is True
    assert positioning["source_positioning_mode"] == "sequential_decode_fallback"
    assert positioning["source_seek_set_succeeded"] is True
    assert positioning["source_seek_fallback_reason"] == "seek_first_frame_decode_failed"
    assert positioning["source_sequential_discarded_frames"] == 3
    assert len(reopened) == 1
    assert capture is reopened[0]
    assert reopened[0].read_indices == [0, 1, 2, 3]
    assert int(first_frame[0, 0, 0]) == 3

    ok, next_frame = capture.read()
    assert ok is True
    assert next_frame is not None
    assert int(next_frame[0, 0, 0]) == 4
    assert reopened[0].read_indices == [0, 1, 2, 3, 4]
    capture.release()


def test_verified_random_seek_prefetches_target_once_without_fallback() -> None:
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(7)]
    capture = _FakeCapture(frames)

    def unexpected_factory(_path: str) -> _FakeCapture:
        raise AssertionError("verified random seek must not reopen the source")

    positioned, first_frame, positioning = _position_capture_for_bounded_window(
        Path("fixture.mp4"),
        capture,
        source_start_frame=2,
        capture_factory=unexpected_factory,
    )

    assert positioned is capture
    assert positioning["source_positioning_mode"] == "random_seek_verified"
    assert int(first_frame[0, 0, 0]) == 2
    ok, next_frame = positioned.read()
    assert ok is True
    assert next_frame is not None
    assert int(next_frame[0, 0, 0]) == 3
    assert capture.read_indices == [2, 3]
    capture.release()


def test_builder_creates_bounded_deterministic_grouped_variants(tmp_path: Path) -> None:
    sources = [tmp_path / f"source_{index}.mp4" for index in range(5)]
    for index, source in enumerate(sources):
        _write_video(source, color_offset=index * 5)
    clean_manifest = tmp_path / "clean.csv"
    _write_multi_clean_manifest(clean_manifest, sources, split="heldout")

    manifests: list[Path] = []
    loaded_runs: list[list[dict]] = []
    for run in (1, 2):
        manifest = tmp_path / f"run_{run}" / "dataset.csv"
        manifests.append(manifest)
        build_a4_training_dataset(
            source_manifest_path=clean_manifest,
            output_dir=tmp_path / f"run_{run}" / "clips",
            output_manifest_path=manifest,
            authoritative_manifest_path=AUTHORITATIVE_MANIFEST,
            max_frames_per_video=8,
            clip_duration_s=1.0,
            generator_seed=77,
            attack_start_frame=2,
            attack_ramp_frames=2,
            _anchor_provider=CenterAnchorProvider(),
        )
        rows, metadata = load_a4_dataset_manifest(manifest)
        assert metadata["base_count"] == 5
        assert metadata["clip_count"] == 30
        assert metadata["variants_per_base"] == 6
        assert metadata["authoritative_video_count"] == 36
        loaded_runs.append(rows)

    first, second = loaded_runs
    assert len(first) == len(second) == 30
    assert {row["split"] for row in first} == {"heldout"}
    assert len({row["scene_id"] for row in first}) == 5
    assert len({row["base_clip_id"] for row in first}) == 5
    assert {row["attack_type"] for row in first} == {"clean", *REQUIRED_ATTACK_TYPES}
    assert {
        row["trajectory_mode"]
        for row in first
        if row["attack_type"] == "adv_patch"
    } == set(ADV_PATCH_TRAJECTORY_MODES)
    assert {int(row["frames"]) for row in first} == {8}
    for base_clip_id in {row["base_clip_id"] for row in first}:
        base_rows = [row for row in first if row["base_clip_id"] == base_clip_id]
        assert len({row["source_start_frame"] for row in base_rows}) == 1
        assert len({row["source_end_frame_exclusive"] for row in base_rows}) == 1
        assert len({row["source_positioning_mode"] for row in base_rows}) == 1
        assert len({row["source_seek_fallback_reason"] for row in base_rows}) == 1
        assert int(base_rows[0]["source_end_frame_exclusive"]) - int(
            base_rows[0]["source_start_frame"]
        ) == 8

    stable_fields = (
        "clip_id",
        "content_sha256",
        "provenance_id",
        "variant_seed",
        "source_start_frame",
        "source_end_frame_exclusive",
        "source_positioning_mode",
        "source_seek_fallback_reason",
    )
    assert [tuple(row[field] for field in stable_fields) for row in first] == [
        tuple(row[field] for field in stable_fields) for row in second
    ]
    jump = next(
        row for row in first if row["trajectory_mode"] == "discrete_jump/jitter"
    )
    provenance = json.loads(jump["provenance_json"])
    assert provenance["trajectory_algorithm"].startswith("piecewise_seeded")
    assert provenance["source_start_frame"] == int(jump["source_start_frame"])
    assert provenance["source_end_frame_exclusive"] == int(
        jump["source_end_frame_exclusive"]
    )
    assert provenance["source_positioning"]["source_positioning_mode"] == jump[
        "source_positioning_mode"
    ]
    assert provenance["trajectory_parameters"]["jump_interval_frames"] in {
        4,
        6,
        8,
        12,
        16,
    }
