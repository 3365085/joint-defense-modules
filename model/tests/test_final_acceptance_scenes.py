from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from defense.diagnostics.final_acceptance_matrix import load_acceptance_manifest
from defense.diagnostics.final_acceptance_scenes import (
    SceneGenerationConfig,
    WrappedSceneConfig,
    generate_final_acceptance_scenes,
    generate_wrapped_final_acceptance_scenes,
)


def _write_video(path: Path, *, frames: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (96, 64))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            frame = np.full((64, 96, 3), 90 + idx % 12, dtype=np.uint8)
            cv2.rectangle(frame, (36 + idx % 3, 12), (62 + idx % 3, 52), (120, 150, 170), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_generate_final_acceptance_scenes_writes_manifest_and_videos(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    for stem in ("source_a", "source_b"):
        _write_video(source_root / f"{stem}.mp4")
    manifest_path = tmp_path / "matrix.json"
    negative_manifest = tmp_path / "negative.json"
    negative_manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_id": "n1_case",
                        "path": str(tmp_path / "negative.mp4"),
                        "category": "N1",
                        "label": "negative",
                        "attack_start_frame": None,
                        "attack_end_frame": None,
                        "source_id": "negative-source",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = generate_final_acceptance_scenes(
        SceneGenerationConfig(
            source_root=source_root,
            output_root=tmp_path / "out",
            manifest_path=manifest_path,
            negative_manifest_path=negative_manifest,
            source_stems=("source_a", "source_b"),
            max_frames=36,
            max_width=96,
            attack_start_frame=8,
            attack_ramp_frames=4,
            seed=123,
        )
    )

    assert manifest_path.is_file()
    assert len(report["clips"]) == 7
    positives = [row for row in report["clips"] if row["category"].startswith("P")]
    negatives = [row for row in report["clips"] if row["category"].startswith("N")]
    assert {row["category"] for row in positives} == {"P1", "P2", "P3"}
    assert all(row["label"] == "positive" for row in positives)
    assert all(row["attack_start_frame"] == 8 for row in positives)
    assert all(row["attack_end_frame"] == 35 for row in positives)
    assert all(Path(row["path"]).is_file() for row in positives)
    assert negatives == [
        {
            "clip_id": "n1_case",
            "path": str(tmp_path / "negative.mp4"),
            "category": "N1",
            "label": "negative",
            "attack_start_frame": None,
            "attack_end_frame": None,
            "source_id": "negative-source",
        }
    ]

    loaded = load_acceptance_manifest(manifest_path)
    assert len(loaded) == 7
    assert sum(1 for clip in loaded if clip.is_positive) == 6
    assert all(clip.attack_start_frame == 8 for clip in loaded if clip.is_positive)

    disk_report = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert disk_report["generation"] == "final_acceptance_scenes_v2"
    assert disk_report["design_contract"]["P1"].startswith("target-related")


def test_generate_wrapped_final_acceptance_scenes_adds_prefix_and_tail(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean"
    attack_root = tmp_path / "attack"
    attack_clips: dict[str, tuple[Path, ...]] = {}
    for category, subdir in (("P1", "adv_patch"), ("P2", "glare"), ("P3", "occlusion")):
        clips = []
        for idx in range(4):
            clean = clean_root / f"{category}_{idx}.mp4"
            attack = attack_root / subdir / f"{category}_{idx}__{subdir}.mp4"
            _write_video(clean, frames=96)
            _write_video(attack, frames=72)
            attack.with_suffix(".meta.json").write_text(
                json.dumps(
                    {
                        "source": str(clean),
                        "start_frame": 0,
                        "attack_delay": 3,
                        "attack_ramp": 2,
                    }
                ),
                encoding="utf-8",
            )
            clips.append(attack)
        attack_clips[category] = tuple(clips)
    manifest_path = tmp_path / "wrapped.json"

    report = generate_wrapped_final_acceptance_scenes(
        WrappedSceneConfig(
            output_root=tmp_path / "wrapped",
            manifest_path=manifest_path,
            negative_manifest_path=None,
            attack_clips=attack_clips,
            clean_context_video=clean_root / "P1_0.mp4",
            clean_prefix_s=2.0,
            attack_duration_s=2.0,
            clean_tail_s=2.0,
            max_width=96,
        )
    )

    assert manifest_path.is_file()
    assert len(report["clips"]) == 12
    assert {row["category"] for row in report["clips"]} == {"P1", "P2", "P3"}
    assert all(row["attack_start_frame"] == 24 for row in report["clips"])
    assert all(row["attack_end_frame"] == 47 for row in report["clips"])
    assert all(Path(row["path"]).is_file() for row in report["clips"])
    metas = [Path(row["path"]).with_suffix(".meta.json") for row in report["clips"]]
    assert all(json.loads(path.read_text(encoding="utf-8"))["clean_tail_frames"] == 24 for path in metas)
