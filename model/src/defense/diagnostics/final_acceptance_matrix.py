from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATEGORY_ORDER = ("P1", "P2", "P3", "N1", "N2", "N3", "N4")
DEFAULT_MAX_POST_ATTACK_LINGER_FRAMES = 20
DEFAULT_MAX_FIRST_CONFIRMATION_DELAY_S = 2.0
REQUIRED_MANIFEST_FIELDS = (
    "clip_id",
    "path",
    "category",
    "label",
    "attack_start_frame",
    "attack_end_frame",
    "source_id",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "diagnostics" / "final_acceptance_matrix_report.json"


class ManifestValidationError(ValueError):
    """Raised when an acceptance manifest does not satisfy the explicit schema."""


@dataclass(frozen=True, slots=True)
class AcceptanceClip:
    clip_id: str
    path: Path
    category: str
    label: str
    is_positive: bool
    attack_start_frame: int | None
    attack_end_frame: int | None
    source_id: str

    def report_fields(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "path": str(self.path),
            "category": self.category,
            "label": self.label,
            "is_positive": self.is_positive,
            "attack_start_frame": self.attack_start_frame,
            "attack_end_frame": self.attack_end_frame,
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class AcceptanceFrame:
    frame: Any
    source_frame_idx: int
    source_time_s: float


FrameSourceFactory = Callable[[AcceptanceClip], Iterable[Any]]
ConfirmationSelector = Callable[[Any], bool]


def load_acceptance_manifest(manifest_path: str | Path) -> list[AcceptanceClip]:
    """Load and validate a CSV or JSON final-acceptance manifest.

    Frame annotations are zero-based and inclusive. Positive categories require
    both attack bounds. Negative categories require those fields to be empty.
    Relative media paths are resolved against the manifest directory.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"acceptance manifest does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv_rows(path)
    elif suffix == ".json":
        rows = _load_json_rows(path)
    else:
        raise ManifestValidationError(
            f"acceptance manifest must be CSV or JSON, got: {path.suffix or '<no suffix>'}"
        )
    if not rows:
        raise ManifestValidationError(f"acceptance manifest has no clips: {path}")

    clips: list[AcceptanceClip] = []
    seen_clip_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        clip = _parse_manifest_row(row, manifest_dir=path.parent, row_number=row_number)
        if clip.clip_id in seen_clip_ids:
            raise ManifestValidationError(
                f"manifest row {row_number}: duplicate clip_id {clip.clip_id!r}"
            )
        seen_clip_ids.add(clip.clip_id)
        clips.append(clip)
    return clips


def iter_video_frames(clip: AcceptanceClip) -> Iterable[Any]:
    """Decode every frame in a manifest clip through OpenCV."""

    import cv2

    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for clip {clip.clip_id}: {clip.path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not (fps > 0.0):
        fps = 1.0
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            source_frame_idx = decoded
            decoded += 1
            yield AcceptanceFrame(
                frame=frame,
                source_frame_idx=source_frame_idx,
                source_time_s=float(source_frame_idx) / fps,
            )
    finally:
        capture.release()
    if decoded == 0:
        raise RuntimeError(f"video decoded zero frames for clip {clip.clip_id}: {clip.path}")


def default_confirmation_selector(pipeline_output: Any) -> bool:
    """Extract the confirmed-alert signal from production or fake outputs."""

    value = pipeline_output
    if isinstance(value, (tuple, list)):
        if not value:
            raise KeyError("pipeline output sequence is empty")
        value = value[-1]
    if isinstance(value, Mapping) and "info" in value and isinstance(value["info"], Mapping):
        value = value["info"]
    if isinstance(value, Mapping):
        if "alert_confirmed" not in value:
            raise KeyError("pipeline output is missing 'alert_confirmed'")
        return bool(value["alert_confirmed"])
    if hasattr(value, "alert_confirmed"):
        return bool(getattr(value, "alert_confirmed"))
    if isinstance(value, bool):
        return value
    raise TypeError(
        "pipeline output must expose alert_confirmed in a mapping, object, or final tuple item"
    )


def evaluate_acceptance_clip(
    clip: AcceptanceClip,
    *,
    pipeline: Any,
    frame_source_factory: FrameSourceFactory = iter_video_frames,
    confirmation_selector: ConfirmationSelector = default_confirmation_selector,
) -> dict[str, Any]:
    """Run one full clip and calculate event-level acceptance metrics."""

    reset = getattr(pipeline, "reset", None)
    if callable(reset):
        reset()
    start_clip = getattr(pipeline, "start_clip", None)
    if callable(start_clip):
        start_clip(clip)

    attack_start = (
        _required_attack_bound(clip.attack_start_frame, clip, "start")
        if clip.is_positive
        else None
    )
    attack_end = (
        _required_attack_bound(clip.attack_end_frame, clip, "end")
        if clip.is_positive
        else None
    )
    frame_count = 0
    confirmed_frame_count = 0
    pre_attack_confirmed_frames = 0
    attack_window_confirmed_frames = 0
    post_attack_confirmed_frames = 0
    post_attack_observed_frames = 0
    post_attack_linger_frames = 0
    first_hit_frame: int | None = None
    first_hit_time_s: float | None = None
    attack_start_time_s: float | None = None
    linger_open = True
    previous_frame: AcceptanceFrame | None = None
    last_source_frame_idx: int | None = None
    temporal_input_frames = 0
    temporal_previous_applied_frames = 0
    temporal_strict_predecessor_frames = 0
    temporal_gap_violation_frames = 0

    frames = iter(frame_source_factory(clip))
    try:
        for fallback_frame_idx, frame in enumerate(frames):
            current_frame = _coerce_acceptance_frame(
                frame,
                fallback_frame_idx=fallback_frame_idx,
            )
            frame_idx = current_frame.source_frame_idx
            if last_source_frame_idx is not None and frame_idx <= last_source_frame_idx:
                raise RuntimeError(
                    f"clip {clip.clip_id} source frame indices must increase: "
                    f"{last_source_frame_idx} -> {frame_idx}"
                )
            output = _process_pipeline_frame(
                pipeline,
                current_frame,
                previous_frame=previous_frame,
            )
            previous_frame = current_frame
            last_source_frame_idx = frame_idx
            temporal_input = _pipeline_temporal_input(output)
            if temporal_input is not None:
                temporal_input_frames += 1
                if temporal_input.get("previous_frame_applied", False):
                    temporal_previous_applied_frames += 1
                if temporal_input.get("strict_source_predecessor", False):
                    temporal_strict_predecessor_frames += 1
                gap_frames = temporal_input.get("source_gap_frames")
                if gap_frames is not None and int(gap_frames) != 1:
                    temporal_gap_violation_frames += 1
            confirmed = bool(confirmation_selector(output))
            frame_count += 1
            if confirmed:
                confirmed_frame_count += 1

            if not clip.is_positive:
                continue

            assert attack_start is not None and attack_end is not None
            if frame_idx < attack_start:
                if confirmed:
                    pre_attack_confirmed_frames += 1
            elif frame_idx <= attack_end:
                if attack_start_time_s is None:
                    attack_start_time_s = current_frame.source_time_s
                if confirmed:
                    attack_window_confirmed_frames += 1
                    if first_hit_frame is None:
                        first_hit_frame = frame_idx
                        first_hit_time_s = current_frame.source_time_s
            else:
                post_attack_observed_frames += 1
                if confirmed:
                    post_attack_confirmed_frames += 1
                if linger_open:
                    if confirmed:
                        post_attack_linger_frames += 1
                    else:
                        linger_open = False
    finally:
        close_frames = getattr(frames, "close", None)
        if callable(close_frames):
            close_frames()
        finish_clip = getattr(pipeline, "finish_clip", None)
        if callable(finish_clip):
            finish_clip(clip)

    if frame_count <= 0:
        raise RuntimeError(f"clip {clip.clip_id} produced no frames")

    result = clip.report_fields()
    result.update(
        {
            "status": "completed",
            "error": None,
            "frame_count": frame_count,
            "confirmed_frame_count": confirmed_frame_count,
            "temporal_input_frames": temporal_input_frames,
            "temporal_previous_applied_frames": temporal_previous_applied_frames,
            "temporal_strict_predecessor_frames": temporal_strict_predecessor_frames,
            "temporal_gap_violation_frames": temporal_gap_violation_frames,
            "temporal_strict_predecessor_complete": (
                temporal_input_frames == frame_count
                and temporal_strict_predecessor_frames == max(0, frame_count - 1)
                and temporal_previous_applied_frames == max(0, frame_count - 1)
                and temporal_gap_violation_frames == 0
            )
            if temporal_input_frames > 0
            else None,
        }
    )
    if not clip.is_positive:
        result.update(
            {
                "pre_attack_false_positive": None,
                "pre_attack_confirmed_frames": None,
                "hit_after_onset": None,
                "first_hit_frame": None,
                "first_delay_frames": None,
                "first_hit_time_s": None,
                "first_delay_s": None,
                "attack_window_confirmed_frames": None,
                "post_attack_confirmed_frames": None,
                "post_attack_observed_frames": None,
                "post_attack_linger_frames": None,
                "positive_missed": None,
                "negative_false_positive": confirmed_frame_count > 0,
                "event_outcome": (
                    "negative_false_positive" if confirmed_frame_count > 0 else "negative_clean"
                ),
            }
        )
        return result

    assert attack_start is not None and attack_end is not None
    if last_source_frame_idx is None or last_source_frame_idx < attack_end:
        raise RuntimeError(
            f"clip {clip.clip_id} ended at frame {last_source_frame_idx}, "
            f"before attack_end_frame {attack_end}"
        )

    hit_after_onset = first_hit_frame is not None
    result.update(
        {
            "pre_attack_false_positive": pre_attack_confirmed_frames > 0,
            "pre_attack_confirmed_frames": pre_attack_confirmed_frames,
            "hit_after_onset": hit_after_onset,
            "first_hit_frame": first_hit_frame,
            "first_delay_frames": (
                first_hit_frame - attack_start if first_hit_frame is not None else None
            ),
            "first_hit_time_s": first_hit_time_s,
            "first_delay_s": (
                first_hit_time_s - attack_start_time_s
                if first_hit_time_s is not None and attack_start_time_s is not None
                else None
            ),
            "attack_window_confirmed_frames": attack_window_confirmed_frames,
            "post_attack_confirmed_frames": post_attack_confirmed_frames,
            "post_attack_observed_frames": post_attack_observed_frames,
            "post_attack_linger_frames": post_attack_linger_frames,
            "positive_missed": not hit_after_onset,
            "negative_false_positive": None,
            "event_outcome": "positive_hit" if hit_after_onset else "positive_miss",
        }
    )
    return result


def run_acceptance_matrix(
    manifest_path: str | Path,
    *,
    pipeline: Any | None = None,
    pipeline_factory: Callable[[], Any] | None = None,
    frame_source_factory: FrameSourceFactory = iter_video_frames,
    confirmation_selector: ConfirmationSelector = default_confirmation_selector,
    output_path: str | Path | None = None,
    close_pipeline: bool | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all manifest clips with an injected pipeline or pipeline factory."""

    if (pipeline is None) == (pipeline_factory is None):
        raise ValueError("provide exactly one of pipeline or pipeline_factory")

    clips = load_acceptance_manifest(manifest_path)
    owned_pipeline = pipeline is None
    active_pipeline = pipeline_factory() if pipeline_factory is not None else pipeline
    if active_pipeline is None:
        raise RuntimeError("pipeline_factory returned None")

    results: list[dict[str, Any]] = []
    try:
        for clip in clips:
            try:
                result = evaluate_acceptance_clip(
                    clip,
                    pipeline=active_pipeline,
                    frame_source_factory=frame_source_factory,
                    confirmation_selector=confirmation_selector,
                )
            except Exception as exc:
                result = _error_result(clip, exc)
            results.append(result)
    finally:
        should_close = owned_pipeline if close_pipeline is None else bool(close_pipeline)
        close = getattr(active_pipeline, "close", None)
        if should_close and callable(close):
            close()

    manifest = Path(manifest_path).expanduser().resolve()
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest),
        "frame_indexing": "zero_based_inclusive_attack_bounds",
        "confirmation_signal": "alert_confirmed",
        "runtime": dict(runtime_metadata or {}),
        "clips": results,
        "summary": summarize_acceptance_results(results),
    }
    if output_path is not None:
        write_acceptance_report(output_path, report)
    return report


def run_runtime_acceptance_matrix(
    manifest_path: str | Path,
    *,
    output_path: str | Path = DEFAULT_REPORT_PATH,
    config_path: str | Path | None = None,
    profile: str = "desktop_rtx",
) -> dict[str, Any]:
    """Build the configured production pipeline and run the acceptance matrix."""

    from defense.runtime.config import DEFAULT_CONFIG_PATH
    from defense.runtime.pipeline_factory import PipelineCache

    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else DEFAULT_CONFIG_PATH.resolve()
    )
    cache = PipelineCache(config_path=resolved_config_path)
    try:
        bundle = cache.get(profile=str(profile or "default"))
        runtime_metadata = {
            "config_path": str(resolved_config_path),
            "profile": str(profile or "default"),
            "backend": bundle.backend,
            "model_family": bundle.model_family,
            "artifact_path": bundle.artifact_path,
            "warmup_error": bundle.warmup_error or None,
        }
        report = run_acceptance_matrix(
            manifest_path,
            pipeline=bundle.pipeline,
            output_path=None,
            close_pipeline=False,
            runtime_metadata=runtime_metadata,
        )
    finally:
        cache.clear()
    write_acceptance_report(output_path, report)
    return report


def summarize_acceptance_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate clip-level event outcomes overall and by matrix category."""

    rows = [dict(result) for result in results]
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    positives = [row for row in completed if row.get("is_positive") is True]
    negatives = [row for row in completed if row.get("is_positive") is False]

    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORY_ORDER:
        category_rows = [row for row in rows if row.get("category") == category]
        category_completed = [
            row for row in category_rows if row.get("status") == "completed"
        ]
        category_positive_misses = sum(
            1 for row in category_completed if row.get("positive_missed") is True
        )
        category_negative_false_positives = sum(
            1 for row in category_completed if row.get("negative_false_positive") is True
        )
        by_category[category] = {
            "clips": len(category_rows),
            "completed_clips": len(category_completed),
            "failed_clips": len(category_rows) - len(category_completed),
            "positive_missed_clips": category_positive_misses,
            "negative_false_positive_clips": category_negative_false_positives,
        }

    positive_missed_clips = sum(
        1 for row in positives if row.get("positive_missed") is True
    )
    negative_false_positive_clips = sum(
        1 for row in negatives if row.get("negative_false_positive") is True
    )
    return {
        "clips": len(rows),
        "completed_clips": len(completed),
        "failed_clips": len(failed),
        "positive_clips": sum(1 for row in rows if row.get("is_positive") is True),
        "positive_completed_clips": len(positives),
        "positive_missed_clips": positive_missed_clips,
        "positive_miss_rate": _rate(positive_missed_clips, len(positives)),
        "negative_clips": sum(1 for row in rows if row.get("is_positive") is False),
        "negative_completed_clips": len(negatives),
        "negative_false_positive_clips": negative_false_positive_clips,
        "negative_false_positive_rate": _rate(
            negative_false_positive_clips, len(negatives)
        ),
        "positive_pre_attack_false_positive_clips": sum(
            1 for row in positives if row.get("pre_attack_false_positive") is True
        ),
        "positive_post_attack_linger_clips": sum(
            1
            for row in positives
            if int(row.get("post_attack_linger_frames") or 0) > 0
        ),
        "by_category": by_category,
    }


def write_acceptance_report(
    output_path: str | Path, report: Mapping[str, Any]
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def report_exit_code(report: Mapping[str, Any]) -> int:
    """Return nonzero when the acceptance matrix fails any gate."""

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return 2
    gate_fields = (
        "failed_clips",
        "positive_missed_clips",
        "negative_false_positive_clips",
        "positive_pre_attack_false_positive_clips",
    )
    if any(int(summary.get(field) or 0) > 0 for field in gate_fields):
        return 1
    runtime = report.get("runtime")
    runtime_options = runtime if isinstance(runtime, Mapping) else {}
    max_post_attack_linger_frames = int(
        runtime_options.get(
            "max_post_attack_linger_frames",
            DEFAULT_MAX_POST_ATTACK_LINGER_FRAMES,
        )
    )
    max_first_confirmation_delay_s = float(
        runtime_options.get(
            "max_first_confirmation_delay_s",
            DEFAULT_MAX_FIRST_CONFIRMATION_DELAY_S,
        )
    )
    clips = report.get("clips")
    if isinstance(clips, Iterable):
        for row in clips:
            if not isinstance(row, Mapping):
                continue
            temporal_complete = row.get("temporal_strict_predecessor_complete")
            if temporal_complete is False:
                return 1
            if row.get("is_positive") is True:
                first_delay_s = row.get("first_delay_s")
                if first_delay_s is not None:
                    try:
                        if float(first_delay_s) > max_first_confirmation_delay_s:
                            return 1
                    except (TypeError, ValueError):
                        return 1
                post_attack_linger_frames = int(
                    row.get("post_attack_linger_frames") or 0
                )
                if post_attack_linger_frames > max_post_attack_linger_frames:
                    return 1
                post_attack_confirmed_frames = int(
                    row.get("post_attack_confirmed_frames") or 0
                )
                if post_attack_confirmed_frames - post_attack_linger_frames > 0:
                    return 1
    return 0


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in fieldnames]
        if missing:
            raise ManifestValidationError(
                f"CSV manifest is missing required columns: {', '.join(missing)}"
            )
        return [dict(row) for row in reader]


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, Mapping):
        data = data.get("clips")
    if not isinstance(data, list):
        raise ManifestValidationError(
            "JSON manifest root must be a list or an object containing a 'clips' list"
        )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, Mapping):
            raise ManifestValidationError(
                f"JSON manifest row {index} must be an object"
            )
        rows.append(dict(row))
    return rows


def _parse_manifest_row(
    row: Mapping[str, Any], *, manifest_dir: Path, row_number: int
) -> AcceptanceClip:
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
    if missing:
        raise ManifestValidationError(
            f"manifest row {row_number}: missing required fields: {', '.join(missing)}"
        )

    clip_id = _required_text(row["clip_id"], row_number, "clip_id")
    path_text = _required_text(row["path"], row_number, "path")
    category = _required_text(row["category"], row_number, "category").upper()
    source_id = _required_text(row["source_id"], row_number, "source_id")
    if category not in CATEGORY_ORDER:
        raise ManifestValidationError(
            f"manifest row {row_number}: category must be one of "
            f"{', '.join(CATEGORY_ORDER)}, got {category!r}"
        )

    label_value = row["label"]
    is_positive = _parse_label(label_value, row_number)
    expected_positive = category.startswith("P")
    if is_positive != expected_positive:
        raise ManifestValidationError(
            f"manifest row {row_number}: label {label_value!r} conflicts with "
            f"category {category}"
        )

    attack_start = _optional_nonnegative_int(
        row["attack_start_frame"], row_number, "attack_start_frame"
    )
    attack_end = _optional_nonnegative_int(
        row["attack_end_frame"], row_number, "attack_end_frame"
    )
    if is_positive:
        if attack_start is None or attack_end is None:
            raise ManifestValidationError(
                f"manifest row {row_number}: positive clips require attack frame bounds"
            )
        if attack_end < attack_start:
            raise ManifestValidationError(
                f"manifest row {row_number}: attack_end_frame must be >= "
                "attack_start_frame"
            )
    elif attack_start is not None or attack_end is not None:
        raise ManifestValidationError(
            f"manifest row {row_number}: negative clips must leave attack frame bounds empty"
        )

    media_path = Path(path_text).expanduser()
    if not media_path.is_absolute():
        media_path = manifest_dir / media_path
    return AcceptanceClip(
        clip_id=clip_id,
        path=media_path.resolve(),
        category=category,
        label=str(label_value).strip(),
        is_positive=is_positive,
        attack_start_frame=attack_start,
        attack_end_frame=attack_end,
        source_id=source_id,
    )


def _required_text(value: Any, row_number: int, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestValidationError(
            f"manifest row {row_number}: {field} must not be empty"
        )
    return text


def _parse_label(value: Any, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    token = str(value or "").strip().lower()
    if token in {"1", "true", "positive", "attack", "adversarial", "p"}:
        return True
    if token in {"0", "false", "negative", "normal", "clean", "benign", "n"}:
        return False
    raise ManifestValidationError(
        f"manifest row {row_number}: unsupported label {value!r}; "
        "use positive/negative, attack/normal, true/false, or 1/0"
    )


def _optional_nonnegative_int(
    value: Any, row_number: int, field: str
) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ManifestValidationError(
            f"manifest row {row_number}: {field} must be a non-negative integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"manifest row {row_number}: {field} must be a non-negative integer"
        ) from exc
    if str(value).strip() not in {str(parsed), f"+{parsed}"}:
        raise ManifestValidationError(
            f"manifest row {row_number}: {field} must be an integer, got {value!r}"
        )
    if parsed < 0:
        raise ManifestValidationError(
            f"manifest row {row_number}: {field} must be non-negative"
        )
    return parsed


def _required_attack_bound(
    value: int | None, clip: AcceptanceClip, bound_name: str
) -> int:
    if value is None:
        raise ManifestValidationError(
            f"positive clip {clip.clip_id} is missing attack {bound_name} frame"
        )
    return int(value)


def _coerce_acceptance_frame(
    frame: Any,
    *,
    fallback_frame_idx: int,
) -> AcceptanceFrame:
    if isinstance(frame, AcceptanceFrame):
        return frame
    return AcceptanceFrame(
        frame=frame,
        source_frame_idx=int(fallback_frame_idx),
        source_time_s=float(fallback_frame_idx),
    )


def _pipeline_temporal_input(output: Any) -> Mapping[str, Any] | None:
    value = output
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[-1]
    if isinstance(value, Mapping) and "info" in value and isinstance(value["info"], Mapping):
        value = value["info"]
    if not isinstance(value, Mapping):
        return None
    temporal_input = value.get("temporal_input")
    return temporal_input if isinstance(temporal_input, Mapping) else None


def _process_pipeline_frame(
    pipeline: Any,
    frame: AcceptanceFrame,
    *,
    previous_frame: AcceptanceFrame | None,
) -> Any:
    process_runtime_frame = getattr(pipeline, "process_runtime_frame", None)
    if callable(process_runtime_frame):
        return process_runtime_frame(
            frame.frame,
            timestamp=frame.source_time_s,
            previous_frame=(
                previous_frame.frame if previous_frame is not None else None
            ),
            current_source_frame_idx=frame.source_frame_idx,
            previous_source_frame_idx=(
                previous_frame.source_frame_idx
                if previous_frame is not None
                else None
            ),
            previous_source_time_s=(
                previous_frame.source_time_s
                if previous_frame is not None
                else None
            ),
        )
    process_frame = getattr(pipeline, "process_frame", None)
    if callable(process_frame):
        return process_frame(frame.frame)
    if callable(pipeline):
        return pipeline(frame.frame)
    raise TypeError(
        "pipeline must be callable or expose process_runtime_frame(...) "
        "or process_frame(frame)"
    )


def _error_result(clip: AcceptanceClip, exc: Exception) -> dict[str, Any]:
    result = clip.report_fields()
    result.update(
        {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "frame_count": None,
            "confirmed_frame_count": None,
            "pre_attack_false_positive": None,
            "pre_attack_confirmed_frames": None,
            "hit_after_onset": None,
            "first_hit_frame": None,
            "first_delay_frames": None,
            "attack_window_confirmed_frames": None,
            "post_attack_confirmed_frames": None,
            "post_attack_observed_frames": None,
            "post_attack_linger_frames": None,
            "positive_missed": None,
            "negative_false_positive": None,
            "event_outcome": "evaluation_error",
        }
    )
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)
