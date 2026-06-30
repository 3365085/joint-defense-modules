from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PPE_LABELS = ("person", "head", "helmet")
MATURE_STABILIZER_SELECTION_REASONS = {
    "overlap_safe_protected_mature_track",
    "overlap_safe_mature_duplicate_track",
}

FIELDNAMES = [
    "record_index",
    "overlay_seq",
    "frame_idx",
    "video_time_s",
    "shadow_overlap_profile",
    "visible_person_count",
    "visible_head_count",
    "visible_helmet_count",
    "raw_person_count",
    "raw_head_count",
    "raw_helmet_count",
    "shadow_decision_count",
    "shadow_profile_kept_decision_count",
    "shadow_profile_rejected_decision_count",
    "person_conditioned_missing_head_helmet",
    "material_suggested_use",
    "has_suppressed_head_or_helmet",
    "has_weak_head_or_helmet",
    "roi_redetect_enabled",
    "roi_redetect_roi_count",
    "roi_redetect_base_box_count",
    "roi_redetect_candidate_box_count",
    "roi_redetect_final_box_count",
    "roi_redetect_nms_suppressed_count",
    "roi_redetect_final_roi_source_count",
    "roi_redetect_final_full_frame_source_count",
    "roi_redetect_suppressed_roi_source_count",
    "roi_redetect_suppressed_full_frame_source_count",
]


def summarize_ppe_overlay_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [ppe_overlay_row(record, record_index=index) for index, record in enumerate(records)]
    profile_counts: Counter[str] = Counter()
    material_use_counts: Counter[str] = Counter()
    ppe_reason_counts: Counter[str] = Counter()
    suppression_reason_counts: Counter[str] = Counter()
    suppressed_area_buckets = {"head": Counter(), "helmet": Counter()}
    nonzero_records = {
        "visible": Counter(),
        "raw": Counter(),
    }
    abrupt_label_drop_sum: Counter[str] = Counter()
    abrupt_event_total = 0
    abrupt_drop_sum = 0
    prev_visible: dict[str, int] | None = None

    shadow_decision_total = 0
    shadow_profile_kept_decision_total = 0
    shadow_profile_rejected_decision_total = 0
    shadow_same_target_reasons: Counter[str] = Counter()
    shadow_selection_reasons: Counter[str] = Counter()
    shadow_profile_reject_reasons: Counter[str] = Counter()
    shadow_profile_rejected_analysis: Counter[str] = Counter()
    shadow_profile_rejected_examples: list[dict[str, Any]] = []
    shadow_protected_mature_examples: list[dict[str, Any]] = []
    shadow_mature_stabilizer_examples: list[dict[str, Any]] = []
    shadow_mature_stabilizer_reasons: Counter[str] = Counter()
    shadow_profile_kept_render_status: Counter[str] = Counter()
    shadow_profile_kept_missing_drop_reasons: Counter[str] = Counter()
    person_conditioned_missing_records = 0
    suppressed_records = 0
    weak_records = 0
    person_conditioned_flags: list[bool] = []
    shadow_profile_kept_flags: list[bool] = []
    roi_redetect_rows: list[dict[str, Any]] = []

    for row in rows:
        profile_counts[str(row["shadow_overlap_profile"] or "legacy")] += 1
        material_use_counts[str(row["material_suggested_use"] or "unknown")] += 1
        shadow_decision_total += int(row["shadow_decision_count"])
        shadow_profile_kept_decision_total += int(row["shadow_profile_kept_decision_count"])
        shadow_profile_rejected_decision_total += int(row["shadow_profile_rejected_decision_count"])
        person_conditioned = bool(row["person_conditioned_missing_head_helmet"])
        shadow_kept = int(row["shadow_profile_kept_decision_count"]) > 0
        person_conditioned_flags.append(person_conditioned)
        shadow_profile_kept_flags.append(shadow_kept)
        if person_conditioned:
            person_conditioned_missing_records += 1
        if bool(row["has_suppressed_head_or_helmet"]):
            suppressed_records += 1
        if bool(row["has_weak_head_or_helmet"]):
            weak_records += 1
        roi_merge = _as_dict(row.get("_record", {}).get("detector_roi_redetect_merge"))
        if roi_merge:
            roi_redetect_rows.append(row)

        visible = {label: int(row[f"visible_{label}_count"]) for label in PPE_LABELS}
        raw = {label: int(row[f"raw_{label}_count"]) for label in PPE_LABELS}
        for label in PPE_LABELS:
            if visible[label] > 0:
                nonzero_records["visible"][label] += 1
            if raw[label] > 0:
                nonzero_records["raw"][label] += 1
        if prev_visible is not None:
            record_drop = 0
            for label in PPE_LABELS:
                drop = max(0, prev_visible[label] - visible[label])
                if drop:
                    abrupt_label_drop_sum[label] += drop
                    record_drop += drop
            if record_drop:
                abrupt_event_total += 1
                abrupt_drop_sum += record_drop
        prev_visible = visible

    for record in rows:
        original = record.get("_record")
        if not isinstance(original, dict):
            continue
        tracking = _as_dict(original.get("ppe_tracking_diagnostics"))
        for decision in _as_list(tracking.get("shadow_decisions")):
            if not isinstance(decision, dict):
                continue
            shadow_same_target_reasons[str(decision.get("same_target_reason") or "unknown")] += 1
            selection_reason = str(decision.get("selection_reason") or "unknown")
            shadow_selection_reasons[selection_reason] += 1
            if selection_reason == "overlap_safe_protected_mature_track" and len(shadow_protected_mature_examples) < 24:
                shadow_protected_mature_examples.append(
                    {
                        "record_index": _int(record.get("record_index")),
                        "overlay_seq": _int(original.get("overlay_seq")),
                        "frame_idx": _int(original.get("frame_idx")),
                        "same_target_reason": str(decision.get("same_target_reason") or ""),
                        "kept": _shadow_decision_track_sample(decision.get("kept")),
                        "dropped": _shadow_decision_track_sample(decision.get("dropped")),
                    }
                )
            if selection_reason in MATURE_STABILIZER_SELECTION_REASONS:
                shadow_mature_stabilizer_reasons[selection_reason] += 1
                if len(shadow_mature_stabilizer_examples) < 24:
                    shadow_mature_stabilizer_examples.append(
                        {
                            "record_index": _int(record.get("record_index")),
                            "overlay_seq": _int(original.get("overlay_seq")),
                            "frame_idx": _int(original.get("frame_idx")),
                            "selection_reason": selection_reason,
                            "same_target_reason": str(decision.get("same_target_reason") or ""),
                            "kept": _shadow_decision_track_sample(decision.get("kept")),
                            "dropped": _shadow_decision_track_sample(decision.get("dropped")),
                        }
                    )
        reject_counts = _as_dict(tracking.get("shadow_profile_reject_reason_counts"))
        if reject_counts:
            for reason, value in reject_counts.items():
                shadow_profile_reject_reasons[str(reason)] += _int(value)
        else:
            for decision in _as_list(tracking.get("shadow_profile_rejected_decisions")):
                if not isinstance(decision, dict):
                    continue
                shadow_profile_reject_reasons[str(decision.get("reject_reason") or "unknown")] += 1
        for decision in _as_list(tracking.get("shadow_profile_rejected_decisions")):
            if not isinstance(decision, dict):
                continue
            analysis = _shadow_profile_rejected_decision_analysis(decision)
            category = str(analysis.get("category") or "unknown")
            shadow_profile_rejected_analysis[category] += 1
            if len(shadow_profile_rejected_examples) < 24:
                shadow_profile_rejected_examples.append(
                    {
                        "record_index": _int(record.get("record_index")),
                        "overlay_seq": _int(original.get("overlay_seq")),
                        "frame_idx": _int(original.get("frame_idx")),
                        **analysis,
                    }
                )
        linkage = _as_dict(tracking.get("shadow_profile_kept_render_linkage"))
        for status, value in _as_dict(linkage.get("render_status_counts")).items():
            shadow_profile_kept_render_status[str(status)] += _int(value)
        for reason, value in _as_dict(linkage.get("missing_drop_reason_counts")).items():
            shadow_profile_kept_missing_drop_reasons[str(reason)] += _int(value)
        small = _as_dict(original.get("ppe_small_target_diagnostics"))
        reason = str(small.get("reason") or original.get("ppe_reason") or "")
        if reason:
            ppe_reason_counts[reason] += 1
        for name, value in _as_dict(small.get("suppression_reason_counts")).items():
            suppression_reason_counts[str(name)] += _int(value)
        buckets = _as_dict(small.get("suppressed_area_buckets"))
        for label in ("head", "helmet"):
            for bucket, value in _as_dict(buckets.get(label)).items():
                suppressed_area_buckets[label][str(bucket)] += _int(value)

    return {
        "record_count": len(rows),
        "profile_counts": dict(sorted(profile_counts.items())),
        "shadow_decision_total": shadow_decision_total,
        "shadow_profile_kept_decision_total": shadow_profile_kept_decision_total,
        "shadow_profile_rejected_decision_total": shadow_profile_rejected_decision_total,
        "shadow_same_target_reason_counts": _sorted_counter(shadow_same_target_reasons),
        "shadow_selection_reason_counts": _sorted_counter(shadow_selection_reasons),
        "shadow_protected_mature_track": {
            "selection_total": int(shadow_selection_reasons.get("overlap_safe_protected_mature_track", 0)),
            "examples": shadow_protected_mature_examples,
        },
        "shadow_mature_stabilizer_track": {
            "selection_total": sum(int(value) for value in shadow_mature_stabilizer_reasons.values()),
            "selection_reason_counts": _sorted_counter(shadow_mature_stabilizer_reasons),
            "examples": shadow_mature_stabilizer_examples,
        },
        "shadow_profile_reject_reason_counts": _sorted_counter(shadow_profile_reject_reasons),
        "shadow_profile_rejected_decision_analysis": {
            "category_counts": _sorted_counter(shadow_profile_rejected_analysis),
            "strict_new_track_duplicate_total": int(
                shadow_profile_rejected_analysis.get("strict_new_track_duplicate", 0)
            ),
            "mature_pair_review_total": int(
                shadow_profile_rejected_analysis.get("strict_mature_pair_review", 0)
                + shadow_profile_rejected_analysis.get("mature_pair_no_overlap_safe_evidence", 0)
            ),
            "examples": shadow_profile_rejected_examples,
        },
        "shadow_profile_kept_render_status_counts": _sorted_counter(shadow_profile_kept_render_status),
        "shadow_profile_kept_missing_drop_reason_counts": _sorted_counter(
            shadow_profile_kept_missing_drop_reasons
        ),
        "abrupt_event_total": abrupt_event_total,
        "abrupt_drop_sum": abrupt_drop_sum,
        "abrupt_label_drop_sum": _sorted_counter(abrupt_label_drop_sum),
        "visible_nonzero_records": _label_counter_dict(nonzero_records["visible"]),
        "raw_nonzero_records": _label_counter_dict(nonzero_records["raw"]),
        "material_screening": {
            "suggested_use_counts": dict(sorted(material_use_counts.items())),
            "person_conditioned_missing_records": person_conditioned_missing_records,
            "person_conditioned_missing_max_run": _max_true_run(person_conditioned_flags),
            "person_conditioned_missing_ranges": _flag_ranges(rows, person_conditioned_flags),
            "has_suppressed_head_or_helmet_records": suppressed_records,
            "has_weak_head_or_helmet_records": weak_records,
            "person_conditioned_same_person_track": _small_target_person_track_summary(rows),
        },
        "ppe_reason_counts": _sorted_counter(ppe_reason_counts),
        "suppression_reason_counts": _sorted_counter(suppression_reason_counts),
        "suppressed_area_buckets": {
            "head": _sorted_counter(suppressed_area_buckets["head"]),
            "helmet": _sorted_counter(suppressed_area_buckets["helmet"]),
        },
        "track_lifecycle": _track_lifecycle_summary(rows),
        "shadow_profile_decision_lifecycle": _shadow_profile_decision_lifecycle_summary(rows),
        "roi_redetect_merge": _roi_redetect_merge_summary(roi_redetect_rows),
        "candidate_flags": {
            "overlap_safe_profile_active": any(key == "overlap_safe_v1" for key in profile_counts),
            "overlap_safe_kept_records_present": shadow_profile_kept_decision_total > 0,
            "overlap_safe_kept_record_count": sum(1 for flag in shadow_profile_kept_flags if flag),
            "overlap_safe_kept_ranges": _flag_ranges(rows, shadow_profile_kept_flags),
            "overlap_safe_protected_decisions_present": shadow_selection_reasons.get("overlap_safe_protected_mature_track", 0) > 0,
            "overlap_safe_mature_stabilizer_decisions_present": sum(
                int(value) for value in shadow_mature_stabilizer_reasons.values()
            ) > 0,
            "small_target_person_conditioned_records_present": person_conditioned_missing_records > 0,
            "roi_redetect_records_present": bool(roi_redetect_rows),
        },
    }


def compare_ppe_overlay_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_person_conditioned_run: int = 3,
) -> dict[str, Any]:
    deltas = {
        "record_count": _int(candidate.get("record_count")) - _int(baseline.get("record_count")),
        "shadow_decision_total": _int(candidate.get("shadow_decision_total"))
        - _int(baseline.get("shadow_decision_total")),
        "shadow_profile_kept_decision_total": _int(candidate.get("shadow_profile_kept_decision_total"))
        - _int(baseline.get("shadow_profile_kept_decision_total")),
        "overlap_safe_protected_mature_track_selections": _protected_mature_selection_count(candidate)
        - _protected_mature_selection_count(baseline),
        "overlap_safe_mature_stabilizer_selections": _mature_stabilizer_selection_count(candidate)
        - _mature_stabilizer_selection_count(baseline),
        "abrupt_event_total": _int(candidate.get("abrupt_event_total"))
        - _int(baseline.get("abrupt_event_total")),
        "abrupt_drop_sum": _int(candidate.get("abrupt_drop_sum"))
        - _int(baseline.get("abrupt_drop_sum")),
        "person_conditioned_missing_records": _material_count(candidate, "person_conditioned_missing_records")
        - _material_count(baseline, "person_conditioned_missing_records"),
        "person_conditioned_missing_max_run": _material_count(candidate, "person_conditioned_missing_max_run")
        - _material_count(baseline, "person_conditioned_missing_max_run"),
        "appeared_track_total": _track_lifecycle_count(candidate, "appeared_track_total")
        - _track_lifecycle_count(baseline, "appeared_track_total"),
        "disappeared_track_total": _track_lifecycle_count(candidate, "disappeared_track_total")
        - _track_lifecycle_count(baseline, "disappeared_track_total"),
        "misses_exceed_render_cap_disappearances": _track_lifecycle_reason_count(
            candidate,
            "misses_exceed_render_cap",
        )
        - _track_lifecycle_reason_count(baseline, "misses_exceed_render_cap"),
        "held_track_not_eligible_disappearances": _track_lifecycle_reason_count(
            candidate,
            "held_track_not_eligible",
        )
        - _track_lifecycle_reason_count(baseline, "held_track_not_eligible"),
        "unexplained_track_disappearances": _track_lifecycle_reason_count(
            candidate,
            "not_in_current_overlay",
        )
        - _track_lifecycle_reason_count(baseline, "not_in_current_overlay"),
        "shadow_profile_touched_next_disappearances": _shadow_profile_decision_lifecycle_count(
            candidate,
            "all",
            "next_disappeared_track_total",
        )
        - _shadow_profile_decision_lifecycle_count(
            baseline,
            "all",
            "next_disappeared_track_total",
        ),
        "shadow_profile_kept_next_disappearances": _shadow_profile_decision_lifecycle_count(
            candidate,
            "kept",
            "next_disappeared_track_total",
        )
        - _shadow_profile_decision_lifecycle_count(
            baseline,
            "kept",
            "next_disappeared_track_total",
        ),
        "shadow_profile_rejected_next_disappearances": _shadow_profile_decision_lifecycle_count(
            candidate,
            "rejected",
            "next_disappeared_track_total",
        )
        - _shadow_profile_decision_lifecycle_count(
            baseline,
            "rejected",
            "next_disappeared_track_total",
        ),
    }
    overlap_gate = _overlap_safe_text_gate(candidate, deltas)
    small_target_gate = _small_target_text_gate(
        candidate,
        min_person_conditioned_run=max(1, int(min_person_conditioned_run)),
    )
    return {
        "baseline_record_count": _int(baseline.get("record_count")),
        "candidate_record_count": _int(candidate.get("record_count")),
        "deltas": deltas,
        "overlap_safe_text_gate": overlap_gate,
        "small_target_material_gate": small_target_gate,
        "ready_for_visual_validation": bool(overlap_gate["ready_for_visual_validation"]),
        "ready_for_small_target_candidate_design": bool(small_target_gate["ready_for_candidate_design"]),
    }


def ppe_overlay_csv_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in ppe_overlay_row(record, record_index=index).items() if key in FIELDNAMES}
        for index, record in enumerate(records)
    ]


def load_ppe_overlay_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return extract_ppe_overlay_records(payload)


def extract_ppe_overlay_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("overlay payload must be a JSON object or list")

    candidates = [
        payload.get("records"),
        _as_dict(payload.get("overlay")).get("records"),
        payload.get("overlay_records"),
    ]
    for value in candidates:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    raise ValueError("overlay payload does not contain records")


def build_ppe_overlay_summary_report(
    input_path: str | Path,
    *,
    baseline_path: str | Path | None = None,
    min_person_conditioned_run: int = 3,
) -> dict[str, Any]:
    input_records = load_ppe_overlay_records(input_path)
    summary = summarize_ppe_overlay_records(input_records)
    report = {
        "input_path": str(input_path),
        "record_count": len(input_records),
        "summary": summary,
    }
    if baseline_path is not None:
        baseline_records = load_ppe_overlay_records(baseline_path)
        baseline_summary = summarize_ppe_overlay_records(baseline_records)
        report["baseline_path"] = str(baseline_path)
        report["baseline_record_count"] = len(baseline_records)
        report["baseline_summary"] = baseline_summary
        report["comparison"] = compare_ppe_overlay_summaries(
            baseline_summary,
            summary,
            min_person_conditioned_run=min_person_conditioned_run,
        )
    return report


def write_ppe_overlay_summary_report(
    report: dict[str, Any],
    *,
    summary_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
) -> None:
    if summary_path is not None:
        summary_out = Path(summary_path)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if comparison_path is not None and "comparison" in report:
        comparison_out = Path(comparison_path)
        comparison_out.parent.mkdir(parents=True, exist_ok=True)
        comparison_out.write_text(
            json.dumps(report["comparison"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def write_ppe_overlay_csv_rows(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ppe_overlay_csv_rows(records)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ppe_overlay_row(record: dict[str, Any], *, record_index: int = 0) -> dict[str, Any]:
    visible = _visible_label_counts(record)
    raw = _raw_label_counts(record)
    tracking = _as_dict(record.get("ppe_tracking_diagnostics"))
    small = _as_dict(record.get("ppe_small_target_diagnostics"))
    material = _as_dict(small.get("material_screening"))
    roi_merge = _as_dict(record.get("detector_roi_redetect_merge"))
    roi_final_sources = _source_counts(roi_merge.get("final_sources"))
    roi_suppressed_sources = _source_counts(roi_merge.get("suppressed_sources"))
    return {
        "_record": record,
        "record_index": int(record_index),
        "overlay_seq": _int(record.get("overlay_seq", record.get("seq", record_index))),
        "frame_idx": _int(record.get("frame_idx")),
        "video_time_s": _float(record.get("video_time_s")),
        "shadow_overlap_profile": str(
            record.get("ppe_shadow_overlap_profile")
            or tracking.get("shadow_overlap_profile")
            or "legacy"
        ),
        "visible_person_count": visible["person"],
        "visible_head_count": visible["head"],
        "visible_helmet_count": visible["helmet"],
        "raw_person_count": raw["person"],
        "raw_head_count": raw["head"],
        "raw_helmet_count": raw["helmet"],
        "shadow_decision_count": _int(tracking.get("shadow_decision_count")),
        "shadow_profile_kept_decision_count": _int(tracking.get("shadow_profile_kept_decision_count")),
        "shadow_profile_rejected_decision_count": _int(
            tracking.get("shadow_profile_rejected_decision_count")
        ),
        "person_conditioned_missing_head_helmet": bool(
            material.get("person_conditioned_missing_head_helmet")
        ),
        "material_suggested_use": str(material.get("suggested_use") or "unknown"),
        "has_suppressed_head_or_helmet": bool(material.get("has_suppressed_head_or_helmet")),
        "has_weak_head_or_helmet": bool(material.get("has_weak_head_or_helmet")),
        "roi_redetect_enabled": bool(roi_merge.get("enabled")) if roi_merge else False,
        "roi_redetect_roi_count": _int(roi_merge.get("roi_count")),
        "roi_redetect_base_box_count": _int(roi_merge.get("base_box_count")),
        "roi_redetect_candidate_box_count": _int(roi_merge.get("candidate_box_count", roi_merge.get("candidate_count"))),
        "roi_redetect_final_box_count": _int(roi_merge.get("final_box_count")),
        "roi_redetect_nms_suppressed_count": _int(roi_merge.get("nms_suppressed_count")),
        "roi_redetect_final_roi_source_count": _int(roi_final_sources.get("roi_redetect")),
        "roi_redetect_final_full_frame_source_count": _int(roi_final_sources.get("full_frame")),
        "roi_redetect_suppressed_roi_source_count": _int(roi_suppressed_sources.get("roi_redetect")),
        "roi_redetect_suppressed_full_frame_source_count": _int(roi_suppressed_sources.get("full_frame")),
    }


def _visible_label_counts(record: dict[str, Any]) -> dict[str, int]:
    tracks = record.get("ppe_tracks")
    if isinstance(tracks, list):
        counts: Counter[str] = Counter()
        for track in tracks:
            if isinstance(track, dict):
                label = str(track.get("label") or "")
                if label in PPE_LABELS:
                    counts[label] += 1
        return _label_counter_dict(counts)
    class_counts = _as_dict(record.get("ppe_class_counts"))
    if class_counts:
        return {label: _int(class_counts.get(label)) for label in PPE_LABELS}
    return {
        "person": _int(record.get("ppe_person_count")),
        "head": _int(record.get("ppe_head_count")),
        "helmet": _int(record.get("ppe_helmet_count")),
    }


def _raw_label_counts(record: dict[str, Any]) -> dict[str, int]:
    class_counts = _as_dict(record.get("raw_class_counts"))
    if class_counts:
        return {label: _int(class_counts.get(label)) for label in PPE_LABELS}
    ppe_raw_counts = _as_dict(record.get("ppe_raw_class_counts"))
    if ppe_raw_counts:
        return {label: _int(ppe_raw_counts.get(label)) for label in PPE_LABELS}
    return {
        "person": _int(record.get("ppe_raw_person_count")),
        "head": _int(record.get("ppe_raw_head_count")),
        "helmet": _int(record.get("ppe_raw_helmet_count")),
    }


def _source_counts(sources: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for source in _as_list(sources):
        if isinstance(source, dict):
            counts[str(source.get("source") or "unknown")] += 1
    return counts


def _overlap_safe_text_gate(candidate: dict[str, Any], deltas: dict[str, int]) -> dict[str, Any]:
    flags = _as_dict(candidate.get("candidate_flags"))
    kept_render_counts = _as_dict(candidate.get("shadow_profile_kept_render_status_counts"))
    rejected_analysis = _as_dict(candidate.get("shadow_profile_rejected_decision_analysis"))
    mature_stabilizer_total = _mature_stabilizer_selection_count(candidate)
    lifecycle_net_improved = bool(
        mature_stabilizer_total > 0
        and _int(deltas.get("disappeared_track_total")) < 0
        and _int(deltas.get("appeared_track_total")) <= 0
    )
    reasons: list[str] = []
    if not bool(flags.get("overlap_safe_profile_active")):
        reasons.append("overlap_safe_profile_not_active")
    if _int(candidate.get("shadow_profile_kept_decision_total")) <= 0 and mature_stabilizer_total <= 0:
        reasons.append("no_overlap_safe_kept_decisions")
    kept_render_total = sum(_int(value) for value in kept_render_counts.values())
    not_all_rendered = kept_render_total - _int(kept_render_counts.get("all_rendered"))
    if not_all_rendered > 0:
        reasons.append("overlap_safe_kept_not_all_rendered")
    if _int(deltas.get("abrupt_event_total")) > 0 and not lifecycle_net_improved:
        reasons.append("abrupt_event_regression")
    if _int(deltas.get("abrupt_drop_sum")) > 0 and not lifecycle_net_improved:
        reasons.append("abrupt_drop_regression")
    if _int(deltas.get("disappeared_track_total")) > 0:
        reasons.append("track_disappearance_regression")
    if _int(deltas.get("misses_exceed_render_cap_disappearances")) > 0 and not lifecycle_net_improved:
        reasons.append("render_cap_disappearance_regression")
    if _int(deltas.get("held_track_not_eligible_disappearances")) > 0 and not lifecycle_net_improved:
        reasons.append("held_track_disappearance_regression")
    if _int(deltas.get("unexplained_track_disappearances")) > 0:
        reasons.append("unexplained_track_disappearance_regression")
    lifecycle = _as_dict(candidate.get("shadow_profile_decision_lifecycle"))
    kept_lifecycle = _as_dict(_as_dict(lifecycle.get("by_decision_type")).get("kept"))
    if _int(kept_lifecycle.get("next_disappeared_track_total")) > 0:
        reasons.append("overlap_safe_kept_track_disappeared_next")
    ready = not reasons
    return {
        "status": "candidate_for_visual_validation" if ready else "insufficient_text_evidence",
        "ready_for_visual_validation": ready,
        "reasons": reasons,
        "diagnostics": {
            "rejected_decision_analysis": rejected_analysis,
            "mature_stabilizer_selection_total": mature_stabilizer_total,
            "lifecycle_net_improved": lifecycle_net_improved,
        },
    }


def _small_target_text_gate(candidate: dict[str, Any], *, min_person_conditioned_run: int) -> dict[str, Any]:
    material = _as_dict(candidate.get("material_screening"))
    reasons: list[str] = []
    if _int(material.get("person_conditioned_missing_records")) <= 0:
        reasons.append("no_person_conditioned_missing_records")
    if _int(material.get("person_conditioned_missing_max_run")) < int(min_person_conditioned_run):
        reasons.append("person_conditioned_missing_not_continuous")
    if (
        _int(material.get("has_suppressed_head_or_helmet_records")) <= 0
        and _int(material.get("has_weak_head_or_helmet_records")) <= 0
    ):
        reasons.append("no_small_target_weak_or_suppressed_evidence")
    same_person = _as_dict(material.get("person_conditioned_same_person_track"))
    if _int(material.get("person_conditioned_missing_records")) > 0:
        if _int(same_person.get("candidate_person_track_event_total")) <= 0:
            reasons.append("no_person_conditioned_visible_person_track")
        elif _int(same_person.get("max_run")) < int(min_person_conditioned_run):
            reasons.append("person_conditioned_same_person_track_not_continuous")
    ready = not reasons
    return {
        "status": "candidate_material_found" if ready else "insufficient_stable_material",
        "ready_for_candidate_design": ready,
        "min_person_conditioned_run": int(min_person_conditioned_run),
        "reasons": reasons,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _label_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in PPE_LABELS}


def _track_lifecycle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records_with_track_ids = 0
    transitions_with_track_ids = 0
    skipped_run_boundary_transitions = 0
    continued_total = 0
    appeared_total = 0
    disappeared_total = 0
    continued_labels: Counter[str] = Counter()
    appeared_labels: Counter[str] = Counter()
    disappeared_labels: Counter[str] = Counter()
    disappearance_reasons: Counter[str] = Counter()
    business_filter_disappearance_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    track_maps: list[dict[int, dict[str, Any]]] = []
    for row in rows:
        track_map = _visible_track_map(_as_dict(row.get("_record")))
        if track_map:
            records_with_track_ids += 1
        track_maps.append(track_map)

    for index in range(1, len(rows)):
        previous_row = rows[index - 1]
        current_row = rows[index]
        previous_record = _as_dict(previous_row.get("_record"))
        current_record = _as_dict(current_row.get("_record"))
        previous_run_id = _int(previous_record.get("run_id"))
        current_run_id = _int(current_record.get("run_id"))
        if previous_run_id and current_run_id and previous_run_id != current_run_id:
            skipped_run_boundary_transitions += 1
            continue

        previous = track_maps[index - 1]
        current = track_maps[index]
        if not previous and not current:
            continue
        transitions_with_track_ids += 1
        continued_ids = sorted(set(previous) & set(current))
        appeared_ids = sorted(set(current) - set(previous))
        disappeared_ids = sorted(set(previous) - set(current))
        continued_total += len(continued_ids)
        appeared_total += len(appeared_ids)
        disappeared_total += len(disappeared_ids)

        for track_id in continued_ids:
            continued_labels[str(current[track_id].get("label") or "unknown")] += 1
        for track_id in appeared_ids:
            appeared_labels[str(current[track_id].get("label") or "unknown")] += 1

        drop_lookup = _track_drop_lookup(current_record)
        disappeared_items: list[dict[str, Any]] = []
        for track_id in disappeared_ids:
            label = str(previous[track_id].get("label") or "unknown")
            disappeared_labels[label] += 1
            drop = drop_lookup.get(track_id)
            reason = str(drop.get("reason") if drop else "not_in_current_overlay")
            disappearance_reasons[reason] += 1
            stage = str(drop.get("stage") if drop else "")
            if stage == "business_filter":
                business_filter_disappearance_reasons[reason] += 1
            disappeared_items.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "reason": reason,
                    "stage": stage,
                }
            )

        if (appeared_ids or disappeared_ids) and len(examples) < 12:
            examples.append(
                {
                    "from_record_index": index - 1,
                    "to_record_index": index,
                    "from_frame_idx": _int(previous_row.get("frame_idx")),
                    "to_frame_idx": _int(current_row.get("frame_idx")),
                    "from_overlay_seq": _int(previous_row.get("overlay_seq")),
                    "to_overlay_seq": _int(current_row.get("overlay_seq")),
                    "continued_count": len(continued_ids),
                    "appeared": [_track_event_item(track_id, current[track_id]) for track_id in appeared_ids[:12]],
                    "disappeared": disappeared_items[:12],
                }
            )

    return {
        "record_count": len(rows),
        "transition_count": max(0, len(rows) - 1),
        "records_with_track_ids": records_with_track_ids,
        "transitions_with_track_ids": transitions_with_track_ids,
        "skipped_run_boundary_transitions": skipped_run_boundary_transitions,
        "continued_track_total": continued_total,
        "appeared_track_total": appeared_total,
        "disappeared_track_total": disappeared_total,
        "continued_label_counts": _sorted_counter(continued_labels),
        "appeared_label_counts": _sorted_counter(appeared_labels),
        "disappeared_label_counts": _sorted_counter(disappeared_labels),
        "disappearance_reason_counts": _sorted_counter(disappearance_reasons),
        "business_filter_disappearance_reason_counts": _sorted_counter(
            business_filter_disappearance_reasons
        ),
        "transition_examples": examples,
    }


def _shadow_profile_decision_lifecycle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {
        "kept": _new_shadow_profile_lifecycle_bucket(),
        "rejected": _new_shadow_profile_lifecycle_bucket(),
    }
    track_maps = [_visible_track_map(_as_dict(row.get("_record"))) for row in rows]

    for index, row in enumerate(rows):
        record = _as_dict(row.get("_record"))
        tracking = _as_dict(record.get("ppe_tracking_diagnostics"))
        decision_groups = (
            ("kept", _as_list(tracking.get("shadow_profile_kept_decisions"))),
            ("rejected", _as_list(tracking.get("shadow_profile_rejected_decisions"))),
        )
        for decision_type, decisions in decision_groups:
            bucket = buckets[decision_type]
            for decision_index, decision in enumerate(decisions):
                if not isinstance(decision, dict):
                    continue
                bucket["decision_count"] += 1
                touched_tracks = _shadow_profile_decision_track_items(decision)
                bucket["touched_track_event_total"] += len(touched_tracks)
                next_state = _next_row_state(rows, index)
                current_tracks = track_maps[index] if index < len(track_maps) else {}
                next_tracks = track_maps[index + 1] if next_state["available"] else {}
                current_drop_lookup = _track_drop_lookup(record)
                next_drop_lookup = (
                    _track_drop_lookup(_as_dict(rows[index + 1].get("_record")))
                    if next_state["available"]
                    else {}
                )

                for track in touched_tracks:
                    track_id = _int(track.get("track_id"))
                    label = str(track.get("label") or "unknown")
                    if track_id <= 0:
                        continue
                    bucket["unique_touched_track_ids"].add(track_id)
                    bucket["touched_label_counts"][label] += 1
                    event = {
                        "decision_type": decision_type,
                        "record_index": int(index),
                        "decision_index": int(decision_index),
                        "overlay_seq": _int(row.get("overlay_seq")),
                        "frame_idx": _int(row.get("frame_idx")),
                        "track_id": track_id,
                        "label": label,
                        "same_target_reason": str(decision.get("same_target_reason") or ""),
                        "reject_reason": str(decision.get("reject_reason") or ""),
                        "overlap_safe_reason": _overlap_safe_reason(decision),
                    }

                    if track_id not in current_tracks:
                        drop = current_drop_lookup.get(track_id)
                        reason = str(drop.get("reason") if drop else "not_rendered_in_decision_record")
                        bucket["current_missing_track_total"] += 1
                        bucket["current_missing_reason_counts"][reason] += 1
                        event.update(
                            {
                                "current_status": "not_rendered",
                                "current_reason": reason,
                                "current_stage": str(drop.get("stage") if drop else ""),
                                "next_status": "not_applicable",
                            }
                        )
                        _append_shadow_profile_lifecycle_example(bucket, event)
                        continue

                    bucket["current_rendered_track_total"] += 1
                    event["current_status"] = "rendered"
                    if not next_state["available"]:
                        reason = str(next_state["reason"])
                        bucket["next_unavailable_track_total"] += 1
                        bucket["next_unavailable_reason_counts"][reason] += 1
                        event.update(
                            {
                                "next_status": "unavailable",
                                "next_reason": reason,
                            }
                        )
                        _append_shadow_profile_lifecycle_example(bucket, event)
                        continue

                    event["next_overlay_seq"] = _int(rows[index + 1].get("overlay_seq"))
                    event["next_frame_idx"] = _int(rows[index + 1].get("frame_idx"))
                    if track_id in next_tracks:
                        bucket["next_continued_track_total"] += 1
                        event["next_status"] = "continued"
                    else:
                        drop = next_drop_lookup.get(track_id)
                        reason = str(drop.get("reason") if drop else "not_in_current_overlay")
                        bucket["next_disappeared_track_total"] += 1
                        bucket["next_disappearance_reason_counts"][reason] += 1
                        event.update(
                            {
                                "next_status": "disappeared",
                                "next_reason": reason,
                                "next_stage": str(drop.get("stage") if drop else ""),
                            }
                        )
                    _append_shadow_profile_lifecycle_example(bucket, event)

    kept = _finalize_shadow_profile_lifecycle_bucket(buckets["kept"])
    rejected = _finalize_shadow_profile_lifecycle_bucket(buckets["rejected"])
    reason_counts = Counter()
    for bucket in (kept, rejected):
        for reason, value in _as_dict(bucket.get("next_disappearance_reason_counts")).items():
            reason_counts[str(reason)] += _int(value)
    unique_ids = set(buckets["kept"]["unique_touched_track_ids"]) | set(
        buckets["rejected"]["unique_touched_track_ids"]
    )
    return {
        "decision_count": _int(kept.get("decision_count")) + _int(rejected.get("decision_count")),
        "touched_track_event_total": _int(kept.get("touched_track_event_total"))
        + _int(rejected.get("touched_track_event_total")),
        "unique_touched_track_count": len(unique_ids),
        "current_rendered_track_total": _int(kept.get("current_rendered_track_total"))
        + _int(rejected.get("current_rendered_track_total")),
        "current_missing_track_total": _int(kept.get("current_missing_track_total"))
        + _int(rejected.get("current_missing_track_total")),
        "next_continued_track_total": _int(kept.get("next_continued_track_total"))
        + _int(rejected.get("next_continued_track_total")),
        "next_disappeared_track_total": _int(kept.get("next_disappeared_track_total"))
        + _int(rejected.get("next_disappeared_track_total")),
        "next_unavailable_track_total": _int(kept.get("next_unavailable_track_total"))
        + _int(rejected.get("next_unavailable_track_total")),
        "next_disappearance_reason_counts": _sorted_counter(reason_counts),
        "by_decision_type": {
            "kept": kept,
            "rejected": rejected,
        },
    }


def _roi_redetect_merge_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records_with_roi = 0
    roi_count = 0
    candidate_count = 0
    final_box_count = 0
    nms_input_count = 0
    nms_kept_count = 0
    nms_suppressed_count = 0
    decision_counts: Counter[str] = Counter()
    final_source_counts: Counter[str] = Counter()
    suppressed_source_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in rows:
        record = row.get("_record")
        if not isinstance(record, dict):
            continue
        merge = _as_dict(record.get("detector_roi_redetect_merge"))
        if not merge:
            continue
        records_with_roi += 1
        roi_count += _int(merge.get("roi_count"))
        candidate_count += _int(merge.get("candidate_box_count", merge.get("candidate_count")))
        final_box_count += _int(merge.get("final_box_count"))
        nms_input_count += _int(merge.get("nms_input_count"))
        nms_kept_count += _int(merge.get("nms_kept_count"))
        nms_suppressed_count += _int(merge.get("nms_suppressed_count"))
        for source in _as_list(merge.get("final_sources")):
            if isinstance(source, dict):
                final_source_counts[str(source.get("source") or "unknown")] += 1
        for source in _as_list(merge.get("suppressed_sources")):
            if isinstance(source, dict):
                suppressed_source_counts[str(source.get("source") or "unknown")] += 1
        for roi in _as_list(merge.get("rois")):
            if not isinstance(roi, dict):
                continue
            for decision in _as_list(roi.get("decisions")):
                if isinstance(decision, dict):
                    decision_counts[str(decision.get("decision") or "unknown")] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "record_index": _int(row.get("record_index")),
                    "overlay_seq": _int(record.get("overlay_seq")),
                    "frame_idx": _int(record.get("frame_idx")),
                    "roi_count": _int(merge.get("roi_count")),
                    "candidate_count": _int(merge.get("candidate_box_count", merge.get("candidate_count"))),
                    "final_box_count": _int(merge.get("final_box_count")),
                    "decision_counts": _decision_counts_for_merge(merge),
                    "final_sources": [
                        str(source.get("source") or "unknown")
                        for source in _as_list(merge.get("final_sources"))
                        if isinstance(source, dict)
                    ],
                    "suppressed_sources": [
                        str(source.get("source") or "unknown")
                        for source in _as_list(merge.get("suppressed_sources"))
                        if isinstance(source, dict)
                    ],
                }
            )

    return {
        "records_with_roi": records_with_roi,
        "roi_count": roi_count,
        "candidate_count": candidate_count,
        "final_box_count": final_box_count,
        "nms_input_count": nms_input_count,
        "nms_kept_count": nms_kept_count,
        "nms_suppressed_count": nms_suppressed_count,
        "decision_counts": _sorted_counter(decision_counts),
        "final_source_counts": _sorted_counter(final_source_counts),
        "suppressed_source_counts": _sorted_counter(suppressed_source_counts),
        "examples": examples,
    }


def _decision_counts_for_merge(merge: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for roi in _as_list(merge.get("rois")):
        if not isinstance(roi, dict):
            continue
        for decision in _as_list(roi.get("decisions")):
            if isinstance(decision, dict):
                counts[str(decision.get("decision") or "unknown")] += 1
    return _sorted_counter(counts)


def _small_target_person_track_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active: dict[int, dict[str, Any]] = {}
    ranges: list[dict[str, Any]] = []
    track_ids: set[int] = set()
    candidate_person_track_event_total = 0
    records_with_candidate_person_track = 0
    missing_without_person_track_records = 0
    previous_run_id = 0

    for index, row in enumerate(rows):
        record = _as_dict(row.get("_record"))
        run_id = _int(record.get("run_id"))
        if previous_run_id and run_id and previous_run_id != run_id:
            ranges.extend(_close_small_target_person_ranges(active))
        if run_id:
            previous_run_id = run_id

        small = _as_dict(record.get("ppe_small_target_diagnostics"))
        material = _as_dict(small.get("material_screening"))
        is_candidate_record = bool(material.get("person_conditioned_missing_head_helmet"))
        person_tracks = _visible_person_track_map(record) if is_candidate_record else {}
        current_ids = set(person_tracks)

        for track_id in sorted(set(active) - current_ids):
            ranges.append(_finalize_small_target_person_range(active.pop(track_id)))

        if not is_candidate_record:
            continue
        if current_ids:
            records_with_candidate_person_track += 1
        else:
            missing_without_person_track_records += 1
            continue

        for track_id, track in sorted(person_tracks.items()):
            track_ids.add(track_id)
            candidate_person_track_event_total += 1
            if track_id not in active:
                active[track_id] = {"track_id": track_id, "events": []}
            active[track_id]["events"].append(
                _small_target_person_track_event(
                    row,
                    track,
                    small,
                    record_index=index,
                )
            )

    ranges.extend(_close_small_target_person_ranges(active))
    ranges.sort(
        key=lambda item: (
            -_int(item.get("count")),
            _int(item.get("start_record_index")),
            _int(item.get("track_id")),
        )
    )
    max_run = max((_int(item.get("count")) for item in ranges), default=0)
    stable_min_run = 3
    stable_ranges = [item for item in ranges if _int(item.get("count")) >= stable_min_run]
    return {
        "stable_min_run": stable_min_run,
        "candidate_person_track_event_total": candidate_person_track_event_total,
        "records_with_candidate_person_track": records_with_candidate_person_track,
        "person_conditioned_missing_without_person_track_records": missing_without_person_track_records,
        "track_count": len(track_ids),
        "range_count": len(ranges),
        "stable_range_count": len(stable_ranges),
        "max_run": max_run,
        "ranges": ranges[:24],
        "stable_ranges": stable_ranges[:12],
    }


def _close_small_target_person_ranges(active: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    ranges = [_finalize_small_target_person_range(active.pop(track_id)) for track_id in list(active)]
    return ranges


def _small_target_person_track_event(
    row: dict[str, Any],
    track: dict[str, Any],
    small: dict[str, Any],
    *,
    record_index: int,
) -> dict[str, Any]:
    weak_counts = _as_dict(small.get("weak_counts"))
    suppressed_counts = _as_dict(small.get("suppressed_counts"))
    center = _track_center(track)
    return {
        "record_index": int(record_index),
        "overlay_seq": _int(row.get("overlay_seq")),
        "frame_idx": _int(row.get("frame_idx")),
        "video_time_s": _float(row.get("video_time_s")),
        "track_id": _track_id(track),
        "label": str(track.get("label") or "unknown"),
        "age": _int(track.get("age")),
        "misses": _int(track.get("misses")),
        "confidence": _float(track.get("confidence")),
        "area_ratio": _float(track.get("area_ratio")),
        "box": list(track.get("box") or []),
        "center_x": center[0],
        "center_y": center[1],
        "visible_head_count": _int(row.get("visible_head_count")),
        "visible_helmet_count": _int(row.get("visible_helmet_count")),
        "raw_head_count": _int(row.get("raw_head_count")),
        "raw_helmet_count": _int(row.get("raw_helmet_count")),
        "weak_head_count": _int(weak_counts.get("head")),
        "weak_helmet_count": _int(weak_counts.get("helmet")),
        "suppressed_head_count": _int(suppressed_counts.get("head")),
        "suppressed_helmet_count": _int(suppressed_counts.get("helmet")),
        "suggested_use": str(_as_dict(small.get("material_screening")).get("suggested_use") or ""),
        "reason": str(small.get("reason") or ""),
        "suppression_reason_counts": _as_dict(small.get("suppression_reason_counts")),
    }


def _finalize_small_target_person_range(state: dict[str, Any]) -> dict[str, Any]:
    events = list(state.get("events") or [])
    if not events:
        return {"track_id": _int(state.get("track_id")), "count": 0}
    first = events[0]
    last = events[-1]
    confidences = [_float(event.get("confidence")) for event in events]
    ages = [_int(event.get("age")) for event in events]
    misses = [_int(event.get("misses")) for event in events]
    area_ratios = [_float(event.get("area_ratio")) for event in events]
    first_center = (_float(first.get("center_x")), _float(first.get("center_y")))
    max_center_drift_px = max(
        (
            ((_float(event.get("center_x")) - first_center[0]) ** 2 + (_float(event.get("center_y")) - first_center[1]) ** 2)
            ** 0.5
            for event in events
        ),
        default=0.0,
    )
    weak_counts = Counter()
    suppressed_counts = Counter()
    suppression_reasons: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for event in events:
        weak_counts["head"] += _int(event.get("weak_head_count"))
        weak_counts["helmet"] += _int(event.get("weak_helmet_count"))
        suppressed_counts["head"] += _int(event.get("suppressed_head_count"))
        suppressed_counts["helmet"] += _int(event.get("suppressed_helmet_count"))
        reason = str(event.get("reason") or "")
        if reason:
            reasons[reason] += 1
        for name, value in _as_dict(event.get("suppression_reason_counts")).items():
            suppression_reasons[str(name)] += _int(value)
    duration = max(0.0, _float(last.get("video_time_s")) - _float(first.get("video_time_s")))
    return {
        "track_id": _int(state.get("track_id")),
        "count": len(events),
        "duration_s": duration,
        "start_record_index": _int(first.get("record_index")),
        "end_record_index": _int(last.get("record_index")),
        "start_frame_idx": _int(first.get("frame_idx")),
        "end_frame_idx": _int(last.get("frame_idx")),
        "start_overlay_seq": _int(first.get("overlay_seq")),
        "end_overlay_seq": _int(last.get("overlay_seq")),
        "start_video_time_s": _float(first.get("video_time_s")),
        "end_video_time_s": _float(last.get("video_time_s")),
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "min_age": min(ages) if ages else 0,
        "max_age": max(ages) if ages else 0,
        "max_misses": max(misses) if misses else 0,
        "min_area_ratio": min(area_ratios) if area_ratios else 0.0,
        "max_area_ratio": max(area_ratios) if area_ratios else 0.0,
        "max_center_drift_px": max_center_drift_px,
        "weak_counts": _label_head_helmet_counter_dict(weak_counts),
        "suppressed_counts": _label_head_helmet_counter_dict(suppressed_counts),
        "suppression_reason_counts": _sorted_counter(suppression_reasons),
        "reason_counts": _sorted_counter(reasons),
        "first_event": _small_target_range_event_sample(first),
        "last_event": _small_target_range_event_sample(last),
    }


def _small_target_range_event_sample(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_idx": _int(event.get("frame_idx")),
        "overlay_seq": _int(event.get("overlay_seq")),
        "video_time_s": _float(event.get("video_time_s")),
        "age": _int(event.get("age")),
        "misses": _int(event.get("misses")),
        "confidence": _float(event.get("confidence")),
        "area_ratio": _float(event.get("area_ratio")),
        "box": list(event.get("box") or []),
    }


def _new_shadow_profile_lifecycle_bucket() -> dict[str, Any]:
    return {
        "decision_count": 0,
        "touched_track_event_total": 0,
        "unique_touched_track_ids": set(),
        "touched_label_counts": Counter(),
        "current_rendered_track_total": 0,
        "current_missing_track_total": 0,
        "current_missing_reason_counts": Counter(),
        "next_continued_track_total": 0,
        "next_disappeared_track_total": 0,
        "next_unavailable_track_total": 0,
        "next_disappearance_reason_counts": Counter(),
        "next_unavailable_reason_counts": Counter(),
        "examples": [],
    }


def _finalize_shadow_profile_lifecycle_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_count": _int(bucket.get("decision_count")),
        "touched_track_event_total": _int(bucket.get("touched_track_event_total")),
        "unique_touched_track_count": len(bucket.get("unique_touched_track_ids") or set()),
        "touched_label_counts": _sorted_counter(bucket.get("touched_label_counts") or Counter()),
        "current_rendered_track_total": _int(bucket.get("current_rendered_track_total")),
        "current_missing_track_total": _int(bucket.get("current_missing_track_total")),
        "current_missing_reason_counts": _sorted_counter(
            bucket.get("current_missing_reason_counts") or Counter()
        ),
        "next_continued_track_total": _int(bucket.get("next_continued_track_total")),
        "next_disappeared_track_total": _int(bucket.get("next_disappeared_track_total")),
        "next_unavailable_track_total": _int(bucket.get("next_unavailable_track_total")),
        "next_disappearance_reason_counts": _sorted_counter(
            bucket.get("next_disappearance_reason_counts") or Counter()
        ),
        "next_unavailable_reason_counts": _sorted_counter(
            bucket.get("next_unavailable_reason_counts") or Counter()
        ),
        "examples": list(bucket.get("examples") or []),
    }


def _append_shadow_profile_lifecycle_example(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    if len(bucket["examples"]) < 16 and event.get("next_status") != "continued":
        bucket["examples"].append(event)


def _next_row_state(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index + 1 >= len(rows):
        return {"available": False, "reason": "no_next_record"}
    current_record = _as_dict(rows[index].get("_record"))
    next_record = _as_dict(rows[index + 1].get("_record"))
    current_run_id = _int(current_record.get("run_id"))
    next_run_id = _int(next_record.get("run_id"))
    if current_run_id and next_run_id and current_run_id != next_run_id:
        return {"available": False, "reason": "run_boundary"}
    return {"available": True, "reason": ""}


def _shadow_profile_decision_track_items(decision: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[int] = set()
    for key in ("left", "right", "kept", "dropped"):
        item = _as_dict(decision.get(key))
        track_id = _track_id(item)
        if track_id <= 0 or track_id in seen:
            continue
        seen.add(track_id)
        items.append(
            {
                "track_id": track_id,
                "label": str(item.get("label") or item.get("stable_label") or "unknown"),
            }
        )
    return items


def _shadow_decision_track_sample(value: Any) -> dict[str, Any]:
    item = _as_dict(value)
    if not item:
        return {}
    return {
        "track_id": _track_id(item),
        "label": str(item.get("label") or item.get("stable_label") or "unknown"),
        "misses": _int(item.get("misses")),
        "age": _int(item.get("age")),
        "confidence": _float(item.get("confidence")),
    }


def _overlap_safe_reason(decision: dict[str, Any]) -> str:
    overlap_safe = _as_dict(decision.get("overlap_safe"))
    return str(overlap_safe.get("reason") or "")


def _shadow_profile_rejected_decision_analysis(decision: dict[str, Any]) -> dict[str, Any]:
    left = _as_dict(decision.get("left"))
    right = _as_dict(decision.get("right"))
    reject_reason = str(decision.get("reject_reason") or _overlap_safe_reason(decision) or "unknown")
    same_target_reason = str(decision.get("same_target_reason") or "")
    iou = _float(decision.get("iou"))
    distance = _float(decision.get("center_distance_ratio"))
    containment = _float(decision.get("containment"))
    left_age = _int(left.get("age"))
    right_age = _int(right.get("age"))
    left_misses = _int(left.get("misses"))
    right_misses = _int(right.get("misses"))
    min_age = min(left_age, right_age)
    max_misses = max(left_misses, right_misses)
    same_label = same_target_reason in {"same_head_overlap", "same_helmet_overlap"}
    strict_geometry = reject_reason == "strict_geometry_would_remove"

    if (
        strict_geometry
        and min_age <= 2
        and (iou >= 0.55 or distance <= 0.018 or containment >= 0.85)
    ):
        category = "strict_new_track_duplicate"
    elif (
        strict_geometry
        and same_label
        and min_age >= 3
        and max_misses <= 1
        and distance <= 0.060
        and containment < 0.95
    ):
        category = "strict_mature_pair_review"
    elif (
        reject_reason == "no_overlap_safe_evidence"
        and same_label
        and min_age >= 3
        and max_misses <= 1
    ):
        category = "mature_pair_no_overlap_safe_evidence"
    elif strict_geometry:
        category = "strict_geometry_other"
    else:
        category = reject_reason or "unknown"

    return {
        "category": category,
        "reject_reason": reject_reason,
        "same_target_reason": same_target_reason,
        "iou": iou,
        "center_distance_ratio": distance,
        "containment": containment,
        "left_track_id": _track_id(left),
        "right_track_id": _track_id(right),
        "left_label": str(left.get("label") or left.get("stable_label") or ""),
        "right_label": str(right.get("label") or right.get("stable_label") or ""),
        "left_age": left_age,
        "right_age": right_age,
        "left_misses": left_misses,
        "right_misses": right_misses,
    }


def _visible_track_map(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tracks = record.get("ppe_tracks")
    if not isinstance(tracks, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = _track_id(track)
        label = str(track.get("label") or "")
        if track_id <= 0 or label not in PPE_LABELS:
            continue
        out[track_id] = {
            "track_id": track_id,
            "label": label,
            "misses": _int(track.get("misses")),
            "confidence": _float(track.get("confidence")),
        }
    return out


def _visible_person_track_map(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tracks = record.get("ppe_tracks")
    if not isinstance(tracks, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = _track_id(track)
        if track_id <= 0 or str(track.get("label") or "") != "person":
            continue
        out[track_id] = dict(track)
    return out


def _track_drop_lookup(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tracking = _as_dict(record.get("ppe_tracking_diagnostics"))
    out: dict[int, dict[str, Any]] = {}
    for drop in _as_list(tracking.get("drops")):
        if not isinstance(drop, dict):
            continue
        track_id = _track_id(drop)
        if track_id > 0 and track_id not in out:
            out[track_id] = drop
    business = _as_dict(tracking.get("business_filter"))
    for drop in _as_list(business.get("dropped")):
        if not isinstance(drop, dict):
            continue
        track_id = _track_id(drop)
        if track_id <= 0 or track_id in out:
            continue
        business_drop = dict(drop)
        business_drop["stage"] = str(business_drop.get("stage") or "business_filter")
        business_drop["reason"] = str(business_drop.get("reason") or "business_filtered")
        out[track_id] = business_drop
    return out


def _track_event_item(track_id: int, track: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": int(track_id),
        "label": str(track.get("label") or "unknown"),
        "misses": _int(track.get("misses")),
        "confidence": _float(track.get("confidence")),
    }


def _track_center(track: dict[str, Any]) -> tuple[float, float]:
    box = track.get("box")
    if not isinstance(box, list | tuple) or len(box) < 4:
        return (0.0, 0.0)
    x1, y1, x2, y2 = (_float(box[0]), _float(box[1]), _float(box[2]), _float(box[3]))
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _track_id(track: dict[str, Any]) -> int:
    return _int(track.get("track_id", track.get("id")))


def _track_lifecycle_count(summary: dict[str, Any], key: str) -> int:
    return _int(_as_dict(summary.get("track_lifecycle")).get(key))


def _track_lifecycle_reason_count(summary: dict[str, Any], reason: str) -> int:
    lifecycle = _as_dict(summary.get("track_lifecycle"))
    return _int(_as_dict(lifecycle.get("disappearance_reason_counts")).get(reason))


def _protected_mature_selection_count(summary: dict[str, Any]) -> int:
    return _int(_as_dict(summary.get("shadow_protected_mature_track")).get("selection_total"))


def _mature_stabilizer_selection_count(summary: dict[str, Any]) -> int:
    return _int(_as_dict(summary.get("shadow_mature_stabilizer_track")).get("selection_total"))


def _shadow_profile_decision_lifecycle_count(
    summary: dict[str, Any],
    decision_type: str,
    key: str,
) -> int:
    lifecycle = _as_dict(summary.get("shadow_profile_decision_lifecycle"))
    if decision_type == "all":
        return _int(lifecycle.get(key))
    by_type = _as_dict(lifecycle.get("by_decision_type"))
    return _int(_as_dict(by_type.get(decision_type)).get(key))


def _material_count(summary: dict[str, Any], key: str) -> int:
    return _int(_as_dict(summary.get("material_screening")).get(key))


def _label_head_helmet_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in ("head", "helmet")}



def _max_true_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _flag_ranges(rows: list[dict[str, Any]], flags: list[bool]) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start_index: int | None = None
    previous_index: int | None = None
    for index, flag in enumerate(flags):
        if flag:
            if start_index is None:
                start_index = index
            previous_index = index
            continue
        if start_index is not None and previous_index is not None:
            ranges.append(_range_record(rows, start_index, previous_index))
        start_index = None
        previous_index = None
    if start_index is not None and previous_index is not None:
        ranges.append(_range_record(rows, start_index, previous_index))
    return ranges


def _range_record(rows: list[dict[str, Any]], start_index: int, end_index: int) -> dict[str, int]:
    start = rows[start_index]
    end = rows[end_index]
    return {
        "start_record_index": int(start_index),
        "end_record_index": int(end_index),
        "count": int(end_index - start_index + 1),
        "start_frame_idx": _int(start.get("frame_idx")),
        "end_frame_idx": _int(end.get("frame_idx")),
        "start_overlay_seq": _int(start.get("overlay_seq")),
        "end_overlay_seq": _int(end.get("overlay_seq")),
    }
