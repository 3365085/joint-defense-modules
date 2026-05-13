from __future__ import annotations

from collections import deque
from statistics import fmean, median, pstdev
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from ..types import ROI


class _ClipClassifierProtocol(Protocol):
    """Subset of :class:`TorchLogisticFusion` we actually consume.

    Declared as a :class:`typing.Protocol` so tests can inject a fake that
    returns a deterministic ``classifier_p_adv`` without paying for a real
    JSON artifact on disk or a torch import at test-collection time.
    """

    def compute(self, features: dict[str, float]) -> dict[str, Any]: ...


class SourceAuthenticityDetector:
    """Clip-level detector for suspected generated/replayed video streams.

    This branch is intentionally conservative. It records cheap per-frame
    statistics and only evaluates a clip-level score on a keyframe interval.
    The score is reported as ``p_synth`` and does not change Module A
    ``p_adv``.

    Task 6.4 adds an optional ``clip_classifier`` (a
    :class:`defense.module_a.fusion.classifier_fusion.TorchLogisticFusion`
    artifact, typically
    ``experiments/configs/module_a_synth_classifier_v1.json``) that, when
    wired up AND enabled AND its clip-length rolling buffer is full, takes
    over ``p_synth`` via its ``classifier_p_adv`` output. When the
    classifier is not configured, not enabled, or its window has not yet
    filled, the hand-weighted score from the pre-Task-6.4 formula is used
    unchanged. This preserves bit-for-bit behaviour of the default baseline
    (``synth_classifier_enabled=false``, Req 6.3 / 9.1 spirit).

    Classifier input contract (Task 6.3):
      * 5 retained raw features per ``_evaluate`` call:
        ``highfreq_std``, ``highfreq_mean``, ``edge_mean``, ``diff_std_mean``,
        ``score_oversmooth``.
      * 3 aggregations per raw feature (``mean``, ``median``, ``std``) across
        the last ``classifier_window`` evaluations -> up to 15-dim vector.
      * ``feature_names`` in the artifact are ``{raw}_{agg}`` (e.g.
        ``highfreq_std_mean``), matching ``tools.train_synth_classifier``.
    """

    _CLASSIFIER_FEATURE_NAMES: tuple[str, ...] = (
        "highfreq_std",
        "highfreq_mean",
        "edge_mean",
        "diff_std_mean",
        "score_oversmooth",
    )
    _CLASSIFIER_AGGREGATIONS: tuple[str, ...] = ("mean", "median", "std")

    def __init__(
        self,
        enabled: bool = False,
        interval: int = 3,
        window: int = 30,
        min_window: int = 8,
        threshold: float = 0.40,
        warning_window: int = 4,
        warning_trigger_count: int = 2,
        hold_frames: int = 10,
        repeated_diff_threshold: float = 0.0035,
        low_motion_threshold: float = 0.010,
        low_edge_threshold: float = 0.020,
        flicker_threshold: float = 0.018,
        roi_jitter_threshold: float = 0.020,
        clip_classifier: _ClipClassifierProtocol | None = None,
        classifier_enabled: bool = False,
        classifier_window: int = 60,
    ):
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.window = max(3, int(window))
        self.min_window = max(2, min(int(min_window), self.window))
        self.threshold = float(threshold)
        self.warning_history: deque[int] = deque(maxlen=max(1, int(warning_window)))
        self.warning_trigger_count = max(
            1,
            min(int(warning_trigger_count), self.warning_history.maxlen or 1),
        )
        self.hold_frames = max(0, int(hold_frames))
        self.hold_remaining = 0
        self.repeated_diff_threshold = float(repeated_diff_threshold)
        self.low_motion_threshold = float(low_motion_threshold)
        self.low_edge_threshold = float(low_edge_threshold)
        self.flicker_threshold = float(flicker_threshold)
        self.roi_jitter_threshold = float(roi_jitter_threshold)
        self.samples: deque[dict[str, float]] = deque(maxlen=self.window)
        self.prev_roi_signature: dict[str, tuple[float, float, float, float]] = {}

        # Task 6.4 — Synth_Classifier hookup.
        # ``clip_classifier is None`` means the artifact was never loaded,
        # which is also the default baseline state (Req 6.3 / 9.1 spirit).
        # ``classifier_enabled`` is the rollout gate: when False the
        # classifier is scored into shadow fields but does NOT override
        # ``p_synth``; mirrors the Static_Media_Classifier gating from
        # Task 5.4. The rolling buffer length matches Task 6.3's
        # ``clip_window_frames=60`` default so training and inference
        # consume identical clip aggregations.
        self.clip_classifier = clip_classifier
        self.classifier_enabled = bool(classifier_enabled)
        self.classifier_window = max(1, int(classifier_window))
        self._classifier_rolling: deque[dict[str, float]] = deque(maxlen=self.classifier_window)

        self.last_result = self._empty()

    def reset(self) -> None:
        self.samples.clear()
        self.warning_history.clear()
        self.prev_roi_signature.clear()
        self.hold_remaining = 0
        # Also drop the clip-classifier rolling buffer so a new stream
        # cannot carry stale statistics forward across resets (e.g. when
        # ``stream_geometry_changed`` forces a pipeline reset).
        self._classifier_rolling.clear()
        self.last_result = self._empty()

    def compute(
        self,
        prev_gray: torch.Tensor | None,
        curr_gray: torch.Tensor,
        rois: list[ROI] | None,
        frame_idx: int,
        temporal: dict[str, Any] | None = None,
        motion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            self.last_result = self._empty(enabled=False)
            return self.last_result

        sample = self._sample(prev_gray, curr_gray, rois or [])
        self.samples.append(sample)
        should_evaluate = prev_gray is not None and frame_idx % self.interval == 0
        if not should_evaluate:
            result = dict(self.last_result)
            result.update(
                {
                    "source_authenticity_enabled": True,
                    "source_authenticity_evaluated": False,
                    "source_authenticity_frame_delta": sample["frame_delta"],
                    "source_authenticity_edge_mean": sample["edge_mean"],
                    "source_authenticity_highfreq_mean": sample["highfreq_mean"],
                    "source_authenticity_roi_jitter": sample["roi_jitter"],
                    "source_authenticity_window_size": len(self.samples),
                }
            )
            self.last_result = result
            return result

        result = self._evaluate(sample, temporal or {}, motion or {})
        self.last_result = result
        return result

    def _sample(
        self,
        prev_gray: torch.Tensor | None,
        curr_gray: torch.Tensor,
        rois: list[ROI],
    ) -> dict[str, float]:
        curr = curr_gray.float() / 255.0
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            frame_delta = 0.0
            diff_std = 0.0
        else:
            diff = torch.abs(curr_gray.float() - prev_gray.float()) / 255.0
            frame_delta = float(diff.mean().item())
            diff_std = float(diff.std(unbiased=False).item())

        dx = torch.abs(curr[:, :, :, 1:] - curr[:, :, :, :-1]).mean()
        dy = torch.abs(curr[:, :, 1:, :] - curr[:, :, :-1, :]).mean()
        edge_mean = float(((dx + dy) * 0.5).item())
        smooth = F.avg_pool2d(curr, kernel_size=5, stride=1, padding=2)
        highfreq_mean = float(torch.abs(curr - smooth).mean().item())
        roi_jitter = self._roi_jitter(rois)
        return {
            "frame_delta": frame_delta,
            "diff_std": diff_std,
            "edge_mean": edge_mean,
            "highfreq_mean": highfreq_mean,
            "roi_jitter": roi_jitter,
            "roi_count": float(len(rois)),
        }

    def _evaluate(
        self,
        sample: dict[str, float],
        temporal: dict[str, Any],
        motion: dict[str, Any],
    ) -> dict[str, Any]:
        samples = list(self.samples)
        enough = len(samples) >= self.min_window
        frame_deltas = [item["frame_delta"] for item in samples if item["frame_delta"] > 0.0]
        edge_values = [item["edge_mean"] for item in samples]
        highfreq_values = [item["highfreq_mean"] for item in samples]
        roi_jitters = [item["roi_jitter"] for item in samples]

        repeated_ratio = self._ratio_below(frame_deltas, self.repeated_diff_threshold)
        low_motion_ratio = self._ratio_below(frame_deltas, self.low_motion_threshold)
        mean_delta = self._mean(frame_deltas)
        mean_edge = self._mean(edge_values)
        mean_highfreq = self._mean(highfreq_values)
        highfreq_std = self._std(highfreq_values)
        roi_jitter_mean = self._mean(roi_jitters)
        diff_std_mean = self._mean([item["diff_std"] for item in samples])

        repeat_score = self._ramp(repeated_ratio, 0.35, 0.75)
        oversmooth_score = 0.5 * self._ramp(
            self.low_edge_threshold - mean_edge, 0.0, self.low_edge_threshold
        )
        oversmooth_score += 0.5 * self._ramp(
            self.low_edge_threshold - mean_highfreq, 0.0, self.low_edge_threshold
        )
        synthetic_motion_score = self._ramp(mean_delta, 0.012, 0.018) * max(
            self._ramp(self.low_edge_threshold - mean_edge, 0.0, self.low_edge_threshold),
            self._ramp(self.low_edge_threshold - mean_highfreq, 0.0, self.low_edge_threshold),
        )
        flicker_score = self._ramp(
            highfreq_std, self.flicker_threshold, self.flicker_threshold * 2.5
        )
        jitter_score = self._ramp(
            roi_jitter_mean, self.roi_jitter_threshold, self.roi_jitter_threshold * 3.0
        )
        temporal_score = self._ramp(float(temporal.get("change_t", 0.0)), 0.06, 0.18)
        local_motion_score = self._ramp(float(motion.get("local_max_ratio", 0.0)), 0.45, 0.85)

        handcrafted_p_synth = min(
            1.0,
            0.28 * repeat_score
            + 0.22 * oversmooth_score
            + 0.30 * synthetic_motion_score
            + 0.10 * flicker_score
            + 0.06 * jitter_score
            + 0.02 * temporal_score
            + 0.02 * local_motion_score,
        )

        # --- Task 6.4 — Synth_Classifier clip-level override --------------
        # Push the 5 retained raw features for this evaluation onto the
        # clip-classifier rolling buffer. ``score_oversmooth`` corresponds
        # to the ``oversmooth`` entry of ``source_authenticity_scores``;
        # Task 6.3 reads that exact same key via
        # ``analyze_synth_features.flatten_feature_block`` so training and
        # inference consume the same definition.
        raw_clip_features = {
            "highfreq_std": float(highfreq_std),
            "highfreq_mean": float(mean_highfreq),
            "edge_mean": float(mean_edge),
            "diff_std_mean": float(diff_std_mean),
            "score_oversmooth": float(oversmooth_score),
        }
        self._classifier_rolling.append(raw_clip_features)

        classifier_buffer_ready = len(self._classifier_rolling) >= self.classifier_window
        classifier_available = classifier_buffer_ready and self.clip_classifier is not None
        classifier_p_synth: float | None = None
        classifier_output: dict[str, Any] | None = None
        if classifier_available:
            # Always score when the artifact is loaded AND the window is
            # full, regardless of the rollout gate. This matches the
            # Static_Media_Classifier pattern from Task 5.4: shadow
            # scoring is always emitted so offline replay / event
            # evidence can attribute decisions, while the gate controls
            # whether the score actually feeds back into ``p_synth``.
            clip_features = self._aggregate_classifier_features()
            classifier_output = self.clip_classifier.compute(clip_features)  # type: ignore[union-attr]
            classifier_p_synth = float(classifier_output["classifier_p_adv"])

        classifier_active = classifier_p_synth is not None and self.classifier_enabled
        p_synth = float(classifier_p_synth if classifier_active else handcrafted_p_synth)
        # --- end Task 6.4 insertion --------------------------------------

        clip_suspicious = bool(enough and p_synth >= self.threshold)
        self.warning_history.append(1 if clip_suspicious else 0)
        confirmed = enough and sum(self.warning_history) >= self.warning_trigger_count
        if confirmed:
            self.hold_remaining = self.hold_frames
        elif self.hold_remaining > 0:
            self.hold_remaining -= 1

        warning = bool(confirmed or self.hold_remaining > 0)
        reasons: list[str] = []
        if repeat_score >= 0.6:
            reasons.append("重复帧/帧差过低")
        if oversmooth_score >= 0.6:
            reasons.append("画面过度平滑")
        if synthetic_motion_score >= 0.6:
            reasons.append("过平滑区域存在异常帧间变化")
        if flicker_score >= 0.6:
            reasons.append("细节闪烁异常")
        if jitter_score >= 0.6:
            reasons.append("目标轨迹抖动")
        if classifier_active and not reasons:
            reasons.append("Synth_Classifier clip 级告警")
        if not reasons and p_synth >= self.threshold:
            reasons.append("clip级统计异常")

        return {
            "source_authenticity_enabled": True,
            "source_authenticity_evaluated": True,
            "source_authenticity_available": bool(enough),
            "p_synth": float(p_synth),
            "source_authenticity_warning": warning,
            "source_authenticity_confirmed": bool(confirmed),
            "source_authenticity_clip_suspicious": clip_suspicious,
            "source_authenticity_hold_remaining": int(self.hold_remaining),
            "source_authenticity_reason": "、".join(reasons),
            "source_authenticity_window_size": len(samples),
            "source_authenticity_repeated_ratio": float(repeated_ratio),
            "source_authenticity_low_motion_ratio": float(low_motion_ratio),
            "source_authenticity_frame_delta_mean": float(mean_delta),
            "source_authenticity_edge_mean": float(mean_edge),
            "source_authenticity_highfreq_mean": float(mean_highfreq),
            "source_authenticity_highfreq_std": float(highfreq_std),
            "source_authenticity_diff_std_mean": float(diff_std_mean),
            "source_authenticity_roi_jitter": float(roi_jitter_mean),
            "source_authenticity_frame_delta": float(sample["frame_delta"]),
            "source_authenticity_scores": {
                "repeat": float(repeat_score),
                "oversmooth": float(oversmooth_score),
                "synthetic_motion": float(synthetic_motion_score),
                "flicker": float(flicker_score),
                "roi_jitter": float(jitter_score),
                "temporal": float(temporal_score),
                "local_motion": float(local_motion_score),
            },
            # Task 6.4 shadow fields — always stamped (see docstring).
            # ``classifier_p_synth`` is 0.0 when the classifier has not been
            # scored yet (artifact absent, rolling buffer not yet full, etc.);
            # use ``classifier_available`` below to tell "no score" apart
            # from "scored to exactly 0.0".
            "source_authenticity_handcrafted_p_synth": float(handcrafted_p_synth),
            "source_authenticity_classifier_p_synth": (
                0.0 if classifier_p_synth is None else float(classifier_p_synth)
            ),
            "source_authenticity_classifier_enabled": bool(self.classifier_enabled),
            "source_authenticity_classifier_available": bool(classifier_available),
            "source_authenticity_classifier_active": bool(classifier_active),
            "source_authenticity_classifier_window": int(self.classifier_window),
            "source_authenticity_classifier_buffer_size": len(self._classifier_rolling),
            "source_authenticity_classifier_artifact": (
                ""
                if classifier_output is None
                else str(classifier_output.get("classifier_artifact", ""))
            ),
            "source_authenticity_classifier_kind": (
                ""
                if classifier_output is None
                else str(classifier_output.get("classifier_kind", ""))
            ),
            "source_authenticity_classifier_threshold": (
                0.0
                if classifier_output is None
                else float(classifier_output.get("classifier_threshold", 0.0))
            ),
            "source_authenticity_backend": "gpu_clip_stats",
        }

    def _aggregate_classifier_features(self) -> dict[str, float]:
        """Build the clip-level 15-dim feature dict from the rolling buffer.

        Matches ``tools.train_synth_classifier.clip_feature_names``:
        ``<raw>_<agg>`` for each raw feature × each aggregation
        (``mean``/``median``/``std``). Standard deviation uses the
        population formula (``pstdev``); the training script likewise
        uses :func:`statistics.pstdev` so the two paths agree bit-for-bit
        on single-video windows.

        Raises
        ------
        RuntimeError
            If invoked while the buffer is not full enough for the
            contract. Callers must gate on ``classifier_available``
            before calling.
        """
        buffer_size = len(self._classifier_rolling)
        if buffer_size < self.classifier_window:
            raise RuntimeError(
                "Synth_Classifier rolling buffer is not full: "
                f"{buffer_size}/{self.classifier_window}"
            )
        # Slice the last ``classifier_window`` samples so the classifier
        # always sees a fixed-length clip. Using ``list(self._classifier_rolling)``
        # is O(N) but N is bounded by ``classifier_window`` (default 60),
        # so the cost is negligible compared to the torch frame work.
        buffer = list(self._classifier_rolling)[-self.classifier_window :]
        features: dict[str, float] = {}
        for raw_name in self._CLASSIFIER_FEATURE_NAMES:
            values = [float(entry[raw_name]) for entry in buffer]
            agg_mean = float(fmean(values)) if values else 0.0
            agg_median = float(median(values)) if values else 0.0
            agg_std = float(pstdev(values)) if len(values) >= 2 else 0.0
            features[f"{raw_name}_mean"] = agg_mean
            features[f"{raw_name}_median"] = agg_median
            features[f"{raw_name}_std"] = agg_std
        return features

    def _roi_jitter(self, rois: list[ROI]) -> float:
        current: dict[str, tuple[float, float, float, float]] = {}
        movements: list[float] = []
        for roi in rois:
            if roi.label not in {"person", "helmet", "head"}:
                continue
            x1, y1, x2, y2 = roi.bbox
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            sig = (
                (x1 + x2) / 1280.0,
                (y1 + y2) / 1280.0,
                w / 640.0,
                h / 640.0,
            )
            current[roi.roi_id] = sig
            previous = self.prev_roi_signature.get(roi.roi_id)
            if previous is None:
                continue
            movements.append(sum(abs(sig[i] - previous[i]) for i in range(4)) / 4.0)
        self.prev_roi_signature = current
        return self._mean(movements)

    @staticmethod
    def _ratio_below(values: list[float], threshold: float) -> float:
        if not values:
            return 0.0
        return sum(1 for value in values if value <= threshold) / len(values)

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @classmethod
    def _std(cls, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = cls._mean(values)
        return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)

    @staticmethod
    def _ramp(value: float, start: float, end: float) -> float:
        if end <= start:
            return 1.0 if value >= end else 0.0
        return min(1.0, max(0.0, (value - start) / (end - start)))

    @staticmethod
    def _empty(enabled: bool = True) -> dict[str, Any]:
        return {
            "source_authenticity_enabled": bool(enabled),
            "source_authenticity_evaluated": False,
            "source_authenticity_available": False,
            "p_synth": 0.0,
            "source_authenticity_warning": False,
            "source_authenticity_confirmed": False,
            "source_authenticity_clip_suspicious": False,
            "source_authenticity_hold_remaining": 0,
            "source_authenticity_reason": "",
            "source_authenticity_window_size": 0,
            "source_authenticity_repeated_ratio": 0.0,
            "source_authenticity_low_motion_ratio": 0.0,
            "source_authenticity_frame_delta_mean": 0.0,
            "source_authenticity_edge_mean": 0.0,
            "source_authenticity_highfreq_mean": 0.0,
            "source_authenticity_highfreq_std": 0.0,
            "source_authenticity_diff_std_mean": 0.0,
            "source_authenticity_roi_jitter": 0.0,
            "source_authenticity_frame_delta": 0.0,
            "source_authenticity_scores": {},
            # Task 6.4 shadow fields — default state (no classifier yet).
            "source_authenticity_handcrafted_p_synth": 0.0,
            "source_authenticity_classifier_p_synth": 0.0,
            "source_authenticity_classifier_enabled": False,
            "source_authenticity_classifier_available": False,
            "source_authenticity_classifier_active": False,
            "source_authenticity_classifier_window": 0,
            "source_authenticity_classifier_buffer_size": 0,
            "source_authenticity_classifier_artifact": "",
            "source_authenticity_classifier_kind": "",
            "source_authenticity_classifier_threshold": 0.0,
            "source_authenticity_backend": "gpu_clip_stats",
        }
