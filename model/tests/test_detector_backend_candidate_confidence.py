from __future__ import annotations

from defense.module_a.backends.detector_backend import (
    UltralyticsDetectorBackend,
    YoloV5DetectorBackend,
    configured_class_names,
)


def test_ultralytics_candidate_confidence_applies_to_three_class_models():
    backend = UltralyticsDetectorBackend.__new__(UltralyticsDetectorBackend)
    backend.confidence = 0.25
    backend.candidate_confidence = 0.18
    backend.names = {0: "helmet", 1: "head"}

    assert backend._prediction_confidence() == 0.18

    backend.names = {0: "helmet", 1: "head", 2: "person"}

    assert backend._prediction_confidence() == 0.18


def test_yolov5_candidate_confidence_applies_to_three_class_models():
    backend = YoloV5DetectorBackend.__new__(YoloV5DetectorBackend)
    backend.confidence = 0.25
    backend.candidate_confidence = 0.18
    backend.names = {0: "helmet", 1: "head", 2: "person"}

    assert backend._prediction_confidence() == 0.18


def test_candidate_confidence_treats_human_aliases_as_person_capable():
    backend = UltralyticsDetectorBackend.__new__(UltralyticsDetectorBackend)
    backend.confidence = 0.25
    backend.candidate_confidence = 0.18

    for alias in ("person", "worker", "human", "pedestrian"):
        backend.names = {0: "helmet", 1: "head", 2: alias}
        assert backend._prediction_confidence() == 0.18


def test_configured_class_names_accepts_three_class_person_first_mapping():
    config = {"inference": {"class_names": ["person", "head", "helmet"]}}

    assert configured_class_names(config) == {0: "person", 1: "head", 2: "helmet"}


def test_configured_class_names_accepts_string_key_mapping():
    config = {"inference": {"names": {"0": "person", "1": "head", "2": "helmet"}}}

    assert configured_class_names(config) == {0: "person", 1: "head", 2: "helmet"}
