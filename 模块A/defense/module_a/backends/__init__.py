"""Detector inference backends for Module A."""

from .detector_backend import (
    DetectionFrameResult,
    UltralyticsDetectorBackend,
    create_detector_backend,
)

__all__ = ["DetectionFrameResult", "UltralyticsDetectorBackend", "create_detector_backend"]
