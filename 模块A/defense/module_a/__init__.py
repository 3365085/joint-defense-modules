"""General Module A physical perturbation detector."""

from .calibration import FeatureCalibration
from .detector import ModuleADetector
from .types import ROI, ModuleAInput, ModuleAResult
from .utils import ensure_ultralytics_settings_isolated, module_a_package_root

__all__ = [
    "ROI",
    "FeatureCalibration",
    "ModuleADetector",
    "ModuleAInput",
    "ModuleAResult",
    "ensure_ultralytics_settings_isolated",
    "module_a_package_root",
]
