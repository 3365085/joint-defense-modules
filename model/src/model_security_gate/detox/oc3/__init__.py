"""OC3 object-context counterfactual detox algorithms."""
from __future__ import annotations

from .oc3_detox import (
    CandidateBox,
    OC3Config,
    OC3LossTerms,
    OC3Plan,
    OC3Stage,
    OC3Witness,
    build_oc3_plan,
)
from .oc3_trainer_v4 import OC3TrainV4Config, OC3TrainV4Result, train_oc3_adapter_v4

__all__ = [
    "CandidateBox",
    "OC3Config",
    "OC3LossTerms",
    "OC3Plan",
    "OC3Stage",
    "OC3TrainV4Config",
    "OC3TrainV4Result",
    "OC3Witness",
    "build_oc3_plan",
    "train_oc3_adapter_v4",
]
