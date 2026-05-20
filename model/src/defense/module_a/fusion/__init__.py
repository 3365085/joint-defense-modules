"""Fusion strategies for Module A."""

from .classifier_fusion import TorchLogisticFusion
from .rule_fusion import GPURuleFusion
from .target_anchored import TargetAnchoredAnalyzer

__all__ = ["GPURuleFusion", "TorchLogisticFusion", "TargetAnchoredAnalyzer"]
