"""Attack-zoo builders and specifications for OD backdoor experiments."""

from .specs import (
    AttackSpec,
    PoisonModelSpec,
    default_poison_model_matrix,
    default_t0_attack_specs,
    load_attack_specs,
)
from .yolo_builder import (
    AttackZooBuildConfig,
    AttackZooBuildResult,
    build_attack_zoo_dataset,
)
from .poison_train import (
    PoisonDatasetConfig,
    PoisonDatasetResult,
    build_poison_train_dataset,
)

__all__ = [
    "AttackSpec",
    "PoisonModelSpec",
    "default_poison_model_matrix",
    "default_t0_attack_specs",
    "load_attack_specs",
    "AttackZooBuildConfig",
    "AttackZooBuildResult",
    "build_attack_zoo_dataset",
    "PoisonDatasetConfig",
    "PoisonDatasetResult",
    "build_poison_train_dataset",
    "VariantSpec",
    "ExpandedSuiteResult",
    "default_variant_grid",
    "expand_hard_suite",
]

from .hard_suite_expander import VariantSpec, ExpandedSuiteResult, default_variant_grid, expand_hard_suite
