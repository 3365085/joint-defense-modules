"""AutoDetox closed-loop controller for model-security-gate.

The package turns the previous manual tuning cycle into a deterministic,
auditable controller:

    evidence -> diagnosis -> strategy route -> safe candidate recipes
    -> hard-gated acceptance / rollback.
"""

from .schema import (
    AutoDetoxPlan,
    AutoDiagnosis,
    CandidateRecipe,
    CandidateResult,
    EvidenceEvent,
    GateSpec,
    MetricSnapshot,
)
from .controller import (
    AutoDetoxInputs,
    build_autodetox_plan,
    evaluate_candidate_result,
    execute_plan,
    execute_recipe,
    render_plan_markdown,
    select_best_candidate,
    write_plan,
)
from .diagnosis import diagnose_snapshot
from .gates import candidate_score, evaluate_gate, hard_gate_pass

__all__ = [
    "AutoDetoxInputs",
    "AutoDetoxPlan",
    "AutoDiagnosis",
    "CandidateRecipe",
    "CandidateResult",
    "EvidenceEvent",
    "GateSpec",
    "MetricSnapshot",
    "build_autodetox_plan",
    "candidate_score",
    "diagnose_snapshot",
    "evaluate_candidate_result",
    "evaluate_gate",
    "execute_plan",
    "execute_recipe",
    "hard_gate_pass",
    "render_plan_markdown",
    "select_best_candidate",
    "write_plan",
]
