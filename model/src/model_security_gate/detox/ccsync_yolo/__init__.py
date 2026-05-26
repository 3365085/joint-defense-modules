"""CCSync on YOLO: real-detector integration.

Wires the closed-form CCSync purification onto a frozen YOLO model.
The pipeline:

  1. Forward trigger pool images through the poisoned YOLO with hooks
     on cls-head modules; collect (B*H*W, C) flattened activations
     per FPN scale.
  2. Forward clean target-absent images through the same model;
     collect equivalent activations.
  3. For each scale, compute excess correlation S = ρ_trig - ρ_clean,
     sync degree σ = sum(positive S above τ), and per-channel α
     via softmax-based concentration.
  4. Find the matching cls-head Conv2d in BOTH W_poisoned AND W_clean
     (the project keeps clean-baseline checkpoints for every family),
     interpolate weights per output channel using α.
  5. Save the per-scale α vectors and the purified merged checkpoint.

This is fundamentally different from the previous CCS (channel
do-intervention at activations) and CELS-OD/JES (post-head energy
surgery): CCSync **modifies weights**, producing a STATIC purified
.pt with no inference overhead.

Patent positioning: independent claim 2-of-five.  Mathematically,
CCSync ⊃ Backbone-Soup (uniform σ recovers Soup at α=alpha_max).
"""
from .pipeline import (
    CCSyncYoloConfig,
    CCSyncYoloResult,
    purify_yolo_with_ccsync,
)

__all__ = [
    "CCSyncYoloConfig",
    "CCSyncYoloResult",
    "purify_yolo_with_ccsync",
]
