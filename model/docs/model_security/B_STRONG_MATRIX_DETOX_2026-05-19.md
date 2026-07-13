# B strong poison detox matrix

Date: 2026-05-19

## Scope

This document records the latest B-source strong non-patch poison validation.
All four B strong poison checkpoints reached the usable hard-regression bar:

```text
poison external ASR = 79/79 = 100%
external hard-suite images = 79 target-present samples
strict ASR ceiling target = Wilson 95% upper bound < 5%
```

The detox side now includes explicit `invisible` trigger support in
`model_security_gate/detox/asr_aware_dataset.py`, so bounded full-image noise
can be used in ASR-aware training and internal ASR regression rather than only
in external evaluation.

## Result Matrix

| Poison attack | Poison ASR | Selected defended model | External ASR | Strict ceiling | Internal ASR | Clean mAP50-95 | mAP50-95 drop |
|---|---:|---|---:|---|---:|---:|---:|
| `b_sig_lowfreq_hi_oda` | `79/79 = 100%` | `runs/B_sig_lowfreq_hi_clean_anchor_weight_soup_2026-05-18/candidates/anchor01_mask_bd_v2_clean_baseline_alpha1p0.pt` | `0/79 = 0%` | PASS | `1/99 = 1.01%` | `0.4459385267` | `0.182 pp` |
| `b_sig_multiperiod_oda` | `79/79 = 100%` | `runs/B_sig_multiperiod_weight_soup_2026-05-18/candidates/anchor01_mask_bd_v2_clean_baseline_alpha0p8.pt` | `0/79 = 0%` | PASS | `1/99 = 1.01%` | `0.4470528956` | `-2.625 pp` |
| `b_warp_lowfreq_strong_combo_oda` | `79/79 = 100%` | `runs/B_warp_lowfreq_combo_detox_full_2026-05-18/02_cycle_01_phase_03_clean_recovery/ultralytics_recovery/clean_recovery/weights/best.pt` | `0/79 = 0%` | PASS | `3/99 = 3.03%` | `0.4213206563` | `2.539 pp` |
| `b_invisible_noise_hi_oda` | `79/79 = 100%` | `runs/B_invisible_noise_hi_weight_soup_2026-05-18/candidates/anchor01_mask_bd_v2_clean_baseline_alpha0p8.pt` | `0/79 = 0%` | PASS | `1/99 = 1.01%` | `0.4498148427` | `-0.312 pp` |

Negative mAP drop means the selected model scored above the poisoned baseline
on clean validation.

## Important Interpretation

The B strong matrix is solved under the current pipeline assumptions, but the
solution modes are different:

- `b_warp_lowfreq_strong_combo_oda` is the strongest pure Hybrid-PURIFY closure:
  the selected checkpoint comes from Hybrid-PURIFY plus clean recovery and stays
  inside both external and internal ASR gates.
- `b_invisible_noise_hi_oda` needed Hybrid-PURIFY first, then clean-anchor
  weight soup. Raw Hybrid reduced external ASR from `100%` to `2/79 = 2.53%`,
  but internal ASR remained `12.12%`; clean-anchor soup closed it to `0/79`
  external and `1.01%` internal.
- `b_sig_multiperiod_oda` was externally closed by raw Hybrid, but clean-anchor
  soup was needed to reduce internal ASR from `8.08%` to `1.01%` and recover
  clean mAP.
- `b_sig_lowfreq_hi_oda` has a strict-pass fallback at `alpha=1.0`, which is
  effectively the clean anchor. This is valid as a clean-anchor replacement
  baseline, but it should not be described as a pure poisoned-checkpoint repair.

## Evidence Files

Low-frequency:

```text
runs/B_sig_lowfreq_hi_clean_anchor_weight_soup_2026-05-18/alpha1p0_external/external_hard_suite_asr.json
runs/B_sig_lowfreq_hi_clean_anchor_weight_soup_2026-05-18/alpha1p0_strict_ceiling/strict_asr_ceiling_plan.json
runs/B_sig_lowfreq_hi_clean_anchor_weight_soup_2026-05-18/alpha1p0_internal_asr/asr_regression.json
runs/B_sig_lowfreq_hi_clean_anchor_weight_soup_2026-05-18/alpha1p0_clean_metrics.json
```

Multi-period:

```text
runs/B_sig_multiperiod_weight_soup_2026-05-18/last_mile_weight_soup_manifest.json
runs/B_sig_multiperiod_weight_soup_2026-05-18/selected_strict_ceiling/strict_asr_ceiling_plan.json
runs/B_sig_multiperiod_weight_soup_2026-05-18/selected_internal_asr/asr_regression.json
```

Warp plus low-frequency combo:

```text
runs/B_warp_lowfreq_combo_detox_full_2026-05-18/hybrid_purify_manifest.json
runs/B_warp_lowfreq_combo_detox_full_2026-05-18/selected_strict_ceiling/strict_asr_ceiling_plan.json
```

Invisible noise:

```text
runs/B_invisible_noise_hi_detox_full_2026-05-18/hybrid_purify_manifest.json
runs/B_invisible_noise_hi_weight_soup_2026-05-18/last_mile_weight_soup_manifest.json
runs/B_invisible_noise_hi_weight_soup_2026-05-18/selected_strict_ceiling/strict_asr_ceiling_plan.json
runs/B_invisible_noise_hi_weight_soup_2026-05-18/selected_internal_asr/asr_regression.json
```

## New Configs

```text
configs/b_frontier_sig_multiperiod_detox.yaml
configs/b_frontier_sig_multiperiod_asr_regression.yaml
configs/b_frontier_warp_lowfreq_combo_detox.yaml
configs/b_frontier_warp_lowfreq_combo_asr_regression.yaml
configs/b_frontier_invisible_noise_hi_detox.yaml
configs/b_frontier_invisible_noise_hi_asr_regression.yaml
```

## Current Open Caveat

The current pipeline can solve the B strong non-patch matrix when a trusted
clean anchor is available. For a stricter "repair the poisoned checkpoint
without replacing it with a clean anchor" claim, the remaining research target
is the low-frequency family: the best pure/near-poison repair closes external
ASR, but the strongest strict result currently uses the clean-anchor fallback.
