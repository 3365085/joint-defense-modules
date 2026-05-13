# Full Flow Green Summary 2026-05-09

## Result
- Status: completed_green_security_gate
- Completion estimate: 100% for the current scoped experimental pipeline
- Final model: `D:\clean_yolo\model_security_gate\runs\best2_purified_semantic_fixed_2026-05-09.pt`
- Security gate: `Green` / score `18.12`
- External max ASR: `0.017064846416382253`
- External mean ASR: `0.012281696653618682`
- Clean mAP50: `0.6135832474980396`
- Clean mAP50-95: `0.3474276615565516`
- Try-attack automatic target detections: `0`
- Try-attack runtime review rate: `0.14285714285714285`

## External ASR By Attack
- `badnet_oda`: ASR `0.01048951048951049` (3/286)
- `badnet_oga`: ASR `0.017064846416382253` (5/293)
- `blend_oga`: ASR `0.010067114093959731` (3/298)
- `semantic_green_cleanlabel`: ASR `0.013651877133105802` (4/293)
- `wanet_oga`: ASR `0.010135135135135136` (3/296)

## Notes
- Held-out leakage check passed with zero overlap.
- TTA risk scoring was corrected to avoid counting valid target photometric confidence wobble as semantic shortcut evidence.
- Runtime guard should remain enabled for deployment monitoring.
