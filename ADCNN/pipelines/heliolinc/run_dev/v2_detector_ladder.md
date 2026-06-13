# ADCNN v2 detector-level ladder (dev, 21 fields, @0.5 retention floor)

| model | per-sighting all | faint (SNR<10) | SNR2-5 | SNR5-10 | SNR10-31 | dets/panel @0.5 | dets/panel @0.8 |
|---|---|---|---|---|---|---|---|
| v1 (baseline)        | 36.6% | 22.9% | 11% | 37% | 57% | 135 | 7.4 |
| v2_B (lr5e-5, stk0.6)| 37.9% | 24.1% | 11% | 39% | 58% |  75 | 0.1 |

- Stage-1 fine-tune @0.5: SMALL recall gain (+1.3pt all, +1.2pt faint) — NOT the factor-level
  lift the recall^2 math hoped for; the 2-field early read (54->72%) was a high-baseline-night
  artifact, not the 21-field truth.
- Raw load HALVED (135->75 dets/panel) — a real efficiency win.
- @0.8 alert floor COLLAPSES (7.4->0.1 dets/panel): v1 stage-2 CNN scores v2's recalibrated
  segmentation channels near-zero (the chimera). The alert-level question is UNANSWERABLE without
  a stage-2 refit on v2 outputs — structurally, not a tuning issue.
