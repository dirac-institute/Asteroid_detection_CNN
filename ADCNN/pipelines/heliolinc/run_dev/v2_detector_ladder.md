# ADCNN v2 detector-level ladder (dev, 21 fields, @0.5 retention floor) — FINAL

| model | recall all | faint (SNR<10) | SNR2-5 | SNR5-10 | SNR10-31 | dets/panel @0.5 | dets/panel @0.8 |
|---|---|---|---|---|---|---|---|
| v1 (baseline)          | 36.6% | 22.9% | 11% | 37% | 57% | 135 | 7.4 |
| v2_B (lr5e-5, stk0.6)  | 37.9% | 24.1% | 11% | 39% | 58% |  75 | 0.1 |
| v2_Blow (lr2e-5,stk0.6)| 31.7% | 20.1% |  9% | 32% | 49% |  61 | 0.0 |
| v2_D (lr5e-5, stk0.85) | 43.1% | 27.6% | 13% | 44% | 66% |  94 | 0.1 |

WINNER: v2_D (hard-positive oversampling of the stack-found/ADCNN-missed pool — the #251 lever).
- +6.5pt recall overall, +4.7pt faint (22.9->27.6%), gains in EVERY SNR bin, at LOWER load than v1.
- Confirms it's the oversampling, not fine-tuning per se: v2_B (plain) barely moved, v2_Blow (low LR) hurt.
- ALL @0.8 columns structurally-invalidated (v1 stage-2 CNN scores v2 seg channels ~0 -> the chimera).
  The alert-level product metric is UNMEASURABLE for any v2 without a stage-2 refit on v2 outputs.

NEXT DECISION (user): stage-2 refit on v2_D (canonical train_end_to_end flow; leakage-clean cnn panels
from the ~21k unsampled dev panels; op value 0.80 unchanged) -> true alert-level v1-vs-v2_D ladder ->
gate -> ONE blind shot. Without it, v2_D's detector win cannot be converted to an alert-stream number.
