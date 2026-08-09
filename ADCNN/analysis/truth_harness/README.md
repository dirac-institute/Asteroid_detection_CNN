# Truth harness — injection-recovery tooling (2026-08-09)

Measures the same-night 2-visit pipeline against injected truth: movers injected into REAL diffims,
pushed through the REAL detector and linker, then matched back by position.

These scripts live here, not under `outputs/`, because `outputs/` is gitignored — an earlier commit
(e9fafb5) claimed to add them and silently added nothing. Working copies still run from
`outputs/runs/pa_validate/`; this is the tracked source of record.

| script | what it establishes |
|---|---|
| `build_rank_table.py` | alert feature/label table. LABEL HYGIENE: true only if BOTH epochs match the SAME oid (a chance link pairing one injected det with one FP is a WRONG link). FEATURE HYGIENE: no visit-pair properties, since the CV groups ARE visit pairs. |
| `fit_ranker.py` | sign-constrained logistic ranker, fold-fraction budgets, score-inflation stress curve. |
| `audit_op.py` | every op-point cut scored in isolation: kills_true vs kills_fp. |
| `inject_night.py` | cross-night harness (`INJ_RUN`/`INJ_TAG`). |
| `inject_asym.py` | per-epoch detection thresholds (deep vs shallow band). |
| `analyze_asym.py` | faint-fast completeness AND alert volume across the threshold curve. |
| `mf_length.py` | matched-filter trail-length estimator vs the shipped footprint extent. |

## Headline measurements

- **Every co-pointed revisit in the campaign is CROSS-BAND** (i→z, i→r, g→r; not one same-band pair
  on any night), with a measured depth gap of 0.45–0.62 mag. The shallow epoch discards 84–87% of
  the faint-fast movers the deep epoch found, so `snr_t` is a FIRST-EPOCH SNR and the binding
  constraint is the shallower epoch.
- **`rate_hi_2v: 8.0` is harmful**: kills 16.7% of true movers for 6.9% of FPs (ratio 0.41).
- **`chi2_2v_max: 8.0` sits at the MEDIAN of the true-mover chi2 distribution** (7.4), discarding 46%
  before ranking; the bias is by TRAIL LENGTH, not brightness.
- **Ranking is not the bottleneck**: a fitted ranker buys +0.3 pts (t=0.58) inside the shipped gate,
  reproducing the 2026-06-10 82-field "chi2 is a gate, not a weight" result exactly.
- **`--gate-mode any` is REFUTED**: it trades FAST movers for slow ones (gains 65 at 1.6 deg/day,
  loses 131 at 9.0 deg/day) via greedy-claim competition.
- **Matched-filter trail length WORKS** once its near-constant bias is calibrated out: |rate error|
  20.5%→11.9% overall and 14.2%→8.1% for fast movers, truncation 15.4%→1.9%, better in EVERY length
  bin, and the dspeed chi2 penalty on fast movers goes 0.45σ→0.01σ. Robust to a 31% PSF mismatch.
  Measures INPUT QUALITY only — the end-to-end completeness gain is NOT yet measured.
