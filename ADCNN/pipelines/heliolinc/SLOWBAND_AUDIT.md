# Slow-band (1–2 deg/day) audit — len_db floor refuted, root cause traced to length de-bias

**Date:** 2026-07-24. **Evidence:** run_lambda 82-field injection campaign (the formal threshold
evidence), regenerated floor-4 caches, per-object kill-chain census. All numbers below are
reproducible from committed caches + `outputs/runs/run_lambda`.

## Verdict

1. **`len_db_min` 6→4 is REFUTED.** Regenerated all 82 evidence caches at floor 4
   (`outputs/runs/run_lambda/_nomfsnr_cache_len4/`, isolated from the frozen caches;
   default-floor path proven byte-identical to the committed cache on field 74).
   The pre-declared selection rule picks the SAME op (S=0.80 / mfsnr=5.0);
   completeness identical (C=6.073%, 159/2618); purity +1.05pp as a side-effect
   (extra 4–6 px dets raise recurrence counts → `recur<2` kills 17 more FP pairs).
   The floor-4 evidence contains **zero TP pairs with min_len<6 at any score/mfsnr**
   (778 band pairs, all FP). **The frozen op stands; no op files changed.**

2. **The slow band is structurally dead** at the frozen op (true trail 6.25–12.5 px
   = 1–2 deg/day; 896 of 2618 ff-recoverable = 34% of the designed bin):

   | true band | recoverable | recovered | C |
   |---|---|---|---|
   | 6.25–9 px (1.0–1.44 °/d) | 493 | 0 | 0.00% |
   | 9–12.5 px (1.44–2 °/d) | 403 | 1 | 0.25% |
   | 12.5–25 px (2–4 °/d) | 909 | 65 | 7.15% |
   | 25–50 px (4–8 °/d) | 813 | 93 | 11.44% |

   The completeness cliff at ~12.5 px is exactly the `dspeed` pre-gate pass boundary
   (see mechanism below).

## Kill chain (all 896 slow-band objects, floor 0, production art/recur cuts)

| stage | n | % |
|---|---|---|
| never detected (any score) | 341 | 38.1% |
| <2 visits at S≥0.6 | 229 | 25.6% |
| <2 visits at S≥0.8 | 71 | 7.9% |
| no pair in rate window | 1 | 0.1% |
| mfsnr≥5 kills all pairs | 140 | 15.6% |
| χ²≤5 kills all pairs | 111 | 12.4% |
| recovered | 3 | 0.3% |

**64% is stage-1** (detection/scoring) — the linker never sees it.
The 251 linker-gate deaths (28%, = +9.6pp overall-C ceiling) all trace to ONE mechanism:

## Mechanism: the MF_LEN de-bias collapses slow-band endpoints

- `detect_night.py` builds trail endpoints as center ± `len_db`/2 along beta, so the
  endpoint span (and hence `pair_chi2`'s trail velocity `tv()`) IS the de-biased length.
- On run_lambda injections, slow-band SNR 2–10 truth dets at S≥0.8 measure
  **`length_raw ≈ true + 2.0 px`** (strict 3-px positional match, n=182) — but the frozen
  de-bias assumes the run_ft bloom (`raw ≈ true + 5–6 px` at the short end, offset 7.67).
  Subtracting 7.67 crushes them: **len_db med 1.81 px for true ~8.5 px trails; 38% clip to ~0**.
- Collapsed endpoints → trail speed ≈ 0 → `dspeed = |0 − rate|/rate ≈ 1.0` → the 0.6
  pre-gate auto-rejects (dspeed>0.6 for 99% of the 251's best pairs; p25 = 1.00 is the
  zero-speed signature). Rescue tests that rescale endpoint magnitude (linear or quadratic
  de-bias) rescue **0/251** — the length information is already destroyed in the catalog;
  PA is also noisy at these lengths (dpa_tm>20° for 47%).
- Relaxing the mfsnr gate instead is measured useless: on the χ²-passing (cached) pairs,
  every length-aware mfsnr rule tried adds ≤5 TP pairs while adding 32–189 FP
  (P 76.9% → 66.7–74.9%). χ² (via the collapsed velocity) is the binding gate, not mfsnr.

## Open calibration question: the two injectors disagree

`ADCNN/calibration/mflen_fit_pairs.csv` (run_ft rendering) shows short-trail bloom
+5–6 px; run_lambda injections show +2 px (SNR-independent, verified up to SNR 31).
The frozen de-bias constants (7.67 / 0.9425) fit run_ft and destroy run_lambda's short
end. Which rendering matches real LSSTCam short trails is UNRESOLVED — diff the trail
renderers (`build_ft_dataset.py` vs `inject_trails.py`) on the `trail_length` definition
before touching the de-bias. Note the ends-bloom is also nonlinear (quadratic fit
`raw = 5.21 + 1.20L − 0.0052L²` halves the short-end residual on the fit pairs).

## Implications / recommended order

1. **Stage-1 is the dominant slow-band lever (64%)** → rerun the miss audit on v2_D
   catalogs (MISS_AUDIT.md predates v2_D) and fold slow-band misses into the next
   fine-tune round.
2. **Resolve the injector disagreement**, then decide whether the de-bias (and the
   endpoint construction) needs an SNR/length-aware form. Any change goes through
   `calibrate_mflen` + full cache regen + `threshold_selection` re-derivation.
3. Even with perfect lengths, short-pair 2v physics is weak (PA ±18° at 6 px; the
   measured FP:TP ratio in the sub-6 px band is 390:0 at χ²≤5) — slow-band *discovery*
   realistically needs 3v confirmation, consistent with the 2v-=-alert-tier scope.
4. Do NOT loosen mfsnr or χ² gates for short pairs — measured to buy ~nothing at
   material purity cost.

## Tooling added

`exact_lowS_pairs.py --len-db-min <F>` regenerates evidence caches at a non-default
floor into an isolated `_nomfsnr_cache_len<F>/` subdir (standard filenames inside, so
`threshold_selection --cache-dir` reads them unchanged); the frozen floor-6 caches can
never be clobbered. Default floor is byte-identical to the committed caches.
