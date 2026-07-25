# v2_D miss audit — catalog stages + pixel-level segmentation response

**Date:** 2026-07-25. Successor to MISS_AUDIT.md (2026-06-11, v1-era) and companion to
SLOWBAND_AUDIT.md. Evidence: (A) run_lambda 82-field injection campaign (v2_D catalogs,
confirmed via detect slurm → active pipeline → models/current → v2_D symlinks);
(B) `probe_seg_response.py` — 3,600 injected trails (7 SNR × 7 length grid, 150 live
DM-53195 panels, run_dev manifests) through the EXACT production chain
(`predict_panel_overlap_3ch_full` → `panel_to_catalog_rows`, rl=zeros, low cnn_thr so
sub-threshold candidates are visible); (C) same probe on real 0630 embargo panels.

## A. Full-bin stage table (run_lambda, per ff-recoverable object, frozen-op chain)

Stages: A never detected (any score) · B <2 visits at S≥0.6 · C <2 visits at S≥0.8 ·
E mfsnr≥5 kills all pairs · F χ²≤5 kills all pairs · G recovered.

| | len 6.25–12.5 px | len 12.5–25 | len 25–50 |
|---|---|---|---|
| SNR 2–5 | A 53 B 27 C 7 E 9 F 4 **G 0.2** | A 49 B 27 C 9 E 9 F 5 **G 3.0** | A 55 B 26 C 7 E 6 F 4 **G 2.7** |
| SNR 5–10 | A 20 B 24 C 9 E 24 F 23 **G 0.5** | A 15 B 19 C 10 E 18 F 20 **G 17.7** | A 18 B 27 C 9 E 7 F 14 **G 26.0** |

(percent of n=488/507/447/408/402/366)

- **SNR 2–5 (≈55% of the bin) is ~80% stage-1 (A+B) at EVERY length.** The sub-5σ
  population — the pipeline's design target — dies before the linker, length-independent.
- The linker-gate deaths (E+F) are an SNR 5–10 phenomenon, worst for short trails
  (46% at 6.25–12.5 px vs 21% at 25–50) — mechanism in SLOWBAND_AUDIT.md
  (de-bias endpoint collapse → dspeed pre-gate).

## B. Pixel-level response probe (dev panels, production v2_D chain)

Per-sighting outcomes for slow trails (L ≤ 12.5 px):

| | zero-response (pmax<0.1) | weak (0.1–0.5) | seg fires, lost downstream | production-detected |
|---|---|---|---|---|
| SNR 2–5 | **50%** | 5% | 18% | 27% |
| SNR 5–10 | 16% | 2% | 9% | 73% |

- Of the "seg fires (pmax≥0.5) but lost" class at SNR 2–5: **92% produce a candidate that
  the stage-2 focal-cutout CNN scores just BELOW the 0.5 floor** (median 0.39, p75 0.44);
  8% lose the candidate stage; 0% die at the nn_pmax gate. → Stage-2 is a concentrated,
  cheap fine-tune/recalibration target, separate from the 50% true seg-blindness.
- Detection prob is essentially length-independent at fixed SNR (surface-brightness
  scaling already encoded in mag): SNR 2 ≈ 9–20% prod, SNR 4 ≈ 39–58%, SNR 6 ≈ 61–76%,
  SNR 10 ≈ 81–90%.
- mf_snr measures LOW vs nominal: at snr_target 6 only 50% of detected trails pass
  mf_snr≥5 (median exactly 5.0); at snr_target 4, 33%. The mfsnr≥5 2v gate is
  effectively an SNR≈7 gate per sighting.

### Length bias (ends-bloom) on dev panels

| true L (px) | raw−L med (all) | SNR≤4 | SNR≥8 | len_db<6 |
|---|---|---|---|---|
| 6.25 | +6.35 | +7.78 | +5.93 | 59% |
| 8 | +5.46 | +7.24 | +5.04 | 47% |
| 10 | +4.76 | +6.29 | +4.42 | 26% |
| 12.5 | +4.89 | +5.59 | +4.55 | 9% |
| 16 | +5.04 | +3.50 | +5.44 | 4% |
| 25 | +4.41 | −0.69 | +5.46 | 7% |
| 40 | +0.08 | −13.80 | +3.04 | 7% |

- Dev panels REPRODUCE the committed mflen fit (+5–6 px short-end bloom) — the frozen
  de-bias constants are correct **for dev-like panels**. No dependence on panel noise
  tercile (+5.3/+5.7/+5.9) — contrast is NOT the driver of the run_lambda discrepancy.
- The run_lambda anomaly is now characterized: strict-matched unclipped dets give
  `raw = 1.03 + 1.239·L` on lambda vs `raw ≈ 5.2 + 1.20·L` on dev — **same slope,
  ~4 px less constant bloom**. Half-length-rendering refuted (slope would be ~0.6).
  Something suppresses the fixed per-end footprint extension on the dense ecliptic
  lambda fields (competing background structure in the seg map is the leading suspect);
  either way the de-bias intercept is field-population-dependent.
- Long faint trails TRUNCATE (L=40, SNR≤4: −13.8 px) — the footprint breaks up at low
  surface brightness; a single global linear de-bias cannot represent this
  (bloom-short / truncate-long, SNR-dependent).

## C. Reality check: same probe on real 0630 embargo panels

3,600 trails on 150 REAL 0630 embargo panels (S3, job 33065125). Real panels are noisier
(MAD sigma med 53.9 vs 36.7 on dev) — same nominal SNR is harder.

**Miss split, slow trails (L≤12.5):**

| | zero-response | weak | seg fires, stage-2 kills | production-detected |
|---|---|---|---|---|
| SNR 2–5 | **66%** | 5% | 14% (88% cand, score med 0.37) | 15% |
| SNR 5–10 | 23% | 3% | 11% | 63% |

Both fine-tune targets CONFIRMED on production-realistic panels; the stage-2
just-below-floor signature is identical (score med 0.37).

**Length bias on real panels** (med raw−L): +7.1 / +6.1 / +5.4 / +5.0 / +5.0 / +4.8 / −4.4
for L = 6.25…40; snr≤4 at L≥25 truncates hard (−3.3 / −20.0).

- **Short-end bloom on real panels MATCHES dev (+6–7 px)** — the frozen de-bias constants
  are approximately right in production; run_lambda's +2 px was the outlier (dense-field
  campaign artifact). Consequently real-panel `len_db` for slow trails is only ~0.2–2 px
  low (med 6.07 at true 6.25) — **endpoints do NOT collapse on real data**; the
  SLOWBAND_AUDIT endpoint-collapse catastrophe is largely a lambda-evidence artifact, and
  the dspeed/χ² physics should function for real detected slow trails.
- The long-faint truncation (−20 px at L=40, SNR≤4) is REAL and un-representable by the
  global linear de-bias.
- **mf_snr is the binding real-data linker gate**: only 23–33% of detected trails at
  nominal SNR ≤6 pass mf_snr≥5 (SNR 8: 50%, SNR 10: 70%) — on real panels mfsnr≥5 acts
  as a per-sighting SNR≈8–9 gate, squarely inside the design bin.

## Implications (recommended order)

**Binding constraint (user, 2026-07-25): long-trail / current performance must be SAME OR
BETTER under any change.** Every lever below ships with non-regression gates: fast-band C
at the frozen op (12.5–25 px and 25–50 px) >= current, overall C >= 6.07%, purity floor
>= 75% still met, blind CLEAN-24 confirm not below the frozen headline, and the 12
production 0630 alerts preserved by member position. A candidate that trades long for
short is rejected regardless of its slow-end gain.

1. **Fine-tune round, two distinct targets** (formal path: build_ft-style dataset →
   stage-1 + stage-2 → train_and_validate re-derivation), confirmed on real panels:
   a. stage-2 rescue of the seg-fires-but-score-0.33–0.45 population (SNR 2–5, short) —
      cheapest single lever, 14–18% of the sub-5σ slow band per sighting;
   b. stage-1 seg fine-tune targeting the 50–66% zero-response at SNR 2–5 (hard: may be
      close to the information floor; measure what a realneg-style faint-short round buys).
2. **mfsnr≥5 is the binding real-data linker gate** (per-sighting SNR≈8–9 on real panels).
   A redesign now has a measured real-panel basis, but its FP cost must still come from an
   injection campaign — and lambda evidence showed relaxation without a measurement fix
   buys nothing THERE; re-evaluate on evidence whose length/mfsnr behavior matches real
   panels (dev-like, or a fresh real-panel campaign).
3. **De-bias**: frozen constants are approximately right for real short trails (bloom
   +6–7 px); the real defect is the long-faint truncation (un-representable by a global
   linear). Any overhaul = joint (length_raw, mf_snr) correction, gated by the
   no-regression constraint for L≳16 px.
4. **Evidence-transfer caveat (major)**: run_lambda — the formal threshold evidence — has
   anomalous length measurement (intercept 1.0 vs 5.2/11.4 dev/real short-end); its
   slow-band E/F stage attribution (endpoint collapse) does not transfer to real panels.
   Real-data slow-band bottlenecks: detection (66% zero-response) then mfsnr. Weigh this
   when reading SLOWBAND_AUDIT.md's linker-gate percentages.

## Tooling

`probe_seg_response.py` (+ `probe_seg.slurm`, RUN/OUT/PANELS env): inject controlled
SNR×length grids into live diffims (posix or s3), run the active pipeline exactly as
production, record per-trail seg response at truth + catalog outcome + length bias.
Reusable for any panel set with a manifest (fits_path column).
