# #250 — Stack-seeded trail-measurement branch (design, pre-registered)

**Goal.** Exploit the blind-measured complementarity (BLIND_TEST_REPORT.md §3.1): the 5σ stack
recovers ~1,760 injected sightings ADCNN@0.80 misses, at a marginal cost already shown to be 23×
more FP-efficient than relaxing the stack to 4σ. Convert those stack-only detections into
*alert-capable* candidates (trail state + score) and measure whether the FROZEN 2v alert op
(`op_2v_alert.json`, read-only) gains faint-fast completeness over the blind 3.64% at comparable
purity/load. **Measurement only — no threshold, config, or model changes.**

## Why a new component is needed (grounded in the code)

ADCNN's trail state (`mf_length`/`len_db`, `mf_beta`, `mf_snr`) is produced by
`ADCNN/inference/catalog.py:panel_to_catalog_rows` from the **segmentation probability map** — for
stack-only peaks, segmentation is precisely what did not fire, so there is no `prob` to measure on.
The branch therefore needs a SEEDED measurement: a matched-filter scan (or Veres trailed-PSF fit)
applied directly to a cutout around each stack peak.

## Pipeline (per blind field k)

1. **Seed set:** `stack_full_s5_{k}_peaks.csv` peaks farther than 10 px from every
   `adcnn_dets_masked_{k}.csv` detection (the stack-only additions; ~280k dets over 26 fields,
   of which ~1.76k match injected sightings).
2. **Pixels:** re-create the injected panel deterministically (`inject_trails.add_trails`, same
   seed/inject.csv → byte-identical pixels) and cut 64×64 stamps at each seed.
3. **Trail state (seeded MF):** template bank scan on the stamp — PA ∈ [0,180°) in ~7.5° steps,
   L ∈ {6…40} px trailed-PSF templates (PSF FWHM 3.77 px) → best (PA, L), `mf_snr` = peak matched
   filter S/N, `len_db` via the MF_LEN_* de-bias of catalog.py. Endpoints from (x,y) ± L/2 along PA,
   sky via the manifest `wcs_json` WCS (same transform as the whole blind chain).
4. **Score (same stage-2 CNN, unchanged):** `models/cnn_postproc.pt` on the stamp → `score`.
   *Honesty guard, sharpened after reading the input contract:* the CNN's 3 channels are
   `[diffim/σ, seg_prob, seg_agg]` (`cnn_postproc.make_cutouts`) — **two of three channels are
   segmentation outputs**, and stack-only peaks are by definition locations where segmentation
   response was low. Scoring them honestly (re-running segmentation to supply the true low
   prob/agg) is expected to produce low scores *structurally*, not incidentally. If so, the
   measured conclusion is "stack-seeded candidates cannot enter the frozen alert stream without
   model work (a stage-2 variant not conditioned on segmentation response)" — that conclusion goes
   to the #251 audit/retraining objective. It is NOT a license to lower the floor or to score with
   fabricated channels. Order of experiments is therefore: **(A) seeded-MF trail state first**
   (cheap, CPU, segmentation-independent — answers whether stack-only TP additions even carry 2v
   geometry: real trail PA/length vs truth, mf_snr distribution TP vs FP); (B) score measurement
   second (needs a segmentation re-run over inject panels to supply honest ch2/ch3).
5. **Merge:** emit rows in the `adcnn_dets_masked` schema tagged `src=stack`, positional dedup
   (10 px) against the ADCNN catalog, then `mask_flags`-equivalent artifact veto.
6. **Evaluate at the FROZEN op:** rerun the exact pair machinery (`exact_lowS_pairs` conventions)
   on the merged catalog → faint-fast object completeness, injection-set pair purity (T2), alerts
   per field-night, top-N truth fraction — side-by-side with the blind ADCNN-only row (3.64% /
   99.1% off-ecl / 12.5 per night).

## Success / failure criteria (pre-registered)

- **Success:** merged-catalog faint-fast 2v completeness > 3.64% (off-ecl band [2.32–4.68] cleared)
  at injection-set pair purity within the blind off-ecliptic band (≥98%) and alert load ≤ 2× blind.
- **Informative failure modes:** (a) seeded MF gives trail states whose chi2/PA fail the frozen
  gates (stack additions are point-like → no 2v geometry) → complementarity is detection-level only;
  (b) stage-2 scores OOD-low → score floor blocks the branch pending retraining (#251 input).
- Either way the result is reported; nothing is tuned.

## Deliverables

`stack_seed_measure.py` (seeded MF + CNN scoring driver, GPU array over 26 fields),
`stack_seed_merge_eval.py` (merge + frozen-op reduction), results appended to
BLIND_TEST_REPORT.md §10 follow-ups or a STACK_SEEDED_RESULT.md.

---

## RESULT — experiment A (pilot, fields 2+3, 642 panels, 19,050 stack-only peaks)

`run_blind/stack_seed_mf_2_3.csv` (35 tp / 19,015 fp stack-only peaks):

- **Trail geometry of the stack-only TP additions is REAL and usable:** seeded-MF PA error vs truth
  median **4.5°**, 91% within 15° (inside the linker's pa_tol_2v=10° for most). The additions are
  genuine trails — median true length 17.9 px, median snr_target 14.3 (mid-SNR, NOT the faint bin) —
  that segmentation failed to fire on: direct evidence for the §4 domain-shift recall diagnosis.
- **But there is NO per-detection discriminator for the stack-only stream:** seeded mf_snr tp median
  20.5 vs fp median 18.6 — non-separating, structurally: every stack-only peak IS 5σ-significant
  flux, so a flux-template S/N cannot reject them, and the smallest templates degenerate toward
  point-source matching. tp:fp base rate 1:543.
- **Consequence at the frozen op:** admitting the stream would raise per-panel candidate density
  ~7× (9.7 → ~70) and chance-pair volume ~50× (λ∝ρ²) with no compensating gate — destroying the
  alert tier's purity, the very thing the tier exists for. The stage-2 score cannot rescue it
  structurally (2 of 3 input channels are segmentation outputs ≈ 0 at these seeds).

**Verdict (pre-registered failure mode (a)+(b) jointly): the stack-seeded branch CANNOT enter the
frozen alert stream as designed.** The blind-measured stack/ADCNN complementarity is detection-level
only; converting it to alert level requires a per-detection discriminator that does not currently
exist for segmentation-missed trails. That requirement — "can a stage-2 variant scored WITHOUT
segmentation-conditioned channels separate stack-only true trails from 5σ artifacts?" — is exactly
a retraining-objective question and is handed to the **#251 detector miss audit**. No gate, score
floor, or threshold was changed; nothing from this experiment enters the shipped pipeline.
