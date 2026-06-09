# ADCNN threshold decision for same-night NEO discovery — evidence-backed

**Scope:** the science target is the **faint-fast bin** — detection-SNR ∈ [2,10] AND apparent rate > 1 deg/day
(trail ≥ ~6 px). All completeness numbers below are resolved to THIS bin (see
`run_lambda/PHYSICAL_LEVERS_2V.md`, `faint-fast-bin-only-scope`).

## Selection rule (committed methodology)
Report the measured completeness C(S) and false-link rate λ(S); fix the real base rate ρ from
Granvik(2018) × Sorcha (ρ ≈ 0.14 faint-fast same-night-2v movers / deg² / night); compute
**purity(S) = C·ρ / (C·ρ + λ)**; choose the operating point by a **purity-floor** rule (a discovery claim
needs purity above a floor; an alert stream does not). The per-pair 3σ FAR (λ_pair ≤ 1.35×10⁻³) is reported
as the per-link statistical significance but is **base-rate-blind**, so it is NOT the criterion for a
discovery claim — purity is.

## Evidence (82 off-ecliptic injection fields, realistic 34-min WFD cadence; FAR tables = null-MC)

### Completeness is SNR-resolved — the faint-fast bin is NOT the blended number
`run_lambda/lambda_vs_S.csv` reports a *blended* completeness over all injected SNR. That blend is
**dominated by bright (SNR>10) movers** (426/1929 recovered at S=0.80) and is NOT the faint-fast figure.
Resolving to the science bin (within-bin counts, `comp_snr2_5`,`comp_snr5_10`; recoverable totals
1442 + 1176 = **2618** faint-fast injected movers). **This committed CSV is the SHIPPED 2v op**
(`mfsnr_min_2v=10, chi2_2v_max=5, rate∈[1,8]` — verified by re-deriving `field_eval` against the sweep
cache, 5/5 fields):

| score S | faint-fast C (SNR 2–10), shipped op | blended C (all SNR) | per-pair σ (FAR) |
|---|---|---|---|
| 0.80 | **1.7%** (45/2618) | 10.4% | 2.80 |
| 0.85 | 1.6% (42/2618) | 9.7% | 2.99 |
| 0.90 | 1.3% (33/2618) | 8.1% | 3.17 |

Faint-fast 2-visit completeness is **single-digit-percent and monotone-decreasing in S** over the validated
sweep; S=0.80 is the **lowest validated score** and therefore the highest faint-fast completeness of the three
validated points (1.7%). It is a floor, not a peak — lowering S further is unvalidated and would raise
completeness only by also flooding the 2v stream (see below). The monotone-in-S behaviour is mechanism-level
(higher S → fewer detections → fewer recovered pairs) and op-independent, so the score decision below does not
depend on which 2v cut set is used.

### The shipped photometric cut (mfsnr≥10) costs completeness — it does NOT delete the bin
The shipped 2v op applies a matched-filter trail-SNR floor `mfsnr_min_2v=10` as a per-link purity lever
(commit d23bab4). Measured effect on the faint-fast science bin (same 82 fields, S=0.80):

| 2v op | faint-fast C (SNR 2–10) | SNR 2–5 recovered | SNR 5–10 recovered |
|---|---|---|---|
| shipped: mfsnr≥10, chi2≤5, rate[1,8] | **1.7%** | 1 | 44 |
| no mfsnr, **chi2≤5** (matched) | **≈4.3%** (mfsnr-only effect, ~2.5×) | — | — |
| no mfsnr, chi2≤3 | 4.1% | 12 | 96 |
| no mfsnr, looser geometry | 6.1% | 23 | 137 |

So mfsnr≥10 does **not** delete the faint-fast bin — 44 SNR 5–10 movers survive it. What it does: nearly
zero the faintest sub-bin (SNR 2–5) and cut SNR 5–10, for a **~2.5×** lower faint-fast completeness. That
factor is the **mfsnr-only** effect at *matched* chi2≤5 — verified by re-running `field_eval` on a 6-field
sample with mfsnr toggled at fixed chi2 (6 vs 15 faint-fast recoveries, the cut biting both sub-bins: SNR 2–5
4→1, SNR 5–10 11→5); it is not an artifact of also tightening chi2. (The no-mfsnr chi2≤3 row, 4.1%, is a
consistent chi2-mixed cross-check.) **Correcting an earlier overstatement** (carried in
`faint-fast-bin-only-scope` memory too): the premise "the movers ARE mf_snr 2–10" conflates *matched-filter
trail SNR* (mf_snr) with *detection SNR* — a long-trail fast mover can have mf_snr≥10 at detection-SNR 5–10,
which is exactly why 44 survive. mfsnr≥10 buys per-link 2v purity at a real ~2.5× completeness cost; it does
not erase the science target.

**Open item (flagged, not changed):** an *alert* stream is not gated on purity, yet the shipped faint-fast 2v
alert tier still applies `mfsnr_min_2v=10` (`trail_state_link` reads it from `op_point.json`). For the alert
tier specifically that pays ~2.4× completeness for purity the tier does not use — a genuine lever to drop
`mfsnr_min_2v` for faint-fast alerts. Left as shipped pending an explicit decision (it is correct for the
general/3σ 2v op-point; the question is only the faint-fast alert sub-stream).

### 2-visit purity at the base rate
chi2-optimized op, ρ=0.14: caps at **~8%** at C~2.3% (chi2≤3, S≥0.95), ~4% at C~3.6%, ~1% at C~6.6% —
exhaustively shown to be a hard wall (base rate + per-detection-inseparable faint streak FP; λ_2v∝N_FP²).
NO score threshold lifts faint-fast 2v above single-digit purity.

### 3-visit false-link rate (the discovery tier)
`link_fpp_calib_3visit.json` is a FAR table: λ_3v ≈ **0** per night at every S ≥ 0.8 (0.0575 at 0.6,
0.0025 at 0.7, 0.0 at ≥0.8) → **3v purity ≈ 100%** (false detections cannot align across 3 epochs — the
(FP)^N collapse). Recovers 2025 NY2 at every S in the table.

**3v discovery-tier faint-fast completeness is cadence-limited, not threshold-limited.** WFD is
pair-dominated by design (`two-vs-three-sighting-fast-neo`: even before SNR/linking, fast movers are
observed ≥3× the *same night* only ~17% of the time). So same-night standalone 3+visit discovery of
faint-fast movers is a *fraction* of the already-small 1.7% 2v-linked set — only those with a real 3rd
same-night detection. The 2v→3v promotion tier (`--promote-3v`) recovers exactly those: it attaches a real
3rd same-night detection lying on the precise 2-centroid chord track, elevating **33% of true 2v
candidates** to the pure 3v tier at **0 false** (validated). Same-night faint-fast discovery completeness is
therefore intrinsically small; the bulk of faint-fast discovery requires **multi-night** linking (out of
scope here).

## DECISION
The pipeline emits two tiers from **one global ADCNN score floor**:

**Global detection floor: `score_min = 0.80`.** Backed by: (i) it is the **lowest validated score** in the
λ(S) sweep, so it gives the highest validated faint-fast completeness (1.7%, vs 1.3% at 0.90) — faint-fast
completeness is monotone-decreasing in S; (ii) the 3+visit discovery tier is FAR-pure there (λ_3v=0/night,
recovers NY2), so raising S buys **no** 3v-purity gain while costing completeness; (iii) the chi2-gated 2v
false-link rate is manageable for the alert stream (0.12 / field-night). Lowering S below 0.80 is
**unvalidated** in the faint-fast sweep and floods the 2v alert stream (λ rises steeply). 0.80 is thus the
single-floor compromise. Held in `link_op_point.json`.

**Tier 1 — 3+visit (incl. 2v→3v promotions) = DISCOVERY (3σ).** Purity ≈100% at score_min=0.80
(λ_3v=0/night). This is the evidence-backed standalone-discovery channel for the faint-fast bin. Its
same-night completeness is small and **cadence-bound** (WFD pairs-dominated → ≥3 same-night detections are
rare); the promotion tier captures the multi-visit subset (33% of true 2v → pure 3v, 0 false). Higher
same-night faint-fast discovery completeness is not available at this cadence — it needs multi-night linking.

**Tier 2 — 2-visit = ALERT / follow-up candidates, NOT discovery.** By the purity-floor rule, faint-fast 2v
purity (≤8% at any S) is below any discovery floor → it is emitted as the chi2-ranked `alerts.jsonl` stream
for follow-up, never claimed as a discovery. The shipped 2v op (chi2_2v_max=5, rate∈[1,8], mfsnr_min_2v=10)
sets candidate quality. The photometric floor mfsnr≥10 lifts per-link 2v purity but, as shown above, costs
~2.4× faint-fast completeness (≈4%→1.7%) **without** reaching discovery-grade purity for this bin — so for the
faint-fast *alert* sub-stream it is a poor trade (alerts don't need the purity it buys). It is retained as
shipped for now (it is appropriate for the general/3σ 2v op-point) and flagged above as the open lever.

**Bottom line:** for the faint-fast bin, **score_min = 0.80** with **3+visit = discovery, 2-visit = alerts**.
No single-night 2-visit threshold yields discovery-grade purity (proven exhaustively: parallax, orbit-prior,
sky-region, partner-recovery, pixel forced-phot, CNN fine-tune). Standalone same-night faint-fast discovery
comes from the cadence-limited 3+visit tier (pure but small); the rest needs multi-night linking.

**Cleaner future architecture (noted, not yet adopted):** decouple the tiers — give the pure 3v discovery
tier a *lower* score floor (it stays FAR-pure, gaining 3v completeness) and gate the 2v alert stream
*separately* at a higher score. The current single 0.80 floor is pinned upward only by 2v-alert flooding,
not by any 3v-purity constraint.
