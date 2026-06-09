# ρ/FP sky map — is there a base-rate-favorable sub-region for faint-fast 2-visit purity?

**Bin:** faint-fast same-night movers ONLY — detection-SNR ∈ [2,10], rate > 1 deg/day.
**Question:** off-ecliptic (|β|∈[20,46]) we measure 2v purity ~8%. Toward the ecliptic the real
NEO density ρ rises — but does it rise *enough* to lift purity to a discovery-grade level?

## Headline (load-bearing, slope-independent)
When the channel is FP-dominated, purity ∝ C·ρ/(C·ρ+λ) ≈ ρ/FP, so a sky sub-region helps only if it
raises the base rate ρ substantially. **It does not.** ρ rises at most **~2.5×** toward the ecliptic
(below). Even in the most charitable case — FP perfectly **flat** across the sky — that takes 8% purity
to 8%×2.5 = **~20%**, still far short of discovery-grade (need ~6×+). A ≤2.5× base-rate lever **cannot
close the gap**, regardless of how FP actually behaves. Sky selection is not a viable purity lever for
this bin. (FP almost certainly rises toward the ecliptic from crowding, which only makes it worse — but
the verdict does not depend on that.)

## (A) ρ_true(|β|) — Sorcha/Granvik NEOs through real OpSim cadence
Faint-fast (SNR 2–10, rate>1) same-night-2v-observable obj-nights, 559 total over 201 nights,
relative **sighting-count** vs the off-ecliptic anchor |β|∈[20,46] = 1.0:

| \|β\| band | obj-nights | ρ_rel |
|---|---|---|
| 0–5   | 106 | 2.41 |
| 5–10  | 111 | 2.53 |
| 10–15 | 79  | 1.80 |
| 15–20 | 54  | 1.23 |
| 20–30 | 120 | 1.37 |
| 30–46 | 89  | 0.63 |

ρ peaks at |β|<10 but only **~2.5×** above the off-ecliptic band. Caveat (cuts in our favor): this is a
**sighting-count** ratio, not per-deg² surface density — it conflates intrinsic density with how much
OpSim points at each |β|. The true per-area enhancement is likely *smaller*, strengthening the verdict.

Elongation: 86% of faint-fast 2v movers sit at elongation 120–180° (near opposition) — exactly where
WFD points at night, so we are **already in the favorable elongation regime**; no extra lever there.

## (B) FP false-link density vs |β| — physical prior, NOT a measurement
Only **10** false links exist across the 82-field set at score 0.80, and per-field rates are
**non-monotone** in the populated bins ([20,25)=0.148, [25,30)=0.250) — the apparent inward rise is
driven entirely by zero events in the outer |β|=35–50 band (~1.5–2σ down-fluctuation on ~3 expected).
At the bin's actual operating point (score≈0.95) there is ≈1 false link in 82 fields → the β-dependence
is **unmeasurable**. So: treat "FP rises toward the ecliptic" as a well-motivated **physical prior**
(crowding → more subtraction artifacts; on-ecliptic stellar density, absent from these off-ecliptic
deep fields, would push it higher still) — **not** a fitted slope. We deliberately do **not** quote a
quantitative FP_enh / purity_rel table; the earlier −0.12/deg extrapolation was not supported by the data.
(Also: the sweepcache `false` count is total false-links, not faint-fast-only — a further reason not to
lean on it quantitatively.)

## (C) Verdict — NO favorable sub-region
The off-ecliptic deep fields where we measured ~8% are **already the most favorable sky** for this bin:
- The base-rate gain available anywhere is ≤2.5× → caps any sky-selected purity at ~20% even under the
  most optimistic (FP-flat) assumption — not discovery-grade.
- FP is expected (physical prior) to rise toward the ecliptic, only lowering that ceiling.
- We are already near opposition (the favorable elongation regime).

**Consequence:** the "base-rate-favorable sky" lever (advisor menu option 1) is **closed**. The faint-fast
2-visit purity wall is not escapable by sky selection. Multi-night (≥3 epochs **across nights**, cadence-
compatible — NOT 3-same-night) remains the only path to discovery-grade purity for this bin.

Script: `/tmp/rho_fp_map.py` (reproducible from `run_test2/sorcha/test2neo_opsim.csv` +
`run_lambda/_sweepcache/` + `fields.csv`).
