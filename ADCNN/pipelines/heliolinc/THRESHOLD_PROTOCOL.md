# Formal threshold-selection protocol — same-night 2v alert product (paper methodology)

**Purpose:** derive the operating thresholds from a pre-defined objective on VALIDATION data, freeze them,
and only then run the blinded injection-on-real test for the reported numbers. This protects the result
from "threshold tuned on the final test" criticism. Frozen 2026-06-10, BEFORE the blind test.

## 1. Product and objective (defined first)
- **Product:** same-night 2-visit follow-up ALERT (never a discovery claim; 3+visit / multi-night are
  separate products with separate thresholds — do not conflate in the paper).
- **Objective:** J(S, mfsnr | budget) = expected number of faint-fast injected NEOs (detection-SNR 2–10,
  rate 1–8°/day) whose alert ranks INSIDE the follow-up budget, under the shipped ranking
  (recalibrated priorityScore = weakest-member ADCNN score; see ALERT_SWEEP_DECISION.md addendum 2).
- **Primary budget:** global top-50/night; per-field budgets (top-5/10/20/50) reported for sensitivity.
- **Hard constraints:** chi2 ≤ 5 acceptance gate; rate ∈ [1,8]°/day; search regions actionable at
  +30/+60/+90 min (9″/80″ — threshold-independent, set by the predictor); NY2 + known-recovery
  regressions intact; no FP flood (night-level margin below).

## 2. Validation data (not the biased known-object table)
82 off-ecliptic injection-on-real fields: synthetic faint-fast NEO trails (Sorcha×Granvik, SNR down to 2)
injected into real DP2 difference images at realistic ~34-min WFD cadence → real FP population + exact
truth. Denominator = 2,618 faint-fast objects injected into ≥2 same-night visits. Per-pair evidence table:
`measure_nomfsnr` v2 (uncapped at floor 0.80 → exact FP statistics; floor<0.80 truth-exact with a
Poisson rank model for the FP-subsampled band — see §4 note).

## 3. The sweep (J in expected faint-fast objects per 82 field-nights)
### J(S) at mfsnr≥5 — figure `threshold_sensitivity.png`, left panel
| S floor | top-5 | top-10 | top-20 | top-50 |
|---|---|---|---|---|
| 0.60 | 77.0 | 104.9 | 133.6 | 167.0 |
| 0.70 | 73.3 | 100.9 | 129.0 | 162.0 |
| 0.75 | 72.3 | 99.9 | 128.0 | 161.0 |
| **0.80** | **71.0** | **98.0** | **126.0** | **159.0** |
| 0.825 | 71.0 | 97.0 | 124.0 | 157.0 |
| 0.85 | 69.0 | 94.0 | 120.0 | 152.0 |
| 0.875 | 67.0 | 91.0 | 117.0 | 146.0 |
| 0.90 | 65.0 | 85.0 | 105.0 | 127.0 |

**J(S) is a PLATEAU over 0.60–0.825 (±4%) with a knee at ~0.85 and a steeper fall by 0.90. No cliff.**
S=0.80 is selected ON the plateau; J(0.75)/J(0.80) = +1.6%, J(0.85)/J(0.80) = −4.8% (top-20). The paper
claim: *S=0.80 is the selected operating point and the result is stable over a nearby range.* The sub-0.80
branch's small gain (≤+6%) additionally carries the seeding-tractability cost measured elsewhere
(40–144 M chord pairs/field at 0.60) — the plateau means we pay nothing for staying at 0.80.

### J(mfsnr) at S=0.80 — right panel (exact, uncapped table)
| mfsnr | top-5 | top-10 | top-20 | top-50 | FP/field | FP outranking truth /30-field night |
|---|---|---|---|---|---|---|
| 0 | 71 | 101 | 137 | 180 | 41.1 | 36.6 |
| 3 | 72 | 102 | 138 | 180 | 23.4 | 30.0 |
| **5** | **71** | **98** | **126** | **159** | **3.4** | **13.9** |
| 7 | 45 | 63 | 84 | 101 | 1.0 | 5.5 |
| 10 | 20 | 27 | 37 | 44 | 0.12 | 1.7 |

**The mfsnr optimum depends on the budget UNIT, and the paper must state this precisely:**
- Under a **per-field** budget (top-20/field), mfsnr=3 maximizes J (138 vs 126, +10%) — the recalibrated
  ranking surfaces truth even in the FP-heavier stream.
- Under the **shipped per-NIGHT global budget** (top-50/night over ~30 fields), the deciding quantity is
  the FP load that outranks typical truth: mfsnr≥5 → 13.9/night (28% of budget, 3.6× safety margin);
  mfsnr=3 → 30/night (60%, 1.7×); mfsnr=0 → 36.6/night (73%, marginal). **mfsnr≥5 is selected for the
  night-level robustness margin**; mfsnr=3 is documented as the per-field-budget variant (+10% J) for
  deployments with per-field follow-up capacity.
- mfsnr≥7 collapses J (−33%) and ≥10 deletes the product (−71%): the photometric cut is a dial with a
  sharp cost above 5 — this IS the cliff in the system, and it sits in mfsnr, not in S.

## 4. Selected operating point (FROZEN before the blind test)
```
ADCNN score floor   S >= 0.80      (plateau member; tractable downstream; validated anchors)
photometric cut     mfsnr >= 5     (night-budget safety margin 3.6x)
acceptance gate     pair_chi2 <= 5 (gate ONLY -- excluded from ranking by measurement)
ranking             priorityScore = tier base + 0.95 * weakest-member ADCNN score
budget              top-50/night global (alerts_top_n=50); uncapped per-field
```
The blinded injection-on-real test reports final completeness/purity/alert yield at THIS point; it is not
used for tuning. Estimator note: floor<0.80 J values combine the exact ≥0.80 branch (rank domination: pairs
below the floor cannot outrank ≥0.80 pairs under the score_min ranking) with a Poisson rank model over the
FP-subsampled 0.60-table band; all selected-point and mfsnr-row numbers are exact (uncapped v2 table).

## 5. Separate products, separate thresholds (do not mix in the paper)
- 3+visit same-night confirm: S≥0.70, mfsnr off (geometry carries purity; op_3v_confirm.json).
- Multi-night discovery: S≥0.80 default; S≥0.60 reservoir VALID only behind the length-split hybrid
  router (HYBRID_LADDER_RESULT.md). These are independent constrained optima of different objectives.
