# 2-visit alert-tier operating point — mfsnr 5 vs 7 sweep (real nights, schema-1.1 ranked packets)

**Question:** with the alert tier now a separate product (`op_2v_alert.json`, ranked `priorityScore`
packets, `--alerts-top-n` budget), where does the photometric floor go — mfsnr ≥ 5 or ≥ 7 — and what
follow-up budget does it need?

## Evidence

### A. Injection calibration (82 off-ecliptic dense fields, S=0.80 — the pair-rich UPPER BOUND)
| mfsnr | faint-fast completeness | false 2v links / field-night (real FP×FP) |
|---|---|---|
| ≥10 (discovery op) | 1.7% | 0.12 |
| ≥7 | 3.9% | 1.0 |
| **≥5** | **6.1%** | **3.4** |

### B. Real-night sweep (this run; `alert_sweep/summary.csv`)
Substrates: run_band RA[345,348] box (10 real DP2 nights, main-belt field) and the 2025 NY2 discovery
night (run_night8731), each linked with `op_2v_alert.json` at mfsnr 5 and 7.

| run | alerts/night | 2v NEW | 3+visit | known rec | top pscore | search radius +30/+90 min |
|---|---|---|---|---|---|---|
| band mfsnr5 | 2 | 1 (chi2=1.4, mfsnr≈5.1) | 1 (5.6°/day, unmatched) | 0 | 3.38 | 8.9″ / 80″ |
| band mfsnr7 | 1 | 0 | 1 (same) | 0 | 3.45 | 8.9″ / 80″ |
| NY2 mfsnr5 | 1 | 0 | 1 | **2025 NY2** | 0.87 | 9.0″ / 80″ |
| NY2 mfsnr7 | 1 | 0 | 1 | **2025 NY2** | 0.87 | 9.0″ / 80″ |

Key facts:
- **Real alert load is tiny: 1–2 alerts/night/box** — an order of magnitude below the injection-field
  bound (3.4/field-night). Real WFD-ish cadence offers few same-night pairs; the dense-field number is
  the worst case for pair-rich cadences (deep drilling), not the typical night.
- **The lever behaves exactly as calibrated:** the single extra mfsnr5 alert is a 2v NEW with
  *excellent* geometry (chi2 = 1.4, vs true-pair median ~4) at mfsnr ≈ 5.1 — precisely the
  faint-but-well-measured candidate class the mfsnr≥7/10 floors delete.
- **Ranking works:** the unmatched fast mover (3+visit, 5.6°/day) tops both streams
  (priorityScore 3.4); the known recovery correctly ranks last (0.87); the extra mfsnr5 candidate
  slots between (3.11 median). A top-N consumer reads the headline first.
- **Follow-up boxes are trivially actionable:** search radius (linear ⊕ close-NEO admissible-region
  curvature at ρ=0.01 AU) is ~9″ at +30 min, ~80″ at +90 min — any follow-up aperture covers it.
  (+1 night is degree-scale for close objects — same-night follow-up is the point.)
- **Regression anchor intact:** NY2 night yields exactly 1 track = 2025 NY2 CONFIRMED at both settings.

## DECISION
**Default 2v-alert op: `mfsnr_min_2v = 5`** (as shipped in `op_2v_alert.json`), with a follow-up budget
of **`alerts_top_n = 50` per night** (generous: measured real load is 1–2/night/box; the cap exists for
pathological nights/dense cadences and is logged, never silent).

Rationale: the completeness gain (1.7→6.1% faint-fast, ~3.6×) is the alert tier's whole purpose; the
false-rate cost lands on a RANKED, CAPPED stream whose real measured volume is minutes of vetting per
night; and priorityScore cleanly separates the headline (fast 3+visit candidate) from the tail.
**Fallback:** if a deployment's follow-up capacity saturates (pair-rich cadence, ecliptic fields),
`--mfsnr-min-2v 7` cuts the load ~3.4× while keeping 2.3× the discovery-op completeness — the documented
knob, not a re-calibration.

Scope note (honest): this sweep covers two real substrates (a main-belt-field box + one NEO discovery
night) — enough to set the *default* because A gives the volume upper bound and B confirms ranking,
load, and the regression anchor on real nights. The NEO-specific alert completeness number still comes
from the blinded injection-on-real test (planned; see op_multinight_discovery.json notes).

Reproduce: `trail_state_link --dets <night dets> --known <known.csv> --op-point op_2v_alert.json
[--mfsnr-min-2v 7] --out <dir>/tracks.csv` → `alerts.jsonl` (ranked) + `summarize_alert_sweep.py`.

---
# ADDENDUM (2026-06-10): mfsnr 5->0 flip REJECTED — the bounded-alert gate failed

Question: drop mfsnr>=5 entirely and rely on ranked top-N? Evidence: v2 per-pair table (82 injection
fields, exact FP, full priorityScore inputs incl. pair_chi2), ranked by the REAL alert_stream priority.

| config                  | faint-fast C | alerts/field | top-N purity |
|-------------------------|--------------|--------------|--------------|
| shipped mfsnr>=5 (uncapped) | **6.07% (159 objs)** | 15.2 | 76.9% |
| mfsnr=0 (uncapped)      | 6.99% (183)  | 54.2         | 22.4% |
| mfsnr=0 top-10/field    | 2.44% (64)   | 9.1          | 67.7% |
| mfsnr=0 top-20/field    | 4.47% (117)  | 16.5         | 53.3% |
| shipped + top-up to 20  | 4.70% (123)  | 16.5         | 55.1% |

VERDICT — **shipped default FROZEN: S>=0.80, mfsnr>=5, uncapped per-field, global top-50/night.**
The mfsnr<5 band DOES hold truth (+24 objects; NY2-night check even recovered an extra real object,
2006 QX198, at mfsnr=0) — but that truth lives too deep in the FP-heavy tail to surface under any
realistic per-field budget (needs 40+ alerts/field). mfsnr=0 is documented ONLY as a high-budget
diagnostic / follow-up-rich mode, never the default.

DIAGNOSTIC (drives the next fix): the current priorityScore RANKS WORSE than a plain weakest-member-score
proxy (top-20 truth 117 vs 137) — within the chi2<=5 survivors the FP chi2 distribution has a fat LOW tail
(chance 2-point fits can be perfect), so the dominant exp(-chi2/5) term injects variance that lifts FPs
over truth. chi2 is excellent for REJECTION, bad as a strong RANKING weight inside its own gate.
-> priorityScore recalibration task: chi2 stays a hard gate; ranking re-weighted toward weakest-member
score + trail-consistency terms (field-grouped CV, top-N metrics, interpretable linear form).
