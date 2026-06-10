# 2v alert-reservoir experiment — 0.60 rung RESULT (2026-06-10)

Per-pair table: measure_nomfsnr --smin 0.6, all 82 injection fields (faint-fast truth to SNR 2),
chi2<=5 + rate[1,8] gates, truth-aware FP-subsample (cap 30k seeds/field).

## The decisive numbers (completeness EXACT; truth-aware sampling)
| mode                          | faint-fast C | objects |
|-------------------------------|--------------|---------|
| shipped alert (0.80, mfsnr5)  | 6.07%        | 159     |
| 0.80, no mfsnr                | 6.99%        | 183     |
| A anchored (max>=0.8,min>=0.6)| 7.52%        | 197     |
| B low-low fast-trail (len>=6) | 7.52%        | 197 (identical set) |
| raw 0.60 (forbidden ref)      | 7.52%        | 197 (identical set) |

ALL low-threshold modes converge to the SAME +14 objects: every marginal true pair at floor 0.60 already
has a >=0.80 anchor member AND both trails >=6px. The reservoir's whole 0.60-rung upside = +8% relative,
bounded by per-sighting DETECTION recall (recall^2 funnel), not by threshold or routing cleverness.

## The tractability fact
Chord seeding at floor 0.60 generates 40-144 MILLION pairs per field (all 82 fields hit the 30k cap;
median sampled fraction 9e-4). Same-night 2v at 0.60 is computationally explosive at the SEEDING stage --
an anchored-only seeder (KD from strong anchors only) would fix tractability but cannot fix the +8% ceiling.
(FP-load columns from this run are unquotable -- weights ~1e4/row; prior art bounds it: asym_sweep purity
0.55% at S_low=0.70 => ~10-100x the baseline FP load. TP/FP ranking proxy does separate: med weakest-member
score 0.94 TP vs 0.66-0.70 FP -- ranked top-N would surface the truth, but over a flooded stream.)

## VERDICT
Do NOT ship a same-night 2v reservoir at 0.60: tiny exact upside (+14 objects / 82 field-nights), seeding
explosion, FP load bounded-bad by prior art. The reservoir architecture pays off where geometry across
EPOCHS supplies the purity (multi-night hybrid: S>=0.60 reopened, HYBRID_LADDER_RESULT.md); same-night 2v
lacks the third epoch and the detection funnel caps the gain. The 0.50 rung + NEO-rich substrate remains
folded into the blinded injection-on-real test as Product B, with LOW expectations stated up front.
