# Static veto: what coverage costs, and what replaces it

**Date:** 2026-07-28. **Question:** the static veto needs a DRP coadd catalogue, which exists only
where DRP has processed coadds. On night 20260629 just **0.15%** of alerts had any catalogue to
check against, so 8 were flagged instead of the ~19% a covered night sees — silently, since the
class counts merely look clean. Can the veto be made coverage-independent **without losing purity
or completeness**?

## What the veto is

It flags an alert when one of its detections sits on a catalogued stationary source: a "mover"
with a member on a known star is almost certainly a template-subtraction residual. It acts twice —
as SEED EXCLUSION (a static–static pair is never seeded) and as a FLAG on surviving alerts.

## Measured on injected truth (run_calib_0630, 5,226 pairable injected objects)

All rows: `claim-order preal`, `op_2v_stream`, identical detections. Only the catalogue differs.

| static catalogue | sources | seed pairs excluded | alerts | true pairs | completeness | purity |
|---|---|---|---|---|---|---|
| none (control) | 0 | 0.00% | 11,765 | 987 | 18.89% | 8.39% |
| self-cal (recurrence) | 3,065 | 0.23% | 11,759 | 987 | 18.89% | 8.39% |
| all-sky refcat (the_monster) | 135,749 | 0.56% | 11,737 | 986 | 18.87% | 8.40% |
| DRP coadd objects | 1,072,589 | 1.77% | 11,693 | 986 | 18.87% | **8.43%** |

## Conclusions

1. **The static veto is worth far less than its flag rate suggests.** Even the full DRP catalogue —
   1.07M sources, the densest option — moves purity 8.39% → 8.43% and costs 1 real pair. It flags
   ~19% of alerts on a production night, but flagging is not removing: FLAG-not-drop means those
   alerts stay in the stream, merely demoted. The purity of the delivered stream barely moves.
2. **Therefore 0629's missing coverage is not the crisis it appeared to be.** Its alerts are not
   materially less pure for lacking the veto; they are less *annotated*.
3. **An all-sky refcat is a safe, coverage-complete substitute** where annotation matters: it
   reaches every night the telescope observes (135,749 sources from 88/88 sky shards on this
   footprint, versus DRP's patchy tracts) at 60% of DRP's purity gain and no extra completeness
   cost. `build_static_refcat.py --refcat the_monster_20250219 --collection refcats`.
4. **Self-calibration does not work.** A mover is at a position once and a static in every visit
   covering it, so recurrence *should* identify statics — but marginal residuals fluctuate across
   the detection threshold and so fail to recur, while a catalogue knows the star is always there.
   3,065 sources, 0.23% seed exclusion, zero measurable effect. Kept for the record, not for use.

## Recommendation

Use DRP where it exists and the refcat to fill the gaps (the veto accepts a concatenation) — but
treat this as annotation quality, not purity. Do NOT block a night on missing static coverage, and
DO record the coverage fraction (`stream_summary.static_coverage`) so a low-coverage night is never
mistaken for a clean one. The real purity levers remain the CNN score, the orbit chi2 and mf_snr,
per ALERT_RANKING.md.

## Caveat

Measured on ONE night's injected set (603 panels, 0630 sky) with synthetic positives. The
FLAG-not-drop design means the veto's value is mostly in RANKING, which this table does not score;
a veto-aware term in the calibrated P(real) model would measure that properly and is the natural
follow-up.
