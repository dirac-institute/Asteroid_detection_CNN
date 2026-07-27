# Alert ranking: how to combine CNN score and orbit-fit chi2, measured

**Date:** 2026-07-26. **Question:** at alert-stream volume, what is the right way to combine the
per-detection CNN score with the two-visit orbit-fit chi2 (and mf_snr) into ONE ranking, and where
should the threshold go? **Answer:** rank by a calibrated `P(real)` fit on a real night's own
false-positive population; the weighting is measured, not tuned.

## Method

For a fixed alert budget the optimal ranking is the likelihood ratio `p(x|real)/p(x|false)`
(Neyman–Pearson), so the relative weight of the features is determined by data. The features are
(near-)conditionally independent given the label, so log-evidence adds and the justified form is a
logistic/naive-Bayes model:

```
logit P(real) = a + b·log(score_min) + c·log(chi2) + d·log(mfsnr_min)
```

Measured within-class correlations of `log score` vs `log chi2`:

| evidence set | real class | false class |
|---|---|---|
| 82 off-ecliptic injection fields | −0.047 | −0.004 |
| real night 20260630 (own FP null) | −0.243 | −0.015 |

The false class — which dominates the denominator — is essentially independent in both, so the
additive form stands.

## The calibration set (this is the part that matters)

A likelihood ratio is only as good as `p(x | false)`. Injection *fields* supply the wrong null:
their negatives are faint random chance links in sparse off-ecliptic sky, whereas a real survey
night's negatives are STRUCTURED residuals — static template subtractions, satellite trains — which
are bright and morphologically real. So the calibration set is built ON the night
(`ADCNN/calibration/night_pair_injection.py`): pair-consistent synthetic movers injected into
20260630's own repeat-pointing regions, re-detected and re-linked through the production chain, so
every negative is that night's genuine residual.

Night structure matters: 20260630's 27 visits span 11 h over 18 pointings and do NOT tile one
field. Injecting over the global sky bounding box put 6000 objects into empty gaps → 6 sightings,
0 paired. Grouping visits by boresight and keeping only groups whose epoch gap lies inside the
linker window (a 1-min repeat cannot show 1–8 °/day motion; 88–92 min is outside `max_arc_2v_min`)
gave 5 usable groups → 603 panels, 15,029 objects, 5,226 paired → **11,663 labelled 2-visit alerts,
936 real (8.0%), 10,727 false**.

## Result (cross-validated BY POINTING GROUP)

| ranking | AUC |
|---|---|
| CNN score alone | 0.9842 |
| chi2 alone | 0.8455 |
| mf_snr alone | 0.9751 |
| current `priorityScore` | 0.9842 |
| CV score + chi2 | 0.9861 |
| **CV score + chi2 + mf_snr** | **0.9890** |

```
logit P(real) = 4.499 + 11.525·log(score) − 1.041·log(chi2) + 1.219·log(mfsnr)
```

Real movers kept at a fixed nightly budget:

| budget | score | chi2 | score+chi2 | all three |
|---|---|---|---|---|
| 500 | 51.0% | 27.2% | 52.4% | 51.3% |
| 1000 | 88.9% | 44.0% | 90.6% | **90.8%** |
| 2000 | 96.9% | 63.5% | 97.2% | **98.0%** |
| 5000 | 98.9% | 87.8% | 98.9% | **99.6%** |

Operating points on the calibrated axis (out-of-fold probabilities, so honest):

| P(real) ≥ | alerts/night | completeness | purity |
|---|---|---|---|
| 0.90 | 301 | 30.3% | 94.4% |
| 0.70 | 673 | 64.9% | 90.2% |
| 0.50 | 874 | 80.8% | 86.5% |
| 0.30 | 1037 | 87.7% | 79.2% |
| 0.10 | 1459 | 93.6% | 60.0% |
| 0.02 | 3303 | 96.9% | 27.5% |

That table IS the threshold answer: pick the budget or the purity you want and read off the cut.

## A claim of mine that this REFUTES

Earlier I reported that chi2 ranks validated alerts ~40× better than CNN score, from the 12 frozen
production alerts of 20260630 landing at stream ranks 3771–9942 by score but 22–260 by chi2. That
inference was wrong, because those 12 alerts are **not truth**: none has an MPC match and 9 of the
12 are veto-flagged (static/train), i.e. they are as likely to be residuals as movers. Ranking them
highly is not evidence of a good ranking. Under controlled injected truth, CNN score is the single
strongest feature (AUC 0.984 vs 0.846) and chi2 contributes a real but modest improvement. The
controlled measurement supersedes the proxy.

## Honest limitations

- **The positive class is synthetic.** `add_trails` renders a Gaussian-PSF trail; if the CNN finds
  those easier than real asteroid trails, score's dominance is inflated. This experiment fixed the
  NULL, not the positives. Settling it needs real labelled movers — MPC-recovered knowns on a night
  that has them (20260630 has none).
- One night, one instrument configuration. The coefficients should be re-fit per night (the machinery
  is now a single command) before being treated as universal.
- mf_snr is currently also a hard GATE (`mfsnr_min_2v`). It carries real ranking information
  (AUC 0.975), so it is better used as a soft term than a cut: on real panels the ≥5 gate acts as a
  per-sighting SNR≈8–9 cut, which removes exactly the faint-fast objects the detector exists to find.

## Reproduce

```
python -m ADCNN.calibration.night_pair_injection --manifest <wcs manifest> --dets <night dets> \
    --out-dir outputs/runs/run_calib_<night>
sbatch ADCNN/calibration/calib_detect.slurm            # detect --inject + mask_flags
python -m ADCNN.linking.link_2visit --dets <calib masked dets> ... --claim-order quality
python -m ADCNN.calibration.fit_alert_ranking --alerts <calib alerts> --inject <inject.csv>
```
