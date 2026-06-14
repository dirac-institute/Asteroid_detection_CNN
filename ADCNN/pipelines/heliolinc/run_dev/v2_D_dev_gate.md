# ADCNN v2_D — DEV ALERT GATE: PASS (20 off-ecliptic fields, frozen op)

Frozen op: S>=0.80, mf_snr>=5, chi2<=5 (gate), rate[1,8]. v2_D = stage-1 domain fine-tune
(hard-positive oversampling) + stage-2 refit + MF_LEN recalibration (offset 7.67, slope 0.9425).

| model | tp | fp | purity | faint-fast C | alerts/field-night |
|---|---|---|---|---|---|
| v1    | 95 | 5  | 95.0% | 1.42% | 5.0 |
| v2_D  | 287| 20 | 93.5% | 5.67% | 15.3 |

- faint-fast alert completeness +299% (~4x) at maintained purity (95.0->93.5%, no collapse).
- PRODUCT metric (linked 2v alert completeness) improved, not just detector recall.
- Load 3x (5->15.3/fn) but purity held; ranked top-50 stream.
- Gate PASS: >=20% rel faint-C gain AND purity >=0.9x v1 AND no collapse.

The MF_LEN diagnosis chain (the detour that made this work): v2_D's domain-adapted stage-1 has a
TIGHTER ends-bloom (offset 7.7px vs v1 33.4) -> v1's de-bias constant zeroed v2_D len_db -> the
frozen len_db>=6 floor deleted v2_D's real detections (apparent "0 pairs FAIL"). Re-deriving the
2-param de-bias (field-held-out, residual ~1px) restored len_db (17%->95% >=6px) and unlocked the
4x product gain. Three v1-fit constants were invalidated by the stage-1 change and re-derived:
stage-2 score, [FP-budget thr not needed], MF_LEN trail-length de-bias.

NEXT: ONE blind shot (job 28694740 -> run_blind_v2eval/, run_blind untouched) v1 vs v2_D at the
frozen op vs the v1 blind row (faint-C 3.64%, purity 99.1%). No retune after blind.
