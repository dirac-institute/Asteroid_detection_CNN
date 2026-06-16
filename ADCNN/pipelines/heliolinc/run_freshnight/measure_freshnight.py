#!/usr/bin/env python3
"""Measure purity + completeness for ONE fresh injected full-night field at the FROZEN alert op,
restricted to the training-range NEOs (detection-SNR in [2,10] AND trail length in [6,60] px).

Uses the SAME per-pair definition as the validation (exact_lowS_pairs.eval_field_exact): a 2-visit
pair is TP iff both members match the SAME injected objID (10 px), FP otherwise; completeness =
distinct recovered training-range objects / recoverable training-range objects (injected into >=2
same-night visits). Purity is the in-sample injected-truth fraction at the op.

Usage:  PYTHONPATH=<repo> python measure_freshnight.py --dir <run> --k 0 [--out metrics.json]
"""
import argparse, json, os
import pandas as pd
from ADCNN.pipelines.heliolinc import exact_lowS_pairs as ex

# frozen alert op (op_2v_alert.json) + the training range
OP = dict(score_min=0.80, mfsnr_min=5.0, chi2_max=5.0, rate_lo=1.0, rate_hi=8.0, len_min=6.0)
SNR_LO, SNR_HI = 2.0, 10.0
LEN_LO, LEN_HI = 6.0, 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # full chord-pair enumeration at floor 0.60 (then we apply the frozen op below) -- the exact,
    # FP-uncapped evidence table, identical machinery to the threshold-selection validation.
    rows, recoverable, n_seed = ex.eval_field_exact(a.dir, a.k, 0.60)

    # training-range denominator: recoverable objs (>=2 same-night sightings) with SNR in [2,10]
    # AND injected trail length in [6,60] px (joined from truth).
    truth = pd.read_csv(f"{a.dir}/truth_{a.k}.csv")
    lencol = "trail_px" if "trail_px" in truth.columns else "trail_length"
    tlen = dict(zip(truth.objID.astype(str), truth[lencol].astype(float)))
    # recoverable/pair objIDs are field-prefixed ("<k>_SNEO...."); truth objID is bare ("SNEO....").
    def bare(o):
        o = str(o)
        return o.split("_", 1)[1] if "_" in o else o
    def in_train(obj, snr):
        return (SNR_LO <= snr < SNR_HI) and (LEN_LO <= tlen.get(bare(obj), -1) <= LEN_HI)
    recoverable_tr = {o for o, s in recoverable.items() if in_train(o, s)}

    # apply the frozen op to the pairs; count TP/FP and the distinct recovered training-range objs
    tp = fp = 0
    rec_objs = set()
    for (mn, mf, rate, label, nfp, obj, mx, ln, c2, dpa, dsp, perp) in rows:
        if not (mn >= OP["score_min"] and mf >= OP["mfsnr_min"] and c2 <= OP["chi2_max"]
                and OP["rate_lo"] <= rate <= OP["rate_hi"] and ln >= OP["len_min"]):
            continue
        if label == "tp" and obj is not None and obj in recoverable_tr:
            tp += 1
            rec_objs.add(obj)
        elif label != "tp":
            fp += 1
    # FP pairs are training-range-agnostic (a false link has no true object); count all FP at the op.

    n_recoverable = len(recoverable_tr)
    purity = 100.0 * tp / (tp + fp) if (tp + fp) else None
    completeness = 100.0 * len(rec_objs) / n_recoverable if n_recoverable else None
    res = {
        "operating_point": OP, "training_range": {"snr": [SNR_LO, SNR_HI], "len_px": [LEN_LO, LEN_HI]},
        "n_recoverable_trainingrange_objs": n_recoverable,
        "n_recovered_objs": len(rec_objs),
        "completeness_pct": round(completeness, 2) if completeness is not None else None,
        "n_tp_pairs": tp, "n_fp_pairs": fp,
        "purity_pct": round(purity, 2) if purity is not None else None,
        "n_seed_pairs": n_seed, "n_pairs_passing_physcheck": len(rows),
    }
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
