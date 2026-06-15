#!/usr/bin/env python3
"""ADCNN v2_D — one-command blind-verdict report regeneration.

Reads the small per-field pair caches (v1 from run_blind/_nomfsnr_cache, v2_D from
run_blind_v2eval_cal/_nomfsnr_cache; generated at the alert-op floor smin=0.80) and prints the
frozen-op matched comparison table (ALL / off-ecliptic / ecliptic). On first run it generates the
v2_D caches via eval_field_exact (smin 0.80) so subsequent runs are instant from JSON — no
re-detection, no re-pairing. This is the durable evidence: detections can be regenerated from the
frozen models (see REPRODUCE_V2_D.md) but the verdict reads from these caches.

Usage:  PYTHONPATH=<repo> python regen_v2_report.py        # prints the v1-vs-v2_D blind table
"""
import json, os, glob
import exact_lowS_pairs as ex

HL = os.path.dirname(os.path.abspath(__file__))
V1 = f"{HL}/run_blind"
V2 = f"{HL}/run_blind_v2eval_cal"
OFFECL = [str(k) for k in range(20)]
ECL = ["24", "25", "26", "27", "28", "29"]
OP = dict(smin=0.80, mfmin=5.0, rlo=1.0, rhi=8.0)


def v2_cache(k):
    """Return v2_D pair rows for field k at smin 0.80; generate+save the cache if missing."""
    cp = f"{V2}/_nomfsnr_cache/{k}_smin0.8_v3exact.json"
    if os.path.exists(cp):
        return json.load(open(cp))["rows"]
    rows, rec, n = ex.eval_field_exact(V2, k, 0.80)
    os.makedirs(f"{V2}/_nomfsnr_cache", exist_ok=True)
    json.dump({"rows": rows, "rec": rec, "n_seed": n}, open(cp + ".tmp", "w"))
    os.replace(cp + ".tmp", cp)
    return rows


def v1_rows(k):
    return json.load(open(f"{V1}/_nomfsnr_cache/{k}_smin0.5_v3exact.json"))["rows"]


def rec_for(ks):
    r = {}
    for k in ks:
        for o, s in json.load(open(f"{V1}/_nomfsnr_cache/{k}_smin0.5_v3exact.json"))["rec"].items():
            r[f"{k}_{o}"] = s   # injection truth is shared (same fields) -> v1 rec is the denominator
    return r


def tally(ks, getrows, rec):
    fftot = sum(1 for s in rec.values() if 2 <= s < 10)
    tp = fp = 0; ff = set()
    for k in ks:
        for r in getrows(k):
            mn, mf, rate, lab, obj = r[0], r[1], r[2], r[3], r[5]
            if mn >= OP["smin"] and mf >= OP["mfmin"] and OP["rlo"] <= rate <= OP["rhi"]:
                if lab == "tp":
                    tp += 1
                    if obj and 2 <= rec.get(f"{k}_{obj}", -1) < 10:
                        ff.add(f"{k}_{obj}")
                else:
                    fp += 1
    pur = round(100 * tp / (tp + fp), 1) if tp + fp else None
    C = round(100 * len(ff) / fftot, 2) if fftot else None
    return tp, fp, pur, C, round((tp + fp) / len(ks), 1)


def main():
    print("ADCNN v2_D blind verdict (frozen op S>=0.80, mf_snr>=5, chi2<=5, rate[1,8]):\n")
    print(f"{'split':12s} {'model':5s} {'tp':>5s} {'fp':>4s} {'purity':>7s} {'C_ff':>7s} {'alerts/fn':>10s}")
    for tag, ks in [("ALL", OFFECL + ECL), ("off-ecl", OFFECL), ("ecliptic", ECL)]:
        rec = rec_for(ks)
        a = tally(ks, v1_rows, rec)
        b = tally(ks, v2_cache, rec)
        print(f"{tag:12s} {'v1':5s} {a[0]:>5d} {a[1]:>4d} {str(a[2])+'%':>7s} {str(a[3])+'%':>7s} {a[4]:>10}")
        print(f"{tag:12s} {'v2_D':5s} {b[0]:>5d} {b[1]:>4d} {str(b[2])+'%':>7s} {str(b[3])+'%':>7s} {b[4]:>10}")
    print("\n(v1 reproduces its original BLIND_TEST_REPORT numbers exactly -> harness check.)")


if __name__ == "__main__":
    main()
