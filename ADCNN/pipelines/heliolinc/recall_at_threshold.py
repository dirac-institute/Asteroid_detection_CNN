"""Unbiased recovery efficiency vs faintness at a given ADCNN score floor, on the INJECTED test set.

Uses the held-out injected truth (DATA/test.csv: 5321 trails with true mag/SNR/trail_length and the
5-sigma STACK-detection flag) + an ADCNN detection catalog run at --cnn-thr 0 (all scored candidates).
For each score floor it matches truth<->detections (segment overlap, tol_px) and reports the
PER-DETECTION recall vs measured SNR and magnitude -- overall and for the STACK-MISSED subset (the
faint discovery population the catalogue is blind to). Then folds the per-detection recall through the
">=3 detections in a night" requirement to give the 3-visit linkage completeness at the 3-sigma point.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
from ADCNN.evaluation.catalog_match import match_trail_catalogs
from math import comb

TRUTH = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/DATA/test.csv"
MEAS  = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/Evaluation/catalogs_thr0/test_detections.csv"
TOLPX = 20.0
FLOORS = [0.59, 0.80]
SNR_BINS = [0, 3, 4, 5, 6, 8, 100]
MAG_BINS = [0, 22, 23, 23.5, 24, 24.5, 25, 99]

def recall_by(truth, col, bins, thr_floors_matched):
    lab = pd.cut(truth[col], bins)
    out = []
    for b, idx in truth.groupby(lab).groups.items():
        row = {col: str(b), "n": len(idx)}
        for thr, tm in thr_floors_matched.items():
            row[f"r@{thr}"] = float(tm.loc[idx].mean()) if len(idx) else float("nan")
        out.append(row)
    return pd.DataFrame(out)

def main():
    truth = pd.read_csv(TRUTH)
    meas = pd.read_csv(MEAS)
    out = []
    out.append(f"truth {len(truth)} injected trails | measured {len(meas)} candidates | tol {TOLPX}px")
    out.append(f"truth cols incl SNR_estimation, mag, stack_detection_5sigma; meas score range "
               f"[{meas.score.min():.2f},{meas.score.max():.2f}]")
    # match at each score floor -> per-truth 'matched' boolean
    matched = {}
    for thr in FLOORS:
        m = meas[meas.score >= thr]
        to, _, c = match_trail_catalogs(m, truth, tol_px=TOLPX, flag_col="det")
        matched[thr] = to["det"].reset_index(drop=True)
        out.append(f"\nscore>={thr}: {len(m)} dets | overall recall {c['TP']/(c['TP']+c['FN']):.3f}")
    truth = truth.reset_index(drop=True)
    # recall vs measured SNR
    out.append("\n=== recall vs SNR_estimation (ALL injected trails) ===")
    out.append(recall_by(truth, "SNR_estimation", SNR_BINS, matched).to_string(index=False))
    # recall vs SNR for STACK-MISSED subset (the discovery population)
    miss = truth[~truth.stack_detection_5sigma.astype(bool)]
    mm = {t: matched[t].loc[miss.index] for t in FLOORS}
    out.append(f"\n=== recall vs SNR_estimation (STACK-MISSED only, n={len(miss)}) ===")
    out.append(recall_by(miss.reset_index(drop=True).assign(**{f"_m{t}":mm[t].values for t in FLOORS}),
                         "SNR_estimation", SNR_BINS,
                         {t: pd.Series(mm[t].values) for t in FLOORS}).to_string(index=False))
    # recall vs magnitude
    out.append("\n=== recall vs mag (ALL injected trails) ===")
    out.append(recall_by(truth, "mag", MAG_BINS, matched).to_string(index=False))
    # fold per-detection recall (score>=0.80) -> 3-visit completeness
    out.append("\n=== 3-VISIT completeness at score>=0.80 = P(>=3 of N detections), per SNR bin ===")
    lab = pd.cut(truth.SNR_estimation, SNR_BINS)
    for b, idx in truth.groupby(lab).groups.items():
        r = float(matched[0.80].loc[idx].mean()) if len(idx) else 0.0
        def pge3(N, r): return sum(comb(N,k)*r**k*(1-r)**(N-k) for k in range(3, N+1))
        out.append(f"  SNR {str(b):>12}: per-det r={r:.3f} -> 3-visit eff: N=3 {pge3(3,r):.3f}  N=4 {pge3(4,r):.3f}  N=5 {pge3(5,r):.3f}")
    open("/tmp/recall_out.txt","w").write("\n".join(out)+"\n")
    print("DONE")

if __name__ == "__main__":
    main()
