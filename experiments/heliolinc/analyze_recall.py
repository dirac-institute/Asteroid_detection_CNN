"""Unbiased detection-capability analysis on the SYNTHETIC test sets (uniform trail-length & SNR).

The real-data recall (test_real) is selection-biased: it is built from *catalogued* fast movers,
which skew short and bright, and v7 was designed for elongated trails. The synthetic test sets
sample trail_length (6-60 px) and detection-SNR (2-8) uniformly, so recall measured there is the
fair measure of what the detector can do across the parameter space.

This sweeps the RF threshold on the rf_thr=0 catalogs and reports, per test set:
  * recall (object-level, trail-overlap) + FP/panel vs threshold,
  * recall vs trail_length and vs SNR at a chosen threshold,
then re-runs the HelioLinC linkability simulation using the *unbiased* per-sighting recall.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.evaluation.catalog_match import match_trail_catalogs

THRS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]


def recall_at(meas: pd.DataFrame, truth: pd.DataFrame, thr: float, tol_px: float = 20.0):
    m = meas[meas.score_rf >= thr]
    t_out, _, c = match_trail_catalogs(m, truth, tol_px=tol_px)
    n_pan = truth.image_id.nunique()
    return c["TP"] / max(c["TP"] + c["FN"], 1), c["FP"] / max(n_pan, 1), t_out


def binned_recall(t_out: pd.DataFrame, col: str, edges, flag="nn_detected"):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t_out[col] >= lo) & (t_out[col] < hi)
        n = int(m.sum())
        out.append((f"{lo:g}-{hi:g}", n, t_out.loc[m, flag].mean() if n else float("nan")))
    return out


def linkability(truth_win: pd.DataFrame, p: float, trials=40, seed=0):
    rng = np.random.default_rng(seed)
    tw = truth_win.copy()
    tw["night"] = np.floor(tw.mjd - 0.5).astype(int) if "mjd" in tw else np.floor(tw.image_id)
    vals = []
    for _ in range(trials):
        keep = tw[rng.random(len(tw)) < p]
        g = keep.groupby(["ObjID", "night"]).size()
        tn = (g >= 2).reset_index(name="t")
        cnt = tn[tn.t].groupby("ObjID").size()
        vals.append(int((cnt >= 3).sum()))
    return np.mean(vals), np.std(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat-dir", default=str(REPO / "Evaluation/catalogs_thr0"))
    ap.add_argument("--sets", nargs="*", default=["test_5sigma", "test_4sigma", "test_3sigma"])
    ap.add_argument("--report-thr", type=float, default=0.0, help="threshold for the per-bin breakdown")
    a = ap.parse_args()

    t_out_ref = {}
    for s in a.sets:
        meas = pd.read_csv(Path(a.cat_dir) / f"{s}_detections.csv")
        truth = pd.read_csv(REPO / f"DATA_DIFFIM/{s}/test.csv")
        print(f"\n================  {s}  (uniform: len 6-60px, SNR 2-8; {truth.image_id.nunique()} panels) ================")
        print(f"{'thr':>5} {'recall':>8} {'FP/panel':>9}")
        for thr in THRS:
            rec, fpp, t_out = recall_at(meas, truth, thr)
            if abs(thr - a.report_thr) < 1e-9:
                t_out_ref[s] = t_out
            print(f"{thr:5.2f} {rec:8.3f} {fpp:9.1f}")

    # parameter breakdown at report-thr (use 5sigma = cleanest)
    ref = "test_5sigma" if "test_5sigma" in t_out_ref else a.sets[0]
    t_out = t_out_ref[ref]
    print(f"\n=== recall vs trail_length  ({ref}, thr={a.report_thr}) ===")
    for lab, n, r in binned_recall(t_out, "trail_length", [6, 12, 20, 30, 45, 60]):
        print(f"  len {lab:>7} px : n={n:4d}  recall={r:.3f}")
    print(f"\n=== recall vs detection-SNR  ({ref}, thr={a.report_thr}) ===")
    for lab, n, r in binned_recall(t_out, "SNR", [2, 3, 4, 5, 6, 8]):
        print(f"  SNR {lab:>6} : n={n:4d}  recall={r:.3f}")

    # linkability with UNBIASED recall (the synthetic-test recall at report-thr)
    rec_unbiased = float(t_out["nn_detected"].mean())
    truth_win = pd.read_csv(REPO / "experiments/heliolinc/run_truth/truth_dets.csv")
    truth_win = truth_win[(truth_win.mjd >= 60866) & (truth_win.mjd < 60880)]
    print(f"\n=== HelioLinC linkability (window ceiling 37 objects) under UNBIASED recall ===")
    print(f"  using synthetic {ref} recall@{a.report_thr} = {rec_unbiased:.3f}")
    for p in sorted({rec_unbiased, 0.6, 0.75, 0.9}):
        mu, sd = linkability(truth_win, p)
        tag = " <- unbiased" if abs(p - rec_unbiased) < 1e-9 else ""
        print(f"    p={p:.3f}: linkable objects = {mu:.1f} +/- {sd:.1f}{tag}")


if __name__ == "__main__":
    main()
