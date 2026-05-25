"""A/B experiment: does fixing the trail ORIENTATION before the RF improve the second stage?

Single-variable comparison on identical data + pipeline. For every panel we run v7 ONCE and
build the 72-feature candidate table twice:
  - "pca"    : orientation = footprint principal axis (the corrected extraction; ~8-10° MAD).
  - "nnhead" : orientation = NN sin2β/cos2β-head angle (the original; r≈0 vs truth).
Only the 7 orientation features (or_beta, or_snr_L*, or_flux_L*) differ between the two; the
nnhead table is the pca table with just those columns recomputed (no doubled feature cost).

RF training data = the SAME 64 held-out validation panels the deployed RF uses (train_end_to_end
stage 2 = train_rf_from_val, neg5). We train one RandomForest per variant, then evaluate both on
the synthetic test sets (recall vs FP/panel via the trail-overlap matcher).

Panels (train + all test sets) are sharded round-robin across the visible GPUs — each GPU runs
v7 + feature extraction on its shard in its own process; the main process gathers the small
candidate tables, fits the two RFs, and evaluates. Streaming per panel -> bounded memory.

    python experiments/heliolinc/rf_orient_compare.py --n-gpus 4
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))

REAL = REPO / "DATA_DIFFIM_realistic"
# Canonical RF training data = the NN's held-out VALIDATION panels (train_end_to_end stage 2:
# train_rf_from_val on the first n_val_panels=64 of shard_3_val.csv). NOT a big shard sweep.
VAL_H5 = f"{REAL}/shard_3/train.h5"
VAL_CSV = f"{REAL}/shard_3_val.csv"
N_VAL_PANELS = 64
TESTSETS = ["test_5sigma", "test_4sigma", "test_3sigma"]
DATA = REPO / "DATA_DIFFIM"
EVAL_COLS = None  # set per worker after importing RF_FEATURES_V2


def _both_variants(model, predict, compute, add_orient, mad_sigma, img, rl, device):
    """Run v7 once; return (cand_pca, cand_nnhead, prob). nnhead = pca with the orientation
    columns recomputed via the NN-head angle (single v7 pass, single feature pass)."""
    p, s, c, a = predict(model, img, rl, device=device)
    prob = p.astype(np.float32)[None]
    # gate_pmax=0.10 matches the deployed eval (make_eval_catalogs) — cheaply drops sub-threshold
    # noise candidates (val-validated, 0 TP loss) so the 72-feature suite isn't run on thousands
    # of 3σ/4σ false candidates per panel. Same gate for both variants -> A/B stays fair.
    cand, _ = compute(prob, img.astype(np.float32)[None], s[None], c[None], a[None],
                      real_labels=rl[None], orient_mode="pca", gate_pmax=0.10, verbose=False)
    if not len(cand):
        return cand, cand, prob
    cand_nn = cand.copy()
    add_orient(cand_nn, {0: prob[0]}, {0: img.astype(np.float32)},
               {0: float(mad_sigma(img.astype(np.float32)))},
               {0: s}, {0: c}, {0: a}, orient_mode="nnhead")
    return cand, cand_nn, prob


def _gpu_worker(gpu_id, work, v7_ckpt, q):
    """Process this shard's panels on one GPU. `work` = list of (kind, h5, csv, pid) where kind
    is 'train' or a test-set name. Returns (train_Xp, train_Xn, train_y, eval_rows)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import h5py
    import torch
    from ADCNN.data.preprocessing import diffim_mad_sigma
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import compute_v2_features, _add_orient, RF_FEATURES_V2
    from ADCNN.inference.rf_postproc import label_candidates_by_injection_overlap

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(v7_ckpt, map_location=dev).eval()
    feats = list(RF_FEATURES_V2)

    def pool(cand, labels):
        keep = (labels == 1) | ((labels == 0) & (cand["frac_real_label_overlap"].to_numpy() < 0.5))
        return cand.loc[keep, feats].fillna(0.0).to_numpy(np.float32), labels[keep]

    Xp, Xn, ys, eval_rows = [], [], [], []
    # group work by (h5) to open each file once
    csv_cache = {}
    by_file = {}
    for kind, h5, csv, pid in work:
        by_file.setdefault(h5, []).append((kind, csv, pid))
    for h5, items in by_file.items():
        with h5py.File(h5, "r") as f:
            for kind, csv, pid in items:
                if csv not in csv_cache:
                    csv_cache[csv] = pd.read_csv(csv)
                cat = csv_cache[csv]
                img = f["images"][pid][:].astype(np.float32)
                rl = f["real_labels"][pid][:].astype(np.uint16)
                cand, cand_nn, prob = _both_variants(
                    model, predict_panel_overlap_3ch_full, compute_v2_features,
                    _add_orient, diffim_mad_sigma, img, rl, dev)
                if not len(cand):
                    continue
                if kind == "train":
                    cp = cat[cat.image_id == pid].copy(); cp["image_id"] = 0
                    labels = label_candidates_by_injection_overlap(cand, cp, prob)
                    xp, yp = pool(cand, labels); xn, _ = pool(cand_nn, labels)
                    Xp.append(xp); Xn.append(xn); ys.append(yp)
                else:  # eval: emit measured rows for both variants
                    for variant, cc in (("pca", cand), ("nn", cand_nn)):
                        d = cc.copy()
                        d["image_id"] = pid
                        d["x"] = cc["x_centroid"]; d["y"] = cc["y_centroid"]
                        d["beta"] = cc["or_beta"]; d["length"] = cc["mf_length"]
                        d["__set"] = kind; d["__variant"] = variant
                        eval_rows.append(d[["__set", "__variant", "image_id", "x", "y",
                                            "beta", "length"] + feats])
    q.put((
        np.concatenate(Xp) if Xp else np.empty((0, len(feats)), np.float32),
        np.concatenate(Xn) if Xn else np.empty((0, len(feats)), np.float32),
        np.concatenate(ys) if ys else np.empty((0,), np.int64),
        pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame(),
    ))


def build_work(n_val_panels, eval_panels):
    work = []
    val_ids = sorted(pd.read_csv(VAL_CSV).image_id.unique())[:n_val_panels]
    for pid in val_ids:
        work.append(("train", VAL_H5, VAL_CSV, int(pid)))
    for s in TESTSETS:
        ids = sorted(pd.read_csv(DATA / s / "test.csv").image_id.unique())
        if eval_panels:
            ids = ids[:eval_panels]
        for pid in ids:
            work.append((s, str(DATA / s / "test.h5"), str(DATA / s / "test.csv"), int(pid)))
    return work


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--n-val-panels", type=int, default=N_VAL_PANELS)
    ap.add_argument("--eval-panels", type=int, default=0, help="0 = all test panels")
    ap.add_argument("--neg-ratio", type=int, default=5, help="deployed reg2 = neg5")
    ap.add_argument("--n-gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--out-pca", default=str(REPO / "models/rf_postproc_pca.pkl"))
    ap.add_argument("--out-nn", default=str(REPO / "models/rf_postproc_nnhead.pkl"))
    a = ap.parse_args()

    from ADCNN.inference.features import RF_FEATURES_V2
    from ADCNN.inference.rf_train import train_rf
    from ADCNN.inference.rf_postproc import save_rf
    from ADCNN.evaluation.catalog_match import evaluate_catalog
    feats = list(RF_FEATURES_V2)

    n_gpus = a.n_gpus or max(1, torch.cuda.device_count())
    work = build_work(a.n_val_panels, a.eval_panels)
    shards = [work[g::n_gpus] for g in range(n_gpus)]
    print(f"=== {len(work)} panels ({a.n_val_panels} train + test) across {n_gpus} GPUs ===", flush=True)

    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gpu_worker, args=(g, shards[g], a.v7, q))
             for g in range(n_gpus) if shards[g]]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    Xp = np.concatenate([r[0] for r in results])
    Xn = np.concatenate([r[1] for r in results])
    y = np.concatenate([r[2] for r in results])
    ev = pd.concat([r[3] for r in results], ignore_index=True)
    print(f"[pool] {len(y)} candidates ({int(y.sum())} pos / {int((y == 0).sum())} neg)", flush=True)

    rf_pca = train_rf(Xp, y, neg_ratio=a.neg_ratio, seed=2026)
    rf_nn = train_rf(Xn, y, neg_ratio=a.neg_ratio, seed=2026)
    save_rf(rf_pca, a.out_pca); save_rf(rf_nn, a.out_nn)
    print(f"saved -> {a.out_pca} , {a.out_nn}", flush=True)
    for tag, rf in (("pca", rf_pca), ("nnhead", rf_nn)):
        imp = rf.feature_importances_
        ob = imp[feats.index("or_beta")]
        osnr = sum(imp[feats.index(f"or_snr_L{L}")] for L in (30, 50, 80))
        print(f"[imp:{tag:6s}] or_beta={ob:.4f}  sum(or_snr_L*)={osnr:.4f}", flush=True)

    print("\n=== eval on synthetic test sets (recall, FP/panel) ===", flush=True)
    print(f"{'set':12s} {'thr':>4s} | {'pca recall':>11s} {'pca fp/pan':>10s} | "
          f"{'nn recall':>10s} {'nn fp/pan':>9s}")
    thrs = [0.3, 0.5]
    for s in TESTSETS:
        truth = pd.read_csv(DATA / s / "test.csv")
        if a.eval_panels:
            truth = truth[truth.image_id.isin(sorted(truth.image_id.unique())[:a.eval_panels])]
        for variant, rf in (("pca", rf_pca), ("nn", rf_nn)):
            m = ev[(ev["__set"] == s) & (ev["__variant"] == variant)].copy()
            m["score_rf"] = rf.predict_proba(m[feats].fillna(0.0).to_numpy(np.float32))[:, 1]
            globals()[f"_res_{variant}"] = {
                thr: evaluate_catalog(m[m.score_rf >= thr], truth, tol_px=10.0)[0] for thr in thrs}
        for thr in thrs:
            rp = globals()["_res_pca"][thr]; rn = globals()["_res_nn"][thr]
            print(f"{s:12s} {thr:4.1f} | {rp['recall']*100:10.1f}% {rp['fp_per_panel']:10.2f} | "
                  f"{rn['recall']*100:9.1f}% {rn['fp_per_panel']:9.2f}", flush=True)
    print("ORIENT_COMPARE DONE", flush=True)


if __name__ == "__main__":
    main()
