"""Score the REAL-asteroid test set (``DATA_DIFFIM/test_real``) with v7 + the
V2 RF post-processor and compare against the LSST stack.

The real catalog is *selection-biased bright* (it only contains discovered
asteroids), so its recall numbers do NOT measure the network's value — the
faint-regime value is established on the synthetic sets. ``test_real``'s
informative signal is the **second-stage gain** (asteroids the stack missed
that the network recovers) and the **false-positive rate on real difference
images**. The report therefore centres on stack-missed recovery + FP.

Streams panel-by-panel (the full (N,H,W) stack is far too big to hold) and
reuses the production path: ``predict_panel_overlap_3ch_full`` →
``compute_v2_features`` → ``apply_rf_v2`` → ``materialize_label_mask_v2`` →
``objectwise_confusion``. A failed panel is recorded as a *conservative
NN-miss* (never silently dropped, never kills a shard).

CLI (asteroid_cnn env, GPU)::

    # one shard of an array (writes results/parts/{sight,fp}_<shard>.csv)
    python -m ADCNN.evaluation.real_eval score --shard K --nshards M \\
        --data DATA_DIFFIM/test_real --model <scripted.pt> --rf <rf.pkl>
    # after all shards: merge -> summary.txt + per_sighting/per_panel csv
    python -m ADCNN.evaluation.real_eval merge --data DATA_DIFFIM/test_real
"""
from __future__ import annotations

import argparse
import glob
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from ADCNN.inference.diffim_eval import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    DEFAULT_THR, RF_FEATURES_V2, apply_rf_v2, compute_v2_features, load_rf,
    materialize_label_mask_v2)
from ADCNN.evaluation.detection import objectwise_confusion


# ======================================================================
# Per-shard scoring
# ======================================================================
def score_shard(data_dir, model_path, rf_path, *, shard=0, nshards=1,
                 thr=DEFAULT_THR, limit=0, results_dir=None):
    data = Path(data_dir)
    res = Path(results_dir) if results_dir else data / "results"
    (res / "parts").mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=dev).eval()
    rf = load_rf(rf_path)
    cat = pd.read_csv(data / "test.csv")
    panels = pd.read_csv(data / "panels.csv")
    cat_by_img = {int(i): g for i, g in cat.groupby("image_id")}

    ids = sorted(int(i) for i in panels["image_id"].unique())
    if limit:
        ids = ids[:limit]
    my = ids[shard::nshards] if nshards > 1 else ids
    print(f"[real_eval s{shard}/{nshards}] {len(my)}/{len(ids)} panels "
          f"model={Path(model_path).name} thr={thr}", flush=True)

    obj_rows, fp_rows = [], []
    t0 = time.time()
    with h5py.File(data / "test.h5", "r") as f:
        for n, idx in enumerate(my):
            prow = panels[panels["image_id"] == idx]
            if prow.empty:
                continue
            prow = prow.iloc[0]
            role = prow["role"]
            try:
                img = f["images"][idx][:].astype(np.float32)
                rl = f["real_labels"][idx][:]
                prob, sn, cs, ag = predict_panel_overlap_3ch_full(
                    model, img, rl, device=dev)
                prob = prob.astype(np.float32)
                cand, ppd = compute_v2_features(
                    {0: prob}, {0: img}, {0: sn.astype(np.float32)},
                    {0: cs.astype(np.float32)}, {0: ag.astype(np.float32)},
                    real_labels={0: rl}, verbose=False)
                if len(cand):
                    fc = list(RF_FEATURES_V2)
                    cand[fc] = cand[fc].replace([np.inf, -np.inf], np.nan)
                    cand = apply_rf_v2(cand, rf)
                    kept = cand[cand["score_rf"] >= thr].copy()
                else:
                    kept = cand
                H, W = prob.shape
                m1 = materialize_label_mask_v2(kept, ppd, (1, H, W))
                ncomp = int(len(np.unique(m1))
                            - (1 if (m1 == 0).any() else 0))
                if role == "asteroid" and idx in cat_by_img:
                    sub = cat_by_img[idx].copy()
                    sub["image_id"] = 0
                    _, ofp, _, catm = objectwise_confusion(
                        sub, m1, 0.5, use_threads=True, max_workers=4)
                    for _, r in catm.iterrows():
                        obj_rows.append({
                            "ObjID": r["ObjID"], "image_id": idx,
                            "visit": int(r["visit"]),
                            "detector": int(r["detector"]),
                            "trail_length": float(r["trail_length"]),
                            "speed_deg_day": float(r.get("speed_deg_day",
                                                         np.nan)),
                            "band": r.get("physical_filter", ""),
                            "nn_detected": bool(r["nn_detected"]),
                            "stack_detected": bool(r["stack_detection"]),
                            "failed": False})
                    fp_rows.append({"image_id": idx, "role": role,
                                    "nn_fp": int(ofp),
                                    "stack_dia": int(prow["n_dia"]),
                                    "failed": False})
                else:
                    fp_rows.append({"image_id": idx, "role": role,
                                    "nn_fp": ncomp,
                                    "stack_dia": int(prow["n_dia"]),
                                    "failed": False})
            except Exception as e:
                print(f"[s{shard} PANEL-FAIL idx={idx}] "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                if role == "asteroid" and idx in cat_by_img:
                    for _, r in cat_by_img[idx].iterrows():
                        obj_rows.append({
                            "ObjID": r["ObjID"], "image_id": idx,
                            "visit": int(r["visit"]),
                            "detector": int(r["detector"]),
                            "trail_length": float(r["trail_length"]),
                            "speed_deg_day": float(r.get("speed_deg_day",
                                                         np.nan)),
                            "band": r.get("physical_filter", ""),
                            "nn_detected": False,
                            "stack_detected": bool(r["stack_detection"]),
                            "failed": True})
                fp_rows.append({"image_id": idx, "role": role, "nn_fp": 0,
                                "stack_dia": int(prow["n_dia"]),
                                "failed": True})
            if n % 25 == 0 or n == len(my) - 1:
                el = time.time() - t0
                print(f"[s{shard} {n+1}/{len(my)}] panel={idx} role={role} "
                      f"({el:.0f}s {el/max(n+1,1):.1f}s/panel)", flush=True)
            if n % 50 == 0 and (obj_rows or fp_rows):
                pd.DataFrame(obj_rows).to_csv(
                    res / "parts" / f"sight_{shard}.csv", index=False)
                pd.DataFrame(fp_rows).to_csv(
                    res / "parts" / f"fp_{shard}.csv", index=False)
    pd.DataFrame(obj_rows).to_csv(res / "parts" / f"sight_{shard}.csv",
                                  index=False)
    pd.DataFrame(fp_rows).to_csv(res / "parts" / f"fp_{shard}.csv",
                                 index=False)
    print(f"[real_eval s{shard}] done: {len(obj_rows)} sightings, "
          f"{len(fp_rows)} panels", flush=True)
    if nshards == 1:
        merge(data_dir, results_dir=results_dir)


# ======================================================================
# Report
# ======================================================================
def build_report(obj: pd.DataFrame, fp: pd.DataFrame) -> str:
    L, P = [], None
    out = []
    P = out.append
    n_s = len(obj)
    s_nn = int(obj["nn_detected"].sum())
    s_st = int(obj["stack_detected"].sum())
    g = obj.groupby("ObjID").agg(nn=("nn_detected", "any"),
                                 st=("stack_detected", "any"))
    n_o = len(g)
    o_nn = int(g["nn"].sum()); o_st = int(g["st"].sum())
    o_nn_only = int((g["nn"] & ~g["st"]).sum())
    o_st_only = int((~g["nn"] & g["st"]).sum())
    o_both = int((g["nn"] & g["st"]).sum())
    o_neither = int((~g["nn"] & ~g["st"]).sum())
    emp = fp[fp["role"] == "empty"]; ast = fp[fp["role"] == "asteroid"]

    P("=" * 64)
    P("  test_real : v7 + V2 RF  vs  LSST stack @ 5sigma")
    P("=" * 64)
    P(f"panels: {len(fp)} (asteroid={len(ast)} empty={len(emp)})")
    P(f"asteroid sightings: {n_s}   unique objects: {n_o}")
    if "failed" in fp.columns and int(fp["failed"].sum()):
        P(f"  ({int(fp['failed'].sum())} panels failed -> conservative "
          f"NN-miss, not dropped)")
    P("")
    P("-- PER-OBJECT recovery (detected in >=1 sighting) --")
    P(f"  network (v7+V2) : {o_nn}/{n_o} ({100*o_nn/max(n_o,1):.1f}%)")
    P(f"  LSST stack      : {o_st}/{n_o} ({100*o_st/max(n_o,1):.1f}%)")
    P(f"  both={o_both}  NETWORK-ONLY={o_nn_only}  stack-only={o_st_only}  "
      f"neither={o_neither}")
    P("")
    P("-- PER-SIGHTING recovery --")
    P(f"  network {s_nn}/{n_s} ({100*s_nn/max(n_s,1):.1f}%)  "
      f"stack {s_st}/{n_s} ({100*s_st/max(n_s,1):.1f}%)")
    P("")
    P("-- FALSE POSITIVES on empty real diffims --")
    if len(emp):
        P(f"  NN FP/panel mean={emp['nn_fp'].mean():.2f} "
          f"median={emp['nn_fp'].median():.0f} max={emp['nn_fp'].max()} "
          f"total={int(emp['nn_fp'].sum())}")
        P(f"  empty panels with 0 NN FP: "
          f"{int((emp['nn_fp']==0).sum())}/{len(emp)}")
    if len(ast):
        P(f"  NN extra FP on asteroid panels: "
          f"mean={ast['nn_fp'].mean():.2f} total={int(ast['nn_fp'].sum())}")
    P("")
    P("###### SECOND-STAGE VALUE: NN on TOP of the 5sigma stack ######")
    miss = obj[~obj["stack_detected"]]
    P(f"  stack-missed sightings: {len(miss)}/{n_s}")
    if len(miss):
        rec = int(miss["nn_detected"].sum())
        P(f"  -> NN recovers {rec} ({100*rec/len(miss):.1f}%)  "
          f"<== pure 2nd-stage gain (sightings)")
    never = g[~g["st"]]
    P(f"  objects stack NEVER detected: {len(never)}/{n_o}")
    if len(never):
        P(f"  -> NN newly recovers {int(never['nn'].sum())} "
          f"({100*never['nn'].mean():.1f}%)  <== NEW objects")
    comb = int((g['nn'] | g['st']).sum())
    P(f"  combined recall {comb}/{n_o} ({100*comb/max(n_o,1):.1f}%) "
      f"vs stack-alone {o_st}/{n_o} -> +{comb-o_st} objects")
    P("")
    P("-- NN recovery of stack-MISSED sightings, by trail length --")
    if len(miss):
        m2 = miss.copy()
        m2["Lbin"] = pd.cut(m2["trail_length"], [0, 8, 12, 20, 40, 1e4],
                            labels=["<8", "8-12", "12-20", "20-40", ">40"])
        P(m2.groupby("Lbin", observed=True).agg(
            n=("nn_detected", "size"),
            nn_recall=("nn_detected", "mean")).round(3).to_string())
    return "\n".join(out)


def merge(data_dir, results_dir=None):
    data = Path(data_dir)
    res = Path(results_dir) if results_dir else data / "results"

    def _cat(pat):
        fs = sorted(glob.glob(str(res / "parts" / pat)))
        return (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
                if fs else pd.DataFrame())

    obj = _cat("sight_*.csv").drop_duplicates(["ObjID", "image_id"])
    fp = _cat("fp_*.csv").drop_duplicates(["image_id"])
    print(f"[merge] {len(obj)} sightings, {len(fp)} panels", flush=True)
    obj.to_csv(res / "per_sighting.csv", index=False)
    fp.to_csv(res / "per_panel_fp.csv", index=False)
    rep = build_report(obj, fp)
    print("\n" + rep)
    (res / "summary.txt").write_text(rep + "\n")
    print(f"\n[saved] {res}/summary.txt  per_sighting.csv  per_panel_fp.csv")


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser("real_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--data", required=True)
    sc.add_argument("--model", required=True)
    sc.add_argument("--rf", required=True)
    sc.add_argument("--shard", type=int, default=0)
    sc.add_argument("--nshards", type=int, default=1)
    sc.add_argument("--thr", type=float, default=DEFAULT_THR)
    sc.add_argument("--limit", type=int, default=0)
    sc.add_argument("--results-dir", default=None)
    mg = sub.add_parser("merge")
    mg.add_argument("--data", required=True)
    mg.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    if a.cmd == "score":
        score_shard(a.data, a.model, a.rf, shard=a.shard, nshards=a.nshards,
                    thr=a.thr, limit=a.limit, results_dir=a.results_dir)
    else:
        merge(a.data, results_dir=a.results_dir)


if __name__ == "__main__":
    main()
