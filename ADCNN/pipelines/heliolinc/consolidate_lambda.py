"""Build the master per-detection catalog for the lambda campaign and publish the deliverables into the
Butler collection u/mrakovci/ADCNN/samenight_link_lambda.

Master catalog = every ADCNN detection over all fields, annotated with: re-timed mjd, injected objID + target
SNR (truth), and a stack-5sigma coincidence flag (did Rubin's 5sigma SourceDetection also find it, within
tol_px). This is the "what stack detected vs what ADCNN detected" record. Then publish it + lambda(S) curve
+ S* result via register_collection.publish().
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

from ADCNN.pipelines.heliolinc.retime_cadence import apply_retime
from ADCNN.pipelines.heliolinc.sweep_S import label_injected
from ADCNN.pipelines.heliolinc import register_collection as reg


def merge_stack(d, stack):
    """Tag each INJECTED-matched detection with whether the 5sigma stack recovered that sighting (per
    visit,detector,objID). Real-FP detections get stack5=NaN (we don't dump raw 5sigma FP positions --
    only their per-panel density)."""
    d = d.copy(); d["stack5"] = np.nan
    if stack is None or not len(stack) or "objID" not in stack:
        return d
    key = {(int(v), int(det), str(o)): bool(s)
           for v, det, o, s in zip(stack.visit, stack.detector, stack.objID, stack.stack_det)}
    mask = d.objID.notna()
    d.loc[mask, "stack5"] = [key.get((int(v), int(det), str(o)), np.nan)
                             for v, det, o in zip(d.loc[mask, "visit"], d.loc[mask, "detector"], d.loc[mask, "objID"])]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--publish", action="store_true", help="publish to Butler collection (else just write parquet)")
    a = ap.parse_args()
    rundir = Path(a.dir)

    keep = ["field", "visit", "detector", "mjd", "x", "y", "ra", "dec", "beta", "len_db",
            "score", "art_frac", "objID", "snr_target", "stack5"]
    parts = []
    for f in sorted(glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv")):
        k = f.split("adcnn_dets_masked_")[1].split(".csv")[0]
        d = pd.read_csv(f)
        rmf, injf, stf = f"{a.dir}/retime_{k}.csv", f"{a.dir}/inject_{k}.csv", f"{a.dir}/stack_dets_{k}.csv"
        if Path(rmf).exists():
            d = apply_retime(d, pd.read_csv(rmf))
        inj = pd.read_csv(injf) if Path(injf).exists() else None
        d = label_injected(d, inj)
        d = d.merge(inj[["objID", "snr_target"]].drop_duplicates("objID"), on="objID", how="left") if inj is not None else d.assign(snr_target=np.nan)
        d = merge_stack(d, pd.read_csv(stf) if Path(stf).exists() else None)
        d["field"] = int(k)
        for c in keep:
            if c not in d:
                d[c] = np.nan
        parts.append(d[keep])
        print(f"[consolidate] field {k}: {len(d)} dets, {int(d.objID.notna().sum())} injected-matched, "
              f"{int(d.stack5.sum())} stack-5sigma", flush=True)
    master = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)
    master.to_parquet(rundir / "master_detections.parquet")
    print(f"[consolidate] master: {len(master)} detections -> master_detections.parquet", flush=True)

    if a.publish:
        reg.publish("samenight_lambda_detections", master)
        for fn, name in [("lambda_vs_S.csv", "samenight_lambda_curve"), ("s_star.csv", "samenight_lambda_result")]:
            p = rundir / fn
            if p.exists():
                reg.publish(name, pd.read_csv(p))
    print("CONSOLIDATE_DONE")


if __name__ == "__main__":
    main()
