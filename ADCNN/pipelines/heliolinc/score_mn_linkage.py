#!/usr/bin/env python3
"""Score a heliolinx multi-night linkage run on run_band real data vs the known-asteroid truth.

Inputs (per --tag): run_band/mn_box/dets_<tag>.csv (the catalog fed to heliolinx, ordered),
LPclust2det_<tag>.csv (linkage_id -> detection row index into dets file). Truth = run_band/known.csv.

A linkage is TRUE if its detections crossmatch (within tol, per visit) to a SINGLE known ObjID on a
majority of member detections; else FALSE (chance link). Completeness = distinct true knowns recovered /
knowns observed AND ADCNN-detected on >=3 nights inside the box at this score floor. Reports completeness,
false-link count, purity, and runtime context.
"""
import sys, argparse
import pandas as pd, numpy as np
from scipy.spatial import cKDTree
from ADCNN.pipelines.heliolinc.trail_state_link import radec_to_unit, _chord_radius


def _nearest_vk(mjd, kvks, tol_day=35.0 / 86400.0):
    """Snap each mjd to the NEAREST known visit-key within tol (handles trail-as-tracklet endpoint
    pseudo-obs at mjd +/- exptime/2, which break exact visit-key equality). Returns vk array (NaN = none)."""
    kv = np.sort(np.unique(kvks))
    i = np.clip(np.searchsorted(kv, mjd), 1, len(kv) - 1)
    lo, hi = kv[i - 1], kv[i]
    near = np.where(np.abs(mjd - lo) <= np.abs(mjd - hi), lo, hi)
    return np.where(np.abs(mjd - near) <= tol_day, near, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="run_band/mn_box")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--known", default="run_band/known.csv")
    ap.add_argument("--dets-file", default=None,
                    help="explicit input-dets CSV for the truth denominator (default {dir}/dets_{tag}.csv; "
                         "NOTE run_heliolinx overwrites that name with its heliolinx-format file -- pass the "
                         "original culled catalog here for trail-as-tracklet runs)")
    ap.add_argument("--tol-arcsec", type=float, default=5.0)
    ap.add_argument("--out-json", default=None,
                    help="write {recovered:[ObjID...], n_true, n_false, nlink, truth3:[...]} for chain union")
    a = ap.parse_args()

    dets = pd.read_csv(a.dets_file or f"{a.dir}/dets_{a.tag}.csv")   # ordered as fed; row i == heliolinx index i
    dets["night"] = np.floor(dets.mjd - 0.5).astype(int)
    k = pd.read_csv(a.known); k["night"] = np.floor(k.mjd - 0.5).astype(int); k["vk"] = k.mjd.round(5)
    dets["vk"] = _nearest_vk(dets.mjd.to_numpy(), k.vk.to_numpy())
    # restrict known to the box footprint (by RA/Dec span of dets) so the denominator is in-box
    ramin, ramax = dets.ra.min() - 0.05, dets.ra.max() + 0.05
    dmin, dmax = dets.dec.min() - 0.05, dets.dec.max() + 0.05
    k = k[(k.ra.between(ramin, ramax)) & (k.dec.between(dmin, dmax))].reset_index(drop=True)
    tol = _chord_radius(a.tol_arcsec / 3600.0)

    # label each det row with its matched ObjID (per visit KD match)
    det_obj = np.full(len(dets), None, dtype=object)
    didx = dict(tuple(dets.reset_index().groupby("vk")))
    for vk, kg in k.groupby("vk"):
        if vk not in didx:
            continue
        dg = didx[vk]
        tree = cKDTree(radec_to_unit(kg.ra.to_numpy(), kg.dec.to_numpy()))
        dd, ii = tree.query(radec_to_unit(dg.ra.to_numpy(), dg.dec.to_numpy()), k=1)
        oid = kg.ObjID.to_numpy()
        for di, h, jj in zip(dg["index"].to_numpy(), dd <= tol, ii):
            if h:
                det_obj[di] = oid[jj]
    dets["obj"] = det_obj

    # TRUTH denominator: knowns ADCNN-detected (matched) on >=3 distinct nights inside the box
    md = dets[dets.obj.notna()]
    det3 = md.groupby("obj").night.nunique()
    truth3 = set(det3[det3 >= 3].index)
    # also Rubin-observed >=3 nights in box (the cadence ceiling)
    obs3 = set((k.groupby("ObjID").night.nunique() >= 3).pipe(lambda s: s[s].index))

    # read linkages. LPclust2det maps linkage_id -> PAIRDETS row index (NOT input dets order).
    # pairdets carries its own MJD/RA/Dec -> crossmatch those to known directly (same as h2h_metrics).
    import os
    lp = f"{a.dir}/LPclust2det_{a.tag}.csv"
    pdf_path = f"{a.dir}/pairdets_{a.tag}.csv"
    if not os.path.exists(lp):
        print(f"NO LINKAGE FILE {lp} (link_purify produced none or failed)"); return
    pdf = pd.read_csv(pdf_path)
    pdf.columns = [c.lstrip("#") for c in pdf.columns]       # '#MJD' -> 'MJD'
    pdf["vk"] = _nearest_vk(pdf.MJD.to_numpy(), k.vk.to_numpy())
    # label each pairdets row with matched ObjID (per visit KD match to known)
    pobj = np.full(len(pdf), None, dtype=object)
    pidx = dict(tuple(pdf.reset_index().groupby("vk")))
    for vk, kg in k.groupby("vk"):
        if vk not in pidx:
            continue
        pg = pidx[vk]
        tree = cKDTree(radec_to_unit(kg.ra.to_numpy(), kg.dec.to_numpy()))
        dd, ii = tree.query(radec_to_unit(pg.RA.to_numpy(), pg.Dec.to_numpy()), k=1)
        oid = kg.ObjID.to_numpy()
        for pi, h, jj in zip(pg["index"].to_numpy(), dd <= tol, ii):
            if h:
                pobj[pi] = oid[jj]
    c2d = pd.read_csv(lp)
    c2d.columns = [c.lstrip("#") for c in c2d.columns]       # '#clusternum' -> 'clusternum'
    clcol, dtcol = c2d.columns[0], c2d.columns[1]
    nlink = c2d[clcol].nunique()
    true_objs = set(); n_false = 0; n_true = 0
    for cl, g in c2d.groupby(clcol):
        idx = g[dtcol].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < len(pdf))]
        objs = pobj[idx]
        objs = objs[pd.notna(objs)]
        if len(objs) and (np.sum(objs == pd.Series(objs).mode()[0]) >= max(2, 0.6 * len(idx))):
            true_objs.add(pd.Series(objs).mode()[0]); n_true += 1
        else:
            n_false += 1
    comp = len(true_objs) / max(len(truth3), 1)
    purity = n_true / max(nlink, 1)
    print(f"=== tag={a.tag} ===")
    print(f"dets fed: {len(dets)}  in-box knowns Rubin-obs>=3n: {len(obs3)}  ADCNN-det>=3n (truth denom): {len(truth3)}")
    print(f"linkages: {nlink}  TRUE(real obj): {n_true}  FALSE(chance): {n_false}")
    print(f"distinct knowns recovered: {len(true_objs)}")
    print(f"COMPLETENESS (vs ADCNN-det>=3n) = {comp:.3f}   PURITY = {purity:.3f}   false-tracks = {n_false}")
    if a.out_json:
        import json
        json.dump(dict(tag=a.tag, recovered=sorted(str(o) for o in true_objs), n_true=int(n_true),
                       n_false=int(n_false), nlink=int(nlink), truth3=sorted(str(o) for o in truth3)),
                  open(a.out_json, "w"))
        print(f"-> {a.out_json}")


if __name__ == "__main__":
    main()
