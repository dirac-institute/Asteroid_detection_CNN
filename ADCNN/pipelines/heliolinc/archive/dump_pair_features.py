"""Dump per-pair features (chi2 + bound-orbit a/e + the 5 chi2 components + chord rate + trail geometry)
for TRUE (injected) vs FALSE 2-visit pairs at a fixed score floor, to investigate NON-ML cuts that could
buy 3σ purity at ~75% completeness. The linker computes a,e but never cuts on orbital PHYSICALITY -- this
tests whether (a,e,q) or tighter component cuts separate true movers from chance FP chords.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from ADCNN.linking.link_2visit import chord_seed_pairs, pair_chi2
from ADCNN.pipelines.heliolinc.sweep_S import _load_field, PC


# fine artifact planes a REAL moving source should be ~0 on (per [[lsst-mask-fp-filter]] /
# [[fp-rejection-investigation]]); art_frac only thresholds the MAX, these are individually informative.
ART = ["m_SPIKE", "m_CR", "m_STREAK", "m_CROSSTALK", "m_DETECTED_NEGATIVE", "m_HIGH_VARIANCE",
       "m_ITL_DIP", "m_SAT", "m_EDGE", "m_SENSOR_EDGE", "m_SUSPECT", "m_INTRP", "m_CLIPPED"]
DENS_PX = 120.0   # local crowding radius (px) -- FP cluster in artifact-prone regions


def _field(args):
    from scipy.spatial import cKDTree
    d_dir, k, S = args
    _, d, recoverable = _load_field(d_dir, k, 6.0, 0.3, 2)
    artcols = [c for c in ART if c in d.columns]
    # per-panel k-d trees over ALL detections (any score) -> local density (crowding) per endpoint
    trees = {key: cKDTree(g[["x", "y"]].values) for key, g in d.groupby(["visit", "detector"])}
    ds = d[d.score >= S].reset_index(drop=True)
    oid = ds.objID.to_numpy(); sc = ds.score.to_numpy()
    ra = ds.ra.to_numpy(); dec = ds.dec.to_numpy(); mjd = ds.mjd.to_numpy()
    lenb = ds.len_db.to_numpy(); xs = ds.x.to_numpy(); ys = ds.y.to_numpy()
    vis = ds.visit.to_numpy(); det = ds.detector.to_numpy()
    mfsnr = ds.mf_snr.to_numpy() if "mf_snr" in ds else np.full(len(ds), np.nan)
    nnp = ds.nn_pmax.to_numpy() if "nn_pmax" in ds else np.full(len(ds), np.nan)
    artv = ds[artcols].to_numpy() if artcols else np.zeros((len(ds), 0))

    def dens(t):
        tree = trees.get((int(vis[t]), int(det[t])))
        return (len(tree.query_ball_point([xs[t], ys[t]], DENS_PX)) - 1) if tree is not None else 0

    rows = []
    for i, j in chord_seed_pairs(ds, max_arc_min=PC["max_arc_2v_min"]):
        g = ds.iloc[[i, j]]
        chi2, info = pair_chi2(g)
        if not np.isfinite(chi2):
            continue
        dt = abs(mjd[j] - mjd[i]); cd = np.cos(np.radians(dec[i]))
        sep = np.hypot((ra[j] - ra[i]) * cd, dec[j] - dec[i])
        rate = sep / dt if dt > 0 else np.nan
        is_true = bool(pd.notna(oid[i]) and oid[i] == oid[j])
        obj = str(oid[i]) if is_true else ""
        q = info["a"] * (1 - info["e"]) if np.isfinite(info["a"]) and np.isfinite(info["e"]) else np.nan
        r = dict(field=k, is_true=is_true, obj=obj, chi2=chi2, a=info["a"], e=info["e"], q=q,
                 perp=info["perp"], resid=info["resid"], dsnr=info["dsnr"], dpa_tm=info["dpa_tm"],
                 dspeed=info["dspeed"], rate=rate, len_min=min(lenb[i], lenb[j]),
                 len_ratio=max(lenb[i], lenb[j]) / max(min(lenb[i], lenb[j]), 1), score_min=min(sc[i], sc[j]),
                 # DEEPER: photometry, secondary ADCNN, spatial crowding, fine mask proximity
                 mfsnr_min=np.nanmin([mfsnr[i], mfsnr[j]]), nnp_min=np.nanmin([nnp[i], nnp[j]]),
                 dens_max=max(dens(i), dens(j)), dens_min=min(dens(i), dens(j)))
        if artcols:
            a_i = artv[i]; a_j = artv[j]
            r["art_sum_max"] = float(max(a_i.sum(), a_j.sum()))       # total artifact-plane proximity
            r["art_any_max"] = float(max(a_i.max(), a_j.max()))       # = the art_frac (<0.3 by construction)
            r["m_detneg_max"] = float(max(a_i[artcols.index("m_DETECTED_NEGATIVE")] if "m_DETECTED_NEGATIVE" in artcols else 0,
                                          a_j[artcols.index("m_DETECTED_NEGATIVE")] if "m_DETECTED_NEGATIVE" in artcols else 0))
        rows.append(r)
    return rows, list(recoverable.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--score", type=float, default=0.80)
    ap.add_argument("--n-fields", type=int, default=20)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ks = [f.split("adcnn_dets_masked_")[1].split(".csv")[0]
          for f in sorted(glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv"))][:a.n_fields]
    print(f"[dump] {len(ks)} fields at S>={a.score}", flush=True)
    rows = []; recoverable = set()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(_field, (a.dir, k, a.score)) for k in ks]):
            r, rec = fut.result(); rows.extend(r); recoverable.update(rec)
    df = pd.DataFrame(rows)
    df.attrs["n_recoverable"] = len(recoverable)
    out = a.out or f"{a.dir}/pair_features_S{int(a.score*100)}.parquet"
    df.to_parquet(out)
    Path(out + ".meta").write_text(str(len(recoverable)))   # recoverable denominator for completeness
    print(f"[dump] recoverable (>=2 sighting) injected objects across {len(ks)} fields: {len(recoverable)}", flush=True)
    nt = int(df.is_true.sum()); nf = int((~df.is_true).sum())
    print(f"[dump] {len(df)} pairs passing pre-gate: {nt} TRUE, {nf} FALSE -> {out}", flush=True)
    # chi2<=3.0 survivors (the shipped op-point) -- the set we must purify
    s = df[df.chi2 <= 3.0]
    print(f"[dump] chi2<=3.0 survivors: {int(s.is_true.sum())} true / {int((~s.is_true).sum())} false", flush=True)
    print("DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
