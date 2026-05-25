"""Step 2 (asteroid_cnn, GPU): STREAM difference_image FITS straight from the Butler datastore
into the two-stage detector and emit a HelioLinC detection catalog -- no intermediate dataset,
no big files, bounded memory.

Given the manifest of FITS paths (butler_manifest.py), the panels are sharded across the visible
GPUs. Each GPU process prefetches FITS with a small thread pool (astropy reads image HDU 1 + WCS +
MJD directly -- validated bit-identical to the lsst stack, WCS agree to 0.001") so disk I/O hides
behind the GPU, then runs v7 -> 72-feature RF (deployed operating point, thr 0.5) on each panel and
converts kept detections (x,y) -> (RA,Dec) via the panel's own WCS. Output: adcnn_dets.csv
[detid, mjd, ra, dec, mag, band, obscode, visit, detector, x, y, score_rf] + colformat.txt.
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings
# Cap BLAS/OMP threads BEFORE numpy import. We run n_gpus shard processes per node, each calling
# compute_v2_features (NumPy/BLAS-heavy); with default threading each shard grabs ALL cores ->
# n_gpus×ncores oversubscription (observed loadavg ~121 on a 32-core node, GPUs starved at ~70%).
# It was never filesystem I/O (jobs run on different nodes, iowait≈0) — it's CPU oversubscription.
# Default 6 threads/shard (4 shards × 6 = 24 ≤ 32 cores); override with ADCNN_BLAS_THREADS.
_NTHREAD = os.environ.get("ADCNN_BLAS_THREADS", "6")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NTHREAD)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))  # so spawned workers can import ADCNN regardless of cwd
OBSCODE = "I11"  # Rubin Observatory / LSST
COLFORMAT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"


def read_fits_panel(path: str):
    """Read one diffim FITS directly: (image float32, astropy WCS, mjd-mid). HDU1=IMAGE (validated)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            img = np.nan_to_num(hdul[1].data.astype(np.float32))
            wcs = WCS(hdul[1].header)
            h0 = hdul[0].header
            mjd = h0.get("DATE-AVG") or h0.get("MJD-AVG") or h0.get("MJD-OBS") or h0.get("MJD-BEG")
            if isinstance(mjd, str):  # DATE-AVG is ISO; convert
                from astropy.time import Time
                mjd = Time(mjd, format="isot").mjd
    return img, wcs, float(mjd)


def _prefetch(paths, workers):
    """Yield (idx, (img, wcs, mjd)) in submission order, prefetching up to `workers` ahead."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        nxt = 0
        for i in range(min(workers * 2, len(paths))):
            futs[i] = ex.submit(read_fits_panel, paths[i])
        submitted = len(futs)
        while nxt < len(paths):
            res = futs.pop(nxt).result()
            if submitted < len(paths):  # keep the pipeline full
                futs[submitted] = ex.submit(read_fits_panel, paths[submitted]); submitted += 1
            yield nxt, res
            nxt += 1


def run_shard(gpu_id, rows, v7_ckpt, rf_pkl, rf_thr, prefetch, out_csv, feat_out=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.rf_postproc import compute_v2_features, apply_rf_v2, load_rf, RF_FEATURES_V2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(v7_ckpt, map_location=dev).eval()
    rf = load_rf(rf_pkl)
    # RESUME: skip (visit,detector) panels already processed (sidecar .done log) so a preempted +
    # requeued job continues instead of losing everything. Detections are appended per panel.
    done_path = out_csv + ".done"
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            p = line.strip().split(",")
            if len(p) == 2:
                done.add((int(p[0]), int(p[1])))
    rows = [r for r in rows if (int(r["visit"]), int(r["detector"])) not in done]
    write_header = not (os.path.exists(out_csv) and os.path.getsize(out_csv) > 0)
    donef = open(done_path, "a")
    paths = [r["fits_path"] for r in rows]
    out, feats, n_done = [], [], len(done)
    for i, (img, wcs, mjd) in _prefetch(paths, prefetch):
        r = rows[i]
        rl = np.zeros(img.shape, dtype=np.uint16)
        prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
        cand, _ = compute_v2_features(prob[None], img[None], sin[None], cos[None], agg[None],
                                      real_labels=rl[None], gate_pmax=0.10, verbose=False)
        if not len(cand):
            donef.write(f'{int(r["visit"])},{int(r["detector"])}\n'); donef.flush()
            continue
        feat = [c for c in RF_FEATURES_V2 if c in cand.columns]
        cand[feat] = cand[feat].replace([np.inf, -np.inf], np.nan)
        cand = apply_rf_v2(cand, rf)
        xy = cand[["x_centroid", "y_centroid"]].to_numpy(np.float64)
        sky_all = wcs.all_pix2world(xy, 0)
        # Trail ENDPOINTS for trail->tracklet linking: half-length along mf_beta (PCA), with the
        # +~30px ADCNN ends-bloom removed (L_true ≈ (mf_length-33.4)/0.887; Veres-validated) so the
        # endpoint separation matches the true on-sky motion over the exposure. -> RA/Dec via WCS.
        beta_rad = np.radians(cand.get("mf_beta", pd.Series(np.zeros(len(cand)))).to_numpy(np.float64))
        L_db = np.clip((cand.get("mf_length", pd.Series(np.zeros(len(cand)))).to_numpy(np.float64) - 33.4) / 0.887, 0, None)
        hdx = 0.5 * L_db * np.cos(beta_rad); hdy = 0.5 * L_db * np.sin(beta_rad)
        sky0 = wcs.all_pix2world(np.stack([xy[:, 0] - hdx, xy[:, 1] - hdy], 1), 0)
        sky1 = wcs.all_pix2world(np.stack([xy[:, 0] + hdx, xy[:, 1] + hdy], 1), 0)
        cand = cand.assign(ra=sky_all[:, 0], dec=sky_all[:, 1], mjd=mjd,
                           ra0=sky0[:, 0], dec0=sky0[:, 1], ra1=sky1[:, 0], dec1=sky1[:, 1],
                           len_db=L_db,
                           visit=int(r["visit"]), detector=int(r["detector"]), band=str(r["band"])[:1] or "r")
        if feat_out is not None:  # dump EVERY candidate's features+sky for RF (re)training
            feats.append(cand)
        keep = cand[cand.score_rf >= rf_thr]
        prows = [dict(mjd=mjd, ra=float(c.ra), dec=float(c.dec), mag=21.0, band=c.band,
                      obscode=OBSCODE, visit=int(r["visit"]), detector=int(r["detector"]),
                      x=float(c.x_centroid), y=float(c.y_centroid), score_rf=float(c.score_rf),
                      length=float(c.get("mf_length", np.nan)), len_db=float(c.len_db),
                      mf_snr=float(c.get("mf_snr", np.nan)),
                      ra0=float(c.ra0), dec0=float(c.dec0), ra1=float(c.ra1), dec1=float(c.dec1),
                      # trail PA from footprint PCA (or_beta = NN head is r≈0 vs truth, kept as diagnostic)
                      beta=float(c.get("mf_beta", np.nan)), beta_nn=float(c.get("or_beta", np.nan)),
                      nn_pmax=float(c.get("max_p", np.nan))) for _, c in keep.iterrows()]
        if prows:   # append THIS panel's detections immediately (preemption-safe)
            pd.DataFrame(prows).to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False; out.extend(prows)
        donef.write(f'{int(r["visit"])},{int(r["detector"])}\n'); donef.flush()
    donef.close()
    if feat_out is not None and feats:
        pd.concat(feats, ignore_index=True).to_parquet(feat_out, index=False)
    print(f"[gpu{gpu_id}] {len(rows)} new panels (+{n_done} done before) -> {len(out)} new detections"
          f"{' (+features dumped)' if feat_out else ''}", flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "experiments/heliolinc/run_disco/manifest.csv"))
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--rf", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--rf-thr", type=float, default=0.5, help="deployed operating point (eval-consistent)")
    ap.add_argument("--prefetch", type=int, default=6, help="FITS reads in flight per GPU (bounds memory)")
    ap.add_argument("--n-gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/run_disco/adcnn_dets.csv"))
    ap.add_argument("--dump-features", default=None,
                    help="also write EVERY candidate's 72 features + ra/dec/mjd here (parquet, for RF retraining)")
    ap.add_argument("--limit", type=int, default=0, help="first N panels only (smoke test)")
    a = ap.parse_args()

    man = pd.read_csv(a.manifest)
    if a.limit:
        man = man.head(a.limit)
    n_gpus = a.n_gpus or max(1, torch.cuda.device_count())
    shards = [man.iloc[g::n_gpus].to_dict("records") for g in range(n_gpus)]
    tmp = Path(a.out).parent
    tmp.mkdir(parents=True, exist_ok=True)
    shard_csvs = [str(tmp / f"_shard{g}.csv") for g in range(n_gpus)]
    feat_pqs = [str(tmp / f"_feat{g}.parquet") if a.dump_features else None for g in range(n_gpus)]

    if n_gpus == 1:
        run_shard(0, shards[0], a.v7, a.rf, a.rf_thr, a.prefetch, shard_csvs[0], feat_pqs[0])
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=run_shard,
                             args=(g, shards[g], a.v7, a.rf, a.rf_thr, a.prefetch, shard_csvs[g], feat_pqs[g]))
                 for g in range(n_gpus) if shards[g]]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    cat = pd.concat([pd.read_csv(c) for c in shard_csvs if Path(c).exists() and os.path.getsize(c) > 1],
                    ignore_index=True)
    cat = cat.sort_values(["mjd", "visit", "detector"]).reset_index(drop=True)
    cat.insert(0, "detid", range(len(cat)))
    cat.to_csv(a.out, index=False)
    (Path(a.out).parent / "colformat.txt").write_text(COLFORMAT)
    if a.dump_features:
        feats = pd.concat([pd.read_parquet(f) for f in feat_pqs if f and Path(f).exists()], ignore_index=True)
        feats.to_parquet(a.dump_features, index=False)
        print(f"[discover] dumped {len(feats)} candidate feature rows -> {a.dump_features}", flush=True)
        for f in feat_pqs:
            if f:
                Path(f).unlink(missing_ok=True)
    for c in shard_csvs:
        Path(c).unlink(missing_ok=True)
        Path(c + ".done").unlink(missing_ok=True)   # resume sidecars (only here, after success)
    print(f"[discover] {len(cat)} detections over {man.shape[0]} panels (thr={a.rf_thr}) -> {a.out}", flush=True)
    print(f"[discover] {cat.visit.nunique()} visits, {len({int(str(v)[:8]) for v in cat.visit})} nights", flush=True)


if __name__ == "__main__":
    main()
