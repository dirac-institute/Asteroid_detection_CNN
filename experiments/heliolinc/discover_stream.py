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


def run_shard(gpu_id, rows, v7_ckpt, rf_pkl, rf_thr, prefetch, out_csv):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.rf_postproc import compute_v2_features, apply_rf_v2, load_rf, RF_FEATURES_V2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(v7_ckpt, map_location=dev).eval()
    rf = load_rf(rf_pkl)
    paths = [r["fits_path"] for r in rows]
    out = []
    for i, (img, wcs, mjd) in _prefetch(paths, prefetch):
        r = rows[i]
        rl = np.zeros(img.shape, dtype=np.uint16)
        prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
        cand, _ = compute_v2_features(prob[None], img[None], sin[None], cos[None], agg[None],
                                      real_labels=rl[None], verbose=False)
        if not len(cand):
            continue
        feat = [c for c in RF_FEATURES_V2 if c in cand.columns]
        cand[feat] = cand[feat].replace([np.inf, -np.inf], np.nan)
        keep = apply_rf_v2(cand, rf)
        keep = keep[keep.score_rf >= rf_thr]
        if not len(keep):
            continue
        sky = wcs.all_pix2world(keep[["x_centroid", "y_centroid"]].to_numpy(np.float64), 0)
        for (ra, dec), (_, c) in zip(sky, keep.iterrows()):
            out.append(dict(mjd=mjd, ra=float(ra), dec=float(dec), mag=21.0, band=str(r["band"])[:1] or "r",
                            obscode=OBSCODE, visit=int(r["visit"]), detector=int(r["detector"]),
                            x=float(c.x_centroid), y=float(c.y_centroid), score_rf=float(c.score_rf)))
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"[gpu{gpu_id}] {len(rows)} panels -> {len(out)} detections", flush=True)


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

    if n_gpus == 1:
        run_shard(0, shards[0], a.v7, a.rf, a.rf_thr, a.prefetch, shard_csvs[0])
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=run_shard,
                             args=(g, shards[g], a.v7, a.rf, a.rf_thr, a.prefetch, shard_csvs[g]))
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
    for c in shard_csvs:
        Path(c).unlink(missing_ok=True)
    print(f"[discover] {len(cat)} detections over {man.shape[0]} panels (thr={a.rf_thr}) -> {a.out}", flush=True)
    print(f"[discover] {cat.visit.nunique()} visits, {len({int(str(v)[:8]) for v in cat.visit})} nights", flush=True)


if __name__ == "__main__":
    main()
