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
# Pin to 1: parallelism comes from the per-GPU feature-WORKER POOL (n_gpus×n_workers ≈ ncores,
# each worker 1 BLAS thread). Workers inherit this env, so 1 thread each -> no oversubscription.
_NTHREAD = os.environ.get("ADCNN_BLAS_THREADS", "1")
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


def run_shard(gpu_id, rows, v7_ckpt, rf_pkl, rf_thr, prefetch, out_csv, n_workers=8, feat_out=None):
    """Stream FITS -> v7 (GPU) -> feature+RF PROCESS POOL (CPU) -> sky catalog. The GPU runs v7
    inference continuously while `n_workers` CPU processes compute the 72 features + RF in parallel
    across panels (catalog.py's engine) -> GPU and all cores stay busy (the old inline-per-shard
    version left the GPU idle ~90% while one process computed features). Detections + a .done log
    are written incrementally (preemption-safe / resumable)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import multiprocessing as mp
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.catalog import _worker, _worker_init, InferenceConfig
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(v7_ckpt, map_location=dev).eval()
    config = InferenceConfig(rf_thr=rf_thr, gate_pmax=0.10)
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
    n_done = len(done); n_det = 0
    pending = deque()

    def drain():
        """Pop one finished feature-worker result, convert pixels->sky (+ trail endpoints), append."""
        nonlocal write_header, n_det
        fut, r, wcs, mjd = pending.popleft()
        cand = fut.result()   # catalog public schema (x,y,beta=PCA,length,score_rf>=thr) or None
        if cand is not None and len(cand):
            xy = cand[["x", "y"]].to_numpy(np.float64)
            sky = wcs.all_pix2world(xy, 0)
            # Trail ENDPOINTS for tracklets: half-length along mf_beta (=cand.beta), +30px ADCNN
            # ends-bloom removed (L_true≈(mf_length-33.4)/0.887) so endpoint sep matches sky motion.
            beta_rad = np.radians(cand["beta"].to_numpy(np.float64))
            L_db = np.clip((cand["length"].to_numpy(np.float64) - 33.4) / 0.887, 0, None)
            hdx = 0.5 * L_db * np.cos(beta_rad); hdy = 0.5 * L_db * np.sin(beta_rad)
            sky0 = wcs.all_pix2world(np.stack([xy[:, 0] - hdx, xy[:, 1] - hdy], 1), 0)
            sky1 = wcs.all_pix2world(np.stack([xy[:, 0] + hdx, xy[:, 1] + hdy], 1), 0)
            out = pd.DataFrame(dict(
                mjd=mjd, ra=sky[:, 0], dec=sky[:, 1], mag=21.0, band=str(r["band"])[:1] or "r",
                obscode=OBSCODE, visit=int(r["visit"]), detector=int(r["detector"]),
                x=xy[:, 0], y=xy[:, 1], score_rf=cand["score_rf"].to_numpy(),
                length=cand["length"].to_numpy(), len_db=L_db, mf_snr=cand["mf_snr"].to_numpy(),
                ra0=sky0[:, 0], dec0=sky0[:, 1], ra1=sky1[:, 0], dec1=sky1[:, 1],
                beta=cand["beta"].to_numpy(),
                beta_nn=cand.get("beta_nn", pd.Series(np.nan, index=cand.index)).to_numpy(),
                nn_pmax=cand["nn_pmax"].to_numpy()))
            out.to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False; n_det += len(out)
        donef.write(f'{int(r["visit"])},{int(r["detector"])}\n'); donef.flush()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(2, n_workers), mp_context=ctx,
                             initializer=_worker_init, initargs=(rf_pkl, config)) as pool:
        for i, (img, wcs, mjd) in _prefetch(paths, prefetch):
            rl = np.zeros(img.shape, dtype=np.uint16)
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            pending.append((pool.submit(_worker, (i, prob, img, sin, cos, agg, rl)), rows[i], wcs, mjd))
            if len(pending) >= 2 * max(2, n_workers):   # backpressure: bound RAM + queue depth
                drain()
        while pending:
            drain()
    donef.close()
    print(f"[gpu{gpu_id}] {len(rows)} new panels (+{n_done} done) -> {n_det} new detections", flush=True)


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
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 8
    n_workers = max(2, cores // n_gpus - 1)   # feature procs per GPU shard (n_gpus×n_workers ≈ cores)
    print(f"[discover] {n_gpus} GPUs × {n_workers} feature workers ({cores} cores)", flush=True)
    shards = [man.iloc[g::n_gpus].to_dict("records") for g in range(n_gpus)]
    tmp = Path(a.out).parent
    tmp.mkdir(parents=True, exist_ok=True)
    shard_csvs = [str(tmp / f"_shard{g}.csv") for g in range(n_gpus)]
    feat_pqs = [str(tmp / f"_feat{g}.parquet") if a.dump_features else None for g in range(n_gpus)]

    if n_gpus == 1:
        run_shard(0, shards[0], a.v7, a.rf, a.rf_thr, a.prefetch, shard_csvs[0], n_workers, feat_pqs[0])
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=run_shard,
                             args=(g, shards[g], a.v7, a.rf, a.rf_thr, a.prefetch, shard_csvs[g], n_workers, feat_pqs[g]))
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
