"""Detect stage of the same-night pipeline (asteroid_cnn, GPU): STREAM difference-image FITS straight
from the Butler datastore into the two-stage ADCNN detector and emit the detection catalog -- no
intermediate dataset, bounded memory.

Given the manifest of FITS paths, the panels are sharded across the visible GPUs. Each GPU process
prefetches FITS with a small thread pool (one read per panel: IMAGE HDU + header WCS + DATE-AVG, and
the MASK HDU with its plane cards) so disk/S3 I/O hides behind the GPU, then runs segmentation model
-> focal-cutout CNN stage-2 filter on each panel and converts kept detections (x,y) -> (RA,Dec) via
the panel's own WCS, with trail endpoints (ra0,dec0,ra1,dec1) and the mask-plane fractions along each
trail (ADCNN.pipelines.usdf_night.mask_planes). Detections and a .done log are appended per panel to
a per-GPU shard, so a killed job resumes where it stopped.

The merge writes adcnn_dets.csv [detid,mjd,ra,dec,mag,band,obscode,visit,detector,x,y,score,length,
len_db,mf_snr,length_raw,ra0,dec0,ra1,dec1,beta,nn_pmax,m_elong,m_dipole,m_neg_ratio,
m_neg_blob,is_dipole,m_<PLANE>...,art_frac] only when every manifest panel is in a .done log; otherwise
it keeps the shards and exits 3. A panel whose FITS cannot be read is not marked done; it is logged to
<out>.unreadable (visit,detector,error) and retried by the next run. --allow-partial runs no detection
and merges the panels that are done (used after the last attempt).
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
# Cap BLAS/OMP threads BEFORE numpy import. Multiple shard processes per node each run the
# NumPy/BLAS-heavy candidate-extraction worker pool; with default threading each shard would
# grab all cores -> n_gpus x ncores oversubscription. Pin to 1: parallelism comes from the
# per-GPU feature-worker pool (n_gpus x n_workers ~= ncores, each worker on 1 BLAS thread).
_NTHREAD = os.environ.get("ADCNN_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _NTHREAD)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root: $ADCNN_REPO if set, else inferred from this file's location (ADCNN/inference/ -> root).
# Portable across deployments; only override ADCNN_REPO for a relocated checkout.
REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO))  # so spawned workers can import ADCNN regardless of cwd
from ADCNN.pipelines.usdf_night.mask_planes import COLUMNS as MASK_COLUMNS, fractions, plane_values

OBSCODE = "X05"  # Rubin Observatory (join writes the same code into hl_dets.csv)
DET_COLUMNS = ["mjd", "ra", "dec", "mag", "band", "obscode", "visit", "detector", "x", "y", "score",
               "length", "len_db", "mf_snr", "length_raw", "ra0", "dec0", "ra1", "dec1", "beta", "nn_pmax",
               "m_elong", "m_dipole", "m_neg_ratio", "m_neg_blob", "is_dipole"] + MASK_COLUMNS


def _wcs_any(hdr):
    """WCS from a diffim header: primary FITS-WCS or the alternate 'A' key if it is celestial.
    RAISES if neither is a sky WCS -- some DRP outputs write 'A' as a CTYPE='PIXEL' bookkeeping
    transform, and silently using that produced pixel-valued 'sky' coordinates."""
    from astropy.wcs import WCS as _W
    try:
        w = _W(hdr)
        if w.has_celestial:
            return w
    except Exception:
        pass
    w = _W(hdr, key="A")
    if not w.has_celestial:
        raise ValueError("no celestial WCS in FITS header")
    return w


def read_fits_panel(path: str):
    """Read one diffim panel in a single open: (image float32, astropy WCS, mjd-mid, mask, plane values).
    HDU1=IMAGE (validated). `path` is a Butler datastore URI: a local file path, or an s3:// URI
    fetched in-memory -- see ADCNN.inference.diffim_io.open_diffim. Image/WCS errors raise; a missing
    MASK HDU or plane cards give mask=values=None (logged), so the panel still yields detections
    with NaN mask columns."""
    from ADCNN.inference.diffim_io import open_diffim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open_diffim(path, memmap=False) as hdul:
            img = np.nan_to_num(hdul[1].data.astype(np.float32))
            wcs = _wcs_any(hdul[1].header)
            h0 = hdul[0].header
            mjd = h0.get("DATE-AVG") or h0.get("MJD-AVG") or h0.get("MJD-OBS") or h0.get("MJD-BEG")
            if isinstance(mjd, str):  # DATE-AVG is ISO; convert
                from astropy.time import Time
                mjd = Time(mjd, format="isot").mjd
            try:
                values = plane_values(hdul["MASK"].header)
                mask = np.asarray(hdul["MASK"].data)
            except Exception as e:
                print(f"[mask] {path}: {type(e).__name__}: {e} -> mask columns NaN", flush=True)
                mask = values = None
    return img, wcs, float(mjd), mask, values


def _prefetch(paths, workers):
    """Yield (idx, panel, error) in submission order, prefetching up to `workers` ahead.
    panel is read_fits_panel's tuple, or None with the error message when the read failed."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i in range(min(workers * 2, len(paths))):
            futs[i] = ex.submit(read_fits_panel, paths[i])
        submitted = len(futs)
        for nxt in range(len(paths)):
            try:
                res, err = futs.pop(nxt).result(), None
            except Exception as e:    # a corrupt/unreadable FITS must not kill the whole GPU shard
                res, err = None, f"{type(e).__name__}: {e}"
                print(f"[prefetch] skip panel {nxt} ({paths[nxt]}): {err}", flush=True)
            if submitted < len(paths):  # keep the pipeline full
                futs[submitted] = ex.submit(read_fits_panel, paths[submitted]); submitted += 1
            yield nxt, res, err


def _log_unreadable(path, visit, detector, err):
    """Append one unreadable panel to the <out>.unreadable log (visit,detector,error; no header)."""
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([int(visit), int(detector), err])


def run_shard(gpu_id, rows, seg_ckpt, cnn_model, thr, prefetch, out_csv, unreadable, n_workers=8):
    """Stream FITS -> segmentation model (GPU) -> feature + stage-2 focal-cutout-CNN PROCESS POOL (CPU) -> sky catalog.
    The stage-2 filter is the focal-cutout CNN (models/v2_D2/cnn_postproc.pt). The GPU runs segmentation model
    continuously while `n_workers` CPU processes compute features + the CNN in parallel across panels.
    Detections + a .done log are written incrementally (preemption-safe / resumable)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.catalog import _worker, _worker_init, InferenceConfig
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(seg_ckpt, map_location=dev).eval()
    # gate_pmax=0.10 is INERT at the shipped operating point and has been all along: over 2,703,888
    # detections across 15,198 panels, ZERO have max_p < 0.10, because the adaptive t_low in
    # candidates.py binds first at a measured 0.4447-0.4769 (mu+6sd) on real panels -- not at the
    # nominal t_low=0.05 the config advertises, and not at the 0.5 ceiling either. The knob is live
    # (it does cut at 0.50/0.90), it simply sits far below what binds. Left in place because removing
    # it changes nothing and the real operating point is the adaptive threshold, which is what should
    # be documented and calibrated.
    config = InferenceConfig(cnn_thr=thr, gate_pmax=0.10)
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
    n_done = len(done); n_det = 0; n_unread = 0; n_nomask = 0
    total = len(rows); t0 = time.time(); last = t0
    pending = deque()

    def drain():
        """Pop one finished feature-worker result, convert pixels->sky (+ trail endpoints, mask fractions), append."""
        nonlocal write_header, n_det
        fut, r, wcs, mjd, mask, values = pending.popleft()
        cand = fut.result()   # catalog public schema (catalog._COLMAP; score>=thr) or None
        if cand is not None and len(cand):
            xy = cand[["x", "y"]].to_numpy(np.float64)
            sky = wcs.all_pix2world(xy, 0)
            # Trail ENDPOINTS for tracklets: half-length along cand.beta. The catalog `length` is
            # ALREADY the physical trail length (catalog.panel_to_catalog_rows), so endpoint
            # separation matches the on-sky motion directly -- no re-correction here.
            beta = cand["beta"].to_numpy(np.float64)
            beta_rad = np.radians(beta)
            L_db = np.clip(cand["length"].to_numpy(np.float64), 0, None)
            hdx = 0.5 * L_db * np.cos(beta_rad); hdy = 0.5 * L_db * np.sin(beta_rad)
            sky0 = wcs.all_pix2world(np.stack([xy[:, 0] - hdx, xy[:, 1] - hdy], 1), 0)
            sky1 = wcs.all_pix2world(np.stack([xy[:, 0] + hdx, xy[:, 1] + hdy], 1), 0)
            out = dict(
                # mag is NOT set here: the GPU streaming detector has no PhotoCalib. NaN = not measured
                # (mf_snr below is the per-detection significance).
                mjd=mjd, ra=sky[:, 0], dec=sky[:, 1], mag=np.nan,
                band=(str(r["band"])[:1] if str(r["band"]).lower() not in ("nan", "") else "r"),
                obscode=OBSCODE, visit=int(r["visit"]), detector=int(r["detector"]),
                x=xy[:, 0], y=xy[:, 1], score=cand["score"].to_numpy(),
                length=cand["length"].to_numpy(), len_db=L_db, mf_snr=cand["mf_snr"].to_numpy(),
                length_raw=cand["length_raw"].to_numpy(),   # RAW footprint length
                ra0=sky0[:, 0], dec0=sky0[:, 1], ra1=sky1[:, 0], dec1=sky1[:, 1],
                beta=beta,
                nn_pmax=cand["nn_pmax"].to_numpy(),
                # DETECTION-TIME dipole/RING morphology (catalog._attach_dipole_morphology).
                m_elong=cand["m_elong"].to_numpy(), m_dipole=cand["m_dipole"].to_numpy(),
                m_neg_ratio=cand["m_neg_ratio"].to_numpy(), m_neg_blob=cand["m_neg_blob"].to_numpy(),
                is_dipole=cand["is_dipole"].to_numpy())
            out.update(fractions(mask, values, xy[:, 0], xy[:, 1], beta, L_db))
            pd.DataFrame(out)[DET_COLUMNS].to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False; n_det += len(cand)
        donef.write(f'{int(r["visit"])},{int(r["detector"])}\n'); donef.flush()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(2, n_workers), mp_context=ctx,
                             initializer=_worker_init, initargs=(cnn_model, config)) as pool:
        for i, data, err in _prefetch(paths, prefetch):
            if data is None:                       # unreadable FITS: logged, NOT marked done (retried on resume)
                _log_unreadable(unreadable, rows[i]["visit"], rows[i]["detector"], err)
                n_unread += 1
                continue
            img, wcs, mjd, mask, values = data
            n_nomask += mask is None
            rl = np.zeros(img.shape, dtype=np.uint16)
            prob, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
            pending.append((pool.submit(_worker, (prob, img, agg, rl)), rows[i], wcs, mjd, mask, values))
            if len(pending) >= 2 * max(2, n_workers):   # backpressure: bound RAM + queue depth
                drain()
            now = time.time()
            if now - last >= 20.0:                       # progress heartbeat (panels segmentation model-processed)
                rate = (i + 1) / max(now - t0, 1e-6); eta = (total - i - 1) / rate if rate > 0 else 0.0
                print(f"[gpu{gpu_id}] {i+1}/{total} panels | {n_det} det | {rate:.1f} pan/s | ETA {eta/60:.1f}m", flush=True)
                last = now
        while pending:
            drain()
    donef.close()
    print(f"[gpu{gpu_id}] {len(rows)} new panels (+{n_done} done) -> {n_det} new detections; "
          f"{n_unread} unreadable, {n_nomask} without mask", flush=True)


def run_children(target, argsets):
    """Run target(*args) in one spawned process per argument tuple; return their exit codes."""
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=target, args=args) for args in argsets]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return [p.exitcode for p in procs]


def n_devices(requested, cuda_ok, count, allow_cpu):
    """Shard processes to run: --n-gpus (0 = all visible), never more than the visible GPUs.
    Without CUDA the run is refused unless allow_cpu (ADCNN_ALLOW_CPU=1), then `requested` or 1."""
    if not cuda_ok:
        if not allow_cpu:
            raise SystemExit("[discover] ERROR: CUDA unavailable (set ADCNN_ALLOW_CPU=1 for a CPU run)")
        return requested or 1
    if requested > count:
        raise SystemExit(f"[discover] ERROR: --n-gpus {requested} exceeds the {count} visible GPUs")
    return requested or count


def resolve_threshold(cnn_thr, cnn_path, floor):
    """(threshold, source): --cnn-thr if given, else the sidecar's calibrated 'threshold', else the
    pipeline floor."""
    if cnn_thr is not None:
        return float(cnn_thr), "--cnn-thr"
    side = json.loads(Path(cnn_path).with_suffix(".json").read_text())
    if "threshold" in side:
        return float(side["threshold"]), "sidecar"
    return float(floor), "pipeline floor"


def _done_panels(tmp, tag):
    """(visit, detector) pairs listed in any shard .done log of output `tag` in directory `tmp`."""
    done = set()
    for f in sorted(tmp.glob(f"_shard_{tag}_*.csv.done")):
        for line in open(f):
            p = line.strip().split(",")
            if len(p) == 2:
                done.add((int(p[0]), int(p[1])))
    return done


def merge(man, out, allow_partial=False):
    """Merge the shards of `out` into `out`; returns the exit code.
    Coverage = manifest panels listed in any shard .done log. Below 100% nothing is written and the
    shards/.done logs are kept (return 3), unless allow_partial: then the done panels are merged and
    the shards are kept. At 100% the shards and .done logs are deleted after the write."""
    tmp, tag = Path(out).parent, Path(out).stem
    expected = set(zip(man["visit"].astype(np.int64), man["detector"].astype(np.int64)))
    done = _done_panels(tmp, tag) & expected
    n_miss = len(expected) - len(done)
    print(f"[discover] panel coverage {len(done)}/{len(expected)}", flush=True)
    if n_miss and not allow_partial:
        print(f"[discover] {n_miss} panels missing: shards kept, {out} not written", flush=True)
        return 3
    if n_miss:
        print(f"[discover] --allow-partial: merging without the {n_miss} missing panels", flush=True)
    # Read ALL shard files for this output (glob): a run that resumed with a DIFFERENT --n-gpus would
    # otherwise orphan the old shards. The dedup below makes the merge idempotent.
    all_shards = sorted(tmp.glob(f"_shard_{tag}_*.csv"))
    frames = [pd.read_csv(c, low_memory=False) for c in all_shards if os.path.getsize(c) > 1]
    cat = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DET_COLUMNS)
    # defensive: drop any torn/misaligned rows (visit must be a 13-digit visit id); coerce to int
    vnum = pd.to_numeric(cat.visit, errors="coerce")
    bad = ~(vnum.notna() & (vnum >= 1e12) & (vnum < 1e13))
    if bad.any():
        print(f"[discover] dropping {int(bad.sum())} malformed rows (bad visit)", flush=True)
        cat = cat[~bad].copy()
    cat["visit"] = pd.to_numeric(cat.visit).astype(np.int64)
    cat["detector"] = pd.to_numeric(cat.detector, errors="coerce").astype("Int64")
    # only panels logged done: rows written before a crash but not logged done are redone on resume
    keep = np.fromiter((k in done for k in zip(cat.visit, cat.detector)), dtype=bool, count=len(cat))
    cat = cat[keep]
    # RESUME IDEMPOTENCY: a panel whose CSV rows were written but whose .done line was not (crash between the
    # two, or the panel reassigned to another shard after an n_gpus change) is re-detected and appended again.
    # Detection is deterministic, so the duplicate rows are byte-identical -> dedup on the panel+position key.
    _n = len(cat)
    cat = cat.drop_duplicates(["visit", "detector", "x", "y", "score"]).reset_index(drop=True)
    if len(cat) < _n:
        print(f"[discover] dedup: dropped {_n - len(cat)} duplicate detections (resume/re-detect)", flush=True)
    cat = cat.sort_values(["mjd", "visit", "detector"]).reset_index(drop=True)
    cat.insert(0, "detid", range(len(cat)))   # deterministic: assigned post-dedup, post-sort
    part = f"{out}.tmp"
    cat.to_csv(part, index=False)
    os.replace(part, out)
    if not n_miss:
        # .done logs before shards: a kill in between leaves panels to redo, never a panel logged done
        # whose rows are gone (the next merge would call the night complete without them)
        for c in sorted(tmp.glob(f"_shard_{tag}_*.csv.done")) + all_shards:
            Path(c).unlink(missing_ok=True)
    print(f"[discover] {len(cat)} detections from {len(done)} panels, {cat.visit.nunique()} visits -> {out}",
          flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from ADCNN.config import ACTIVE as _PIPE  # active pipeline (ADCNN_PIPELINE selects; default=v2_D2)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cnn-thr", type=float, default=None,
                    help="CNN retention floor. detect.sbatch always passes the night's FLOOR (status.py "
                         "DEFAULT_FLOOR); without it, the cnn_postproc.json 'threshold' (v2_D2 has none), "
                         "else the pipeline.json cnn_thr_floor")
    ap.add_argument("--prefetch", type=int, default=6, help="FITS reads in flight per GPU (bounds memory)")
    ap.add_argument("--n-gpus", type=int, default=0, help="0 = all visible")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="no detection: merge the panels already done even if coverage is incomplete")
    a = ap.parse_args(argv)
    man = pd.read_csv(a.manifest)
    if not a.allow_partial:
        # Fail fast: validate the pipeline config, model files + the op-point sidecar exist BEFORE allocating
        # GPUs (else the worker crashes seconds in with an opaque torch error).
        if _PIPE is None:
            raise SystemExit("[discover] ERROR: no ADCNN pipeline config (set $ADCNN_PIPELINE)")
        seg_model, cnn = str(_PIPE.seg_model), str(_PIPE.cnn_model)
        for _m, _label in ((seg_model, "segmentation model"), (cnn, "focal-cutout CNN")):
            if not os.path.exists(_m):
                raise SystemExit(f"[discover] ERROR: {_label} not found: {_m} (check $ADCNN_PIPELINE)")
        if a.cnn_thr is None and not os.path.exists(str(Path(cnn).with_suffix(".json"))):
            raise SystemExit(f"[discover] ERROR: op-point sidecar {Path(cnn).with_suffix('.json')} missing; pass --cnn-thr explicitly")
        thr, src = resolve_threshold(a.cnn_thr, cnn, _PIPE.cnn_thr_floor)
        print(f"[discover] stage-2 filter = focal-cutout CNN ({Path(cnn).name}) @ thr {thr} (from {src})", flush=True)
        import torch
        n_gpus = n_devices(a.n_gpus, torch.cuda.is_available(), torch.cuda.device_count(),
                           os.environ.get("ADCNN_ALLOW_CPU") == "1")
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or 8
        n_workers = max(2, cores // n_gpus - 1)   # feature procs per GPU shard (n_gpus×n_workers ≈ cores)
        print(f"[discover] {n_gpus} {'GPUs' if torch.cuda.is_available() else 'CPU shards'} × {n_workers} "
              f"feature workers ({cores} cores)", flush=True)
        shards = [man.iloc[g::n_gpus].to_dict("records") for g in range(n_gpus)]
        tmp = Path(a.out).parent
        tmp.mkdir(parents=True, exist_ok=True)
        # shard paths MUST be unique per output, else concurrent fields sharing a RUN dir append to the same
        # _shard{g}.csv and interleave/tear each other's rows (corruption). Tag with the output stem.
        _tag = Path(a.out).stem
        shard_csvs = [str(tmp / f"_shard_{_tag}_{g}.csv") for g in range(n_gpus)]
        unreadable = a.out + ".unreadable"
        if n_gpus == 1:
            run_shard(0, shards[0], seg_model, cnn, thr, a.prefetch, shard_csvs[0], unreadable, n_workers)
        else:
            gs = [g for g in range(n_gpus) if shards[g]]
            codes = run_children(run_shard, [(g, shards[g], seg_model, cnn, thr, a.prefetch, shard_csvs[g],
                                              unreadable, n_workers) for g in gs])
            failed = [(g, c) for g, c in zip(gs, codes) if c != 0]
            if failed:
                raise SystemExit(f"[discover] ERROR: shard process(es) failed (gpu, exit code): {failed}; "
                                 f"shards kept, {a.out} not written")
    raise SystemExit(merge(man, a.out, a.allow_partial))


if __name__ == "__main__":
    main()
