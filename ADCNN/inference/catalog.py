"""ENTRY POINT — end-to-end ADCNN inference: diffim panels -> detection catalog.

Runs the full two-stage detector over every panel of an h5 and emits ONE ROW PER KEPT
DETECTION (RF score >= rf_thr) as a CSV catalog:

    v7 segmentation  ->  candidate components + 72 features  ->  RandomForest score

Each row carries the *measured* trail geometry (centroid x/y, orientation ``beta``,
``length``), brightness (``flux``), the raw NN peak, and the RF score — everything an
evaluator needs to overlap-match this catalog against a truth catalog, and everything
HelioLinC needs once sky coordinates are attached.

Sky coordinates (RA/Dec/MJD) are deliberately NOT added here: they require the per-panel
Butler WCS (``lsst_distrib`` env, no torch). This engine runs in the torch env and emits
the pixel-space catalog plus routing keys (``image_id`` + ``visit``/``detector``/``band``
when a ``panels.csv`` is supplied). ``experiments/heliolinc/adcnn_wcs.py`` is the Butler
step that turns those into the HelioLinC catalog (``detid,mjd,ra,dec,mag,band,obscode``).

PERFORMANCE: the GPU runs v7 inference in the main process while a pool of worker
processes computes the 72 candidate features + RF score in parallel across panels (the
feature stage is CPU-bound and single-threaded per panel, so this is the dominant cost).
Panels are independent, so the output is identical to a serial run (rows sorted by
``image_id``). Set ``n_workers`` (default = allocated cores).

    python -m ADCNN.inference.catalog \
        --h5 DATA_DIFFIM/test_5sigma/test.h5 \
        --panels DATA_DIFFIM/test_5sigma/panels.csv \
        --out detections_5sigma.csv
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ADCNN.inference.rf_postproc import DEFAULT_THR

REPO = Path(__file__).resolve().parents[2]

# Public detection-catalog schema: internal candidate column -> emitted column.
# Keep stable — the eval matcher (ADCNN.evaluation.catalog_match) and the HelioLinC
# bridge read these names.
_COLMAP = {
    "image_id": "image_id",
    "x_centroid": "x",          # measured centroid (px)
    "y_centroid": "y",
    "or_beta": "beta",          # measured orientation (deg, image convention 0=+x)
    "mf_length": "length",      # measured trail length (px)
    "mf_flux": "flux",          # integrated matched-filter flux (brightness proxy)
    "mf_snr": "mf_snr",
    "area": "area",
    "elongation": "elongation",
    "max_p": "nn_pmax",         # peak NN segmentation probability
    "score_rf": "score_rf",     # stage-2 RF score (operating cut applied before emit)
}
CATALOG_COLUMNS = list(_COLMAP.values())


def _panel_to_catalog(pid, prob, img, sin, cos, agg, rl, rf, rf_thr, gate_pmax=0.0):
    """Stage-2 for one panel: 72 features -> RF score -> keep score>=rf_thr -> schema.
    Pure-CPU (no torch/GPU); safe to run in a worker process. Returns a DataFrame slice
    in the public schema, or None if no detection survives."""
    from ADCNN.inference.rf_postproc import RF_FEATURES_V2, compute_v2_features, apply_rf_v2
    cand, _ = compute_v2_features(prob[None], img[None], sin[None], cos[None], agg[None],
                                  real_labels=rl[None], gate_pmax=gate_pmax, verbose=False)
    if not len(cand):
        return None
    cand[list(RF_FEATURES_V2)] = cand[list(RF_FEATURES_V2)].replace([np.inf, -np.inf], np.nan)
    cand = apply_rf_v2(cand, rf)
    cand = cand[cand["score_rf"] >= rf_thr].copy()
    if not len(cand):
        return None
    cand["image_id"] = int(pid)
    return cand[[c for c in _COLMAP if c in cand.columns]].rename(columns=_COLMAP)


# --- worker process state (one RF per worker, loaded once) ---
_RF = None
_THR = DEFAULT_THR
_GATE = 0.0


def _worker_init(rf_pkl, rf_thr, gate_pmax):
    """Isolate each feature worker: no GPU, single-threaded BLAS (we parallelise across
    panels, so per-worker thread pools would only oversubscribe), one shared RF."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    global _RF, _THR, _GATE
    from ADCNN.inference.rf_postproc import load_rf
    _RF = load_rf(str(rf_pkl))
    try:
        _RF.n_jobs = 1
    except Exception:
        pass
    _THR = float(rf_thr)
    _GATE = float(gate_pmax)


def _worker(args):
    pid, prob, img, sin, cos, agg, rl = args
    return _panel_to_catalog(pid, prob, img, sin, cos, agg, rl, _RF, _THR, _GATE)


def build_detection_catalog(h5_path, v7_ckpt, rf_pkl, *, panels_csv=None,
                            rf_thr: float = DEFAULT_THR, device: str = "cuda",
                            panel_ids=None, n_workers=None, gate_pmax: float = 0.0,
                            stride: int = 64) -> pd.DataFrame:
    """Run the two-stage detector over `h5_path`; return one row per kept detection.

    GPU v7 inference (main process) is pipelined with a pool of `n_workers` CPU processes
    that compute features + RF in parallel across panels. `rf_thr` is the pre-chosen RF
    operating point. `panels_csv` (optional) attaches visit/detector/band by `image_id`
    for the downstream HelioLinC WCS step.
    """
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full

    if n_workers is None:
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
        except AttributeError:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128px tiles -> autotune once
    model = torch.jit.load(str(v7_ckpt), map_location=dev).eval()
    parts: list[pd.DataFrame] = []

    def _read(f, pid):
        return (f["images"][pid][:].astype(np.float32), f["real_labels"][pid][:].astype(np.uint16))

    if n_workers <= 1:
        from ADCNN.inference.rf_postproc import load_rf
        rf = load_rf(str(rf_pkl))
        with h5py.File(h5_path, "r") as f:
            ids = range(int(f["images"].shape[0])) if panel_ids is None else panel_ids
            for pid in ids:
                img, rl = _read(f, pid)
                prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev, stride=stride)
                r = _panel_to_catalog(int(pid), prob, img, sin, cos, agg, rl, rf, rf_thr, gate_pmax)
                if r is not None:
                    parts.append(r)
    else:
        ctx = mp.get_context("spawn")
        pending: deque = deque()

        def drain():
            r = pending.popleft().result()
            if r is not None and len(r):
                parts.append(r)

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=_worker_init, initargs=(str(rf_pkl), rf_thr, gate_pmax)) as pool, \
             h5py.File(h5_path, "r") as f:
            ids = range(int(f["images"].shape[0])) if panel_ids is None else panel_ids
            for pid in ids:
                img, rl = _read(f, pid)
                prob, sin, cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev, stride=stride)
                pending.append(pool.submit(_worker, (int(pid), prob, img, sin, cos, agg, rl)))
                if len(pending) >= 2 * n_workers:   # backpressure: bound RAM + queue depth
                    drain()
            while pending:
                drain()

    if parts:
        cat = pd.concat(parts, ignore_index=True).sort_values("image_id").reset_index(drop=True)
    else:
        cat = pd.DataFrame(columns=CATALOG_COLUMNS)

    if panels_csv:
        pan = pd.read_csv(panels_csv)
        keep = [c for c in ("image_id", "visit", "detector", "band") if c in pan.columns]
        if len(keep) > 1:
            cat = cat.merge(pan[keep], on="image_id", how="left")
    return cat


def _gpu_shard_worker(gpu_id, h5_path, v7_ckpt, rf_pkl, shard, rf_thr, n_workers, batch, gate_pmax, stride, q):
    """Run the engine on one panel shard pinned to GPU `gpu_id`. Spawned process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["ADCNN_TILE_BATCH"] = str(batch)
    os.environ["ADCNN_PREP_WORKERS"] = str(max(2, n_workers))  # CPU prep threads (overlap GPU)
    cat = build_detection_catalog(h5_path, v7_ckpt, rf_pkl, rf_thr=rf_thr,
                                  device="cuda", panel_ids=shard, n_workers=n_workers,
                                  gate_pmax=gate_pmax, stride=stride)
    q.put(cat)


def build_detection_catalog_multigpu(h5_path, v7_ckpt, rf_pkl, *, panels_csv=None,
                                     rf_thr: float = DEFAULT_THR, n_gpus=None,
                                     tile_batch: int = 64, gate_pmax: float = 0.0,
                                     stride: int = 64) -> pd.DataFrame:
    """Data-parallel catalog build: panels are round-robin sharded across `n_gpus`, each
    GPU runs the engine (with its own CPU feature pool) in a separate process. Identical
    output to the single-GPU path (rows sorted by image_id). Falls back to single-GPU when
    n_gpus<=1. `tile_batch` sets the per-forward tile batch (env ADCNN_TILE_BATCH)."""
    import torch
    if n_gpus is None:
        n_gpus = max(1, torch.cuda.device_count())
    with h5py.File(h5_path, "r") as f:
        n_panels = int(f["images"].shape[0])
    if n_gpus <= 1:
        os.environ["ADCNN_TILE_BATCH"] = str(tile_batch)
        return build_detection_catalog(h5_path, v7_ckpt, rf_pkl, panels_csv=panels_csv,
                                       rf_thr=rf_thr, device="cuda", gate_pmax=gate_pmax, stride=stride)

    shards = [list(range(g, n_panels, n_gpus)) for g in range(n_gpus)]  # round-robin balance
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or (2 * n_gpus)
    per = max(1, cores // n_gpus - 1)  # CPU feature workers per GPU process

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gpu_shard_worker,
                         args=(g, str(h5_path), str(v7_ckpt), str(rf_pkl), shards[g],
                               rf_thr, per, tile_batch, gate_pmax, stride, q))
             for g in range(n_gpus) if shards[g]]
    for p in procs:
        p.start()
    parts = [q.get() for _ in procs]   # drain queue before join (avoids deadlock on large items)
    for p in procs:
        p.join()

    nonempty = [c for c in parts if len(c)]
    cat = (pd.concat(nonempty, ignore_index=True).sort_values("image_id").reset_index(drop=True)
           if nonempty else pd.DataFrame(columns=CATALOG_COLUMNS))
    if panels_csv:
        pan = pd.read_csv(panels_csv)
        keep = [c for c in ("image_id", "visit", "detector", "band") if c in pan.columns]
        if len(keep) > 1:
            cat = cat.merge(pan[keep], on="image_id", how="left")
    return cat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="diffim panel h5 (images + real_labels)")
    ap.add_argument("--panels", help="optional panels.csv -> attach visit/detector/band")
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--rf", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--rf-thr", type=float, default=DEFAULT_THR,
                    help="pre-chosen RF operating point (default = shipped DEFAULT_THR)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all panels")
    ap.add_argument("--n-workers", type=int, default=0, help="0 = all allocated cores")
    ap.add_argument("--n-gpus", type=int, default=1, help=">1 = data-parallel across GPUs (ignores --limit)")
    ap.add_argument("--tile-batch", type=int, default=24, help="tiles per GPU forward (64 is a good A100 value)")
    ap.add_argument("--gate-pmax", type=float, default=0.0, help="skip features for candidates with peak NN prob < this (pre-chosen on val)")
    ap.add_argument("--stride", type=int, default=64, help="sliding-window stride (larger = fewer tiles = faster, slightly different maps)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.n_gpus and a.n_gpus > 1:
        cat = build_detection_catalog_multigpu(a.h5, a.v7, a.rf, panels_csv=a.panels,
                                               rf_thr=a.rf_thr, n_gpus=a.n_gpus, tile_batch=a.tile_batch,
                                               gate_pmax=a.gate_pmax, stride=a.stride)
    else:
        os.environ["ADCNN_TILE_BATCH"] = str(a.tile_batch)
        pids = range(a.limit) if a.limit else None
        cat = build_detection_catalog(a.h5, a.v7, a.rf, panels_csv=a.panels, rf_thr=a.rf_thr,
                                      device=a.device, panel_ids=pids, n_workers=(a.n_workers or None),
                                      gate_pmax=a.gate_pmax, stride=a.stride)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(a.out, index=False)
    npan = cat["image_id"].nunique() if len(cat) else 0
    print(f"[catalog] {len(cat)} detections (score>={a.rf_thr}) over {npan} panels -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
