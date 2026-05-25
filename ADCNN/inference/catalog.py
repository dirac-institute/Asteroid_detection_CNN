"""ENTRY POINT — end-to-end ADCNN inference: diffim panels -> detection catalog.

Runs the full two-stage detector over every panel of an h5 and emits ONE ROW PER KEPT
DETECTION (RF score >= ``config.rf_thr``) as a CSV catalog:

    v7 segmentation  ->  candidate components + 72 features  ->  RandomForest score

Each row carries the *measured* trail geometry (centroid x/y, orientation ``beta``,
``length``), brightness (``flux``), the raw NN peak and the RF score — everything an
evaluator needs to overlap-match against a truth catalog, and everything HelioLinC needs
once sky coordinates are attached.

Sky coordinates (RA/Dec/MJD) are deliberately NOT added here: they require the per-panel
Butler WCS (``lsst_distrib`` env, no torch). This engine runs in the torch env and emits the
pixel-space catalog plus routing keys (``image_id`` + ``visit``/``detector``/``band`` when a
``panels.csv`` is supplied); ``experiments/heliolinc/adcnn_wcs.py`` is the Butler step that
turns those into the HelioLinC catalog (``detid,mjd,ra,dec,mag,band,obscode``).

Performance: the GPU runs v7 inference (with parallel per-tile CPU prep) in the main process
while a pool of worker processes computes the 72 features + RF score in parallel across
panels — the feature stage is CPU-bound per panel, so this overlaps it behind the GPU and
keeps the node busy. ``build_detection_catalog_multigpu`` shards panels across all GPUs. The
output is independent of worker/GPU count (rows sorted by ``image_id``).

    python -m ADCNN.inference.catalog --h5 DATA_DIFFIM/test_5sigma/test.h5 \
        --panels DATA_DIFFIM/test_5sigma/panels.csv --n-gpus 4 --out detections.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import h5py
import numpy as np
import pandas as pd

from ADCNN.inference.rf_postproc import DEFAULT_THR

__all__ = ["InferenceConfig", "build_detection_catalog",
           "build_detection_catalog_multigpu", "CATALOG_COLUMNS"]

REPO = Path(__file__).resolve().parents[2]

# Public detection-catalog schema: internal candidate column -> emitted column. Keep stable —
# the eval matcher (ADCNN.evaluation.catalog_match) and the HelioLinC bridge read these names.
_COLMAP = {
    "image_id": "image_id",
    "x_centroid": "x",          # measured centroid (px)
    "y_centroid": "y",
    "mf_beta": "beta",          # trail PA from footprint PCA (deg, 0=+x); recovers truth ~8-10deg MAD
    "or_beta": "beta_nn",       # NN sin2β/cos2β-head orientation (DIAGNOSTIC ONLY: r≈0 vs truth)
    "mf_length": "length",      # measured trail length (px)
    "mf_flux": "flux",          # integrated matched-filter flux (brightness proxy)
    "mf_snr": "mf_snr",
    "area": "area",
    "elongation": "elongation",
    "max_p": "nn_pmax",         # peak NN segmentation probability
    "score_rf": "score_rf",     # stage-2 RF score (operating cut applied before emit)
}
CATALOG_COLUMNS = list(_COLMAP.values())
_ROUTING_KEYS = ("image_id", "visit", "detector", "band")  # joined from panels.csv for HelioLinC


@dataclass(frozen=True)
class InferenceConfig:
    """Knobs for the two-stage detector.

    Detection-affecting (pre-chosen, never tuned on the eval set):
      rf_thr   : RF score operating point (keep detections with score >= this).
      gate_pmax: skip the expensive features for candidates whose peak NN prob < this
                 (the RF scores them ~0 anyway; 0 = no gate).
      stride   : sliding-window stride for v7 inference.
    Speed-only (do not change detections):
      tile_batch: tiles per GPU forward.
    """
    rf_thr: float = DEFAULT_THR
    gate_pmax: float = 0.0
    stride: int = 64
    tile_batch: int = 64


DEFAULT_CONFIG = InferenceConfig()


def _panel_to_catalog(pid: int, prob, img, sin, cos, agg, rl, rf,
                      config: InferenceConfig) -> Optional[pd.DataFrame]:
    """Stage 2 for one panel: 72 features -> RF score -> keep score>=rf_thr -> public schema.
    Pure-CPU (no torch/GPU); safe to run in a worker process. Returns the per-panel catalog
    slice, or None if no detection survives."""
    from ADCNN.inference.rf_postproc import RF_FEATURES_V2, compute_v2_features, apply_rf_v2
    cand, _ = compute_v2_features(prob[None], img[None], sin[None], cos[None], agg[None],
                                  real_labels=rl[None], gate_pmax=config.gate_pmax, verbose=False)
    if not len(cand):
        return None
    cand[list(RF_FEATURES_V2)] = cand[list(RF_FEATURES_V2)].replace([np.inf, -np.inf], np.nan)
    cand = apply_rf_v2(cand, rf)
    cand = cand[cand["score_rf"] >= config.rf_thr].copy()
    if not len(cand):
        return None
    cand["image_id"] = int(pid)
    return cand[[c for c in _COLMAP if c in cand.columns]].rename(columns=_COLMAP)


def _attach_routing_keys(cat: pd.DataFrame, panels_csv) -> pd.DataFrame:
    """Left-join visit/detector/band from `panels_csv` (for the downstream HelioLinC WCS step)."""
    if not panels_csv:
        return cat
    pan = pd.read_csv(panels_csv)
    keep = [c for c in _ROUTING_KEYS if c in pan.columns]
    return cat.merge(pan[keep], on="image_id", how="left") if len(keep) > 1 else cat


def _finalize(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-panel slices into a deterministic catalog (sorted by image_id)."""
    if not parts:
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values("image_id").reset_index(drop=True)


# --- feature-worker process state (one RF + config per worker, set once at spawn) ---
_RF = None
_CONFIG = DEFAULT_CONFIG


def _worker_init(rf_pkl: str, config: InferenceConfig) -> None:
    """Isolate each feature worker: hide the GPU and pin BLAS to one thread (we parallelise
    across panels, so per-worker thread pools would only oversubscribe), and load one RF."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    global _RF, _CONFIG
    from ADCNN.inference.rf_postproc import load_rf
    _RF = load_rf(str(rf_pkl))
    try:
        _RF.n_jobs = 1
    except Exception:
        pass
    _CONFIG = config


def _worker(args):
    return _panel_to_catalog(*args, _RF, _CONFIG)


def _iter_panel_outputs(model, h5_path, panel_ids, device, config: InferenceConfig,
                        prep_workers) -> Iterator[tuple]:
    """Yield ``(pid, prob, img, sin, cos, agg, rl)`` for each panel: read the diffim + DIA
    mask and run v7 sliding-window inference (GPU, with parallel CPU prep)."""
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    with h5py.File(h5_path, "r") as f:
        ids = range(int(f["images"].shape[0])) if panel_ids is None else panel_ids
        for pid in ids:
            img = f["images"][pid][:].astype(np.float32)
            rl = f["real_labels"][pid][:].astype(np.uint16)
            prob, sin, cos, agg = predict_panel_overlap_3ch_full(
                model, img, rl, device=device, stride=config.stride,
                tile_batch=config.tile_batch, prep_workers=prep_workers)
            yield int(pid), prob, img, sin, cos, agg, rl


def build_detection_catalog(h5_path, v7_ckpt, rf_pkl, *, config: InferenceConfig = DEFAULT_CONFIG,
                            panels_csv=None, panel_ids: Optional[Iterable[int]] = None,
                            device: str = "cuda", n_workers: Optional[int] = None,
                            prep_workers: Optional[int] = None) -> pd.DataFrame:
    """Run the two-stage detector over `h5_path`; return one row per kept detection.

    v7 inference (GPU, main process) is pipelined with a pool of `n_workers` CPU processes
    computing features + RF in parallel across panels (`n_workers<=1` runs them inline). The
    result is independent of `n_workers`. `panels_csv` attaches visit/detector/band for the
    HelioLinC WCS step. `prep_workers` sets the per-tile CPU prep threads (default = predict's).
    """
    import torch
    if n_workers is None:
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
        except AttributeError:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128px tiles -> autotune once
    model = torch.jit.load(str(v7_ckpt), map_location=dev).eval()
    panels = _iter_panel_outputs(model, h5_path, panel_ids, dev, config, prep_workers)
    parts: list[pd.DataFrame] = []

    if n_workers <= 1:
        from ADCNN.inference.rf_postproc import load_rf
        rf = load_rf(str(rf_pkl))
        for pid, prob, img, sin, cos, agg, rl in panels:
            r = _panel_to_catalog(pid, prob, img, sin, cos, agg, rl, rf, config)
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
                                 initializer=_worker_init, initargs=(str(rf_pkl), config)) as pool:
            for out in panels:
                pending.append(pool.submit(_worker, out))
                if len(pending) >= 2 * n_workers:   # backpressure: bound RAM + queue depth
                    drain()
            while pending:
                drain()

    return _attach_routing_keys(_finalize(parts), panels_csv)


def _gpu_shard_worker(gpu_id, h5_path, v7_ckpt, rf_pkl, shard, config, n_workers, q):
    """Run the engine on one panel shard pinned to GPU `gpu_id` (spawned process)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cat = build_detection_catalog(h5_path, v7_ckpt, rf_pkl, config=config, panel_ids=shard,
                                  device="cuda", n_workers=n_workers, prep_workers=max(2, n_workers))
    q.put(cat)


def build_detection_catalog_multigpu(h5_path, v7_ckpt, rf_pkl, *,
                                     config: InferenceConfig = DEFAULT_CONFIG,
                                     panels_csv=None, panel_ids: Optional[Iterable[int]] = None,
                                     n_gpus: Optional[int] = None) -> pd.DataFrame:
    """Data-parallel catalog build: panels are round-robin sharded across `n_gpus`, each GPU
    runs the engine (with its own CPU feature pool) in a separate process. Output is identical
    to the single-GPU path (sorted by image_id). Falls back to single-GPU when n_gpus<=1.
    `panel_ids` restricts processing to those panels (e.g. a discovery window)."""
    import torch
    if n_gpus is None:
        n_gpus = max(1, torch.cuda.device_count())
    if n_gpus <= 1:
        return build_detection_catalog(h5_path, v7_ckpt, rf_pkl, config=config,
                                       panels_csv=panels_csv, panel_ids=panel_ids)

    if panel_ids is None:
        with h5py.File(h5_path, "r") as f:
            ids = list(range(int(f["images"].shape[0])))
    else:
        ids = list(panel_ids)
    shards = [ids[g::n_gpus] for g in range(n_gpus)]  # round-robin load balance
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or (2 * n_gpus)
    per = max(1, cores // n_gpus - 1)  # CPU feature workers per GPU process

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_gpu_shard_worker,
                         args=(g, str(h5_path), str(v7_ckpt), str(rf_pkl), shards[g], config, per, q))
             for g in range(n_gpus) if shards[g]]
    for p in procs:
        p.start()
    parts = [q.get() for _ in procs]   # drain queue before join (avoid deadlock on large items)
    for p in procs:
        p.join()

    return _attach_routing_keys(_finalize([c for c in parts if len(c)]), panels_csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="diffim panel h5 (images + real_labels)")
    ap.add_argument("--panels", help="optional panels.csv -> attach visit/detector/band")
    ap.add_argument("--v7", default=str(REPO / "models/v7_diffim_scripted.pt"))
    ap.add_argument("--rf", default=str(REPO / "models/rf_postproc.pkl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rf-thr", type=float, default=DEFAULT_THR, help="RF operating point (pre-chosen)")
    ap.add_argument("--gate-pmax", type=float, default=0.0, help="skip features below this peak NN prob (val-chosen)")
    ap.add_argument("--stride", type=int, default=64, help="sliding-window stride")
    ap.add_argument("--tile-batch", type=int, default=64, help="tiles per GPU forward (speed only)")
    ap.add_argument("--n-gpus", type=int, default=1, help=">1 = data-parallel across GPUs")
    ap.add_argument("--n-workers", type=int, default=0, help="single-GPU: feature procs (0 = all cores)")
    ap.add_argument("--limit", type=int, default=0, help="single-GPU: first N panels only (0 = all)")
    ap.add_argument("--panel-ids", help="restrict to these image_ids: a CSV with an 'image_id' column, or a comma list")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    panel_ids = None
    if a.panel_ids:
        if Path(a.panel_ids).exists():
            panel_ids = sorted(pd.read_csv(a.panel_ids)["image_id"].astype(int).unique())
        else:
            panel_ids = [int(s) for s in a.panel_ids.split(",")]
    elif a.limit:
        panel_ids = list(range(a.limit))

    config = InferenceConfig(rf_thr=a.rf_thr, gate_pmax=a.gate_pmax, stride=a.stride, tile_batch=a.tile_batch)
    if a.n_gpus and a.n_gpus > 1:
        cat = build_detection_catalog_multigpu(a.h5, a.v7, a.rf, config=config,
                                               panels_csv=a.panels, panel_ids=panel_ids, n_gpus=a.n_gpus)
    else:
        cat = build_detection_catalog(a.h5, a.v7, a.rf, config=config, panels_csv=a.panels,
                                      panel_ids=panel_ids, device=a.device, n_workers=(a.n_workers or None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(a.out, index=False)
    npan = cat["image_id"].nunique() if len(cat) else 0
    print(f"[catalog] {len(cat)} detections (score>={a.rf_thr}) over {npan} panels -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
