"""ENTRY POINT — end-to-end ADCNN inference: diffim panels -> detection catalog.

Runs the full two-stage detector over every panel of an h5 and emits ONE ROW PER KEPT
DETECTION (CNN score >= ``config.cnn_thr``) as a CSV catalog:

    segmentation model segmentation  ->  candidate components + trail measurement  ->  focal cutout-CNN score

Each row carries the *measured* trail geometry (centroid x/y, orientation ``beta``, ``length``),
brightness (``flux``), the raw NN peak and the stage-2 CNN ``score`` — everything an evaluator
needs to overlap-match against a truth catalog, and everything HelioLinC needs once sky
coordinates are attached.

Sky coordinates (RA/Dec/MJD) are deliberately NOT added here: they require the per-panel Butler
WCS (``lsst_distrib`` env, no torch). This engine runs in the torch env and emits the pixel-space
catalog plus routing keys (``image_id`` + ``visit``/``detector``/``band`` when a ``panels.csv`` is
supplied).

Performance: the GPU runs segmentation model inference (with parallel per-tile CPU prep) in the main process while
a pool of worker processes computes candidate features in parallel across panels (CPU-bound per
panel, so it overlaps behind the GPU). The small width-40 cutout CNN scores on the GPU alongside
segmentation model. ``build_detection_catalog_multigpu`` shards panels across all GPUs. The output is independent
of worker/GPU count (rows sorted by ``image_id``).

    python -m ADCNN.inference.catalog --h5 DATA_DIFFIM/test/test.h5 \
        --panels DATA_DIFFIM/test/panels.csv --n-gpus 4 --out detections.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import h5py
import numpy as np
import pandas as pd

from ADCNN.inference.cnn_postproc import CNN_DEFAULT_THR

__all__ = ["InferenceConfig", "build_detection_catalog",
           "build_detection_catalog_multigpu", "panel_to_catalog_rows",
           "CATALOG_COLUMNS", "MF_LEN_OFFSET", "MF_LEN_SLOPE"]

REPO = Path(__file__).resolve().parents[2]

# Public detection-catalog schema: internal candidate column -> emitted column. Keep stable —
# the eval matcher (ADCNN.evaluation.catalog_match) and the HelioLinC bridge read these names.
_COLMAP = {
    "image_id": "image_id",
    "x_centroid": "x",          # measured centroid (px)
    "y_centroid": "y",
    "mf_beta": "beta",          # trail PA from footprint PCA (deg, 0=+x); recovers truth ~8-10deg MAD
    "mf_length": "length",      # measured trail length (px), de-biased to physical length (see MF_LEN_*)
    "mf_length_raw": "length_raw",  # RAW pre-debias matched-filter length (px); for model-specific MF_LEN recalibration
    "mf_flux": "flux",          # integrated matched-filter flux (brightness proxy)
    "mf_snr": "mf_snr",
    "area": "area",
    "elongation": "elongation",
    "max_p": "nn_pmax",         # peak NN segmentation probability
    "score": "score",           # stage-2 cutout-CNN score (operating cut applied before emit)
    # DETECTION-TIME morphology (bright-star diffim dipole/RING measure, DMTN-007): computed on the
    # diffim stamp while it is in memory (no extra IO), so linking can refuse ring seeds pre-link
    # instead of a post-link QA stamp pass. See panel_to_catalog_rows + ADCNN.qa.alert_morphology.
    "m_elong": "m_elong", "m_dipole": "m_dipole", "m_neg_ratio": "m_neg_ratio",
    "m_neg_blob": "m_neg_blob", "is_dipole": "is_dipole",
}
CATALOG_COLUMNS = list(_COLMAP.values())
_ROUTING_KEYS = ("image_id", "visit", "detector", "band")  # joined from panels.csv for HelioLinC


@dataclass(frozen=True)
class InferenceConfig:
    """Knobs for the two-stage detector.

    Detection-affecting (pre-chosen, never tuned on the eval set):
      cnn_thr  : CNN score operating point (keep detections with score >= this).
      gate_pmax: skip the expensive candidate features for candidates whose peak NN prob < this
                 (the CNN scores them ~0 anyway; 0 = no gate).
      stride   : sliding-window stride for segmentation model inference.
    Speed-only (do not change detections):
      tile_batch: tiles per GPU forward.
    """
    cnn_thr: float = CNN_DEFAULT_THR
    gate_pmax: float = 0.0
    stride: int = 64
    tile_batch: int = 64


DEFAULT_CONFIG = InferenceConfig()
PROGRESS_S = 20.0  # heartbeat interval (s) for the per-shard progress print

# segmentation model's segmentation/matched-filter over-extends trail ends ("ends bloom"): the raw mf_length is
# biased, mf_length ≈ MF_LEN_SLOPE*L_true + MF_LEN_OFFSET. The emitted `length` INVERTS this to the
# physical trail length (median residual ~0px vs truth), so eval parameter-recovery is unbiased and
# HelioLinC gets the true length. Single source of truth: downstream consumers read the corrected
# `length` directly (no re-correction).
#
# The de-bias is MODEL-SPECIFIC (a domain-adapted stage-1 has a different ends-bloom), so it lives in
# the active pipeline config (ADCNN/config.py -> models/<pipeline>/pipeline.json) and TRAVELS WITH THE
# MODEL. Mixing one model with another's de-bias silently corrupts `len_db` (the linker length gate then
# deletes real detections). Select a pipeline with ADCNN_PIPELINE; the current default carries 7.67/0.9425.
# Escape hatch for FITTING a new de-bias: ADCNN_MF_LEN_OFFSET=0 ADCNN_MF_LEN_SLOPE=1 -> raw length
# (env override is applied inside load_pipeline()).
from ADCNN.config import ACTIVE as _PIPE  # resolved once at import (honors ADCNN_PIPELINE / env overrides)

if _PIPE is not None:
    MF_LEN_OFFSET = _PIPE.mf_len_offset
    MF_LEN_SLOPE = _PIPE.mf_len_slope
else:  # no pipeline config on disk (bootstrap) -> last-resort env, else raw (no de-bias)
    MF_LEN_OFFSET = float(os.environ.get("ADCNN_MF_LEN_OFFSET", "0"))
    MF_LEN_SLOPE = float(os.environ.get("ADCNN_MF_LEN_SLOPE", "1"))
import sys as _sys
# TEMPLATE-BANK trail length. ON by default: measured end-to-end at the binding 1k budget it lifts
# delivered completeness 9.39% -> 12.70% (McNemar p<1e-4) at flat purity, with the mechanism confirmed
# (detection bit-identical, linking efficiency of both-detected movers 23.5% -> 28.7%).
# ADCNN_MF_TEMPLATE_LENGTH=0 restores the footprint estimator + ends-bloom de-bias exactly.
MF_TEMPLATE_LENGTH = os.environ.get("ADCNN_MF_TEMPLATE_LENGTH", "1") not in ("0", "false", "False")

print(f"[catalog] MF_LEN de-bias offset={MF_LEN_OFFSET} slope={MF_LEN_SLOPE} "
      f"(pipeline={_PIPE.name if _PIPE else 'none'})", file=_sys.stderr, flush=True)


def panel_to_catalog_rows(pid: int, prob, img, agg, rl, cnn,
                          config: InferenceConfig) -> Optional[pd.DataFrame]:
    """Stage 2 for one panel: candidate extraction -> cutout-CNN score -> keep score>=cnn_thr
    -> public schema. Returns the per-panel catalog slice, or None if no detection survives.

    This is the single primitive used by both the disk-h5 engine
    (:func:`build_detection_catalog`) and the live-Butler stream
    (:mod:`ADCNN.inference.stream_real_inference`).
    """
    from ADCNN.inference.features import extract_panel_candidates
    from ADCNN.inference.cnn_postproc import apply_cnn
    cand, _ = extract_panel_candidates(prob[None], img[None], real_labels=rl[None],
                                       gate_pmax=config.gate_pmax)
    if not len(cand):
        return None
    dev = next(cnn.parameters()).device                       # CNN runs on the GPU (see _load_filter)
    cand = apply_cnn(cand, cnn, img, prob, agg, device=dev)   # cutout score -> `score`
    cand = cand[cand["score"] >= config.cnn_thr].copy()
    if not len(cand):
        return None
    cand["image_id"] = int(pid)
    if "mf_length" in cand.columns:   # de-bias the ends-bloom -> physical trail length (see MF_LEN_*)
        cand["mf_length_raw"] = cand["mf_length"]   # preserve RAW (pre-debias) for model-specific MF_LEN recalibration
        if MF_TEMPLATE_LENGTH:
            # TEMPLATE-BANK matched filter (ADCNN.inference.mf_trail_length). The incumbent length is
            # the extent of the THRESHOLDED footprint, so it is contrast-dependent: on a faint trail
            # the threshold keeps only the bright middle and the ends are lost (measured median
            # 13.17px against a truth median of ~24px on a faint population). A faint FAST mover then
            # reads as a SLOW one, the linker's dspeed chi2 fires, and a real detection is discarded.
            #
            # The ends-bloom de-bias is DELIBERATELY NOT APPLIED here: MF_LEN_OFFSET/SLOPE were
            # calibrated for the FOOTPRINT estimator's bias, so applying them to template lengths
            # would double-correct. The template estimator carries its own single calibration (MF_K).
            from ADCNN.inference.mf_trail_length import refine_trail_length
            # Reuse the panel sigma computed during candidate extraction. Recomputing it dominated
            # the estimator -- 277 ms of a 318 ms call, five times the template bank that needs it --
            # and a second full-panel median would simply pay that cost again.
            _sig = (float(cand["panel_sigma"].iloc[0]) if "panel_sigma" in cand.columns
                    and np.isfinite(cand["panel_sigma"].iloc[0]) else None)
            _L, _B = refine_trail_length(cand["x_centroid"].to_numpy(), cand["y_centroid"].to_numpy(),
                                         img, cand["mf_length"].to_numpy(),
                                         cand["mf_beta"].to_numpy() if "mf_beta" in cand.columns
                                         else np.zeros(len(cand)),
                                         sigma=_sig)
            # REGRESSION GUARD: refine_trail_length returns the INCUMBENT footprint length for any
            # detection it cannot handle (within MF_STAMP/2 of a panel edge). In template mode the
            # de-bias else-branch below never runs, so those fallback rows kept a RAW footprint
            # length -- measured median 24.09 px against 17.42 px under the previous pipeline (ratio
            # 1.370, 84.8% inflated by >20%) on 0.87% of detections (~33k/night). detect_night builds
            # the on-sky trail endpoints from this, so their trail was ~37% too long and fed the
            # linker's dspeed chi2 as a real disagreement. Apply the ends-bloom de-bias to exactly
            # the rows that kept the footprint estimate.
            _fellback = np.isclose(_L, cand["mf_length"].to_numpy(), rtol=0, atol=0)
            _L = np.asarray(_L, float).copy()
            if _fellback.any():
                _L[_fellback] = np.clip(
                    (_L[_fellback] - MF_LEN_OFFSET) / MF_LEN_SLOPE, 0.0, None)
                print(f"[catalog] {int(_fellback.sum())} edge detections kept the footprint length "
                      f"-> ends-bloom de-bias applied to those rows only", flush=True)
            cand["mf_length"] = np.clip(_L, 0.0, None)
            if "mf_beta" in cand.columns:
                cand["mf_beta"] = _B
        else:
            cand["mf_length"] = np.clip((cand["mf_length"] - MF_LEN_OFFSET) / MF_LEN_SLOPE, 0.0, None)
    _attach_dipole_morphology(cand, img)             # DETECTION-TIME ring flag (no extra pixel IO)
    return cand[[c for c in _COLMAP if c in cand.columns]].rename(columns=_COLMAP)


# DMTN-007 dipole/RING morphology at DETECTION time -----------------------------------------------
# The stage-1 image (diffim) is already in memory here, so the per-detection dipole measure costs no
# extra pixel IO. Emitting it as a catalog column lets the LINKER refuse ring seeds pre-link (rings
# never form tracklets) instead of the old post-link QA stamp pass (ADCNN.qa.alert_morphology on the
# alert cutouts). ONE flag definition (alert_morphology.ripple_flag) is shared by both paths.
_MORPH_K = 96          # stamp size for the feature (matches the QA mosaic stamp; _features uses win=28)

def _attach_dipole_morphology(cand: pd.DataFrame, img) -> None:
    from ADCNN.qa.alert_morphology import _features, ripple_flag
    from ADCNN.inference.cnn_postproc import _cutout
    xs = cand["x_centroid"].to_numpy(); ys = cand["y_centroid"].to_numpy()
    E = np.empty(len(xs)); D = np.empty(len(xs)); SN = np.empty(len(xs))
    NR = np.empty(len(xs)); NB = np.empty(len(xs))
    for i in range(len(xs)):
        e, d, _nd, pk, nr, nb = _features(_cutout(img, xs[i], ys[i], _MORPH_K))
        E[i], D[i], SN[i], NR[i], NB[i] = e, d, pk, nr, nb
    cand["m_elong"] = E; cand["m_dipole"] = D; cand["m_neg_ratio"] = NR; cand["m_neg_blob"] = NB
    cand["is_dipole"] = ripple_flag(E, D, SN, NR, NB)


def _attach_routing_keys(cat: pd.DataFrame, panels_csv) -> pd.DataFrame:
    """Left-join visit/detector/band from `panels_csv` (for the downstream HelioLinC WCS step)."""
    if not panels_csv:
        return cat
    pan = pd.read_csv(panels_csv).drop_duplicates("image_id")  # dup image_id would multiply rows
    keep = [c for c in _ROUTING_KEYS if c in pan.columns]
    return cat.merge(pan[keep], on="image_id", how="left") if len(keep) > 1 else cat


def _finalize(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-panel slices into a deterministic catalog (sorted by image_id)."""
    if not parts:
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values("image_id").reset_index(drop=True)


# --- feature-worker process state (one CNN + config per worker, set once at spawn) ---
_CNN = None
_CONFIG = DEFAULT_CONFIG


def _load_filter(cnn_pt: str, config: InferenceConfig):
    """Load the stage-2 focal cutout CNN (on the GPU when available)."""
    import torch
    from ADCNN.inference.cnn_postproc import load_cnn
    return load_cnn(str(cnn_pt), device=("cuda" if torch.cuda.is_available() else "cpu"))


def _worker_init(cnn_pt: str, config: InferenceConfig) -> None:
    """Isolate each feature worker: pin BLAS to one thread (we parallelise across panels, so
    per-worker thread pools would only oversubscribe) and load one CNN. The cutout CNN scores
    on the GPU, so the worker keeps the shard's CUDA_VISIBLE_DEVICES (shared with segmentation model)."""
    # Setting these here is TOO LATE: this module imports numpy at module scope, which under spawn
    # runs BEFORE the initializer, so the BLAS pool is already sized from the visible core count.
    # MEASURED: threadpool_info() inside a worker reported 64 OpenBLAS threads, with n_workers =
    # cores-1 = 127 -- i.e. 127x64 threads on 128 cores. Kept for child processes that import later;
    # threadpoolctl below is what actually pins the already-initialised pool.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    global _CNN, _CONFIG
    _CNN = _load_filter(cnn_pt, config)
    _CONFIG = config


def _worker(args):
    return panel_to_catalog_rows(*args, _CNN, _CONFIG)


def _iter_panel_outputs(model, h5_path, panel_ids, device, config: InferenceConfig,
                        prep_workers) -> Iterator[tuple]:
    """Yield ``(pid, prob, img, agg, rl)`` for each panel: read the diffim + DIA mask and run
    the segmentation sliding-window inference (GPU, parallel CPU prep). The seg model's sin/cos
    orientation heads aren't used downstream by the lean candidate extractor, so we discard
    them here to keep memory and worker IPC small."""
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    with h5py.File(h5_path, "r") as f:
        ids = range(int(f["images"].shape[0])) if panel_ids is None else panel_ids
        for pid in ids:
            img = f["images"][pid][:].astype(np.float32)
            rl = f["real_labels"][pid][:].astype(np.uint16)
            prob, _sin, _cos, agg = predict_panel_overlap_3ch_full(
                model, img, rl, device=device, stride=config.stride,
                tile_batch=config.tile_batch, prep_workers=prep_workers)
            yield int(pid), prob, img, agg, rl


def build_detection_catalog(h5_path, seg_ckpt, cnn_pt, *, config: InferenceConfig = DEFAULT_CONFIG,
                            panels_csv=None, panel_ids: Optional[Iterable[int]] = None,
                            device: str = "cuda", n_workers: Optional[int] = None,
                            prep_workers: Optional[int] = None) -> pd.DataFrame:
    """Run the two-stage detector over `h5_path`; return one row per kept detection.

    segmentation model inference (GPU, main process) is pipelined with a pool of `n_workers` CPU processes
    computing candidate features + the cutout CNN in parallel across panels (`n_workers<=1` runs
    them inline). The result is independent of `n_workers`. `panels_csv` attaches
    visit/detector/band for the HelioLinC WCS step. `prep_workers` sets the per-tile CPU prep
    threads (default = predict's).
    """
    import torch
    if n_workers is None:
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
        except AttributeError:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

    # Pin BLAS in the PARENT before spawning: children inherit the environment at spawn time, and by
    # the time _worker_init runs the child has already imported numpy (measured: 64 OpenBLAS threads
    # per worker with n_workers=127, i.e. 8,128 threads on 128 cores).
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed 128px tiles -> autotune once
    if panel_ids is not None:
        panel_ids = list(panel_ids)
        total = len(panel_ids)
    else:
        with h5py.File(h5_path, "r") as f:
            total = int(f["images"].shape[0])
    model = torch.jit.load(str(seg_ckpt), map_location=dev).eval()
    panels = _iter_panel_outputs(model, h5_path, panel_ids, dev, config, prep_workers)
    parts: list[pd.DataFrame] = []

    # progress heartbeat: panels segmentation model-processed / total + detections so far + rate + ETA, every PROGRESS_S
    tag = os.environ.get("CUDA_VISIBLE_DEVICES") or str(device)
    t0 = time.time(); last = [t0]; n = [0]

    def tick():
        now = time.time()
        if now - last[0] >= PROGRESS_S or n[0] == total:
            nd = sum(len(p) for p in parts); rate = n[0] / max(now - t0, 1e-6)
            eta = (total - n[0]) / rate if rate > 0 else 0.0
            print(f"[catalog gpu{tag}] {n[0]}/{total} panels | {nd} det | {rate:.1f} pan/s | ETA {eta/60:.1f}m",
                  flush=True)
            last[0] = now

    if n_workers <= 1:
        cnn = _load_filter(cnn_pt, config)
        for pid, prob, img, agg, rl in panels:
            r = panel_to_catalog_rows(pid, prob, img, agg, rl, cnn, config)
            if r is not None:
                parts.append(r)
            n[0] += 1; tick()
    else:
        ctx = mp.get_context("spawn")
        pending: deque = deque()

        def drain():
            r = pending.popleft().result()
            if r is not None and len(r):
                parts.append(r)

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                 initializer=_worker_init, initargs=(str(cnn_pt), config)) as pool:
            for out in panels:
                pending.append(pool.submit(_worker, out))
                n[0] += 1; tick()             # n = panels segmentation model-processed (GPU is the bottleneck)
                if len(pending) >= 2 * n_workers:   # backpressure: bound RAM + queue depth
                    drain()
            while pending:
                drain()
        n[0] = total; tick()                  # final line with the full detection count

    return _attach_routing_keys(_finalize(parts), panels_csv)


def _gpu_shard_worker(gpu_id, h5_path, seg_ckpt, cnn_pt, shard, config, n_workers, q):
    """Run the engine on one panel shard pinned to GPU `gpu_id` (spawned process)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cat = build_detection_catalog(h5_path, seg_ckpt, cnn_pt, config=config, panel_ids=shard,
                                  device="cuda", n_workers=n_workers, prep_workers=max(2, n_workers))
    q.put(cat)


def build_detection_catalog_multigpu(h5_path, seg_ckpt, cnn_pt, *,
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
        return build_detection_catalog(h5_path, seg_ckpt, cnn_pt, config=config,
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
                         args=(g, str(h5_path), str(seg_ckpt), str(cnn_pt), shards[g], config, per, q))
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
    _seg_def = str(_PIPE.seg_model) if _PIPE else str(REPO / "models/current/segmentation_scripted.pt")
    _cnn_def = str(_PIPE.cnn_model) if _PIPE else str(REPO / "models/current/cnn_postproc.pt")
    ap.add_argument("--seg-model", default=_seg_def, help="stage-1 segmentation (default: active pipeline)")
    ap.add_argument("--cnn", default=_cnn_def, help="stage-2 cutout CNN (default: active pipeline)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cnn-thr", type=float, default=CNN_DEFAULT_THR, help="CNN operating point (pre-chosen)")
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

    config = InferenceConfig(cnn_thr=a.cnn_thr, gate_pmax=a.gate_pmax,
                             stride=a.stride, tile_batch=a.tile_batch)
    if a.n_gpus and a.n_gpus > 1:
        cat = build_detection_catalog_multigpu(a.h5, a.seg_model, a.cnn, config=config,
                                               panels_csv=a.panels, panel_ids=panel_ids, n_gpus=a.n_gpus)
    else:
        cat = build_detection_catalog(a.h5, a.seg_model, a.cnn, config=config, panels_csv=a.panels,
                                      panel_ids=panel_ids, device=a.device, n_workers=(a.n_workers or None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(a.out, index=False)
    npan = cat["image_id"].nunique() if len(cat) else 0
    print(f"[catalog] {len(cat)} detections (CNN score>={a.cnn_thr}) over {npan} panels -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
