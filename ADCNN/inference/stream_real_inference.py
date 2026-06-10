"""GPU-side consumer for the real-data streaming pipeline (multi-GPU).

Reads length-prefixed pickled diffim messages from stdin (produced by
``stream_real_butler.py`` running in the LSST stack env), runs the shipped two-stage detector
on every visible GPU in parallel (one worker thread per device), and emits a per-detection
CSV matching the public catalog schema for the real-data evaluation notebook.

The threading layout: one reader thread consumes stdin and pushes panels onto a bounded queue;
N worker threads (one per device) each own a (seg, cnn) pair on their device and pump panels
through ``panel_to_catalog_rows``; the main thread accumulates the per-detection rows.
PyTorch releases the GIL during CUDA kernel launches, so the workers run concurrently across GPUs.
"""
from __future__ import annotations
import argparse
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ADCNN.inference.catalog import (CATALOG_COLUMNS, InferenceConfig,
                                     panel_to_catalog_rows)
from ADCNN.inference.cnn_postproc import CUTOUT_K, read_threshold

# LSSTCam has 189 science detectors per visit; used only to project the per-visit wall time
# in the summary.
N_DETECTORS_PER_VISIT = 189


def _read_frame(stream):
    hdr = stream.read(8)
    if not hdr or len(hdr) < 8:
        return None
    n = int.from_bytes(hdr, "little")
    body = stream.read(n)
    if len(body) < n:
        return None
    return pickle.loads(body)


def _load_models_on_device(seg_path, cnn_path, device):
    import torch
    from ADCNN.inference.cnn_postproc import load_cnn
    seg = torch.jit.load(seg_path, map_location=device).eval()
    cnn = load_cnn(cnn_path, device=device)
    return seg, cnn


def _detect_one(seg, cnn, img, device, config: InferenceConfig):
    """Run stage 1 (seg) + stage 2 (cutout CNN) on one diffim panel via the shared
    ``panel_to_catalog_rows`` primitive. Returns ``(rows_df_or_None, (t_nn, t_ext_cnn))``.
    """
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    # ch3 = 0 in production. ABLATION CLOSED (2026-06-10, 6 test panels, stored clean-diffim mask vs
    # zeros): recall IDENTICAL (85/120 vs 84/120) -> ch3=0 is NOT a recall bug, no retrain needed; all
    # shipped calibrations (injection sweeps via discover_stream) also ran ch3=0, so production and
    # calibration are consistent. Measured upside: stored context suppresses HIGH-SCORE FP ~1.5x at fixed
    # recall (n(S>=0.8) 111 vs 167) -> wiring the real DIA mask plane (diffim HDU2, cf. mask_flags.py) in
    # here is an OPTIONAL FP-density lever for reopening lower score floors, behind the length-split
    # hybrid linker in priority. Do not re-open this as a correctness issue.
    rl = np.zeros_like(img, dtype=np.uint16)  # real-data stream has no DIA mask channel
    t0 = time.perf_counter()
    prob, _sin, _cos, agg = predict_panel_overlap_3ch_full(
        seg, img, rl, device=device, stride=config.stride, tile_batch=config.tile_batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_nn = time.perf_counter() - t0
    prob = prob.astype(np.float32)
    agg = np.asarray(agg, np.float32)
    t0 = time.perf_counter()
    rows = panel_to_catalog_rows(0, prob, img, agg, rl, cnn, config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_ext_cnn = time.perf_counter() - t0
    return rows, (t_nn, t_ext_cnn)


def _worker(device, seg_path, cnn_path, in_q, out_q, config: InferenceConfig, stats):
    """One thread per device. Pull panels until poison-pill, push results."""
    import torch
    dev = torch.device(device)
    torch.backends.cudnn.benchmark = True
    seg, cnn = _load_models_on_device(seg_path, cnn_path, dev)
    while True:
        item = in_q.get()
        if item is None:
            out_q.put(None)  # propagate sentinel
            return
        msg, image_id = item
        img = msg["image"]
        try:
            rows, (t_nn, t_ec) = _detect_one(seg, cnn, img, dev, config)
            n_kept = int(len(rows)) if rows is not None else 0
            row_df = None
            if n_kept:
                rows = rows.copy()
                rows["image_id"] = int(image_id)
                rows["visit"]    = int(msg["visit"])
                rows["detector"] = int(msg["detector"])
                wanted = ["image_id", "visit", "detector"] + [c for c in CATALOG_COLUMNS
                                                              if c not in ("image_id",) and c in rows.columns]
                row_df = rows[wanted]
            out_q.put({"image_id": image_id, "visit": msg["visit"], "detector": msg["detector"],
                       "rows": row_df, "n_kept": n_kept, "t_nn": t_nn, "t_ext_cnn": t_ec,
                       "t_butler": msg.get("t_butler_s", 0.0),
                       "t_subtract": msg.get("t_subtract_s", 0.0)})
            stats[device]["n"] += 1
        except Exception as e:
            out_q.put({"image_id": image_id, "visit": msg["visit"], "detector": msg["detector"],
                       "rows": None, "err": f"{type(e).__name__}: {e}"})


def _reader(in_q, src, n_workers, n_fail_counter):
    """Consume stdin pickled frames, push (msg, image_id) onto in_q for the workers."""
    image_id = 0
    while True:
        msg = _read_frame(src)
        if msg is None or msg.get("_eof"):
            break
        if msg.get("err"):
            n_fail_counter["n"] += 1
            print(f"  UPSTREAM ({msg.get('visit')}, {msg.get('detector')}): {msg['err']}",
                  file=sys.stderr, flush=True)
            continue
        in_q.put((msg, image_id))
        image_id += 1
    for _ in range(n_workers):
        in_q.put(None)  # poison pills, one per worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(REPO / "models/segmentation_model.pt"))
    ap.add_argument("--cnn", default=str(REPO / "models/cnn_postproc.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=0, help="0 = use all visible GPUs")
    ap.add_argument("--queue-size", type=int, default=16, help="bounded queue depth (RAM cap)")
    ap.add_argument("--cnn-thr", type=float, default=None,
                    help="override the sidecar-calibrated CNN operating point")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--tile-batch", type=int, default=64)
    a = ap.parse_args()

    import torch
    n_avail = torch.cuda.device_count()
    n_workers = a.gpus if a.gpus > 0 else (n_avail or 1)
    devices = [f"cuda:{i}" for i in range(n_workers)] if n_avail else ["cpu"]
    print(f"[inference] devices={devices}", file=sys.stderr, flush=True)

    thr = a.cnn_thr if a.cnn_thr is not None else read_threshold(a.cnn)
    config = InferenceConfig(cnn_thr=thr, stride=a.stride, tile_batch=a.tile_batch)
    print(f"[inference] cnn_thr={thr}  k={CUTOUT_K}", file=sys.stderr, flush=True)

    in_q = queue.Queue(maxsize=max(a.queue_size, 2 * len(devices)))
    out_q = queue.Queue()
    stats = {d: {"n": 0} for d in devices}
    n_fail = {"n": 0}

    threads = []
    for d in devices:
        t = threading.Thread(target=_worker, args=(d, a.seg, a.cnn, in_q, out_q, config, stats),
                             daemon=True)
        t.start(); threads.append(t)

    rt = threading.Thread(target=_reader, args=(in_q, sys.stdin.buffer, len(devices), n_fail),
                          daemon=True)
    rt.start()

    rows = []
    n_panels = 0; n_emit = 0
    t_nn, t_ec, t_butler, t_subtract = [], [], [], []
    finished_workers = 0
    t_wall0 = time.perf_counter()

    while finished_workers < len(devices):
        item = out_q.get()
        if item is None:
            finished_workers += 1
            continue
        if item.get("err"):
            print(f"  panel ({item['visit']},{item['detector']}): WORKER {item['err']}",
                  file=sys.stderr, flush=True)
            continue
        n_panels += 1
        n_emit += int(item["n_kept"])
        t_nn.append(item["t_nn"]); t_ec.append(item["t_ext_cnn"])
        t_butler.append(item["t_butler"]); t_subtract.append(item["t_subtract"])
        if item["rows"] is not None:
            rows.append(item["rows"])
        if n_panels % 25 == 0:
            elapsed = time.perf_counter() - t_wall0
            rate = n_panels / max(elapsed, 1e-9)
            mu = lambda v: float(np.mean(v)) if v else 0.0
            print(f"  [{n_panels}]  rate={rate:.3f} pan/s ({rate*60:.1f}/min)  "
                  f"mean butler={mu(t_butler):.2f}s nn={mu(t_nn):.2f}s ext+cnn={mu(t_ec):.2f}s",
                  file=sys.stderr, flush=True)

    wall = time.perf_counter() - t_wall0

    if rows:
        out = pd.concat(rows, ignore_index=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)

    print("\n=== STREAM SUMMARY ===", file=sys.stderr)
    print(f"  devices: {len(devices)}  ({devices})", file=sys.stderr)
    print(f"  panels OK: {n_panels}  FAIL upstream: {n_fail['n']}  rows emitted: {n_emit}",
          file=sys.stderr)
    if n_panels:
        mu = lambda v: float(np.mean(v)) if v else 0.0
        per = {"butler_s": mu(t_butler), "subtract_s": mu(t_subtract),
               "nn_s": mu(t_nn), "ext_cnn_s": mu(t_ec)}
        gpu = per["nn_s"] + per["ext_cnn_s"]
        print(f"  mean per panel: butler={per['butler_s']:.2f}s subtract={per['subtract_s']:.2f}s | "
              f"GPU(nn+ext+cnn)={gpu:.2f}s", file=sys.stderr)
        print(f"  WALL: {wall:.1f}s   THROUGHPUT: {n_panels/wall:.3f} pan/s "
              f"({n_panels*60/wall:.1f} pan/min = "
              f"{N_DETECTORS_PER_VISIT*wall/n_panels:.1f} s/visit)", file=sys.stderr)
        for label, target_s in [("1 visit/min (60 s/visit)", 60.0),
                                ("LSST cadence (30 s/visit)", 30.0)]:
            pps = N_DETECTORS_PER_VISIT / target_s
            need_g = pps * gpu
            need_c = pps * per["butler_s"]
            scale_factor = pps / (n_panels / wall)
            print(f"  -> {label:30s}: ~{need_g:5.1f} GPUs + {need_c:5.1f} Butler workers "
                  f"(scale-out factor {scale_factor:.2f}× from this {len(devices)}-GPU run)",
                  file=sys.stderr)
    print("[inference] done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
