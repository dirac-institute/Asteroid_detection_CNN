"""GPU-side consumer for the streaming test_real pipeline (multi-GPU).

Reads length-prefixed pickled diffim messages from stdin (produced by
``stream_real_butler.py`` running in the LSST stack env), runs the shipped two-stage detector
on every visible GPU in TRUE parallel (one worker thread per device), and emits a per-detection
CSV matching the public catalog schema for the Evaluation_Real notebook.

The threading layout: one reader thread consumes stdin and pushes panels onto a bounded queue;
N worker threads (one per device) each own a (seg, cnn) pair on their device and pump panels
through the inference path; a writer thread accumulates the per-detection rows. PyTorch releases
the GIL during CUDA kernel launches, so the workers run concurrently across GPUs.
"""
from __future__ import annotations
import argparse
import json
import os
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

CAT_COLS = ["image_id", "visit", "detector", "x", "y", "beta", "length", "flux",
            "mf_snr", "area", "elongation", "nn_pmax", "score"]
MF_LEN_OFFSET = 33.4
MF_LEN_SLOPE = 0.887


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


def _detect_one(seg, cnn, img, device, thr, k):
    import torch
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import extract_panel_candidates
    from ADCNN.inference.cnn_postproc import apply_cnn
    rl = np.zeros_like(img, dtype=np.uint16)
    t0 = time.perf_counter()
    prob, _, _, agg = predict_panel_overlap_3ch_full(seg, img, rl, device=device)
    if device.type == "cuda": torch.cuda.synchronize(device)
    t_nn = time.perf_counter() - t0
    prob = prob.astype(np.float32); agg = np.asarray(agg, np.float32)
    t0 = time.perf_counter()
    cand, _ = extract_panel_candidates({0: prob}, {0: img}, real_labels={0: rl})
    t_ext = time.perf_counter() - t0
    t0 = time.perf_counter()
    if len(cand):
        cand = apply_cnn(cand, cnn, img, prob, agg, device=device, k=k)
        kept = cand[cand["score"] >= thr]
    else:
        kept = cand
    if device.type == "cuda": torch.cuda.synchronize(device)
    t_cnn = time.perf_counter() - t0
    return kept, (t_nn, t_ext, t_cnn)


def _worker(device, seg_path, cnn_path, in_q, out_q, thr, k, stats):
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
            kept, (t_nn, t_ext, t_cnn) = _detect_one(seg, cnn, img, dev, thr, k)
            n_kept = int(len(kept)) if kept is not None else 0
            row_df = None
            if n_kept:
                kept = kept.copy()
                if "mf_length" in kept.columns:
                    kept["mf_length"] = np.clip(
                        (kept["mf_length"] - MF_LEN_OFFSET) / MF_LEN_SLOPE, 0.0, None)
                kept = kept.rename(columns={
                    "x_centroid": "x", "y_centroid": "y", "mf_beta": "beta",
                    "mf_length": "length", "mf_flux": "flux", "max_p": "nn_pmax"})
                kept["image_id"] = int(image_id)
                kept["visit"]    = int(msg["visit"])
                kept["detector"] = int(msg["detector"])
                row_df = kept[[c for c in CAT_COLS if c in kept.columns]]
            out_q.put({"image_id": image_id, "visit": msg["visit"], "detector": msg["detector"],
                       "rows": row_df, "n_kept": n_kept, "t_nn": t_nn, "t_ext": t_ext,
                       "t_cnn": t_cnn, "t_butler": msg.get("t_butler_s", 0.0),
                       "t_subtract": msg.get("t_subtract_s", 0.0),
                       "source": msg.get("source", "")})
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
            print(f"  UPSTREAM ({msg.get('visit')}, {msg.get('detector')}): {msg['err']}", file=sys.stderr, flush=True)
            continue
        in_q.put((msg, image_id))
        image_id += 1
    # poison pills: one per worker
    for _ in range(n_workers):
        in_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=str(REPO / "models/seg_v2_segmentation_scripted.pt"))
    ap.add_argument("--cnn", default=str(REPO / "models/seg_v2_cnn_postproc.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=0, help="0 = use all visible GPUs")
    ap.add_argument("--queue-size", type=int, default=16, help="bounded queue depth (RAM cap)")
    a = ap.parse_args()

    import torch
    n_avail = torch.cuda.device_count()
    n_workers = a.gpus if a.gpus > 0 else (n_avail or 1)
    devices = [f"cuda:{i}" for i in range(n_workers)] if n_avail else ["cpu"]
    print(f"[inference] devices={devices}", file=sys.stderr, flush=True)

    sc = Path(a.cnn).with_suffix(".json")
    info = json.loads(sc.read_text()) if sc.exists() else {}
    thr = float(info.get("threshold", 0.6))
    k = int(info.get("k", 96))
    print(f"[inference] thr={thr}  k={k}", file=sys.stderr, flush=True)

    in_q = queue.Queue(maxsize=max(a.queue_size, 2 * len(devices)))
    out_q = queue.Queue()
    stats = {d: {"n": 0} for d in devices}
    n_fail = {"n": 0}

    # Start the workers (one per device).
    threads = []
    for d in devices:
        t = threading.Thread(target=_worker, args=(d, a.seg, a.cnn, in_q, out_q, thr, k, stats), daemon=True)
        t.start(); threads.append(t)

    # Start the stdin reader.
    rt = threading.Thread(target=_reader, args=(in_q, sys.stdin.buffer, len(devices), n_fail), daemon=True)
    rt.start()

    rows = []
    n_panels = 0; n_emit = 0
    t_nn = []; t_ext = []; t_cnn = []; t_butler = []; t_subtract = []
    finished_workers = 0
    t_wall0 = time.perf_counter()

    while finished_workers < len(devices):
        item = out_q.get()
        if item is None:
            finished_workers += 1
            continue
        if item.get("err"):
            print(f"  panel ({item['visit']},{item['detector']}): WORKER {item['err']}", file=sys.stderr, flush=True)
            continue
        n_panels += 1
        n_emit += int(item["n_kept"])
        t_nn.append(item["t_nn"]); t_ext.append(item["t_ext"]); t_cnn.append(item["t_cnn"])
        t_butler.append(item["t_butler"]); t_subtract.append(item["t_subtract"])
        if item["rows"] is not None:
            rows.append(item["rows"])
        if n_panels % 25 == 0:
            elapsed = time.perf_counter() - t_wall0
            rate = n_panels / max(elapsed, 1e-9)
            mu = lambda v: float(np.mean(v)) if v else 0.0
            print(f"  [{n_panels}]  rate={rate:.3f} pan/s ({rate*60:.1f}/min)  "
                  f"mean butler={mu(t_butler):.2f}s nn={mu(t_nn):.2f}s ext={mu(t_ext):.2f}s "
                  f"cnn={mu(t_cnn):.2f}s", file=sys.stderr, flush=True)

    wall = time.perf_counter() - t_wall0

    if rows:
        out = pd.concat(rows, ignore_index=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, index=False)

    print("\n=== STREAM-REAL SUMMARY ===", file=sys.stderr)
    print(f"  devices: {len(devices)}  ({devices})", file=sys.stderr)
    print(f"  panels OK: {n_panels}  FAIL upstream: {n_fail['n']}  rows emitted: {n_emit}", file=sys.stderr)
    if n_panels:
        mu = lambda v: float(np.mean(v)) if v else 0.0
        per = {"butler_s": mu(t_butler), "subtract_s": mu(t_subtract),
               "nn_s": mu(t_nn), "extract_s": mu(t_ext), "cnn_s": mu(t_cnn)}
        gpu = per["nn_s"] + per["extract_s"] + per["cnn_s"]
        print(f"  mean per panel: butler={per['butler_s']:.2f}s subtract={per['subtract_s']:.2f}s | "
              f"GPU(nn+ext+cnn)={gpu:.2f}s", file=sys.stderr)
        print(f"  WALL: {wall:.1f}s   THROUGHPUT: {n_panels/wall:.3f} pan/s "
              f"({n_panels*60/wall:.1f} pan/min = {189*wall/n_panels:.1f} s/visit)", file=sys.stderr)
        for label, target_s in [("1 visit/min (60 s/visit)", 60.0),
                                  ("LSST cadence (30 s/visit)", 30.0)]:
            pps = 189 / target_s
            need_g = pps * gpu     # GPU workers needed
            need_c = pps * per["butler_s"]   # CPU workers (Butler workers) needed
            scale_factor = pps / (n_panels / wall)
            print(f"  -> {label:30s}: ~{need_g:5.1f} GPUs + {need_c:5.1f} Butler workers "
                  f"(measured scale-out factor {scale_factor:.2f}× from this {len(devices)}-GPU run)", file=sys.stderr)
    print("[inference] done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
