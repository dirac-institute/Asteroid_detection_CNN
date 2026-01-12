from __future__ import annotations
import numpy as np
import sys
from pathlib import Path
proj = Path("../../../")
sys.path.insert(0, str(proj))
sys.path.insert(0, str(proj / "ADCNN" / "data" / "dataset_creation"))
import ADCNN.evaluation as evaluation


def two_threshold_prediction (predictions, t_low=0.1, pixel_gap=10, score_method="max"):
    predicted_scores = np.copy(predictions)
    for image_id in range(predictions.shape[0]):
        lab, n = evaluation._label_components_fds(predictions[image_id] >= t_low, pixel_gap=pixel_gap)
        for i in range(1, n + 1):
            print (f"\rProcessing image {image_id + 1}, component {i}/{n}   ", end="")
            comp_mask = lab == i
            if score_method == "max":
                comp_score = predictions[image_id][comp_mask].max()
            elif score_method == "topk_mean":
                vals = predictions[image_id][comp_mask]
                k = int (vals.size*0.05) if int (vals.size*0.05) > 0 else 1
                top = np.partition(vals, vals.size - k)[-k:]
                comp_score = float(top.mean())
            else:
                raise ValueError(f"Unknown score_method={score_method!r}")
            predicted_scores[image_id][comp_mask] = comp_score
    return predicted_scores


import os
import math
import numpy as np
import concurrent.futures as cf
import multiprocessing as mp
from multiprocessing import shared_memory

import ADCNN.evaluation as evaluation


def _process_chunk(args) -> None:
    """
    Attach to shm inside the task (robust in notebooks / spawn).
    Args contains (shm_in_name, shm_out_name, shape, dtype_str, t_low, pixel_gap, score_method, start, stop)
    """
    (shm_in_name, shm_out_name, shape, dtype_str,
     t_low, pixel_gap, score_method, start, stop) = args

    shm_in = shared_memory.SharedMemory(name=shm_in_name)
    shm_out = shared_memory.SharedMemory(name=shm_out_name)
    try:
        dtype = np.dtype(dtype_str)
        inp = np.ndarray(shape, dtype=dtype, buffer=shm_in.buf)
        out = np.ndarray(shape, dtype=dtype, buffer=shm_out.buf)

        for image_id in range(start, stop):
            pred = inp[image_id]  # (H,W)
            lab, n = evaluation._label_components_fds(pred >= t_low, pixel_gap=pixel_gap)
            if n <= 0:
                continue

            if score_method == "max":
                for i in range(1, n + 1):
                    comp = (lab == i)
                    if not comp.any():
                        continue
                    comp_score = float(pred[comp].max())
                    out[image_id][comp] = comp_score

            elif score_method == "topk_mean":
                for i in range(1, n + 1):
                    comp = (lab == i)
                    if not comp.any():
                        continue
                    vals = pred[comp]
                    k = int(vals.size * 0.05)
                    if k < 1:
                        k = 1
                    top = np.partition(vals, vals.size - k)[-k:]
                    comp_score = float(top.mean())
                    out[image_id][comp] = comp_score
            else:
                raise ValueError(f"Unknown score_method={score_method!r}")

    finally:
        shm_in.close()
        shm_out.close()


def two_threshold_prediction_mp(
    predictions: np.ndarray,
    *,
    t_low: float = 0.1,
    pixel_gap: int = 10,
    score_method: str = "max",
    workers: int | None = None,
    chunk_images: int | None = None,
    start_method: str = "spawn",  # notebook-safe default
) -> np.ndarray:
    """
    Multiprocessed two-threshold postprocess using shared memory.
    Works reliably in Jupyter (spawn), and also works in scripts (fork/spawn).
    """
    if predictions.ndim != 3:
        raise ValueError(f"predictions must be (N,H,W); got {predictions.shape}")

    pred = np.ascontiguousarray(predictions, dtype=np.float32)
    out = pred.copy()

    N, H, W = pred.shape
    workers = int(workers or max(1, (os.cpu_count() or 1) - 1))

    if chunk_images is None:
        # ~6 chunks per worker
        chunk_images = max(1, int(math.ceil(N / (workers * 6))))
    chunk_images = int(chunk_images)

    shm_in = shared_memory.SharedMemory(create=True, size=pred.nbytes)
    shm_out = shared_memory.SharedMemory(create=True, size=out.nbytes)

    try:
        shm_in_arr = np.ndarray(pred.shape, dtype=pred.dtype, buffer=shm_in.buf)
        shm_out_arr = np.ndarray(out.shape, dtype=out.dtype, buffer=shm_out.buf)
        shm_in_arr[...] = pred
        shm_out_arr[...] = out

        tasks = []
        for s in range(0, N, chunk_images):
            e = min(N, s + chunk_images)
            tasks.append((
                shm_in.name, shm_out.name, pred.shape, pred.dtype.str,
                float(t_low), int(pixel_gap), str(score_method),
                int(s), int(e)
            ))

        ctx = mp.get_context(start_method)
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            list(ex.map(_process_chunk, tasks, chunksize=1))

        return np.array(shm_out_arr, copy=True)

    finally:
        try:
            shm_in.close(); shm_in.unlink()
        except Exception:
            pass
        try:
            shm_out.close(); shm_out.unlink()
        except Exception:
            pass