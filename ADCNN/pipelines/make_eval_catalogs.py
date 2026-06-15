"""ENTRY POINT — produce detection catalogs for the evaluation test sets (the inference
half of the catalog-based evaluation), then print trail-overlap metrics per set.

For each test set it runs the optimized multi-GPU engine (``build_detection_catalog_multigpu``:
segmentation model inference with parallel CPU prep, pipelined across all GPUs; candidate features + cutout CNN
in a process pool; ``gate_pmax`` cheap-gate) and writes ``<out>/<set>_detections.csv``. It then
matches each catalog against the set's truth CSV (``evaluate_catalog``) and prints
recall / FP-per-panel / wall-time. Inference only — no training, fixed CNN operating point.

Speed (4×A100, parallel-prep + batch 64 + gate 0.10): ~1.0 s/panel on sparse synthetic
panels, ~1.4 s/panel on the candidate-dense real set — i.e. a full ~189-detector LSST visit
in ~3–4 min, images→catalog (NN + features + CNN all included, postprocessing hidden behind
the GPU forward).

    python -m ADCNN.pipelines.make_eval_catalogs                 # the simulated `test` set
    python -m ADCNN.pipelines.make_eval_catalogs --sets test test_real
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

from ADCNN.inference.catalog import build_detection_catalog_multigpu, InferenceConfig
from ADCNN.inference.cnn_postproc import CNN_DEFAULT_THR
from ADCNN.evaluation.catalog_match import evaluate_catalog

REPO = Path(__file__).resolve().parents[2]
# The simulated test set is now ONE build-once/detect-many set (`test.h5`/`test.csv`): the NN runs
# once and its metrics are sigma-independent, so a single inference covers all stack-sigma baselines
# (the per-sigma stack comparison lives in the evaluation notebook).
DEFAULT_SETS = ["test"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS, help="test-set dirs under DATA_DIFFIM/")
    from ADCNN.config import ACTIVE as _PIPE  # active pipeline (ADCNN_PIPELINE selects; default=current)
    _seg_def = str(_PIPE.seg_model) if _PIPE else str(REPO / "models/current/segmentation_scripted.pt")
    _cnn_def = str(_PIPE.cnn_model) if _PIPE else str(REPO / "models/current/cnn_postproc.pt")
    ap.add_argument("--data-root", default=str(REPO / "DATA_DIFFIM"))
    ap.add_argument("--seg-model", default=_seg_def, help="stage-1 segmentation (default: active pipeline)")
    ap.add_argument("--cnn", default=_cnn_def, help="stage-2 cutout CNN (default: active pipeline)")
    ap.add_argument("--out", default=str(REPO / "Evaluation/catalogs"))
    ap.add_argument("--cnn-thr", type=float, default=CNN_DEFAULT_THR, help="CNN operating point (pre-chosen)")
    ap.add_argument("--gate-pmax", type=float, default=0.10,
                    help="cheap candidate gate: skip feature extraction where peak NN prob < this. "
                         "0.10 was validated TP-safe on val; raising it can drop genuine faint trails.")
    ap.add_argument("--tile-batch", type=int, default=64)
    ap.add_argument("--tol-px", type=float, default=20.0, help="trail-overlap match tolerance (fixed, pre-chosen)")
    ap.add_argument("--n-gpus", type=int, default=0, help="0 = all visible GPUs")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_gpus = a.n_gpus or torch.cuda.device_count()
    data_root = Path(a.data_root)

    for name in a.sets:
        # Accept BOTH layouts: flat (<data_root>/<name>.h5 + <name>.csv, from make_sim_data) and
        # the legacy subdir (<data_root>/<name>/test.h5 + test.csv).
        flat_h5 = data_root / f"{name}.h5"
        if flat_h5.exists():
            h5, truth_csv, panels = flat_h5, data_root / f"{name}.csv", data_root / f"{name}_panels.csv"
        else:
            d = data_root / name
            h5, truth_csv, panels = d / "test.h5", d / "test.csv", d / "panels.csv"
        if not h5.exists():
            print(f"[skip] {name}: no {h5}", flush=True)
            continue
        import h5py
        with h5py.File(h5, "r") as _f:        # true panel count for the fp_per_panel denominator
            n_panels = int(_f["images"].shape[0])
        t0 = time.time()
        cfg = InferenceConfig(cnn_thr=a.cnn_thr, gate_pmax=a.gate_pmax, tile_batch=a.tile_batch)
        cat = build_detection_catalog_multigpu(
            str(h5), a.seg_model, a.cnn, config=cfg, n_gpus=n_gpus,
            panels_csv=str(panels) if panels.exists() else None,
        )
        out_csv = out / f"{name}_detections.csv"
        cat.to_csv(out_csv, index=False)
        dt = time.time() - t0

        if truth_csv.exists():
            c, _ = evaluate_catalog(cat, truth_csv, tol_px=a.tol_px, n_panels=n_panels)
            metrics = {k: c[k] for k in ("TP", "FP", "FN", "recall", "fp_per_panel", "n_panels") if k in c}
            metrics.update(set=name, seg_model=a.seg_model, cnn=a.cnn, cnn_thr=a.cnn_thr, tol_px=a.tol_px)
            (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"[{name}] {len(cat)} det -> {out_csv.name} | TP={c['TP']} FP={c['FP']} FN={c['FN']} "
                  f"recall={c['recall']:.3f} FP/panel={c['fp_per_panel']:.1f} | "
                  f"{dt:.0f}s ({dt/max(c['n_panels'],1):.2f}s/panel)", flush=True)
        else:
            print(f"[{name}] {len(cat)} det -> {out_csv.name} | {dt:.0f}s (no truth CSV to score)", flush=True)
    print("make_eval_catalogs DONE", flush=True)


if __name__ == "__main__":
    main()
