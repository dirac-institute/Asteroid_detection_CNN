#!/usr/bin/env python
"""Pre-flight reproducibility check for the promoted diffim pipeline.

CPU-only, no DATA_DIFFIM, no GPU, ~20 s. It does not retrain or re-evaluate;
it asserts that the *promoted artifacts* on disk still satisfy the invariants
the result depends on, so a "recreate from zero" run is starting from a sane
state. Run it before / after `promote.sh`, or as a fast CI-style guard.

What it verifies:
  1. The lazy ADCNN.inference API surface resolves (Stage-1 + Stage-2 names).
  2. DEFAULT_THR == 0.50 and len(RF_FEATURES_V2) == 72  (operating point).
  3. The promoted scripted v7 loads on CPU and emits the 5-tuple
     (seg_logits, orient_sin, orient_cos, raw_seg, agg) at the right shapes
     via the real sliding-window entrypoint.
  4. The promoted RF's feature contract is in lock-step with RF_FEATURES_V2
     (n_features_in_ and, when present, feature_names_in_ match exactly).
  5. The full Stage-1-features -> Stage-2-RF -> mask path runs end to end on
     a synthetic panel and yields finite scores in [0, 1].

Exit 0 = PASS. Exit 0 + "SKIP" = promoted ckpts absent (fresh checkout;
gitignored) — nothing to validate yet. Exit 1 = a contract broke.

Run from the repo root with the asteroid_cnn env active (ADCNN must be
importable — the SLURM wrappers set PYTHONPATH; locally, cwd=repo root):

    python ADCNN/scripts/test_real/validate_pipeline.py
    python ADCNN/scripts/test_real/validate_pipeline.py --repo-dir /path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Standalone script in a non-package dir: ensure the repo root (which holds the
# ADCNN package) is importable even when launched as `python <path>` without
# PYTHONPATH set. The SLURM wrappers also set PYTHONPATH; this is idempotent.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from ADCNN.inference import (
    predict_panel_overlap_3ch_full,
    compute_v2_features,
    apply_rf_v2,
    materialize_label_mask_v2,
    load_rf,
    RF_FEATURES_V2,
    DEFAULT_THR,
)
from ADCNN.utils.helpers import draw_one_line


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-dir",
        default=str(Path(__file__).resolve().parents[3]),
        help="repo root (default: inferred from this file's location)",
    )
    args = ap.parse_args()
    repo = Path(args.repo_dir)
    ck = repo / "experiments/diffim_runs/pilot_v7/ckpts"
    model_pt = ck / "v7_scripted.pt"
    rf_pkl = ck / "rf_postproc_v2.pkl"

    print("== ADCNN promoted-pipeline validation ==")

    # 1 + 2: API surface + operating-point constants ------------------------
    if len(RF_FEATURES_V2) != 72:
        _fail(f"RF_FEATURES_V2 has {len(RF_FEATURES_V2)} features, expected 72")
    if abs(DEFAULT_THR - 0.50) > 1e-9:
        _fail(f"DEFAULT_THR={DEFAULT_THR}, expected 0.50 (fine-tuned v7 point)")
    if len(set(RF_FEATURES_V2)) != len(RF_FEATURES_V2):
        _fail("RF_FEATURES_V2 contains duplicate names")
    print(f"  ok    API resolves; DEFAULT_THR={DEFAULT_THR}; "
          f"{len(RF_FEATURES_V2)} unique RF features")

    if not (model_pt.exists() and rf_pkl.exists()):
        print(f"  SKIP  promoted ckpts not on disk under {ck} "
              "(gitignored — run the build/finetune/promote pipeline first)")
        print("VALIDATE: SKIP (no artifacts) — API surface OK")
        sys.exit(0)

    device = torch.device("cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    # 3: promoted scripted model loads + emits the 5-tuple ------------------
    model = torch.jit.load(str(model_pt), map_location=device).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 128, 128, dtype=torch.float32))
    if not (isinstance(out, (tuple, list)) and len(out) == 5):
        _fail(f"scripted model returned {type(out)} of len "
              f"{len(out) if hasattr(out, '__len__') else '?'}, expected 5")
    seg_logits = out[0]
    if tuple(seg_logits.shape) != (2, 1, 128, 128):
        _fail(f"seg_logits shape {tuple(seg_logits.shape)}, expected (2,1,128,128)")
    print(f"  ok    v7_scripted.pt loads on CPU; 5-tuple; "
          f"seg_logits{tuple(seg_logits.shape)}")

    # 4: RF feature contract is in lock-step with RF_FEATURES_V2 ------------
    rf = load_rf(rf_pkl)
    n_in = getattr(rf, "n_features_in_", None)
    if n_in is not None and int(n_in) != 72:
        _fail(f"rf.n_features_in_={n_in}, expected 72 == len(RF_FEATURES_V2)")
    names = getattr(rf, "feature_names_in_", None)
    if names is not None and tuple(names) != tuple(RF_FEATURES_V2):
        diff = [(i, a, b) for i, (a, b) in
                enumerate(zip(list(names), RF_FEATURES_V2)) if a != b]
        _fail(f"rf.feature_names_in_ != RF_FEATURES_V2 (first mismatch: "
              f"{diff[0] if diff else 'len differs'})")
    print(f"  ok    RF contract: n_features_in_={n_in}; "
          f"feature_names {'match' if names is not None else 'n/a (positional)'}")

    # 5: full Stage-1-features -> Stage-2 path on a synthetic panel ---------
    H = W = 256
    rng = np.random.default_rng(0)
    diffim = rng.standard_normal((H, W)).astype(np.float32)
    draw_one_line(diffim, origin=(60.0, 70.0), angle_deg=27.0,
                  length=90.0, true_value=8, line_thickness=3)
    probs = np.zeros((H, W), dtype=np.float32)
    # a deterministic high-probability streak so >=1 candidate is extracted
    yy, xx = np.ogrid[:H, :W]
    line = (np.abs((yy - 70) - np.tan(np.deg2rad(27.0)) * (xx - 60)) < 2.5) & \
           (xx >= 60) & (xx <= 60 + 90)
    probs[line] = 0.95
    zeros = np.zeros((H, W), dtype=np.float32)
    real_labels = np.zeros((H, W), dtype=np.float32)

    # Stage-1 sliding-window entrypoint on the real promoted model.
    pmap, smap, cmap, amap = predict_panel_overlap_3ch_full(
        model, diffim, real_labels, device=device, tile=128, stride=64)
    for nm, m in (("prob", pmap), ("sin", smap), ("cos", cmap), ("agg", amap)):
        if tuple(m.shape) != (H, W):
            _fail(f"predict_* {nm} map shape {tuple(m.shape)}, expected {(H, W)}")
        if not np.isfinite(np.asarray(m, dtype=np.float32)).all():
            _fail(f"predict_* {nm} map has non-finite values")
    print(f"  ok    predict_panel_overlap_3ch_full -> 4 finite {H}x{W} maps")

    # Stage-2: features (synthetic probs guarantee a candidate) -> RF -> mask.
    # compute_v2_features wants (N,H,W) stacks / {pid:(H,W)}; pass one panel.
    cand_df, probs_dict = compute_v2_features(
        probs[None], diffim[None], zeros[None], zeros[None], zeros[None],
        real_labels=real_labels[None])
    if len(cand_df) == 0:
        _fail("compute_v2_features produced 0 candidates on the synthetic streak")
    missing = [c for c in RF_FEATURES_V2 if c not in cand_df.columns]
    if missing:
        _fail(f"cand_df missing {len(missing)} RF feature cols, e.g. {missing[:5]}")
    scored = apply_rf_v2(cand_df, rf)
    if "score_rf" not in scored.columns:
        _fail("apply_rf_v2 did not set score_rf")
    s = scored["score_rf"].to_numpy(dtype=np.float64)
    if not np.isfinite(s).all() or s.min() < 0.0 or s.max() > 1.0:
        _fail(f"score_rf out of range/finite: min={s.min()} max={s.max()}")
    # materialize wants the full (N,H,W) output shape (one panel here).
    masks = materialize_label_mask_v2(
        scored[scored.score_rf >= DEFAULT_THR], probs_dict, (1, H, W))
    if tuple(masks.shape) != (1, H, W):
        _fail(f"materialize_label_mask_v2 shape {tuple(masks.shape)}, "
              f"expected {(1, H, W)}")
    print(f"  ok    Stage-2: {len(cand_df)} cand -> RF score "
          f"[{s.min():.3f},{s.max():.3f}]; mask{tuple(masks.shape)} OK")

    print(f"VALIDATE: PASS — promoted v7 + RF consistent at DEFAULT_THR="
          f"{DEFAULT_THR} (72-feature lock-step held)")
    sys.exit(0)


if __name__ == "__main__":
    main()
