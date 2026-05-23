"""Full evaluation pipeline for the diffim NN (consolidated from
the diffim eval prototype).

Loads a trained checkpoint (or TorchScript file), runs sliding-window
inference on the test_{5,4,3}sigma splits, extracts candidates, matches
them to injection truths, and writes:
  - <split>_panel_probs.npz       per-panel probability maps (float16)
  - <split>_candidates.csv        per-candidate features + matched_injection_id
  - <split>_per_injection.csv     per-injection match info incl. stack_detection
  - froc plots, completeness plots, summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ADCNN.training.ema import EMAModel
from ADCNN.core.diffim_model import UNetResSEOrientHough
from ADCNN.data.diffim_dataset import build_3channel, diffim_mad_sigma
from ADCNN.inference.diffim_candidates import (
    extract_candidates,
    candidate_pixel_mask,
    CandidateExtractorConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "DATA_DIFFIM"
SPLIT_PATHS = {
    "test_5sigma": (DATA_DIR / "test_5sigma" / "test.h5", DATA_DIR / "test_5sigma" / "test.csv"),
    "test_4sigma": (DATA_DIR / "test_4sigma" / "test.h5", DATA_DIR / "test_4sigma" / "test.csv"),
    "test_3sigma": (DATA_DIR / "test_3sigma" / "test.h5", DATA_DIR / "test_3sigma" / "test.csv"),
}


# ---------------------------------------------------------------------------
# Inlined inference + candidate-matching helpers.
# ---------------------------------------------------------------------------
def hann2d(tile: int) -> np.ndarray:
    """2D Hann window for blending overlapping tile predictions."""
    w = np.hanning(tile + 2)[1:-1]
    return (w[:, None] * w[None, :]).astype(np.float32)


def _draw_line(mask: np.ndarray, x: float, y: float, beta_deg: float,
               length: float, value: int, thickness: int = 3) -> np.ndarray:
    """In-place draw of a thick line centered at (x, y). Same geometry as
    ADCNN/data/dataset_creation/common.py::draw_one_line, inlined here to
    avoid the lsst.geom import at eval time."""
    xs = float(length) * math.cos(math.radians(beta_deg))
    ys = float(length) * math.sin(math.radians(beta_deg))
    x1 = int(round(x - xs / 2.0)); y1 = int(round(y - ys / 2.0))
    x2 = int(round(x + xs / 2.0)); y2 = int(round(y + ys / 2.0))
    line = cv2.line(np.zeros(mask.shape, dtype=np.uint8), (x2, y2), (x1, y1), 1,
                    thickness=int(thickness))
    mask[line != 0] = value
    return mask


def build_truth_id_mask(masks_panel: np.ndarray, df_panel: pd.DataFrame) -> np.ndarray:
    """Paint each injection's drawn-line truth into an int32 id mask."""
    H, W = masks_panel.shape
    id_mask = np.zeros((H, W), dtype=np.int32)
    if len(df_panel) == 0:
        return id_mask
    for _, row in df_panel.sort_values("trail_length", ascending=True).iterrows():
        _draw_line(
            id_mask, float(row["x"]), float(row["y"]),
            float(row["beta"]), float(row["trail_length"]),
            value=int(row["injection_idx"]),
            thickness=3,
        )
    return id_mask


def match_candidates_to_injections(
    candidates: pd.DataFrame,
    truth_id_mask: np.ndarray,
    panel_prob: np.ndarray,
    *,
    t_low: float,
    min_overlap: int = 1,
) -> pd.DataFrame:
    """Add columns matched_injection_id, matched_overlap_px, matched_iou."""
    out = candidates.copy()
    matched_id = np.full(len(out), -1, dtype=np.int64)
    matched_ov = np.zeros(len(out), dtype=np.int64)
    matched_iou = np.zeros(len(out), dtype=np.float64)
    for i, (_, cand) in enumerate(out.iterrows()):
        c_t_low = float(cand["effective_t_low"]) if "effective_t_low" in cand else float(t_low)
        m = candidate_pixel_mask(truth_id_mask.shape, cand, panel_prob, t_low=c_t_low)
        if not m.any():
            continue
        ids = truth_id_mask[m]
        ids = ids[ids > 0]
        if ids.size == 0:
            continue
        vals, counts = np.unique(ids, return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] >= min_overlap:
            matched_id[i] = int(vals[best])
            matched_ov[i] = int(counts[best])
            truth_for_id = (truth_id_mask == vals[best])
            iou = float((m & truth_for_id).sum()) / float((m | truth_for_id).sum() + 1e-12)
            matched_iou[i] = iou
    out["matched_injection_id"] = matched_id
    out["matched_overlap_px"] = matched_ov
    out["matched_iou"] = matched_iou
    return out


def compute_metrics(
    inj_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    score_grid: np.ndarray,
    *,
    score_col: str = "max_p",
    real_label_thresh: float = 0.5,
) -> dict:
    """Per-score-threshold object-level metrics."""
    panels = sorted(inj_df["panel_id"].unique().tolist())
    n_panels = len(panels)
    results = []
    rec_score = inj_df["matched_score"].to_numpy()
    rec_id = inj_df["matched_candidate_id"].to_numpy()
    stk = inj_df["stack_detection"].astype(bool).to_numpy()
    for s in score_grid:
        nn_rec = (rec_id >= 0) & (rec_score >= s)
        cand_above = cand_df[cand_df[score_col] >= s]
        unmatched = cand_above["matched_injection_id"].to_numpy() < 0
        is_info = (cand_above["frac_real_label_overlap"].to_numpy() >= real_label_thresh)
        spurious = unmatched & ~is_info
        informational = unmatched & is_info
        union = stk | nn_rec
        lsst_missed_nn = (~stk) & nn_rec
        results.append({
            "score": float(s),
            "N_total": int(len(inj_df)),
            "N_lsst": int(stk.sum()),
            "N_nn": int(nn_rec.sum()),
            "N_union": int(union.sum()),
            "N_lsst_missed_nn_recovered": int(lsst_missed_nn.sum()),
            "N_lsst_missed_total": int((~stk).sum()),
            "recall_lsst": float(stk.mean()),
            "recall_nn": float(nn_rec.mean()),
            "recall_union": float(union.mean()),
            "recall_lsst_missed_nn": float(lsst_missed_nn.sum() / max(int((~stk).sum()), 1)),
            "n_candidates_above_score": int(len(cand_above)),
            "n_spurious_above_score": int(spurious.sum()),
            "n_informational_above_score": int(informational.sum()),
            "spurious_per_panel": float(spurious.sum() / max(n_panels, 1)),
            "informational_per_panel": float(informational.sum() / max(n_panels, 1)),
            "candidates_per_panel": float(len(cand_above) / max(n_panels, 1)),
        })
    return {"sweep": results, "n_panels": n_panels}


def plot_froc(sweep: list[dict], out_path: Path, label: str):
    sp = np.array([s["spurious_per_panel"] for s in sweep])
    rec_nn = np.array([s["recall_nn"] for s in sweep])
    rec_union = np.array([s["recall_union"] for s in sweep])
    rec_lsst = sweep[0]["recall_lsst"]
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.plot(sp, rec_nn, "-o", label="NN-only recall", markersize=3)
    ax.plot(sp, rec_union, "-^", label="NN ∪ LSST recall", markersize=3)
    ax.axhline(rec_lsst, ls="--", color="k", label=f"LSST-only recall ({rec_lsst:.2f})")
    ax.set_xlabel("Spurious candidates per panel")
    ax.set_ylabel("Object recall")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"FROC — {label}")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_completeness_vs(inj_df, nn_score_threshold, field, bins, out_path, label):
    stk = inj_df["stack_detection"].astype(bool).to_numpy()
    rec_nn = (inj_df["matched_candidate_id"].to_numpy() >= 0) & (
        inj_df["matched_score"].to_numpy() >= nn_score_threshold
    )
    union = stk | rec_nn
    lsst_missed_nn = (~stk) & rec_nn

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    f = inj_df[field].to_numpy()

    def binned(mask, denom_mask):
        idx = np.digitize(f, bins) - 1
        x, y = [], []
        for i in range(len(bins) - 1):
            in_bin = (idx == i) & denom_mask
            if int(in_bin.sum()) == 0:
                continue
            x.append(0.5 * (bins[i] + bins[i + 1]))
            y.append(float(mask[in_bin].sum()) / int(in_bin.sum()))
        return np.array(x), np.array(y)

    all_mask = np.ones(len(inj_df), dtype=bool)
    for arr, lbl, marker in [(stk, "LSST", "-o"),
                             (rec_nn, "NN", "-s"),
                             (union, "NN ∪ LSST", "-^")]:
        x, y = binned(arr, all_mask)
        if x.size:
            ax.plot(x, y, marker, label=lbl, markersize=3)
    x, y = binned(lsst_missed_nn, ~stk)
    if x.size:
        ax.plot(x, y, "-d", label="NN on LSST-missed (per denom)", markersize=3)
    ax.set_xlabel(field)
    ax.set_ylabel("Object recall")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Completeness vs {field} — {label}\n(NN score ≥ {nn_score_threshold:.2f})")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_visual_panels(run_dir, split, h5_path, panel_probs, cand_df, inj_df,
                       nn_score_threshold, n_each=5):
    out_dir = Path(run_dir) / "panels" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    matched = inj_df["matched_candidate_id"].to_numpy() >= 0
    rec_nn = matched & (inj_df["matched_score"].to_numpy() >= nn_score_threshold)
    stk = inj_df["stack_detection"].astype(bool).to_numpy()
    categories = {
        "tp": inj_df[rec_nn].sample(n=min(n_each, int(rec_nn.sum())),
                                    random_state=int(rng.integers(0, 1_000_000)))
              if rec_nn.any() else inj_df.iloc[:0],
        "lsst_missed_nn_rec": inj_df[(~stk) & rec_nn].sample(
            n=min(n_each, int(((~stk) & rec_nn).sum())),
            random_state=int(rng.integers(0, 1_000_000))) if ((~stk) & rec_nn).any() else inj_df.iloc[:0],
        "nn_missed": inj_df[~rec_nn].sample(n=min(n_each, int((~rec_nn).sum())),
                                            random_state=int(rng.integers(0, 1_000_000)))
              if (~rec_nn).any() else inj_df.iloc[:0],
    }
    sp = cand_df[(cand_df["matched_injection_id"] < 0)
                 & (cand_df["max_p"] >= nn_score_threshold)
                 & (cand_df["frac_real_label_overlap"] < 0.5)]
    categories["fp"] = sp.sample(n=min(n_each, len(sp)),
                                 random_state=int(rng.integers(0, 1_000_000))) \
        if len(sp) else sp.iloc[:0]

    with h5py.File(h5_path, "r") as f:
        images = f["images"]; masks = f["masks"]; real_labels = f["real_labels"]
        for cat, rows in categories.items():
            for k, (_, r) in enumerate(rows.iterrows()):
                pid = int(r["panel_id"])
                if cat == "fp":
                    cy, cx = int(r["y_centroid"]), int(r["x_centroid"])
                else:
                    cy, cx = int(r["y"]), int(r["x"])
                half = 96
                y0 = max(0, cy - half); y1 = min(images.shape[1], y0 + 2 * half); y0 = max(0, y1 - 2 * half)
                x0 = max(0, cx - half); x1 = min(images.shape[2], x0 + 2 * half); x0 = max(0, x1 - 2 * half)
                img = images[pid, y0:y1, x0:x1].astype(np.float32)
                msk = masks[pid, y0:y1, x0:x1].astype(bool)
                rl = (real_labels[pid, y0:y1, x0:x1] > 0)
                prob = panel_probs[pid][y0:y1, x0:x1].astype(np.float32)
                sig = max(float(1.4826 * np.median(np.abs(img))), 1e-6)
                disp = np.clip(img / sig, -5, 5)
                fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                axs[0].imshow(disp, cmap="gray", vmin=-5, vmax=5, origin="lower")
                if msk.any():
                    axs[0].contour(msk.astype(np.uint8), levels=[0.5], colors=["#ff3030"], linewidths=0.8)
                if rl.any():
                    axs[0].contour(rl.astype(np.uint8), levels=[0.5], colors=["#3070ff"], linewidths=0.5, alpha=0.7)
                axs[0].set_title("diffim (red=truth, blue=real_labels)")
                axs[1].imshow(prob, cmap="magma", vmin=0, vmax=1, origin="lower")
                axs[1].set_title(f"NN prob (panel {pid})")
                axs[2].imshow(disp, cmap="gray", vmin=-5, vmax=5, origin="lower")
                axs[2].contour(prob >= nn_score_threshold, levels=[0.5], colors=["#30ff30"], linewidths=0.8)
                axs[2].set_title(f"NN candidate(p≥{nn_score_threshold:.2f}) overlay")
                for a in axs:
                    a.set_xticks([]); a.set_yticks([])
                fig.suptitle(f"cat={cat}", fontsize=9)
                fig.savefig(out_dir / f"{cat}_{k:02d}_pid{pid}_y{cy}_x{cx}.png", dpi=130)
                plt.close(fig)


@torch.no_grad()
def predict_panel_overlap_3ch(
    model: torch.nn.Module,
    panel_image: np.ndarray,
    panel_real_labels: np.ndarray,
    *,
    device,
    tile: int = 128,
    stride: int = 64,
    clip: float = 5.0,
    stats_crop: int = 1024,
) -> np.ndarray:
    """Sliding-window inference with Hann-weighted averaging on 3-channel input."""
    H, W = panel_image.shape
    s = min(stats_crop, H, W)
    h0c = (H - s) // 2
    w0c = (W - s) // 2
    sigma = diffim_mad_sigma(panel_image[h0c:h0c + s, w0c:w0c + s])

    prob_acc = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    hann = hann2d(tile)

    def starts(N, t, sstep):
        out = list(range(0, max(N - t, 0) + 1, sstep))
        if out[-1] != N - t:
            out.append(N - t)
        return out
    ys = starts(H, tile, stride)
    xs = starts(W, tile, stride)

    batch_xs, batch_locs = [], []
    BATCH = 24  # the line aggregator + 3-channel input make this the GPU-memory limit

    def flush():
        if not batch_xs:
            return
        # Each entry is already (3, T, T); stack to (B, 3, T, T)
        xb = torch.from_numpy(np.stack(batch_xs)).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seg_logits, _, _, _, _ = model(xb)
        probs = torch.sigmoid(seg_logits).cpu().numpy().astype(np.float32)
        for (y0, x0), p in zip(batch_locs, probs[:, 0]):
            prob_acc[y0:y0 + tile, x0:x0 + tile] += p * hann
            weight_acc[y0:y0 + tile, x0:x0 + tile] += hann
        batch_xs.clear(); batch_locs.clear()

    for y0 in ys:
        for x0 in xs:
            diffim_tile = panel_image[y0:y0 + tile, x0:x0 + tile]
            rl_tile = panel_real_labels[y0:y0 + tile, x0:x0 + tile]
            x3 = build_3channel(diffim_tile, rl_tile, panel_sigma=sigma, clip=clip)
            batch_xs.append(x3)
            batch_locs.append((y0, x0))
            if len(batch_xs) >= BATCH:
                flush()
    flush()

    out = prob_acc / np.maximum(weight_acc, 1e-6)
    return out.astype(np.float16)


@torch.no_grad()
def predict_panel_overlap_3ch_full(
    model: torch.nn.Module,
    panel_image: np.ndarray,
    panel_real_labels: np.ndarray,
    *,
    device,
    tile: int = 128,
    stride: int = 64,
    clip: float = 5.0,
    stats_crop: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window inference returning all auxiliary heads.

    Returns (prob, orient_sin, orient_cos, agg) — each (H, W) float16. `prob`
    is sigmoid(seg_logits); `orient_sin`/`orient_cos` are tanh-bounded
    sin(2β)/cos(2β); `agg` is the raw line-aggregator logit. All four maps
    use Hann-weighted overlap blending (same convention as
    `predict_panel_overlap_3ch`).
    """
    H, W = panel_image.shape
    s = min(stats_crop, H, W)
    h0c = (H - s) // 2
    w0c = (W - s) // 2
    sigma = diffim_mad_sigma(panel_image[h0c:h0c + s, w0c:w0c + s])

    prob_acc = np.zeros((H, W), dtype=np.float32)
    sin_acc  = np.zeros((H, W), dtype=np.float32)
    cos_acc  = np.zeros((H, W), dtype=np.float32)
    agg_acc  = np.zeros((H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    hann = hann2d(tile)

    def starts(N, t, sstep):
        out = list(range(0, max(N - t, 0) + 1, sstep))
        if out[-1] != N - t:
            out.append(N - t)
        return out
    ys = starts(H, tile, stride)
    xs = starts(W, tile, stride)

    batch_xs, batch_locs = [], []
    BATCH = 24

    def flush():
        if not batch_xs:
            return
        xb = torch.from_numpy(np.stack(batch_xs)).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seg_logits, sn, cs, _, ag = model(xb)
        probs = torch.sigmoid(seg_logits).detach().float().cpu().numpy()
        sn = sn.detach().float().cpu().numpy()
        cs = cs.detach().float().cpu().numpy()
        ag = ag.detach().float().cpu().numpy()
        for (y0, x0), p, s_, c_, a_ in zip(batch_locs, probs[:, 0],
                                            sn[:, 0], cs[:, 0], ag[:, 0]):
            prob_acc[y0:y0+tile, x0:x0+tile] += p * hann
            sin_acc[y0:y0+tile, x0:x0+tile]  += s_ * hann
            cos_acc[y0:y0+tile, x0:x0+tile]  += c_ * hann
            agg_acc[y0:y0+tile, x0:x0+tile]  += a_ * hann
            weight_acc[y0:y0+tile, x0:x0+tile] += hann
        batch_xs.clear(); batch_locs.clear()

    for y0 in ys:
        for x0 in xs:
            diffim_tile = panel_image[y0:y0 + tile, x0:x0 + tile]
            rl_tile = panel_real_labels[y0:y0 + tile, x0:x0 + tile]
            x3 = build_3channel(diffim_tile, rl_tile, panel_sigma=sigma, clip=clip)
            batch_xs.append(x3)
            batch_locs.append((y0, x0))
            if len(batch_xs) >= BATCH:
                flush()
    flush()

    wmax = np.maximum(weight_acc, 1e-6)
    return (
        (prob_acc / wmax).astype(np.float16),
        (sin_acc  / wmax).astype(np.float16),
        (cos_acc  / wmax).astype(np.float16),
        (agg_acc  / wmax).astype(np.float16),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "experiments" / "diffim_runs"))
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--splits", nargs="+",
                    default=["test_5sigma", "test_4sigma", "test_3sigma"])
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--t-low", type=float, default=0.05)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--kernel-lens", type=int, nargs="+", default=[11, 21, 41])
    ap.add_argument("--n-angles", type=int, default=12)
    ap.add_argument("--widths", type=int, nargs="+", default=[48, 96, 192, 384, 768])
    ap.add_argument("--nn-score-grid", nargs="+", type=float,
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--report-nn-score", type=float, default=0.3)
    ap.add_argument("--real-label-thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", action="store_true",
                    help="Use raw model weights instead of EMA-applied. Needed when "
                         "EMA hasn't had time to track late-emerging parameters "
                         "(e.g. agg_alpha that ramps from 0).")
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.run_name
    ckpt_path = run_dir / "ckpts" / args.checkpoint
    assert ckpt_path.exists(), f"missing checkpoint {ckpt_path}"
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_info = json.loads((run_dir / "split.json").read_text())
    train_panels = set(int(x) for x in split_info["train_panels"])
    train_csv = pd.read_csv(REPO_ROOT / "DATA_DIFFIM" / "train.csv")
    train_visits = set(train_csv[train_csv["image_id"].isin(train_panels)]["visit"].unique().tolist())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNetResSEOrientHough(in_ch=3, widths=tuple(args.widths),
                                  kernel_lens=tuple(args.kernel_lens),
                                  n_angles=args.n_angles).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt and not args.no_ema:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)
        print(f"[init] EMA applied")
    elif args.no_ema:
        print(f"[init] no-ema: using raw checkpoint weights")
    model.eval()
    print(f"[init] loaded {args.checkpoint} epoch={ckpt.get('epoch','?')} val_auc={ckpt.get('val_pixel_auc','?')} "
          f"agg_alpha={float(model.agg_alpha):.4f}")

    cfg_extract = CandidateExtractorConfig(t_low=args.t_low, min_area=args.min_area)
    score_grid = np.array(sorted(args.nn_score_grid))

    all_split_summaries: dict[str, dict] = {}
    for split in args.splits:
        h5_path, csv_path = SPLIT_PATHS[split]
        print(f"[{split}] loading {h5_path}", flush=True)
        csv = pd.read_csv(csv_path).copy()
        csv["injection_idx"] = csv.groupby("image_id").cumcount() + 1

        with h5py.File(h5_path, "r") as f:
            N, H, W = f["images"].shape
            panel_ids = list(range(N))

            t0 = time.time()
            panel_probs: dict[int, np.ndarray] = {}
            for pid in panel_ids:
                img = f["images"][pid][:]
                rl = f["real_labels"][pid][:]
                panel_probs[pid] = predict_panel_overlap_3ch(
                    model, img, rl,
                    device=device, tile=args.tile, stride=args.stride,
                )
            print(f"[{split}] sliding-window inference done in {time.time()-t0:.1f}s", flush=True)

            np.savez_compressed(out_dir / f"{split}_panel_probs.npz",
                                **{f"pid_{pid}": panel_probs[pid] for pid in panel_ids})

            cand_dfs = []
            inj_rows = []
            t1 = time.time()
            for pid in panel_ids:
                rl = f["real_labels"][pid][:]
                msk = f["masks"][pid][:]
                pp = panel_probs[pid].astype(np.float32)
                cand = extract_candidates(pp, real_labels=rl, cfg=cfg_extract, panel_id=pid)

                df_panel = csv[csv["image_id"] == pid].copy()
                df_panel["injection_idx"] = df_panel["injection_idx"].astype(np.int64)
                tid_mask = build_truth_id_mask(msk, df_panel)
                cand = match_candidates_to_injections(cand, tid_mask, pp, t_low=args.t_low)
                cand_dfs.append(cand)

                for _, r in df_panel.iterrows():
                    cm = cand[cand["matched_injection_id"] == int(r["injection_idx"])]
                    if len(cm):
                        best = cm.sort_values("max_p", ascending=False).iloc[0]
                        matched_cid = int(best["candidate_id"])
                        matched_score = float(best["max_p"])
                        matched_iou = float(best["matched_iou"])
                    else:
                        matched_cid, matched_score, matched_iou = -1, 0.0, 0.0
                    inj_rows.append({
                        "panel_id": pid,
                        "injection_idx": int(r["injection_idx"]),
                        "x": int(r["x"]), "y": int(r["y"]),
                        "visit": int(r["visit"]),
                        "detector": int(r["detector"]),
                        "physical_filter": r["physical_filter"],
                        "SNR": float(r["SNR"]),
                        "SNR_estimation": float(r.get("SNR_estimation", float("nan"))),
                        "trail_length": float(r["trail_length"]),
                        "beta": float(r["beta"]),
                        "mag": float(r["mag"]),
                        "stack_detection": bool(r["stack_detection"]),
                        "stack_snr": float(r.get("stack_snr", float("nan"))),
                        "matched_candidate_id": matched_cid,
                        "matched_score": matched_score,
                        "matched_iou": matched_iou,
                    })
            print(f"[{split}] extracted+matched in {time.time()-t1:.1f}s", flush=True)

        cand_df = pd.concat(cand_dfs, ignore_index=True) if cand_dfs else pd.DataFrame()
        inj_df = pd.DataFrame(inj_rows)

        cand_df.to_csv(out_dir / f"{split}_candidates.csv", index=False)
        inj_df.to_csv(out_dir / f"{split}_per_injection.csv", index=False)

        full = compute_metrics(inj_df, cand_df, score_grid, real_label_thresh=args.real_label_thresh)
        strict_inj = inj_df[~inj_df["visit"].isin(train_visits)].copy()
        strict_cand = cand_df.merge(
            strict_inj[["panel_id"]].drop_duplicates(),
            on="panel_id", how="inner",
        )
        strict = compute_metrics(strict_inj, strict_cand, score_grid, real_label_thresh=args.real_label_thresh)

        plot_froc(full["sweep"], out_dir / f"{split}_froc_full.png", label=f"{split} (full)")
        if len(strict_inj):
            plot_froc(strict["sweep"], out_dir / f"{split}_froc_strict.png",
                      label=f"{split} (strict visit-disjoint)")
        plot_completeness_vs(inj_df, args.report_nn_score, "SNR",
                             bins=np.linspace(2, 8, 13),
                             out_path=out_dir / f"{split}_completeness_vs_SNR.png", label=split)
        plot_completeness_vs(inj_df, args.report_nn_score, "trail_length",
                             bins=np.linspace(6, 60, 10),
                             out_path=out_dir / f"{split}_completeness_vs_length.png", label=split)
        plot_completeness_vs(inj_df, args.report_nn_score, "beta",
                             bins=np.linspace(0, 180, 10),
                             out_path=out_dir / f"{split}_completeness_vs_beta.png", label=split)
        save_visual_panels(run_dir, split, h5_path,
                           {pid: panel_probs[pid] for pid in panel_ids},
                           cand_df, inj_df, nn_score_threshold=args.report_nn_score)

        all_split_summaries[split] = {
            "full": full,
            "strict_visit_disjoint": strict,
            "n_train_visits": len(train_visits),
            "n_strict_injections": int(len(strict_inj)),
        }

    (out_dir / "summary.json").write_text(json.dumps(all_split_summaries, indent=2))
    print(f"DONE. summary written to {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
