from __future__ import annotations

from ADCNN.utils.utils import draw_one_line
import ADCNN.evals.eval_utils as eval_utils
import os, gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures as cf
import multiprocessing as mp


# -------------------------
# Worker globals (avoid pickling predictions/groups per task)
# -------------------------
_G_PRED = None
_G_GROUPS = None
_G_PIXEL_GAP = None
_G_PSF_WIDTH = None
_G_STACK_FP = None


# -------------------------
# Helpers
# -------------------------
def _prepare_catalog_groups_np(catalog: pd.DataFrame) -> dict[int, np.ndarray]:
    need = ["image_id", "x", "y", "beta", "trail_length"]
    missing = [c for c in need if c not in catalog.columns]
    if missing:
        raise ValueError(f"catalog is missing columns: {missing}")

    groups: dict[int, np.ndarray] = {}
    for img_id, df in catalog.groupby("image_id", sort=False):
        groups[int(img_id)] = df[["x", "y", "beta", "trail_length"]].to_numpy()
    return groups

def _init_worker(
    predictions: np.ndarray,
    groups: dict[int, np.ndarray],
    pixel_gap: int,
    psf_width: int,
    stack_fp: np.ndarray | None,
):
    """
    stack_fp:
      - None => no special handling (current behavior)
      - array of shape (N,H,W), truthy where STACK says "this is a false positive region"
        Any NN component overlapping stack_fp is NOT counted as FP (ignored).
    """
    global _G_PRED, _G_GROUPS, _G_PIXEL_GAP, _G_PSF_WIDTH, _G_STACK_FP
    _G_PRED = predictions
    _G_GROUPS = groups
    _G_PIXEL_GAP = int(pixel_gap)
    _G_PSF_WIDTH = int(psf_width)
    _G_STACK_FP = stack_fp  # may be None


def _objectwise_one_image(args):
    """
    args = (img_id, thr_index, thr_value)
    Returns (thr_index, tp, fp, fn)

    FP counting rule:
      - Compute connected components on (pred >= thr)
      - Match GT objects to components (one-to-one) via removed_labels LUT
      - Remaining components are candidate FP components
      - If stack_fp is provided:
          any remaining component that overlaps stack_fp[img_id] is NOT counted as FP
    """
    img_id, thr_i, thr = args
    pred2d = _G_PRED[img_id]
    rows_np = _G_GROUPS[img_id]
    pixel_gap = _G_PIXEL_GAP
    psf_width = _G_PSF_WIDTH

    pred_bin = (pred2d >= thr) if pred2d.dtype != np.bool_ else pred2d.copy()
    lab, n = eval_utils._label_components_fds(pred_bin, pixel_gap=pixel_gap)
    predicted_positive = int(n)

    removed_labels = np.zeros(predicted_positive + 1, dtype=bool)

    tp_img = 0
    fn_img = 0
    H, W = pred_bin.shape
    half = int(psf_width / 2)

    # --- match GT objects ---
    for x, y, beta, trail_length in rows_np:
        pad = half + 4

        dx = abs(np.cos(beta)) * trail_length
        dy = abs(np.sin(beta)) * trail_length

        x0 = int(max(0, np.floor(x - dx - pad)))
        x1 = int(min(W, np.ceil(x + dx + pad)))
        y0 = int(max(0, np.floor(y - dy - pad)))
        y1 = int(min(H, np.ceil(y + dy + pad)))

        if x1 <= x0 or y1 <= y0:
            fn_img += 1
            continue

        roi_h = y1 - y0
        roi_w = x1 - x0

        mask_roi = draw_one_line(
            np.zeros((roi_h, roi_w), dtype=np.uint8),
            [x - x0, y - y0],
            beta,
            trail_length,
            true_value=1,
            line_thickness=half,
        ).astype(bool)

        rr, cc = np.nonzero(mask_roi)
        if rr.size == 0:
            fn_img += 1
            continue

        lab_vals = lab[rr + y0, cc + x0]
        lab_vals = lab_vals[(lab_vals != 0) & (~removed_labels[lab_vals])]

        if lab_vals.size:
            tp_img += 1
            removed_labels[np.unique(lab_vals)] = True
        else:
            fn_img += 1

    # --- FP counting (optionally ignore those overlapping stack_fp) ---
    if predicted_positive <= 0:
        fp_img = 0
    elif _G_STACK_FP is None:
        fp_img = max(predicted_positive - tp_img, 0)
    else:
        stack_mask = _G_STACK_FP[img_id]
        # accept bool/uint8/etc.
        stack_mask = (stack_mask != 0)
        # labels that overlap stack FP regions
        overlap_labels = lab[stack_mask]
        if overlap_labels.size == 0:
            ignored_remaining = 0
        else:
            overlap_labels = np.unique(overlap_labels)
            overlap_labels = overlap_labels[overlap_labels != 0]
            if overlap_labels.size == 0:
                ignored_remaining = 0
            else:
                # ignore only if the label is NOT already removed by GT matching
                ignored_remaining = int(np.sum(~removed_labels[overlap_labels]))

        fp_img = predicted_positive - tp_img - ignored_remaining
        if fp_img < 0:
            fp_img = 0

    return int(thr_i), int(tp_img), int(fp_img), int(fn_img)


# -------------------------
# Pixelwise: one-pass histogram scan returning ARRAYS aligned with thresholds
# -------------------------
def pixelwise_scan(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: np.ndarray,
    nbins: int = 4096,
    chunk_pixels: int = 50_000_000,
):
    pred = np.asarray(predictions, dtype=np.float32)
    gt = np.asarray(ground_truth)

    if pred.shape != gt.shape:
        raise ValueError(f"predictions.shape {pred.shape} != ground_truth.shape {gt.shape}")

    pred_flat = pred.reshape(-1)
    gt_pos_flat = (gt.reshape(-1) != 0)

    pred_flat = np.clip(pred_flat, 0.0, 1.0)

    hist_pos = np.zeros(nbins, dtype=np.int64)
    hist_neg = np.zeros(nbins, dtype=np.int64)

    n = pred_flat.size
    for start in range(0, n, chunk_pixels):
        end = min(n, start + chunk_pixels)
        p = pred_flat[start:end]
        gpos = gt_pos_flat[start:end]
        b = (p * (nbins - 1)).astype(np.int32)

        if gpos.any():
            hist_pos += np.bincount(b[gpos], minlength=nbins)
        if (~gpos).any():
            hist_neg += np.bincount(b[~gpos], minlength=nbins)

    P = int(hist_pos.sum())
    N = int(hist_neg.sum())

    tp_cum = np.cumsum(hist_pos[::-1])[::-1]
    fp_cum = np.cumsum(hist_neg[::-1])[::-1]

    thr = np.asarray(thresholds, dtype=np.float32)
    thr = np.clip(thr, 0.0, 1.0)
    thr_bin = (thr * (nbins - 1)).astype(np.int32)

    m = thr.shape[0]
    pix_tp = np.empty(m, dtype=np.int64)
    pix_fp = np.empty(m, dtype=np.int64)
    pix_fn = np.empty(m, dtype=np.int64)
    pix_tn = np.empty(m, dtype=np.int64)

    for i, b in enumerate(thr_bin.tolist()):
        tp = int(tp_cum[b])
        fp = int(fp_cum[b])
        pix_tp[i] = tp
        pix_fp[i] = fp
        pix_fn[i] = P - tp
        pix_tn[i] = N - fp

    return {
        "pix_tp": pix_tp, "pix_fp": pix_fp, "pix_fn": pix_fn, "pix_tn": pix_tn
    }


def scan_thresholds(
    catalog: pd.DataFrame,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    *,
    stack_fp: np.ndarray | None = None,
    thr_min: float = 0.0,
    thr_max: float = 1.0,
    n_points: int = 51,
    pixel_gap: int = 2,
    psf_width: int = 40,
    obj_max_workers: int | None = None,
    inflight_thresholds: int = 8,
    gc_every: int = 0,
    force_fork: bool = True,
    nbins: int = 4096,
    chunk_pixels: int = 50_000_000,
) -> pd.DataFrame:
    """
    Returns df with raw counts:
      - pix_tp, pix_fp, pix_fn, pix_tn
      - obj_tp, obj_fp, obj_fn

    If stack_fp is provided:
      - stack_fp shape must be (N,H,W) matching predictions/ground_truth
      - any remaining NN component overlapping stack_fp is ignored (not counted as FP)
    """
    if obj_max_workers is None:
        obj_max_workers = max(1, (os.cpu_count() or 2) - 1)

    thresholds = np.linspace(thr_min, thr_max, n_points, dtype=np.float64)

    groups = _prepare_catalog_groups_np(catalog)
    image_ids = [img_id for img_id in groups.keys() if 0 <= img_id < predictions.shape[0]]
    n_images = len(image_ids)
    if n_images == 0:
        raise ValueError("No valid image_ids in catalog within predictions.shape[0].")

    if stack_fp is not None:
        stack_fp = np.asarray(stack_fp)
        if stack_fp.shape != predictions.shape:
            raise ValueError(f"stack_fp.shape {stack_fp.shape} must match predictions.shape {predictions.shape}")

    # ---- pixelwise in ONE pass ----
    pixA = pixelwise_scan(
        predictions=predictions,
        ground_truth=ground_truth,
        thresholds=thresholds,
        nbins=nbins,
        chunk_pixels=chunk_pixels,
    )

    # ---- objectwise pipelined ----
    acc_tp = np.zeros(n_points, dtype=np.int64)
    acc_fp = np.zeros(n_points, dtype=np.int64)
    acc_fn = np.zeros(n_points, dtype=np.int64)
    acc_done = np.zeros(n_points, dtype=np.int32)

    thr_iter = iter(range(n_points))
    future_to_thr_i: dict[cf.Future, int] = {}

    def submit_threshold_index(thr_i: int, pool: cf.ProcessPoolExecutor):
        thr = float(thresholds[thr_i])
        for img_id in image_ids:
            fut = pool.submit(_objectwise_one_image, (img_id, thr_i, thr))
            future_to_thr_i[fut] = thr_i

    mp_context = mp.get_context("fork") if force_fork else None

    completed = 0
    with cf.ProcessPoolExecutor(
        max_workers=obj_max_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(predictions, groups, pixel_gap, psf_width, stack_fp),
    ) as pool:
        for _ in range(min(inflight_thresholds, n_points)):
            try:
                submit_threshold_index(next(thr_iter), pool)
            except StopIteration:
                break

        while future_to_thr_i:
            done_set, _ = cf.wait(future_to_thr_i, return_when=cf.FIRST_COMPLETED)
            for fut in done_set:
                thr_i = future_to_thr_i.pop(fut)
                thr_i2, tpi, fpi, fni = fut.result()
                if thr_i2 != thr_i:
                    thr_i = thr_i2

                acc_tp[thr_i] += tpi
                acc_fp[thr_i] += fpi
                acc_fn[thr_i] += fni
                acc_done[thr_i] += 1

                completed += 1
                if gc_every and (completed % gc_every == 0):
                    gc.collect()

                if acc_done[thr_i] == n_images:
                    try:
                        submit_threshold_index(next(thr_iter), pool)
                    except StopIteration:
                        pass

    df = pd.DataFrame({
        "thr": thresholds.astype(float),
        "pix_tp": pixA["pix_tp"],
        "pix_fp": pixA["pix_fp"],
        "pix_fn": pixA["pix_fn"],
        "pix_tn": pixA["pix_tn"],
        "obj_tp": acc_tp,
        "obj_fp": acc_fp,
        "obj_fn": acc_fn,
    }).sort_values("thr").reset_index(drop=True)

    df["obj_precision"] = eval_utils.precision(df["obj_tp"], df["obj_fp"])
    df["obj_recall"] = eval_utils.recall(df["obj_tp"], df["obj_fn"])
    df["obj_f1"] = eval_utils.f1(df["obj_tp"], df["obj_fp"], df["obj_fn"])
    df["obj_f2"] = eval_utils.f2(df["obj_tp"], df["obj_fp"], df["obj_fn"])

    df["pix_precision"] = eval_utils.precision(df["pix_tp"], df["pix_fp"])
    df["pix_recall"] = eval_utils.recall(df["pix_tp"], df["pix_fn"])
    df["pix_f1"] = eval_utils.f1(df["pix_tp"], df["pix_fp"], df["pix_fn"])
    df["pix_f2"] = eval_utils.f2(df["pix_tp"], df["pix_fp"], df["pix_fn"])

    return df

def compute_froc(
    thr: np.ndarray,
    obj_tp: np.ndarray,
    obj_fp: np.ndarray,
    obj_fn: np.ndarray,
    *,
    n_images: int,
    fp_denom: str = "all",            # "all" or "neg"
    n_neg_images: int | None = None,  # required if fp_denom="neg"
    make_monotonic: bool = True,
    sort_by: str = "fppi",            # "fppi" (recommended) or "thr"
):
    """
    Compute FROC arrays from threshold-scan counts.

    Returns a dict with:
      thr, fppi, recall, recall_plot (monotone if requested), denom_images
    Arrays are sorted by `sort_by`.
    """
    thr = np.asarray(thr, dtype=np.float64)
    tp = np.asarray(obj_tp, dtype=np.float64)
    fp = np.asarray(obj_fp, dtype=np.float64)
    fn = np.asarray(obj_fn, dtype=np.float64)

    if not (thr.shape == tp.shape == fp.shape == fn.shape):
        raise ValueError("thr, obj_tp, obj_fp, obj_fn must have the same shape")

    if fp_denom not in ("all", "neg"):
        raise ValueError("fp_denom must be 'all' or 'neg'")

    denom_images = n_images if fp_denom == "all" else n_neg_images
    if denom_images is None or int(denom_images) <= 0:
        raise ValueError("Provide n_neg_images > 0 when fp_denom='neg'")

    fppi = eval_utils.fppi(fp, int(denom_images))
    recall = eval_utils.recall(tp, fn)

    if sort_by == "fppi":
        order = np.argsort(fppi, kind="mergesort")
    elif sort_by == "thr":
        order = np.argsort(thr, kind="mergesort")
    else:
        raise ValueError("sort_by must be 'fppi' or 'thr'")

    thr_s = thr[order]
    fppi_s = fppi[order]
    recall_s = recall[order]

    if make_monotonic and sort_by != "fppi":
        # monotonicity is defined w.r.t. FPPI; enforce in FPPI order
        order2 = np.argsort(fppi_s, kind="mergesort")
        thr_s = thr_s[order2]
        fppi_s = fppi_s[order2]
        recall_s = recall_s[order2]

    recall_plot = eval_utils.make_monotone_increasing(recall_s) if make_monotonic else recall_s

    return {
        "thr": thr_s,
        "fppi": fppi_s,
        "recall": recall_s,
        "recall_plot": recall_plot,
        "denom_images": int(denom_images),
        "make_monotonic": bool(make_monotonic),
    }

def compute_roc(tp, fp, tn, fn, *, sort_by="fpr"):
    """
    Build ROC arrays from per-threshold confusion counts.

    Inputs:
      tp, fp, tn, fn: 1D arrays (same length), per threshold.
      sort_by: "fpr" (recommended) or "thr" (keeps given order)

    Returns dict with:
      fpr, tpr, thresholds_index (ordering used), tp, fp, tn, fn
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    tn = np.asarray(tn, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    if not (tp.shape == fp.shape == tn.shape == fn.shape):
        raise ValueError("tp, fp, tn, fn must have the same shape")

    denom_tpr = tp + fn
    denom_fpr = fp + tn

    tpr = np.divide(tp, denom_tpr, out=np.zeros_like(tp), where=denom_tpr > 0)
    fpr = np.divide(fp, denom_fpr, out=np.zeros_like(fp), where=denom_fpr > 0)

    idx = np.arange(tp.size)
    if sort_by == "fpr":
        # stable sort to keep deterministic ordering in ties
        order = np.argsort(fpr, kind="mergesort")
        fpr = fpr[order]
        tpr = tpr[order]
        tp = tp[order]; fp = fp[order]; tn = tn[order]; fn = fn[order]
        idx = idx[order]
    elif sort_by == "thr":
        pass
    else:
        raise ValueError("sort_by must be 'fpr' or 'thr'")

    return {
        "fpr": fpr,
        "tpr": tpr,
        "order_idx": idx,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

def roc_auc(roc, *, x_max=None, normalize=False, ensure_anchors=True):
    """
    Compute AUC under ROC curve.

    Params:
      roc: output dict from roc_from_counts (must contain fpr, tpr)
      x_max: integrate only over fpr in [0, x_max]. If None uses max fpr.
      normalize: if True, returns AUC / x_max (range [0,1] if ensure_anchors and x_max=1)
      ensure_anchors: if True, ensure curve contains (0,0) and (x_max, tpr_at_xmax).

    Returns:
      auc (float)
    """
    x = np.asarray(roc["fpr"], dtype=np.float64)
    y = np.asarray(roc["tpr"], dtype=np.float64)

    if x.size == 0:
        return 0.0

    if x_max is None:
        x_max = float(x.max()) if x.size else 0.0
    x_max = float(x_max)
    if x_max <= 0:
        return 0.0

    # keep points <= x_max
    m = x <= x_max
    x2 = x[m]
    y2 = y[m]

    # helper: interpolate y at x_target
    def _interp_at(x_target):
        j = np.searchsorted(x, x_target, side="left")
        if j <= 0:
            return float(y[0])
        if j >= len(x):
            return float(y[-1])
        x0, x1 = float(x[j - 1]), float(x[j])
        y0, y1 = float(y[j - 1]), float(y[j])
        if x1 <= x0:
            return y0
        t = (x_target - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    if ensure_anchors:
        # anchor at x=0
        if x2.size == 0:
            # no points <= x_max; build minimal segment
            y_at = _interp_at(x_max)
            x2 = np.array([0.0, x_max], dtype=np.float64)
            y2 = np.array([0.0, y_at], dtype=np.float64)
        else:
            if x2[0] > 0.0:
                x2 = np.insert(x2, 0, 0.0)
                y2 = np.insert(y2, 0, 0.0)
            if x2[-1] < x_max:
                y_at = _interp_at(x_max)
                x2 = np.append(x2, x_max)
                y2 = np.append(y2, y_at)

    area = float(np.trapz(y2, x2))
    if normalize:
        area /= x_max
    return area

def plot_froc(
    froc: dict,
    *,
    x_max: float | None = None,
    title: str = "FROC",
):
    x = np.asarray(froc["fppi"], dtype=np.float64)
    y = np.asarray(froc["recall_plot"], dtype=np.float64)
    denom = froc.get("denom_images", None)

    if x.size == 0:
        raise ValueError("Empty FROC arrays")

    x_max_eff = float(np.nanmax(x)) if x_max is None else float(x_max)

    plt.figure()
    plt.plot(x, y)
    if x_max is not None:
        plt.xlim(0, x_max_eff)
    plt.ylim(0, 1.0)
    plt.xlabel(f"FP per image")
    plt.ylabel("Purity")
    plt.title(f"{title}")
    plt.show()

def plot_roc(roc, *, auc=None, x_max=1.0, title="ROC (pixelwise)", show_diagonal=True):
    """
    Plot ROC curve (fpr vs tpr).
    """
    x = np.asarray(roc["fpr"], dtype=np.float64)
    y = np.asarray(roc["tpr"], dtype=np.float64)

    if auc is None:
        auc = roc_auc(roc, x_max=x_max, normalize=False)

    plt.figure()
    plt.plot(x, y, label=f"AUC={auc:.4f}")
    if show_diagonal:
        plt.plot([0, x_max], [0, min(1.0, x_max)], linestyle="--", label="random")

    plt.xlim(0.0, float(x_max))
    plt.ylim(0.0, 1.0)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    #plt.grid(True)
    plt.legend()
    plt.show()

def plot_fscore(
    thr,
    *,
    pix_f1=None,
    pix_f2=None,
    obj_f1=None,
    obj_f2=None,
    title="F scores vs threshold",
):
    thr = np.asarray(thr, dtype=float)

    plt.figure()

    def _plot(y, label):
        y = np.asarray(y, dtype=float)
        if y.shape != thr.shape:
            raise ValueError(f"{label} has shape {y.shape}, expected {thr.shape}")
        plt.plot(thr, y, label=label)

    if pix_f1 is not None:
        _plot(pix_f1, "pixelwise F1")
    if pix_f2 is not None:
        _plot(pix_f2, "pixelwise F2")
    if obj_f1 is not None:
        _plot(obj_f1, "objectwise F1")
    if obj_f2 is not None:
        _plot(obj_f2, "objectwise F2")

    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.title(title)
    #plt.grid(True)
    plt.legend()
    plt.show()