import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures as cf
from ADCNN.utils.utils import draw_one_line
import ADCNN.evals.eval_utils as eval_utils

def _prepare_catalog_groups_np(catalog: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
      groups[img_id] = (rows_np, row_index_np)

    rows_np: float array with columns [x, y, beta, trail_length]
    row_index_np: original dataframe index for writing nn_detected back
    """
    need = ["image_id", "x", "y", "beta", "trail_length"]
    missing = [c for c in need if c not in catalog.columns]
    if missing:
        raise ValueError(f"catalog is missing columns: {missing}")

    groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for img_id, df in catalog.groupby("image_id", sort=False):
        img_id = int(img_id)
        rows_np = df[["x", "y", "beta", "trail_length"]].to_numpy()
        idx_np = df.index.to_numpy()
        groups[img_id] = (rows_np, idx_np)
    return groups


def _objectwise_and_mark_one_image_local(args):
    """
    args = (img_id, rows_np, idx_np, pred2d, thr, pixel_gap, psf_width, stack_fp2d)

    Returns:
      (tp_img, fp_img, fn_img, idx_np, nn_detected_bool_array)
    """
    (img_id, rows_np, idx_np, pred2d, thr, pixel_gap, psf_width, stack_fp2d) = args

    pred_bin = (pred2d >= thr) if pred2d.dtype != np.bool_ else pred2d.copy()

    lab, n = eval_utils._label_components_fds(pred_bin, pixel_gap=pixel_gap)
    predicted_positive = int(n)

    removed_labels = np.zeros(predicted_positive + 1, dtype=bool)

    tp_img = 0
    fn_img = 0

    H, W = pred_bin.shape
    half = int(psf_width / 2)

    nn_det = np.zeros(rows_np.shape[0], dtype=bool)

    for j, (x, y, beta, trail_length) in enumerate(rows_np):
        pad = half + 4

        dx = abs(np.cos(beta)) * trail_length
        dy = abs(np.sin(beta)) * trail_length

        x0 = int(max(0, np.floor(x - dx - pad)))
        x1 = int(min(W, np.ceil (x + dx + pad)))
        y0 = int(max(0, np.floor(y - dy - pad)))
        y1 = int(min(H, np.ceil (y + dy + pad)))

        if x1 <= x0 or y1 <= y0:
            fn_img += 1
            nn_det[j] = False
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
            nn_det[j] = False
            continue

        lab_vals_all = lab[rr + y0, cc + x0]

        # per-object marking: any overlap with any component
        nn_det[j] = np.any(lab_vals_all != 0)

        # TP/FN matching with LUT removal
        lab_vals = lab_vals_all[(lab_vals_all != 0) & (~removed_labels[lab_vals_all])]
        if lab_vals.size:
            tp_img += 1
            removed_labels[np.unique(lab_vals)] = True
        else:
            fn_img += 1

    # --- FP counting ---
    if stack_fp2d is None:
        fp_img = max(predicted_positive - tp_img, 0)
    else:
        stack_mask = (stack_fp2d != 0)
        if np.any(stack_mask):
            stack_labels = np.unique(lab[stack_mask])
            stack_labels = stack_labels[stack_labels != 0]
            ignored = np.zeros(predicted_positive + 1, dtype=bool)
            ignored[stack_labels] = True
        else:
            ignored = np.zeros(predicted_positive + 1, dtype=bool)

        fp_img = int(np.sum((~removed_labels[1:]) & (~ignored[1:])))

    return int(tp_img), int(fp_img), int(fn_img), idx_np, nn_det


def objectwise_confusion_and_mark(
    catalog: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float,
    *,
    pixel_gap: int = 2,
    psf_width: int = 40,
    max_workers: int | None = None,
    use_threads: bool = False,
    stack_fp: np.ndarray | None = None,  # shape (N,H,W) or None
):
    """
    One pass per image (one FDS per image):
      - computes (obj_tp, obj_fp, obj_fn)
      - returns catalog copy with nn_detected filled

    stack_fp (optional):
      - predicted components overlapping stack_fp[img_id] are ignored (not counted as FP)
      - if stack_fp is None: behaves like before

    Notes:
      - No fork/globals initializer. For N=50 images this is fine.
      - If you see pickling overhead or RAM spikes, switch back to initializer+fork.
    """
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    cat = catalog.copy()
    cat["nn_detected"] = False

    groups = _prepare_catalog_groups_np(cat)
    image_ids = [img_id for img_id in groups.keys() if 0 <= img_id < predictions.shape[0]]
    if not image_ids:
        return 0, 0, 0, cat

    thr = float(threshold)

    tasks = []
    for img_id in image_ids:
        rows_np, idx_np = groups[img_id]
        pred2d = predictions[img_id]
        stack_fp2d = None if stack_fp is None else stack_fp[img_id]
        tasks.append((img_id, rows_np, idx_np, pred2d, thr, pixel_gap, psf_width, stack_fp2d))

    ex = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor

    tp = fp = fn = 0
    with ex(max_workers=max_workers) as pool:
        for tpi, fpi, fni, idxs, det in pool.map(_objectwise_and_mark_one_image_local, tasks, chunksize=1):
            tp += tpi
            fp += fpi
            fn += fni
            cat.loc[idxs, "nn_detected"] = det

    return int(tp), int(fp), int(fn), cat

def print_confusion_matrix(cm, title="Confusion Matrix"):
    cm = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"], dtype="Int64")
    print(title)
    print (f"F1 Score: {cm[1,1]*2/(cm[1,1]*2 + cm[0,1] + cm[1,0]):.4f}, F2 Score: {cm[1,1]*5/(cm[1,1]*5 + cm[0,1] + cm[1,0]*4):.4f}")
    print(df)
    print()

def pixelwise_confusion(p_full, gt_full, thr):
    gt = (gt_full > 0).astype(np.uint8); pred = (p_full >= thr).astype(np.uint8)
    tp = int(((pred==1)&(gt==1)).sum()); fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum()); tn = int(((pred==0)&(gt==0)).sum())
    return int(tp),int(fp),int(fn),int(tn)

def confusion_matrix(catalog: pd.DataFrame,
                     ground_truth: np.ndarray,
                     predictions: np.ndarray,
                     threshold: float,
                     stack_fp: np.ndarray | None = None,
                     verbose: bool = False):
    obj_tp, obj_fp, obj_fn, cat = objectwise_confusion_and_mark(catalog=catalog,
                                                                predictions=predictions,
                                                                stack_fp=stack_fp,
                                                                threshold=threshold)
    pix_tp, pix_fp, pix_fn, pix_tn = pixelwise_confusion(predictions,
                                                         ground_truth,
                                                         threshold)
    if verbose:
        print_confusion_matrix({"TP": pix_tp, "FP": pix_fp, "FN": pix_fn, "TN": pix_tn},
                               title="Pixel-wise Confusion Matrix")
        print_confusion_matrix({"TP" : obj_tp, "FP": obj_fp, "FN": obj_fn, "TN": pd.NA},
                               title="Object-wise Confusion Matrix")
    return (obj_tp, obj_fp, obj_fn), (pix_tp, pix_fp, pix_fn, pix_tn), cat

def plot_detect_hist(cat, field, bins=12, title=None, density=False, xlim=None):
    nn_det = cat[cat["nn_detected"]]
    stk_det = cat[cat["stack_detection"]]
    cum_det = cat[cat["nn_detected"] | cat["stack_detection"]]
    vals = cat[field].to_numpy(); vals = vals[np.isfinite(vals)]
    if xlim is not None:
        vals = vals[(vals>=xlim[0]) & (vals<=xlim[1])]
    edges = np.histogram_bin_edges(vals, bins=bins)
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    if not density:
        ax.hist(cat[field],     bins=edges, histtype="step", label="All injected", alpha=0.7)
        ax.hist(cum_det[field], bins=edges, histtype="step", label="Cumulative (NN ∪ LSST)")
        ax.hist(nn_det[field],  bins=edges, histtype="step", label="NN detected")
        ax.hist(stk_det[field], bins=edges, histtype="step", label="LSST stack detected")
        ax.set_xlabel(field.replace("_"," ")); ax.set_ylabel("Count")
    else:
        all_i = np.histogram(cat[field],     bins=edges)[0]
        cum_i = np.histogram(cum_det[field], bins=edges)[0]
        nn_i  = np.histogram(nn_det[field],  bins=edges)[0]
        stk_i = np.histogram(stk_det[field], bins=edges)[0]
        bin_widths = np.diff(edges)
        ax.stairs(cum_i/all_i, edges, label="Cumulative (NN ∪ LSST)")
        ax.stairs(nn_i/all_i,  edges, label="NN detected")
        ax.stairs(stk_i/all_i, edges, label="LSST stack detected")
        ax.set_xlabel(field.replace("_"," ")); ax.set_ylabel("Completeness")
    if title: ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    return fig, ax