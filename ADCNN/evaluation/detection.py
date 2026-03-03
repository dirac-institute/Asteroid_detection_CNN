"""
Object-level detection evaluation.

Provides functions for:
- Object-wise confusion matrices (TP/FP/FN at detection level)
- Pixel-wise confusion matrices
- Detection marking in catalogs
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import concurrent.futures as cf
from scipy import ndimage as ndi

from ADCNN.utils.angle_utils import deg2rad
from ADCNN.evaluation.geometry import label_components

try:
    from ADCNN.utils.helpers import draw_one_line
except ImportError:
    # Fallback if draw_one_line not available
    def draw_one_line(mask, origin, angle_deg, length, true_value=1, line_thickness=3):
        raise NotImplementedError("draw_one_line requires cv2")



def robust_stats_mad(arr):
    med = np.median(arr); mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * (mad + 1e-12)
    return np.float32(med), np.float32(sigma)

def _disk_mask(shape, yc, xc, r):
    H,W = shape; y,x = np.ogrid[:H,:W]
    return (y-yc)**2 + (x-xc)**2 <= r*r

def _line_mask(shape, yc, xc, beta_deg, length, width=1):
    H,W = shape; mask = np.zeros((H,W), bool)
    L = float(length)/2.0; th = np.deg2rad(float(beta_deg))
    dy,dx = np.sin(th), np.cos(th)
    y0,x0 = float(yc)-L*dy, float(xc)-L*dx
    y1,x1 = float(yc)+L*dy, float(xc)+L*dx
    steps = max(2, int(np.ceil(length*2)))
    ys = np.clip(np.rint(np.linspace(y0,y1,steps)).astype(int), 0, H-1)
    xs = np.clip(np.rint(np.linspace(x0,x1,steps)).astype(int), 0, W-1)
    mask[ys,xs] = True
    if width>1:
        rad = max(1,int(width//2))
        mask = ndi.binary_dilation(mask, structure=np.ones((2*rad+1,2*rad+1), bool))
    return mask

def _label_components_fds(mask_bool, pixel_gap=3):
    if pixel_gap>1:
        grown = ndi.binary_dilation(mask_bool, structure=np.ones((2*pixel_gap+1,2*pixel_gap+1), bool))
    else:
        grown = mask_bool
    labels, n = ndi.label(grown, structure=np.ones((3,3), bool))  # 8-connectivity
    return labels, int(n)

def mark_nn_and_stack_fds(csv_path, p_full, thr=0.5, pixel_gap=3, line_width=1):
    cat = pd.read_csv(csv_path).copy()
    need = {"image_id","x","y"}
    miss = need - set(cat.columns)
    if miss: raise ValueError(f"CSV missing columns: {miss}")
    if "stack_detection" in cat.columns: cat["stack_detected"] = cat["stack_detection"].astype(bool)
    elif "stack_mag" in cat.columns:    cat["stack_detected"] = ~cat["stack_mag"].isna()
    else:                               cat["stack_detected"] = False
    H,W = p_full.shape[1:]
    pred_bin = (p_full >= thr)
    labels_list = []
    for i in range(p_full.shape[0]):
        labels_list.append(_label_components_fds(pred_bin[i], pixel_gap=pixel_gap)[0])
    nn = np.zeros(len(cat), bool)
    has_beta = "beta" in cat.columns; has_len = "trail_length" in cat.columns
    for pid, grp in cat.groupby("image_id"):
        pid = int(pid)
        if pid<0 or pid>=len(labels_list): continue
        lab = labels_list[pid]
        for idx in grp.index:
            xc = int(np.clip(int(cat.at[idx,"x"]), 0, W-1))
            yc = int(np.clip(int(cat.at[idx,"y"]), 0, H-1))
            if has_beta and has_len and np.isfinite(cat.at[idx,"beta"]) and np.isfinite(cat.at[idx,"trail_length"]):
                m = _line_mask((H,W), yc, xc, beta_deg=float(cat.at[idx,"beta"]),
                               length=float(cat.at[idx,"trail_length"]), width=line_width)
            else:
                m = _disk_mask((H,W), yc, xc, r=max(1,pixel_gap))
            nn[idx] = (lab[m] > 0).any()
    cat["nn_detected"] = nn
    return cat

# =============================================================================
# Object-wise Evaluation
# =============================================================================

def _prepare_catalog_groups(catalog: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Group catalog by image_id for efficient processing.

    Returns:
        dict mapping image_id -> (rows_array, index_array)
        rows_array: [x, y, beta, trail_length] for each object
        index_array: original dataframe indices
    """
    required = ["image_id", "x", "y", "beta", "trail_length"]
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")

    groups = {}
    for img_id, df in catalog.groupby("image_id", sort=False):
        rows_np = df[["x", "y", "beta", "trail_length"]].to_numpy()
        idx_np = df.index.to_numpy()
        groups[int(img_id)] = (rows_np, idx_np)

    return groups


def _evaluate_single_image(args):
    """
    Evaluate single image for object-level TP/FP/FN.

    Args tuple:
        (img_id, rows_np, idx_np, pred2d, thr, pixel_gap, psf_width, stack_fp2d)

    Returns:
        (tp, fp, fn, idx_np, nn_detected_array)
    """
    (img_id, rows_np, idx_np, pred2d, thr, pixel_gap, psf_width, stack_fp2d) = args

    # Threshold predictions
    pred_bin = (pred2d >= thr) if pred2d.dtype != np.bool_ else pred2d.copy()

    # Label connected components
    labels, n_components = label_components(pred_bin, pixel_gap=pixel_gap)

    # Track which components have been matched to GT objects
    matched_labels = np.zeros(n_components + 1, dtype=bool)

    tp = 0
    fn = 0
    H, W = pred_bin.shape
    half_psf = int(psf_width / 2)

    # Track detections per object
    nn_detected = np.zeros(rows_np.shape[0], dtype=bool)

    # Evaluate each GT object
    for j, (x, y, beta_deg, trail_length) in enumerate(rows_np):
        pad = half_psf + 4

        # Compute bounding box (beta in degrees, convert to radians)
        beta_rad = deg2rad(beta_deg)
        dx = abs(np.cos(beta_rad)) * trail_length
        dy = abs(np.sin(beta_rad)) * trail_length

        x0 = int(max(0, np.floor(x - dx - pad)))
        x1 = int(min(W, np.ceil(x + dx + pad)))
        y0 = int(max(0, np.floor(y - dy - pad)))
        y1 = int(min(H, np.ceil(y + dy + pad)))

        if x1 <= x0 or y1 <= y0:
            fn += 1
            nn_detected[j] = False
            continue

        # Create trail mask in ROI
        roi_h = y1 - y0
        roi_w = x1 - x0

        try:
            mask_roi = draw_one_line(
                np.zeros((roi_h, roi_w), dtype=np.uint8),
                (x - x0, y - y0),
                beta_deg,
                trail_length,
                true_value=1,
                line_thickness=half_psf,
            ).astype(bool)
        except:
            # Fallback to simple mask if draw_one_line fails
            mask_roi = np.zeros((roi_h, roi_w), dtype=bool)
            cy, cx = int(y - y0), int(x - x0)
            if 0 <= cy < roi_h and 0 <= cx < roi_w:
                mask_roi[cy, cx] = True

        rr, cc = np.nonzero(mask_roi)
        if rr.size == 0:
            fn += 1
            nn_detected[j] = False
            continue

        # Check overlap with predicted components
        label_vals_all = labels[rr + y0, cc + x0]

        # Mark as detected if any overlap
        nn_detected[j] = np.any(label_vals_all != 0)

        # Match to first unmatched component (TP/FN)
        label_vals = label_vals_all[(label_vals_all != 0) & (~matched_labels[label_vals_all])]

        if label_vals.size > 0:
            tp += 1
            matched_labels[np.unique(label_vals)] = True
        else:
            fn += 1

    # Count false positives (unmatched components)
    if stack_fp2d is None:
        fp = max(n_components - tp, 0)
    else:
        # Ignore components that overlap with stack false positives
        stack_mask = (stack_fp2d != 0)
        if np.any(stack_mask):
            stack_labels = np.unique(labels[stack_mask])
            stack_labels = stack_labels[stack_labels != 0]
            ignored = np.zeros(n_components + 1, dtype=bool)
            ignored[stack_labels] = True
        else:
            ignored = np.zeros(n_components + 1, dtype=bool)

        fp = int(np.sum((~matched_labels[1:]) & (~ignored[1:])))

    return int(tp), int(fp), int(fn), idx_np, nn_detected


def objectwise_confusion(
    catalog: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float,
    *,
    pixel_gap: int = 2,
    psf_width: int = 40,
    max_workers: Optional[int] = None,
    use_threads: bool = False,
    stack_fp: Optional[np.ndarray] = None,
) -> tuple[int, int, int, pd.DataFrame]:
    """
    Compute object-level confusion matrix.

    Args:
        catalog: DataFrame with columns [image_id, x, y, beta, trail_length]
        predictions: Prediction array (N, H, W)
        threshold: Detection threshold
        pixel_gap: Gap tolerance for component labeling
        psf_width: PSF width for trail matching
        max_workers: Number of parallel workers
        use_threads: Use threads instead of processes
        stack_fp: Optional (N, H, W) mask of stack false positives to ignore

    Returns:
        (tp, fp, fn, catalog_with_detections)
    """
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    cat = catalog.copy()
    cat["nn_detected"] = False

    groups = _prepare_catalog_groups(cat)
    image_ids = [img_id for img_id in groups.keys() if 0 <= img_id < predictions.shape[0]]

    if not image_ids:
        return 0, 0, 0, cat

    # Prepare tasks
    tasks = []
    for img_id in image_ids:
        rows_np, idx_np = groups[img_id]
        pred2d = predictions[img_id]
        stack_fp2d = None if stack_fp is None else stack_fp[img_id]
        tasks.append((img_id, rows_np, idx_np, pred2d, float(threshold),
                     pixel_gap, psf_width, stack_fp2d))

    # Execute in parallel
    executor = cf.ThreadPoolExecutor if use_threads else cf.ProcessPoolExecutor

    tp = fp = fn = 0
    with executor(max_workers=max_workers) as pool:
        for tpi, fpi, fni, idxs, detected in pool.map(_evaluate_single_image, tasks, chunksize=1):
            tp += tpi
            fp += fpi
            fn += fni
            cat.loc[idxs, "nn_detected"] = detected

    return int(tp), int(fp), int(fn), cat


# =============================================================================
# Pixel-wise Evaluation
# =============================================================================

def pixelwise_confusion(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float
) -> tuple[int, int, int, int]:
    """
    Compute pixel-level confusion matrix.

    Args:
        predictions: Prediction array (N, H, W) or (H, W)
        ground_truth: Ground truth mask (N, H, W) or (H, W)
        threshold: Detection threshold

    Returns:
        (tp, fp, fn, tn)
    """
    gt = (ground_truth > 0).astype(np.uint8)
    pred = (predictions >= threshold).astype(np.uint8)

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    return tp, fp, fn, tn


# =============================================================================
# Combined Evaluation
# =============================================================================

def mark_detections(
    catalog: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float,
    **kwargs
) -> pd.DataFrame:
    """
    Mark which catalog objects were detected.

    Convenience wrapper that returns only the marked catalog.
    """
    _, _, _, cat = objectwise_confusion(catalog, predictions, threshold, **kwargs)
    return cat


def full_confusion(
    catalog: pd.DataFrame,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    **kwargs
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], pd.DataFrame]:
    """
    Compute both object-level and pixel-level confusion matrices.

    Returns:
        ((obj_tp, obj_fp, obj_fn), (pix_tp, pix_fp, pix_fn, pix_tn), catalog)
    """
    obj_tp, obj_fp, obj_fn, cat = objectwise_confusion(
        catalog, predictions, threshold, **kwargs
    )

    pix_tp, pix_fp, pix_fn, pix_tn = pixelwise_confusion(
        predictions, ground_truth, threshold
    )

    return (obj_tp, obj_fp, obj_fn), (pix_tp, pix_fp, pix_fn, pix_tn), cat

