import numpy as np
import scipy.ndimage as ndi

def _safe_div(a, b) -> float:
    return float(a) / float(b) if b else 0.0

def _label_components_fds(mask_bool, pixel_gap=3):
    if pixel_gap>1:
        grown = ndi.binary_dilation(mask_bool, structure=np.ones((2*pixel_gap+1,2*pixel_gap+1), bool))
    else:
        grown = mask_bool
    labels, n = ndi.label(grown, structure=np.ones((3,3), bool))  # 8-connectivity
    return labels, int(n)

def fppi(fp: np.ndarray, denom_images: int) -> np.ndarray:
    fp = np.asarray(fp, dtype=np.float64)
    if denom_images <= 0:
        raise ValueError("denom_images must be > 0")
    return fp / float(denom_images)

def precision(tp, fp):
    """
    Precision = TP / (TP + FP)
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = tp + fp
    m = denom > 0
    out[m] = tp[m] / denom[m]
    return out

def recall(tp, fn):
    """
    Recall = TP / (TP + FN)
    """
    tp = np.asarray(tp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = tp + fn
    m = denom > 0
    out[m] = tp[m] / denom[m]
    return out


def f1(tp, fp, fn):
    """
    F1 = 2 TP / (2 TP + FP + FN)
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = 2 * tp + fp + fn
    m = denom > 0
    out[m] = (2 * tp[m]) / denom[m]
    return out


def f2(tp, fp, fn):
    """
    F2 = 5 TP / (5 TP + FP + 4 FN)
    """
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)

    out = np.zeros_like(tp, dtype=np.float64)
    denom = 5 * tp + fp + 4 * fn
    m = denom > 0
    out[m] = (5 * tp[m]) / denom[m]
    return out

def make_monotone_increasing(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.maximum.accumulate(y)