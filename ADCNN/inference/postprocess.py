from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.ndimage as ndi

from ADCNN.evaluation.geometry import label_components


RECOMMENDED_POSTPROCESS_CONFIG = {
    "threshold": 0.20,
    "pixel_gap": 2,
    "min_area": 3,
    "max_area": None,
    "min_score": 0.20,
    "min_peak_probability": 0.20,
    "score_method": "topk_mean",
    "topk_fraction": 0.05,
    "return_label_mask": False,
}
"""
Best current extractor settings from the repo notebook experiments.

Important:
- these settings are for candidate extraction only
- the strongest overall results still came from applying a rescue-specific reranker
  after extraction
- the aggressive low-threshold / large-gap settings from the optimistic sweep were
  not kept because they failed strict validation
"""


@dataclass
class AsteroidDetection:
    """One detected asteroid candidate."""

    label_id: int
    x: float
    y: float
    area: int
    score: float
    peak_probability: float
    mean_probability: float
    topk_mean_probability: float
    bbox_x0: int
    bbox_x1: int
    bbox_y0: int
    bbox_y1: int


def _component_score(values: np.ndarray, method: str = "topk_mean", topk_fraction: float = 0.05) -> float:
    if values.size == 0:
        return 0.0
    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method == "sum":
        return float(values.sum())
    if method == "topk_mean":
        k = max(1, int(np.ceil(values.size * float(topk_fraction))))
        top = np.partition(values, values.size - k)[-k:]
        return float(top.mean())
    raise ValueError(f"Unknown score_method={method!r}")


def _keep_component(
    values: np.ndarray,
    area: int,
    *,
    min_area: int,
    max_area: int | None,
    min_score: float,
    min_peak_probability: float,
    score_method: str,
    topk_fraction: float,
) -> tuple[bool, float]:
    score = _component_score(values, method=score_method, topk_fraction=topk_fraction)
    peak = float(values.max()) if values.size else 0.0

    if area < int(min_area):
        return False, score
    if max_area is not None and area > int(max_area):
        return False, score
    if score < float(min_score):
        return False, score
    if peak < float(min_peak_probability):
        return False, score
    return True, score


def postprocess_prediction(
    prediction: np.ndarray,
    *,
    threshold: float = RECOMMENDED_POSTPROCESS_CONFIG["threshold"],
    pixel_gap: int = RECOMMENDED_POSTPROCESS_CONFIG["pixel_gap"],
    min_area: int = RECOMMENDED_POSTPROCESS_CONFIG["min_area"],
    max_area: int | None = RECOMMENDED_POSTPROCESS_CONFIG["max_area"],
    min_score: float = RECOMMENDED_POSTPROCESS_CONFIG["min_score"],
    min_peak_probability: float = RECOMMENDED_POSTPROCESS_CONFIG["min_peak_probability"],
    score_method: str = RECOMMENDED_POSTPROCESS_CONFIG["score_method"],
    topk_fraction: float = RECOMMENDED_POSTPROCESS_CONFIG["topk_fraction"],
    return_label_mask: bool = False,
) -> tuple[np.ndarray, list[dict]] | tuple[np.ndarray, list[list[dict]]]:
    """
    Convert NN probability maps into:
    1. a mask of detected asteroids
    2. a list of detections with centroid coordinates

    Important behavior:
    - grouping uses `label_components(..., pixel_gap=...)` to bridge small gaps
    - the returned mask and centroids are computed on the ORIGINAL thresholded support
      inside each grown component, not on the fully grown mask

    Args:
        prediction:
            Either one 2D probability map with shape (H, W) or a batch with
            shape (N, H, W).
        threshold:
            Probability threshold used to define strict support pixels.
        pixel_gap:
            Gap-bridging radius used only for grouping components.
        min_area:
            Minimum number of strict-support pixels required to keep a component.
        max_area:
            Optional maximum strict-support area.
        min_score:
            Minimum component score required to keep a detection.
        min_peak_probability:
            Minimum peak probability inside the strict support.
        score_method:
            One of: "topk_mean", "max", "mean", "sum"
        topk_fraction:
            Used only when `score_method="topk_mean"`.
        return_label_mask:
            If True, output is an integer label mask. Otherwise it is a boolean mask.

    Returns:
        If input is 2D:
            detection_mask:
                Boolean mask or integer label mask of accepted detections.
            detections:
                List of dicts with centroid x/y and basic component metadata.

        If input is 3D:
            masks:
                Array with shape (N, H, W)
            detections_per_image:
                List of detection lists, one entry per image.
    """
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 3:
        return postprocess_predictions(
            prediction,
            threshold=threshold,
            pixel_gap=pixel_gap,
            min_area=min_area,
            max_area=max_area,
            min_score=min_score,
            min_peak_probability=min_peak_probability,
            score_method=score_method,
            topk_fraction=topk_fraction,
            return_label_mask=return_label_mask,
        )
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be 2D (H, W), got shape={prediction.shape}")

    strict_support = prediction >= float(threshold)
    grown_labels, n_labels = label_components(strict_support, pixel_gap=int(pixel_gap))

    if return_label_mask:
        detection_mask = np.zeros(prediction.shape, dtype=np.int32)
    else:
        detection_mask = np.zeros(prediction.shape, dtype=bool)

    if n_labels <= 0:
        return detection_mask, []

    detections: list[dict] = []
    slices = ndi.find_objects(grown_labels)
    out_label = 0

    for label_id in range(1, n_labels + 1):
        slc = slices[label_id - 1]
        if slc is None:
            continue

        grown_local = grown_labels[slc] == label_id
        support_local = strict_support[slc] & grown_local
        rr_local, cc_local = np.nonzero(support_local)
        if rr_local.size == 0:
            continue

        rr = rr_local + int(slc[0].start)
        cc = cc_local + int(slc[1].start)
        values = prediction[rr, cc]
        area = int(rr.size)

        keep, score = _keep_component(
            values,
            area,
            min_area=min_area,
            max_area=max_area,
            min_score=min_score,
            min_peak_probability=min_peak_probability,
            score_method=score_method,
            topk_fraction=topk_fraction,
        )
        if not keep:
            continue

        out_label += 1
        y0 = int(rr.min())
        y1 = int(rr.max()) + 1
        x0 = int(cc.min())
        x1 = int(cc.max()) + 1

        if return_label_mask:
            detection_mask[rr, cc] = out_label
        else:
            detection_mask[rr, cc] = True

        topk_mean = _component_score(values, method="topk_mean", topk_fraction=topk_fraction)
        det = AsteroidDetection(
            label_id=out_label,
            x=float(cc.mean()),
            y=float(rr.mean()),
            area=area,
            score=float(score),
            peak_probability=float(values.max()),
            mean_probability=float(values.mean()),
            topk_mean_probability=float(topk_mean),
            bbox_x0=x0,
            bbox_x1=x1,
            bbox_y0=y0,
            bbox_y1=y1,
        )
        detections.append(asdict(det))

    return detection_mask, detections


def postprocess_predictions(
    predictions: np.ndarray,
    *,
    threshold: float = RECOMMENDED_POSTPROCESS_CONFIG["threshold"],
    pixel_gap: int = RECOMMENDED_POSTPROCESS_CONFIG["pixel_gap"],
    min_area: int = RECOMMENDED_POSTPROCESS_CONFIG["min_area"],
    max_area: int | None = RECOMMENDED_POSTPROCESS_CONFIG["max_area"],
    min_score: float = RECOMMENDED_POSTPROCESS_CONFIG["min_score"],
    min_peak_probability: float = RECOMMENDED_POSTPROCESS_CONFIG["min_peak_probability"],
    score_method: str = RECOMMENDED_POSTPROCESS_CONFIG["score_method"],
    topk_fraction: float = RECOMMENDED_POSTPROCESS_CONFIG["topk_fraction"],
    return_label_mask: bool = False,
) -> tuple[np.ndarray, list[list[dict]]]:
    """
    Batch version of `postprocess_prediction`.

    Args:
        predictions:
            Array with shape (N, H, W)

    Returns:
        masks:
            Array with shape (N, H, W)
        detections_per_image:
            A list of detection lists, one entry per image.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 3:
        raise ValueError(f"predictions must be 3D (N, H, W), got shape={predictions.shape}")

    mask_dtype = np.int32 if return_label_mask else bool
    masks = np.zeros(predictions.shape, dtype=mask_dtype)
    detections_per_image: list[list[dict]] = []

    for image_id in range(predictions.shape[0]):
        mask, detections = postprocess_prediction(
            predictions[image_id],
            threshold=threshold,
            pixel_gap=pixel_gap,
            min_area=min_area,
            max_area=max_area,
            min_score=min_score,
            min_peak_probability=min_peak_probability,
            score_method=score_method,
            topk_fraction=topk_fraction,
            return_label_mask=return_label_mask,
        )
        masks[image_id] = mask
        for det in detections:
            det["image_id"] = int(image_id)
        detections_per_image.append(detections)

    return masks, detections_per_image


def two_threshold_prediction(
    predictions: np.ndarray,
    *,
    t_low: float = RECOMMENDED_POSTPROCESS_CONFIG["threshold"],
    pixel_gap: int = RECOMMENDED_POSTPROCESS_CONFIG["pixel_gap"],
    min_area: int = RECOMMENDED_POSTPROCESS_CONFIG["min_area"],
    max_area: int | None = RECOMMENDED_POSTPROCESS_CONFIG["max_area"],
    min_score: float = RECOMMENDED_POSTPROCESS_CONFIG["min_score"],
    min_peak_probability: float = RECOMMENDED_POSTPROCESS_CONFIG["min_peak_probability"],
    score_method: str = RECOMMENDED_POSTPROCESS_CONFIG["score_method"],
    topk_fraction: float = RECOMMENDED_POSTPROCESS_CONFIG["topk_fraction"],
    return_label_mask: bool = False,
) -> tuple[np.ndarray, list[list[dict]]]:
    """
    Backward-compatible wrapper around `postprocess_predictions`.

    The old implementation returned a score-painted array. The new implementation
    returns the actually useful postprocessed outputs:
    - detection masks
    - detected asteroid centroids / metadata
    """
    return postprocess_predictions(
        predictions,
        threshold=t_low,
        pixel_gap=pixel_gap,
        min_area=min_area,
        max_area=max_area,
        min_score=min_score,
        min_peak_probability=min_peak_probability,
        score_method=score_method,
        topk_fraction=topk_fraction,
        return_label_mask=return_label_mask,
    )
