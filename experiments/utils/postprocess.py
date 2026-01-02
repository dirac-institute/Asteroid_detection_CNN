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