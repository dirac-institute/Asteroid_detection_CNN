

def two_stage_objectwise_confusion (catalog, predictions, t_low, threshold, pixel_gap=10, psf_width=40, score_method="max"):
    true_positive = 0
    false_negative = 0
    false_positive = 0
    cat = catalog.copy()

    for image_id in np.sort(catalog["image_id"].unique()):
        print (f"\rProcessing {image_id +1} / {catalog["image_id"].max()+1}", end="", flush=True)
        predicted_positive = 0
        lab, n = evaluation._label_components_fds(predictions[image_id]>=t_low, pixel_gap=pixel_gap)
        lab_predictions = np.zeros_like(predictions[image_id])
        for i in range(1, n + 1):
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
            lab_predictions[comp_mask] = comp_score
            if comp_score >= threshold:
                predicted_positive += 1
        lab_predictions = lab_predictions >= threshold
        for i, row in catalog[catalog["image_id"]==image_id].iterrows():
            catalog_id = np.argwhere((catalog["x"] == row["x"]) & (catalog["y"] == row["y"]) & (catalog["image_id"] == row["image_id"]))[0][0]
            mask = draw_one_line(np.zeros((predictions.shape[1], predictions.shape[2]), dtype=np.uint8), [row["x"], row["y"]], row["beta"], row["trail_length"], true_value=1, line_thickness=int(psf_width/2))
            mask = mask != 0
            intersection = lab_predictions & mask
            lab_ids = np.unique(lab[intersection])
            lab_ids = lab_ids[lab_ids!=0]  # remove background
            lab_removed = np.isin(lab, lab_ids)
            lab_predictions[lab_removed] = 0
            if np.any(intersection):
                true_positive += 1
                cat.at[catalog_id, "nn_detected"] = True
            else:
                false_negative += 1
                cat.at[catalog_id, "nn_detected"] = False
        false_positive += max(predicted_positive - true_positive , 0)
    print ()
    return true_positive, false_positive, false_negative, cat