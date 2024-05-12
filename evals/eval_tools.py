import sys, os
import tensorflow as tf
sys.path.append("..")
import tools
from astroML.crossmatch import crossmatch_angular
import numpy as np
import pandas as pd
import multiprocessing

def create_NN_prediction(dataset_path, model_path="../DATA/Trained_model", threshold=0.5, batch_size=1024):
    dataset_test = tf.data.TFRecordDataset([dataset_path])
    tfrecord_shape = tools.model.get_shape_of_quadratic_image_tfrecord(dataset_test)
    dataset_test = dataset_test.map(tools.model.parse_function(img_shape=tfrecord_shape, test=True))
    dataset_test = dataset_test.batch(batch_size)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(model_path, compile=False)
        predictions = model.predict(dataset_test, verbose=0)
    predictions = tools.data.npy_merge(predictions > threshold, (4176, 2048))
    return predictions


def one_hit(p, injected_calexp, catalog_row, i):
    origin = injected_calexp.getWcs().skyToPixelArray(np.array([catalog_row["ra"]]), np.array([catalog_row["dec"]]),
                                                      degrees=True)
    angle = catalog_row["beta"]
    length = catalog_row["trail_length"]
    mask = np.zeros(injected_calexp.image.array.shape)
    mask = tools.data.draw_one_line(mask, origin, angle, length, line_thickness=6)
    return {'injection_id': catalog_row['injection_id'], 'ra': catalog_row['ra'], 'dec': catalog_row['dec'],
            'trail_length': catalog_row['trail_length'], 'beta': catalog_row['beta'],
            'mag': catalog_row['mag'], 'n': i, 'x': round(origin[0][0]), 'y': round(origin[1][0]),
            'detected': int(((mask == 1) & (p == 1)).sum() > 0)}


def compare_NN_predictions(p, repo, output_coll, val_index=None, batch_size=None):
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    catalog_ref = list(butler.registry.queryDatasets("injected_postISRCCD_catalog",
                                                     collections=output_coll,
                                                     instrument='HSC'))
    ref = list(butler.registry.queryDatasets("injected_calexp",
                                             collections=output_coll,
                                             instrument='HSC'))
    parameters = []
    if batch_size is None:
        batch_size = os.cpu_count() - 1
    if val_index is None:
        val_index = list(range(len(catalog_ref)))
    for i, index in enumerate(val_index):
        injected_calexp = butler.get("injected_calexp",
                                     dataId=ref[index].dataId,
                                     collections=output_coll)
        catalog = butler.get("injected_postISRCCD_catalog",
                             dataId=catalog_ref[index].dataId,
                             collections=output_coll)
        # catalog[catalog["injection_flag"] == 0]
        parameters += [(p[i], injected_calexp, k, i) for k in catalog]
    pool = multiprocessing.Pool(batch_size)
    list_cat = pool.starmap(one_hit, parameters)
    pool.close()
    pool.join()
    return pd.DataFrame(list_cat)


def NN_comparation_histogram_data(model_path, tf_dataset_path, val_index_path, repo, output_coll,
                                  column_name="trail_length", batch_size=None, threshold=0.5):
    predictions = create_NN_prediction(tf_dataset_path, model_path, threshold=threshold)
    with open(val_index_path, 'rb') as f:
        val_index = np.load(f)
        val_index.sort()
    cat = compare_NN_predictions(predictions, repo, output_coll, val_index=val_index, batch_size=batch_size)
    return cat[cat["detected"] == 1][column_name].to_numpy(), cat[column_name].to_numpy()

def one_LSST_stack_comparison(butler, output_coll, injection_catalog_id, source_catalog_id, calexp_id,
                              column_name):
    injection_catalog = butler.get("injected_postISRCCD_catalog",
                                   dataId=injection_catalog_id.dataId,
                                   collections=output_coll, )
    # injection_catalog[injection_catalog["injection_flag"] == 0]
    original_source_catalog = butler.get("src",
                                         dataId=source_catalog_id.dataId,
                                         collections=output_coll, )
    source_catalog = butler.get("injected_src",
                                dataId=source_catalog_id.dataId,
                                collections=output_coll, )
    calexp = butler.get("injected_calexp",
                        dataId=calexp_id.dataId,
                        collections=output_coll)
    sc = source_catalog.asAstropy().to_pandas()
    osc = original_source_catalog.asAstropy().to_pandas()
    dist, ind = crossmatch_angular(sc[['coord_ra', 'coord_dec']].values,
                                   osc[['coord_ra', 'coord_dec']].values, 0.04 / 3600)
    source_origin = calexp.getWcs().skyToPixelArray(np.array([source_catalog["coord_ra"][np.isinf(dist)]]),
                                                    np.array([source_catalog["coord_dec"][np.isinf(dist)]]),
                                                    degrees=False)
    injected_origin = calexp.getWcs().skyToPixelArray(np.array([injection_catalog["ra"]]),
                                                      np.array([injection_catalog["dec"]]),
                                                      degrees=True)
    angle = injection_catalog["beta"]
    length = injection_catalog["trail_length"]
    mask_source = np.zeros(calexp.image.array.shape)
    mask_source[source_origin[1].astype(int), source_origin[0].astype(int)] = 1
    matched_values = np.array([])
    for j in range(len(angle)):
        mask_inject = tools.data.draw_one_line(np.zeros(calexp.image.array.shape),
                                               (injected_origin[0][j], injected_origin[1][j]),
                                               angle[j], length[j])
        if (mask_inject * mask_source).sum() > 0:
            matched_values = np.append(matched_values, injection_catalog[column_name][j])
    return matched_values

def LSST_stack_comparation_histogram_data(repo, output_coll, val_index_path,
                                          column_name="trail_length", batch_size=None):
    from lsst.daf.butler import Butler
    with open(val_index_path, 'rb') as f:
        val_index = np.load(f)
        val_index.sort()
    butler = Butler(repo)
    injection_catalog_ids = list(butler.registry.queryDatasets("injected_postISRCCD_catalog", collections=output_coll, instrument='HSC'))
    source_catalog_ids = list(butler.registry.queryDatasets("injected_src", collections=output_coll, instrument='HSC'))
    calexp_ids = list(butler.registry.queryDatasets("injected_calexp", collections=output_coll, instrument='HSC'))
    parameters = [(butler, output_coll,
                    injection_catalog_ids[i], source_catalog_ids[i],
                    calexp_ids[i], column_name) for i in val_index]
    if batch_size is None:
        batch_size = os.cpu_count() - 1
    pool = multiprocessing.Pool(batch_size)
    list_cat = pool.starmap(one_LSST_stack_comparison, parameters)
    pool.close()
    pool.join()
    return np.concatenate(list_cat)


def FDS(img, roots, pixel_gap, visited_pixels=None):
    if visited_pixels is None:
        visited_pixels = set()
    height = img.shape[0]
    width = img.shape[1]
    todo = [(roots[0], roots[1])]
    mask = np.zeros((height, width))
    while todo:
        j, i = todo.pop()
        visited_pixels.add((j, i))
        if img[j, i] != 0:
            img[j, i] = 0
            mask[j, i] = 1
            for gap_x in range(-pixel_gap - 1, pixel_gap + 1):
                for gap_y in range(-pixel_gap - 1, pixel_gap + 1):
                    if not (j + gap_x, i + gap_y) in visited_pixels and (0 <= j + gap_x < height) and (
                            0 <= i + gap_y < width):
                        todo += [(j + gap_x, i + gap_y)]
    return mask, visited_pixels


def get_one_image_mask(true_img, prediction_img, pixel_gap=15):
    p_img = np.copy(prediction_img)
    t_img = np.copy(true_img)
    mask = np.zeros((t_img.shape))
    tp = 0
    fp = 0
    fn = 0
    while p_img.sum() != 0:
        roots = np.where(p_img != 0)
        mask_p, __ = FDS(p_img, (roots[0][0], roots[1][0]), pixel_gap)
        if np.any(mask_p * true_img != 0):
            tp += 1
            mask[mask_p != 0] = 1
        else:
            fp += 1
            mask[mask_p != 0] = 2
    while t_img.sum() != 0:
        roots = np.where(t_img != 0)
        mask_t, __ = FDS(t_img, (roots[0][0], roots[1][0]), pixel_gap=1)
        if not np.any(mask_t * prediction_img != 0):
            fn += 1
            mask[mask_t != 0] = 3
    return tp, fp, fn, mask


def get_mask(truths, predictions, batch_size=None):
    parameters = [(truths[i], predictions[i]) for i in range(truths.shape[0])]
    masks = np.empty(truths.shape)
    true_positive = np.empty(truths.shape[0])
    false_positive = np.empty(truths.shape[0])
    false_negative = np.empty(truths.shape[0])
    if batch_size is None:
        batch_size = os.cpu_count() - 3
    pool = multiprocessing.Pool(batch_size)
    list_cat = pool.starmap(get_one_image_mask, parameters)
    pool.close()
    pool.join()
    for i, (tp, fp, fn, mask) in enumerate(list_cat):
        true_positive[i] = tp
        false_positive[i] = fp
        false_negative[i] = fn
        masks[i, :, :] = mask
    return true_positive, false_positive, false_negative, masks


def f1_score(tp, fp, fn):
    return tp / (tp + 0.5 * (fp + fn))


def precision(tp, fp, fn):
    return tp / (tp + fp)


def recall(tp, fp, fn):
    return tp / (tp + fn)