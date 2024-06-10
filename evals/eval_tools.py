import sys, os
import tensorflow as tf

sys.path.append("..")
import tools
from astroML.crossmatch import crossmatch_angular
import numpy as np
import pandas as pd
import multiprocessing
from collections import deque
from joblib import Parallel, delayed


def create_NN_prediction(dataset_path, model_path="../DATA/Trained_model", threshold=0.5, batch_size=1024,
                         verbose=True):
    if type(dataset_path) is str:
        dataset_path = [dataset_path]
        dataset_path_iterable = False
    else:
        dataset_path_iterable = True
    predictions_list = ()
    if len(tf.config.list_physical_devices('GPU')) == 0:
        if verbose:
            print("No GPU detected")
        mirrored_strategy = tf.distribute.get_strategy()
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        for i, dataset in enumerate(dataset_path):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset {dataset} not found")

            dataset_test = tf.data.TFRecordDataset([dataset])
            tfrecord_shape = tools.model.get_shape_of_quadratic_image_tfrecord(dataset_test)
            dataset_test = dataset_test.interleave(lambda x: tf.data.Dataset.from_tensors(
                tools.model.parse_function(img_shape=tfrecord_shape, test=True)(x)),
                                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_test = dataset_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            predictions = model.predict(dataset_test, verbose=1 if verbose else 0)
            if threshold > 0:
                predictions = (predictions > threshold).astype(float)
            else:
                predictions = predictions.astype(float)
            if not tuple(model.outputs[0].shape[1:]) == tfrecord_shape:
                predictions = np.array(tf.image.resize(predictions, tfrecord_shape[:-1]))
            if threshold > 0:
                predictions = np.ceil(predictions)
            predictions = tools.data.npy_merge(predictions, (4176, 2048))
            if not dataset_path_iterable:
                return predictions
            else:
                predictions_list += (predictions,)
    return predictions_list


def one_image_hits(p, butler, ref, catalog_ref, output_coll, calexp_dimensions, n):
    injected_calexp = butler.get("injected_calexp.wcs",
                                 dataId=ref.dataId,
                                 collections=output_coll)
    catalog = butler.get("injected_postISRCCD_catalog",
                         dataId=catalog_ref.dataId,
                         collections=output_coll)
    list_cat = [None] * len(catalog)
    for i, catalog_row in enumerate(catalog):
        origin = injected_calexp.skyToPixelArray(np.array([catalog_row["ra"]]), np.array([catalog_row["dec"]]),
                                                 degrees=True)
        angle = catalog_row["beta"]
        length = catalog_row["trail_length"]
        mask = np.zeros(calexp_dimensions)
        mask = tools.data.draw_one_line(mask, origin, angle, length, line_thickness=6)
        list_cat[i] = {'injection_id': catalog_row['injection_id'], 'ra': catalog_row['ra'], 'dec': catalog_row['dec'],
                       'trail_length': catalog_row['trail_length'], 'beta': catalog_row['beta'],
                       'mag': catalog_row['mag'], 'n': n, 'x': round(origin[0][0]), 'y': round(origin[1][0]),
                       'detected': int(((mask == 1) & (p == 1)).sum() > 0)}
    return list_cat


def compare_NN_predictions(p, repo, output_coll, val_index=None, multiprocess_size=10):
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    catalog_ref = list(butler.registry.queryDatasets("injected_postISRCCD_catalog",
                                                     collections=output_coll,
                                                     instrument='HSC'))
    ref = list(butler.registry.queryDatasets("injected_calexp",
                                             collections=output_coll,
                                             instrument='HSC'))
    calexp_dimensions = butler.get("injected_calexp.dimensions", dataId=ref[0].dataId, collections=output_coll)
    calexp_dimensions = (calexp_dimensions.y, calexp_dimensions.x)
    if val_index is None:
        val_index = list(range(len(catalog_ref)))
    parameters = []
    parameters += [(p[j], butler, ref[i], catalog_ref[i], output_coll, calexp_dimensions, i) for j, i in
                   enumerate(val_index)]
    if multiprocess_size is None:
        multiprocess_size = max(1, min(os.cpu_count() - 1, len(parameters)))
    if multiprocess_size > 1:
        with multiprocessing.Pool(multiprocess_size) as pool:
            list_cat = pool.starmap(one_image_hits, parameters)
    else:
        list_cat = [None] * len(parameters)
        for i, p in enumerate(parameters):
            list_cat[i] = one_image_hits(*p)
    return pd.DataFrame(list(np.array(list_cat).flatten()))


def NN_comparation_histogram_data(predictions, val_index_path, repo, output_coll,
                                  column_name="trail_length", multiprocess_size=10):
    with open(val_index_path, 'rb') as f:
        val_index = np.load(f)
        val_index.sort()
    cat = compare_NN_predictions(predictions, repo, output_coll, val_index=val_index,
                                 multiprocess_size=multiprocess_size)
    return cat[cat["detected"] == 1][column_name].to_numpy(), cat[column_name].to_numpy()


def one_LSST_stack_comparison(butler, output_coll, injection_catalog_id, source_catalog_id, calexp_id,
                              calexp_dimensions, column_name):
    injection_catalog = butler.get("injected_postISRCCD_catalog",
                                   dataId=injection_catalog_id.dataId,
                                   collections=output_coll, )
    original_source_catalog = butler.get("src",
                                         dataId=source_catalog_id.dataId,
                                         collections=output_coll, )
    source_catalog = butler.get("injected_src",
                                dataId=source_catalog_id.dataId,
                                collections=output_coll, )
    calexp = butler.get("injected_calexp.wcs",
                        dataId=calexp_id.dataId,
                        collections=output_coll)
    sc = source_catalog.asAstropy().to_pandas()
    osc = original_source_catalog.asAstropy().to_pandas()
    dist, ind = crossmatch_angular(sc[['coord_ra', 'coord_dec']].values,
                                   osc[['coord_ra', 'coord_dec']].values, 0.04 / 3600)
    source_origin = calexp.skyToPixelArray(np.array([source_catalog["coord_ra"][np.isinf(dist)]]),
                                           np.array([source_catalog["coord_dec"][np.isinf(dist)]]),
                                           degrees=False)
    injected_origin = calexp.skyToPixelArray(np.array([injection_catalog["ra"]]),
                                             np.array([injection_catalog["dec"]]),
                                             degrees=True)
    angle = injection_catalog["beta"]
    length = injection_catalog["trail_length"]
    mask_source = np.zeros(calexp_dimensions)
    mask_source[source_origin[1].astype(int), source_origin[0].astype(int)] = 1
    matched_values = deque([])
    for j in range(len(angle)):
        mask_inject = tools.data.draw_one_line(np.zeros(calexp_dimensions),
                                               (injected_origin[0][j], injected_origin[1][j]),
                                               angle[j], length[j])
        if (mask_inject * mask_source).sum() > 0:
            matched_values.append(j)
    if type(column_name) is str:
        column_name = [column_name]
    return injection_catalog[column_name].to_pandas().iloc[list(matched_values)]


def LSST_stack_comparation_histogram_data(repo, output_coll, val_index_path,
                                          column_name="trail_length", multiprocess_size=None):
    from lsst.daf.butler import Butler
    with open(val_index_path, 'rb') as f:
        val_index = np.load(f)
        val_index.sort()
    butler = Butler(repo)
    injection_catalog_ids = list(
        butler.registry.queryDatasets("injected_postISRCCD_catalog", collections=output_coll, instrument='HSC'))
    source_catalog_ids = list(butler.registry.queryDatasets("injected_src", collections=output_coll, instrument='HSC'))
    calexp_ids = list(butler.registry.queryDatasets("injected_calexp", collections=output_coll, instrument='HSC'))
    calexp_dimensions = butler.get("injected_calexp.dimensions", dataId=calexp_ids[0].dataId, collections=output_coll)
    calexp_dimensions = (calexp_dimensions.y, calexp_dimensions.x)
    parameters = [(butler, output_coll,
                   injection_catalog_ids[i], source_catalog_ids[i],
                   calexp_ids[i], calexp_dimensions, column_name) for i in val_index]
    if multiprocess_size is None:
        multiprocess_size = max(1, os.cpu_count() - 1)
    if multiprocess_size > 1:
        with multiprocessing.Pool(multiprocess_size) as pool:
            list_cat = pool.starmap(one_LSST_stack_comparison, parameters)
    else:
        list_cat = [None] * len(parameters)
        for i, p in enumerate(parameters):
            list_cat[i] = one_LSST_stack_comparison(*p)
    return pd.concat(list_cat, ignore_index=True).to_numpy().squeeze()


def FDS(img, roots, pixel_gap, visited_pixels=None):
    if visited_pixels is None:
        visited_pixels = np.zeros(img.shape, dtype=bool)
    height, width = img.shape
    todo = deque([(roots[0], roots[1])])
    mask = np.zeros((height, width), dtype=bool)

    while todo:
        j, i = todo.pop()
        if not visited_pixels[j, i] and img[j, i] != 0:
            visited_pixels[j, i] = True
            img[j, i] = False
            mask[j, i] = True
            j_min = max(j - pixel_gap, 0)
            j_max = min(j + pixel_gap + 1, height)
            i_min = max(i - pixel_gap, 0)
            i_max = min(i + pixel_gap + 1, width)
            for jj in range(j_min, j_max):
                for ii in range(i_min, i_max):
                    if not visited_pixels[jj, ii]:
                        todo.append((jj, ii))

    return mask, visited_pixels


def get_one_image_mask(true_img, prediction_img, pixel_gap=15):
    p_img = prediction_img != 0
    t_img = true_img != 0
    mask = np.zeros((t_img.shape))
    tp = 0
    fp = 0
    fn = 0
    visited = None
    while p_img.sum() != 0:
        roots = np.where(p_img != 0)
        mask_p, visited = FDS(p_img, (roots[0][0], roots[1][0]), pixel_gap, visited_pixels=visited)
        if np.any(mask_p * true_img != 0):
            tp += 1
            mask[mask_p != 0] = 1
        else:
            fp += 1
            mask[mask_p != 0] = 2
    visited = None
    while t_img.sum() != 0:
        roots = np.where(t_img != 0)
        mask_t, visited = FDS(t_img, (roots[0][0], roots[1][0]), pixel_gap=1, visited_pixels=visited)
        if not np.any(mask_t * prediction_img != 0):
            fn += 1
            mask[mask_t != 0] = 3
    return tp, fp, fn, mask


def get_mask(truths, predictions, multiprocess_size=None):
    if multiprocess_size is None:
        multiprocess_size = max(1, min(os.cpu_count() - 1, truths.shape[0]))
    if multiprocess_size > 1:
        parameters = [(truths[i], predictions[i]) for i in range(truths.shape[0])]
        with multiprocessing.Pool(multiprocess_size) as pool:
            results = pool.starmap(get_one_image_mask, parameters)
    else:
        results = [None] * truths.shape[0]
        for i in range(truths.shape[0]):
            results[i] = get_one_image_mask(truths[i], predictions[i])
    masks = np.empty(truths.shape)
    true_positive = np.empty(truths.shape[0])
    false_positive = np.empty(truths.shape[0])
    false_negative = np.empty(truths.shape[0])

    for i, (tp, fp, fn, mask) in enumerate(results):
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
