import tools
from astroML.crossmatch import crossmatch_angular
import numpy as np
import pandas as pd
import multiprocessing


def create_nn_prediction(dataset_path, model_path="../DATA/Trained_model", threshold=0.5, batch_size=1024,
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
            with tf.device("/cpu:0"):
                predictions = np.array(tf.image.resize(predictions, tfrecord_shape[:-1]))
        if threshold > 0:
            predictions = np.ceil(predictions)
        predictions = tools.data.npy_merge(predictions, (4176, 2048))
        if not dataset_path_iterable:
            return predictions
        else:
            predictions_list += (predictions,)
    return predictions_list


def get_injection_catalog(butler, collection):
    injection_catalog_ids = np.unique(np.array(list(butler.registry.queryDatasets("injection_catalog",
                                                                                  collections=collection,
                                                                                  instrument='HSC',
                                                                                  findFirst=True))))
    injection_catalog = [butler.get("injection_catalog",
                                    dataId=i.dataId,
                                    collections=collection).to_pandas() for i in injection_catalog_ids]
    return pd.concat(injection_catalog).set_index("injection_id").sort_index()


def one_image_hits(butler, injected_calexp_ref, postisrccd_catalog_ref,
                   output_coll, calexp_dimensions, n, stack_source_catalog_id=None,
                   nn_predictions=None):
    injected_calexp_wcs = butler.get("injected_calexp.wcs",
                                     dataId=injected_calexp_ref.dataId,
                                     collections=output_coll)
    injected_postisrccd_catalog = butler.get("injected_postISRCCD_catalog",
                                             dataId=postisrccd_catalog_ref.dataId,
                                             collections=output_coll)
    results = [None] * len(injected_postisrccd_catalog)
    if stack_source_catalog_id is not None:
        src_catalog = butler.get("src",
                                 dataId=stack_source_catalog_id.dataId,
                                 collections=output_coll)
        injected_src_catalog = butler.get("injected_src",
                                          dataId=stack_source_catalog_id.dataId,
                                          collections=output_coll)

        sc = src_catalog.asAstropy().to_pandas()
        isc = injected_src_catalog.asAstropy().to_pandas()
        dist, ind = crossmatch_angular(isc[['coord_ra', 'coord_dec']].values,
                                       sc[['coord_ra', 'coord_dec']].values, 0.04 / 3600)

        stack_detection_origins = injected_calexp_wcs.skyToPixelArray(
            np.array([injected_src_catalog["coord_ra"][np.isinf(dist)]]),
            np.array([injected_src_catalog["coord_dec"][np.isinf(dist)]]),
            degrees=False)
        stack_predictions = np.zeros(calexp_dimensions)
        stack_predictions[stack_detection_origins[1].astype(int), stack_detection_origins[0].astype(int)] = 1
    else:
        stack_predictions = 0

    injected_origin = injected_calexp_wcs.skyToPixelArray(np.array([injected_postisrccd_catalog["ra"]]),
                                                          np.array([injected_postisrccd_catalog["dec"]]),
                                                          degrees=True)
    injected_angle = injected_postisrccd_catalog["beta"]
    injected_length = injected_postisrccd_catalog["trail_length"]
    for i, catalog_row in enumerate(injected_postisrccd_catalog):
        injected_mask = tools.data.draw_one_line(np.zeros(calexp_dimensions),
                                                 (injected_origin[0][i], injected_origin[1][i]),
                                                 injected_angle[i], injected_length[i])
        results[i] = {'injection_id': catalog_row['injection_id'],
                      'ra': catalog_row['ra'],
                      'dec': catalog_row['dec'],
                      'trail_length': catalog_row['trail_length'],
                      'beta': catalog_row['beta'],
                      'surface_brightness': catalog_row['mag'],
                      'n': n,
                      'x': round(injected_origin[0][i]),
                      'y': round(injected_origin[1][i])}
        if nn_predictions is not None:
            results[i]["NN_detected"] = int(((injected_mask == 1) & (nn_predictions == 1)).sum() > 0)
        if stack_source_catalog_id is not None:
            results[i]["stack_detected"] = int((injected_mask * stack_predictions).sum() > 0)
    return results


def recovered_sources(repo, collection, nn_predictions=None, val_index=None, n_parallel=None):
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    postisrccd_catalog_ref = np.unique(np.array(list(butler.registry.queryDatasets("injected_postISRCCD_catalog",
                                                                                   collections=collection,
                                                                                   instrument='HSC',
                                                                                   findFirst=True))))
    injected_calexp_ref = np.unique(np.array(list(butler.registry.queryDatasets("injected_calexp",
                                                                                collections=collection,
                                                                                instrument='HSC',
                                                                                findFirst=True))))
    source_catalog_ids = np.unique(np.array(list(butler.registry.queryDatasets("injected_src",
                                                                               collections=collection,
                                                                               instrument='HSC',
                                                                               findFirst=True))))
    calexp_dimensions = butler.get("injected_calexp.dimensions",
                                   dataId=injected_calexp_ref[0].dataId,
                                   collections=collection)
    calexp_dimensions = (calexp_dimensions.y, calexp_dimensions.x)
    injection_catalog = get_injection_catalog(butler, collection)
    if val_index is None:
        val_index = list(range(len(injected_calexp_ref)))
    if nn_predictions is None:
        nn_predictions = [None] * len(injected_calexp_ref)
    parameters = [(butler, injected_calexp_ref[i], postisrccd_catalog_ref[i],
                   collection, calexp_dimensions, i, source_catalog_ids[i], nn_predictions[i]) for i in val_index]
    if n_parallel > 1:
        with multiprocessing.Pool(n_parallel) as pool:
            results = pool.starmap(one_image_hits, parameters)
    else:
        results = [None] * len(parameters)
        for i, p in enumerate(parameters):
            results[i] = one_image_hits(*p)
            print("\r", i + 1, "/", len(parameters), end="")
    results = pd.DataFrame(list(np.array(results).flatten())).set_index("injection_id").sort_index()
    return injection_catalog.merge(results)
