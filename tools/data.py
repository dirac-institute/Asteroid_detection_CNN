import numpy as np
import tensorflow as tf
import multiprocessing
import cv2
import os
import time


def split(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    if arr.shape[0] % nrows != 0:
        arr = np.vstack([arr, np.zeros(shape=(nrows - (arr.shape[0] % nrows), arr.shape[1]))])
    if arr.shape[1] % ncols != 0:
        arr = np.hstack([arr, np.zeros(shape=(arr.shape[0], ncols - (arr.shape[1] % ncols)))])
    return (arr.reshape(arr.shape[0] // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))


def tfdataset_merge(dataset, shape):
    def tiles(x, y):
        y_tiled = tf.reshape(y, (x_rows, y_rows, y.shape[1], y.shape[2]))
        y_tiled = tf.transpose(y_tiled, perm=[0, 2, 1, 3])
        y_tiled = tf.reshape(y_tiled, (x_rows * y.shape[1], y_rows * y.shape[2]))
        return y_tiled[:shape[0], :shape[1]]

    if len(dataset.take(1).get_single_element()[0].shape) > 3:
        dataset = dataset.unbatch()
    image_x, image_y = dataset.take(1).get_single_element()
    img_shape = image_x.shape
    x_rows = int(np.ceil(shape[0] / img_shape[0]))
    y_rows = int(np.ceil(shape[1] / img_shape[1]))
    dataset = dataset.batch(x_rows * y_rows)
    return dataset.map(tiles)

def npy_merge (array, shape):
    img_shape = array.shape [1:]
    x_rows = int(np.ceil(shape[0] / img_shape[0]))
    y_rows = int(np.ceil(shape[1] / img_shape[1]))
    array_tiled = np.reshape (array, (-1, x_rows, y_rows, img_shape[0], img_shape[1]))
    array_tiled = np.transpose(array_tiled, axes=[0, 1, 3, 2, 4])
    array_tiled = np.reshape(array_tiled, (-1, x_rows * img_shape[0], y_rows * img_shape[1]))
    return array_tiled[:,:shape[0], :shape[1]]

def get_mask_layer(calexp, mask_name):
    bit_global = calexp.mask.getPlaneBitMask(mask_name)
    return np.where(np.bitwise_and(calexp.mask.array, bit_global), True, False)


def dataset_to_numpy(dataset):
    for i, a in enumerate(dataset):
        s = a.shape
    array = np.empty((i+1)+s)
    for i, a in enumerate(dataset):
        array[i] = a
    return array


def get_asteroid_num(img):
    img=np.copy(img)
    height = img.shape[0]
    width = img.shape[1]
    n_clusters = 0
    while img.sum()!=0:
        roots = np.where(img==1)
        n_clusters += 1
        todo = [(roots[0][0], roots[1][0])]
        visited_pixels = set()
        while todo:
            j, i = todo.pop()
            if (0 <= j < height) and (0 <= i < width) and (img[j, i] > 0):
                visited_pixels.add((j, i))
                img[j, i] = 0
                if not (j + 1, i) in visited_pixels:
                    todo += [(j + 1, i)]
                if not (j - 1, i) in visited_pixels:
                    todo += [(j - 1, i)]
                if not (j, i + 1) in visited_pixels:
                    todo += [(j, i + 1)]
                if not (j, i - 1) in visited_pixels:
                    todo += [(j, i - 1)]
                if not (j - 1, i - 1) in visited_pixels:
                    todo += [(j - 1, i - 1)]
                if not (j + 1, i + 1) in visited_pixels:
                    todo += [(j + 1, i + 1)]
    return n_clusters

def depthfirstsearch(img, root_j, root_i):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros([height, width])
    try:
        _ = iter(root_j)
    except TypeError:
        todo = [(int(root_j), int(root_i))]
    else:
        assert len(root_j) == len(root_i)
        todo = []
        for k in range(len(root_j)):
            todo += [(int(root_j[k]), int(root_i[k]))]
    visited_pixels = set()
    while todo:
        j, i = todo.pop()
        if (0 <= j < height) and (0 <= i < width) and (img[j, i] > 0):
            visited_pixels.add((j, i))
            mask[j, i] = 1
            if not (j + 1, i) in visited_pixels:
                todo += [(j + 1, i)]
            if not (j - 1, i) in visited_pixels:
                todo += [(j - 1, i)]
            if not (j, i + 1) in visited_pixels:
                todo += [(j, i + 1)]
            if not (j, i - 1) in visited_pixels:
                todo += [(j, i - 1)]
            if not (j - 1, i - 1) in visited_pixels:
                todo += [(j - 1, i - 1)]
            if not (j + 1, i + 1) in visited_pixels:
                todo += [(j + 1, i + 1)]
    return mask


def draw_one_line(mask, origin, angle, length, line_thickness=2):
    x0 = origin[0]
    y0 = origin[1]
    x_size = length * np.cos((np.pi / 180) * angle)
    y_size = length * np.sin((np.pi / 180) * angle)
    x1 = x0 - x_size / 2
    y1 = y0 - y_size / 2
    x0 = x0 + x_size / 2
    y0 = y0 + y_size / 2
    line = cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=line_thickness)
    return line


def draw_mask_lines(catalog, calexp):
    mask = np.zeros(calexp.image.array.shape)
    for k in range(len(catalog)):
        origin = calexp.getWcs().skyToPixelArray(np.array([catalog[k]["ra"]]), np.array([catalog[k]["dec"]]),
                                                 degrees=True)
        angle = catalog[k]["beta"]
        length = catalog[k]["trail_length"]
        mask = draw_one_line(mask, origin, angle, length)
    return mask


def one_visit_IO(exp_ref, cat_ref, butler, output_coll, shape=(512, 512)):
    injected_calexp = butler.get("injected_calexp",
                                 dataId=exp_ref.dataId,
                                 collections=output_coll)
    catalog = butler.get("injected_postISRCCD_catalog",
                         dataId=cat_ref.dataId,
                         collections=output_coll)
    mask = draw_mask_lines(catalog, injected_calexp)
    split_injected_calexp = split(injected_calexp.image.array, shape[0], shape[1])
    split_mask = split(mask, shape[0], shape[1])
    return split_injected_calexp, split_mask

def one_iteration(i, exp_ref, cat_ref, butler, output_coll, shape):
    inp, outp = one_visit_IO(exp_ref, cat_ref, butler, output_coll, shape)
    serialized_list = [""] * len(inp)
    counter = 0
    for x, y in zip(inp, outp):
        x = x[:, :, np.newaxis]
        y = y[:, :, np.newaxis].astype(int)
        feature = {}
        feature['x'] = tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten()))
        feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        serialized_list[counter] = serialized
        counter += 1
    return serialized_list

def convert_butler_tfrecords(repo, output_coll, shape, filename_train, filename_test="", train_split=0.25,verbose=True):
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    catalog_ref = list(butler.registry.queryDatasets("injected_postISRCCD_catalog",
                                                     collections=output_coll,
                                                     instrument='HSC'))
    ref = list(butler.registry.queryDatasets("injected_calexp",
                                             collections=output_coll,
                                             instrument='HSC'))
    if train_split < 0 or train_split > 1:
        raise ValueError("train_split must be between 0 and 1")
    elif train_split != 0:
        index = np.arange(0, len(ref))
        np.random.shuffle(index)
        index = index[:int(len(ref) * train_split)]
        if filename_test == "":
            filename_test = repo + "test.tfrecord"
        elif not filename_test.endswith(".tfrecord"):
            filename_test += ".tfrecord"
    else:
        filename_test  = "/dev/null"
        index = []

    if filename_train == "":
        filename_train = repo + "train.tfrecord"
    elif not filename_train.endswith(".tfrecord"):
        filename_train += ".tfrecord"
    batch_size = os.cpu_count() - 1
    counter = 0
    with tf.io.TFRecordWriter(filename_train) as writer_train:
        with tf.io.TFRecordWriter(filename_test) as writer_test:
            while counter < len(ref):
                difference = min(len(ref) - counter, batch_size)
                data_ref = [(i, ref[i], catalog_ref[i], butler, output_coll, shape) for i in range(counter, counter+difference)]
                pool = multiprocessing.Pool(batch_size)
                serialized_tf = pool.starmap(one_iteration, data_ref)
                pool.close()
                pool.join()
                for c, serialized in enumerate(serialized_tf):
                    if counter + c in index:
                        for s in serialized:
                            writer_test.write(s)
                    else:
                        for s in serialized:
                            writer_train.write(s)
                    if verbose:
                        print("\r", counter + c+1, "/", len(ref), end="")
                counter += difference

def convert_butler_numpy(repo, output_coll, shape=(512, 512), parallelize=True):
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    catalog_ref = list(butler.registry.queryDatasets("injected_postISRCCD_catalog",
                                                     collections=output_coll,
                                                     instrument='HSC'))
    ref = list(butler.registry.queryDatasets("injected_calexp",
                                             collections=output_coll,
                                             instrument='HSC'))
    if parallelize:
        data_ref = [(ref[i], catalog_ref[i], butler, output_coll, shape) for i in range(len(ref))]
        pool = multiprocessing.Pool(os.cpu_count() - 1)
        a = pool.starmap(one_visit_IO, data_ref)
        pool.close()
        pool.join()
        inputs = [inp[0] for inp in a]
        outputs = [inp[1] for inp in a]
    else:
        inputs = []
        outputs = []
        print("\r", 0, "/", len(ref), end="")
        for i in range(len(ref)):
            inp, outp = one_visit_IO(ref[i], catalog_ref[i], butler, output_coll, shape)
            inputs.append(inp)
            outputs.append(outp)
            print("\r", i, "/", len(ref), end="")
    return np.concatenate(inputs), np.concatenate(outputs)


def convert_npy_tfrecords(inputs, labels, filename_train, filename_test):
    i = 0
    index = np.arange(0, inputs.shape[0])
    np.random.shuffle(index)
    index = index[:int(inputs.shape[0] * 0.25)]
    with tf.io.TFRecordWriter(filename_train) as writer_train:
        with tf.io.TFRecordWriter(filename_test) as writer_test:
            for X, y in zip(inputs, labels):
                i += 1
                print("\r", i, "/", inputs.shape[0], end="")
                # Feature contains a map of string to feature proto objects
                feature = {}
                feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
                feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))

                # Construct the Example proto object
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize the example to a string
                serialized = example.SerializeToString()

                # write the serialized object to the disk
                if i in index:
                    writer_test.write(serialized)
                else:
                    writer_train.write(serialized)


REPO = "/epyc/ssd/users/kmrakovc/DATA/rc2_subset/SMALL_HSC/"
OUTPUT_COLL = "u/kmrakovc/single_frame_injection_01"