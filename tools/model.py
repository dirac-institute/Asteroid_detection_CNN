import tensorflow as tf
import numpy as np
from tools.attention_module import attach_attention_module


def parse_function(img_shape=(128, 128, 1), test=False, clip=True):
    """

    :param img_shape:
    :param test:
    :param clip:
    :return:
    """
    def parsing(example_proto):
        keys_to_features = {'x': tf.io.FixedLenFeature(shape=img_shape, dtype=tf.float32),
                            'y': tf.io.FixedLenFeature(shape=img_shape, dtype=tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        parsed_features['y'] = tf.cast(parsed_features['y'], tf.float32)
        if clip:
            parsed_features['x'] = tf.clip_by_value(parsed_features['x'], -166.43, 169.96)
        if test:
            return parsed_features['x']
        else:
            return parsed_features['x'], parsed_features['y']

    return parsing


def reshape_outputs(img_shape=(32, 32)):
    def reshaping(inputs, targets):
        targets = tf.image.resize(targets, img_shape)
        targets = tf.math.ceil(targets)
        return inputs, targets

    return reshaping


def get_shape_of_quadratic_image_tfrecord(raw_dataset):
    keys_to_features = {'x': tf.io.VarLenFeature(dtype=tf.float32),
                        'y': tf.io.VarLenFeature(dtype=tf.int64)}
    for i in raw_dataset.take(1):
        parsed_features = tf.io.parse_single_example(i, keys_to_features)
        return int(np.sqrt(parsed_features["x"].shape[0])), int(np.sqrt(parsed_features["x"].shape[0])), 1


def get_architecture_from_model(model):
    """
    Extracts the architecture of a model and returns it as a dictionary.
    :param model: tensorflow model
    :return: dictionary with the architecture
    """
    architecture = {
        "downFilters": [],
        "downActivation": [],
        "downDropout": [],
        "downMaxPool": [],
        "upFilters": [],
        "upActivation": [],
        "upDropout": [], }
    for layer in model.layers:
        if ("block" in layer.name.lower()) and ("conv1" in layer.name.lower()):
            if layer.name.lower()[0] == "e":
                architecture["downFilters"].append(layer.filters)
                architecture["downActivation"].append(layer.activation.__name__)
            elif layer.name.lower()[0] == "d":
                architecture["upFilters"].append(layer.filters)
                architecture["upActivation"].append(layer.activation.__name__)
        elif ("block" in layer.name.lower()) and ("drop" in layer.name.lower()):
            if layer.name.lower()[0] == "e":
                architecture["downDropout"].append(layer.rate)
            elif layer.name.lower()[0] == "d":
                architecture["upDropout"].append(layer.rate)
        elif ("eblock" in layer.name.lower()) and ("pool" in layer.name.lower()):
            current_layer = int(layer.name.lower()[6])
            if len(architecture["downMaxPool"]) < current_layer:
                for i in range(current_layer - len(architecture["downMaxPool"])):
                    architecture["downMaxPool"].append(False)
            architecture["downMaxPool"].append(True)
    return architecture


def attention_gate(g, s, num_filters, kernel_size=1, name=""):
    wg = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", name="attention" + name + "_sconv")(g)
    wg = tf.keras.layers.BatchNormalization(name="attention" + name + "_snorm")(wg)

    ws = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", name="attention" + name + "_gconv")(s)
    ws = tf.keras.layers.BatchNormalization(name="attention" + name + "_gnorm")(ws)

    out = tf.keras.layers.add([wg, ws], name="attention" + name + "_sum")
    #out = tf.keras.layers.BatchNormalization(name="attention" + name + "_sum_norm")(out)
    out = tf.keras.layers.Activation("relu", name="attention" + name + "_relu")(out)
    out = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", name="attention" + name + "_conv")(out)
    out = tf.keras.layers.BatchNormalization(name="attention" + name + "_norm")(out)
    out = tf.keras.layers.Activation("sigmoid", name="attention" + name + "_sigmoid")(out)

    #s = tf.keras.layers.DepthwiseConv2D(kernel_size+2, activation="sigmoid", padding="same", name="attention" + name + "_depthwise")(s)
    out = tf.keras.layers.multiply([out, s], name="attention" + name + "_multiply")
    return out


def encoder_mini_block(inputs, n_filters=32, kernel_size=3, activation="relu", dropout_prob=0.3, max_pooling=True,
                       attention=True, name=""):
    """
    Encoder mini block for U-Net architecture. It consists of two convolutional layers with the same activation function
    and number of filters. Optionally, a dropout layer can be added after the second convolutional layer. If max_pooling
    is set to True, a max pooling layer is added at the end of the block. The skip connection is the output of the second
    convolutional layer.

    :param inputs: Input tensor to the block
    :param n_filters: Number of filters for the convolutional layers
    :param activation: Activation function for the convolutional layers
    :param dropout_prob: Dropout probability for the dropout layer (0 means no dropout)
    :param max_pooling: Boolean to add a max pooling layer at the end of the block
    :param name: Name of the block (Optional)
    :return: The output tensor of the block and the skip connection tensor
    """

    conv = tf.keras.layers.Conv2D(n_filters,
                                  kernel_size,  # filter size
                                  strides=1,
                                  activation="linear",
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="eblock" + name + "_conv1")(inputs)

    conv = tf.keras.layers.BatchNormalization(name="eblock" + name + "_norm1")(conv)
    conv = tf.keras.layers.Activation(activation=activation, name="eblock" + name + "_" + activation + "1")(conv)

    conv = tf.keras.layers.Conv2D(n_filters,
                                  kernel_size,  # filter size
                                  strides=1,
                                  activation="linear",
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="eblock" + name + "_conv2")(conv)
    conv = tf.keras.layers.BatchNormalization(name="eblock" + name + "_norm2")(conv)
    conv = tf.keras.layers.Activation(activation=activation, name="eblock" + name + "_" + activation + "2")(conv)
    if attention:
        conv = attach_attention_module(conv, "cbam_block")

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob, name="eblock" + name + "_dropout")(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="eblock" + name + "_pool")(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection


def decoder_mini_block(prev_layer_input, skip_layer_input=None, n_filters=32, kernel_size=3, activation="relu", dropout_prob=0.3,
                       max_pooling=True, attention=True, name=""):
    """
    Decoder mini block for U-Net architecture that consists of a transposed convolutional layer followed by two
    convolutional layers. The skip connection is the concatenation of the transposed convolutional layer and the
    corresponding encoder skip connection.

    :param prev_layer_input: Input tensor to the block from the previous layer
    :param skip_layer_input: Input tensor to the block from the corresponding encoder skip connection
    :param n_filters: Number of filters for the convolutional layers
    :param activation: Activation function for the convolutional layers
    :param name: Name of the block (Optional)
    :return: The output tensor of the block
    """
    if max_pooling:
        prev_layer_input = tf.keras.layers.Conv2DTranspose(skip_layer_input.shape[-1], strides=2, padding='same',
                                                          kernel_size=kernel_size, name="dblock" + name + "_upsampling")(prev_layer_input)
        prev_layer_input = tf.keras.layers.BatchNormalization(name="dblock" + name + "_norm0")(prev_layer_input)
        prev_layer_input = tf.keras.layers.Activation(activation=activation,
                                                      name="dblock" + name + "_" + activation + "0")(prev_layer_input)

        #prev_layer_input = tf.keras.layers.UpSampling2D(interpolation="bilinear", name="dblock" + name + "_upsampling")(prev_layer_input)
        #prev_layer_input = tf.nn.depth_to_space(prev_layer_input, block_size=2, name="dblock" + name + "_upsampling")
    if skip_layer_input is not None:
        """prev_layer_input = tf.keras.layers.Conv2D(skip_layer_input.shape[-1],
                                                  kernel_size,  # filter size
                                                  strides=1,
                                                  activation="linear",
                                                  padding='same',
                                                  kernel_initializer='HeNormal',
                                                  name="dblock" + name + "_conv0")(prev_layer_input)
        prev_layer_input = tf.keras.layers.BatchNormalization(name="dblock" + name + "_norm0")(prev_layer_input)
        prev_layer_input = tf.keras.layers.Activation(activation=activation,
                                                      name="dblock" + name + "_" + activation + "0")(prev_layer_input)"""

        skip_layer_input = attention_gate(prev_layer_input, skip_layer_input, skip_layer_input.shape[-1], name=name)
        merge = tf.keras.layers.concatenate([prev_layer_input, skip_layer_input], name="dblock" + name + "_merge")
    else:
        merge = prev_layer_input
    conv = tf.keras.layers.Conv2D(n_filters,
                                  kernel_size,  # filter size
                                  strides=1,
                                  activation="linear",
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="dblock" + name + "_conv1")(merge)
    conv = tf.keras.layers.BatchNormalization(name="dblock" + name + "_norm1")(conv)
    conv = tf.keras.layers.Activation(activation=activation, name="dblock" + name + "_" + activation + "1")(conv)

    conv = tf.keras.layers.Conv2D(n_filters,
                                  kernel_size,  # filter size
                                  strides=1,
                                  activation="linear",
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="dblock" + name + "_conv2")(conv)
    conv = tf.keras.layers.BatchNormalization(name="dblock" + name + "_norm2")(conv)
    conv = tf.keras.layers.Activation(activation=activation, name="dblock" + name + "_" + activation + "2")(conv)
    if attention:
        conv = attach_attention_module(conv, "cbam_block")
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob, name="dblock" + name + "_dropout")(conv)

    return conv


def unet_model(input_size, arhitecture, kernel_size=3):
    """
    U-Net model for semantic segmentation. The model consists of an encoder and a decoder. The encoder downsamples the
    input image and extracts features. The decoder upsamples the features and generates the segmentation mask. Skip
    connections are used to concatenate the encoder features with the decoder features. The model is created from the
    architecture dictionary that contains the number of filters, activation functions, dropout probabilities, and max
    pooling for each mini block.

    :param input_size: Size of the input image
    :param arhitecture: Dictionary containing the architecture of the U-Net model
    :return: U-Net model
    """
    inputs = tf.keras.layers.Input(input_size, name="input")
    layer = tf.keras.layers.BatchNormalization(name="input_normalisation")(inputs)
    skip_connections = []
    arhitecture["downMaxPool"][len(arhitecture["downFilters"])-1] = False
    # Encoder
    for i in range(len(arhitecture["downFilters"])):
        layer, skip = encoder_mini_block(layer,
                                         kernel_size=kernel_size,
                                         n_filters=arhitecture["downFilters"][i],
                                         activation=arhitecture["downActivation"][i],
                                         dropout_prob=arhitecture["downDropout"][i],
                                         max_pooling=arhitecture["downMaxPool"][i],
                                         attention=False if i == 0 else True,
                                         name=str(i))
        if i != len(arhitecture["downFilters"])-1:
            skip_connections.append(skip)
        else:
            skip_connections.append(None)

    # Decoder
    for i in range(len(arhitecture["upFilters"])):
        skip_con = skip_connections[len(skip_connections) - 1 - i]
        layer = decoder_mini_block(layer,
                                   skip_con,
                                   kernel_size=kernel_size,
                                   n_filters=arhitecture["upFilters"][i],
                                   activation=arhitecture["upActivation"][i],
                                   attention=True,
                                   dropout_prob=arhitecture["upDropout"][i],
                                   max_pooling=arhitecture["downMaxPool"][len(arhitecture["downMaxPool"]) - 1 - i],
                                   name=str(len(arhitecture["upFilters"]) - 1 - i))

    outputs = tf.keras.layers.Conv2D(1, kernel_size, padding='same', name="output_conv")(layer)
    #outputs = tf.keras.layers.BatchNormalization(name="output_norm")(outputs)
    outputs = tf.keras.layers.Activation(activation="sigmoid", name="output_sigmoid")(outputs)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="AsteroidNET")
    return model


if __name__ == "__main__":
    pass
