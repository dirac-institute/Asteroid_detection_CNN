import tensorflow as tf
import numpy as np

def parse_function(img_shape=(128, 128, 1), test=False):
    def parsing(example_proto):
        keys_to_features = {'x':tf.io.FixedLenFeature(shape=img_shape, dtype=tf.float32),
                        'y': tf.io.FixedLenFeature(shape=img_shape, dtype=tf.int64)}
        parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
        parsed_features['y'] = tf.cast(parsed_features['y'], tf.float32)
        parsed_features['x'] = tf.clip_by_value(parsed_features['x'], -100, 100)
        if test:
            return parsed_features['x']
        else:
            return parsed_features['x'], parsed_features['y']
    return parsing

def get_shape_of_quadratic_image_tfrecord(raw_dataset):
    keys_to_features = {'x': tf.io.VarLenFeature(dtype=tf.float32),
                        'y': tf.io.VarLenFeature(dtype=tf.int64)}
    for i in raw_dataset.take(1):
        parsed_features = tf.io.parse_single_example(i, keys_to_features)
        return (int(np.sqrt(parsed_features["x"].shape[0])), int(np.sqrt(parsed_features["x"].shape[0])), 1)


def custom_loss_sum(losses):
    def custom_loss(y_true, y_pred):
        loss = 0
        for i, l in enumerate(losses):
            loss += l(y_true, y_pred)
        return loss
    return custom_loss

class F1_Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.counter = self.add_weight(name='counter', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)
        self.count = self.add_weight(name='F1ScoreCount', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign_add(2 * ((p * r) / (p + r + 1e-6)))
        self.count.assign_add(1)


    def result(self):
        return self.f1/self.count

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)
        self.count.assign(0)

def get_architecture_from_model(model):
    """
    Extracts the architecture of a model and returns it as a dictionary.
    :param model: tensorflow model
    :return: dictionary with the architecture
    """
    architecture = {
        "downFilters":[],
        "downActivation": [],
        "downDropout": [],
        "downMaxPool": [],
        "upFilters": [],
        "upActivation": [],
        "upDropout": []}
    for layer in model.layers:
        if ("block" in layer.name.lower()) and ("conv1" in layer.name.lower()):
            if layer.name.lower()[0]=="e":
                architecture["downFilters"].append(layer.filters)
                architecture["downActivation"].append(layer.activation.__name__)
            elif layer.name.lower()[0]=="d":
                architecture["upFilters"].append(layer.filters)
                architecture["upActivation"].append(layer.activation.__name__)
        elif ("block" in layer.name.lower()) and ("drop" in layer.name.lower()):
            if layer.name.lower()[0]=="e":
                architecture["downDropout"].append(layer.rate)
            elif layer.name.lower()[0]=="d":
                architecture["upDropout"].append(layer.rate)
        elif ("eblock" in layer.name.lower()) and ("pool" in layer.name.lower()):
            current_layer = int(layer.name.lower()[6])
            if len(architecture["downMaxPool"])<current_layer:
                for i in range(current_layer-len(architecture["downMaxPool"])):
                    architecture["downMaxPool"].append(False)
            architecture["downMaxPool"].append(True)
    return architecture
def encoder_mini_block(inputs, n_filters=32, activation="relu", dropout_prob=0.3, max_pooling=True, name=""):
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
    inputs = tf.keras.layers.BatchNormalization(name="eblock" + name + "norm") (inputs)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,  # filter size
                                  activation=activation,
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="eblock" + name + "conv1")(inputs)
    
    """
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,  # filter size
                                  activation=activation,
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="eblock" + name + "conv2")(conv)
    """
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob, name="eblock" + name + "drop")(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="eblock" + name + "pool")(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection


def decoder_mini_block(prev_layer_input, skip_layer_input, n_filters=32, activation="relu", dropout_prob=0.3,
                       max_pooling=True, name=""):
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
        prev_layer_input = tf.keras.layers.Conv2DTranspose(n_filters,
                                         (3, 3),
                                         strides=(2, 2),
                                         padding='same',
                                         name="dblock" + name + "convT")(prev_layer_input)
        #prev_layer_input = tf.keras.layers.UpSampling2D((2, 2), name="dblock" + name + "pool")(prev_layer_input)
    merge = tf.keras.layers.concatenate([prev_layer_input, skip_layer_input], axis=-1, name="dblock" + name + "concat")
    conv = tf.keras.layers.BatchNormalization(name="dblock" + name + "norm")(merge)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,  # filter size
                                  activation=activation,
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="dblock" + name + "conv1")(conv)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob, name="dblock" + name + "drop")(conv)
    """
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,  # filter size
                                  activation=activation,
                                  padding='same',
                                  kernel_initializer='HeNormal',
                                  name="dblock" + name + "conv2")(conv)
    """

    return conv


def unet_model(input_size, arhitecture):
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
    #inputs = tf.keras.layers.Normalization(axis=None, mean=10.653569, variance=170.63159, name="input_normalisation")(inputs)

    #inputs = tf.keras.layers.BatchNormalization(name="input_normalisation")(inputs, training=True)
    skip_connections = []
    layer = inputs
    # Encoder
    for i in range(len(arhitecture["downFilters"])):
        layer, skip = encoder_mini_block(layer,
                                         n_filters=arhitecture["downFilters"][i],
                                         activation=arhitecture["downActivation"][i],
                                         dropout_prob=arhitecture["downDropout"][i],
                                         max_pooling=arhitecture["downMaxPool"][i],
                                         name=str(i))
        skip_connections.append(skip)

    # Decoder
    for i in range(len(arhitecture["upFilters"])):
        layer = decoder_mini_block(layer,
                                   skip_connections[len(arhitecture["upFilters"])-1-i],
                                   n_filters=arhitecture["upFilters"][i],
                                   activation=arhitecture["upActivation"][i],
                                   dropout_prob=arhitecture["upDropout"][i],
                                   max_pooling=arhitecture["downMaxPool"][len(arhitecture["upFilters"])-1-i],
                                   name=str(len(arhitecture["upFilters"])-1-i))

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name="output")(layer)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="AsteroidNET")
    return model


if __name__ == "__main__":
    arhit = {"downFilters":[16, 32, 64],
             "downActivation": ["relu", "relu", "relu"],
             "downDropout": [0.11, 0.12, 0.13],
             "downMaxPool": [True, False, False],
             "upFilters": [64, 32, 16],
             "upActivation": ["relu", "relu", "relu"],
             "upDropout": [0.1, 0.2, 0.3]}
    model = unet_model((128, 128, 1), arhit)
    arhit1 = get_architecture_from_model(model)
    model.summary()
    print (arhit1)
