import argparse
import sys
import tensorflow as tf
sys.path.append("../")
import tools.model
import json

def main (args):
    with open(args.arhitecture) as f:
        arhitecture = json.load(f)

    dataset_train = tf.data.TFRecordDataset([args.train_dataset_path])
    tfrecord_shape = tools.model.get_shape_of_quadratic_image_tfrecord(dataset_train)
    dataset_train = dataset_train.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    dataset_train = dataset_train.shuffle(5*args.batch_size).batch(args.batch_size).prefetch(2)
    dataset_val = tf.data.TFRecordDataset([args.test_dataset_path])
    dataset_val = dataset_val.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    dataset_val = dataset_val.batch(args.batch_size).prefetch(2)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    FE = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=args.class_balancing_alpha)
    CE = tf.keras.losses.BinaryCrossentropy()
    with mirrored_strategy.scope():
        model = tools.model.unet_model((128, 128, 1), arhitecture)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.start_lr), loss=[FE, CE],
                      metrics=["Precision", "Recall", tools.model.F1_Score()])
    earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2*args.decay_lr_patience, verbose=1,
                                                        restore_best_weights=True)
    terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
    reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.decay_lr_rate,
                                                                patience=args.decay_lr_patience, verbose=1)
    results = model.fit(dataset_train, epochs=args.epochs, validation_data=dataset_val,
                        callbacks=[earlystopping_kb, terminateonnan_kb, reducelronplateau_kb], verbose=2)
    model.save(args.model_destination)



def parse_arguments(args):
    """Parse command line arguments.
    Args:
        args (list): Command line arguments.
    Returns:
        args (Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str,
                        default='../DATA/train.tfrecord',
                        help='Path to training dataset.')
    parser.add_argument('--test_dataset_path', type=str,
                        default='../DATA/test.tfrecord',
                        help='Path to test dataset.')
    parser.add_argument('--arhitecture', type=str,
                        default="../DATA/arhitecture.json",
                        help='Path to a JSON containing definition of an arhitecture.')
    parser.add_argument('--model_destination', type=str,
                        default="Trained_model",
                        help='Path where to save the model once trained.')
    parser.add_argument('--epochs', type=int,
                        default=150,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=256,
                        help='Batch size.')
    parser.add_argument('--class_balancing_alpha', type=float,
                        default=0.95,
                        help='How much to weight the positive class in the loss function.')
    parser.add_argument('--start_lr', type=float,
                        default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--decay_lr_rate', type=float,
                        default=0.95,
                        help='Rate at which to decay the learning rate upon reaching the plateau.')
    parser.add_argument('--decay_lr_patience', type=float,
                        default=2 * 362,
                        help='Number of iteration to wait upon reaching the plataeau.')
    return parser.parse_args(args)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))