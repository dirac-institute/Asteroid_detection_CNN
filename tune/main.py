import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import argparse
import sys
import keras_tuner as kt
import json
import tensorflow as tf
import time

sys.path.append("../")
import tools


def main(args):
    print("Program started at: ", time.ctime())
    start_time = time.time()
    dataset_train = tf.data.TFRecordDataset([args.train_dataset_path])
    tfrecord_shape = tools.model.get_shape_of_quadratic_image_tfrecord(dataset_train)
    dataset_train = dataset_train.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    dataset_train = dataset_train.shuffle(5 * args.batch_size).batch(args.batch_size).prefetch(2)
    dataset_val = tf.data.TFRecordDataset([args.test_dataset_path])
    dataset_val = dataset_val.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    dataset_val = dataset_val.batch(args.batch_size).prefetch(2)
    strategy = tf.distribute.MirroredStrategy()
    #strategy = tf.distribute.get_strategy()
    FE = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=args.class_balancing_alpha)
    CE = tf.keras.losses.BinaryCrossentropy()
    earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5 * args.decay_lr_patience,
                                                        verbose=1,
                                                        restore_best_weights=True)
    terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
    reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.decay_lr_rate,
                                                                patience=args.decay_lr_patience, verbose=1)
    tuner = kt.Hyperband(hypermodel=tools.hypertuneModels.StockHyperModel(tfrecord_shape, [FE, CE]),
                         objective=kt.Objective('val_f1_score', "max"),
                         max_epochs=args.epochs,
                         factor=int(args.factor),
                         hyperband_iterations=args.hyperband_iterations,
                         directory=args.tuner_destination,
                         overwrite=args.overwrite,
                         project_name="AsteroidsNN_Tuner",
                         distribution_strategy=strategy)

    print("Overhead time: ", time.time() - start_time, " seconds.")
    tuner.search(dataset_train, epochs=args.epochs, verbose=2, validation_data=dataset_val,
                 callbacks=[earlystopping_kb, terminateonnan_kb, reducelronplateau_kb])
    if os.environ.get('KERASTUNER_TUNER_ID', "chief") == "chief":
        best_hps = tools.hypertuneModels.get_best_hyperparameters(tuner, num_trials=10)
        arhitecture = {}
        for j, hyperparameters in enumerate(best_hps):
            best_model = tuner.hypermodel.build(hyperparameters)
            arhitecture[str(j)] = tools.model.get_architecture_from_model(best_model)
            print(arhitecture[str(j)])
        with open(args.arhitecture_destination, 'w') as f:
            json.dump(arhitecture, f)
    print("Program ended at: ", time.ctime(), " with duration: ", time.time() - start_time, " seconds.")


def parse_arguments(args):
    """Parse command line arguments.
    Args:
        args (list): Command line arguments.
    Returns:
        args (Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str,
                        default='../DATA/train1.tfrecord',
                        help='Path to training dataset.')
    parser.add_argument('--test_dataset_path', type=str,
                        default='../DATA/test1.tfrecord',
                        help='Path to test dataset.')
    parser.add_argument('--tuner_destination', type=str,
                        default="../../Tuner/",
                        help='Path where to save the tuner.')
    parser.add_argument('--arhitecture_destination', type=str,
                        default='../DATA/arhitecture_tuned.json',
                        help='Path where to save the tuner.')
    parser.add_argument('-o', '--overwrite', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Overwrite previous data.')
    parser.add_argument('--epochs', type=int,
                        default=32,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='Batch size.')
    parser.add_argument('--class_balancing_alpha', type=float,
                        default=0.95,
                        help='How much to weight the positive class in the loss function.')
    parser.add_argument('--start_lr', type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--decay_lr_rate', type=float,
                        default=0.95,
                        help='Rate at which to decay the learning rate upon reaching the plateau.')
    parser.add_argument('--decay_lr_patience', type=float,
                        default=6,
                        help='Number of iteration to wait upon reaching the plateau.')
    parser.add_argument('--factor', type=int,
                        default=4,
                        help='Number of hyperband rounds.')
    parser.add_argument('--hyperband_iterations', type=int,
                        default=1,
                        help='Repetitions of each full hyperband algorithm.')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
