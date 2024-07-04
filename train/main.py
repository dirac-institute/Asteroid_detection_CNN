import argparse
import sys, os
import tensorflow as tf

sys.path.append("../")
import tools.model
import json


def main(args):
    if args.multiworker:
        slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=8000)
        communication = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
        mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver,
                                                                      communication_options=communication)
        print('Number of replicas:', mirrored_strategy.num_replicas_in_sync)
        task_type, task_id = (mirrored_strategy.cluster_resolver.task_type,
                              mirrored_strategy.cluster_resolver.task_id)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        task_type, task_id = (None, None)
    print("GPUS detected:", len(tf.config.list_physical_devices('GPU')))
    with open(args.arhitecture) as f:
        arhitecture = json.load(f)
    if "0" in arhitecture.keys():
        arhitecture = arhitecture["0"]
    if args.model_destination[-6:] != ".keras":
        args.model_destination += ".keras"
    dataset_train = tf.data.TFRecordDataset([args.train_dataset_path])
    tfrecord_shape = tools.model.get_shape_of_quadratic_image_tfrecord(dataset_train)
    dataset_train = dataset_train.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    dataset_val = tf.data.TFRecordDataset([args.test_dataset_path])
    dataset_val = dataset_val.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    with mirrored_strategy.scope():
        if os.path.isfile(args.model_destination):
            model = tf.keras.models.load_model(args.model_destination, compile=False)
        else:
            model = tools.model.unet_model(tfrecord_shape, arhitecture, kernel_size=args.kernel_size)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.start_lr),
                      loss=tools.metrics.FocalTversky(alpha=args.alpha, gamma=args.gamma),
                      metrics=["Precision", "Recall", tools.metrics.F1_Score()])

    if tuple(model.outputs[0].shape[1:]) != tfrecord_shape:
        dataset_train = dataset_train.map(tools.model.reshape_outputs(img_shape=tuple(model.outputs[0].shape[1:-1])))
        dataset_val = dataset_val.map(tools.model.reshape_outputs(img_shape=tuple(model.outputs[0].shape[1:-1])))

    dataset_train = dataset_train.repeat().shuffle(1000).batch(args.batch_size).prefetch(10)
    dataset_val = dataset_val.batch(args.batch_size).prefetch(10)
    if args.multiworker:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset_train = dataset_train.with_options(options)
        dataset_train = mirrored_strategy.experimental_distribute_dataset(dataset_train)

    earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5 * args.decay_lr_patience,
                                                        verbose=1,
                                                        restore_best_weights=True)
    terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
    reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=args.decay_lr_rate,
                                                                patience=2 * args.decay_lr_patience,
                                                                cooldown=args.decay_lr_patience,
                                                                verbose=1)
    checkpoint_kb = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_destination, save_weights_only=False,
                                                       monitor='val_f1_score', mode='max', save_best_only=True)
    if (task_type == 'worker' and task_id == 0) or task_type is None:
        kb = [terminateonnan_kb, reducelronplateau_kb, checkpoint_kb]
    else:
        kb = [terminateonnan_kb, reducelronplateau_kb]
    if args.verbose:
        verbose = 1
    else:
        verbose = 2
    #tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    if (task_type == 'worker' and task_id == 0) or task_type is None:
        results = model.fit(dataset_train, epochs=args.epochs, validation_data=dataset_val, callbacks=kb, verbose=verbose,
                        steps_per_epoch=10000, validation_steps=100)
    else:
        results = model.fit(dataset_train, epochs=args.epochs, callbacks=kb, verbose=0,
                        steps_per_epoch=10000)
    #if (task_type == 'worker' and task_id == 0) or task_type is None:
    #    model.save(args.model_destination)


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

    parser.add_argument('--arhitecture', type=str,
                        default="../DATA/arhitecture_tuned.json",
                        help='Path to a JSON containing definition of an arhitecture.')

    parser.add_argument('--model_destination', type=str,
                        default="../DATA/Trained_model3",
                        help='Path where to save the model once trained.')

    parser.add_argument('-d', '--multiworker', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Use multiworker strategy.')

    parser.add_argument('--kernel_size', type=int,
                        default=3,
                        help='Size of the kernel.')

    parser.add_argument('--merge_operation', type=str,
                        default="concat",
                        help='Merge operation to be used in the model.')

    parser.add_argument('--epochs', type=int,
                        default=8,
                        help='Number of epochs.')

    parser.add_argument('--alpha', type=float,
                        default=0.9,
                        help='Alpha parameter in loss function.')

    parser.add_argument('--gamma', type=float,
                        default=3,
                        help='Gamma parameter in loss function.')

    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='Batch size.')

    parser.add_argument('--start_lr', type=float,
                        default=0.001,
                        help='Initial learning rate.')

    parser.add_argument('--decay_lr_rate', type=float,
                        default=0.75,
                        help='Rate at which to decay the learning rate upon reaching the plateau.')

    parser.add_argument('--decay_lr_patience', type=float,
                        default=2,
                        help='Number of iteration to wait upon reaching the plateau.')

    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Verbose output.')

    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
