import argparse
import sys
import tensorflow as tf
sys.path.append("../")
import tools.model
import json

def main (args):
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
    dataset_train = dataset_train.shuffle(5*args.batch_size).batch(args.batch_size).prefetch(2)
    dataset_val = tf.data.TFRecordDataset([args.test_dataset_path])
    dataset_val = dataset_val.map(tools.model.parse_function(img_shape=tfrecord_shape, test=False))
    dataset_val = dataset_val.batch(args.batch_size).prefetch(2)
    with mirrored_strategy.scope():
        model = tools.model.unet_model((128, 128, 1), arhitecture)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.start_lr), loss=tools.metrics.FocalTversky (alpha=0.9, gamma=2),
                      metrics=["Precision", "Recall", tools.metrics.F1_Score()])
    earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5*args.decay_lr_patience, verbose=1,
                                                        restore_best_weights=True)
    terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
    reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.decay_lr_rate,
                                                                patience=args.decay_lr_patience, verbose=1)
    checkpoint_kb = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_destination, save_weights_only=False,
                                                       monitor='val_f1_score', mode='max', save_best_only=True)
    if (task_type == 'worker' and task_id == 0) or task_type is None:
        kb = [terminateonnan_kb, reducelronplateau_kb, checkpoint_kb]
    else:
        kb = [terminateonnan_kb, reducelronplateau_kb]
    results = model.fit(dataset_train, epochs=args.epochs, validation_data=dataset_val,
                        callbacks=kb, verbose=2)
    if (task_type == 'worker' and task_id == 0) or task_type is None:
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
    parser.add_argument('--epochs', type=int,
                        default=64,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='Batch size.')
    parser.add_argument('--start_lr', type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--decay_lr_rate', type=float,
                        default=0.6,
                        help='Rate at which to decay the learning rate upon reaching the plateau.')
    parser.add_argument('--decay_lr_patience', type=float,
                        default=3,
                        help='Number of iteration to wait upon reaching the plateau.')
    return parser.parse_args(args)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
