import time
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append("..")
import os
import argparse
import tools
import evals
import numpy as np
from lsst.daf.butler import Butler


def plot_completeness_magnitude_histogram(table, bins=0, absolute_numbers=False):
    fig, ax = plt.subplots()
    true_ast = table["integrated_mag"]
    if bins == 0:
        # if not user specified, make bins 0.5 wide
        bins = int((np.ceil(np.max(true_ast)) - np.floor(np.min(true_ast))) / 0.5) + 1
    edges = np.linspace(np.floor(np.min(true_ast)), np.ceil(np.max(true_ast)), bins, endpoint=True)
    true_hist, __ = np.histogram(true_ast, bins=edges)
    if "stack_detected" in table.columns:
        stack_predictions = table[table["stack_detected"] == 1]["integrated_mag"]
        stack_hist, __ = np.histogram(stack_predictions, bins=edges)
        if not absolute_numbers:
            stack_c = stack_hist / true_hist
        else:
            stack_c = stack_hist
        ax.stairs(stack_c, edges, label="LSST stack detected asteroids")
    if "NN_detected" in table.columns:
        nn_predictions = table[table["NN_detected"] == 1]["integrated_mag"]
        nn_hist, __ = np.histogram(nn_predictions, bins=edges)
        if not absolute_numbers:
            nn_c = nn_hist / true_hist
        else:
            nn_c = nn_hist
        ax.stairs(nn_c, edges, label="NN detected asteroids")
    if absolute_numbers:
        ax.stairs(true_hist, edges, label="True asteroids")
        ax.set_ylabel("Number")
    else:
        ax.set_ylabel("Completeness")
    ax.set_xlabel("Magnitude")
    _ = ax.set_title("Trail lenght: " +
                     str(int(np.floor(np.min(table["trail_length"])))) +
                     " - " +
                     str(int(np.ceil(np.max(table["trail_length"])))) +
                     " px")
    ax.legend()
    return ax


def main(args):
    if args.val_index_path == "":
        args.val_index_path = None
    collections = args.collection.split(',')
    tf_dataset_paths = args.tf_dataset_path.split(',')
    if len(collections) != len(tf_dataset_paths):
        raise ValueError("Number of collections and TFrecords files should be the same")
    model_name = args.model_path.split("_")[-1].split(".")[0]
    if args.output_path[-1] != "/":
        args.output_path += "/"
    args.output_path += model_name + "/"
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.verbose:
        print("Model evaluating started", flush=True)
    start_time = time.time()
    predictions = evals.eval_tools.create_nn_prediction(tf_dataset_paths,
                                                        args.model_path,
                                                        threshold=args.threshold,
                                                        batch_size=args.batch_size,
                                                        verbose=False)
    if args.verbose:
        print("NN predictions created in", time.time() - start_time, "seconds", flush=True)
    for i in range(len(collections)):
        dataset_name = tf_dataset_paths[i].split("/")[-1].split(".")[0]
        output_path = args.output_path + dataset_name
        table = recovered_sources(args.repo_path, collections[i],
                                  nn_predictions=predictions[i], n_parallel=args.cpu_count)
        ax_magnitude_histogram = plot_completeness_magnitude_histogram(table)


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default="../DATA/Trained_model_56735424.keras",
                        help='Path to the model.')
    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='Batch size for the evaluation.')
    parser.add_argument('--tf_dataset_path', type=str,
                        default="../DATA/test_01.tfrecord,../DATA/test_02.tfrecord,../DATA/test_03.tfrecord,../DATA/test_04.tfrecord",
                        help='Comma-separated list of paths to the TFrecords files.')
    parser.add_argument('--output_path', type=str,
                        default="../RESULTS/",
                        help='Path to the output folder.')
    parser.add_argument('--repo_path', type=str,
                        default="../rc2_subset/SMALL_HSC/",
                        help='Path to the Butler repo.')
    parser.add_argument('--collection', type=str,
                        default="u/kmrakovc/single_frame_injection_01,u/kmrakovc/single_frame_injection_02,u/kmrakovc/single_frame_injection_03,u/kmrakovc/single_frame_injection_04",
                        help='Comma-separated list of collection names in the Butler repo.')
    parser.add_argument('--val_index_path', type=str,
                        default="",
                        help='Path to the validation index file.')
    parser.add_argument('--cpu_count', type=int,
                        default=9,
                        help='Number of CPUs to use.')
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help='Threshold for the predictions.')
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Verbose output.')

    return parser.parse_args(args)
