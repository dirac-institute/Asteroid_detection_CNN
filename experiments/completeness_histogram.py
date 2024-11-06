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
import pandas as pd

def plot_completeness_histogram(table, column_name, x_name="", bins=0, absolute_numbers=False):
    fig, ax = plt.subplots()
    true_ast = table[column_name].array

    # Calculate the number of bins if not specified
    if bins == 0:
        if "mag" in column_name:
            bins = int((np.ceil(np.max(true_ast)) - np.floor(np.min(true_ast))) / 0.5) + 1
        else:
            bins = int((np.ceil(np.max(true_ast)) - np.floor(np.min(true_ast))) / 5) + 1

    # Define the edges of the bins
    edges = np.linspace(np.floor(np.min(true_ast)), np.ceil(np.max(true_ast)), bins, endpoint=True)

    # Calculate the histogram for the true values
    true_hist, _ = np.histogram(true_ast, bins=edges)

    # Create a mask for non-zero bins to avoid division by zero
    non_zero_mask = true_hist > 0
    filtered_edges = edges[:-1][non_zero_mask]  # Exclude the last edge to match histogram shape
    filtered_edges = np.append(filtered_edges, edges[-1])  # Add the last edge back

    if "stack_detected" in table.columns:
        stack_predictions = table[table["stack_detected"] == 1][column_name]
        stack_hist, _ = np.histogram(stack_predictions, bins=edges)
        stack_hist = stack_hist[non_zero_mask]  # Filter out zero bins
        if not absolute_numbers:
            stack_c = stack_hist / true_hist[non_zero_mask]
        else:
            stack_c = stack_hist
        ax.stairs(stack_c, filtered_edges, label="LSST stack detected")

    if "NN_detected" in table.columns:
        nn_predictions = table[table["NN_detected"] == 1][column_name]
        nn_hist, _ = np.histogram(nn_predictions, bins=edges)
        nn_hist = nn_hist[non_zero_mask]  # Filter out zero bins
        if not absolute_numbers:
            nn_c = nn_hist / true_hist[non_zero_mask]
        else:
            nn_c = nn_hist
        ax.stairs(nn_c, filtered_edges, label="NN detected")

    if absolute_numbers:
        ax.stairs(true_hist[non_zero_mask], filtered_edges, label="True")
        ax.set_ylabel("Number")
    elif ("NN_detected" in table.columns) and ("stack_detected" in table.columns):
        both_predictions = table[(table["stack_detected"] == 1) | (table["NN_detected"] == 1)][column_name]
        both_hist, _ = np.histogram(both_predictions, bins=edges)
        both_hist = both_hist[non_zero_mask]  # Filter out zero bins
        both_c = both_hist / true_hist[non_zero_mask]
        ax.stairs(both_c, filtered_edges, label="NN or LSST stack")
        ax.set_ylabel("Completeness")
    else:
        ax.set_ylabel("Completeness")

    ax.set_xlabel(x_name)
    ax.legend()
    return fig, ax


def create_prediction_table(args):
    collections = args.collection.split(',')
    tf_dataset_paths = args.tf_dataset_path.split(',')
    if len(collections) != len(tf_dataset_paths):
        raise ValueError("Number of collections and TFrecords files should be the same")
    if args.val_index_path == "":
        val_index = [None for i in range(len(collections))]
    else:
        val_index = []
        for path in  args.val_index_path.split(','):
            val_index.append(np.load(path))
        if len(val_index) != len(collections):
            raise ValueError("Number of validation index files and collections should be the same")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.verbose:
        print("Model evaluating started", flush=True)
    start_time = time.time()
    predictions = evals.eval_tools.create_nn_prediction(tf_dataset_paths,
                                                        args.model_path,
                                                        threshold=args.threshold,
                                                        batch_size=args.batch_size,
                                                        verbose=True)
    if args.verbose:
        print("NN predictions created in", round(time.time() - start_time, 2), "seconds", flush=True)
    table = ["" for i in range(len(collections))]
    for i in range(len(collections)):
        dataset_name = tf_dataset_paths[i].split("/")[-1].split(".")[0]
        output_path = args.output_path + dataset_name
        if not os.path.exists(args.cutouts_path + dataset_name + "/"):
            os.makedirs(args.cutouts_path + dataset_name + "/")
        table[i] = evals.eval_tools.recovered_sources(args.repo_path, collections[i],
                                                   nn_predictions=predictions[i],
                                                   n_parallel=args.cpu_count,
                                                   val_index=val_index[i],
                                                   cutouts_path=args.cutouts_path + dataset_name + "/",)
        table[i].to_csv(output_path + "_prediction_table.csv")
    return table


def check_if_prediction_exist(args):
    collections = args.collection.split(',')
    tf_dataset_paths = args.tf_dataset_path.split(',')
    #model_name = args.model_path.split("_")[-1].split(".")[0]
    for i in range(len(collections)):
        dataset_name = tf_dataset_paths[i].split("/")[-1].split(".")[0]
        output_path = args.output_path + dataset_name
        if not os.path.exists(output_path + "_prediction_table.csv"):
            return False
    return True



def main(args):
    model_name = args.model_path.split("_")[-1].split(".")[0]
    if args.output_path[-1] != "/":
        args.output_path += "/"
    if args.cutouts_path[-1] != "/":
        args.cutouts_path += "/"
    args.output_path += model_name + "/"
    if args.predict or not check_if_prediction_exist(args):
        tables = create_prediction_table(args)
    collections = args.collection.split(',')
    tf_dataset_paths = args.tf_dataset_path.split(',')
    for i in range(len(collections)):
        dataset_name = tf_dataset_paths[i].split("/")[-1].split(".")[0]
        output_path = args.output_path + "/" + dataset_name
        table = pd.read_csv(output_path + "_prediction_table.csv")
        if i == 0:
            bin_number = 0
        else:
            bin_number = 10
        fig, ax = plot_completeness_histogram(table, column_name="integrated_mag", x_name="Integrated magnitude", bins=bin_number)
        _ = ax.set_title("Trail length: " +
                         str(int(np.floor(np.min(table["trail_length"])))) +
                         " - " +
                         str(int(np.ceil(np.max(table["trail_length"])))) +
                         " px")
        fig.savefig(output_path + "_magnitude_hist.png")
        fig, ax = plot_completeness_histogram(table, column_name="trail_length", x_name="Trail length [px]", bins=0)
        _ = ax.set_title("Integrated magnitude: " +
                         str(int(np.floor(np.min(table["integrated_mag"])))) +
                         " - " +
                         str(int(np.ceil(np.max(table["integrated_mag"])))))
        fig.savefig(output_path + "_trail_length_hist.png")





def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='If set, the model will be used to make predictions. Otherwise, the predictions will be loaded from the output folder.')
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
                        help='Path to the output for the evaluation results.')
    parser.add_argument('--cutouts_path', type=str,
                        default="/astro/users/kmrakovc/epyc/users/kmrakovc/public_html/cutouts",
                        help='Path to the cutouts.')
    parser.add_argument('--repo_path', type=str,
                        default="/epyc/ssd/users/kmrakovc/DATA/rc2_subset/SMALL_HSC/",
                        help='Path to the Butler repo.')
    parser.add_argument('--collection', type=str,
                        default="u/kmrakovc/runs/single_frame_injection_01,u/kmrakovc/runs/single_frame_injection_02,u/kmrakovc/runs/single_frame_injection_03,u/kmrakovc/runs/single_frame_injection_04",
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
                        help='If set, the program will print more information.')
                        

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
    #print (check_if_prediction_exist(args))