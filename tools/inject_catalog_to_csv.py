import sys
from lsst.daf.butler import Butler
import numpy as np
import pandas as pd
import argparse
sys.path.append("../")
import tools.data


def main(args):
    catalog_pandas = tools.data.extract_injection_catalog_to_csv(args.repo_path, args.collection)
    if args.output_csv[-4:] != ".csv":
        args.output_csv += ".csv"
    catalog_pandas.to_csv(args.output_csv)


def parse_arguments(args):
    """Parse command line arguments.
    Args:
        args (list): Command line arguments.
    Returns:
        args (Namespace): Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--repo_path', "--r", type=str,
                        default='/sdf/group/rubin/repo/main',
                        help='Path to butler repository.')
    parser.add_argument('--collection', "--c", type=str,
                        default="u/mrakovci/runs/single_frame_injection_3CD56C01",
                        help='Name of the collection.')
    parser.add_argument('--output_csv', "--o", type=str,
                        default='../DATA/injection_catalog.csv',
                        help='Path to output csv file.')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
