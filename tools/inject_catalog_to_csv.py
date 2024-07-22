import sys
from lsst.daf.butler import Butler
import numpy as np
import pandas as pd
import argparse


def extract_injection_catalog_to_csv(repo, collection):
    butler = Butler(repo)
    list_catalog = []
    postisrccd_catalog_ref = np.unique(np.array(list(butler.registry.queryDatasets("injected_postISRCCD_catalog",
                                                                                   collections=collection,
                                                                                   instrument='HSC',
                                                                                   findFirst=True))))
    for i, catalog_ref in enumerate(postisrccd_catalog_ref):
        injected_postisrccd_catalog = butler.get("injected_postISRCCD_catalog",
                                             dataId=catalog_ref.dataId,
                                             collections=collection).to_pandas()
        list_catalog.append(injected_postisrccd_catalog)
        print ("\r", i+1, "/", len(postisrccd_catalog_ref), "concatenated", end="")
    return pd.concat(list_catalog).set_index("injection_id").sort_index()


def main(args):
    catalog_pandas = extract_injection_catalog_to_csv(args.repo_path, args.collection)
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
