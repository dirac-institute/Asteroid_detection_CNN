import numpy as np
import os
import sys
from astropy.table import QTable, Table, Column
from matplotlib import pyplot as plt
import argparse

def generate_catalog (repo, input_coll, n_inject, trail_length, mag, beta, verbose = True):
    """
    Create a catalog of trails to be injected in the input collection for the source injection. The catalog is saved in the
    astropy table format. The catalog is created by randomly selecting a position in the input collection and then
    randomly selecting the trail length, magnitude and angle of the trail. The catalog will have a visit id for each trail
    that is the same as the visit id of the calexp that the trail is injected into.

    :param repo:
    :param input_coll:
    :param n_inject:
    :param trail_length:
    :param mag:
    :param beta:
    :return:
    """
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    registry = butler.registry
    ra = []
    dec = []
    last_id =0
    injection_catalog = Table(names=('injection_id', 'ra', 'dec', 'source_type', 'trail_length', 'mag', 'beta', 'visit'),
                              dtype=('int64', 'float64', 'float64', 'str', 'int64', 'float64', 'float64', 'int64'))
    length = len(list(registry.queryDatasets("calexp", collections=input_coll, instrument='HSC')))
    for i ,ref in enumerate(registry.queryDatasets("calexp", collections=input_coll, instrument='HSC')):
        raw = butler.get(
            "calexp",
            dataId=ref.dataId,
            collections=input_coll,
        )
        start = raw.wcs.pixelToSky(0 ,0)
        end = raw.wcs.pixelToSky(raw.getDimensions()[0], raw.getDimensions()[1])
        min_ra = start.getRa()
        min_dec = start.getDec()
        max_ra = end.getRa()
        max_dec = end.getDec()
        if not ([min_ra, max_ra] in ra) and not ([min_dec, max_dec] in dec):
            ra.append([min_ra, max_ra])
            dec.append([min_dec, max_dec])
            for k in range(last_id, last_id +n_inject):
                ra_pos = np.random.uniform (low=min_ra.asDegrees(), high=max_ra.asDegrees())
                dec_pos = np.random.uniform (low=min_dec.asDegrees(), high=max_dec.asDegrees())
                inject_length = np.random.uniform (low=trail_length[0], high=trail_length[1])
                magnitude = np.random.uniform (low=mag[0], high=mag[1])
                angle = np.random.uniform (low=beta[0], high=beta[1])
                visitid = raw.getInfo().getVisitInfo().id
                injection_catalog.add_row([k, ra_pos, dec_pos, "Trail", inject_length, magnitude, angle, visitid])
            last_id = k
        if verbose:
            print("\r" , i +1, "/", length, end="")
    if verbose:
        print("")
    return injection_catalog

def write_catalog (catalog, repo, output_coll):
    """
    Write the catalog to the output collection in the repo.

    :param catalog:
    :param repo:
    :param output_coll:
    :return:
    """
    from lsst.daf.butler import Butler
    from lsst.source.injection import ingest_injection_catalog
    writeable_butler = Butler(repo, writeable=True)
    for bands in ["g", "r", "i", "z", "y"]:
        my_injected_datasetRefs = ingest_injection_catalog(
            writeable_butler=writeable_butler,
            table=catalog,
            band=bands,
            output_collection=output_coll)
    return None

def main(args):
    """
    Main function that generates the catalog of trails to be injected in the input collection and writes it to the output
    collection.

    :param args:
    :return:
    """
    catalog = generate_catalog(args.repo, args.input_collection, args.number, args.trail_length, args.magnitude, args.beta)
    write_catalog(catalog, args.repo, args.output_collection)
    return None

def parse_arguments(args):
    """Parse command line arguments.
    Args:
        args (list): Command line arguments.
    Returns:
        args (Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--repo', type=str,
                        default='/epyc/ssd/users/kmrakovc/DATA/rc2_subset/SMALL_HSC/',
                        help='Path to Butler repo.')
    parser.add_argument('-i','--input_collection', type=str,
                        default="u/kmrakovc/RC2_subset/run_1",
                        help='Name of the input collection.')
    parser.add_argument('-o','--output_collection', type=str,
                        default="u/kmrakovc/single_frame_injection_99",
                        help='Name of the output collection.')
    parser.add_argument('-n', '--number', type=int,
                        default=20,
                        help='Number of injections per visit.')
    parser.add_argument('-l','--trail_length', nargs=2 , type=int,
                        default=[4, 74],
                        help='Lower and upper limit for the injected trails length.')
    parser.add_argument('-m','--magnitude', nargs=2 , type=float,
                        default=[20.1, 27.2],
                        help='Lower and upper limit for the injected trails magnitude.')
    parser.add_argument('-b','--beta', nargs=2 , type=float,
                        default=[0.0, 180.0],
                        help='Lower and upper limit for the injected trails rotation angle.')
    return parser.parse_args(args)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))