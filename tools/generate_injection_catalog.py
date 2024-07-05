import numpy as np
import os
import sys
from astropy.table import QTable, Table, Column, vstack
import multiprocessing
import argparse


def generate_one_line(n_inject, trail_length, mag, beta, butler, ref, input_coll, dimensions, source_type,
                      integrated_magnitude=False):
    injection_catalog = Table(
        names=('injection_id', 'ra', 'dec', 'source_type', 'trail_length', 'mag', 'beta', 'visit', 'integrated_mag', 'physical_filter'),
        dtype=('int64', 'float64', 'float64', 'str', 'float64', 'float64', 'float64', 'int64', 'float64', 'str'))
    injection_catalog.add_index('injection_id')
    raw = butler.get(
        source_type + ".wcs",
        dataId=ref.dataId,
        collections=input_coll,
    )
    info = butler.get(
        source_type + ".visitInfo",
        dataId=ref.dataId,
        collections=input_coll,
    )
    filter_name = butler.get(
        source_type + ".filter",
        dataId=ref.dataId,
        collections=input_coll,
    )
    start = raw.pixelToSky(0, 0)
    end = raw.pixelToSky(dimensions[0], dimensions[1])
    min_ra = start.getRa()
    min_dec = start.getDec()
    max_ra = end.getRa()
    max_dec = end.getDec()
    diff_ra = max_ra - min_ra
    diff_dec = max_dec - min_dec
    min_ra = min_ra + 0.02 * diff_ra
    max_ra = max_ra - 0.02 * diff_ra
    min_dec = min_dec + 0.02 * diff_dec
    max_dec = max_dec - 0.02 * diff_dec
    for k in range(n_inject):
        ra_pos = np.random.uniform(low=min_ra.asDegrees(), high=max_ra.asDegrees())
        dec_pos = np.random.uniform(low=min_dec.asDegrees(), high=max_dec.asDegrees())
        inject_length = np.random.uniform(low=trail_length[0], high=trail_length[1])
        if integrated_magnitude:
            # rolling dice for the magnitude then calculating the surface brightness
            magnitude = np.random.uniform(low=mag[0], high=mag[1])
            surface_brightness = magnitude + 2.5 * np.log10(inject_length)
        else:
            # rolling dice for the surface brightness then calculating the magnitude
            surface_brightness = np.random.uniform(low=mag[0], high=mag[1])
            magnitude = surface_brightness - 2.5 * np.log10(inject_length)
        angle = np.random.uniform(low=beta[0], high=beta[1])
        visitid = info.id
        injection_catalog.add_row([k, ra_pos, dec_pos, "Trail", inject_length, surface_brightness, angle, visitid,
                                   magnitude, filter_name.bandLabel])
    return injection_catalog


def generate_catalog(repo, input_coll, n_inject, trail_length, mag, beta, where="", verbose=True,
                     multiprocess_size=None, integrated_magnitude=False):
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
    :param where:
    :param verbose:
    :return:
    """
    from lsst.daf.butler import Butler
    butler = Butler(repo)
    registry = butler.registry
    source_type = "calexp"
    if where == "":
        query = set(registry.queryDatasets(source_type, collections=input_coll, instrument='HSC', findFirst=True))
    else:
        query = set(registry.queryDatasets(source_type, collections=input_coll, instrument='HSC', where=where, findFirst=True))
    length = len(list(query))
    dimensions = butler.get(
        source_type + ".dimensions",
        dataId=list(query)[0].dataId,
        collections=input_coll,
    )
    parameters = [(n_inject, trail_length, mag, beta, butler, ref, input_coll, dimensions, source_type,
                   integrated_magnitude) for ref in query]
    if verbose:
        print("Number of visits found: ", length)
    if multiprocess_size is None:
        multiprocess_size = max(1, min(os.cpu_count() - 1, len(parameters)))
    if multiprocess_size > 1:
        with multiprocessing.Pool(multiprocess_size) as pool:
            injection_catalog = pool.starmap(generate_one_line, parameters)
    else:
        injection_catalog = [None] * len(parameters)
        for i in range(len(parameters)):
            injection_catalog[i] = generate_one_line(*parameters[i])
            if verbose:
                print("\r", i + 1, "/", length, end="", flush=True)
        if verbose:
            print("")
    output_catalog = vstack(injection_catalog, join_type='exact')
    output_catalog["injection_id"] = np.arange(len(output_catalog))
    return output_catalog


def write_catalog(catalog, repo, output_coll):
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
    for bands in np.unique(catalog['physical_filter'].data):
        _ = ingest_injection_catalog(
            writeable_butler=writeable_butler,
            table=catalog[catalog['physical_filter'] == bands],
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
    catalog = generate_catalog(args.repo, args.input_collection, args.number, args.trail_length, args.magnitude,
                               args.beta, where=args.where, verbose=args.verbose, multiprocess_size=args.cpu_count,
                               integrated_magnitude=args.integrated_magnitude)
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
    parser.add_argument('-r', '--repo', type=str,
                        default='/epyc/ssd/users/kmrakovc/DATA/rc2_subset/SMALL_HSC/',
                        help='Path to Butler repo.')
    parser.add_argument('-i', '--input_collection', type=str,
                        default="u/kmrakovc/RC2_subset/run_1",
                        help='Name of the input collection.')
    parser.add_argument('-o', '--output_collection', type=str,
                        default="u/kmrakovc/single_frame_injection_99",
                        help='Name of the output collection.')
    parser.add_argument('-n', '--number', type=int,
                        default=20,
                        help='Number of injections per visit.')
    parser.add_argument('-l', '--trail_length', nargs=2, type=int,
                        default=[4, 74],
                        help='Lower and upper limit for the injected trails length.')
    parser.add_argument('-m', '--magnitude', nargs=2, type=float,
                        default=[20.1, 27.2],
                        help='Lower and upper limit for the injected trails magnitude.')
    parser.add_argument('-b', '--beta', nargs=2, type=float,
                        default=[0.0, 180.0],
                        help='Lower and upper limit for the injected trails rotation angle.')
    parser.add_argument('--where', type=str,
                        default="",
                        help='Filter the collection.')
    parser.add_argument('--cpu_count', type=int,
                        default=1,
                        help='Number of CPUs to use.')
    parser.add_argument('--integrated_magnitude', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='To use magnitude limits as integrated magnitude.')
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Verbose output.')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
