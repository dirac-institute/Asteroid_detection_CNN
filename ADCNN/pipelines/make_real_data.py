"""ENTRY POINT — produce the REAL-asteroid test diffim dataset from the Butler.

Builds a test set of real LSST difference-image panels containing known asteroid
sightings (from an ephemeris/known-object CSV) plus empty control panels, with the
per-sighting truth (object id, ephemeris RA/Dec, trail length, SNR). Used to
evaluate the trained pipeline on real data (test-only — never trained on).

Thin wrapper around ``ADCNN.data.dataset_creation.build_test_real`` (subcommands:
``scan`` to build the panel manifest, ``build`` to render the diffim h5). Example:

    python -m ADCNN.pipelines.make_real_data scan  --real-csv known_objects.csv --out-dir DATA_DIFFIM/test_real
    python -m ADCNN.pipelines.make_real_data build --manifest DATA_DIFFIM/test_real/manifest.csv --out DATA_DIFFIM/test_real
"""
from ADCNN.data.dataset_creation.build_test_real import main

if __name__ == "__main__":
    main()
