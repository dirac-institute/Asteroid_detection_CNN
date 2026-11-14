"""Single chokepoint for reading one difference-image panel's pixels.

Every diffim panel the pipeline consumes is sourced from the Butler datastore: the night manifest
(ADCNN.pipelines.usdf_night.night) records the panel's `fits_path` as the datastore artifact URI
(``Butler.getURI(ref)``). This module is the one place that turns such a URI/path back into
pixels, so the rest of the code ("read FITS") is Butler-native by construction:

  * A LOCAL (POSIX / ``file://``) datastore URI is a plain path; we read it directly with
    astropy (it is the Butler datastore's own file by its on-disk path).
  * A REMOTE (e.g. ``s3://`` embargo prompt-processing) URI is fetched IN-MEMORY with
    ``lsst.resources.ResourcePath`` -- LSST's own URI layer, which handles the
    ``profile@bucket`` alias, S3 endpoint and credentials -- and parsed by astropy from a
    BytesIO. Nothing is written to local disk (no clutter), which is the whole point of the
    remote path.

``lsst.resources`` is imported lazily and only on the remote branch, so local reads never need it.
"""
import contextlib
import io


def is_remote(uri_or_path) -> bool:
    """True for a non-local URI (has a scheme other than ``file``). ``s3://...`` -> True;
    ``/sdf/...`` or ``file:///sdf/...`` -> False."""
    p = str(uri_or_path)
    return ("://" in p) and not p.startswith("file://")


def _local_path(p: str) -> str:
    return p[len("file://"):] if p.startswith("file://") else p


def _ensure_s3_botocore_defaults() -> None:
    """Set the always-safe botocore checksum knobs the SLAC S3 gateway needs.

    botocore >= 1.36 defaults to ``CRC*`` request/response checksum calc+validation, which the
    Rubin SDF S3 gateway handles pathologically slowly -- measured **33 s/panel** without these
    vars vs **~0.65 s cold / ~0.05 s warm** with them (a 50x cliff). The LSST stack env exports
    them; a bare torch env does not, so we set them here before the first S3 read. ``setdefault``
    never overrides an explicit operator choice. Endpoint/profile/credentials stay
    environment-supplied (site- and repo-specific: ``S3_ENDPOINT_URL``,
    ``LSST_RESOURCES_S3_PROFILE_<repo>``, ``AWS_SHARED_CREDENTIALS_FILE``)."""
    import os
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "WHEN_REQUIRED")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "WHEN_REQUIRED")


@contextlib.contextmanager
def open_diffim(uri_or_path, memmap: bool = False):
    """Yield an astropy ``HDUList`` for one diffim panel given a Butler datastore URI or path.

    LOCAL / ``file://`` -> ``astropy.io.fits.open(path, memmap=memmap)``.
    REMOTE (``s3://`` ...) -> ResourcePath bytes -> in-memory ``fits.open`` (``memmap`` ignored;
    the bytes are already resident). Callers use it exactly like ``fits.open``::

        with open_diffim(path) as hdul:
            img = hdul[1].data
    """
    from astropy.io import fits
    p = str(uri_or_path)
    if is_remote(p):
        _ensure_s3_botocore_defaults()
        from lsst.resources import ResourcePath
        data = ResourcePath(p).read()
        with fits.open(io.BytesIO(data), memmap=False) as hdul:
            yield hdul
    else:
        with fits.open(_local_path(p), memmap=memmap) as hdul:
            yield hdul
