import os
import inspect
import lsst.meas.algorithms
from lsst.pipe.tasks.calibrateImage import CalibrateImageTask, CalibrateImageConfig
from lsst.drp.tasks.single_frame_detect_and_measure import SingleFrameDetectAndMeasureTask, SingleFrameDetectAndMeasureConfig
from lsst.ap.association.utils import getRegion
from lsst.ip.isr import IsrTaskLSST


def isr(butler, dataId):
    raw = butler.get("raw", dataId=dataId)

    kwargs = dict(
        bias=butler.get("bias", dataId=dataId),
        dark=butler.get("dark", dataId=dataId),
        flat=butler.get("flat", dataId=dataId),
        ptc=butler.get("ptc", dataId=dataId),
        linearizer=butler.get("linearizer", dataId=dataId),
        crosstalk=butler.get("crosstalk", dataId=dataId),
        defects=butler.get("defects", dataId=dataId),
        camera=butler.get("camera", dataId=dataId),
    )

    try:
        kwargs["deferredChargeCalib"] = butler.get("cti", dataId=dataId)
    except Exception:
        pass

    try:
        kwargs["gainCorrection"] = butler.get("gain_correction", dataId=dataId)
    except Exception:
        pass

    for name in ("bfKernel", "bfk"):
        try:
            kwargs["bfKernel"] = butler.get(name, dataId=dataId)
            break
        except Exception:
            pass

    for name in ("electroBFDistortionMatrix", "electroBfDistortionMatrix"):
        try:
            kwargs["electroBfDistortionMatrix"] = butler.get(name, dataId=dataId)
            break
        except Exception:
            pass

    cfg = IsrTaskLSST.ConfigClass()
    cfg.ampOffset.doApplyAmpOffset = True
    task = IsrTaskLSST(config=cfg)

    allowed = set(inspect.signature(task.run).parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    res = task.run(raw, **kwargs)
    return res.exposure

def _get_schema_names(catalog):
    try:
        return list(catalog.schema.getNames())
    except Exception:
        return [item.field.getName() for item in catalog.schema]

def catalog_to_pandas(catalog, measueTrails=False):
    df = catalog.to_pandas()
    if measueTrails:
        trail_fields = [
            name for name in _get_schema_names(catalog)
            if name.startswith("ext_trailedSources_Naive_") or name.startswith("ext_trailedSources_Veres_")
        ]
        for name in trail_fields:
            if name not in df.columns:
                df[name] = [record[name] for record in catalog]
    return df

def source_detect(exposure, input_background, threshold = 5.0, release_id=0, measueTrails=False):
    if measueTrails:
        try:
            import lsst.meas.extensions.trailedSources  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "measueTrails=True requires lsst.meas.extensions.trailedSources to be set up."
            ) from e

    cfg = SingleFrameDetectAndMeasureConfig()

    cfg.connections.exposure = "preliminary_visit_image"
    cfg.connections.input_background = "preliminary_visit_image_background"
    cfg.connections.sources = "single_visit_star_reprocessed_unstandardized"
    cfg.connections.sources_footprints = "single_visit_star_reprocessed_footprints"
    cfg.connections.background = "preliminary_visit_image_reprocessed_background"

    cfg.id_generator.release_id = release_id

    cfg.detection.thresholdValue = threshold
    cfg.detection.includeThresholdMultiplier = 1.0
    cfg.detection.reEstimateBackground = True
    cfg.detection.doTempLocalBackground = True

    cfg.deblend.maxFootprintArea = -1
    if measueTrails:
        for plugin_name in (
            "base_SdssCentroid",
            "base_SdssShape",
            "ext_trailedSources_Naive",
            "ext_trailedSources_Veres",
        ):
            cfg.measurement.plugins.names.add(plugin_name)
        cfg.measurement.slots.centroid = "base_SdssCentroid"
        cfg.measurement.slots.shape = "base_SdssShape"

    task = SingleFrameDetectAndMeasureTask(config=cfg)
    result = task.run(exposure=exposure, input_background=input_background)
    return result.sources_footprints

def calibrate(butler, postISRCCD, dataId, threshold=5.0, measueTrails=False):
    expanded = butler.registry.expandDataId(dataId)
    exposure_record = expanded.records["exposure"]

    cfg = CalibrateImageConfig()
    cfg.load(os.path.expandvars("$DRP_PIPE_DIR/config/calibrateImage.py"))

    cfg.connections.exposures = "post_isr_image"
    cfg.connections.stars_footprints = "single_visit_star_footprints"
    cfg.connections.psf_stars_footprints = "single_visit_psf_star_footprints"
    cfg.connections.psf_stars = "single_visit_psf_star"
    cfg.connections.initial_stars_schema = "single_visit_star_schema"
    cfg.connections.stars = "single_visit_star_unstandardized"
    cfg.connections.exposure = "preliminary_visit_image"
    cfg.connections.mask = "preliminary_visit_mask"
    cfg.connections.background = "preliminary_visit_image_background"

    cfg.useButlerCamera = True
    cfg.astrometry.matcher.maxOffsetPix = 800
    cfg.astrometry_ref_loader.pixelMargin = 800

    cfg.connections.astrometry_ref_cat = "the_monster_20250219"
    cfg.connections.photometry_ref_cat = "the_monster_20250219"

    cfg.photometry_ref_loader.filterMap = {
        "u": "monster_ComCam_u",
        "g": "monster_ComCam_g",
        "r": "monster_ComCam_r",
        "i": "monster_ComCam_i",
        "z": "monster_ComCam_z",
        "y": "monster_ComCam_y",
    }
    cfg.photometry.applyColorTerms = False
    cfg.photometry.photoCatName = "the_monster_20250219"

    cfg.do_calibrate_pixels = False

    task = CalibrateImageTask(config=cfg)

    region = getRegion(postISRCCD)

    dt = butler.get_dataset_type("the_monster_20250219")
    dims = tuple(dt.dimensions.names)
    skypix_dim = [d for d in dims if d.startswith("htm") or d.startswith("healpix")][0]
    where = f"{skypix_dim}.region OVERLAPS :region"

    refs = list(
        butler.query_datasets(
            "the_monster_20250219",
            collections="refcats",
            where=where,
            bind={"region": region},
            find_first=False,
            with_dimension_records=True,
        )
    )
    if not refs:
        raise RuntimeError("No overlapping refcat shards found.")

    astrometry_loader = lsst.meas.algorithms.ReferenceObjectLoader(
        dataIds=[ref.dataId for ref in refs],
        refCats=[butler.getDeferred(ref) for ref in refs],
        name=cfg.connections.astrometry_ref_cat,
        config=cfg.astrometry_ref_loader,
        log=task.log,
    )
    task.astrometry.setRefObjLoader(astrometry_loader)

    photometry_loader = lsst.meas.algorithms.ReferenceObjectLoader(
        dataIds=[ref.dataId for ref in refs],
        refCats=[butler.getDeferred(ref) for ref in refs],
        name=cfg.connections.photometry_ref_cat,
        config=cfg.photometry_ref_loader,
        log=task.log,
    )
    task.photometry.match.setRefObjLoader(photometry_loader)

    result = task.run(
        exposures=[postISRCCD],
        id_generator=cfg.id_generator.apply(expanded),
        camera_model=butler.get("astrometry_camera", dataId=expanded),
        exposure_record=exposure_record,
        exposure_region=expanded.region,
    )

    calexp = result.exposure
    catalog = source_detect(calexp, result.background, threshold = threshold, measueTrails=measueTrails)

    return calexp, catalog, result.background

def fetch_from_butler(butler, dataId, threshold = 5.0, measueTrails=False):
    calexp = butler.get("preliminary_visit_image", dataId=dataId)
    background = butler.get("preliminary_visit_image_background", dataId=dataId)
    catalog =  source_detect(calexp, background, threshold =threshold, measueTrails=measueTrails)
    return calexp, catalog, background


# ======================================================================================
# Diffim primitives (template fetch + AlardLupton subtract + DIA detect)
# ======================================================================================
#
# Used by simulate.py. The pattern mirrors the proven flow in
# experiments/diffim/stage1_generate/driver.py, with two stack-specific quirks
# baked in:
#   1) `band=:band` bind on template_coadd queries silently no-ops on this
#      stack — must filter Python-side via `r.dataId.get("band") == band`.
#   2) The OVERLAPS clause must use `patch.region`, NOT `template_coadd.region`
#      (region lives on the skypix dim, not the dataset type).

def get_template(butler, pvi, skymap, physical_filter, stage3_collection=None):
    """Fetch all overlapping same-band template_coadd patches and assemble a
    template Exposure aligned to the science PVI bbox/wcs.

    Returns:
        (template_exposure, n_template_refs)
    """
    import lsst.geom
    from lsst.ip.diffim.getTemplate import GetTemplateTask, GetTemplateConfig
    from lsst.sphgeom import ConvexPolygon

    band = pvi.filter.bandLabel
    wcs = pvi.wcs
    bbox = pvi.getBBox()
    corners_pix = [lsst.geom.Point2D(c) for c in bbox.getCorners()]
    corners_sky = [wcs.pixelToSky(p) for p in corners_pix]
    region = ConvexPolygon([c.getVector() for c in corners_sky])

    query_kwargs = dict(
        where="skymap = :skymap AND patch.region OVERLAPS :region",
        bind={"skymap": skymap, "region": region},
        findFirst=True,
    )
    if stage3_collection is not None:
        query_kwargs["collections"] = (
            [stage3_collection] if isinstance(stage3_collection, str) else list(stage3_collection)
        )

    all_refs = list(butler.registry.queryDatasets("template_coadd", **query_kwargs))
    refs = [r for r in all_refs if r.dataId.get("band") == band]
    if not refs:
        bands_seen = sorted({r.dataId.get("band") for r in all_refs})
        raise RuntimeError(
            f"No {band}-band template_coadd overlapping science visit "
            f"(raw_overlap={len(all_refs)}, bands_seen={bands_seen})"
        )

    by_tract: dict = {}
    dataIds: dict = {}
    for ref in refs:
        tract = int(ref.dataId["tract"])
        by_tract.setdefault(tract, []).append(butler.getDeferred(ref))
        dataIds.setdefault(tract, []).append(ref.dataId)

    cfg = GetTemplateConfig()
    task = GetTemplateTask(config=cfg)
    out = task.run(
        coaddExposureHandles=by_tract,
        bbox=bbox,
        wcs=wcs,
        dataIds=dataIds,
        physical_filter=physical_filter,
    )
    return out.template, len(refs)


def run_subtract(template, science, sources):
    """AlardLupton PSF-matched subtraction. `sources` should be the
    single_visit_star_footprints catalog (kernel candidates).

    Returns the SubtractTask Struct with: difference, matchedTemplate, ...
    """
    from lsst.ip.diffim.subtractImages import (
        AlardLuptonSubtractTask,
        AlardLuptonSubtractConfig,
    )
    cfg = AlardLuptonSubtractConfig()
    cfg.doApplyExternalCalibrations = False
    task = AlardLuptonSubtractTask(config=cfg)
    return task.run(template=template, science=science, sources=sources)


def run_detect_diffim(science, matchedTemplate, difference, threshold=5.0, measueTrails=False):
    """DetectAndMeasure on the difference image. Returns the task Struct;
    use `.diaSources` for the catalog. Schema is API-compatible with
    SingleFrameDetectAndMeasureTask outputs (footprints, centroids, fluxes),
    so existing code that calls `.getFootprint()` / `.getCentroid()` works
    unchanged.
    """
    if measueTrails:
        try:
            import lsst.meas.extensions.trailedSources  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "measueTrails=True requires lsst.meas.extensions.trailedSources to be set up."
            ) from e

    from lsst.ip.diffim.detectAndMeasure import (
        DetectAndMeasureTask,
        DetectAndMeasureConfig,
    )
    cfg = DetectAndMeasureConfig()
    cfg.doSkySources = False
    cfg.detection.thresholdValue = float(threshold)
    if measueTrails:
        for plugin_name in (
            "base_SdssCentroid",
            "base_SdssShape",
            "ext_trailedSources_Naive",
            "ext_trailedSources_Veres",
        ):
            cfg.measurement.plugins.names.add(plugin_name)
        cfg.measurement.slots.centroid = "base_SdssCentroid"
        cfg.measurement.slots.shape = "base_SdssShape"

    task = DetectAndMeasureTask(config=cfg)
    return task.run(science=science, matchedTemplate=matchedTemplate, difference=difference)


def fetch_diffim_inputs(butler, dataId, skymap, stage3_collection=None):
    """Fetch the science PVI, the kernel-candidate source list, and the
    template Exposure for one (visit, detector). Caller runs subtract +
    detect themselves (typically twice: clean and injected).

    Returns:
        (pvi, sources, template, physical_filter, n_template_refs)
    """
    pvi = butler.get("preliminary_visit_image", dataId=dataId)
    sources = butler.get("single_visit_star_footprints", dataId=dataId)
    physical_filter = pvi.filter.physicalLabel
    template, n_refs = get_template(butler, pvi, skymap, physical_filter, stage3_collection=stage3_collection)
    return pvi, sources, template, physical_filter, n_refs
