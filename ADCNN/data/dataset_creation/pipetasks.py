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
