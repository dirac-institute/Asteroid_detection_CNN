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

def source_detect(exposure, input_background, threshold = 5.0, release_id=0):
    cfg = SingleFrameDetectAndMeasureConfig()

    cfg.connections.exposure = "preliminary_visit_image"
    cfg.connections.input_background = "preliminary_visit_image_background"
    cfg.connections.sources = "single_visit_star_reprocessed_unstandardized"
    cfg.connections.sources_footprints = "single_visit_star_reprocessed_footprints"
    cfg.connections.background = "preliminary_visit_image_reprocessed_background"

    cfg.id_generator.release_id = release_id

    cfg.detection.thresholdType = "stdev"
    cfg.detection.thresholdValue = threshold
    cfg.detection.includeThresholdMultiplier = 1.0
    #cfg.detection.reEstimateBackground = True
    #cfg.detection.doTempLocalBackground = True

    #cfg.deblend.maxFootprintArea = -1

    task = SingleFrameDetectAndMeasureTask(config=cfg)
    result = task.run(exposure=exposure, input_background=input_background)
    src = result.sources_footprints
    """
    bad_fields = ["base_PixelFlags_flag_bad", "base_PixelFlags_flag_edge", "base_PixelFlags_flag_interpolated",
                  "base_PixelFlags_flag_interpolatedCenter", "base_PixelFlags_flag_nodata", "base_PixelFlags_flag_cr",
                  "base_PixelFlags_flag_saturated", "base_PixelFlags_flag_saturatedCenter", "base_PixelFlags_flag_suspect"]
    #src = src[src["parent"] == 0]
    for field in bad_fields:
        src = src[src[field] == False]"""
    return src

def calibrate(butler, postISRCCD, dataId, threshold=5.0):
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
    catalog = source_detect(calexp, result.background, threshold = threshold)

    return calexp, catalog, result.background

def fetch_from_butler(butler, dataId, threshold = 5.0):
    calexp = butler.get("preliminary_visit_image", dataId=dataId)
    background = butler.get("preliminary_visit_image_background", dataId=dataId)
    catalog =  source_detect(calexp, background, threshold =threshold)
    return calexp, catalog, background
