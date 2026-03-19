import os
import inspect
import lsst.meas.algorithms
from lsst.pipe.tasks.calibrateImage import CalibrateImageTask, CalibrateImageConfig
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
    #cfg.star_detection.thresholdValue = threshold

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
    return result.exposure, result.stars_footprints
