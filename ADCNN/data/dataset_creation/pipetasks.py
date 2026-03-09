from lsst.ip.isr import IsrTaskLSST
from lsst.daf.butler import Butler
from lsst.pipe.tasks.calibrateImage import CalibrateImageTask
from lsst.pipe.tasks.calibrateImage import CalibrateImageConfig
import lsst.afw.image as afwImage

from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.ip.isr import IsrTask


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

    # LSST-specific ISR inputs
    try:
        kwargs["deferredChargeCalib"] = butler.get("cti", dataId=dataId)
    except Exception as e:
        print("No cti -> deferredChargeCalib:", e)

    try:
        kwargs["bfGains"] = butler.get("gain_correction", dataId=dataId)
    except Exception as e:
        print("No gain_correction -> bfGains:", e)

    # Only some repos have this separately
    for name in ("bfKernel", "bfk"):
        try:
            kwargs["bfKernel"] = butler.get(name, dataId=dataId)
            break
        except Exception:
            pass

    cfg = IsrTaskLSST.ConfigClass()
    task = IsrTaskLSST(config=cfg)
    res = task.run(raw, **kwargs)
    return res.exposure

class CalibrateImageNoRefTask(CalibrateImageTask):
    """Use modern CalibrateImageTask internals, but skip astrometry and photometry."""

    def _fit_astrometry(self, exposure, stars, exposure_region=None):
        # Keep whatever WCS is already on the exposure.
        self.metadata["astrometry_matches_count"] = 0
        self.metadata["initial_to_final_wcs"] = float("nan")
        self.metadata["astrom_offset_mean"] = float("nan")
        self.metadata["astrom_offset_std"] = float("nan")
        self.metadata["astrom_offset_median"] = float("nan")
        return [], None

    def _fit_photometry(self, exposure, stars):
        # Keep instrumental units; attach a trivial PhotoCalib if needed.
        photo_calib = exposure.getPhotoCalib()
        if photo_calib is None:
            photo_calib = afwImage.PhotoCalib(1.0, 0.0, bbox=exposure.getBBox())
            exposure.setPhotoCalib(photo_calib)
        self.metadata["photometry_matches_count"] = 0
        return stars, [], None, photo_calib


def calibrate(postISRCCD, threshold=5.0):
    cfg = CalibrateImageConfig()
    cfg.do_calibrate_pixels = False

    # Do not try to persist empty match catalogs
    cfg.optional_outputs = [
        x for x in cfg.optional_outputs
        if x not in ("astrometry_matches", "photometry_matches")
    ]

    # Optional: expose threshold control similar to old CalibrateTask
    cfg.star_detection.thresholdValue = threshold

    task = CalibrateImageNoRefTask(config=cfg)

    # Skip astrometry quality validation entirely
    task.astrometry.check = lambda *args, **kwargs: None

    result = task.run(exposures=[postISRCCD])

    return result.exposure, result.stars, result

def isr_old (butler, dataId):
    raw = butler.get("raw", dataId=dataId)
    linearizer = butler.get("linearizer", dataId=dataId)
    ptc = butler.get("ptc", dataId=dataId)
    dark = butler.get("dark", dataId=dataId)
    bias = butler.get("bias", dataId=dataId)
    crosstalk = butler.get("crosstalk", dataId=dataId)
    defects = butler.get("defects", dataId=dataId)
    flat = butler.get("flat", dataId=dataId)
    cti = butler.get("cti", dataId=dataId)
    camera = butler.get("camera", dataId=dataId)

    cfg = IsrTask.ConfigClass()
    task = IsrTask(config=cfg)
    res = task.run(raw,
                   camera=camera,
                   bias=bias,
                   dark=dark,
                   flat=flat,
                   ptc=ptc,
                   linearizer=linearizer,
                   crosstalk=crosstalk,
                   defects=defects)
    return res.exposure


def characterizeCalibrate(postISRCCD, threshold=5.0):
    char_config = CharacterizeImageTask.ConfigClass()
    char_config.doApCorr = True
    char_config.doDeblend = True
    # Change the detection threshold
    #char_config.detection.thresholdType = "stdev"
    #char_config.detection.thresholdValue = threshold
    char_task = CharacterizeImageTask(config=char_config)
    char_result = char_task.run(postISRCCD)

    calib_config = CalibrateTask.ConfigClass()
    calib_config.doAstrometry = False
    calib_config.doPhotoCal = False
    # Change the detection threshold
    calib_config.detection.thresholdValue = threshold
    calib_task = CalibrateTask(config=calib_config, icSourceSchema=char_result.sourceCat.schema)
    calib_result = calib_task.run(postISRCCD, background=char_result.background, icSourceCat=char_result.sourceCat)
    return calib_result.outputExposure, calib_result.sourceCat
