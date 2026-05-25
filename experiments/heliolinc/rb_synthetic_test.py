"""Does Rubin's real/bogus (RBTransiNet) keep or reject REAL trailed sources?

Definitive synthetic test: inject known trails into real (visit,detector) panels, build the
science / template / difference exposures, and run the DP2 real/bogus model (loaded from the
butler) on cutouts at the injected-trail centroids (TP, known length) vs random/empty positions
(FP). Reports the reliability score of REAL trails by trail length — answering whether running
real/bogus on ADCNN trailed detections would clean FP without killing the trailed TP.

(The run_wide diaSources showed trailed->low reliability, but those may be bogus artefacts; here
the trails are known-real injections, so the score is unambiguous.)

    setup lsst_distrib
    python rb_synthetic_test.py --n-detectors 3 --n-inject 30
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

import lsst.geom as geom
import lsst.afw.table as afwTable
from lsst.geom import Point2D
from lsst.daf.butler import Butler
from lsst.source.injection.inject_exposure import ExposureInjectTask
from lsst.meas.transiNet import RBTransiNetTask, RBTransiNetConfig

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.data.dataset_creation.butler_tasks import fetch_diffim_inputs, run_subtract  # noqa: E402
from ADCNN.utils.helpers import draw_one_line  # noqa: E402
STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP = "lsst_cells_v2"
MODEL_COLL = "pretrained_models/tac_cnn_lsstcam_2026-02-26"


def load_rb(repo_butler):
    payload = repo_butler.get(list(repo_butler.registry.queryDatasets(
        "pretrainedModelPackage", collections=MODEL_COLL, findFirst=False))[0])
    cfg = RBTransiNetConfig(); cfg.modelPackageStorageMode = "butler"
    task = RBTransiNetTask(config=cfg)
    return task, payload


def src_catalog(positions):
    """Minimal SourceCatalog with a centroid slot at the given (x,y) positions."""
    schema = afwTable.SourceTable.makeMinimalSchema()
    ck = afwTable.Point2DKey.addFields(schema, "cen", "centroid", "pixel")
    schema.getAliasMap().set("slot_Centroid", "cen")
    cat = afwTable.SourceCatalog(schema)
    for x, y in positions:
        r = cat.addNew(); r.set(ck, Point2D(float(x), float(y)))
    return cat


def inject(pvi, cat):
    t = ExposureInjectTask()
    return t.run([cat], pvi.clone(), pvi.psf, pvi.photoCalib, pvi.wcs).output_exposure


def _trail_snr(diff_arr, sigma, x, y, L, ang):
    """Matched-filter SNR of the injected trail on the difference image."""
    H, W = diff_arr.shape
    m = np.zeros((H, W), np.uint8); draw_one_line(m, [x, y], ang, L, 1, 2)
    n = int(m.sum())
    if n < 3: return 0.0
    return float(diff_arr[m > 0].sum() / (sigma * np.sqrt(n)))


def one_detector(butler, dataId, rng, n_inject, length_range, mag_range):
    pvi, sources, template, phys, _ = fetch_diffim_inputs(butler, dataId, skymap=SKYMAP, stage3_collection=STAGE3)
    bb = pvi.getBBox(); W, H = bb.getWidth(), bb.getHeight()
    from astropy.table import Table
    rows, truth = [], []
    forbid = np.zeros((H, W), bool)
    for k in range(n_inject):
        L = float(rng.uniform(*length_range)); ang = float(rng.uniform(0, 180)); half = int(L/2)+30
        mag = float(rng.uniform(*mag_range))
        for _ in range(200):
            x = float(rng.uniform(half, W-half)); y = float(rng.uniform(half, H-half))
            tmp = np.zeros((H, W), np.uint8); draw_one_line(tmp, [x, y], ang, L, 1, 4)
            if not ((tmp != 0) & forbid).any():
                forbid |= (tmp != 0); break
        else:
            continue
        sp = pvi.wcs.pixelToSky(x, y)
        rows.append([k, sp.getRa().asDegrees(), sp.getDec().asDegrees(), "Trail", L, mag, ang])
        truth.append(dict(x=x, y=y, trail_length=L, ang=ang, mag=mag))
    inj = Table(rows=rows, names=("injection_id","ra","dec","source_type","trail_length","mag","beta"),
                dtype=("int64","float64","float64","str","float64","float64","float64"))
    pvi_inj = inject(pvi, inj)
    sub = run_subtract(template=template, science=pvi_inj, sources=sources)
    truth = pd.DataFrame(truth)
    # measure each injected trail's SNR on the difference image
    da = sub.difference.image.array.astype(np.float32)
    sig = float(1.4826 * np.median(np.abs(da - np.median(da))))
    truth["snr"] = [_trail_snr(da, sig, r.x, r.y, r.trail_length, r.ang) for r in truth.itertuples()]
    # FP positions: random spots not on an injected trail
    fp = []
    for _ in range(len(truth)):
        fp.append((rng.uniform(60, W-60), rng.uniform(60, H-60)))
    return pvi_inj, sub.matchedTemplate, sub.difference, truth, pd.DataFrame(fp, columns=["x","y"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--where", default="instrument='LSSTCam' AND day_obs>=20250801 AND day_obs<=20250921 AND band='r'")
    ap.add_argument("--n-detectors", type=int, default=3)
    ap.add_argument("--n-inject", type=int, default=30)
    ap.add_argument("--len-min", type=float, default=6.0)
    ap.add_argument("--len-max", type=float, default=30.0, help="cap at ceiling (~30px) to isolate the SNR effect")
    ap.add_argument("--mag-min", type=float, default=20.5)
    ap.add_argument("--mag-max", type=float, default=26.0, help="faint end -> low SNR")
    ap.add_argument("--out", default=str(REPO/"experiments/heliolinc/rb_synth.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    butler = Butler("dp2_prep", collections=[STAGE3, STAGE2])
    task, payload = load_rb(butler)
    refs = list(butler.registry.queryDatasets("preliminary_visit_image", where=a.where, findFirst=True))
    rng.shuffle(refs)
    out, done = [], 0
    for ref in refs:
        if done >= a.n_detectors: break
        did = dict(instrument="LSSTCam", visit=int(ref.dataId["visit"]), detector=int(ref.dataId["detector"]))
        try:
            sci, tmpl, diff, truth, fp = one_detector(butler, did, rng, a.n_inject, (a.len_min, a.len_max), (a.mag_min, a.mag_max))
            cat = src_catalog(list(zip(truth.x, truth.y)) + list(zip(fp.x, fp.y)))
            res = task.run(tmpl, sci, diff, cat, pretrainedModel=payload)
            rel = np.asarray(res.classifications["score"])
        except Exception as e:
            print(f"  skip v={did['visit']} d={did['detector']}: {type(e).__name__}: {e}", flush=True); continue
        nt = len(truth)
        for i in range(nt):
            out.append(dict(**did, tp=True, trail_length=float(truth.trail_length.iloc[i]),
                            snr=float(truth.snr.iloc[i]), mag=float(truth.mag.iloc[i]), reliability=float(rel[i])))
        for j in range(len(fp)):
            out.append(dict(**did, tp=False, trail_length=np.nan, snr=np.nan, mag=np.nan, reliability=float(rel[nt+j])))
        done += 1
        print(f"[{done}/{a.n_detectors}] v={did['visit']} d={did['detector']}: {nt} trails + {len(fp)} FP scored", flush=True)
    df = pd.DataFrame(out); df.to_csv(a.out, index=False)
    if len(df):
        tp, fpp = df[df.tp], df[~df.tp]
        print(f"\n=== real/bogus reliability: {len(tp)} REAL injected trails vs {len(fpp)} FP ===")
        print(f"  injected trails: reliability med {tp.reliability.median():.2f} | rel>=0.5 {100*(tp.reliability>=0.5).mean():.0f}%")
        print(f"  FP/empty       : reliability med {fpp.reliability.median():.2f} | rel>=0.5 {100*(fpp.reliability>=0.5).mean():.0f}%")
        print(f"\n  *** real/bogus KEEP-RATE vs SNR (trails <={a.len_max:.0f}px, inside the length ceiling) ***")
        print(f"  {'SNR bin':12s} {'n':>4s} {'rel med':>8s} {'rel>=0.5 (kept)':>15s}")
        for lo,hi in [(0,5),(5,8),(8,12),(12,20),(20,40),(40,1e9)]:
            s=tp[(tp.snr>=lo)&(tp.snr<hi)]
            if len(s): print(f"  {lo:>4.0f}-{hi if hi<1e8 else 999:<6.0f} {len(s):4d} {s.reliability.median():8.2f} {100*(s.reliability>=0.5).mean():13.0f}%")
    print("RB SYNTH DONE", flush=True)


if __name__ == "__main__":
    main()
