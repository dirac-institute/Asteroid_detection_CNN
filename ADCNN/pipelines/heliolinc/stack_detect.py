"""Rubin "stack" 5-sigma source detection on the (inline-injected) difference images.

The user's pipeline records BOTH detectors on the SAME stored inputs: ADCNN (discover_stream, GPU) and the
stack's own 5-sigma detection (here, CPU). We run `lsst.meas.algorithms.SourceDetectionTask` at 5 sigma
directly on each difference `ExposureF` -- the lightweight, defensible "what the 5-sigma stack pipeline
detects" (the official dia_source_visit uses the same detection upstream of its reliability score; we don't
need templates/science, so this scales to ~440k panels). Injection is applied INLINE with the identical
catalog + seed ADCNN uses (inject_trails.add_trails), so both detectors see byte-identical pixels.

Output stack_dets.csv: visit, detector, mjd, x, y, ra, dec, peak, snr, is_pos -- one row per 5-sigma peak.
Later steps crossmatch this to the ADCNN catalog ("stack vs ADCNN") and to the injection truth (does the
5-sigma stack recover the faint SNR~2 movers? -- expected NO, which is ADCNN's whole point).
"""
from __future__ import annotations
import argparse
import os
import warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0, "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")  # ADCNN importable in LSST env
import numpy as np
import pandas as pd

_TASK = None  # per-worker SourceDetectionTask (built once)


def _get_task(threshold):
    global _TASK
    if _TASK is None:
        import lsst.afw.table as afwTable
        from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
        cfg = SourceDetectionConfig()
        cfg.thresholdValue = float(threshold)
        cfg.thresholdType = "pixel_stdev"      # use the diffim variance plane (per-pixel sigma)
        cfg.thresholdPolarity = "positive"     # moving sources are positive in the difference
        cfg.reEstimateBackground = False       # a difference image is already background-subtracted
        schema = afwTable.SourceTable.makeMinimalSchema()
        _TASK = (SourceDetectionTask(schema=schema, config=cfg), schema, afwTable)
    return _TASK


def _panel(args):
    """Run 5sigma detection, then emit LEAN records: one per-panel FP-density summary row + one row per
    injected sighting on this panel (did the 5sigma stack recover it, at what SNR). Raw FP positions
    (~800/panel of subtraction noise) are NOT dumped -- only their count, the useful quantity."""
    fits_path, recs, inject_rows, threshold, tol, full = args
    from scipy.spatial import cKDTree
    import lsst.afw.image as afwImage
    v, det = recs["visit"], recs["detector"]
    try:
        exp = afwImage.ExposureF.readFits(fits_path)
        if inject_rows:                                  # identical inline injection to ADCNN
            from ADCNN.pipelines.heliolinc.inject_trails import add_trails
            exp.image.array[:] = add_trails(np.array(exp.image.array, copy=True), inject_rows)
        task, schema, afwTable = _get_task(threshold)
        table = afwTable.SourceTable.make(schema)
        res = task.run(table, exp)
        var = exp.variance.array; H, W = var.shape
        wcs = exp.getWcs() if full else None
        try:
            mjd = float(exp.getInfo().getVisitInfo().getDate().toAstropy().mjd) if full else np.nan
        except Exception:
            mjd = np.nan
        pts, snrs = [], []
        peakrows = []
        for src in res.sources:
            fp = src.getFootprint()
            if fp is None or not fp.getPeaks():
                continue
            pk = fp.getPeaks()[0]; x, y = float(pk.getFx()), float(pk.getFy())
            ix, iy = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
            sig = float(np.sqrt(var[iy, ix])) if var[iy, ix] > 0 else np.nan
            snr = float(pk.getPeakValue()) / sig if sig and np.isfinite(sig) else np.nan
            pts.append((x, y)); snrs.append(snr)
            if full and wcs is not None:                 # full 5sigma diaSource catalogue for heliolinc
                try:
                    sp = wcs.pixelToSky(x, y); ra = sp.getRa().asDegrees(); dec = sp.getDec().asDegrees()
                    peakrows.append(dict(kind="peak", visit=v, detector=det, mjd=mjd,
                                         ra=ra, dec=dec, x=x, y=y, snr=snr))
                except Exception:
                    pass
        out = [dict(kind="fpdensity", visit=v, detector=det, n_5sigma=len(pts))]
        out.extend(peakrows)
        if inject_rows:
            tree = cKDTree(pts) if pts else None
            for r in inject_rows:
                hit, snr = False, np.nan
                if tree is not None:
                    dd, ii = tree.query((float(r["x"]), float(r["y"])), distance_upper_bound=tol)
                    if np.isfinite(dd):
                        hit, snr = True, snrs[ii]
                out.append(dict(kind="inj", visit=v, detector=det, objID=r["objID"],
                                x=float(r["x"]), y=float(r["y"]), stack_det=hit, stack_snr=snr))
        return out
    except Exception as e:
        return [dict(kind="err", visit=v, detector=det, _err=f"{type(e).__name__}: {e}")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="visit,detector,fits_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inject", default=None, help="inject.csv (same as ADCNN) -> inline trails before detect")
    ap.add_argument("--retime-map", default=None, help="visit,mjd_retimed -> stamp output mjd")
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--tol-px", type=float, default=10.0, help="match radius injected<->5sigma detection")
    ap.add_argument("--inject-panels-only", action="store_true",
                    help="run only on panels that carry injections (faster; FP density on a representative sample)")
    ap.add_argument("--full-catalog", action="store_true",
                    help="ALSO emit the full 5sigma peak catalogue (visit,detector,mjd,ra,dec,x,y,snr) to "
                         "<out>_peaks.csv -- the diaSource stream heliolinc consumes (FP + injected). Heavier.")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    man = pd.read_csv(a.manifest)[["visit", "detector", "fits_path"]].drop_duplicates(["visit", "detector"])
    inj_map = {}
    if a.inject:
        from ADCNN.pipelines.heliolinc.inject_trails import load_inject_map
        inj_map = load_inject_map(a.inject)
        print(f"[stack] inline-injecting {sum(len(v) for v in inj_map.values())} trails over {len(inj_map)} panels", flush=True)
    if a.inject_panels_only and inj_map:
        man = man[[(int(r.visit), int(r.detector)) in inj_map for r in man.itertuples()]]

    tasks = [(r.fits_path, dict(visit=int(r.visit), detector=int(r.detector)),
              inj_map.get((int(r.visit), int(r.detector))), a.threshold, a.tol_px, a.full_catalog)
             for r in man.itertuples()]
    print(f"[stack] 5sigma detection over {len(tasks)} panels, {a.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_panel, t) for t in tasks])):
            rows.extend(fut.result())
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} panels", flush=True)
    df = pd.DataFrame(rows)
    nerr = int((df.kind == "err").sum()) if "kind" in df else 0
    rm = None
    if a.retime_map:
        r = pd.read_csv(a.retime_map); rm = dict(zip(r.visit.astype(int), r.mjd_retimed.astype(float)))
    # FP-density per panel
    fpd = df[df.kind == "fpdensity"][["visit", "detector", "n_5sigma"]].copy()
    fpd.to_csv(a.out.replace(".csv", "_fpdensity.csv"), index=False)
    # injected-recovery rows (the main output)
    inj = df[df.kind == "inj"][["visit", "detector", "objID", "x", "y", "stack_det", "stack_snr"]].copy()
    if rm is not None and len(inj):
        inj["mjd"] = inj.visit.astype(int).map(rm)
    inj.to_csv(a.out, index=False)
    rec = int(inj.stack_det.sum()) if len(inj) else 0
    npk = 0
    if a.full_catalog and "kind" in df:
        pk = df[df.kind == "peak"][["visit", "detector", "mjd", "ra", "dec", "x", "y", "snr"]].copy()
        peaks_out = a.out.replace(".csv", "_peaks.csv")
        pk.to_csv(peaks_out, index=False); npk = len(pk)
        print(f"[stack] full 5sigma catalogue: {npk} peaks -> {peaks_out}", flush=True)
    print(f"[stack] panels={len(fpd)} median FP/panel={fpd.n_5sigma.median() if len(fpd) else 0:.0f} | "
          f"injected sightings={len(inj)} stack-recovered={rec} | full-peaks {npk} (errs {nerr}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
