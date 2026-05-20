#!/usr/bin/env python
"""LSST stack forced photometry (`base_PsfFlux` + `ext_trailedSources_Naive`)
on the test_real diffims at ephemeris (RA/Dec). Uses the EXACT same
measurement machinery the LSST DRP runs on diaSources, so results match
test.csv `stack_snr` for stack-detected sources (calibration check).
"""
from __future__ import annotations
import argparse, sys, time, concurrent.futures, threading
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
_DSC = f"{_REPO}/ADCNN/data/dataset_creation"
for p in (_REPO, _DSC):
    if p not in sys.path: sys.path.insert(0, p)

from ADCNN.data.dataset_creation.pipetasks import fetch_diffim_inputs, run_subtract  # noqa: E402
from ADCNN.data.dataset_creation.simulate_inject_diffim import format_dataId  # noqa: E402
from lsst.daf.butler import Butler  # noqa: E402
import lsst.afw.table as afwTable  # noqa: E402
import lsst.afw.geom as afwGeom  # noqa: E402
import lsst.afw.detection as afwDet  # noqa: E402
import lsst.geom as geom  # noqa: E402
from lsst.meas.base import (  # noqa: E402
    ForcedMeasurementTask, ForcedMeasurementConfig,
)

STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
SKYMAP = "lsst_cells_v2"
REPO = "dp2_prep"

# Probe for the trailed-source extension once (some envs don't have it).
try:
    import lsst.meas.extensions.trailedSources  # noqa: F401
    HAS_TRAIL = True
except Exception:
    HAS_TRAIL = False

_TLS = threading.local()
def _butler():
    b = getattr(_TLS, "butler", None)
    if b is None:
        b = Butler(REPO, collections=[STAGE3, STAGE2])
        _TLS.butler = b
    return b


def _make_forced_task():
    refSchema = afwTable.SourceTable.makeMinimalSchema()
    cfg = ForcedMeasurementConfig()
    # ext_trailedSources_* are SingleFrame-only plugins (not registered for
    # forced measurement). Forced photometry uses base_PsfFlux at the
    # transformed centroid; that matches LSST stack_snr for point-source
    # photometry. Trailed forced measurement is a separate SFM-based path.
    cfg.plugins.names = ["base_TransformedCentroidFromCoord", "base_PsfFlux"]
    cfg.copyColumns = {"id": "objectId"}
    cfg.slots.centroid = "base_TransformedCentroidFromCoord"
    cfg.slots.psfFlux = "base_PsfFlux"
    # Disable slot requirements we don't have refs for.
    for s in ("shape", "psfShape", "modelFlux", "apFlux",
              "gaussianFlux", "calibFlux"):
        try:
            setattr(cfg.slots, s, None)
        except Exception:
            pass
    task = ForcedMeasurementTask(refSchema, config=cfg)
    return task, refSchema


def _build_ref_catalog(refSchema, sightings, ref_wcs, fp_radius_px=25):
    """SourceCatalog with one entry per sighting at (RA, Dec) with a
    small circular footprint covering the PSF + trail extent."""
    table = afwTable.SourceTable.make(refSchema)
    cat = afwTable.SourceCatalog(table)
    for s in sightings:
        src = cat.addNew()
        src.setCoord(geom.SpherePoint(float(s["ra"]), float(s["dec"]),
                                       geom.degrees))
        # Centroid in pixel space for forced footprint anchor.
        pix = ref_wcs.skyToPixel(src.getCoord())
        ix, iy = int(round(pix.getX())), int(round(pix.getY()))
        # Trail aware radius: span >= trail/2 + PSF margin
        L = float(s.get("trail_length") or 0.0)
        r = int(max(fp_radius_px, np.ceil(L / 2.0) + 10))
        center = geom.Point2I(ix, iy)
        spans = afwGeom.SpanSet.fromShape(r, afwGeom.Stencil.CIRCLE,
                                          offset=center)
        fp = afwDet.Footprint(spans)
        fp.addPeak(float(pix.getX()), float(pix.getY()), 1.0)
        src.setFootprint(fp)
    return cat


def _process_panel(args):
    visit, detector, sightings = args
    out_rows = []
    try:
        b = _butler()
        ref = b.registry.findDataset(
            "preliminary_visit_image",
            dataId={"instrument": "LSSTCam", "visit": int(visit),
                    "detector": int(detector)},
            collections=[STAGE3, STAGE2])
        pvi, sources, template, _f, _n = fetch_diffim_inputs(
            b, format_dataId(ref.dataId), skymap=SKYMAP,
            stage3_collection=STAGE3)
        sub = run_subtract(template=template, science=pvi, sources=sources)
        diffim = sub.difference
        wcs = diffim.getWcs()
        task, refSchema = _make_forced_task()
        refCat = _build_ref_catalog(refSchema, sightings, wcs)
        measCat = task.generateMeasCat(diffim, refCat, wcs)
        task.attachTransformedFootprints(measCat, refCat, diffim, wcs)
        task.run(measCat, diffim, refCat, wcs)
        for src_in, src_meas in zip(sightings, measCat):
            row = dict(src_in)
            try:
                pf = float(src_meas["base_PsfFlux_instFlux"])
                pe = float(src_meas["base_PsfFlux_instFluxErr"])
                ps = pf / pe if pe > 0 else float("nan")
                try:
                    pflag = bool(src_meas["base_PsfFlux_flag"])
                except Exception:
                    pflag = False
            except Exception:
                pf = pe = ps = float("nan"); pflag = True
            row.update({
                "lsst_psf_flux": pf,
                "lsst_psf_fluxErr": pe,
                "lsst_psf_snr": ps,
                "lsst_psf_flag": pflag,
            })
            if HAS_TRAIL:
                try:
                    tf = float(src_meas["ext_trailedSources_Naive_flux"])
                    te = float(src_meas["ext_trailedSources_Naive_fluxErr"])
                    ts = tf / te if te > 0 else float("nan")
                    tlen = float(src_meas["ext_trailedSources_Naive_length"])
                except Exception:
                    # Field names may differ by LSST version; try alternates.
                    try:
                        tf = float(src_meas["ext_trailedSources_Naive_instFlux"])
                        te = float(src_meas["ext_trailedSources_Naive_instFluxErr"])
                        ts = tf / te if te > 0 else float("nan")
                        tlen = float("nan")
                    except Exception:
                        tf = te = ts = tlen = float("nan")
                row.update({
                    "lsst_trail_flux": tf, "lsst_trail_fluxErr": te,
                    "lsst_trail_snr": ts, "lsst_trail_length": tlen,
                })
            out_rows.append(row)
        return ("ok", visit, detector, out_rows, len(sightings))
    except Exception as e:
        import traceback
        return ("err", visit, detector, [
            {**s, "lsst_psf_flux": float("nan"),
             "lsst_psf_fluxErr": float("nan"),
             "lsst_psf_snr": float("nan"),
             "lsst_psf_flag": True,
             "lsst_trail_flux": float("nan"),
             "lsst_trail_fluxErr": float("nan"),
             "lsst_trail_snr": float("nan"),
             "lsst_trail_length": float("nan")}
            for s in sightings], f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-sighting", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--parallel", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    print(f"HAS_TRAIL extension: {HAS_TRAIL}", flush=True)

    ps = pd.read_csv(a.per_sighting)
    tc = pd.read_csv(a.test_csv)
    cols = ["ObjID", "visit", "detector", "x", "y", "beta",
            "trail_length", "ra", "dec"]
    tcj = tc[[c for c in cols if c in tc.columns]].copy()
    if "trail_length" in ps.columns and "trail_length" in tcj.columns:
        tcj = tcj.drop(columns=["trail_length"])
    m = ps.merge(tcj, on=["ObjID", "visit", "detector"], how="left",
                 suffixes=("", "_eph"))
    print(f"[load] sightings={len(m)} no_xy={m['x'].isna().sum()}", flush=True)
    if a.limit:
        m = m.head(a.limit).copy()
        print(f"[limit] {len(m)}", flush=True)

    panels = []
    for (v, d), g in m.groupby(["visit", "detector"], sort=True):
        rows = g.to_dict("records")
        if any(pd.notna(r.get("x")) for r in rows):
            panels.append((int(v), int(d), rows))
    print(f"[panels] {len(panels)}", flush=True)

    t0 = time.time()
    out = []
    ok = err = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.parallel) as ex:
        futs = [ex.submit(_process_panel, p) for p in panels]
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res[0] == "ok": ok += 1
            else:
                err += 1
                if err <= 3:
                    print(f"  ERR panel {res[1]}/{res[2]}: {res[4]}", flush=True)
            out.extend(res[3])
            if n % 20 == 0 or n == len(panels):
                print(f"[{n}/{len(panels)}] ok={ok} err={err} "
                      f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(a.out_csv, index=False)
    valid = int(df['lsst_psf_snr'].notna().sum())
    print(f"[done] n={len(df)} valid_psf={valid} "
          f"median_psf_snr={df['lsst_psf_snr'].median():.2f}", flush=True)
    if "lsst_trail_snr" in df.columns:
        print(f"       valid_trail={int(df['lsst_trail_snr'].notna().sum())} "
              f"median_trail_snr={df['lsst_trail_snr'].median():.2f}",
              flush=True)


if __name__ == "__main__":
    main()
