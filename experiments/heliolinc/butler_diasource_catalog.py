"""Build a diaSource (the LSST 'stack' difference-image detections) catalog over the discovery
window, carrying Rubin's reliability (real/bogus) score -- the basis of how Rubin cleans stack FP.

Reads the consolidated `dia_source_visit` table for each window visit, restricts to the discovery
sky region (the bounding box of the ADCNN run, so it is directly comparable to the ADCNN catalog),
and writes diasources.csv [detid, mjd, ra, dec, mag, band, obscode, reliability, isNegative, snr,
trailLength]. Downstream: keep reliability>=THR (+ not isNegative) -> HelioLinC (Rubin's method),
or use the per-source reliability to gate the ADCNN detections.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from lsst.daf.butler import Butler

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
STAGE4 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
PIXSCALE = 0.2  # arcsec/px (LSSTCam) — trailLength is in pixels
COLFORMAT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO / "experiments/heliolinc/run_disco/manifest.csv"))
    ap.add_argument("--region-from", default=str(REPO / "experiments/heliolinc/run_disco/adcnn_dets_full.csv"),
                    help="csv whose ra/dec bounding box defines the discovery region (for apples-to-apples)")
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/run_disco/diasources.csv"))
    a = ap.parse_args()

    b = Butler("dp2_prep")
    visits = sorted(pd.read_csv(a.manifest).visit.unique())
    reg = pd.read_csv(a.region_from)
    ra0, ra1 = reg.ra.min() - 0.05, reg.ra.max() + 0.05
    dec0, dec1 = reg.dec.min() - 0.05, reg.dec.max() + 0.05
    print(f"[diasrc] region ra [{ra0:.2f},{ra1:.2f}] dec [{dec0:.2f},{dec1:.2f}] over {len(visits)} visits")

    rows = []
    for v in visits:
        refs = list(b.registry.queryDatasets("dia_source_visit", collections=STAGE4,
                                             where=f"instrument='LSSTCam' AND visit={int(v)}", findFirst=True))
        if not refs:
            continue
        df = b.get(refs[0]).to_pandas()
        df = df[(df.ra >= ra0) & (df.ra <= ra1) & (df.dec >= dec0) & (df.dec <= dec1)].copy()
        if not len(df):
            continue
        flux = df.get("psfFlux", pd.Series(np.nan, index=df.index)).to_numpy()
        mag = np.where(flux > 0, 31.4 - 2.5 * np.log10(np.abs(flux) + 1e-9), 21.0)
        ra = df.ra.to_numpy(); dec = df.dec.to_numpy()
        # Trail ENDPOINTS for trail->tracklet linking, from the stack's own trail fit:
        # trailLength (px) + measured trail vector (trailRa/Dec is ~one endpoint vs the ra/dec
        # centroid). Build symmetric endpoints centred on (ra,dec): direction from the trail
        # vector, magnitude = trailLength/2 * pixscale. (No Veres re-run needed.)
        tL = df.get("trailLength", pd.Series(np.nan, index=df.index)).to_numpy()
        tra = df.get("trailRa", pd.Series(ra, index=df.index)).to_numpy()
        tdec = df.get("trailDec", pd.Series(dec, index=df.index)).to_numpy()
        cosd = np.cos(np.radians(dec))
        ux = (tra - ra) * cosd; uy = (tdec - dec)            # trail-vector components (deg)
        nrm = np.hypot(ux, uy); nrm = np.where(nrm > 1e-9, nrm, 1.0)
        half = (tL * PIXSCALE / 3600.0) / 2.0                 # px -> deg, half-length
        dra = (half * ux / nrm) / np.where(cosd > 1e-6, cosd, 1.0); ddec = half * uy / nrm
        rows.append(pd.DataFrame({
            "mjd": df.midpointMjdTai.to_numpy(), "ra": ra, "dec": dec,
            "ra0": ra - dra, "dec0": dec - ddec, "ra1": ra + dra, "dec1": dec + ddec,
            "mag": mag, "band": df.get("band", "r").astype(str).str[:1], "obscode": "I11",
            "reliability": df.reliability.to_numpy(),
            "isNegative": df.get("isNegative", False).astype(bool).to_numpy(),
            "snr": df.get("snr", np.nan).to_numpy(),
            "trailLength": tL, "trailAngle": df.get("trailAngle", np.nan).to_numpy(), "visit": int(v),
        }))
    cat = pd.concat(rows, ignore_index=True).sort_values(["mjd"]).reset_index(drop=True)
    cat.insert(0, "detid", range(len(cat)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(a.out, index=False)
    (Path(a.out).parent / "colformat.txt").write_text(COLFORMAT)
    print(f"[diasrc] {len(cat)} diaSources in region | reliability>=0.5: {(cat.reliability>=0.5).sum()} "
          f"({100*(cat.reliability>=0.5).mean():.0f}%) | not-negative & rel>=0.5: "
          f"{((cat.reliability>=0.5)&(~cat.isNegative)).sum()} -> {a.out}")


if __name__ == "__main__":
    main()
