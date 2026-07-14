"""Faithfully run Ari Heinze's heliolinx chain (make_tracklets -> heliolinc -> link_purify) on a
detection catalogue, using the SHIPPED binaries and auxiliary files unmodified. This is the COMMON LINKER
for the ADCNN-vs-stack head-to-head: both detection streams are passed through this same chain with
identical parameters, so any difference in recovered NEOs is attributable to the DETECTOR, not the linker.

================================ CODE BOUNDARY (read me) ================================
THIS FILE IS OURS. It is a THIN WRAPPER -- it only INVOKES Ari Heinze's upstream binaries; it does not
reimplement any of his algorithm.
  * Ari Heinze's code (NOT OURS): everything under external/heliolinx/ (binaries + C++ src) and
    external/heliolinx-aux/ (test data, hypothesis grids, Earth ephemeris, ObsCodes). Vendored READ-ONLY,
    gitignored, NEVER edited by us. We call the binaries by absolute path (HLX/AUX below) and pass his
    own shipped auxiliary files. Do not modify, copy into, or commit anything from external/.
  * Our code (OURS): this wrapper + ephem_to_inject.py + the rest of ADCNN/pipelines/heliolinc/.
If you ever need to "fix" linker behaviour, fix it HERE (params/inputs) or upstream in his repo separately
-- do not fork his source into our tree.
========================================================================================

Reproduces the verified external/heliolinx-aux TenObjects test invocation, parameterised for NEOs over a
multi-night arc (NEO heliocentric hypothesis grid, minobsnights=3, NEO apparent-rate window). Input is any
CSV with columns mjd, ra, dec (deg) and optionally mag, band; one row per detection. Output is the final
purified linkage set (clust2det) mapping linkage_id -> detection rows of the input catalogue.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
HLX = REPO / "external/heliolinx/bin"                 # tool INPUTS (gitignored local build)
AUX = REPO / "external/heliolinx-aux/tests"
EARTH = AUX / "Earth1day2020s_02a.csv"
OBSCODES = AUX / "ObsCodesNew.txt"
NEO_HYP = AUX / "hypotheses" / "NEO" / "hihyp00aa_neo.txt"
OBSCODE_RUBIN = "X05"  # Simonyi Survey Telescope, Rubin Observatory (verified in ObsCodesNew.txt)


def run(cmd, log):
    log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n"); log.flush()
    r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    log.flush()
    if r.returncode != 0:
        raise SystemExit(f"[heliolinx] FAILED ({r.returncode}): {cmd[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="detection CSV (cols: mjd,ra,dec[,mag,band])")
    ap.add_argument("--outdir", required=True, help="work/output dir")
    ap.add_argument("--tag", default="h2h")
    ap.add_argument("--hypothesis", default=str(NEO_HYP), help="heliocentric r/rdot/accel grid (default NEO)")
    ap.add_argument("--mjd-ref", type=float, default=None, help="reference MJD (default = arc midpoint)")
    # make_tracklets (same-night tracklets, NEO rates)
    ap.add_argument("--maxvel", type=float, default=15.0, help="deg/day max tracklet rate (NEOs up to ~10)")
    ap.add_argument("--minvel", type=float, default=0.0)
    ap.add_argument("--maxtime", type=float, default=1.5, help="hours; max intra-night tracklet gap")
    ap.add_argument("--maxgcr", type=float, default=1.0, help="arcsec; max great-circle residual of tracklet")
    ap.add_argument("--trail-as-tracklet", action="store_true",
                    help="ONE TRAIL = ONE TRACKLET: emit each detection as TWO endpoint pseudo-observations "
                         "at MJD -/+ exptime/2 (needs ra0/dec0/ra1/dec1 columns) and shrink -maxtime to the "
                         "exposure so make_tracklets pairs ONLY each trail's own endpoints (KD radius "
                         "maxvel*30s ~ 20 arcsec -> no O(N x big-radius) same-night pairing explosion; "
                         "observer-ephemeris/imgs generation stays in Heinze's unmodified code). The trail "
                         "supplies the velocity; cross-visit chords are NOT formed (multi-night linking "
                         "carries the association). Also lets 1-detection-per-night objects link.")
    ap.add_argument("--exptime", type=float, default=30.0, help="s; exposure (trail-as-tracklet endpoint dt)")
    # heliolinc
    ap.add_argument("--clustrad", type=float, default=100000.0, help="km clustering radius (close NEOs need >= 1e5)")
    ap.add_argument("--minobsnights", type=int, default=3, help="HelioLinC3D requirement: >=3 nights")
    ap.add_argument("--mintimespan", type=float, default=0.5, help="days")
    ap.add_argument("--maxgeodist", type=float, default=3.0, help="AU; NEO geocentric range")
    # link_purify
    ap.add_argument("--max-astrom-rms", type=float, default=1.0, help="arcsec; orbit-fit acceptance")
    ap.add_argument("--minpointnum", type=int, default=6, help="min detections (>=3 tracklets)")
    ap.add_argument("--default-mag", type=float, default=22.0)
    a = ap.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"heliolinx_{a.tag}.log", "w")

    d = pd.read_csv(a.dets)
    for c in ("mjd", "ra", "dec"):
        if c not in d.columns:
            raise SystemExit(f"[heliolinx] --dets missing column '{c}'")
    d = d.dropna(subset=["mjd", "ra", "dec"]).reset_index(drop=True)
    if a.trail_as_tracklet:
        for c in ("ra0", "dec0", "ra1", "dec1"):
            if c not in d.columns:
                raise SystemExit(f"[heliolinx] --trail-as-tracklet needs trail endpoint column '{c}'")
        half = a.exptime / 2.0 / 86400.0
        e0 = d.copy(); e0["mjd"] = d.mjd - half; e0["ra"] = d.ra0; e0["dec"] = d.dec0
        e1 = d.copy(); e1["mjd"] = d.mjd + half; e1["ra"] = d.ra1; e1["dec"] = d.dec1
        d = pd.concat([e0, e1], ignore_index=True).dropna(subset=["ra", "dec"]).reset_index(drop=True)
        # pair window = the exposure (+20% slack): each trail pairs only with its own twin endpoint
        a.maxtime = a.exptime * 1.2 / 3600.0
        a.minvel = 0.0   # point-like (slow) sources have ~coincident endpoints; keep them
        print(f"[heliolinx] trail-as-tracklet: {len(d)//2} trails -> {len(d)} endpoint pseudo-obs, "
              f"maxtime={a.maxtime*3600:.0f}s", flush=True)
    mag = d["mag"] if "mag" in d.columns else pd.Series(np.nan, index=d.index)
    mag = mag.fillna(a.default_mag)
    band = d["band"].astype(str).str[:1] if "band" in d.columns else pd.Series("r", index=d.index)
    # heliolinx detection input: index,id,mjd,ra,dec,mag,band,obscode (CSV, header)
    hlx = pd.DataFrame(dict(index=np.arange(len(d)), id=[f"d{i}" for i in range(len(d))],
                            mjd=d.mjd.astype(float), ra=d.ra.astype(float), dec=d.dec.astype(float),
                            mag=mag.astype(float), band=band, obscode=OBSCODE_RUBIN))
    detfile = out / f"dets_{a.tag}.csv"
    if Path(a.dets).resolve() == detfile.resolve():
        raise SystemExit(f"[heliolinx] --dets would be OVERWRITTEN by the internal {detfile.name}; "
                         f"rename the input or use a different --tag/--outdir")
    hlx.to_csv(detfile, index=False)
    cf = out / f"colformat_{a.tag}.txt"
    cf.write_text("IDCOL 2\nMJDCOL 3\nRACOL 4\nDECCOL 5\nMAGCOL 6\nBANDCOL 7\nOBSCODECOL 8\n")
    print(f"[heliolinx] {len(hlx)} dets, MJD [{d.mjd.min():.3f},{d.mjd.max():.3f}], "
          f"{int(np.floor(d.mjd-0.5).nunique())} nights -> {detfile.name}", flush=True)

    imgs = out / f"imgs_{a.tag}.txt"; pairdets = out / f"pairdets_{a.tag}.csv"
    trks = out / f"tracklets_{a.tag}.csv"; trk2det = out / f"trk2det_{a.tag}.csv"
    run([HLX / "make_tracklets", "-dets", detfile, "-outimgs", imgs, "-pairdets", pairdets,
         "-tracklets", trks, "-trk2det", trk2det, "-colformat", cf, "-imrad", 2.0,
         "-maxtime", a.maxtime, "-maxvel", a.maxvel, "-minvel", a.minvel, "-maxGCR", a.maxgcr,
         "-earth", EARTH, "-obscode", OBSCODES], log)

    mjd_ref = a.mjd_ref if a.mjd_ref is not None else float((d.mjd.min() + d.mjd.max()) / 2.0)
    sumf = out / f"sum_{a.tag}.csv"; c2d = out / f"clust2det_{a.tag}.csv"
    # NB: plain `heliolinc` (not heliolinc_omp) -- the omp variant emits a 31-col sum that the link_*
    # purifier readers reject (format skew); plain heliolinc's 32-col sum reads cleanly. Verified on the
    # shipped TenObjects test: make_tracklets -> heliolinc -> link_purify reproduces 10/10 linkages.
    run([HLX / "heliolinc", "-imgs", imgs, "-pairdets", pairdets, "-tracklets", trks,
         "-trk2det", trk2det, "-mjd", mjd_ref, "-obspos", EARTH, "-heliodist", a.hypothesis,
         "-clustrad", a.clustrad, "-minobsnights", a.minobsnights, "-mintimespan", a.mintimespan,
         "-maxgeodist", a.maxgeodist, "-outsum", sumf, "-clust2det", c2d], log)

    lflist = out / f"lflist_{a.tag}.txt"; lflist.write_text(f"{sumf.name} {c2d.name}\n")
    lpsum = out / f"LPsum_{a.tag}.csv"; lpc2d = out / f"LPclust2det_{a.tag}.csv"
    # link_purify resolves relative paths in lflist against its CWD
    run(["bash", "-c",
         f"cd {out} && {HLX/'link_purify'} -imgs {imgs.name} -pairdet {pairdets.name} "
         f"-lflist {lflist.name} -max_astrom_rms {a.max_astrom_rms} -minobsnights {a.minobsnights} "
         f"-minpointnum {a.minpointnum} -outsum {lpsum.name} -clust2det {lpc2d.name}"], log)

    # summarise: final purified linkages + their detection rows mapped back to input index
    nlink = 0
    if lpsum.exists():
        try:
            nlink = max(0, sum(1 for _ in open(lpsum)) - 1)
        except Exception:
            nlink = 0
    print(f"[heliolinx] FINAL purified linkages: {nlink} -> {lpsum.name}, {lpc2d.name}", flush=True)
    log.close()
    # write a manifest of what maps to what for the metrics step
    (out / f"heliolinx_{a.tag}_paths.txt").write_text(
        f"detfile={detfile}\npairdets={pairdets}\nLPsum={lpsum}\nLPclust2det={lpc2d}\nmjd_ref={mjd_ref}\n")


if __name__ == "__main__":
    main()
