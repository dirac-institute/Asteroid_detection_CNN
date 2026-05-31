"""Same-night NEO linking: ADCNN trail-tracklets -> heliolinc -> short-arc orbit.

The plan (decided 2026-05-31, from Mario Jurić; see memory heliolinc-algorithm-samenight):
link the multiple sightings of one fast mover *within a single night* and fit a short-arc
orbit, so a NEO can be flagged the same night it is seen — no waiting for later nights.

Why same-night works at all: a TRAILED detection already encodes a tracklet. The trail's two
endpoints are the object's positions at exposure-start and exposure-end, i.e. position + on-sky
velocity from ONE exposure. So "one trail = one tracklet" (the chosen model): each ADCNN trail
becomes a 2-point tracklet. Two such tracklets from two visits in the same night give heliolinc
two epochs -> it links them and link_purify's Method-of-Herget fit returns a (short-arc) orbit.

heliolinc's default `-minobsnights 3` is a false-positive cut, NOT an algorithmic floor; we set
`-minobsnights 1` and a short `-mintimespan`, and run the linker PER NIGHT (each night's own
reference MJD) — which is also the operational "one alert per night" model.

Format strategy: rather than hand-roll heliolinc's imgs/pairdets/tracklets/trk2det files (observer
state vectors with obscode parallax are easy to get wrong), we let Heinze's own `make_tracklets`
build them. We hand it each trail as two detections — one at the detection MJD, one at MJD+exptime
displaced along the trail — at times unique to that trail, so make_tracklets pairs exactly those
two into a tracklet (one-trail=one-tracklet) with no cross-trail mispairing, and emits correctly
formatted inputs for heliolinc.

Two entry points:
  build + run per-night chain on a measured ADCNN catalog (--dets), or
  validate-on-truth: synthesize idealized trails from a known-object ephemeris and score recovery.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
HL = REPO / "ADCNN/pipelines/heliolinc"
BIN = REPO / "external/heliolinx/bin"
AUX = REPO / "external/heliolinx-aux/tests"

# colformat for the detection CSV we hand to make_tracklets (1-indexed columns).
# We put ObjID in IDCOL so heliolinc's pairdets.idstring carries truth for scoring.
COLFORMAT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"
MKTRACK_COLS = ["ObjID", "mjd", "ra", "dec", "mag", "band", "obscode"]


def synth_trails_from_truth(truth, exptime_s=30.0, min_samenight=2):
    """Turn each same-night sighting into an idealized trail using the object's local sky-motion
    (finite-difference of its own ephemeris within the night). Keeps only object-nights with
    >= min_samenight detections (the regime where same-night linking is possible).

    Returns a dets frame: ObjID, night, mjd, ra, dec (trail start), ra_e, dec_e (trail end at
    mjd+exptime), mag, band, obscode. One row per trail (per sighting)."""
    t = truth.copy()
    t["night"] = np.floor(t.mjd - 0.5).astype(int)
    dt = exptime_s / 86400.0
    rows = []
    for (oid, night), g in t.groupby(["ObjID", "night"]):
        if len(g) < min_samenight:
            continue
        g = g.sort_values("mjd").reset_index(drop=True)
        tt = g.mjd.to_numpy(); ra = g.ra.to_numpy(); dec = g.dec.to_numpy()
        # local sky rate (deg/day) via finite difference on the within-night ephemeris
        dradt = np.gradient(ra, tt); ddecdt = np.gradient(dec, tt)
        mag = g.get("mag", pd.Series([21.0] * len(g))).to_numpy()
        band = g.get("band1", g.get("band", pd.Series(["r"] * len(g)))).astype(str).to_numpy()
        obsc = g.get("obscode", pd.Series(["I11"] * len(g))).astype(str).to_numpy()
        for i in range(len(g)):
            rows.append(dict(ObjID=oid, night=int(night), mjd=float(tt[i]),
                             ra=float(ra[i]), dec=float(dec[i]),
                             ra_e=float(ra[i] + dradt[i] * dt),
                             dec_e=float(dec[i] + ddecdt[i] * dt),
                             mag=float(mag[i]), band=str(band[i])[:1] or "r",
                             obscode=str(obsc[i])))
    return pd.DataFrame(rows)


def write_mktrack_input(dets, path, exptime_s=30.0):
    """Write the make_tracklets detection CSV: TWO rows per trail (start at mjd, end at mjd+exptime
    displaced along the trail), at times unique to that trail. Returns the colformat path written
    alongside."""
    dt = exptime_s / 86400.0
    out = []
    for _, d in dets.iterrows():
        out.append((d.ObjID, f"{d.mjd:.8f}",        f"{d.ra:.8f}",   f"{d.dec:.8f}",   f"{d.mag:.3f}", d.band, d.obscode))
        out.append((d.ObjID, f"{d.mjd + dt:.8f}",    f"{d.ra_e:.8f}", f"{d.dec_e:.8f}", f"{d.mag:.3f}", d.band, d.obscode))
    df = pd.DataFrame(out, columns=MKTRACK_COLS)
    df.to_csv(path, index=False)
    cf = Path(path).with_suffix(".colformat.txt")
    cf.write_text(COLFORMAT)
    return cf


def _run(cmd, log_path=None):
    """Run a subprocess, raise on failure, optionally tee stdout+stderr to a log."""
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if log_path is not None:
        Path(log_path).write_text(res.stdout + "\n==STDERR==\n" + res.stderr)
    if res.returncode != 0:
        tail = (res.stdout + res.stderr)[-1500:]
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return res.stdout


def link_one_night(dets_night, work, earth, obscode_file, heliodist, *,
                   exptime_s=30.0, mjdref=None, clustrad=1e5, minobsnights=1,
                   mintimespan=0.0, maxvel=30.0, max_astrom_rms=1.0, npt=2, minpointnum=4):
    """Run make_tracklets -> heliolinc -> link_planarity on one night's trails.
    Returns (lplsum_df, lplclust2det_df) or (None, None) if nothing linked."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    dets_csv = work / "dets.csv"
    cf = write_mktrack_input(dets_night, dets_csv, exptime_s=exptime_s)
    if mjdref is None:
        mjdref = 0.5 * (dets_night.mjd.min() + dets_night.mjd.max())

    imgs = work / "imgs.txt"; pairdets = work / "pairdets.csv"
    tracklets = work / "tracklets.csv"; trk2det = work / "trk2det.csv"
    _run([BIN / "make_tracklets", "-dets", dets_csv, "-outimgs", imgs,
          "-pairdets", pairdets, "-tracklets", tracklets, "-trk2det", trk2det,
          "-colformat", cf, "-imrad", 5.0, "-maxtime", 0.02, "-mintime", 0.0,
          "-maxGCR", 5.0, "-minvel", 0.0, "-maxvel", maxvel, "-mintrkpts", 2,
          "-earth", earth, "-obscode", obscode_file], work / "mktracklets.log")
    if not tracklets.exists() or tracklets.stat().st_size == 0:
        return None, None

    sumf = work / "sum.csv"; c2d = work / "clust2det.csv"
    _run([BIN / "heliolinc", "-imgs", imgs, "-pairdets", pairdets, "-tracklets", tracklets,
          "-trk2det", trk2det, "-mjd", f"{mjdref:.6f}", "-obspos", earth,
          "-heliodist", heliodist, "-clustrad", clustrad, "-npt", npt,
          "-minobsnights", minobsnights, "-mintimespan", mintimespan,
          "-outsum", sumf, "-clust2det", c2d], work / "heliolinc.log")
    if not sumf.exists() or sum(1 for _ in open(sumf)) <= 1:
        return None, None

    # link_planarity resolves lflist entries relative to CWD -> use absolute paths.
    clusterlist = work / "clusterlist"; clusterlist.write_text(f"{sumf.resolve()} {c2d.resolve()}\n")
    lplsum = work / "LPLsum.csv"; lplc2d = work / "LPLclust2det.csv"
    _run([BIN / "link_planarity", "-imgs", imgs, "-pairdet", pairdets, "-lflist", clusterlist,
          "-simptype", 1, "-max_astrom_rms", max_astrom_rms, "-oop", 10000.0,
          "-minobsnights", minobsnights, "-minpointnum", minpointnum, "-ptpow", 3,
          "-maxrms", 400000.0, "-outsum", lplsum, "-clust2det", lplc2d], work / "link_planarity.log")
    if not lplsum.exists() or sum(1 for _ in open(lplsum)) <= 1:
        return None, None
    return pd.read_csv(lplsum), pd.read_csv(lplc2d)


def score_linkages(lplsum, lplc2d, pairdets_path):
    """Score each linkage's purity against the ObjID carried in pairdets.idstring.
    Returns (per_linkage_df, n_pure_objects)."""
    pd_df = pd.read_csv(pairdets_path)
    idcol = "idstring" if "idstring" in pd_df.columns else pd_df.columns[10]
    # clust2det maps cluster number -> detection index (row in pairdets).
    ccol = lplc2d.columns[0]; dcol = lplc2d.columns[1]
    rows = []
    for cnum, grp in lplc2d.groupby(ccol):
        objs = pd_df.iloc[grp[dcol].to_numpy()][idcol].astype(str)
        top = objs.value_counts()
        purity = top.iloc[0] / len(objs)
        rows.append(dict(cluster=cnum, n_det=len(objs), top_obj=top.index[0],
                         purity=purity, n_obj=objs.nunique()))
    per = pd.DataFrame(rows)
    pure = per[per.purity >= 0.8]
    return per, pure.top_obj.nunique()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default=str(HL / "run_truth/truth_dets.csv"),
                    help="validation: known-object detection catalog (detid,mjd,ra,dec,mag,band1,obscode,ObjID)")
    ap.add_argument("--earth", default=str(AUX / "Earth1day2020s_02a.csv"))
    ap.add_argument("--obscode-file", default=str(AUX / "ObsCodesNew.txt"))
    ap.add_argument("--heliodist", default=str(AUX / "hypotheses/NEO/hihyp00aa_neo.txt"),
                    help="heliocentric (r, rdot, accel) hypothesis grid; NEO grid by default")
    ap.add_argument("--out", default=str(HL / "run_samenight"))
    ap.add_argument("--exptime", type=float, default=30.0)
    ap.add_argument("--clustrad", type=float, default=1e5)
    ap.add_argument("--minobsnights", type=int, default=1, help="1 = allow same-night linkages")
    ap.add_argument("--mintimespan", type=float, default=0.0)
    ap.add_argument("--maxvel", type=float, default=30.0, help="deg/day; raise for fast NEOs")
    ap.add_argument("--max-astrom-rms", type=float, default=1.0, help="arcsec; orbit-fit acceptance")
    ap.add_argument("--min-samenight", type=int, default=2, help="keep object-nights with >= this many det")
    ap.add_argument("--nights", type=int, default=0, help="limit to first N nights (0 = all)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(a.truth)
    dets = synth_trails_from_truth(truth, exptime_s=a.exptime, min_samenight=a.min_samenight)
    nights = sorted(dets.night.unique())
    if a.nights:
        nights = nights[:a.nights]
    print(f"[samenight] {dets.ObjID.nunique()} objects, {len(dets)} trails across {len(nights)} nights "
          f"(>= {a.min_samenight} det/night) | heliodist={Path(a.heliodist).name} minobsnights={a.minobsnights}",
          flush=True)

    recovered, total_linkages, total_fp = set(), 0, 0
    per_night = []
    for night in nights:
        dn = dets[dets.night == night].reset_index(drop=True)
        nobj = dn.ObjID.nunique()
        try:
            lplsum, lplc2d = link_one_night(dn, out / f"night_{night}", a.earth, a.obscode_file,
                                            a.heliodist, exptime_s=a.exptime, clustrad=a.clustrad,
                                            minobsnights=a.minobsnights, mintimespan=a.mintimespan,
                                            maxvel=a.maxvel, max_astrom_rms=a.max_astrom_rms)
        except RuntimeError as e:
            print(f"  night {night}: FAILED {e}", flush=True); continue
        if lplsum is None:
            per_night.append(dict(night=night, n_obj=nobj, n_link=0, n_pure=0))
            print(f"  night {night}: {nobj} objs -> 0 linkages", flush=True); continue
        per, n_pure = score_linkages(lplsum, lplc2d, out / f"night_{night}/pairdets.csv")
        nlink = len(per); nfp = int((per.purity < 0.8).sum())
        total_linkages += nlink; total_fp += nfp
        recovered.update(per[per.purity >= 0.8].top_obj.tolist())
        per_night.append(dict(night=night, n_obj=nobj, n_link=nlink, n_pure=int((per.purity >= 0.8).sum())))
        print(f"  night {night}: {nobj} objs -> {nlink} linkages ({nlink-nfp} pure, {nfp} FP)", flush=True)

    pn = pd.DataFrame(per_night); pn.to_csv(out / "per_night.csv", index=False)
    n_linkable = dets.ObjID.nunique()
    print(f"\n[samenight] RESULT: recovered {len(recovered)}/{n_linkable} same-night-linkable objects "
          f"| {total_linkages} linkages total, {total_fp} impure (FP)", flush=True)
    print(f"  per-night summary -> {out/'per_night.csv'}", flush=True)


if __name__ == "__main__":
    main()
