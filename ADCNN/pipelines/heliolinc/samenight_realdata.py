"""Same-night NEO linking on a REAL ADCNN detection catalog (one trail = one tracklet).

ADCNN+Veres give, per trailed detection, two sky endpoints (ra0,dec0 / ra1,dec1) at one exposure —
position + on-sky velocity from a single visit. make_tracklets models one exposure as ONE image, so
it cannot split a single trail into a 2-image tracklet; we therefore author heliolinc's native input
files directly:

  per visit -> TWO images (exposure start at MJD, end at MJD+exptime; observer state propagated by
               the image velocity vector V over exptime)
  per trail -> endpoint0 detection in the start image, endpoint1 in the end image, and ONE tracklet
               (Img1->Img2) linking them == the per-exposure state vector

Observer state vectors (with obscode parallax) are HARVESTED from Heinze's own make_tracklets (run on
the trail centroids), so we never reimplement the geometry. Then heliolinc (-minobsnights 1 -npt 2)
links tracklets across the night and link_planarity fits the short-arc orbit; linkages are
crossmatched to the known-SSObject ephemerides -> CONFIRMED / NEW, per night.
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

CENTROID_COLS = ["detid", "mjd", "ra", "dec", "mag", "band", "obscode"]
CENTROID_COLFMT = "IDCOL 1\nMJDCOL 2\nRACOL 3\nDECCOL 4\nMAGCOL 5\nBANDCOL 6\nOBSCODECOL 7\n"
PAIRDET_HDR = ("MJD,RA,Dec,mag,trail_len,trail_PA,sigmag,sig_across,sig_along,"
               "image,idstring,band,obscode,known_obj,det_qual,origindex")
SOLARDAY = 86400.0


def _run(cmd, log_path=None, env=None):
    full_env = None
    if env:
        import os
        full_env = {**os.environ, **{k: str(v) for k, v in env.items()}}
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=full_env)
    if log_path is not None:
        Path(log_path).write_text(res.stdout + "\n==STDERR==\n" + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(map(str, cmd))}\n"
                           f"{(res.stdout + res.stderr)[-1500:]}")
    return res.stdout


def harvest_observer_images(dets, work, earth, obscode_file, maxvel=40.0):
    """Run make_tracklets on the trail centroids ONLY to get the per-image observer state vectors
    (imgs.txt); its tracklets are discarded. We therefore force near-zero pairing (tiny maxtime +
    maxvel) so make_tracklets does NOT do its O(N^2) all-pairs tracklet search over the full night
    (which on ~20k detections runs for many minutes) — the imgs file is written regardless.
    Returns a frame indexed by image order: mjd, ra, dec, obscode, X,Y,Z, VX,VY,VZ (observer state)."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cen = dets[["detid"]].copy()
    cen["mjd"] = dets.mjd.to_numpy()
    cen["ra"] = 0.5 * (dets.ra0.to_numpy() + dets.ra1.to_numpy())
    cen["dec"] = 0.5 * (dets.dec0.to_numpy() + dets.dec1.to_numpy())
    cen["mag"] = dets.get("mag", pd.Series(21.0, index=dets.index)).fillna(21.0).to_numpy()
    cen["band"] = dets.get("band", pd.Series("r", index=dets.index)).astype(str).str[:1].to_numpy()
    cen["obscode"] = dets.get("obscode", pd.Series("I11", index=dets.index)).astype(str).to_numpy()
    cen_csv = work / "centroids.csv"; cen[CENTROID_COLS].to_csv(cen_csv, index=False)
    (work / "centroids.colformat.txt").write_text(CENTROID_COLFMT)
    imgs = work / "cen_imgs.txt"
    # maxtime/maxvel tiny on purpose: we only want imgs.txt (observer states), not tracklets, so
    # suppress the expensive O(N^2) pairing. (-maxtime in hours; 1e-4 h ~= 0.4 s << inter-visit gap.)
    _run([BIN / "make_tracklets", "-dets", cen_csv, "-outimgs", imgs,
          "-pairdets", work / "cen_pairdets.csv", "-tracklets", work / "cen_tracklets.csv",
          "-trk2det", work / "cen_trk2det.csv", "-colformat", work / "centroids.colformat.txt",
          "-imrad", 5.0, "-maxtime", 1e-4, "-mintime", 0.0, "-maxGCR", 0.1,
          "-minvel", 0.0, "-maxvel", 0.01, "-mintrkpts", 2,
          "-earth", earth, "-obscode", obscode_file], work / "harvest_mktracklets.log")
    rows = []
    for ln in open(imgs):
        ln = ln.strip()
        if not ln or not ln[0].isdigit():
            continue
        p = ln.replace(",", " ").split()
        # MJD RA Dec obscode X Y Z VX VY VZ startind endind exptime
        rows.append(dict(mjd=float(p[0]), ra=float(p[1]), dec=float(p[2]), obscode=p[3],
                         X=float(p[4]), Y=float(p[5]), Z=float(p[6]),
                         VX=float(p[7]), VY=float(p[8]), VZ=float(p[9])))
    return pd.DataFrame(rows)


def author_native(dets, imgframe, work, exptime_s=30.0):
    """Write imgs.txt / pairdets.csv / tracklets.csv / trk2det.csv for the one-trail=one-tracklet
    model. Each visit image -> a start image and an end image (observer propagated by V*exptime).
    Returns (imgs, pairdets, tracklets, trk2det) paths."""
    work = Path(work)
    dt = exptime_s / SOLARDAY
    # Map each detection to its visit image (nearest harvested image MJD).
    img_mjd = imgframe.mjd.to_numpy()
    det_img = np.array([int(np.argmin(np.abs(img_mjd - m))) for m in dets.mjd.to_numpy()])

    # Build the 2-image-per-visit table; only keep visit-images that actually carry detections.
    # The trail spans the exposure, centred on the mid-MJD (DATE-AVG). So endpoint0 is at
    # mid-exptime/2 (shutter open) and endpoint1 at mid+exptime/2 (shutter close); the observer
    # state is propagated by +/- (exptime/2) along its velocity. (Centred, not +exptime offset.)
    half = 0.5 * dt
    half_s = 0.5 * exptime_s
    used = sorted(set(det_img.tolist()))
    start_idx, end_idx = {}, {}
    img_rows = []
    for vi in used:
        s = imgframe.iloc[vi]
        start_idx[vi] = len(img_rows)
        img_rows.append((s.mjd - half, s.ra, s.dec, s.obscode,
                         s.X - s.VX * half_s, s.Y - s.VY * half_s, s.Z - s.VZ * half_s,
                         s.VX, s.VY, s.VZ))
        end_idx[vi] = len(img_rows)
        img_rows.append((s.mjd + half, s.ra, s.dec, s.obscode,
                         s.X + s.VX * half_s, s.Y + s.VY * half_s, s.Z + s.VZ * half_s,
                         s.VX, s.VY, s.VZ))
    imgs = work / "imgs.txt"
    with open(imgs, "w") as f:
        for k, r in enumerate(img_rows):
            mjd, ra, dec, obsc, X, Y, Z, VX, VY, VZ = r
            f.write(f"{mjd:.8f} {ra:.7f} {dec:.7f} {obsc} {X:.4f} {Y:.4f} {Z:.4f} "
                    f"{VX:.6f} {VY:.6f} {VZ:.6f} {k} {k} {exptime_s:.4f}\n")

    # pairdets: two endpoint rows per trail; tracklets: one per trail linking them.
    pd_rows, trk_rows, t2d_rows = [], [], []
    band = dets.get("band", pd.Series("r", index=dets.index)).astype(str).str[:1].to_numpy()
    obsc = dets.get("obscode", pd.Series("I11", index=dets.index)).astype(str).to_numpy()
    mag = dets.get("mag", pd.Series(21.0, index=dets.index)).fillna(21.0).to_numpy()
    detid = dets.get("detid", pd.Series(range(len(dets)), index=dets.index)).astype(str).to_numpy()
    di = 0
    for k in range(len(dets)):
        vi = det_img[k]; si = start_idx[vi]; ei = end_idx[vi]
        m0 = img_rows[si][0]; m1 = img_rows[ei][0]
        r0, d0 = float(dets.ra0.iat[k]), float(dets.dec0.iat[k])
        r1, d1 = float(dets.ra1.iat[k]), float(dets.dec1.iat[k])
        for (mjd, ra, dec, img) in ((m0, r0, d0, si), (m1, r1, d1, ei)):
            pd_rows.append(f"{mjd:.8f},{ra:.7f},{dec:.7f},{mag[k]:.3f},0.00,0.00,1.0000,1.0000,"
                           f"1.0000,{img},{detid[k]},{band[k]},{obsc[k]},-1,-1,{di}")
            di += 1
        a, b = di - 2, di - 1
        trk_rows.append(f"{si},{r0:.7f},{d0:.7f},{ei},{r1:.7f},{d1:.7f},2,{k}")
        t2d_rows.append((k, a)); t2d_rows.append((k, b))

    pairdets = work / "pairdets.csv"
    with open(pairdets, "w") as f:
        f.write("#" + PAIRDET_HDR + "\n"); f.write("\n".join(pd_rows) + "\n")
    tracklets = work / "tracklets.csv"
    with open(tracklets, "w") as f:
        f.write("#Image1,RA1,Dec1,Image2,RA2,Dec2,npts,trk_ID\n"); f.write("\n".join(trk_rows) + "\n")
    trk2det = work / "trk2det.csv"
    with open(trk2det, "w") as f:
        f.write("#trk_ID,detnum\n")
        for t, d in t2d_rows:
            f.write(f"{t},{d}\n")
    return imgs, pairdets, tracklets, trk2det


def _patch_omp_orbit_incl(sumf):
    """heliolinc_omp writes a 31-col sum.csv missing `orbit_incl` (between orbit_e and orbit_MJD);
    regular heliolinc writes 32 and the link_purify/link_planarity reader expects 32. Insert a
    0.0 orbit_incl column (col index 23) so the post-processors can read heliolinc_omp output.
    No-op if already 32 columns."""
    lines = Path(sumf).read_text().splitlines()
    if not lines:
        return
    hdr = lines[0].lstrip("#").split(",")
    if "orbit_incl" in hdr or len(hdr) != 31:
        return  # already correct / unexpected layout — leave alone
    j = hdr.index("orbit_e") + 1   # insert right after orbit_e
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        pre = "#" if ln.startswith("#") else ""
        f = ln.lstrip("#").split(",")
        f.insert(j, "orbit_incl" if i == 0 else "0.000000")
        out.append(pre + ",".join(f))
    Path(sumf).write_text("\n".join(out) + "\n")


def link_purify_sharded(imgs, pairdets, sumf, c2d, work, *, nshard=200, maxproc=110, simptype=1,
                        max_astrom_rms=1.5, rejfrac=0.2, minobsnights=1, minpointnum=6, maxrms=1.5e6):
    """Parallelise link_purify by sharding its input candidate clusters across `nshard` link_purify
    processes (>= cores; concurrency capped at maxproc), then ONE final link_purify combines the
    reduced shard outputs for the global cross-shard dedup.

    ROUND-ROBIN sharding (cluster c -> shard c%nshard, new id c//nshard) so each shard gets a mix of
    cluster sizes — contiguous blocks would dump all the giant impure blobs of one hypothesis region
    into a few shards, which then straggle for hours and set the wall time. heliolinc clusternums are
    0-based sequential and clust2det is grouped by clusternum, so the remap is pure arithmetic.
    Returns (final_sum, final_c2d) paths."""
    import subprocess as sp
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    sdf = pd.read_csv(sumf); sdf.columns = [c.lstrip("#") for c in sdf.columns]
    cdf = pd.read_csv(c2d);  cdf.columns = [c.lstrip("#") for c in cdf.columns]
    ccol = sdf.columns[0]; cc_id = cdf.columns[0]
    n = len(sdf)
    if n == 0:
        return None, None
    nshard = max(1, min(nshard, n))
    cnum = cdf[cc_id].to_numpy()

    def _shard_pair(k):
        idx = np.arange(k, n, nshard)                       # cluster numbers assigned to shard k
        if len(idx) == 0:
            return None
        ss = sdf.iloc[idx].copy(); ss[ccol] = np.arange(len(idx))   # renumber 0..M-1
        m = (cnum % nshard == k); cc = cdf.iloc[np.where(m)[0]].copy(); cc[cc_id] = cc[cc_id] // nshard
        sp_ = work / f"shard_{k}_sum.csv"; cp_ = work / f"shard_{k}_c2d.csv"
        with open(sp_, "w") as f:
            f.write("#" + ",".join(sdf.columns) + "\n"); ss.to_csv(f, header=False, index=False)
        with open(cp_, "w") as f:
            f.write("#" + ",".join(cdf.columns) + "\n"); cc.to_csv(f, header=False, index=False)
        return sp_, cp_

    def _cmd(lf, osum, oc2d):
        return [str(BIN / "link_purify"), "-imgs", str(imgs), "-pairdet", str(pairdets),
                "-lflist", str(lf), "-simptype", "1", "-max_astrom_rms", str(max_astrom_rms),
                "-rejfrac", str(rejfrac), "-minobsnights", str(minobsnights),
                "-minpointnum", str(minpointnum), "-ptpow", "3", "-nightpow", "0", "-timepow", "0",
                "-rmspow", "2", "-maxrms", str(maxrms), "-outsum", str(osum), "-clust2det", str(oc2d)]

    # Launch shards with a concurrency cap of maxproc.
    shard_out = []; running = []
    def _drain_one():
        p, osum, oc2d, log = running.pop(0); p.wait(); log.close()
        if Path(osum).exists() and sum(1 for _ in open(osum)) > 1:
            shard_out.append((osum, oc2d))
    for k in range(nshard):
        pair = _shard_pair(k)
        if pair is None:
            continue
        sp_, cp_ = pair
        lf = work / f"shard_{k}.lflist"; lf.write_text(f"{sp_.resolve()} {cp_.resolve()}\n")
        osum = work / f"shard_{k}_LPsum.csv"; oc2d = work / f"shard_{k}_LPc2d.csv"
        log = open(work / f"shard_{k}.log", "w")
        running.append((sp.Popen(_cmd(lf, osum, oc2d), stdout=log, stderr=sp.STDOUT), osum, oc2d, log))
        while len(running) >= maxproc:
            _drain_one()
    while running:
        _drain_one()
    if not shard_out:
        return None, None

    # Final combine: one link_purify over all reduced shard outputs (global dedup).
    finlf = work / "final.lflist"
    finlf.write_text("".join(f"{Path(s).resolve()} {Path(c).resolve()}\n" for s, c in shard_out))
    psum = work / "LPsum.csv"; pc2d = work / "LPclust2det.csv"
    _run([BIN / "link_purify", "-imgs", imgs, "-pairdet", pairdets, "-lflist", finlf,
          "-simptype", 1, "-max_astrom_rms", max_astrom_rms, "-rejfrac", rejfrac,
          "-minobsnights", minobsnights, "-minpointnum", minpointnum,
          "-ptpow", 3, "-nightpow", 0, "-timepow", 0, "-rmspow", 2, "-maxrms", maxrms,
          "-outsum", psum, "-clust2det", pc2d], work / "link_purify_final.log")
    return psum, pc2d


def link_night(dets_night, work, earth, obscode_file, heliodist, *, exptime_s=30.0, mjdref=None,
               clustrad=5e5, clustchangerad=0.5, npt=3, minobsnights=1, mintimespan=0.02,
               mingeodist=0.01, maxgeodist=3.0, geologstep=1.5, mingeoobs=0.005, minimpactpar=50000.0,
               minpointnum=6, max_astrom_rms=1.5, maxrms=1.5e6, rejfrac=0.2, omp_threads=32, nshard=200):
    """Author native files for one night and run heliolinc -> link_purify. Returns sum, c2d, pairdets.

    Parameters = heliolinx README NEO recipe + Holman 2018, set for single-night NEO work:
    clustrad 5e5 km (scales with geodist, floored by clustchangerad 0.5AU), npt 3 (npt 2 = README's
    'unreasonable FP'; our NEOs span >=3 visits), geo cuts mingeodist 0.01 / mingeoobs 0.005 /
    minimpactpar 5e4 (NEO geometry; suppress near-observer artifacts), minobsnights 1 (same-night),
    mintimespan 0.02 d. link_purify (NOT link_planarity: README says the latter's coplanarity
    pre-screen can drop good points) with max_astrom_rms 1.5" (CNN endpoints ~0.5-1"; cuts astrometric
    INCONSISTENCY, not the expected short-arc orbit degeneracy) + exponents ptpow3/nightpow0/timepow0/rmspow2.
    """
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    imgframe = harvest_observer_images(dets_night, work, earth, obscode_file)
    if imgframe.empty:
        return None, None, None
    imgs, pairdets, tracklets, trk2det = author_native(dets_night, imgframe, work, exptime_s=exptime_s)
    if mjdref is None:
        mjdref = 0.5 * (dets_night.mjd.min() + dets_night.mjd.max())

    sumf = work / "sum.csv"; c2d = work / "clust2det.csv"
    # heliolinc_omp = OpenMP multi-threaded build (same flags); ~13k same-night tracklets is far too
    # slow single-threaded. Threads via OMP_NUM_THREADS.
    _run([BIN / "heliolinc_omp", "-imgs", imgs, "-pairdets", pairdets, "-tracklets", tracklets,
          "-trk2det", trk2det, "-mjd", f"{mjdref:.6f}", "-obspos", earth, "-heliodist", heliodist,
          "-clustrad", clustrad, "-clustchangerad", clustchangerad, "-npt", npt,
          "-mingeodist", mingeodist, "-maxgeodist", maxgeodist, "-geologstep", geologstep,
          "-mingeoobs", mingeoobs, "-minimpactpar", minimpactpar,
          "-minobsnights", minobsnights, "-mintimespan", mintimespan,
          "-outsum", sumf, "-clust2det", c2d], work / "heliolinc.log",
         env={"OMP_NUM_THREADS": omp_threads})
    if not sumf.exists() or sum(1 for _ in open(sumf)) <= 1:
        return None, None, pairdets
    _patch_omp_orbit_incl(sumf)   # heliolinc_omp omits the orbit_incl column the linkers' reader needs

    # link_purify is single-threaded with no _omp build (its core is independent per-cluster Herget
    # fits). Parallelise by SHARDING (Heinze's own model; README runs _p00/_p01/.. shards): split the
    # heliolinc candidate clusters into nshard pieces, link_purify each in parallel, then one final
    # link_purify over the (small) reduced outputs for the global cross-shard dedup.
    psum, pc2d = link_purify_sharded(
        imgs, pairdets, sumf, c2d, work, nshard=nshard,
        simptype=1, max_astrom_rms=max_astrom_rms, rejfrac=rejfrac, minobsnights=minobsnights,
        minpointnum=minpointnum, maxrms=maxrms)
    if psum is None or not psum.exists() or sum(1 for _ in open(psum)) <= 1:
        return None, None, pairdets
    return pd.read_csv(psum), pd.read_csv(pc2d), pairdets


def crossmatch_linkages(lplsum, lplc2d, pairdets_path, dets, known, tol_arcsec=5.0, tol_day=0.02):
    """For each linkage, gather its member detections (via clust2det->pairdets origindex->dets) and
    match their (ra,dec,mjd) to the known-SSObject ephemerides. Returns per-linkage df with the
    best-matching ObjID (or NEW) + match fraction."""
    pdets = pd.read_csv(pairdets_path)
    pdets.columns = [c.lstrip("#") for c in pdets.columns]   # header is '#MJD,RA,Dec,...'
    ccol = lplc2d.columns[0]; dcol = lplc2d.columns[1]
    kra = known.ra.to_numpy(); kdec = known.dec.to_numpy(); kmjd = known.mjd.to_numpy()
    kobj = known.ObjID.astype(str).to_numpy()
    cosd = np.cos(np.radians(np.clip(kdec, -89, 89)))
    rows = []
    for cnum, grp in lplc2d.groupby(ccol):
        pr = pdets.iloc[grp[dcol].to_numpy()]
        ra = pr.RA.to_numpy(); dec = pr.Dec.to_numpy(); mjd = pr.MJD.to_numpy()
        hits = []
        for r, d, m in zip(ra, dec, mjd):
            tsel = np.abs(kmjd - m) <= tol_day
            if not tsel.any():
                continue
            dra = (kra[tsel] - r) * np.cos(np.radians(d)); ddec = kdec[tsel] - d
            sep = np.hypot(dra, ddec) * 3600.0
            j = np.argmin(sep)
            if sep[j] <= tol_arcsec:
                hits.append(kobj[tsel][j])
        if hits:
            vc = pd.Series(hits).value_counts()
            rows.append(dict(cluster=cnum, n_det=len(pr), match_obj=vc.index[0],
                             match_frac=vc.iloc[0] / len(pr), status="CONFIRMED"))
        else:
            rows.append(dict(cluster=cnum, n_det=len(pr), match_obj="", match_frac=0.0, status="NEW"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="ADCNN(+Veres) catalog: mjd,ra0,dec0,ra1,dec1[,visit,mag,band,obscode,detid]")
    ap.add_argument("--known", default=str(HL / "run_night/known.csv"))
    ap.add_argument("--earth", default=str(AUX / "Earth1day2020s_02a.csv"))
    ap.add_argument("--obscode-file", default=str(AUX / "ObsCodesNew.txt"))
    # NEO heliocentric grid (r 1.1-5.9 AU). 'aa' (1884 hyps) is the working set per Holman/heliolinx
    # (a small grid finds the majority; sensitivity is error-limited, not grid-limited) — 'ab'/'ac'
    # (3214/10936 hyps) only chase the last few %. All NEO grids floor at r=1.1; the 3 target NEOs
    # are at rhelio 1.08-1.22 (NY2 at 1.079 links via the r=1.1 bin within clustrad tolerance).
    ap.add_argument("--heliodist", default=str(AUX / "hypotheses/NEO/hihyp00aa_neo.txt"))
    ap.add_argument("--out", default=str(HL / "run_night/link"))
    ap.add_argument("--exptime", type=float, default=30.0, help="LSST exposure (s); Butler VisitInfo = 30.0")
    ap.add_argument("--clustrad", type=float, default=1e5, help="km; Heinze's NEO value is 5e5 (for CULLED input). On our UNCULLED dense input 5e5 explodes the LINKER itself (heliolinc_omp hit 209GB RAM, didn't finish). 1e5 is where heliolinc completes (~1.47M candidates); that's the working value for unculled input. heliolinc default is also 1e5.")
    ap.add_argument("--clustchangerad", type=float, default=0.5, help="AU; floor for clustrad scaling")
    ap.add_argument("--npt", type=int, default=3, help="min tracklets/cluster; README: npt 2 = unreasonable FP")
    ap.add_argument("--minobsnights", type=int, default=1, help="1 = same-night")
    ap.add_argument("--mintimespan", type=float, default=0.02, help="day; min arc (~0.5hr, < one night)")
    ap.add_argument("--mingeodist", type=float, default=0.01, help="AU; README NEO")
    ap.add_argument("--maxgeodist", type=float, default=1.0, help="AU; fast same-night movers are CLOSE (high deg/day => near Earth). Caps the large-geodist bins where clustrad balloons (5e5*geodist) into 40M-detection blobs -> OOM")
    ap.add_argument("--mingeoobs", type=float, default=0.005, help="AU; README NEO")
    ap.add_argument("--minimpactpar", type=float, default=50000.0, help="km; README NEO")
    ap.add_argument("--minpointnum", type=int, default=6, help="link_purify; 2*npt (3 tracklets x 2 endpoints)")
    ap.add_argument("--max-astrom-rms", type=float, default=1.5, help="arcsec; cuts astrometric inconsistency")
    ap.add_argument("--score-min", type=float, default=0.0, help="keep ADCNN score >= this (0 = keep all; do NOT over-gate, faint NEOs have low score)")
    ap.add_argument("--snr-min", type=float, default=0.0, help="keep Veres SNR >= this. DEFAULT 0 (OFF): the whole point of ADCNN is faint sub-5-sigma movers; do NOT gate SNR.")
    ap.add_argument("--art-frac-max", type=float, default=0.3, help="LSST diffim mask cut: drop detections whose trail is >= this fraction on an artifact plane (SPIKE/SAT/CR/STREAK/DETECTED_NEGATIVE/...). Needs the m_*/art_frac columns from mask_flags.py. TP-safe (NEOs survive); removes ~15-20% of FP. 0=off.")
    ap.add_argument("--len-db-min", type=float, default=6.0, help="px; fast-mover trail floor (>=1 deg/day)")
    ap.add_argument("--len-db-max", type=float, default=50.0, help="px; cut long-streak FP. NEOs at 1-3 deg/day trail ~6-18px (object motion over 30s) + seeing; >50px = satellites/spikes")
    ap.add_argument("--tol-arcsec", type=float, default=5.0)
    ap.add_argument("--tol-day", type=float, default=0.02)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.dets)
    if "score" in d and a.score_min > 0:
        d = d[d.score >= a.score_min]
    if "len_db" in d and a.len_db_min > 0:
        d = d[d.len_db >= a.len_db_min]
    if "len_db" in d and a.len_db_max > 0:
        d = d[d.len_db <= a.len_db_max]
    if "snr" in d and a.snr_min > 0:
        d = d[d.snr >= a.snr_min]
    if "art_frac" in d and a.art_frac_max > 0:
        n0 = len(d); d = d[d.art_frac < a.art_frac_max]
        print(f"[realdata] LSST mask cut (art_frac<{a.art_frac_max}): {n0} -> {len(d)} ({100*(1-len(d)/max(n0,1)):.0f}% dropped as masked artifacts)", flush=True)
    need = ["mjd", "ra0", "dec0", "ra1", "dec1"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit(f"--dets missing endpoint columns {miss}")
    if "detid" not in d:
        d = d.reset_index(drop=True); d["detid"] = d.index
    known = pd.read_csv(a.known)
    d["night"] = np.floor(d.mjd - 0.5).astype(int)
    nights = sorted(d.night.unique())
    print(f"[realdata] {len(d)} detections, {len(nights)} night(s) | heliodist={Path(a.heliodist).name} "
          f"| npt={a.npt} minobsnights={a.minobsnights}", flush=True)

    all_link = []
    for night in nights:
        dn = d[d.night == night].reset_index(drop=True)
        try:
            lplsum, lplc2d, pdets = link_night(
                dn, out / f"night_{night}", a.earth, a.obscode_file, a.heliodist,
                exptime_s=a.exptime, clustrad=a.clustrad, clustchangerad=a.clustchangerad, npt=a.npt,
                minobsnights=a.minobsnights, mintimespan=a.mintimespan, mingeodist=a.mingeodist,
                maxgeodist=a.maxgeodist, mingeoobs=a.mingeoobs, minimpactpar=a.minimpactpar,
                minpointnum=a.minpointnum, max_astrom_rms=a.max_astrom_rms)
        except RuntimeError as e:
            print(f"  night {night}: FAILED {e}", flush=True); continue
        if lplsum is None:
            print(f"  night {night}: {len(dn)} dets -> 0 linkages", flush=True); continue
        xm = crossmatch_linkages(lplsum, lplc2d, pdets, dn, known,
                                 tol_arcsec=a.tol_arcsec, tol_day=a.tol_day)
        xm["night"] = night
        all_link.append(xm)
        nconf = int((xm.status == "CONFIRMED").sum()); nnew = int((xm.status == "NEW").sum())
        objs = sorted(xm[xm.status == "CONFIRMED"].match_obj.unique())
        print(f"  night {night}: {len(dn)} dets -> {len(xm)} linkages | {nconf} CONFIRMED "
              f"({len(objs)} objs: {', '.join(objs[:8])}{'...' if len(objs)>8 else ''}) | {nnew} NEW", flush=True)

    if all_link:
        L = pd.concat(all_link, ignore_index=True); L.to_csv(out / "linkages.csv", index=False)
        conf = sorted(L[L.status == "CONFIRMED"].match_obj.unique())
        print(f"\n[realdata] RESULT: {len(L)} linkages | "
              f"{int((L.status=='CONFIRMED').sum())} CONFIRMED ({len(conf)} distinct known objects) | "
              f"{int((L.status=='NEW').sum())} NEW candidates -> {out/'linkages.csv'}", flush=True)
        print(f"[realdata] confirmed objects: {', '.join(conf)}", flush=True)
    else:
        print("\n[realdata] no linkages on any night", flush=True)


if __name__ == "__main__":
    main()
