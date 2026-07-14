"""Build a REAL-asteroid test set (``DATA_DIFFIM/test_real``) from genuine
LSST difference images — no synthetic injection.

This is the non-injection sibling of :mod:`simulate`. For each
``(visit, detector)`` of a catalogued fast-moving asteroid it fetches the
science PVI + template, runs AlardLupton subtraction + DetectAndMeasure on the
*uninjected* PVI, and writes the real difference image (with the real trail
already in it) plus the catalogued empties for a false-positive baseline.

Two CLI subcommands (run inside the LSST stack env, ``setup lsst_distrib``)::

    python -m ADCNN.data.dataset_creation.build_real scan  --real-csv ... --out-dir ...
    python -m ADCNN.data.dataset_creation.build_real build --manifest ... --out ...

``scan`` probes the Butler for which ``(visit,detector)`` pairs are buildable
(PVI + single_visit_star_footprints + an overlapping same-band
template_coadd) and writes ``manifest.csv`` (asteroid panels first, then
N_EMPTY no-asteroid panels for the FP measurement). ``build`` consumes the
manifest and produces ``test.h5`` / ``test.csv`` / ``panels.csv`` byte-
compatible with the synthetic ``test`` sets so the same evaluation
path (the ``Evaluation/Evaluation_Real`` notebook) applies.

Key correctness notes
---------------------
* The catalog ``x,y``/``angle`` columns are STALE w.r.t. the DP2 reprocessing
  (offsets of 150-300 px). Trails are reconstructed from the ephemeris
  ``RA_deg/Dec_deg`` + ``RARateCosDec_deg_day``/``DecRate_deg_day`` through
  the PVI WCS (endpoints at exposure start and start+exposure — the
  convention validated on the diffim itself).
* A single Butler registry connection is NOT thread-safe (psycopg2 named
  cursor corruption) — the availability scan keeps one Butler per thread.
* PVIs are not all the same size in this collection; panels are written
  origin-aligned and zero-padded to the max dimension.
* ``real_labels`` excludes the asteroid-matched diaSource footprint (it is
  the network's channel-3 input, so the target must not be flagged "known").
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import threading
import traceback
import warnings
from multiprocessing import Manager

import numpy as np
import pandas as pd

# --- silence known-harmless stack noise (matches simulate) ----
for _n in ("lsst", "lsst.ip.diffim", "lsst.detectAndMeasure",
           "lsst.meas.algorithms", "ip_diffim_DipoleFit"):
    logging.getLogger(_n).setLevel(logging.ERROR)
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"lsst\.meas\.algorithms\.maskStreaks")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module=r"astropy\.units\.quantity")

import h5py  # noqa: E402
import lsst.geom as geom  # noqa: E402
from lsst.daf.butler import Butler  # noqa: E402
from lsst.geom import Point2D  # noqa: E402
from lsst.pipe.base import (NoWorkFound, UnprocessableDataError,  # noqa: E402
                            UpstreamFailureNoWorkFound)

from ADCNN.utils.helpers import draw_one_line
from ADCNN.data.dataset_creation.photometry import ensure_dir
from ADCNN.data.dataset_creation.butler_tasks import (
    fetch_diffim_inputs, run_detect_diffim, run_subtract)
from ADCNN.data.dataset_creation.simulate import (
    footprints_to_label_mask, format_dataId)

_SKIP = (Exception, NoWorkFound, UnprocessableDataError,
         UpstreamFailureNoWorkFound)

# DP2 Butler — overridable via CLI.
DEF_REPO = os.environ.get("BUTLER_REPO", "main")   # dp2_prep purged 2026-06; live diffims = main DRP
DEF_STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
DEF_STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
DEF_SKYMAP = "lsst_cells_v2"
N_SCIENCE_DETECTORS = 189  # LSSTCam science CCDs 0..188

CSV_COLS = ["ra", "dec", "source_type", "trail_length", "mag", "beta", "visit",
            "detector", "integrated_mag", "PSF_mag", "SNR", "physical_filter",
            "x", "y", "SNR_estimation", "m5_local", "m5_detector",
            "stack_detection", "stack_mag", "stack_mag_err", "stack_snr",
            "ObjID", "speed_deg_day",
            "x_csv", "y_csv", "beta_csv", "trail_length_csv",
            "wcs_len_px", "image_id"]


# ======================================================================
# WCS + ephemeris trail reconstruction
# ======================================================================
def reconstruct_trail(wcs, row, *, bbox_w, bbox_h, min_len_px=4.0):
    """Trail in PVI pixel space from the ephemeris (RA/Dec + sky rates).

    The ephemeris RA_deg/Dec_deg is the trail START (~exposure start); the
    asteroid sweeps to RA+rate*texp over the exposure. ``beta`` is the image
    position angle ``atan2(dy, dx)`` matching ``draw_one_line``. Returns
    ``{x, y, beta, trail_length, on_ccd, wcs_len_px}``.
    """
    ra = float(row["RA_deg"]); dec = float(row["Dec_deg"])
    texp = float(row["exposure_time"]) / 86400.0
    cd = max(np.cos(np.deg2rad(dec)), 1e-6)
    dra = float(row["RARateCosDec_deg_day"]) / cd
    ddec = float(row["DecRate_deg_day"])
    p1 = wcs.skyToPixel(geom.SpherePoint(ra, dec, geom.degrees))
    p2 = wcs.skyToPixel(geom.SpherePoint(ra + dra * texp,
                                         dec + ddec * texp, geom.degrees))
    x1, y1 = p1.getX(), p1.getY()
    x2, y2 = p2.getX(), p2.getY()
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    wcs_len = float(np.hypot(x2 - x1, y2 - y1))
    beta = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    return {"x": float(cx), "y": float(cy), "beta": beta,
            "trail_length": float(max(wcs_len, min_len_px)),
            "on_ccd": bool((0.0 <= cx < bbox_w) and (0.0 <= cy < bbox_h)),
            "wcs_len_px": wcs_len}


# ======================================================================
# Phase A — Butler availability scan
# ======================================================================
_TLS = threading.local()


def _scan_butler(repo, coll):
    b = getattr(_TLS, "butler", None)
    if b is None:
        b = Butler(repo, collections=coll)
        _TLS.butler = b
    return b


def _check_pair(visit, detector, repo, coll, stage3, skymap):
    """Availability of one (visit,detector): pvi + svsf + same-band template."""
    b = _scan_butler(repo, coll)
    did = {"instrument": "LSSTCam", "visit": int(visit),
           "detector": int(detector)}
    out = {"visit": visit, "detector": detector, "band": None, "ok": False}
    try:
        if b.registry.findDataset("preliminary_visit_image", dataId=did,
                                  collections=coll) is None:
            return out
        if b.registry.findDataset("single_visit_star_footprints", dataId=did,
                                  collections=coll) is None:
            return out
        exp = b.registry.expandDataId(did)
        band = exp.get("band")
        out["band"] = band
        allt = b.registry.queryDatasets(
            "template_coadd",
            where="skymap = :skymap AND patch.region OVERLAPS :region",
            bind={"skymap": skymap, "region": exp.region},
            collections=[stage3], findFirst=True)
        out["ok"] = sum(1 for x in allt if x.dataId.get("band") == band) > 0
    except Exception:
        return out
    return out


def scan_availability(real_csv, out_dir, *, repo=DEF_REPO, stage3=DEF_STAGE3,
                      stage2=DEF_STAGE2, skymap=DEF_SKYMAP, n_empty=150,
                      workers=32, seed=123):
    """Probe the Butler and write ``out_dir/manifest.csv`` (asteroid panels
    then ``n_empty`` no-asteroid panels)."""
    coll = [stage3, stage2]
    r = pd.read_csv(real_csv)
    r["visit"] = r["FieldID"].astype("int64")
    r["detector"] = r["detector"].astype("int64")

    def _scan(pairs, label):
        print(f"[scan] {label}: {len(pairs)} pairs, {workers} workers",
              flush=True)
        res, done = [], 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_check_pair, v, d, repo, coll, stage3, skymap):
                    (v, d) for v, d in pairs}
            for fut in concurrent.futures.as_completed(futs):
                res.append(fut.result()); done += 1
                if done % 250 == 0:
                    print(f"  [{done}/{len(pairs)}] "
                          f"ok={sum(x['ok'] for x in res)}", flush=True)
        return pd.DataFrame(res)

    ast = r.groupby(["visit", "detector"]).size().reset_index(name="n_ast")
    av = _scan(list(zip(ast["visit"], ast["detector"])), "asteroid")
    av = av.merge(ast, on=["visit", "detector"], how="left")
    ast_ok = av[av["ok"]].copy(); ast_ok["role"] = "asteroid"
    obj_ok = r.merge(ast_ok[["visit", "detector"]], on=["visit", "detector"])
    print(f"\n[asteroid] {len(ast_ok)}/{len(ast)} pairs buildable | "
          f"{obj_ok['ObjID'].nunique()}/{r['ObjID'].nunique()} objects",
          flush=True)

    rng = random.Random(int(seed))
    real_pairs = set(zip(r["visit"], r["detector"]))
    cand = [(v, d) for v in sorted(ast_ok["visit"].unique())
            for d in range(N_SCIENCE_DETECTORS) if (v, d) not in real_pairs]
    rng.shuffle(cand)
    cand = cand[:max(n_empty * 8, 1200)]
    ev = _scan(cand, "empty-cand")
    emp_ok = ev[ev["ok"]].head(n_empty).copy()
    emp_ok["n_ast"] = 0; emp_ok["role"] = "empty"
    print(f"[empty] {len(emp_ok)}/{n_empty} empty panels secured", flush=True)

    cols = ["visit", "detector", "role", "n_ast", "band"]
    man = pd.concat([ast_ok[cols], emp_ok[cols]], ignore_index=True)
    ensure_dir(out_dir)
    man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    av.to_csv(os.path.join(out_dir, "availability_asteroid.csv"), index=False)
    print(f"[manifest] {len(man)} panels "
          f"({(man.role=='asteroid').sum()} asteroid + "
          f"{(man.role=='empty').sum()} empty) -> {out_dir}/manifest.csv")


# ======================================================================
# Phase B — non-injection real-diffim builder
# ======================================================================
class _Dim:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = int(x); self.y = int(y)


def _match_stack_to_truth(dia, photoCalib, truth, n_obj):
    """Per truth object k (mask==k+1) find best-overlapping diaSource; return
    det/snr/mag arrays + matched diaSource indices (excluded from labels)."""
    H, W = truth.shape
    det = np.zeros(n_obj, bool)
    dmag = np.full(n_obj, np.nan); dmagerr = np.full(n_obj, np.nan)
    dsnr = np.full(n_obj, np.nan); matched = set()
    if len(dia) == 0 or truth.max() == 0:
        return det, dmag, dmagerr, dsnr, matched
    try:
        mags = photoCalib.instFluxToMagnitude(dia, "base_PsfFlux")
    except Exception:
        mags = None
    ys, xs = np.nonzero(truth)
    ids = truth[ys, xs] - 1
    pix = [[] for _ in range(n_obj)]
    for y, x, i in zip(ys, xs, ids):
        if 0 <= i < n_obj:
            pix[i].append((y, x))
    for k in range(n_obj):
        if not pix[k]:
            continue
        yy = np.array([p[0] for p in pix[k]])
        xx = np.array([p[1] for p in pix[k]])
        y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
        th = np.zeros((y1 - y0 + 1, x1 - x0 + 1), bool)
        for y, x in pix[k]:
            th[y - y0, x - x0] = True
        best_ov, best = 0, None
        for idx in range(len(dia)):
            fp = dia[idx].getFootprint()
            bb = fp.getBBox()
            if (bb.getEndX() < x0 or bb.getBeginX() > x1
                    or bb.getEndY() < y0 or bb.getBeginY() > y1):
                continue
            fm = np.zeros_like(th)
            for span in fp.spans:
                y = span.getY()
                if y < y0 or y > y1:
                    continue
                sx0 = max(span.getX0(), x0); sx1 = min(span.getX1(), x1)
                if sx0 <= sx1:
                    fm[y - y0, sx0 - x0: sx1 - x0 + 1] = True
            ov = int((fm & th).sum())
            if ov > best_ov:
                best_ov, best = ov, idx
        if best is not None and best_ov >= 1:
            det[k] = True
            matched.add(best)
            if mags is not None:
                dmag[k] = mags[best, 0]; dmagerr[k] = mags[best, 1]
            try:
                f = float(dia[best].get("base_PsfFlux_instFlux"))
                fe = float(dia[best].get("base_PsfFlux_instFluxErr"))
                dsnr[k] = f / fe if (np.isfinite(f) and np.isfinite(fe)
                                     and fe > 0) else np.nan
            except Exception:
                pass
    return det, dmag, dmagerr, dsnr, matched


def _labels_excluding(dia, matched, dims):
    """footprints_to_label_mask over diaSources EXCEPT asteroid-matched ones."""
    H, W = int(dims.y), int(dims.x)
    if len(dia) == 0:
        return np.zeros((H, W), np.uint16)
    keep = np.ones(len(dia), bool)
    for i in matched:
        keep[i] = False
    if not keep.any():
        return np.zeros((H, W), np.uint16)
    return footprints_to_label_mask(dia[keep].copy(deep=True), dims,
                                    dtype=np.uint16)


def _one_panel(args):
    (idx, visit, detector, role, rows, repo, coll, stage3, skymap,
     Hmax, Wmax, lock, h5p, csvp, panelp, thr) = args
    try:
        butler = Butler(repo, collections=coll)
        ref = butler.registry.findDataset(
            "preliminary_visit_image",
            dataId={"instrument": "LSSTCam", "visit": int(visit),
                    "detector": int(detector)}, collections=coll)
        pvi, sources, template, pf, _ = fetch_diffim_inputs(
            butler, format_dataId(ref.dataId), skymap=skymap,
            stage3_collection=stage3)
        sub = run_subtract(template=template, science=pvi, sources=sources)
        diffim = sub.difference
        det = run_detect_diffim(science=pvi,
                                matchedTemplate=sub.matchedTemplate,
                                difference=diffim, threshold=thr)
        dia = det.diaSources

        arr = diffim.image.array
        H, W = int(arr.shape[0]), int(arr.shape[1])
        dims = _Dim(W, H)
        wcs = pvi.wcs
        recon = []
        for rr in rows:
            t = reconstruct_trail(wcs, rr, bbox_w=W, bbox_h=H)
            if t["on_ccd"]:
                recon.append((rr, t))
        n_cat, n_obj = len(rows), len(recon)

        mask = np.zeros((H, W), np.uint16)
        for i, (rr, t) in enumerate(recon):
            try:
                pw = pvi.psf.getLocalKernel(
                    Point2D(t["x"], t["y"])).getWidth()
            except Exception:
                pw = 7
            draw_one_line(mask, [t["x"], t["y"]], t["beta"],
                          t["trail_length"], true_value=i + 1,
                          line_thickness=max(1, int(pw / 2)))
        d_flag, d_mag, d_magerr, d_snr, matched = _match_stack_to_truth(
            dia, pvi.photoCalib, mask, max(n_obj, 1))
        real_labels = _labels_excluding(dia, matched, dims)

        rec = []
        for i, (rr, t) in enumerate(recon):
            rec.append({
                "ra": float(rr.get("RA_deg", np.nan)),
                "dec": float(rr.get("Dec_deg", np.nan)),
                "source_type": "Trail", "trail_length": float(t["trail_length"]),
                "mag": np.nan, "beta": float(t["beta"]),
                "visit": int(visit), "detector": int(detector),
                "integrated_mag": np.nan, "PSF_mag": np.nan,
                "SNR": float(d_snr[i]) if d_flag[i] else np.nan,
                "physical_filter": pf, "x": float(t["x"]), "y": float(t["y"]),
                "SNR_estimation": np.nan, "m5_local": np.nan,
                "m5_detector": np.nan, "stack_detection": bool(d_flag[i]),
                "stack_mag": float(d_mag[i]), "stack_mag_err": float(d_magerr[i]),
                "stack_snr": float(d_snr[i]), "ObjID": rr.get("ObjID", ""),
                "speed_deg_day": float(rr.get("speed_deg_day", np.nan)),
                "x_csv": float(rr["x"]), "y_csv": float(rr["y"]),
                "beta_csv": float(rr["angle"]),
                "trail_length_csv": float(rr["trail_length"]),
                "wcs_len_px": float(t["wcs_len_px"]), "image_id": idx})

        img_pad = np.zeros((Hmax, Wmax), np.float32)
        msk_pad = np.zeros((Hmax, Wmax), bool)
        rl_pad = np.zeros((Hmax, Wmax), np.uint16)
        img_pad[:H, :W] = arr.astype(np.float32)
        msk_pad[:H, :W] = mask.astype(bool)
        rl_pad[:H, :W] = real_labels
        with lock:
            with h5py.File(h5p, "a") as f:
                f["images"][idx] = img_pad
                f["masks"][idx] = msk_pad
                f["real_labels"][idx] = rl_pad
            if rec:
                df = pd.DataFrame(rec)[CSV_COLS]
                ex = os.path.exists(csvp)
                df.to_csv(csvp, mode="a" if ex else "w",
                          header=not ex, index=False)
            pex = os.path.exists(panelp)
            pd.DataFrame([{
                "image_id": idx, "visit": int(visit),
                "detector": int(detector), "role": role, "n_ast": n_obj,
                "n_cat": n_cat, "n_offccd": n_cat - n_obj,
                "n_stack_det_obj": int(d_flag[:n_obj].sum()) if n_obj else 0,
                "n_dia": len(dia), "band": pf, "img_h": H, "img_w": W,
            }]).to_csv(panelp, mode="a" if pex else "w",
                       header=not pex, index=False)
        return ("ok", idx, role, n_obj,
                int(d_flag[:n_obj].sum()) if n_obj else 0)
    except _SKIP as e:
        return ("err", idx, (int(visit), int(detector)),
                f"{e!r}\n{traceback.format_exc()}")


def build(manifest, out, real_csv, *, repo=DEF_REPO, stage3=DEF_STAGE3,
          stage2=DEF_STAGE2, skymap=DEF_SKYMAP, threshold=5.0, parallel=40,
          chunks=128, limit=0):
    """Build ``out/{test.h5,test.csv,panels.csv}`` from ``manifest``."""
    coll = [stage3, stage2]
    ensure_dir(out)
    man = pd.read_csv(manifest)
    if limit > 0:
        man = man.head(limit).copy()
    man = man.reset_index(drop=True)
    N = len(man)
    print(f"[build] {N} panels "
          f"({(man.role=='asteroid').sum()} asteroid + "
          f"{(man.role=='empty').sum()} empty), thr={threshold}", flush=True)

    real = pd.read_csv(real_csv)
    real["visit"] = real["FieldID"].astype("int64")
    real["detector"] = real["detector"].astype("int64")
    by_pair = {k: g.to_dict("records")
               for k, g in real.groupby(["visit", "detector"])}

    # PVI dims are not uniform — preallocate to the max, zero-pad each panel.
    _tl = threading.local()

    def _dim(v, d):
        b = getattr(_tl, "b", None)
        if b is None:
            b = Butler(repo, collections=coll); _tl.b = b
        try:
            dm = b.get("preliminary_visit_image.dimensions",
                       dataId={"instrument": "LSSTCam", "visit": int(v),
                               "detector": int(d)})
            return int(dm.y), int(dm.x)
        except Exception:
            return None

    pairs = list(zip(man["visit"].astype(int), man["detector"].astype(int)))
    Hmax = Wmax = got = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for r in ex.map(lambda vd: _dim(*vd), pairs):
            if r is not None:
                Hmax = max(Hmax, r[0]); Wmax = max(Wmax, r[1]); got += 1
    if Hmax == 0:
        raise RuntimeError("dimension pre-scan failed for all panels")
    print(f"[build] dim pre-scan {got}/{len(pairs)} -> "
          f"alloc ({Hmax},{Wmax})", flush=True)

    ch = (1, min(chunks, Hmax), min(chunks, Wmax))
    h5p = os.path.join(out, "test.h5")
    csvp = os.path.join(out, "test.csv")
    panelp = os.path.join(out, "panels.csv")
    for p in (h5p, csvp, panelp):
        if os.path.exists(p):
            os.remove(p)
    with h5py.File(h5p, "w") as f:
        for nm, dt in (("images", "float32"), ("masks", "bool"),
                       ("real_labels", "uint16")):
            f.create_dataset(nm, shape=(N, Hmax, Wmax), dtype=dt, chunks=ch,
                             compression="gzip", compression_opts=4,
                             shuffle=True)
    print(f"[build] allocated {h5p} ({N},{Hmax},{Wmax})", flush=True)

    mgr = Manager()
    lock = mgr.Lock()
    tasks = []
    for idx, row in man.iterrows():
        v, d = int(row["visit"]), int(row["detector"])
        rws = by_pair.get((v, d), []) if row["role"] == "asteroid" else []
        tasks.append((idx, v, d, row["role"], rws, repo, coll, stage3,
                      skymap, Hmax, Wmax, lock, h5p, csvp, panelp,
                      float(threshold)))
    ok = err = obj = sd = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
        futs = [ex.submit(_one_panel, t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                o = fut.result()
            except BaseException as e:
                err += 1
                print(f"[{i}/{N}] CRASH {type(e).__name__}: {e}", flush=True)
                continue
            if o[0] == "ok":
                ok += 1; obj += o[3]; sd += o[4]
                if i % 50 == 0 or i == N:
                    print(f"[{i}/{N}] ok={ok} err={err} | objs={obj} "
                          f"stack_det={sd}", flush=True)
            else:
                err += 1
                print(f"[{i}/{N}] ERR idx={o[1]} pair={o[2]}\n{o[3]}",
                      flush=True)
    print(f"\n[done] ok={ok} err={err} | objs={obj} stack_det={sd} "
          f"({100*sd/max(obj,1):.1f}%)\n[done] {h5p}")


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        "build_real", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "build"):
        s = sub.add_parser(name)
        s.add_argument("--repo", default=DEF_REPO)
        s.add_argument("--stage3", default=DEF_STAGE3)
        s.add_argument("--stage2", default=DEF_STAGE2)
        s.add_argument("--skymap", default=DEF_SKYMAP)
        s.add_argument("--real-csv", required=True,
                       help="catalog of real fast movers "
                            "(FieldID,detector,RA_deg,Dec_deg,rates,"
                            "exposure_time,x,y,angle,trail_length,ObjID,...)")
    sc = sub.choices["scan"]
    sc.add_argument("--out-dir", required=True)
    sc.add_argument("--n-empty", type=int, default=150)
    sc.add_argument("--workers", type=int, default=32)
    sc.add_argument("--seed", type=int, default=123)
    bd = sub.choices["build"]
    bd.add_argument("--manifest", required=True)
    bd.add_argument("--out", required=True)
    bd.add_argument("--threshold", type=float, default=5.0)
    bd.add_argument("--parallel", type=int, default=40)
    bd.add_argument("--chunks", type=int, default=128)
    bd.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan_availability(a.real_csv, a.out_dir, repo=a.repo, stage3=a.stage3,
                          stage2=a.stage2, skymap=a.skymap,
                          n_empty=a.n_empty, workers=a.workers, seed=a.seed)
    else:
        build(a.manifest, a.out, a.real_csv, repo=a.repo, stage3=a.stage3,
              stage2=a.stage2, skymap=a.skymap, threshold=a.threshold,
              parallel=a.parallel, chunks=a.chunks, limit=a.limit)


if __name__ == "__main__":
    main()
