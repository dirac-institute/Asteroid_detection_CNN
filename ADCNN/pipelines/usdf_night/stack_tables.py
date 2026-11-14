"""Rubin DIA source tables of every manifest panel -> work/stack_sources.parquet.

One row per dia_source_unfiltered source that is not a sky source. The DPDD columns are computed
from the afw columns with the functor definitions of ap_association data/DiaSource.yaml and
pipe_tasks functors.py (w_2026_46), including the float32 rounding of the APDB schema, so a source
gets the same values whether or not Rubin's reliability filter kept it. On the rows that are in
dia_source_detector every computed column is compared with Rubin's value; the comparison goes to
prep_info.json ("stack_tables" -> "validation").

in_detector = in dia_source_detector (passed Rubin's filter, reliability >= 0.5); in_apdb = in
dia_source_apdb (associated). reliability comes from dia_source_detector (the unfiltered table
carries no reliability values), dia_object_id / ss_object_id from dia_source_apdb; NaN resp. 0
where the source is not in that table. glint_trail is set on every unfiltered row.

A table read that fails is retried once; a panel still unreadable fails the step (exit 1): it is
listed in prep_info.json ("panels_unreadable") and no parquet is written, so the sources of a night
are never silently incomplete and a rerun of prep retries the step.

  python -m ADCNN.pipelines.usdf_night.stack_tables <day> --manifest M --out O [--workers 16]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.pipelines.usdf_night.night import (CredentialError, day_where, normalize_day, open_butler,
                                              update_info, write_atomic)

TABLES = ("dia_source_unfiltered", "dia_source_detector", "dia_source_apdb")
NAIVE = "ext_trailedSources_Naive_"
CD = ("base_LocalWcs_CDMatrix_1_1", "base_LocalWcs_CDMatrix_1_2",
      "base_LocalWcs_CDMatrix_2_1", "base_LocalWcs_CDMatrix_2_2")
# association-flag-map.yaml (DiaSource) bits base_PixelFlags_flag_<s>; flag-rename-rules.yaml
# turns them into the DPDD columns pixelFlags_<s>.
PIXEL_FLAGS = ("offimage", "edge", "interpolated", "saturated", "cr", "bad", "suspect",
               "interpolatedCenter", "saturatedCenter", "crCenter", "suspectCenter", "streak",
               "streakCenter", "injected", "injectedCenter", "injected_template",
               "injected_templateCenter", "nodata", "nodataCenter", "spike")
FLAG_COLUMNS = (["is_negative", "rubin_is_dipole"] + [f"pixelFlags_{s}" for s in PIXEL_FLAGS]
                + ["trail_flag_edge"])
COLUMNS = (["diaSourceId", "visit", "detector", "band", "mjd_tai", "x", "y", "psf_ra", "psf_dec",
            "ra_err", "dec_err", "ra_dec_cov", "trail_ra", "trail_dec", "trail_length_px",
            "trail_length_err_px", "trail_angle_img", "trail_x0", "trail_y0", "trail_x1", "trail_y1",
            "trail_length_arcsec", "trail_length_err_arcsec", "trail_pa_deg", "trail_pa_err_deg",
            "psf_flux_njy", "psf_flux_err_njy", "trail_flux_njy", "trail_flux_err_njy"]
           + FLAG_COLUMNS + ["in_detector", "in_apdb", "reliability", "glint_trail",
                             "dia_object_id", "ss_object_id"])
# our column -> (dia_source_detector column, comparison kind)
COMPARE = {"x": ("x", "f32"), "y": ("y", "f32"), "psf_ra": ("ra", "f64"), "psf_dec": ("dec", "f64"),
           "ra_err": ("raErr", "f32"), "dec_err": ("decErr", "f32"), "ra_dec_cov": ("ra_dec_Cov", "f32"),
           "trail_ra": ("trailRa", "f64"), "trail_dec": ("trailDec", "f64"),
           "trail_length_arcsec": ("trailLength", "f32"),
           "trail_length_err_arcsec": ("trailLengthErr", "f32"),
           "trail_pa_deg": ("trailAngle", "angle"), "trail_pa_err_deg": ("trailAngleErr", "f32"),
           "psf_flux_njy": ("psfFlux", "f32"), "psf_flux_err_njy": ("psfFluxErr", "f32"),
           "trail_flux_njy": ("trailFlux", "f32"), "trail_flux_err_njy": ("trailFluxErr", "f32"),
           "is_negative": ("isNegative", "flag"), "rubin_is_dipole": ("isDipole", "flag"),
           "trail_flag_edge": ("trail_flag_edge", "flag"), "glint_trail": ("glint_trail", "flag"),
           **{f"pixelFlags_{s}": (f"pixelFlags_{s}", "flag") for s in PIXEL_FLAGS},
           "mjd_tai": ("midpointMjdTai", "mjd"), "band": ("band", "str")}
RTOL = 1e-6
TWO_PI = 2.0 * np.pi
HALF_PI = 90.0 * (np.pi / 180.0)


def pixel_scale_arcsec(cd11, cd12, cd21, cd22):
    """ComputePixelScale.pixelScaleArcseconds (CD matrix in radians/pixel)."""
    return 3600 * np.degrees(np.sqrt(np.fabs(cd11 * cd22 - cd12 * cd21)))


def _wrap(a):
    """lsst::geom Angle::wrap, [0, 2pi)."""
    w = np.fmod(a, TWO_PI)
    w = np.where(w < 0.0, w + TWO_PI, w)
    return np.where(w >= TWO_PI, 0.0, w)


def _wrap_ctr(a):
    """lsst::geom Angle::wrapCtr, [-pi, pi)."""
    w = np.fmod(a, TWO_PI)
    lo = w + TWO_PI
    hi = w - TWO_PI
    w = np.where(w < -np.pi, np.where(lo >= np.pi, -np.pi, lo),
                 np.where(w >= np.pi, np.where(hi < -np.pi, -np.pi, hi), w))
    return w


def position_angle_deg(theta, cd11, cd12, cd21, cd22):
    """ConvertDetectorAngleToPositionAngle: detector angle (rad, +y of +x) -> degrees E of N.

    Rubin offsets SpherePoint(0, 0) by the CD-projected unit vector and takes
    (pi/2 - bearingTo).wrapCtr(); this is that computation, vectorised."""
    dx, dy = np.cos(theta), np.sin(theta)
    lon = _wrap(dx * cd11 + dy * cd12)
    lat = dx * cd21 + dy * cd22
    y = np.sin(lon) * np.cos(lat)
    x = np.sin(lat)
    bearing = _wrap(HALF_PI - np.arctan2(y, x))
    return np.rad2deg(_wrap_ctr(HALF_PI - bearing))


def _f32(a):
    return np.asarray(a).astype(np.float32).astype(np.float64)


def dpdd(t) -> dict:
    """DPDD-named values for the rows of `t` (name -> array mapping with afw/slot column names)."""
    g = lambda name: np.asarray(t[name])
    cd = [g(c).astype(np.float64) for c in CD]
    scale = pixel_scale_arcsec(*cd)
    calib = g("base_LocalPhotoCalib")
    out = {
        "x": g("slot_Centroid_x").astype(np.float64), "y": g("slot_Centroid_y").astype(np.float64),
        "psf_ra": g("coord_ra") * 180 / np.pi, "psf_dec": g("coord_dec") * 180 / np.pi,
        "ra_err": _f32(g("coord_raErr") * 180 / np.pi), "dec_err": _f32(g("coord_decErr") * 180 / np.pi),
        "ra_dec_cov": _f32(g("coord_ra_dec_Cov") * (180 / np.pi) ** 2),
        "trail_ra": g(NAIVE + "ra").astype(np.float64), "trail_dec": g(NAIVE + "dec").astype(np.float64),
        "trail_length_px": g(NAIVE + "length").astype(np.float64),
        "trail_length_err_px": g(NAIVE + "lengthErr").astype(np.float64),
        "trail_angle_img": g(NAIVE + "angle").astype(np.float64),
        "trail_x0": g(NAIVE + "x0").astype(np.float64), "trail_y0": g(NAIVE + "y0").astype(np.float64),
        "trail_x1": g(NAIVE + "x1").astype(np.float64), "trail_y1": g(NAIVE + "y1").astype(np.float64),
        "trail_length_arcsec": _f32(g(NAIVE + "length") * scale),
        "trail_length_err_arcsec": _f32(g(NAIVE + "lengthErr") * scale),
        "trail_pa_deg": _f32(position_angle_deg(g(NAIVE + "angle"), *cd)),
        "trail_pa_err_deg": _f32(np.rad2deg(g(NAIVE + "angleErr"))),
        "psf_flux_njy": _f32(g("slot_PsfFlux_instFlux") * calib),
        "psf_flux_err_njy": _f32(g("slot_PsfFlux_instFluxErr") * calib),
        "trail_flux_njy": _f32(g(NAIVE + "flux") * calib),
        "trail_flux_err_njy": _f32(g(NAIVE + "fluxErr") * calib),
        "is_negative": g("is_negative").astype(np.int8),
        "rubin_is_dipole": g("ip_diffim_DipoleFit_classification").astype(np.int8),
        "trail_flag_edge": g(NAIVE + "flag_edge").astype(np.int8),
    }
    for s in PIXEL_FLAGS:
        out[f"pixelFlags_{s}"] = g(f"base_PixelFlags_flag_{s}").astype(np.int8)
    return out


def _col(table, name, fill=None):
    a = table[name]
    return np.asarray(np.ma.filled(a, fill) if fill is not None else a)


def compare(ours: dict, det, pos) -> dict:
    """Per COMPARE column: [n, n_nan_mismatch, n_neq, n_fail, max_abs_diff] on the rows with pos >= 0.

    Columns an older detector-table schema lacks (e.g. pixelFlags_spike before 2026-11) are skipped."""
    sel = pos >= 0
    idx = pos[sel]
    stats = {}
    for col, (theirs, kind) in COMPARE.items():
        if theirs not in det.columns:
            continue
        a = np.asarray(ours[col])[sel]
        b = _col(det, theirs)[idx]
        if kind in ("flag", "str"):
            neq = int(np.sum(a.astype(str if kind == "str" else bool) != b.astype(str if kind == "str" else bool)))
            stats[col] = [int(sel.sum()), 0, neq, neq, float(neq > 0)]
            continue
        b = b.astype(np.float64)
        a = a.astype(np.float32).astype(np.float64) if kind in ("f32", "angle") else a.astype(np.float64)
        na, nb = np.isnan(a), np.isnan(b)
        both = ~na & ~nb
        d = a[both] - b[both]
        if kind == "angle":
            d = (d + 180.0) % 360.0 - 180.0
        d = np.abs(d)
        if kind == "mjd":
            fail = d >= 1.0 / 86400e3
        else:
            fail = d > RTOL * np.abs(b[both])
        nan_mis = int(np.sum(na != nb))
        stats[col] = [int(sel.sum()), nan_mis, nan_mis + int(np.sum(d != 0)), nan_mis + int(fail.sum()),
                      float(d.max()) if d.size else 0.0]
    return stats


def merge_stats(total: dict, part: dict) -> None:
    for col, (n, nanm, neq, fail, mx) in part.items():
        t = total.setdefault(col, [0, 0, 0, 0, 0.0])
        t[0] += n; t[1] += nanm; t[2] += neq; t[3] += fail; t[4] = max(t[4], mx)


def panel_rows(unf, det, apdb, visit, detector, band, mjd_tai):
    """(rows DataFrame, comparison stats, counts) for one panel; det/apdb may be None."""
    sky = np.asarray(unf["sky_source"], bool)
    keep = ~sky
    t = {name: np.asarray(unf[name])[keep] for name in unf.colnames}
    ids = t["id"].astype(np.int64)
    n = len(ids)
    ours = dpdd(t)
    det_ids = _col(det, "diaSourceId").astype(np.int64) if det is not None else np.zeros(0, np.int64)
    apdb_ids = _col(apdb, "diaSourceId").astype(np.int64) if apdb is not None else np.zeros(0, np.int64)
    pos = pd.Index(det_ids).get_indexer(ids) if det_ids.size else np.full(n, -1)
    apos = pd.Index(apdb_ids).get_indexer(ids) if apdb_ids.size else np.full(n, -1)
    in_det, in_apdb = pos >= 0, apos >= 0
    out = {"diaSourceId": ids, "visit": np.full(n, visit, np.int64), "detector": np.full(n, detector, np.int64),
           "band": np.full(n, band, object), "mjd_tai": np.full(n, mjd_tai, np.float64), **ours,
           "in_detector": in_det.astype(np.int8), "in_apdb": in_apdb.astype(np.int8),
           "reliability": np.full(n, np.nan), "glint_trail": (t["glint_trail"].astype(np.float64) if "glint_trail" in t
                                                              else np.full(n, np.nan)),
           "dia_object_id": np.zeros(n, np.int64), "ss_object_id": np.zeros(n, np.int64)}
    if in_det.any():
        out["reliability"][in_det] = _col(det, "reliability").astype(np.float64)[pos[in_det]]
    if in_apdb.any():
        out["dia_object_id"][in_apdb] = _col(apdb, "diaObjectId", 0).astype(np.int64)[apos[in_apdb]]
        out["ss_object_id"][in_apdb] = _col(apdb, "ssObjectId", 0).astype(np.int64)[apos[in_apdb]]
    stats = compare(out, det, pos) if det is not None else {}
    counts = {"rows_unfiltered": int(sky.size), "rows_sky": int(sky.sum()), "rows": n,
              "in_detector": int(in_det.sum()), "in_apdb": int(in_apdb.sum()),
              "detector_rows": int(det_ids.size), "apdb_rows": int(apdb_ids.size),
              "detector_not_in_rows": int((~np.isin(det_ids, ids)).sum()),
              "apdb_not_in_detector": int((~np.isin(apdb_ids, det_ids)).sum()),
              "rows_with_parent": int(np.sum(t["parent"] != 0))}
    return pd.DataFrame(out, columns=COLUMNS), stats, counts


_BUTLER = None


def _init(repo):
    global _BUTLER
    _BUTLER = open_butler(repo)


def _panel(task):
    visit, detector, band, mjd_tai, refs = task
    tables = []
    for name, ref in zip(TABLES, refs):
        if ref is None:
            tables.append(None)
            continue
        try:
            obj = _BUTLER.get(ref)
        except Exception as e:
            msg = (str(e).strip().splitlines() or [""])[0][:200]
            return (visit, detector), None, None, None, f"{name}: {type(e).__name__}: {msg}"
        tables.append(obj.asAstropy() if name == "dia_source_unfiltered" else obj)
    df, stats, counts = panel_rows(*tables, visit, detector, band, mjd_tai)
    return (visit, detector), df, stats, counts, None


def table_refs(butler, day, runs) -> dict:
    """table -> {(run, visit, detector): ref}, each table looked up in the panel's own run."""
    from lsst.daf.butler import MissingDatasetTypeError
    out = {}
    for table in TABLES:
        out[table] = {}
        for run in runs:
            try:
                refs = butler.query_datasets(table, collections=[run], where=day_where(day),
                                             find_first=True, limit=None, explain=False)
            except MissingDatasetTypeError:
                refs = []
            for r in refs:
                out[table][(run, int(r.dataId["visit"]), int(r.dataId["detector"]))] = r
    return out


def build(day, manifest, out, workers=16) -> dict:
    t0 = time.time()
    m = pd.read_csv(manifest)
    repos = sorted(m.repo.unique())
    if len(repos) > 1:
        raise RuntimeError(f"manifest mixes repos {repos}")
    info = {"night": day, "n_panels": len(m), "workers": workers, "panels_without_unfiltered": [],
            "panels_without_detector": [], "panels_without_apdb": [], "panels_unreadable": []}
    frames, stats, counts = [], {}, {}

    def add(df, st, ct):
        frames.append(df)
        merge_stats(stats, st)
        for k, v in ct.items():
            counts[k] = counts.get(k, 0) + v
    if len(m):
        b = open_butler(repos[0])
        refs = table_refs(b, day, sorted(m.run.unique()))
        info["refs"] = {t: len(r) for t, r in refs.items()}
        tasks = []
        for row in m.itertuples(index=False):
            key = (row.run, int(row.visit), int(row.detector))
            rs = tuple(refs[t].get(key) for t in TABLES)
            for r, lst in zip(rs, ("panels_without_unfiltered", "panels_without_detector",
                                   "panels_without_apdb")):
                if r is None:
                    info[lst].append([int(row.visit), int(row.detector)])
            if rs[0] is not None:
                tasks.append((int(row.visit), int(row.detector), str(row.band), float(row.mjd_tai), rs))
        print(f"[stack_tables] {len(tasks)} panels with unfiltered tables of {len(m)} "
              f"(query {time.time() - t0:.1f} s)", flush=True)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init,
                                 initargs=(repos[0],)) as ex:
            retry = []
            for i, (task, (key, df, st, ct, err)) in enumerate(zip(tasks, ex.map(_panel, tasks, chunksize=8)), 1):
                if err:
                    retry.append(task)
                    print(f"[stack_tables] WARNING panel {key}: {err}; retried at the end", flush=True)
                else:
                    add(df, st, ct)
                if i % 500 == 0 or i == len(tasks):
                    print(f"[stack_tables]   {i}/{len(tasks)} panels, {time.time() - t0:.0f} s", flush=True)
            for key, df, st, ct, err in ex.map(_panel, retry):          # one retry of the failed reads
                if err:
                    info["panels_unreadable"].append([key[0], key[1], err])
                    print(f"[stack_tables] ERROR unreadable panel {key}: {err}", flush=True)
                else:
                    add(df, st, ct)
    df = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS))
    df = df.sort_values(["visit", "detector", "diaSourceId"], kind="stable").reset_index(drop=True)
    if df.diaSourceId.duplicated().any():
        raise RuntimeError("duplicate diaSourceId in stack_sources")
    if not info["panels_unreadable"]:
        write_atomic(df, out)
    keys = ["n", "n_nan_mismatch", "n_neq", "n_fail", "max_abs_diff"]
    info["counts"] = counts
    info["validation"] = {c: dict(zip(keys, v)) for c, v in stats.items()}
    if "mjd_tai" in stats:
        info["validation"]["mjd_tai"]["max_abs_diff_ms"] = stats["mjd_tai"][4] * 86400e3
    bad = sorted(c for c, v in stats.items() if v[3])
    info["validation_failed_columns"] = bad
    info["validation_columns_not_in_detector"] = [c for c in COMPARE if c not in stats] if stats else []
    info["seconds"] = round(time.time() - t0, 1)
    update_info(Path(out).parent / "prep_info.json", "stack_tables", info)
    if info["panels_unreadable"]:
        raise RuntimeError(f"{len(info['panels_unreadable'])} panels unreadable after a retry (listed above and in "
                           f"prep_info.json); {out} not written, rerunning prep retries them")
    if bad:
        print(f"[stack_tables] WARNING columns differing from dia_source_detector: {bad}", flush=True)
    print(f"[stack_tables] {len(df):,} sources ({counts.get('in_detector', 0):,} in detector, "
          f"{counts.get('in_apdb', 0):,} in apdb) -> {out} ({info['seconds']} s)", flush=True)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stack_tables", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    try:
        build(normalize_day(a.day), a.manifest, a.out, a.workers)
    except CredentialError as e:
        print(f"stack_tables: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
