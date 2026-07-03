"""Build cross-field NULL visit pairs for the 2-visit chance-link (fpp) calibration.

METHOD (reconstructed 2026-07-03 from the saved run_embargo_0630/null2v/pair_null_*.csv,
frame-transport rotation recovered to <0.01" on all 4 original donors):
take the real anchor visit's detections as-is, and for each DONOR visit (a far-away
same-night field) rigidly rotate ALL its detections onto the TARGET visit's footprint:

    M = Rz(ra_t) @ Ry(-dec_t) @ Ry(dec_d) @ Rz(-ra_d)

with (ra_d, dec_d) / (ra_t, dec_t) the per-visit MEDIAN det centers. Trail endpoints
(ra0,dec0 / ra1,dec1) rotate with the same M; mjd is shifted by a constant to the target
visit's median epoch; the visit id is relabeled to the target's. The pair file =
anchor dets (unchanged) + translated donor dets. Real links are physically impossible
(the two visits image DISJOINT sky), while densities, trail statistics, and the pair
cadence (anchor->target dt) are preserved -- so every surviving 2-visit track is a
CHANCE link, and k_per_det2 = sum(chance) / sum(n1*n2) calibrates fpp_2v_chance.json.

STATIC CATALOGS (per pair, for the production veto stack): the linker flags dets against
one --static-catalog, so each pair gets its own union catalog =
  real statics near the anchor+target region (flags the anchor dets faithfully)
  + donor-region statics rotated by the SAME M (flags the translated donor dets against
    the statics their subtraction residuals actually correlate with).
Cross-terms (donor dets chance-flagged by real target-region statics and vice versa in
the anchor/target overlap) are ~0.6-6% per det (pi*(3")^2 * mag<20 density) -- small
vs the Poisson error of the k fit; documented in fpp_2v_chance.json. Donor fields with
NO static coverage (no DRP `object` tracts -- e.g. 663/712/715 on 20260630) are reported
LOUDLY: their translated dets can never be static-flagged, matching production behavior
on uncovered tracts, and they should be fit/reported SEPARATELY from covered donors.

Usage:
    python -m ADCNN.linking.make_null_pairs \
        --dets   run_embargo_0630/adcnn_dets_masked.csv \
        --anchor-visit 2026063000345 --target-visit 2026063000396 \
        --donors 103,105,106,107,126,128,130,271,272,129,663,712,715 \
        --static-catalog run_embargo_0630/expt_staticveto/static_catalog.parquet \
        --outdir run_embargo_0630/null2v_recal \
        --verify-against run_embargo_0630/null2v
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def transport_matrix(ra_d, dec_d, ra_t, dec_t):
    """Rotation taking the donor center (ra_d, dec_d) to the target center (ra_t, dec_t):
    donor frame -> origin -> target frame. This is the EXACT convention of the original
    2026-06 null pairs (verified <0.01" against all 4 saved pair files)."""
    r = np.radians
    return _rz(r(ra_t)) @ _ry(-r(dec_t)) @ _ry(r(dec_d)) @ _rz(-r(ra_d))


def rotate_radec(ra, dec, M):
    """Apply M to (ra, dec) degrees; NaN-safe (NaN rows stay NaN)."""
    ra = np.asarray(ra, float)
    dec = np.asarray(dec, float)
    rar, decr = np.radians(ra), np.radians(dec)
    u = np.stack([np.cos(decr) * np.cos(rar), np.cos(decr) * np.sin(rar), np.sin(decr)], axis=-1)
    v = u @ M.T
    ra2 = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    dec2 = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    return ra2, dec2


def make_pair(dets, anchor_visit, target_visit, donor_visit):
    """(pair_df, M, meta) -- anchor dets unchanged + donor dets rotated onto the target
    footprint, relabeled to the target visit at the target epoch."""
    anchor = dets[dets.visit == anchor_visit]
    target = dets[dets.visit == target_visit]
    donor = dets[dets.visit == donor_visit]
    for name, df in [("anchor", anchor), ("target", target), ("donor", donor)]:
        if df.empty:
            raise SystemExit(f"[null-pairs] {name} visit has no detections in --dets")
    ra_d, dec_d = donor.ra.median(), donor.dec.median()
    ra_t, dec_t = target.ra.median(), target.dec.median()
    M = transport_matrix(ra_d, dec_d, ra_t, dec_t)
    moved = donor.copy()
    for cra, cdec in [("ra", "dec"), ("ra0", "dec0"), ("ra1", "dec1")]:
        moved[cra], moved[cdec] = rotate_radec(donor[cra], donor[cdec], M)
    dmjd = target.mjd.median() - donor.mjd.median()
    moved["mjd"] = donor.mjd + dmjd
    moved["visit"] = target_visit
    meta = dict(donor_center=(float(ra_d), float(dec_d)), target_center=(float(ra_t), float(dec_t)),
                dmjd_day=float(dmjd), n_anchor=len(anchor), n_donor=len(donor))
    return pd.concat([anchor, moved], ignore_index=True), M, meta


def pair_statics(statics, M, donor_center, anchor_center, target_center, radius_deg):
    """Union static catalog for one pair: real statics near anchor/target + donor-region
    statics rotated by M. Returns (catalog, n_donor_statics)."""
    def near(ra_c, dec_c):
        dra = (statics.ra - ra_c + 180.0) % 360.0 - 180.0
        return (dra * np.cos(np.radians(dec_c))) ** 2 + (statics.dec - dec_c) ** 2 < radius_deg ** 2
    real = statics[near(*anchor_center) | near(*target_center)]
    dstat = statics[near(*donor_center)].copy()
    if len(dstat):
        dstat["ra"], dstat["dec"] = rotate_radec(dstat.ra, dstat.dec, M)
    return pd.concat([real, dstat], ignore_index=True), len(dstat)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="full-night link-input dets CSV (adcnn_dets_masked.csv)")
    ap.add_argument("--anchor-visit", type=int, required=True, help="real visit kept unchanged (e.g. 2026063000345)")
    ap.add_argument("--target-visit", type=int, required=True,
                    help="visit whose footprint/epoch the donors are moved onto (e.g. 2026063000396); "
                    "fixes the pair dt = anchor->target")
    ap.add_argument("--donors", required=True,
                    help="comma list of donor visits; short suffixes allowed (129 -> same prefix as --target-visit)")
    ap.add_argument("--static-catalog", default=None,
                    help="full-depth static catalog (build_static_catalog output); per-pair union catalogs "
                    "are written next to each pair file")
    ap.add_argument("--static-radius-deg", type=float, default=2.5,
                    help="selection radius around each field center (FOV ~1.75 deg + margin)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--verify-against", default=None,
                    help="directory holding original pair_null_<visit>.csv files; rebuilt pairs are "
                    "checked against them (positions to --verify-tol-arcsec, mjd to 1e-6 d)")
    ap.add_argument("--verify-tol-arcsec", type=float, default=0.01)
    a = ap.parse_args()

    dets = pd.read_csv(a.dets)
    prefix = a.target_visit // 1000 * 1000
    donors = [int(x) if int(x) > prefix else prefix + int(x) for x in a.donors.split(",")]
    statics = pd.read_parquet(a.static_catalog) if a.static_catalog else None
    anchor_c = (dets[dets.visit == a.anchor_visit].ra.median(), dets[dets.visit == a.anchor_visit].dec.median())
    target_c = (dets[dets.visit == a.target_visit].ra.median(), dets[dets.visit == a.target_visit].dec.median())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for dv in donors:
        pair, M, meta = make_pair(dets, a.anchor_visit, a.target_visit, dv)
        ppath = out / f"pair_null_{dv}.csv"
        pair.to_csv(ppath, index=False)
        line = (f"[null-pairs] donor {dv}: {meta['n_donor']} dets rotated "
                f"({meta['donor_center'][0]:.2f},{meta['donor_center'][1]:.2f}) -> "
                f"({meta['target_center'][0]:.2f},{meta['target_center'][1]:.2f}), dmjd {meta['dmjd_day']:+.6f} d; "
                f"pair n={len(pair)} -> {ppath}")
        if statics is not None:
            cat, n_donor_statics = pair_statics(statics, M, meta["donor_center"], anchor_c, target_c,
                                                a.static_radius_deg)
            spath = out / f"static_pair_{dv}.parquet"
            cat.to_parquet(spath)
            line += f"; statics {len(cat)} ({n_donor_statics} donor-region) -> {spath.name}"
            if n_donor_statics == 0:
                line += "  ** DONOR FIELD UNCOVERED: its dets can NEVER be static-flagged -- fit separately **"
        print(line, flush=True)

        if a.verify_against:
            ref = pd.read_csv(Path(a.verify_against) / f"pair_null_{dv}.csv")
            if len(ref) != len(pair):
                raise SystemExit(f"[null-pairs] VERIFY donor {dv}: row count {len(pair)} != saved {len(ref)}")
            r = ref.sort_values(["visit", "detid"]).reset_index(drop=True)
            p = pair.sort_values(["visit", "detid"]).reset_index(drop=True)
            if not (r.detid.equals(p.detid) and r.visit.equals(p.visit)):
                raise SystemExit(f"[null-pairs] VERIFY donor {dv}: detid/visit mismatch vs saved")
            dpos = np.hypot((r.ra - p.ra) * np.cos(np.radians(r.dec)), r.dec - p.dec) * 3600.0
            dmjd = (r.mjd - p.mjd).abs().max()
            print(f"  verify vs saved: max dpos {dpos.max():.4f}\" , max dmjd {dmjd:.2e} d", flush=True)
            if dpos.max() > a.verify_tol_arcsec or dmjd > 1e-6:
                raise SystemExit(f"[null-pairs] VERIFY donor {dv}: rebuilt pair deviates from saved "
                                 f"(dpos {dpos.max():.4f}\" > {a.verify_tol_arcsec}\" or dmjd {dmjd:.2e} > 1e-6)")


if __name__ == "__main__":
    main()
