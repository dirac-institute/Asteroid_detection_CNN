"""Head-to-head metrics: map each pipeline's final linkages back to the injected NEO truth and report
completeness + purity, broken out by night-count (same-night vs multi-night) and injected detection-SNR
(the faint sub-5sigma axis where ADCNN should win). Off-ecliptic field => zero real asteroids => any
linkage NOT matching an injected object is a genuine FALSE link.

Two linkage sources:
  * heliolinx (HIS chain): a (pairdets, LPclust2det) pair. pairdets rows carry MJD,RA,Dec; LPclust2det maps
    linkage_id -> pairdets row index. We tag each linked detection to an injected sighting (same night,
    nearest within --tol-arcsec) and call the linkage TRUE if a strict majority of its detections share one
    injected objID.
  * trail_state_link (OURS): tracks.csv with per-track member detections already crossmatched; we re-tag its
    members against the SAME injected truth for an apples-to-apples objID assignment.

Completeness denominators (from truth_objects.csv, field-unique objIDs):
  - multi-night (his regime): injected objects with >=3 distinct nights AND >=6 sightings on panels.
  - same-night (ours): injected objects with >=2 sightings on a single night.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree


def tag_by_truth(ra, dec, mjd, inj, tol_arcsec=3.0):
    """Return injected objID for each (ra,dec,mjd) detection, or None. Match within the same night and
    tol_arcsec; inj has columns objID,ra,dec,mjd."""
    inj = inj.copy()
    inj["night"] = np.floor(inj.mjd - 0.5).astype(int)
    night = np.floor(np.asarray(mjd) - 0.5).astype(int)
    out = np.array([None] * len(ra), dtype=object)
    tol = tol_arcsec / 3600.0
    for n in np.unique(night):
        g = inj[inj.night == n]
        if g.empty:
            continue
        cd = np.cos(np.radians(g.dec.values.mean()))
        tree = cKDTree(np.c_[g.ra.values * cd, g.dec.values])
        m = night == n
        q = np.c_[np.asarray(ra)[m] * cd, np.asarray(dec)[m]]
        d, j = tree.query(q, k=1)
        hit = d < tol
        idx = np.where(m)[0]
        for kk, ish, jj in zip(idx[hit], hit[hit], j[hit]):
            out[kk] = g.objID.values[jj]
    return out


def linkages_from_heliolinx(pairdets_path, clust2det_path, inj, tol_arcsec, frac=0.6):
    pd_df = pd.read_csv(pairdets_path)
    pd_df.columns = [c.lstrip("#") for c in pd_df.columns]
    c2d = pd.read_csv(clust2det_path)
    c2d.columns = [c.lstrip("#") for c in c2d.columns]
    cid, dnum = c2d.columns[0], c2d.columns[1]
    objtag = tag_by_truth(pd_df.RA.values, pd_df.Dec.values, pd_df.MJD.values, inj, tol_arcsec)
    pd_df = pd_df.assign(_obj=objtag, _night=np.floor(pd_df.MJD - 0.5).astype(int))
    links = []
    for lid, grp in c2d.groupby(cid):
        rows = pd_df.iloc[grp[dnum].values]
        objs = rows._obj.dropna()
        true_obj, is_true = None, False
        if len(objs):
            vc = objs.value_counts()
            if vc.iloc[0] >= frac * len(rows):
                true_obj = vc.index[0]; is_true = True
        links.append(dict(linkage=lid, n_det=len(rows), n_nights=rows._night.nunique(),
                          obj=true_obj, is_true=is_true))
    return pd.DataFrame(links)


def linkages_from_tracks(tracks_path, inj, tol_arcsec, frac=0.6):
    import os
    if not os.path.exists(tracks_path) or os.path.getsize(tracks_path) == 0:
        return pd.DataFrame(columns=["linkage", "n_det", "n_nights", "obj", "is_true"])
    try:
        t = pd.read_csv(tracks_path)
    except Exception:
        return pd.DataFrame(columns=["linkage", "n_det", "n_nights", "obj", "is_true"])
    if t.empty:
        return pd.DataFrame(columns=["linkage", "n_det", "n_nights", "obj", "is_true"])
    # tracks.csv has per-track rows? expect columns track_id, ra, dec, mjd (member dets). Fallback: if it
    # stores one row per track with no members, we cannot re-tag -> use its own match_obj column.
    if {"ra", "dec", "mjd"}.issubset(t.columns) and ("track_id" in t.columns or "track" in t.columns):
        tid = "track_id" if "track_id" in t.columns else "track"
        objtag = tag_by_truth(t.ra.values, t.dec.values, t.mjd.values, inj, tol_arcsec)
        t = t.assign(_obj=objtag, _night=np.floor(t.mjd - 0.5).astype(int))
        links = []
        for lid, grp in t.groupby(tid):
            objs = grp._obj.dropna()
            true_obj, is_true = None, False
            if len(objs):
                vc = objs.value_counts()
                if vc.iloc[0] >= frac * len(grp):
                    true_obj = vc.index[0]; is_true = True
            links.append(dict(linkage=lid, n_det=len(grp), n_nights=grp._night.nunique(),
                              obj=true_obj, is_true=is_true))
        return pd.DataFrame(links)
    # fallback: one row per track with status/match_obj
    oc = "match_obj" if "match_obj" in t.columns else ("obj" if "obj" in t.columns else None)
    links = []
    for i, r in t.iterrows():
        o = r[oc] if oc else None
        links.append(dict(linkage=i, n_det=int(r.get("n", 2)), n_nights=int(r.get("n_nights", 1)),
                          obj=(o if isinstance(o, str) and o not in ("", "nan") else None),
                          is_true=isinstance(o, str) and o not in ("", "nan", "NEW")))
    return pd.DataFrame(links)


def report(name, links, truth, label, snr_col="snr_target"):
    recovered = set(links[links.is_true].obj.dropna())
    n_link = len(links); n_true = int(links.is_true.sum()); n_false = n_link - n_true
    purity = n_true / n_link if n_link else float("nan")
    # truth schema: multi-night (ephem_to_inject) has n_nights; same-night (sim_orbits) has only n_sightings.
    if "n_nights" in truth.columns:
        mn = truth[(truth.n_nights >= 3) & (truth.n_sightings >= 6)]   # his multi-night discoverable set
    else:
        mn = truth[truth.n_sightings >= 3]                              # same-night >=3-sighting discoverable
    sn = truth[truth.n_sightings >= 2]                              # any linkable (>=2 sightings)
    def comp(denom):
        if denom.empty:
            return float("nan"), 0, 0
        rec = denom.objID.isin(recovered).sum()
        return rec / len(denom), int(rec), len(denom)
    c_mn = comp(mn); c_sn = comp(sn)
    mn_lbl = "multi-night >=3 nights" if "n_nights" in truth.columns else ">=3 sightings"
    faint = truth[truth[snr_col] < 5]; cf = comp(faint)
    bright = truth[truth[snr_col] >= 10]; cb = comp(bright)
    lines = [f"=== {name} ({label}) ===",
             f"  linkages: {n_link} | TRUE {n_true} / FALSE {n_false} -> purity {purity:.3f}",
             f"  completeness ({mn_lbl}, n={c_mn[2]}): {c_mn[0]:.3f} ({c_mn[1]}/{c_mn[2]})",
             f"  completeness (any >=2 sightings, n={c_sn[2]}): {c_sn[0]:.3f} ({c_sn[1]}/{c_sn[2]})",
             f"  completeness (FAINT SNR<5, n={cf[2]}): {cf[0]:.3f} ({cf[1]}/{cf[2]})",
             f"  completeness (link-bright SNR>=10, n={cb[2]}): {cb[0]:.3f} ({cb[1]}/{cb[2]})"]
    return "\n".join(lines), dict(name=name, label=label, linkages=n_link, true=n_true, false=n_false,
                                  purity=purity, comp_multinight=c_mn[0], comp_any=c_sn[0], comp_faint=cf[0],
                                  recovered=sorted(recovered))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inject", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tol-arcsec", type=float, default=3.0)
    ap.add_argument("--heliolinx", nargs=3, action="append", default=[],
                    metavar=("NAME", "PAIRDETS", "LPCLUST2DET"), help="a heliolinx result to score")
    ap.add_argument("--tracks", nargs=2, action="append", default=[],
                    metavar=("NAME", "TRACKS_CSV"), help="a trail_state_link tracks.csv to score")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    inj = pd.read_csv(a.inject); truth = pd.read_csv(a.truth)
    blocks, rows = [], []
    for name, pdets, c2d in a.heliolinx:
        L = linkages_from_heliolinx(pdets, c2d, inj, a.tol_arcsec)
        txt, row = report(name, L, truth, "stack 5sigma -> his heliolinx chain" if "stack" in name.lower()
                          else "ADCNN -> his heliolinx chain"); blocks.append(txt); rows.append(row)
    for name, trk in a.tracks:
        L = linkages_from_tracks(trk, inj, a.tol_arcsec)
        txt, row = report(name, L, truth, "ADCNN -> our trail_state_link"); blocks.append(txt); rows.append(row)
    rep = "\n".join(blocks)
    print(rep)
    if a.out:
        Path(a.out).write_text(rep + "\n")
        pd.DataFrame(rows).drop(columns=["recovered"]).to_csv(Path(a.out).with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
