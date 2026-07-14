"""Measure PURITY and COMPLETENESS of 2-sighting vs 3-sighting linking on the injected test2 set.

Inputs: the ADCNN detection catalog from the injected off-ecliptic field (adcnn_dets.csv: real FP +
injected-object detections) and the injection truth (inject.csv: per-sighting objID,visit,mjd,ra,dec +
truth_objects.csv: per-objID mag,rate,n_sightings). Off-ecliptic ⇒ no real asteroids, so every detection
is either a matched injected sighting or FP, and every linked track is either a real recovery or a false
link. For k ∈ {2,3} and score floors {0.59, 0.80}:
  - completeness Cₖ = (injected objects with ≥k DETECTED sightings recovered by a ≥k-track) / (those with ≥k detected)
  - purity      Pₖ = (k-tier tracks that recover a true object) / (all k-tier tracks)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np, pandas as pd, sys

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")
from scipy.spatial import cKDTree
from collections import Counter
sys.path.insert(0, str(REPO))
from ADCNN.linking.link_2visit import link, physical_check

TOL_ARCSEC = 5.0


def tag_detections(det, inj):
    """Tag each detection with the injected objID it matches (same visit, within TOL), else 'FP'."""
    det = det.reset_index(drop=True); det["objID"] = "FP"
    iv_by = {v: g for v, g in inj.groupby("visit")}
    for v, dv in det.groupby("visit"):
        iv = iv_by.get(v)
        if iv is None or not len(iv):
            continue
        cd = np.cos(np.radians(iv.dec.to_numpy()))
        tree = cKDTree(np.column_stack([iv.ra.to_numpy() * cd, iv.dec.to_numpy()]))
        cdv = np.cos(np.radians(dv.dec.to_numpy()))
        dist, idx = tree.query(np.column_stack([dv.ra.to_numpy() * cdv, dv.dec.to_numpy()]))
        hit = dist * 3600 <= TOL_ARCSEC
        det.loc[dv.index[hit], "objID"] = iv.objID.to_numpy()[idx[hit]]
    return det


def track_objid(det, members, k):
    """The injected objID a track recovers (>=max(2,k) members share it & not FP), else None (false link)."""
    tags = det.iloc[members].objID.to_numpy()
    c = Counter(t for t in tags if t != "FP")
    if not c:
        return None
    o, n = c.most_common(1)[0]
    return o if n >= min(k, len(members)) and n >= 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(OUTPUTS / "runs/run_test2"))
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--pos-tol", type=float, default=0.05, help="link clustering radius (deg); 0.05 trades cluster-recall up at no purity cost (physical_check is the precision gate)")
    ap.add_argument("--floors", nargs="+", type=float, default=[0.59, 0.80])
    a = ap.parse_args()
    det0 = pd.read_csv(f"{a.run}/adcnn_dets.csv")
    inj = pd.read_csv(f"{a.run}/inject.csv")
    truth = pd.read_csv(f"{a.run}/truth_objects.csv"); truth = truth.set_index("objID")
    det0 = det0[det0.len_db >= a.len_db_min].reset_index(drop=True)
    det0["night"] = np.floor(det0.mjd - 0.5).astype(int)
    det0 = tag_detections(det0, inj)
    out = []
    out.append(f"detections {len(det0)} (len>=6) | real(matched) {int((det0.objID!='FP').sum())} | FP {int((det0.objID=='FP').sum())}")
    out.append(f"injected FAST (>={truth.rate_degday.min():.1f} deg/day) objects {len(truth)} | sightings {len(inj)}")
    # CADENCE CEILING (orbit x real DP2 cadence, Sorcha-grounded): how many fast NEOs are even observable
    # >=k times the same night. This is the hard cap on k-sighting completeness BEFORE any detector/linker
    # loss -- the "some asteroids are only ever seen twice" effect.
    obs = {k: int((truth.k_observable >= k).sum()) for k in (1, 2, 3)}
    out.append(f"CADENCE CEILING: observable >=1x {obs[1]} | >=2x {obs[2]} ({obs[2]/len(truth):.0%}) | "
               f">=3x {obs[3]} ({obs[3]/len(truth):.0%}) of {len(truth)} fast NEOs")
    pc = dict(pa_tol_deg=20, lin_rms_arcsec=1.0, pa_tol_2v_deg=10, orbit_check_2v=True, score_2v_min=0.0)
    rows = []
    for floor in a.floors:
        d = det0[det0.score >= floor].reset_index(drop=True)
        # detected sightings per objID at this floor
        det_sight = d[d.objID != "FP"].groupby("objID").visit.nunique()
        for k in (2, 3):
            _, tracks = link(d, npt=k, min_visits=k, pos_tol_deg=a.pos_tol)
            real_tracks = 0; total_tracks = 0; recovered = set()
            for m in tracks:
                ok, info, nep = physical_check(d, m, min_epochs=k, **pc)
                if not ok:
                    continue
                total_tracks += 1
                o = track_objid(d, m, k)
                if o is not None:
                    real_tracks += 1; recovered.add(o)
            recoverable = set(det_sight[det_sight >= k].index)
            rec = recovered & recoverable
            C = len(rec) / max(len(recoverable), 1)                       # intrinsic linker (given >=k DETECTED)
            C_cad = len(rec) / max(obs[k], 1)                              # survey: of fast NEOs OBSERVABLE >=kx
            P = real_tracks / max(total_tracks, 1)
            rows.append(dict(floor=floor, k=k, tracks=total_tracks, real=real_tracks, false=total_tracks - real_tracks,
                             purity=P, recoverable=len(recoverable), recovered=len(rec), completeness=C,
                             observable=obs[k], completeness_cadence=C_cad))
            out.append(f"floor>={floor} {k}-sighting: tracks {total_tracks} (real {real_tracks}, false {total_tracks-real_tracks}) "
                       f"| PURITY {P:.3f} | C(linker|>={k} detected) {C:.3f} [{len(rec)}/{len(recoverable)}] "
                       f"| C(survey|observable>={k}x) {C_cad:.3f} [{len(rec)}/{obs[k]}]")
    T = pd.DataFrame(rows); T.to_csv(f"{a.run}/recovery_metrics.csv", index=False)
    out.append("\n=== 2x2 for FAST (>=1 deg/day) NEOs at the 3-sigma op-point (score>=0.80) ===")
    for _, r in T[T.floor == max(a.floors)].iterrows():
        out.append(f"  {int(r.k)}-sighting: PURITY {r.purity:.2f} | COMPLETENESS(linker) {r.completeness:.2f} | "
                   f"COMPLETENESS(survey, cadence-folded) {r.completeness_cadence:.2f}  (false links {int(r.false)})")
    # completeness vs mag (3-sighting, floor 0.80)
    open(f"{a.run}/recovery_report.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out)); print("DONE")


if __name__ == "__main__":
    main()
