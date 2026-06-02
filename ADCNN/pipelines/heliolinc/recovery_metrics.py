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
import numpy as np, pandas as pd, sys
from scipy.spatial import cKDTree
from collections import Counter
sys.path.insert(0, "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
from ADCNN.pipelines.heliolinc.trail_state_link import link, physical_check

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
    ap.add_argument("--run", default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/run_test2")
    ap.add_argument("--len-db-min", type=float, default=6.0)
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
    out.append(f"injected objects {len(truth)} | sightings {len(inj)}")
    pc = dict(pa_tol_deg=20, lin_rms_arcsec=1.0, pa_tol_2v_deg=10, orbit_check_2v=True, score_2v_min=0.0)
    rows = []
    for floor in a.floors:
        d = det0[det0.score >= floor].reset_index(drop=True)
        # detected sightings per objID at this floor
        det_sight = d[d.objID != "FP"].groupby("objID").visit.nunique()
        for k in (2, 3):
            _, tracks = link(d, npt=k, min_visits=k)
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
            C = len(recovered & recoverable) / max(len(recoverable), 1)
            P = real_tracks / max(total_tracks, 1)
            rows.append(dict(floor=floor, k=k, tracks=total_tracks, real=real_tracks, false=total_tracks - real_tracks,
                             purity=P, recoverable=len(recoverable), recovered=len(recovered & recoverable), completeness=C))
            out.append(f"floor>={floor} {k}-sighting: tracks {total_tracks} (real {real_tracks}, false {total_tracks-real_tracks}) "
                       f"| PURITY {P:.3f} | recoverable {len(recoverable)} recovered {len(recovered & recoverable)} | COMPLETENESS {C:.3f}")
    T = pd.DataFrame(rows); T.to_csv(f"{a.run}/recovery_metrics.csv", index=False)
    out.append("\n=== 2x2 (at score>=0.80) ===")
    for _, r in T[T.floor == max(a.floors)].iterrows():
        out.append(f"  {int(r.k)}-sighting: purity {r.purity:.2f}  completeness {r.completeness:.2f}  (tracks {int(r.tracks)}, false {int(r.false)})")
    # completeness vs mag (3-sighting, floor 0.80)
    open(f"{a.run}/recovery_report.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out)); print("DONE")


if __name__ == "__main__":
    main()
