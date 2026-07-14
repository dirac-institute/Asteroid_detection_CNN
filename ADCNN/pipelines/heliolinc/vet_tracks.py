"""Per-candidate vetting for same-night discovery tracks (generalized from run_real_main/extend_check.py).
For each 3+visit (discovery-grade) track in <alerts>, extend its linear track across ALL same-night visit
epochs in <dets> and count detections (ANY score) at the predicted position: a REAL mover bright enough to
be caught 3x leaves a consistent chain across MANY visits; a false link / artifact does not. Also flags
trail-length consistency (a real constant-velocity object has ~constant trail length). Prints a verdict.
Usage: python vet_tracks.py --dets dets_masked.csv --alerts alerts.jsonl [--tol 8]"""
import argparse, json, numpy as np, pandas as pd


def sep_arcsec(ra1, dec1, ra2, dec2):
    cd = np.cos(np.radians((dec1 + dec2) / 2))
    return np.hypot((ra1 - ra2) * cd, dec1 - dec2) * 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--tol", type=float, default=8.0)
    ap.add_argument("--out", default=None, help="write per-candidate verdicts CSV")
    a = ap.parse_args()
    out_rows = []
    d = pd.read_csv(a.dets)
    vmjd = d.groupby("visit").mjd.median().sort_values()
    nvis = len(vmjd)
    alerts = [json.loads(l) for l in open(a.alerts) if l.strip()]
    cands = [al for al in alerts if al.get("tier") == "3+visit"]
    print(f"[vet] {a.alerts}: {nvis} same-night visits, {len(alerts)} tracks, {len(cands)} three-visit candidates")
    verdicts = []
    for al in cands:
        ep = al["epochs"]
        t = np.array([e["mjd"] for e in ep]); ra = np.array([e["ra"] for e in ep]); dec = np.array([e["dec"] for e in ep])
        lens = [e.get("trail_len_px") or 0 for e in ep]
        t0 = t.mean(); pra = np.polyfit(t - t0, ra, 1); pdec = np.polyfit(t - t0, dec, 1)
        member_visits = {e["visit"] for e in ep}
        hits = extra = 0; hit_visits = []
        for v, mj in vmjd.items():
            dv = d[d.visit == v]
            if not len(dv):
                continue
            s = sep_arcsec(np.polyval(pra, mj - t0), np.polyval(pdec, mj - t0), dv.ra.values, dv.dec.values)
            if s.min() <= a.tol:
                hits += 1; hit_visits.append(int(v))
                if v not in member_visits:
                    extra += 1
        # consecutive-run check: are the hit visits spread across the night or clustered?
        order = [i for i, v in enumerate(vmjd.index) if int(v) in hit_visits]
        spread = (max(order) - min(order) + 1) if order else 0
        len_consistent = (max(lens) / max(min(lens), 1e-3)) < 2.5 if lens else False
        # verdict: real mover should have extra support spread across the night + consistent trails
        good = extra >= 1 and spread >= max(4, nvis // 4) and len_consistent
        verdict = "PLAUSIBLE" if good else "REJECT"
        verdicts.append((al["alertId"], verdict))
        out_rows.append(dict(alertId=al["alertId"], rate_degday=round(al["motion"]["rate_degday"], 3),
                             ra=round(ep[0]["ra"], 6), dec=round(ep[0]["dec"], 6), mjd=round(ep[0]["mjd"], 5),
                             hits=hits, extra=extra, span=spread, nvis=nvis,
                             len_consistent=len_consistent, verdict=verdict))
        print(f"  {al['alertId']} rate={al['motion']['rate_degday']:.2f}deg/day: {hits}/{nvis} visits on track "
              f"({extra} extra beyond members), span={spread}/{nvis} visit-slots, "
              f"trail_len {[round(x,0) for x in lens]} {'consistent' if len_consistent else 'INCONSISTENT'} -> {verdict}")
    plausible = [aid for aid, v in verdicts if v == "PLAUSIBLE"]
    print(f"[vet] -> {len(plausible)} PLAUSIBLE candidates: {plausible if plausible else 'none'}")
    if a.out and out_rows:
        import csv
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
        print(f"[vet] verdicts -> {a.out}")


if __name__ == "__main__":
    main()
