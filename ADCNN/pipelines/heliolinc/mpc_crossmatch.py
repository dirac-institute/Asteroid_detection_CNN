"""Novelty crossmatch for same-night tracks against the known minor-planet sky (IMCCE SkyBoT),
for runs where the Butler SSObject catalog is unavailable (e.g. the `main`-repo DRP campaigns).

For each alert/track, query SkyBoT conesearch at the FIRST epoch's exact (mjd, ra, dec) from the
observatory (Rubin = X05), match the track position to the nearest catalogued small body, and label:
  KNOWN  -- a catalogued object within --tol arcsec (a re-detection)
  NEW    -- no catalogued object within --tol -> uncatalogued CANDIDATE (vet further)
Output: <run>/crossmatch.csv [alertId, tier, ra, dec, mjd, speed_degday, status, match_obj, match_class,
match_mag, sep_arcsec]. SkyBoT queries are rate-limited (--sleep). Needs outbound HTTPS.
"""
from __future__ import annotations
import argparse, json, time, urllib.request, urllib.parse, math
from pathlib import Path

SKYBOT = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"


def _hms_to_deg(h):
    p = h.split()
    return (abs(float(p[0])) + float(p[1]) / 60 + float(p[2]) / 3600) * 15 * (-1 if h.strip().startswith("-") else 1)


def _dms_to_deg(d):
    p = d.split(); sign = -1 if d.strip().startswith("-") else 1
    return sign * (abs(float(p[0])) + float(p[1]) / 60 + float(p[2]) / 3600)


def _sep_arcsec(ra1, dec1, ra2, dec2):
    cd = math.cos(math.radians((dec1 + dec2) / 2))
    return math.hypot((ra1 - ra2) * cd, dec1 - dec2) * 3600.0


def skybot_conesearch(ra, dec, jd, radius_deg, loc, timeout=30):
    """Return list of dicts for catalogued bodies in the cone at epoch jd. [] if none."""
    params = {"-ra": ra, "-dec": dec, "-rd": radius_deg, "-ep": f"{jd:.6f}", "-loc": loc,
              "-mime": "text", "-output": "object", "-filter": 0}
    url = SKYBOT + "?" + urllib.parse.urlencode(params)
    txt = urllib.request.urlopen(url, timeout=timeout).read().decode()
    out = []
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = [c.strip() for c in line.split("|")]
        if len(f) < 8:
            continue
        try:
            out.append(dict(name=f[1], ra=_hms_to_deg(f[2]), dec=_dms_to_deg(f[3]),
                            cls=f[4], mag=f[5]))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True, help="alerts.jsonl from the linker")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loc", default="X05", help="MPC obs code (Rubin = X05)")
    ap.add_argument("--tol", type=float, default=10.0, help="match radius arcsec")
    ap.add_argument("--radius-deg", type=float, default=0.03, help="conesearch radius (deg)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between queries (be polite)")
    a = ap.parse_args()

    alerts = [json.loads(l) for l in open(a.alerts) if l.strip()]
    rows = []; n_known = n_new = n_err = 0
    for al in alerts:
        ep = al["epochs"][0]
        ra, dec, mjd = float(ep["ra"]), float(ep["dec"]), float(ep["mjd"])
        jd = mjd + 2400000.5
        spd = al.get("motion", {}).get("rate_degday")
        status, mobj, mcls, mmag, msep = "NEW", "", "", "", ""
        try:
            cat = skybot_conesearch(ra, dec, jd, a.radius_deg, a.loc)
            if cat:
                best = min(cat, key=lambda o: _sep_arcsec(ra, dec, o["ra"], o["dec"]))
                sep = _sep_arcsec(ra, dec, best["ra"], best["dec"])
                if sep <= a.tol:
                    status, mobj, mcls, mmag, msep = "KNOWN", best["name"], best["cls"], best["mag"], round(sep, 2)
                else:
                    msep = round(sep, 2)   # nearest catalogued body, but beyond tol
            n_known += status == "KNOWN"; n_new += status == "NEW"
        except Exception as e:
            status = "ERR:" + type(e).__name__; n_err += 1
        rows.append(dict(alertId=al.get("alertId"), tier=al.get("tier"), priority=al.get("priority"),
                         ra=round(ra, 6), dec=round(dec, 6), mjd=round(mjd, 5), speed_degday=spd,
                         status=status, match_obj=mobj, match_class=mcls, match_mag=mmag, sep_arcsec=msep))
        time.sleep(a.sleep)
    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[crossmatch] {len(rows)} tracks -> KNOWN {n_known} | NEW {n_new} | ERR {n_err} -> {a.out}")
    news = [r for r in rows if r["status"] == "NEW"]
    if news:
        print(f"[crossmatch] {len(news)} NEW (uncatalogued) candidates:")
        for r in sorted(news, key=lambda r: (r["priority"] or 9, -(r["speed_degday"] or 0)))[:20]:
            print(f"   {r['alertId']} {r['tier']} ra={r['ra']} dec={r['dec']} "
                  f"speed={r['speed_degday']} nearest_cat={r['sep_arcsec']}\"")


if __name__ == "__main__":
    main()
