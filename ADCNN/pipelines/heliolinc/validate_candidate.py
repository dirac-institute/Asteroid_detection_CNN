"""Validate the single NEW candidate from the band run (night 60867).
  1. Re-link night 60867, find the surviving track, dump its member detections (full provenance).
  2. Randomized-trail-angle NULL test: scramble each trail's PA, re-link+physical_check N times,
     count surviving tracks -> false-link rate for this night.
  3. Nearest known object to the track (confirm truly uncatalogued, not just outside 5").
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
from ADCNN.pipelines.heliolinc.trail_state_link import link, physical_check, fit_residual, crossmatch

HL = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/run_band"
NIGHT = 60867
ART_MAX, LEN_MIN, NPT, PA_TOL, MAX_RMS, MINEP = 0.3, 6.0, 3, 20.0, 1.0, 3

d = pd.read_csv(f"{HL}/adcnn_dets_masked.csv")
if "art_frac" in d: d = d[d.art_frac < ART_MAX]
if "len_db" in d:   d = d[d.len_db >= LEN_MIN]
d = d.reset_index(drop=True)
d["night"] = np.floor(d.mjd - 0.5).astype(int)
dn = d[d.night == NIGHT].reset_index(drop=True)
known = pd.read_csv(f"{HL}/known.csv")
print(f"night {NIGHT}: {len(dn)} dets after cuts")

# 1. find the surviving track + dump members
labels, tracks = link(dn, npt=NPT, min_visits=NPT)
passed = []
for members in tracks:
    ok, info = physical_check(dn, members, pa_tol_deg=PA_TOL, lin_rms_arcsec=MAX_RMS, min_epochs=MINEP)
    if ok:
        passed.append((members, info))
print(f"{len(tracks)} raw tracks -> {len(passed)} pass physical_check\n")
for members, info in passed:
    g = dn.iloc[members].sort_values("mjd")
    rms, speed = fit_residual(dn, members)
    obj, frac = crossmatch(dn, members, known, 5.0, 0.02)
    print(f"=== TRACK  {info}\n    speed {speed:.3f} deg/day  rms {rms:.3f}\"  match='{obj}' frac={frac}")
    cols = [c for c in ["visit","detector","mjd","ra","dec","ra0","dec0","ra1","dec1","len_db","score","snr","art_frac"] if c in g.columns]
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(g[cols].to_string(index=False))
    # nearest known object across the night (any time)
    kk = known.copy()
    cd = np.cos(np.radians(g.dec.mean()))
    best = None
    for _, r in g.iterrows():
        sep = np.hypot((kk.ra - r.ra)*cd, kk.dec - r.dec)*3600
        dt = np.abs(kk.mjd - r.mjd)*86400
        j = int(np.argmin(sep))
        if best is None or sep.iloc[j] < best[0]:
            best = (float(sep.iloc[j]), str(kk.ObjID.iloc[j]), float(dt.iloc[j]))
    print(f"    nearest known: {best[1]}  sep {best[0]:.1f}\"  dt {best[2]:.0f}s\n")

# 2. NULL test: scramble trail angles, keep magnitudes + positions, re-link
def scramble(df, rng):
    o = df.copy()
    cd = np.cos(np.radians(o.dec.to_numpy()))
    half_len = 0.5*np.hypot((o.ra1-o.ra0)*cd, o.dec1-o.dec0).to_numpy()  # deg
    ang = rng.uniform(0, np.pi, len(o))
    dx = half_len*np.cos(ang)/np.where(cd==0,1,cd); dy = half_len*np.sin(ang)
    o["ra0"]=o.ra-dx; o["ra1"]=o.ra+dx; o["dec0"]=o.dec-dy; o["dec1"]=o.dec+dy
    return o

rng = np.random.default_rng(12345)
NTRIAL = 50
counts = []
for _ in range(NTRIAL):
    s = scramble(dn, rng)
    _, tr = link(s, npt=NPT, min_visits=NPT)
    npass = sum(physical_check(s, m, pa_tol_deg=PA_TOL, lin_rms_arcsec=MAX_RMS, min_epochs=MINEP)[0] for m in tr)
    counts.append(npass)
counts = np.array(counts)
print(f"NULL TEST (scrambled trail angles, {NTRIAL} trials): "
      f"mean {counts.mean():.3f} surviving tracks/trial, max {counts.max()}, "
      f"trials with >=1: {(counts>=1).sum()}/{NTRIAL}")
