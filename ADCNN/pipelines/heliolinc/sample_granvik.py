"""Sample a NEO orbit + physical-parameter population APPROXIMATING the Granvik (2018) debiased model,
in Sorcha input format (orbits.csv KEP + physical.csv H/GS). Used to drive Sorcha through the real
LSST cadence so each synthetic NEO gets a realistic number of observable apparitions (orbit x cadence;
some only 2x), realistic on-sky rates, and magnitudes.

Distributions (documented; approximate Granvik 2018 NEO marginals -- not the released particle file):
  a   : lognormal peaking ~1.8 AU, clipped [0.6, 4.0]
  e   : Rayleigh sigma~0.35, clipped [0.02, 0.9]
  i   : Rayleigh sigma~12 deg, clipped [0, 60]
  H   : SFD  P(H) ~ 10^(0.5 H) over [16, 25] (more small bodies)
  node, argPeri, M : uniform [0,360)
  GS  : 0.15 (default slope)
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--epoch-mjd", type=float, default=60797.0, help="orbit epoch (MJD TDB), near the sim window")
    ap.add_argument("--h-min", type=float, default=16.0)
    ap.add_argument("--h-max", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-orbits", required=True)
    ap.add_argument("--out-phys", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.n
    # NEO definition: perihelion q = a(1-e) < 1.3 AU. Sample (a,e,i) approximating Granvik 2018 and
    # REJECT non-NEO (q>=1.3) so the population is Earth-approaching (-> close passes -> fast/bright).
    a_au = np.empty(0); e = np.empty(0); inc = np.empty(0)
    while len(a_au) < n:
        m = n * 3
        aa = np.clip(np.exp(rng.normal(np.log(1.6), 0.45, m)), 0.5, 3.5)
        ee = np.clip(rng.rayleigh(0.42, m), 0.02, 0.92)
        ii = np.clip(rng.rayleigh(13.0, m), 0.0, 60.0)
        keep = aa * (1 - ee) < 1.3                    # NEO perihelion cut
        a_au = np.concatenate([a_au, aa[keep]]); e = np.concatenate([e, ee[keep]]); inc = np.concatenate([inc, ii[keep]])
    a_au = a_au[:n]; e = e[:n]; inc = inc[:n]
    node = rng.uniform(0, 360, n); argp = rng.uniform(0, 360, n); ma = rng.uniform(0, 360, n)
    # H from the SFD  P(H) ~ 10^(0.5 H): inverse-CDF sampling on [h_min,h_max]
    al = 0.5; u = rng.uniform(0, 1, n)
    lo, hi = 10 ** (al * a.h_min), 10 ** (al * a.h_max)
    H = np.log10(lo + u * (hi - lo)) / al
    oid = [f"GNEO{i:06d}" for i in range(n)]
    orbits = pd.DataFrame({
        "ObjID": oid, "FORMAT": "KEP", "a": a_au, "e": e, "inc": inc,
        "node": node, "argPeri": argp, "ma": ma, "epochMJD_TDB": a.epoch_mjd})
    phys = pd.DataFrame({
        "ObjID": oid, "H_r": np.round(H, 3),
        "u-r": 0.0, "g-r": 0.0, "i-r": 0.0, "z-r": 0.0, "y-r": 0.0, "GS": 0.15})
    orbits.to_csv(a.out_orbits, index=False)
    phys.to_csv(a.out_phys, index=False)
    print(f"[granvik] {n} NEOs -> {a.out_orbits} / {a.out_phys}")
    print(f"  a {a_au.min():.2f}-{a_au.max():.2f} (med {np.median(a_au):.2f}) | e med {np.median(e):.2f} | "
          f"i med {np.median(inc):.1f} | H {H.min():.1f}-{H.max():.1f} (med {np.median(H):.1f})")


if __name__ == "__main__":
    main()
