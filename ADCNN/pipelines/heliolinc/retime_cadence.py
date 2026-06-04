"""Re-stamp the visit times of an off-ecliptic deep field so each adjacent same-night pair sits at a
REALISTIC operational delta-t.

Why: the off-ecliptic dense fields (the clean-FP substrate) revisit the same pointing seconds-to-minutes
apart, hundreds of times -- not the WFD same-night pair cadence the linker is meant for. A fast NEO at
~1 deg/day barely moves in seconds, and the per-pair false rate depends on the pair's delta-t (it sets the
position chord / rate band). So we assign each field's visits SYNTHETIC MJDs whose adjacent gaps are drawn
from the real OpSim baseline same-night consecutive-visit delta-t distribution (median ~34 min, real baseline_v2.0_1yr.db). The diffim
PIXELS (and their real FP) are untouched -- only the time each visit is *labelled* with changes, which is
all the injector (mover motion = rate x delta-t) and the linker (chord/rate band, night grouping) consume.

The linker pairs ADJACENT visits within max_arc (~40 min); by spacing consecutive visits at ~21 min every
adjacent pair becomes a valid synthetic WFD pair (sliding window -> ~n_visits-1 pairs/field, matching the
prior count_realfp measurement), while non-adjacent visits (>~40 min apart) correctly do not pair.

Outputs retime_map.csv: visit, mjd_retimed (+ the empirical delta-t distribution is reusable via build_dt_dist).
"""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
DEFAULT_DB = REPO / "ADCNN/pipelines/heliolinc/run_test2/sorcha/baseline_v2.0_1yr.db"
FOV_DEG = 1.5          # group observations into ~field pointings for the same-night pair gaps
DT_MAX_MIN = 60.0      # a same-night PAIR gap is < this (longer = different visit sequence / night)


def build_dt_dist(opsim_db=DEFAULT_DB, fov_deg=FOV_DEG, dt_max_min=DT_MAX_MIN):
    """Empirical same-night consecutive-visit delta-t (minutes) from the OpSim baseline survey."""
    con = sqlite3.connect(str(opsim_db))
    obs = pd.read_sql("SELECT observationStartMJD AS mjd, fieldRA AS ra, fieldDec AS dec FROM observations", con)
    con.close()
    obs["night"] = np.floor(obs.mjd - 0.5).astype(int)
    rak = (obs.ra // fov_deg).astype(int)
    deck = ((obs.dec + 90.0) // fov_deg).astype(int)
    obs["fk"] = obs.night.astype(str) + "_" + rak.astype(str) + "_" + deck.astype(str)
    dts = []
    for _, g in obs.groupby("fk"):
        if len(g) < 2:
            continue
        d = np.diff(np.sort(g.mjd.values)) * 1440.0
        dts.extend(d[(d > 0) & (d < dt_max_min)])
    dts = np.asarray(dts, float)
    if dts.size < 100:
        raise RuntimeError(f"too few same-night pairs ({dts.size}) from {opsim_db}")
    return dts


def make_retime_map(visits, dt_dist, rng, base_mjd=60000.0, dt_cap_min=39.0):
    """Assign synthetic MJDs: visits in chronological order (visit id sorts by time), consecutive gaps
    sampled from dt_dist (clipped < dt_cap_min so every adjacent pair links under max_arc=40)."""
    vs = np.sort(np.asarray(visits, dtype=np.int64))
    n = len(vs)
    gaps = rng.choice(dt_dist, size=max(n - 1, 0), replace=True)
    gaps = np.clip(gaps, 1.0, dt_cap_min) / 1440.0          # minutes -> days
    mjd = base_mjd + np.concatenate([[0.0], np.cumsum(gaps)])
    return pd.DataFrame({"visit": vs, "mjd_retimed": mjd})


def apply_retime(df, retime_map, col="mjd"):
    """Overwrite a detection catalog's mjd column from the retime map (merge on visit)."""
    m = dict(zip(retime_map.visit.astype(int), retime_map.mjd_retimed.astype(float)))
    out = df.copy()
    out[col] = out.visit.astype(int).map(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="csv with a 'visit' column (the field's visits)")
    ap.add_argument("--out", required=True, help="retime_map.csv output")
    ap.add_argument("--opsim-db", default=str(DEFAULT_DB))
    ap.add_argument("--base-mjd", type=float, default=60000.0)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    dt = build_dt_dist(a.opsim_db)
    print(f"[retime] OpSim same-night dt: n={dt.size} median={np.median(dt):.1f}min "
          f"[{np.percentile(dt,10):.0f},{np.percentile(dt,90):.0f}]", flush=True)
    visits = pd.read_csv(a.manifest, usecols=["visit"]).visit.astype(int).unique()
    rm = make_retime_map(visits, dt, np.random.default_rng(a.seed), base_mjd=a.base_mjd)
    rm.to_csv(a.out, index=False)
    span_h = (rm.mjd_retimed.max() - rm.mjd_retimed.min()) * 24.0
    print(f"[retime] {len(rm)} visits -> {a.out} | synthetic span {span_h:.1f}h "
          f"({len(rm)-1} adjacent pairs @ ~{np.median(dt):.0f}min)", flush=True)


if __name__ == "__main__":
    main()
