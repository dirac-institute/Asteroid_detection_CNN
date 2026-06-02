"""Calibrate the ADCNN score threshold to a LINKAGE false-alarm-rate (FPP) budget.

We want any surviving same-night 2-visit track to be trustworthy at a stated significance: set the
ADCNN stage-2 score floor S so the expected number of FALSE 2-visit linkages per night lambda_FP(S)
<= a target false-alarm rate (FAR). Textbook 3-sigma (one-sided) => FAR 1.35e-3/night => P(a track is
a chance link) ~ lambda_FP <= 0.135% (Poisson).

Method
------
1. For each score floor S: keep dets with score>=S (+ the hard len>=6 px and mask cuts), and per night
   (a) run the full 2-visit pipeline (cluster -> physical_check incl. bound-orbit test) to get the REAL
       surviving tracks + which known objects are recovered (the COST side), and
   (b) estimate lambda_FP(S) by a NULL Monte Carlo: many realizations where each visit's detections are
       given an independent random rigid sky offset (CROSS-EPOCH PERMUTATION) -- this destroys any real
       object's cross-visit continuity while preserving each visit's real FP positions, trail lengths and
       trail-angle distribution. Every surviving track is then a chance/false link. Count per realization
       -> lambda_FP/night with Poisson error. (FP-only -- drop known dets -- is run as a cross-check.)
2. lambda_FP scales as the FP density squared (chance-pair law, empirically confirmed); fit
   log lambda_FP vs log rho across the measurable thresholds and extrapolate to the target FAR -> rho*,
   then map back to the score floor S* via the measured rho(S) curve.

Output: a per-threshold table + the calibrated S* for the requested significance.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
from ADCNN.pipelines.heliolinc.trail_state_link import link, physical_check, crossmatch

PC = dict(pa_tol_deg=20.0, lin_rms_arcsec=1.0, min_epochs=2, pa_tol_2v_deg=10.0,
          orbit_check_2v=True, score_2v_min=0.0, max_arc_2v_min=None,
          perp_collinear_2v_arcsec=None, snr_frac_2v=None)
NPT = 2  # min detections (distinct visits) per track; set from --npt in main()


def permute_epochs(dn, rng, shift_deg=0.2):
    """Give each visit an independent random rigid sky offset (|shift|=shift_deg, random direction).
    Decorrelates real objects across visits (shift >> pos_tol 0.017 deg) while preserving each visit's
    internal FP structure + trail vectors (endpoints shift with the centroid)."""
    o = dn.copy()
    cd = np.cos(np.radians(o.dec.mean()))
    for v in o.visit.unique():
        a = rng.uniform(0, 2 * np.pi)
        dra = shift_deg * np.cos(a) / max(cd, 1e-6); ddec = shift_deg * np.sin(a)
        m = (o.visit == v).to_numpy()
        for c in ("ra", "ra0", "ra1"):
            o.loc[m, c] = o.loc[m, c] + dra
        for c in ("dec", "dec0", "dec1"):
            o.loc[m, c] = o.loc[m, c] + ddec
    return o


def count_tracks(dn):
    _, tr = link(dn, npt=NPT, min_visits=NPT)
    return sum(1 for m in tr if physical_check(dn, m, **PC)[0])


def real_pass(dn, known):
    _, tr = link(dn, npt=NPT, min_visits=NPT)
    objs = []
    for m in tr:
        if physical_check(dn, m, **PC)[0]:
            objs.append(crossmatch(dn, m, known, 5.0, 0.02)[0])
    conf = sorted(o for o in set(objs) if o)
    return conf, sum(1 for o in objs if not o)


def lambda_fp(nights, S, rng, target_events=60, nmin=4, nmax=400, shift_deg=0.2):
    """Estimate lambda_FP/night at score floor S, pooled over nights, with adaptive realizations."""
    # quick estimate with nmin realizations to size the run
    quick = []
    for dn in nights:
        for _ in range(nmin):
            quick.append(count_tracks(permute_epochs(dn, rng, shift_deg)))
    lam0 = max(np.mean(quick), 1e-3)
    nreal = int(np.clip(round(target_events / lam0), nmin, nmax))
    counts = list(quick)
    if nreal > nmin:
        for dn in nights:
            for _ in range(nreal - nmin):
                counts.append(count_tracks(permute_epochs(dn, rng, shift_deg)))
    counts = np.array(counts, float)
    # per-night lambda = total false tracks / number of night-realizations
    lam = counts.mean()
    err = counts.std(ddof=1) / np.sqrt(len(counts)) if len(counts) > 1 else lam
    return lam, err, len(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each with adcnn_dets_masked.csv + known.csv)")
    ap.add_argument("--scores", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97])
    ap.add_argument("--npt", type=int, default=2, help="detections (distinct visits) per track: 2 or 3")
    ap.add_argument("--min-epochs", type=int, default=None, help="distinct epochs (default = --npt)")
    ap.add_argument("--target-far", type=float, default=1.35e-3, help="false tracks/night budget (3sigma one-sided=1.35e-3)")
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--max-arc-min", type=float, default=None, help="2-visit Δt window (min); cap the pair arc to the scheduler pair gap (~30). None=no cap")
    ap.add_argument("--orbit-rate-tol", type=float, default=0.5, help="2-visit bound-orbit velocity-residual tolerance (frac of trail speed); tighter=purer (0.5 loose default, ~0.2 strong)")
    ap.add_argument("--perp-collinear", type=float, default=None, help="2-visit 4-endpoint collinearity RMS tol (arcsec); ~0.3 = strong recall-preserving FP cut. None=off")
    ap.add_argument("--snr-frac", type=float, default=None, help="2-visit brightness-consistency tol |dSNR|/min(SNR); ~0.6. None=off")
    ap.add_argument("--art-frac-max", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mc-target-events", type=int, default=60, help="aim for ~this many false events to size realizations")
    ap.add_argument("--mc-nmax", type=int, default=400, help="max null realizations per night (cap MC cost)")
    ap.add_argument("--out", default="/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/ADCNN/pipelines/heliolinc/link_fpp_calib.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    global NPT, PC
    NPT = a.npt
    PC = {**PC, "min_epochs": (a.min_epochs if a.min_epochs is not None else a.npt),
          "max_arc_2v_min": a.max_arc_min, "orbit_rate_tol": a.orbit_rate_tol,
          "perp_collinear_2v_arcsec": a.perp_collinear, "snr_frac_2v": a.snr_frac}
    print(f"[fpp] tier: npt={NPT} min_epochs={PC['min_epochs']} max_arc_2v_min={PC['max_arc_2v_min']} "
          f"orbit_rate_tol={PC['orbit_rate_tol']} perp={PC['perp_collinear_2v_arcsec']} snr_frac={PC['snr_frac_2v']}", flush=True)

    # load all night-fields
    fields = []  # (label, dets_df, known_df)
    for rd in a.runs:
        d = pd.read_csv(Path(rd) / "adcnn_dets_masked.csv")
        d = d[(d.len_db >= a.len_db_min) & (d.get("art_frac", 0) < a.art_frac_max)].reset_index(drop=True)
        d["night"] = np.floor(d.mjd - 0.5).astype(int)
        known = pd.read_csv(Path(rd) / "known.csv"); known["ObjID"] = known.ObjID.astype(str)
        for ni, dn in d.groupby("night"):
            fields.append((f"{Path(rd).name}:{ni}", dn.reset_index(drop=True), known))
    print(f"[fpp] {len(fields)} night-fields from {len(a.runs)} run(s)", flush=True)

    rows = []
    for S in a.scores:
        nights = [dn[dn.score >= S].reset_index(drop=True) for _, dn, _ in fields]
        rho = np.mean([len(dn) for dn in nights])
        conf_all, new_all = set(), 0
        for (_, _, known), dn in zip(fields, nights):
            c, n = real_pass(dn, known); conf_all |= set(c); new_all += n
        lam, err, nr = lambda_fp(nights, S, rng, target_events=a.mc_target_events, nmax=a.mc_nmax)
        rows.append(dict(score=S, dets_per_night=rho, lambda_fp=lam, lambda_err=err, nreal=nr,
                         known_recovered=len(conf_all), new=new_all, ny2=("2025 NY2" in conf_all)))
        print(f"[fpp] S={S:.2f} rho={rho:7.1f} lambda_FP={lam:8.3f}+-{err:.3f} (n={nr}) "
              f"known={len(conf_all)} NY2={'Y' if '2025 NY2' in conf_all else 'n'} NEW={new_all}", flush=True)

    T = pd.DataFrame(rows)
    # fit log lambda vs log rho over the measurable (non-zero) points; extrapolate to the target FAR
    fit = T[(T.lambda_fp > 0) & (T.dets_per_night > 0)]
    Sstar = rho_star = slope = None
    if len(fit) >= 2:
        b, loga = np.polyfit(np.log(fit.dets_per_night), np.log(fit.lambda_fp), 1)
        slope = float(b)
        rho_star = float(np.exp((np.log(a.target_far) - loga) / b))
        ok = T[T.dets_per_night <= rho_star]          # smallest score floor whose density <= rho*
        Sstar = float(ok.score.min()) if len(ok) else float(T.score.max())
    summary = dict(target_far=a.target_far, sigma="3 (one-sided)", fit_slope=slope,
                   rho_star=rho_star, S_star=Sstar, npt=NPT, min_epochs=PC["min_epochs"], table=rows)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[fpp] lambda_FP ~ rho^{slope:.2f}" if slope is not None else "[fpp] no fit (need >=2 non-zero points)", flush=True)
    if rho_star is not None:
        print(f"[fpp] target FAR {a.target_far:.2e}/night -> rho* ~ {rho_star:.1f} dets/night -> S* ~ {Sstar}", flush=True)
    print(f"[fpp] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
