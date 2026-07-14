"""DIRECT, permutation-free 2-visit false-link rate on real off-ecliptic diffims (no injection, no
Monte-Carlo). Off-ecliptic => ~zero real asteroids => every surviving 2-track is a genuine false link.
Links each field-night's masked ADCNN detections with the SHIPPED full 2-visit stack (Δt window +
orbit-residual + collinearity + score floor), counts false 2-tracks, and reports lambda_FP per same-night
PAIR with a Poisson confidence interval, vs the 3-sigma budget 1.35e-3. This replaces the null-MC estimate
(which overestimates because it breaks the real cross-visit FP correlation)."""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd, sys
REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")
sys.path.insert(0, str(REPO))
from ADCNN.linking.link_2visit import link, physical_check, chord_seed_pairs
from ADCNN.pipelines.heliolinc.recurrence import add_recurrence

PC = dict(pa_tol_deg=20.0, lin_rms_arcsec=1.0, min_epochs=2, pa_tol_2v_deg=10.0, orbit_check_2v=True,
          orbit_rate_tol=0.25, max_arc_2v_min=30.0, perp_collinear_2v_arcsec=0.30,
          mfsnr_min_2v=None, rate_lo_2v=None, rate_hi_2v=8.0)


def field_false_tracks(d, S, seed="chord"):
    d = d[d.score >= S].reset_index(drop=True)
    if seed == "chord":
        tracks = chord_seed_pairs(d, max_arc_min=PC.get("max_arc_2v_min") or 1e9)
    else:
        _, tracks = link(d, npt=2, min_visits=2, pos_tol_deg=0.017)
    n = 0
    for m in tracks:
        ok, info, nep = physical_check(d, m, **PC)
        if ok and nep == 2:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUTPUTS / "runs/run_realfp"))
    ap.add_argument("--scores", nargs="+", type=float, default=[0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--max-arc-min", type=float, default=30.0)
    ap.add_argument("--len-db-min", type=float, default=6.0)
    ap.add_argument("--recur-max", type=int, default=None, help="recurrence veto: drop dets recurring at the same sky pos in >= this many other visits (TP-safe; 2 is strong). None=off")
    ap.add_argument("--snr-min", type=float, default=None, help="mf_snr floor (5 = standard detection; trades faint completeness for purity)")
    ap.add_argument("--orbit-rate-tol", type=float, default=0.25)
    ap.add_argument("--perp", type=float, default=0.30, help="collinearity tol (arcsec)")
    ap.add_argument("--dsnr", type=float, default=None, help="brightness-consistency |dSNR|/min tol (snr_frac_2v)")
    ap.add_argument("--seed", choices=["chord", "cluster"], default="chord", help="2-visit seeding (chord=position-chord, default; cluster=trail-velocity)")
    ap.add_argument("--chi2-max", type=float, default=None, help="2-visit combined orbit-fit chi^2 gate (~3.0); preferred over the AND-cut knobs")
    ap.add_argument("--mfsnr-min-2v", type=float, default=None, help="2v PHOTOMETRIC purity cut: fainter member matched-filter TRAIL SNR floor (shipped op-point = 10; THE strongest non-ML 2v lever)")
    ap.add_argument("--rate-lo-2v", type=float, default=None, help="2v NEO apparent-rate band low (deg/day; shipped = 1)")
    ap.add_argument("--rate-hi-2v", type=float, default=8.0, help="2v NEO apparent-rate band high (deg/day; shipped = 8)")
    a = ap.parse_args()
    PC["chi2_2v_max"] = a.chi2_max
    PC["max_arc_2v_min"] = a.max_arc_min
    PC["orbit_rate_tol"] = a.orbit_rate_tol
    PC["perp_collinear_2v_arcsec"] = a.perp
    PC["snr_frac_2v"] = a.dsnr
    PC["mfsnr_min_2v"] = (a.mfsnr_min_2v if a.mfsnr_min_2v and a.mfsnr_min_2v > 0 else None)
    PC["rate_lo_2v"] = a.rate_lo_2v
    PC["rate_hi_2v"] = a.rate_hi_2v
    files = sorted(glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv"))
    out = [f"[realfp] {len(files)} field-nights"]
    # total same-night adjacent pairs (gap <= max_arc) across fields
    fields = []
    for f in files:
        d = pd.read_csv(f)
        d = d[(d.len_db >= a.len_db_min) & (d.get("art_frac", 0) < 0.3)].reset_index(drop=True)
        if a.recur_max is not None:
            d = add_recurrence(d)
            d = d[d.recur < a.recur_max].reset_index(drop=True)   # TP-safe: real >=1deg/day movers have recur==0
        if a.snr_min is not None:
            d = d[d.mf_snr >= a.snr_min].reset_index(drop=True)
        vis = sorted(d.visit.unique())
        mj = {v: d[d.visit == v].mjd.median() for v in vis}
        npair = sum(1 for i in range(len(vis) - 1) if (mj[vis[i + 1]] - mj[vis[i]]) * 1440 <= a.max_arc_min)
        fields.append((f, d, npair))
        out.append(f"  {Path(f).name}: {len(vis)} visits, {npair} pairs, {len(d)} dets")
    NP = sum(p for _, _, p in fields)
    out.append(f"[realfp] TOTAL same-night pairs (<= {a.max_arc_min}min): {NP}")
    out.append(f"{'S':>5} {'false_2tracks':>14} {'lambda/pair':>12} {'95%_CL_upper':>13} {'vs 1.35e-3':>11}")
    rows = []
    for S in a.scores:
        nf = sum(field_false_tracks(d, S, seed=a.seed) for _, d, _ in fields)
        lam = nf / max(NP, 1)
        ul95 = (nf + 1.96 * np.sqrt(nf) + 1.92) / max(NP, 1) if nf > 0 else 3.0 / max(NP, 1)  # ~95% Poisson UL
        rows.append(dict(score=S, false=nf, pairs=NP, lambda_pair=lam, ul95=ul95))
        flag = "MEETS 3sigma" if ul95 <= 1.35e-3 else f"{ul95/1.35e-3:.0f}x over"
        out.append(f"{S:>5.2f} {nf:>14} {lam:>12.2e} {ul95:>13.2e} {flag:>11}")
    pd.DataFrame(rows).to_csv(f"{a.dir}/realfp_lambda.csv", index=False)
    rep = "\n".join(out)
    Path(f"{a.dir}/realfp_report.txt").write_text(rep + "\n")
    print(rep); print("DONE")


if __name__ == "__main__":
    main()
