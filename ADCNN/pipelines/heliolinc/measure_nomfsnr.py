#!/usr/bin/env python3
"""Measure faint-fast 2v ALERT completeness + false-link rate lambda with mfsnr_min_2v swept (incl. DROPPED),
to quantify the completeness we gain by removing the photometric cut on the alert sub-stream.

KEY EFFICIENCY: the expensive work (chord_seed_pairs + physical_check's bound-orbit fit) is INDEPENDENT of
mfsnr and of the score floor above smin (physical_check geometry is score-independent; line 54 monotonicity).
So we do ONE orbit-solve pass per field at the lowest floor and record a PER-PAIR TABLE
(min_score, min_mf_snr, rate, label, n_fp). completeness(mfsnr,S) and lambda(mfsnr,S) are then cheap
post-hoc reductions over that table -- no re-solve for any (mfsnr, S>=smin).

The orbit solve is gated by pair_chi2's cheap PA/speed pre-gate (returns inf before astropy/Lambert), so the
dense-field cliff is pair ENUMERATION, not solves. We record only pairs that PASS the full geometric+chi2 gate
(mfsnr & rate withheld), which is exactly the set whose mfsnr/score we want to sweep.

Per-field JSON cache (resumable). Aggregate anytime with --aggregate-only.
"""
import argparse, json, os, glob
from pathlib import Path
import numpy as np
import pandas as pd

import ADCNN.pipelines.heliolinc.sweep_S as sw
from ADCNN.linking.link_2visit import chord_seed_pairs, physical_check, pair_chi2

REPO = Path(os.environ.get("ADCNN_REPO") or Path(__file__).resolve().parents[3])
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS") or REPO / "outputs")

# Geometry op = shipped, MINUS the mfsnr cut and rate cut (swept/applied post-hoc).
PCHECK = dict(pa_tol_deg=20.0, lin_rms_arcsec=1.0, min_epochs=2, pa_tol_2v_deg=10.0, orbit_check_2v=True,
              orbit_rate_tol=0.5, max_arc_2v_min=40.0, chi2_2v_max=5.0)


def _cache_path(d_dir, k, smin):
    # v2: rows carry chi2 + geometry components (priorityScore inputs) -- new cache namespace so the
    # old 6/8-field caches are not silently reused with the wrong row format.
    return f"{d_dir}/_nomfsnr_cache/{k}_smin{smin}_v2.json"


def eval_field(d_dir, k, smin, max_seed_pairs=None):
    """Single orbit-solve pass: seed at smin, run geometric+chi2 check (no mfsnr/rate), record per-pair table.
    Returns (rows, recoverable{objID:snr}, n_seed, n_capped). rows = [(min_score, min_mfsnr, rate, label, n_fp)]
    label: 'tp' (both same injected obj), 'fp'. n_fp: number of non-injected members (0/1/2)."""
    _, d, recoverable = sw._load_field(d_dir, k, 6.0, 0.3, 2, 1e9)
    ds = d[d.score >= smin].reset_index(drop=True)
    if len(ds) == 0:
        return [], recoverable, 0, 0, 1.0
    sc = ds.score.to_numpy(); oid = ds.objID.to_numpy()
    mfs = ds.mf_snr.to_numpy() if "mf_snr" in ds else np.full(len(ds), np.nan)
    ra = ds.ra.to_numpy(); dec = ds.dec.to_numpy(); mjd = ds.mjd.to_numpy()
    seeds = chord_seed_pairs(ds, max_arc_min=PCHECK["max_arc_2v_min"], max_visit_pairs=200)
    n_seed = len(seeds)
    n_capped = 0
    if max_seed_pairs is not None and n_seed > max_seed_pairs:
        # bound enumeration cost: keep all pairs with >=1 injected member (preserves completeness exactly),
        # subsample FP-FP pairs by fraction f -> rescale FP+FP lambda by 1/f at aggregate (logged).
        inj = [s for s in seeds if (pd.notna(oid[s[0]]) or pd.notna(oid[s[1]]))]
        fp = [s for s in seeds if not (pd.notna(oid[s[0]]) or pd.notna(oid[s[1]]))]
        keep_fp = max_seed_pairs - len(inj)
        if keep_fp < 0:
            keep_fp = 0
        f = keep_fp / max(len(fp), 1)
        # deterministic stride subsample (no RNG; reproducible)
        stride = max(int(round(1.0 / f)), 1) if f > 0 else len(fp) + 1
        fp = fp[::stride]
        n_capped = n_seed - (len(inj) + len(fp))
        seeds = inj + fp
        fp_subsample_f = 1.0 / stride
    else:
        fp_subsample_f = 1.0
    lens = ds.len_db.to_numpy() if "len_db" in ds else np.full(len(ds), np.nan)
    rows = []
    for m in seeds:
        ok, _info, nep = physical_check(ds, m, **PCHECK)
        if not (ok and nep == 2):
            continue
        i, j = m
        cd = np.cos(np.radians(dec[i])); dt = abs(mjd[j] - mjd[i])
        rate = np.hypot((ra[j] - ra[i]) * cd, dec[j] - dec[i]) / dt if dt > 0 else 0.0
        same = pd.notna(oid[i]) and oid[i] == oid[j]
        n_fp = int(pd.isna(oid[i])) + int(pd.isna(oid[j]))
        label = "tp" if same else "fp"
        obj = oid[i] if same else None
        # v2 row: (min_score, min_mfsnr, rate, label, n_fp, obj, max_score, min_len, chi2, dpa_tm, dspeed,
        # perp) -- the full priorityScore/ranking inputs. pair_chi2 re-fit only on PASSING pairs (~45/field,
        # negligible cost); components: dpa_tm = trail-vs-motion PA residual (deg), dspeed = trail-rate vs
        # chord-rate residual (frac), perp = collinearity rms (arcsec).
        c2, ci = pair_chi2(ds.iloc[[i, j]], 30.0)
        rows.append((float(min(sc[i], sc[j])), float(min(mfs[i], mfs[j])), float(rate), label, n_fp,
                     obj if obj is not None else "", float(max(sc[i], sc[j])),
                     float(min(lens[i], lens[j])), float(c2), float(ci.get("dpa_tm", np.nan)),
                     float(ci.get("dspeed", np.nan)), float(ci.get("perp", np.nan))))
    return rows, recoverable, n_seed, n_capped, fp_subsample_f


def _worker(args):
    d_dir, k, smin, max_seed_pairs = args
    cp = _cache_path(d_dir, k, smin)
    if os.path.exists(cp):
        return k, "cached"
    try:
        rows, recoverable, n_seed, n_capped, f = eval_field(d_dir, k, smin, max_seed_pairs)
        json.dump({"rows": rows, "rec": recoverable, "n_seed": n_seed, "n_capped": n_capped, "fp_f": f},
                  open(cp + ".tmp", "w"))
        os.replace(cp + ".tmp", cp)
        return k, f"done seed={n_seed} capped={n_capped} pass={len(rows)}"
    except Exception as e:
        return k, f"ERR {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUTPUTS / "runs/run_lambda"),
                    help="run dir with adcnn_dets_masked_*.csv; fresh caches land in <dir>/_nomfsnr_cache "
                         "(the FROZEN 82-field caches are committed at ADCNN/pipelines/heliolinc/run_lambda/)")
    ap.add_argument("--smin", type=float, default=0.8)
    ap.add_argument("--max-seed-pairs", type=int, default=None,
                    help="per-field enumeration guard; FP-FP pairs subsampled above this, completeness exact")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(f"{a.dir}/_nomfsnr_cache", exist_ok=True)
    ks = sorted({os.path.basename(f).split("adcnn_dets_masked_")[1].rsplit(".csv", 1)[0]
                 for f in glob.glob(f"{a.dir}/adcnn_dets_masked_*.csv")})

    if not a.aggregate_only:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = [(a.dir, k, a.smin, a.max_seed_pairs) for k in ks]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(_worker, t) for t in tasks]):
                k, msg = fut.result()
                print(f"[field {k}] {msg}", flush=True)

    # aggregate
    allrows = []; allrec = {}; fpf = {}; capped_fields = []
    for k in ks:
        cp = _cache_path(a.dir, k, a.smin)
        if not os.path.exists(cp):
            continue
        c = json.load(open(cp))
        for r in c["rows"]:
            allrows.append((k, *r))
        for o, s in c["rec"].items():
            allrec[f"{k}_{o}"] = float(s)
        fpf[k] = c.get("fp_f", 1.0)
        if c.get("n_capped", 0) > 0:
            capped_fields.append((k, c["n_capped"], c.get("fp_f", 1.0)))

    tot25 = sum(1 for s in allrec.values() if 2 <= s < 5)
    tot510 = sum(1 for s in allrec.values() if 5 <= s < 10)
    ff_tot = tot25 + tot510
    print(f"\n# recoverable: total={len(allrec)} faint-fast(2-10)={ff_tot} (2-5={tot25},5-10={tot510})")
    n_fields = len([k for k in ks if os.path.exists(_cache_path(a.dir, k, a.smin))])
    print(f"# fields aggregated: {n_fields}; FP-subsampled fields: {len(capped_fields)} {capped_fields}")

    scores = [round(a.smin + 0.05 * i, 2) for i in range(int((0.9 - a.smin) / 0.05) + 1)]
    rate_lo, rate_hi = 1.0, 8.0
    # lambda is reported as FALSE faint-fast 2v links per field-night (rate-banded). Purity at the REAL base
    # rate is computed offline: purity = C*rho_cnt / (C*rho_cnt + lambda), rho_cnt = expected real recoverable
    # faint-fast movers per field-night (~rho * field_area; rho=0.14/deg^2/night, fields ~one tract).
    # lambda split by #FP members: FPFP (n_fp==2) = the REAL-DATA false rate (injection-independent);
    # injFP (n_fp==1) = injected x real-FP, INFLATED by our artificial injection density (NOT the real rate).
    print(f"\n{'mfsnr':>6} {'S':>5} | {'ff_C%':>7} {'g25':>4} {'g510':>5} | {'lamFPFP':>8} {'lamInjFP':>9}  (FPFP=real-data false rate/field)")
    for mf_thresh in [0.0, 5.0, 7.0, 10.0]:
        for S in scores:
            recset = set(); fpfp = 0.0; injfp = 0.0
            for (k, smin_s, mfmin, rate, label, n_fp, obj, *_extra) in allrows:
                if smin_s < S:
                    continue
                if rate < rate_lo or rate > rate_hi:
                    continue
                if mf_thresh > 0 and mfmin < mf_thresh:
                    continue
                if label == "tp":
                    recset.add(f"{k}_{obj}")
                elif n_fp == 2:
                    fpfp += (1.0 / fpf[k]) if fpf[k] < 1.0 else 1.0   # FP-FP subsampled -> rescale
                else:
                    injfp += 1.0                                       # inj-FP not subsampled
            g25 = sum(1 for o in recset if 2 <= allrec.get(o, -1) < 5)
            g510 = sum(1 for o in recset if 5 <= allrec.get(o, -1) < 10)
            ffC = 100.0 * (g25 + g510) / ff_tot if ff_tot else 0.0
            print(f"{mf_thresh:>6.0f} {S:>5.2f} | {ffC:>6.2f}% {g25:>4} {g510:>5} | "
                  f"{fpfp/max(n_fields,1):>8.3f} {injfp/max(n_fields,1):>9.3f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
