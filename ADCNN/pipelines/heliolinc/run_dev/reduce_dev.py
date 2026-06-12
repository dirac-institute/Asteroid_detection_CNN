#!/usr/bin/env python3
"""Blind-test FINAL reduction at the frozen operating points (EVALUATION_CONTRACT.md). NO TUNING.

Conventions are IDENTICAL to the validation reducer (Evaluation/threshold_selection_plots.py):
  - rows are post-physical_check pairs (chi2<=5 / PA / arc gates already enforced in PCHECK);
  - frozen 2v alert op = min_score>=0.80 AND min_mfsnr>=5 AND rate in [1,8];
  - faint-fast completeness = distinct injected objects (2<=snr_target<10) with >=1 accepted pair,
    over ALL recoverable faint-fast objects (rec dict: n_sightings>=2 on retimed cadence);
  - in-sample purity = TP/(TP+FP) pairs at injected density ("validation injected-truth fraction");
  - field bootstrap 16-84% bands (fields are the independent unit);
  - ranking: priorityScore = base + 0.95*min_score -> per-field-night top-N truth fraction.
Per-latitude split: fields 0-19 off-ecliptic (clean-FP substrate), 24-29 ecliptic (REAL asteroids
present -> their unmatched pairs count as 'fp' here: the ecliptic purity readout is CONSERVATIVE).
"""
import glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{HERE}/_nomfsnr_cache"
OFF_ECL = [str(k) for k in range(21)]
ECL = []
OP = dict(smin=0.80, mfmin=5.0, rlo=1.0, rhi=8.0)   # frozen op_2v_alert.json -- read-only
N_BOOT, SEED = 300, 42


def load():
    rows, rec, ks = [], {}, []
    for f in sorted(glob.glob(f"{CACHE}/*_smin0.5_v3exact.json")):
        k = os.path.basename(f).split("_smin")[0]
        ks.append(k)
        c = json.load(open(f))
        for r in c["rows"]:
            rows.append((k, *r))
        for o, s in c["rec"].items():
            rec[f"{k}_{o}"] = float(s)
    return ks, rows, rec


def reduce_subset(ks, rows, rec, fields, tag):
    sel = [k for k in fields if k in ks]
    selset = set(sel)
    R = [(k, mn, mf, rate, lab, obj) for (k, mn, mf, rate, lab, nfp, obj, mx, ln, c2, dpa, dsp, perp)
         in rows if k in selset]
    recs = {o: s for o, s in rec.items() if o.split("_", 1)[0] in selset}
    ff_tot = sum(1 for s in recs.values() if 2 <= s < 10)
    at_op = [r for r in R if r[1] >= OP["smin"] and r[2] >= OP["mfmin"] and OP["rlo"] <= r[3] <= OP["rhi"]]
    tp = [r for r in at_op if r[4] == "tp"]
    ff_key = lambda k, obj: f"{k}_{obj}"
    is_ff = lambda k, obj: bool(obj) and 2 <= recs.get(ff_key(k, obj), -1) < 10
    ff_obj = {ff_key(k, obj) for (k, mn, mf, rate, lab, obj) in tp if is_ff(k, obj)}
    C = 100 * len(ff_obj) / ff_tot if ff_tot else float("nan")
    P = 100 * len(tp) / len(at_op) if at_op else float("nan")
    # field bootstrap (16-84%)
    fidx = {k: i for i, k in enumerate(sel)}; NF = len(sel)
    tp_f = np.zeros(NF); n_f = np.zeros(NF); den_f = np.zeros(NF); obj_f = np.zeros(NF)
    per = {}
    for (k, mn, mf, rate, lab, obj) in at_op:
        n_f[fidx[k]] += 1
        if lab == "tp":
            tp_f[fidx[k]] += 1
            if is_ff(k, obj):
                per.setdefault(k, set()).add(obj)
    for k, s_ in per.items():
        obj_f[fidx[k]] = len(s_)
    for o, s in recs.items():
        if 2 <= s < 10:
            den_f[fidx[o.split("_", 1)[0]]] += 1
    rng = np.random.default_rng(SEED)
    Cs, Ps = [], []
    for _ in range(N_BOOT):
        b = rng.integers(0, NF, NF)
        Cs.append(100 * obj_f[b].sum() / max(den_f[b].sum(), 1))
        t = n_f[b].sum()
        Ps.append(100 * tp_f[b].sum() / t if t else np.nan)
    # ranked alert stream: per field, rank pairs by priorityScore = 0.95*min_score (monotonic in
    # min_score); top-N truth fraction = TP among the N highest-ranked pairs, summed over fields
    topn = {}
    for N in (5, 10, 50):
        t_n = tot_n = 0
        for k in sel:
            pk = sorted([r for r in at_op if r[0] == k], key=lambda r: -r[1])[:N]
            t_n += sum(1 for r in pk if r[4] == "tp"); tot_n += len(pk)
        topn[f"top{N}"] = dict(tp=t_n, n=tot_n, frac=round(t_n / tot_n, 4) if tot_n else None)
    return dict(tag=tag, n_fields=NF, pairs_post_gate=len(R), pairs_at_op=len(at_op),
                tp_at_op=len(tp), fp_at_op=len(at_op) - len(tp),
                purity_insample_pct=round(P, 2) if P == P else None,
                purity_band=[round(float(np.nanpercentile(Ps, 16)), 2), round(float(np.nanpercentile(Ps, 84)), 2)],
                ff_recovered=len(ff_obj), ff_recoverable=ff_tot,
                completeness_ff_pct=round(C, 2) if C == C else None,
                completeness_band=[round(float(np.percentile(Cs, 16)), 2), round(float(np.percentile(Cs, 84)), 2)],
                alerts_per_fieldnight=round(len(at_op) / NF, 2) if NF else None,
                ranked=topn)


def main():
    ks, rows, rec = load()
    print(f"loaded {len(ks)} field caches: {ks}")
    out = [reduce_subset(ks, rows, rec, ks, "ALL"),
           reduce_subset(ks, rows, rec, OFF_ECL, "off-ecliptic (0-19)"),
           reduce_subset(ks, rows, rec, ECL, "ecliptic (24-29, fp conservative)")]
    for s in out:
        print(json.dumps(s, indent=1))
    json.dump(out, open(f"{HERE}/blind_frozen_op_reduction.json", "w"), indent=1)
    c_all = out[0]["completeness_ff_pct"]
    verdict = "PASS" if (c_all is not None and c_all >= 3.0) else "FAIL"
    print(f"FAILURE CRITERION (faint-fast C >= 3.0% = half validation): C={c_all}% -> {verdict}")
    print("REDUCE_BLIND_DONE")


if __name__ == "__main__":
    main()
