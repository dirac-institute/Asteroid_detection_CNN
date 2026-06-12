#!/usr/bin/env python3
"""#251: detector miss audit on the blind set — defines the retraining objective. Measurement only.

Three-way census:
  (a) stack-found / ADCNN-missed sightings  — by snr_target, true trail length, band, latitude;
  (b) ADCNN-found / stack-missed sightings  — the ADCNN value region, same axes;
  (c) high-score ADCNN FP taxonomy          — mask-plane composition, len_db, mf_snr vs TP at S>=0.80.

Per injected sighting the category at the ADCNN alert floor (S>=0.80) and retention floor (S>=0.50):
both | stack-only | adcnn-only | neither (stack = 5sigma full catalog).
"""
import json, os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
KS = list(range(21))
TOL = 10.0
SNRB = [(2, 5), (5, 10), (10, 31)]
LENB = [(6, 12), (12, 20), (20, 41)]
MASKS = ["m_SPIKE", "m_SAT", "m_CR", "m_STREAK", "m_BAD", "m_SUSPECT", "m_EDGE",
         "m_DETECTED_NEGATIVE", "m_HIGH_VARIANCE", "m_INTRP"]


def main():
    sight = []          # per-sighting rows
    fp_parts, tp_parts = [], []   # (c) high-score det rows
    for k in KS:
        inj = pd.read_csv(f"{HERE}/inject_{k}.csv")
        t = pd.read_csv(f"{HERE}/truth_{k}.csv").set_index("objID")
        man = pd.read_csv(f"{HERE}/manifest_{k}.csv", usecols=["visit", "detector", "band"]).drop_duplicates(["visit", "detector"])
        band = {(int(r.visit), int(r.detector)): r.band for r in man.itertuples()}
        sd = pd.read_csv(f"{HERE}/stack_dets_s5_{k}.csv")          # per-sighting stack hits
        ad = pd.read_csv(f"{HERE}/adcnn_dets_masked_{k}.csv")
        shit = {(r.objID, int(r.visit), int(r.detector)): bool(r.stack_det) for r in sd.itertuples()}
        ga = dict(tuple(ad.groupby(["visit", "detector"])))
        inj["snr_t"] = inj.objID.map(t.snr_target)
        for (v, det), g in inj.groupby(["visit", "detector"]):
            adp = ga.get((v, det))
            if adp is not None and len(adp):
                tree = cKDTree(adp[["x", "y"]].to_numpy())
                d, i = tree.query(g[["x", "y"]].to_numpy(), distance_upper_bound=TOL)
                hit = np.isfinite(d)
                sc = np.where(hit, adp.score.to_numpy()[np.clip(i, 0, len(adp) - 1)], 0.0)
            else:
                hit = np.zeros(len(g), bool); sc = np.zeros(len(g))
            for j, r in enumerate(g.itertuples()):
                sight.append(dict(field=k, ecl=k >= 24, band=band.get((v, det), "?"),
                                  snr_t=float(r.snr_t), tlen=float(r.trail_length),
                                  s5hit=shit.get((r.objID, v, det), False),
                                  a50=bool(hit[j]), a80=bool(hit[j] and sc[j] >= 0.80),
                                  score=float(sc[j])))
        # (c): high-score dets, fp vs tp
        a8 = ad[ad.score >= 0.80].copy()
        if len(a8):
            ti = cKDTree(inj[["x", "y"]].to_numpy())
            lab = np.zeros(len(a8), bool)
            for key, gg in a8.groupby(["visit", "detector"]):
                gi = inj[(inj.visit == key[0]) & (inj.detector == key[1])]
                if len(gi):
                    d, _ = cKDTree(gi[["x", "y"]].to_numpy()).query(gg[["x", "y"]].to_numpy(), distance_upper_bound=TOL)
                    lab[a8.index.get_indexer(gg.index)] = np.isfinite(d)
            a8["is_tp"] = lab
            cols = ["score", "len_db", "mf_snr", "art_frac", "is_tp"] + [m for m in MASKS if m in a8.columns]
            fp_parts.append(a8.loc[~a8.is_tp, cols]); tp_parts.append(a8.loc[a8.is_tp, cols])
    S = pd.DataFrame(sight)
    FP = pd.concat(fp_parts, ignore_index=True); TP = pd.concat(tp_parts, ignore_index=True)

    def cat(m):
        return dict(both=int((m["s5hit"] & m.a80).sum()), stack_only=int((m["s5hit"] & ~m.a80).sum()),
                    adcnn_only=int((~m["s5hit"] & m.a80).sum()), neither=int((~m["s5hit"] & ~m.a80).sum()))

    out = {"overall@a80": cat(S), "overall@a50": dict(
        both=int((S["s5hit"] & S.a50).sum()), stack_only=int((S["s5hit"] & ~S.a50).sum()),
        adcnn_only=int((~S["s5hit"] & S.a50).sum()), neither=int((~S["s5hit"] & ~S.a50).sum()))}
    out["by_snr@a80"] = {f"{lo}-{hi}": cat(S[(S.snr_t >= lo) & (S.snr_t < hi)]) for lo, hi in SNRB}
    out["by_len@a80"] = {f"{lo}-{hi}px": cat(S[(S.tlen >= lo) & (S.tlen < hi)]) for lo, hi in LENB}
    out["by_band@a80"] = {b: cat(S[S.band == b]) for b in sorted(S.band.unique())}
    out["by_lat@a80"] = {"off-ecl": cat(S[~S.ecl]), "ecliptic": cat(S[S.ecl])}
    # stack-only sightings: where does ADCNN lose them — no detection at all, or sub-0.80 score?
    so = S[S["s5hit"] & ~S.a80]
    out["stack_only_decomp"] = dict(n=len(so), no_adcnn_det=int((~so.a50).sum()),
                                    det_but_subthr=int((so.a50).sum()),
                                    subthr_score_median=float(so.loc[so.a50, "score"].median()) if (so.a50).any() else None)
    # (c) FP taxonomy at S>=0.80
    out["hi_score_dets@a80"] = dict(n_fp=len(FP), n_tp=len(TP),
        fp_len_db_med=float(FP.len_db.median()), tp_len_db_med=float(TP.len_db.median()),
        fp_mf_snr_med=float(FP.mf_snr.median()), tp_mf_snr_med=float(TP.mf_snr.median()),
        fp_art_frac_pos=float((FP.art_frac > 0).mean()), tp_art_frac_pos=float((TP.art_frac > 0).mean()),
        mask_rates_fp={m: round(float((FP[m] > 0).mean()), 3) for m in MASKS if m in FP.columns},
        mask_rates_tp={m: round(float((TP[m] > 0).mean()), 3) for m in MASKS if m in TP.columns})
    json.dump(out, open(f"{HERE}/miss_audit.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("MISS_AUDIT_DONE")


if __name__ == "__main__":
    main()
