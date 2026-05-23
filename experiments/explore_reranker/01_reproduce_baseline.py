"""Step 1 — reproduce the promoted FT RandomForest baseline (fp_fix2.txt).

Trains the FT RF via the SAME train_rf_v2 call fp_fix uses on the SAME
panel-disjoint split, then scores the held-out empties and synthetic pool with
the EXACT _fp_gen / _pos_recall formulas (re-expressed on score arrays). If the
table matches fp_fix2.txt's FT columns we have a trustworthy ground truth to
beat. Also stores the held-out-empty + synthetic feature matrices to .npz so
all later model experiments use identical inputs (zero metric drift).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))  # repo root

from harness import (  # noqa: E402
    load_data, build_pool, syn_eval_pool, emp_eval, eval_model_scores,
    score_predict_proba)
from ADCNN.evaluation.fp_analysis import THRS  # noqa: E402
from ADCNN.inference.diffim_postproc_v2 import train_rf_v2  # noqa: E402


def main():
    t0 = time.time()
    syn_f, lab_f, empft, tr_f, ev_f = load_data()
    print(f"[data] syn={syn_f.shape} pos={int(lab_f.sum())} "
          f"neg={int((lab_f == 0).sum())} | empft={empft.shape} "
          f"train_ccd={len(tr_f)} eval_ccd={len(ev_f)}", flush=True)

    X, y, grp, _pool = build_pool(syn_f, lab_f, empft, tr_f)
    print(f"[pool] X={X.shape} pos={int(y.sum())} neg={int((y == 0).sum())} "
          f"groups={len(np.unique(grp))}", flush=True)

    Xs, ys = syn_eval_pool(syn_f, lab_f)
    Xe, fro, n_ccd = emp_eval(empft, ev_f)
    print(f"[eval] syn_pool={Xs.shape} (pos={int((ys == 1).sum())}) "
          f"emp_eval={Xe.shape} n_ccd={n_ccd}", flush=True)

    # --- FT RF trained EXACTLY like fp_fix (train_rf_v2 default hyperparams,
    #     fed comb + comb_lab; it re-derives the same internal pool). ---
    syn_f2 = syn_f.copy()
    emp_ft_tr = empft[empft.image_id.isin(tr_f)].copy()
    emp_ft_tr["panel_id"] = emp_ft_tr["image_id"].to_numpy() + 100000
    from ADCNN.evaluation.fp_analysis import FEATS
    for c in FEATS:
        if c not in emp_ft_tr:
            emp_ft_tr[c] = 0.0
    import pandas as pd
    comb = pd.concat([syn_f2, emp_ft_tr], ignore_index=True, sort=False)
    comb_lab = np.concatenate(
        [lab_f, np.zeros(len(emp_ft_tr), np.int8)]).astype(np.int8)
    t1 = time.time()
    ft_rf = train_rf_v2(comb, labels=comb_lab)
    print(f"[FT RF] trained {time.time() - t1:.0f}s", flush=True)

    s_emp = score_predict_proba(ft_rf, Xe)
    s_syn = score_predict_proba(ft_rf, Xs)
    rows = eval_model_scores(s_emp, fro, n_ccd, s_syn, ys, THRS)

    print("\n  thr |   FPgen/CCD |   posR |   negK   (reproduced FT RF)")
    for t, fp, pr, nk in rows:
        print(f" {t:>4.2f} | {fp:>11.1f} | {pr:>6.3f} | {nk:>6.3f}",
              flush=True)

    # fp_fix2.txt FT columns (the target to beat / sanity baseline)
    ref = {0.05: (68.9, 1.000), 0.10: (37.8, 1.000), 0.20: (18.2, 1.000),
           0.30: (11.7, 1.000), 0.50: (5.6, 0.999), 0.70: (1.4, 0.746)}
    print("\n  thr | repro FP | ref FP | dFP | repro posR | ref posR")
    ok = True
    for t, fp, pr, nk in rows:
        rfp, rpr = ref[t]
        d = fp - rfp
        if abs(d) > max(2.0, 0.06 * rfp) or abs(pr - rpr) > 0.02:
            ok = False
        print(f" {t:>4.2f} | {fp:>8.1f} | {rfp:>6.1f} | {d:>+5.1f} | "
              f"{pr:>10.3f} | {rpr:>7.3f}", flush=True)
    print(f"\n[match fp_fix2.txt FT] {'YES' if ok else 'NO — investigate'}")

    np.savez_compressed(
        HERE / "_arrays.npz",
        X=X, y=y, grp=grp, Xs=Xs, ys=ys, Xe=Xe, fro=fro,
        n_ccd=np.int64(n_ccd),
        base_s_emp=s_emp, base_s_syn=s_syn)
    np.save(HERE / "_base_rows.npy",
            np.array(rows, dtype=np.float64))
    print(f"[saved] {HERE/'_arrays.npz'}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
