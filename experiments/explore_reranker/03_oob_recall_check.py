"""Step 3 — honest cross-checks the head-to-head table cannot show.

(A) Group-disjoint OOB synthetic recall. The baseline metric reads posR
    in-sample (RF trained on the full syn pool, recall measured on it). That
    is the established convention so the head-to-head uses it too — but if a
    model only *looks* better because it overfits the 877 synthetic
    positives, OOB recall on panel-held-out folds will expose it. We compare
    RF_balanced vs HGB_mono_hnm with GroupKFold on the FULL pool (syn panels
    + empty CCDs grouped), reading posR on OOB synthetic positives.

(B) Held-out-empty FP is already fully held out in the main table (the model
    never sees eval CCDs), so no extra check needed there — only recall has
    the in-sample caveat.

This decides whether any GBM Pareto win in step 2 is real or an artifact of
in-sample recall optimism.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from ADCNN.evaluation.fp_analysis import THRS, FEATS  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(HERE))
from harness import score_predict_proba  # noqa: E402
from importlib import import_module  # noqa: E402
zoo = import_module("02_model_zoo")  # reuse mono_vector / balanced_w / hnm


def main():
    A = np.load(HERE / "_arrays.npz")
    X, y, grp = A["X"], A["y"].astype(np.int8), A["grp"]
    # Recover which pool rows are synthetic positives/negatives: synthetic
    # panel ids are < 100000 (empties were offset by +100000 in build_pool).
    # ys is the synthetic recall pool's labels; here we need OOB recall on the
    # *training pool* synthetic positives.
    is_emp = grp >= 100000
    is_syn = ~is_emp
    print(f"[pool] syn rows={int(is_syn.sum())} (pos="
          f"{int(((y == 1) & is_syn).sum())}) emp rows={int(is_emp.sum())}",
          flush=True)

    mono = zoo.mono_vector()
    w = zoo.balanced_w(y)
    gkf = GroupKFold(n_splits=5)

    def oob_recall(make, name):
        oof = np.full(len(y), np.nan)
        t0 = time.time()
        for k, (tr, te) in enumerate(gkf.split(X, y, grp)):
            m = make()
            try:
                m.fit(X[tr], y[tr], sample_weight=w[tr])
            except TypeError:
                m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        synpos = (y == 1) & is_syn
        s = oof[synpos]
        rec = {t: float((s >= t).mean()) for t in THRS}
        # genuine-FP proxy on OOB empty rows (label 0, group>=100000). NB:
        # this is a within-empties OOB proxy, not the held-out 50-CCD number
        # — directional only.
        sneg = oof[is_emp]
        negk = {t: float((sneg >= t).mean()) for t in THRS}
        print(f"\n=== OOB {name}  ({time.time()-t0:.0f}s) ===", flush=True)
        print("  thr | OOB synR | OOB empNegFrac", flush=True)
        for t in THRS:
            print(f" {t:>4.2f} | {rec[t]:>8.3f} | {negk[t]:>13.4f}",
                  flush=True)
        return rec, negk

    oob_recall(lambda: RandomForestClassifier(
        n_estimators=500, max_depth=14, min_samples_leaf=5,
        class_weight="balanced", n_jobs=32, random_state=0),
        "RF_balanced")

    oob_recall(lambda: HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=0.3, early_stopping=False,
        monotonic_cst=mono, random_state=0),
        "HGB_mono")

    print("\n[done]")


if __name__ == "__main__":
    main()
