"""Shared apples-to-apples harness for the reranker exploration.

Reuses the EXACT established metric definitions so numbers are directly
comparable to fp_fix2.txt (the FT RF baseline we must beat):

  * panel-disjoint split  -> ADCNN.evaluation.fp_analysis._split (seed 0, 2/3)
  * genuine FP/CCD        -> _fp_gen  (score>=t & frac_real_label_overlap<EPS)
  * synthetic pos recall  -> _pos_recall on the (pos | hard-neg<0.5) pool
  * training pool          -> exactly fp_fix's FT branch:
        comb = syn_f (full)  +  emp_ft_tr (train empties, label 0)
        comb_lab = [label_v2 , 0...]
        train_rf_v2 internally drops negs with frac_real_label_overlap>=0.5

Anything that swaps ONLY the classifier (keeping pool/split/metric fixed) is a
fair head-to-head vs the promoted FT RandomForest.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ADCNN.evaluation.fp_analysis import (
    FEATS, EPS_GENUINE, THRS, _split, _dedup)
from ADCNN.inference.diffim_postproc_v2 import train_rf_v2

RES = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/"
           "experiments/diffim_runs/test_real/results")
SYN_FT_PKL = RES / "syn5_ft.pkl"
EMPFT_GLOB = str(RES / "parts" / "empft_*.csv")

INFO_MAX = 0.5  # train_rf_v2's informational_fp_max_overlap default


def load_data():
    """Return (syn_f, lab_f, empft, tr_ids, ev_ids) — identical objects to
    what fp_fix's FT branch operates on."""
    syn_f = _dedup(pd.read_pickle(SYN_FT_PKL))
    lab_f = syn_f["label_v2"].to_numpy(np.int8)
    empft = _dedup(pd.concat(
        [pd.read_csv(f) for f in sorted(glob.glob(EMPFT_GLOB))],
        ignore_index=True))
    tr_f, ev_f = _split(empft)            # seed-0 2/3 panel-disjoint split
    return syn_f, lab_f, empft, tr_f, ev_f


def build_pool(syn_f, lab_f, empft, tr_f):
    """Replicate fp_fix FT training pool, then apply train_rf_v2's internal
    informational-FP filter so X/y are the EXACT rows the RF baseline trains
    on. Returns X (float32, FEATS order), y (int8), and a `groups` array
    (panel-disjoint group id) for CV / hard-neg folding.
    """
    emp_ft_tr = empft[empft.image_id.isin(tr_f)].copy()
    emp_ft_tr["panel_id"] = emp_ft_tr["image_id"].to_numpy() + 100000
    for c in FEATS:
        if c not in emp_ft_tr:
            emp_ft_tr[c] = 0.0
    comb = pd.concat([syn_f, emp_ft_tr], ignore_index=True, sort=False)
    comb_lab = np.concatenate(
        [lab_f, np.zeros(len(emp_ft_tr), np.int8)]).astype(np.int8)
    # train_rf_v2 internal pool: pos | (neg & frac_real_label_overlap<0.5)
    fro = comb["frac_real_label_overlap"].to_numpy()
    fp_mask = (comb_lab == 0) & (fro < INFO_MAX)
    pool_mask = (comb_lab == 1) | fp_mask
    X = comb.loc[pool_mask, FEATS].fillna(0.0).to_numpy(np.float32)
    y = comb_lab[pool_mask]
    # group id: synthetic uses panel_id (0..), empties use image_id+100000.
    grp = comb.loc[pool_mask, "panel_id"].to_numpy()
    return X, y, grp, comb.loc[pool_mask].reset_index(drop=True)


def syn_eval_pool(syn_f, lab_f):
    """The recall pool used by _pos_recall: pos | (neg & fro<0.5)."""
    fro = syn_f["frac_real_label_overlap"].to_numpy()
    pool = (lab_f == 1) | ((lab_f == 0) & (fro < INFO_MAX))
    Xs = syn_f.loc[pool, FEATS].fillna(0.0).to_numpy(np.float32)
    ys = lab_f[pool]
    return Xs, ys


def emp_eval(empft, ev_f):
    """Held-out empty eval frame: features (inf->nan->0 as in _fp_gen) and the
    genuine-FP mask source (frac_real_label_overlap)."""
    ev = empft[empft.image_id.isin(ev_f)]
    Xe = (ev[FEATS].replace([np.inf, -np.inf], np.nan)
          .fillna(0.0).to_numpy(np.float32))
    fro = ev["frac_real_label_overlap"].to_numpy()
    n_ccd = ev.image_id.nunique()
    return Xe, fro, n_ccd


def fp_gen_from_scores(s_emp, fro, n_ccd, t):
    return int(((s_emp >= t) & (fro < EPS_GENUINE)).sum()) / n_ccd


def pos_recall_from_scores(s_syn, ys, t):
    return float((s_syn[ys == 1] >= t).mean())


def neg_kept_from_scores(s_syn, ys, t):
    return float((s_syn[ys == 0] >= t).mean())


def score_predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def eval_model_scores(s_emp, fro, n_ccd, s_syn, ys, thrs=THRS):
    """Return list of (t, fp_gen_per_ccd, posR, negK)."""
    out = []
    for t in thrs:
        out.append((
            t,
            fp_gen_from_scores(s_emp, fro, n_ccd, t),
            pos_recall_from_scores(s_syn, ys, t),
            neg_kept_from_scores(s_syn, ys, t),
        ))
    return out
