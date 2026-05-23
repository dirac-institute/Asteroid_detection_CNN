"""False-positive analysis & RF-hardening for the real-data second stage.

Subcommands (``python -m ADCNN.evaluation.fp_analysis <cmd> ...``):

  dump-empty   GPU. V2 candidate features for the real EMPTY CCDs of
               test_real with a given model (real-residual hard negatives +
               FP fact-correction via ``frac_real_label_overlap``). Sharded.
  dump-syn     GPU. Recompute synthetic test_5sigma V2 features + objectwise
               labels with a given model (RF positives/recall, model-
               consistent).
  snr-gain     CPU. Detection-independent matched-filter SNR per real
               sighting -> where (in SNR) a 2nd stage can help vs the stack.
  sweep        GPU. Per-candidate (score, on_truth) dump over empty +
               stack-missed panels. ``sweep-curve`` aggregates -> the
               recovery-vs-FP curve across RF thresholds.
  fp-fix       CPU. Fact-correct FP (all vs genuine) and retrain the V2 RF
               with real empty-CCD hard negatives; ORIGINAL vs (optionally
               fine-tuned) side-by-side, panel-disjoint, recall-guarded.
  bar          GPU. Synthetic-objectwise bar (combined-TP/FP on test_5sigma)
               for a (model, rf) — gate for promoting fine-tuned artifacts.

All heavy steps are sharded for SLURM arrays; CPU steps run anywhere.
"""
from __future__ import annotations

import argparse
import glob
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from ADCNN.inference.predict import predict_panel_overlap_3ch_full
from ADCNN.inference.diffim_postproc_v2 import (
    DEFAULT_THR, RF_FEATURES_V2, apply_rf_v2, compute_v2_features, load_rf,
    materialize_label_mask_v2, train_rf_v2,
    label_candidates_by_injection_overlap)
from ADCNN.inference.diffim_matched_filter import panel_mad_sigma
from ADCNN.utils.helpers import draw_one_line

FEATS = list(RF_FEATURES_V2)
EPS_GENUINE = 0.05  # frac_real_label_overlap below this = genuine new FP
THRS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dedup(df):
    """Drop duplicate columns (KEEP lists re-list cols already in
    RF_FEATURES_V2 e.g. frac_real_label_overlap, max_p)."""
    return df.loc[:, ~df.columns.duplicated()].copy()


# ======================================================================
# dump-empty / dump-syn  (GPU feature dumps)
# ======================================================================
_KEEP_EMP = FEATS + ["panel_id", "candidate_id", "max_p",
                     "y_min", "y_max", "x_min", "x_max"]


def dump_empty(data_dir, model_path, rf_path, *, shard=0, nshards=1,
               tag="emp", results_dir=None):
    data = Path(data_dir)
    res = Path(results_dir) if results_dir else data / "results"
    (res / "parts").mkdir(parents=True, exist_ok=True)
    dev = _dev()
    model = torch.jit.load(str(model_path), map_location=dev).eval()
    rf = load_rf(rf_path)
    panels = pd.read_csv(data / "panels.csv")
    emp = sorted(int(i) for i in
                 panels.loc[panels.role == "empty", "image_id"].unique())
    my = emp[shard::nshards] if nshards > 1 else emp
    print(f"[dump-empty s{shard}/{nshards}] {len(my)}/{len(emp)} CCDs "
          f"tag={tag}", flush=True)
    chunks, t0 = [], time.time()
    with h5py.File(data / "test.h5", "r") as f:
        for k, idx in enumerate(my):
            try:
                img = f["images"][idx][:].astype(np.float32)
                rl = f["real_labels"][idx][:]
                prob, sn, cs, ag = predict_panel_overlap_3ch_full(
                    model, img, rl, device=dev)
                cand, _ = compute_v2_features(
                    {0: prob.astype(np.float32)}, {0: img},
                    {0: sn.astype(np.float32)}, {0: cs.astype(np.float32)},
                    {0: ag.astype(np.float32)}, real_labels={0: rl},
                    verbose=False)
                if not len(cand):
                    continue
                cand[FEATS] = cand[FEATS].replace([np.inf, -np.inf], np.nan)
                cand = apply_rf_v2(cand, rf)
                cand["image_id"] = idx
                chunks.append(cand[[c for c in _KEEP_EMP if c in cand.columns]
                                   + ["score_rf", "image_id"]])
            except Exception as e:
                print(f"[s{shard} FAIL idx={idx}] {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
            if k % 10 == 0 or k == len(my) - 1:
                el = time.time() - t0
                print(f"[s{shard} {k+1}/{len(my)}] {el:.0f}s "
                      f"rows={sum(len(c) for c in chunks)}", flush=True)
            if k % 10 == 0 and chunks:
                pd.concat(chunks, ignore_index=True).to_csv(
                    res / "parts" / f"{tag}_{shard}.csv", index=False)
    if chunks:
        pd.concat(chunks, ignore_index=True).to_csv(
            res / "parts" / f"{tag}_{shard}.csv", index=False)
    print(f"[dump-empty s{shard}] done "
          f"rows={sum(len(c) for c in chunks)}", flush=True)


def dump_syn(syn_dir, model_path, *, tag, results_dir):
    """Recompute synthetic V2 features + objectwise labels with ``model``."""
    d = Path(syn_dir)
    res = Path(results_dir)
    res.mkdir(parents=True, exist_ok=True)
    dev = _dev()
    model = torch.jit.load(str(model_path), map_location=dev).eval()
    csv = pd.read_csv(d / "test.csv")
    pp, dif, sd, cd, ad, rd = {}, {}, {}, {}, {}, {}
    t0 = time.time()
    with h5py.File(d / "test.h5", "r") as f:
        N = int(f["images"].shape[0])
        for i in range(N):
            img = f["images"][i][:].astype(np.float32)
            rl = f["real_labels"][i][:]
            prob, sn, cs, ag = predict_panel_overlap_3ch_full(
                model, img, rl, device=dev)
            pp[i] = prob.astype(np.float32); dif[i] = img
            sd[i] = sn.astype(np.float32); cd[i] = cs.astype(np.float32)
            ad[i] = ag.astype(np.float32); rd[i] = rl
            if i % 10 == 0 or i == N - 1:
                print(f"[dump-syn {i+1}/{N}] "
                      f"{time.time()-t0:.0f}s", flush=True)
    cand, _ = compute_v2_features(pp, dif, sd, cd, ad, real_labels=rd,
                                  verbose=True)
    lab = np.asarray(label_candidates_by_injection_overlap(cand, csv, pp),
                     np.int8)
    cand["label_v2"] = lab
    keep = FEATS + ["panel_id", "candidate_id",
                    "frac_real_label_overlap", "label_v2"]
    _dedup(cand[[c for c in keep if c in cand.columns]]).to_pickle(
        res / f"syn5_{tag}.pkl")
    # Also persist the panel-probs stack so fp-fix's legacy --syn-pp-npy
    # path is reproducible from a tracked step (the pkl already carries
    # label_v2, so fp-fix does not actually need this; kept for parity).
    np.save(res / f"syn5_{tag}_pp.npy",
            np.stack([pp[i] for i in range(N)]).astype(np.float32))
    print(f"[dump-syn] pos={int(lab.sum())} neg={int((lab==0).sum())} -> "
          f"{res}/syn5_{tag}.pkl (+ syn5_{tag}_pp.npy)", flush=True)


# ======================================================================
# snr-gain  (CPU; where are the 2nd-stage gains in SNR)
# ======================================================================
def _snr_panel(args):
    h5p, idx, rows = args
    with h5py.File(h5p, "r") as f:
        im = f["images"][idx][:].astype(np.float32)
    H, W = im.shape
    sig = panel_mad_sigma(im)
    out = []
    for r in rows:
        m = np.zeros((H, W), np.uint8)
        draw_one_line(m, [float(r["x"]), float(r["y"])], float(r["beta"]),
                      float(r["trail_length"]), true_value=1, line_thickness=2)
        mb = m > 0
        n = int(mb.sum())
        snr = (float(im[mb].sum() / (sig * np.sqrt(n)))
               if n and sig > 0 else np.nan)
        out.append((r["ObjID"], idx, snr))
    return out


def snr_gain(data_dir, results_dir=None, workers=32):
    from concurrent.futures import ProcessPoolExecutor
    data = Path(data_dir)
    res = Path(results_dir) if results_dir else data / "results"
    cat = pd.read_csv(data / "test.csv")
    sight = pd.read_csv(res / "per_sighting.csv")
    h5p = str(data / "test.h5")
    tasks = [(h5p, int(i), g[["x", "y", "beta", "trail_length", "ObjID"]]
              .to_dict("records")) for i, g in cat.groupby("image_id")]
    print(f"[snr-gain] {len(cat)} sightings / {len(tasks)} panels", flush=True)
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_snr_panel, tasks, chunksize=4):
            recs.extend(r)
    snr = pd.DataFrame(recs, columns=["ObjID", "image_id", "mf_snr"])
    df = sight.merge(snr, on=["ObjID", "image_id"], how="left")
    df.to_csv(res / "per_sighting_snr.csv", index=False)
    bins = [-1e9, 3, 5, 7, 10, 15, 25, 1e9]
    labs = ["<3", "3-5", "5-7", "7-10", "10-15", "15-25", ">25"]
    df["snrbin"] = pd.cut(df["mf_snr"], bins=bins, labels=labs)
    g = df.groupby("snrbin", observed=True).agg(
        n=("mf_snr", "size"), stack_rec=("stack_detected", "mean"),
        nn_rec=("nn_detected", "mean"))
    gm = df[~df.stack_detected].groupby("snrbin", observed=True).agg(
        miss_n=("mf_snr", "size"),
        nn_recovers_missed=("nn_detected", "mean"))
    rep = ("=" * 70 + "\n  WHERE ARE THE 2nd-STAGE GAINS? (MF-SNR of trail)\n"
           + "=" * 70 + "\n" + g.join(gm).round(3).to_string())
    print("\n" + rep)
    (res / "snr_gain.txt").write_text(rep + "\n")
    print(f"\n[saved] {res}/snr_gain.txt")


# ======================================================================
# sweep / sweep-curve  (recovery-vs-FP curve across RF thresholds)
# ======================================================================
def sweep(data_dir, model_path, rf_path, *, shard=0, nshards=1, smin=0.02,
          results_dir=None):
    data = Path(data_dir)
    res = Path(results_dir) if results_dir else data / "results"
    (res / "parts").mkdir(parents=True, exist_ok=True)
    dev = _dev()
    model = torch.jit.load(str(model_path), map_location=dev).eval()
    rf = load_rf(rf_path)
    cat = pd.read_csv(data / "test.csv")
    panels = pd.read_csv(data / "panels.csv")
    miss = cat[~cat["stack_detection"].astype(bool)]
    miss_by_img = {int(i): g for i, g in miss.groupby("image_id")}
    sel = panels[(panels.role == "empty")
                 | (panels.image_id.isin(miss_by_img))]
    ids = sorted(int(i) for i in sel.image_id.unique())
    my = ids[shard::nshards] if nshards > 1 else ids
    print(f"[sweep s{shard}/{nshards}] {len(my)}/{len(ids)} panels",
          flush=True)
    rows, t0 = [], time.time()
    with h5py.File(data / "test.h5", "r") as f:
        for k, idx in enumerate(my):
            prow = panels[panels.image_id == idx].iloc[0]
            role = prow.role
            try:
                img = f["images"][idx][:].astype(np.float32)
                rl = f["real_labels"][idx][:]
                prob, sn, cs, ag = predict_panel_overlap_3ch_full(
                    model, img, rl, device=dev)
                cand, _ = compute_v2_features(
                    {0: prob.astype(np.float32)}, {0: img},
                    {0: sn.astype(np.float32)}, {0: cs.astype(np.float32)},
                    {0: ag.astype(np.float32)}, real_labels={0: rl},
                    verbose=False)
                if not len(cand):
                    continue
                cand[FEATS] = cand[FEATS].replace([np.inf, -np.inf], np.nan)
                cand = apply_rf_v2(cand, rf)
                cand = cand[cand.score_rf >= smin]
                if not len(cand):
                    continue
                H, W = prob.shape
                tm = np.zeros((H, W), np.uint8)
                objids = []
                if role == "asteroid" and idx in miss_by_img:
                    for j, r in enumerate(miss_by_img[idx].itertuples(), 1):
                        draw_one_line(tm, [float(r.x), float(r.y)],
                                      float(r.beta), float(r.trail_length),
                                      true_value=j, line_thickness=20)
                        objids.append(r.ObjID)
                for cc in cand.itertuples():
                    on = 0
                    if tm.max() > 0:
                        sub = tm[int(cc.y_min):int(cc.y_max) + 1,
                                 int(cc.x_min):int(cc.x_max) + 1]
                        if sub.any():
                            on = int(sub.max())
                    rows.append((idx, role, float(cc.score_rf), on,
                                 objids[on - 1] if on else ""))
            except Exception as e:
                print(f"[s{shard} FAIL idx={idx}] {e}", flush=True)
                traceback.print_exc()
            if k % 25 == 0 or k == len(my) - 1:
                print(f"[s{shard} {k+1}/{len(my)}] "
                      f"{time.time()-t0:.0f}s rows={len(rows)}", flush=True)
            if k % 50 == 0 and rows:
                pd.DataFrame(rows, columns=["image_id", "role", "score_rf",
                             "on_truth", "ObjID"]).to_csv(
                    res / "parts" / f"cand_{shard}.csv", index=False)
    pd.DataFrame(rows, columns=["image_id", "role", "score_rf", "on_truth",
                 "ObjID"]).to_csv(res / "parts" / f"cand_{shard}.csv",
                                  index=False)
    print(f"[sweep s{shard}] done rows={len(rows)}", flush=True)


def sweep_curve(data_dir, results_dir=None):
    data = Path(data_dir)
    res = Path(results_dir) if results_dir else data / "results"
    cand = pd.concat([pd.read_csv(f) for f in
                      sorted(glob.glob(str(res / "parts" / "cand_*.csv")))],
                     ignore_index=True)
    sight = pd.read_csv(res / "per_sighting.csv")
    emp = cand[cand.role == "empty"]
    n_emp = emp.image_id.nunique()
    ast = cand[(cand.role == "asteroid") & (cand.on_truth > 0)]
    miss = sight[~sight.stack_detected]
    g = sight.groupby("ObjID").agg(st=("stack_detected", "any"))
    never = set(g[~g.st].index)
    best_obj = ast.groupby("ObjID").score_rf.max()
    sb = ast.groupby("image_id").score_rf.max()
    miss = miss.assign(best=miss.image_id.map(sb).fillna(-1.0))
    grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]
    lines = [f"empty panels={n_emp} | stack-missed sightings={len(miss)} | "
             f"objects stack NEVER got={len(never)}",
             f"{'thr':>5} {'FP/empty':>9} {'missSight_rec':>14} {'NEW_obj':>9}",
             "-" * 42]
    for t in grid:
        fp = (emp.score_rf >= t).sum() / max(n_emp, 1)
        rs = int((miss.best >= t).sum())
        no = sum(1 for o in never if best_obj.get(o, -1) >= t)
        lines.append(f"{t:>5.2f} {fp:>9.1f} "
                     f"{rs:>5}/{len(miss)} ({100*rs/max(len(miss),1):>4.1f}%)"
                     f" {no:>4}/{len(never)}")
    rep = "\n".join(lines)
    print(rep)
    (res / "threshold_sweep.txt").write_text(rep + "\n")
    print(f"[saved] {res}/threshold_sweep.txt")


# ======================================================================
# fp-fix  (fact-correct + retrain RF with real hard negatives;
#          ORIGINAL vs optional FINE-TUNED, panel-disjoint)
# ======================================================================
def _split(emp, frac=2 / 3):
    ids = np.array(sorted(emp.image_id.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    cut = int(len(ids) * frac)
    return set(ids[:cut]), set(ids[cut:])


def _fp_gen(df, rf, n, t):
    X = df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
        np.float32)
    s = rf.predict_proba(X)[:, 1]
    return int(((s >= t) & (df["frac_real_label_overlap"].to_numpy()
                            < EPS_GENUINE)).sum()) / n


def _pos_recall(df, lab, rf, t):
    s = rf.predict_proba(df[FEATS].fillna(0.0).to_numpy(np.float32))[:, 1]
    return (s[lab == 1] >= t).mean(), (s[lab == 0] >= t).mean()


def fp_fix(results_dir, syn_cached_pkl, syn_pp_npy, syn_csv, *,
           old_rf_path, ft_syn_pkl=None, old_tag="emp", ft_tag="empft",
           ckpt_out=None):
    """Fact-correct FP and retrain the RF with real empty-CCD hard negatives.

    ``old_*`` = original-model synthetic artifacts. Two ways to supply them:

    * **reproducible (preferred):** ``--syn-cached-pkl`` points at a
      ``dump-syn``-produced ``syn5_<tag>.pkl`` (ORIGINAL model). It already
      carries an objectwise ``label_v2`` column (computed by the *same*
      ``label_candidates_by_injection_overlap`` definition fp-fix would use),
      so ``--syn-pp-npy`` is not needed.
    * **legacy:** a cached scored-cand pkl *without* ``label_v2`` — then
      ``--syn-pp-npy`` (original panel_probs) is required to recompute the
      labels. Numerically identical to the first path.

    If ``ft_syn_pkl`` and fine-tuned empty parts (``{ft_tag}_*.csv``) are
    present, also produces the ORIGINAL-vs-FINE-TUNED table and a retrained
    ``rf_postproc_v2_ft.pkl``. The retrained (promoted) RF is built solely
    from ``ft_syn_pkl`` + the ``{ft_tag}_*.csv`` empties — it does NOT depend
    on the original-model artifacts, which only feed the comparison columns.
    """
    import joblib
    res = Path(results_dir)
    log = []
    P = lambda *a: (print(*a, flush=True),
                    log.append(" ".join(map(str, a))))

    emp_o = _dedup(pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        str(res / "parts" / f"{old_tag}_*.csv")))], ignore_index=True))
    tr_o, ev_o = _split(emp_o)
    emp_o_ev = emp_o[emp_o.image_id.isin(ev_o)]
    old_rf = joblib.load(old_rf_path)
    syn_o = _dedup(pd.read_pickle(syn_cached_pkl))
    csv5 = pd.read_csv(syn_csv)
    if "label_v2" in syn_o.columns:
        # dump-syn already computed objectwise labels with the ORIGINAL model
        # (identical label_candidates_by_injection_overlap definition) — fully
        # reproducible, no legacy panel_probs npy needed.
        lab_o = syn_o["label_v2"].to_numpy(np.int8)
        P(f"[fp-fix] using label_v2 from {Path(syn_cached_pkl).name} "
          f"(reproducible path)")
    else:
        if not syn_pp_npy or not Path(syn_pp_npy).exists():
            raise SystemExit(
                "fp-fix: --syn-cached-pkl has no 'label_v2' column; the "
                "legacy path needs an existing --syn-pp-npy. Prefer pointing "
                "--syn-cached-pkl at a dump-syn syn5_<tag>.pkl instead.")
        pp_o = np.load(syn_pp_npy)
        lab_o = np.asarray(label_candidates_by_injection_overlap(
            syn_o, csv5, pp_o), np.int8)
        P(f"[fp-fix] recomputed labels from legacy {Path(syn_pp_npy).name}")
    pool_o = (lab_o == 1) | ((lab_o == 0) & (
        syn_o["frac_real_label_overlap"].to_numpy() < 0.5))
    syn_o_p, lab_o_p = syn_o[pool_o], lab_o[pool_o]

    P("==== FP/CCD on HELDOUT real empties — ORIGINAL rf ====")
    P(f"{'thr':>5} {'FP_genuine/CCD':>15}")
    for t in THRS:
        P(f"{t:>5.2f} {_fp_gen(emp_o_ev, old_rf, len(ev_o), t):>15.1f}")
    P("(FP_genuine = NN comp NOT on a stack residual the stack already "
      "flags = what a deployed 2nd stage truly adds)")

    ft_parts = sorted(glob.glob(str(res / "parts" / f"{ft_tag}_*.csv")))
    if ft_syn_pkl and Path(ft_syn_pkl).exists() and ft_parts:
        empft = _dedup(pd.concat([pd.read_csv(f) for f in ft_parts],
                                 ignore_index=True))
        tr_f, ev_f = _split(empft)
        emp_ft_tr = empft[empft.image_id.isin(tr_f)].copy()
        emp_ft_ev = empft[empft.image_id.isin(ev_f)]
        syn_f = _dedup(pd.read_pickle(ft_syn_pkl))
        lab_f = syn_f["label_v2"].to_numpy(np.int8)
        emp_ft_tr["panel_id"] = emp_ft_tr["image_id"].to_numpy() + 100000
        for c in FEATS:
            if c not in emp_ft_tr:
                emp_ft_tr[c] = 0.0
        comb = pd.concat([syn_f, emp_ft_tr], ignore_index=True, sort=False)
        comb_lab = np.concatenate([lab_f, np.zeros(len(emp_ft_tr), np.int8)])
        P(f"\n[FT] syn pos={int(lab_f.sum())} + {len(emp_ft_tr)} real-empty "
          f"neg; training RF ...")
        ft_rf = train_rf_v2(comb, labels=comb_lab)
        if ckpt_out:
            joblib.dump(ft_rf, ckpt_out)
            P(f"[FT] saved {ckpt_out}")
        pool_f = (lab_f == 1) | ((lab_f == 0) & (
            syn_f["frac_real_label_overlap"].to_numpy() < 0.5))
        syn_f_p, lab_f_p = syn_f[pool_f], lab_f[pool_f]
        P("\n================ ORIGINAL vs FINE-TUNED ================")
        P(f"{'thr':>5} | {'OLD FPgen':>9} {'FT FPgen':>9} | "
          f"{'OLD posR':>8} {'FT posR':>8}")
        for t in THRS:
            ofp = _fp_gen(emp_o_ev, old_rf, len(ev_o), t)
            ffp = _fp_gen(emp_ft_ev, ft_rf, len(ev_f), t)
            opr, _ = _pos_recall(syn_o_p, lab_o_p, old_rf, t)
            fpr, _ = _pos_recall(syn_f_p, lab_f_p, ft_rf, t)
            P(f"{t:>5.2f} | {ofp:>9.1f} {ffp:>9.1f} | "
              f"{opr:>8.3f} {fpr:>8.3f}")
        P("posR = synthetic true-trail recall (must stay ~1.0). "
          "WIN = FT FPgen << OLD at matched posR.")
    (res / "fp_fix.txt").write_text("\n".join(log) + "\n")
    print(f"\n[saved] {res}/fp_fix.txt")


# ======================================================================
# bar  (synthetic objectwise bar gate)
# ======================================================================
def bar(syn_root, model_path, rf_path, tag, *, splits=("test_5sigma",),
        results_dir):
    from ADCNN.evaluation.detection import (
        objectwise_confusion, combined_objectwise_confusion_separate)
    res = Path(results_dir)
    res.mkdir(parents=True, exist_ok=True)
    dev = _dev()
    model = torch.jit.load(str(model_path), map_location=dev).eval()
    rf = load_rf(rf_path)
    thrs = [0.10, 0.20, 0.50]
    lines = [f"=== BAR {tag} === model={Path(model_path).name} "
             f"rf={Path(rf_path).name}",
             f"{'split':>12} {'thr':>5} | {'NN_TP':>6} {'NN_FP':>7} "
             f"{'NN_FN':>6} | {'cTP':>5} {'cFP':>7} {'cFN':>5} {'nObj':>5}"]
    for sp in splits:
        d = Path(syn_root) / sp
        cat = pd.read_csv(d / "test.csv")
        with h5py.File(d / "test.h5", "r") as f:
            N, H, W = f["images"].shape
            per = {t: np.zeros((N, H, W), np.int32) for t in thrs}
            sfp = np.zeros((N, H, W), np.uint16)
            t0 = time.time()
            for i in range(N):
                img = f["images"][i][:].astype(np.float32)
                rl = f["real_labels"][i][:]
                sfp[i] = rl
                prob, sn, cs, ag = predict_panel_overlap_3ch_full(
                    model, img, rl, device=dev)
                cand, ppd = compute_v2_features(
                    {0: prob.astype(np.float32)}, {0: img},
                    {0: sn.astype(np.float32)}, {0: cs.astype(np.float32)},
                    {0: ag.astype(np.float32)}, real_labels={0: rl},
                    verbose=False)
                if len(cand):
                    cand[FEATS] = cand[FEATS].replace([np.inf, -np.inf],
                                                      np.nan)
                    cand = apply_rf_v2(cand, rf)
                for t in thrs:
                    kept = cand[cand.score_rf >= t] if len(cand) else cand
                    per[t][i] = materialize_label_mask_v2(
                        kept, ppd, (1, H, W))[0]
                if i % 10 == 0 or i == N - 1:
                    print(f"  [{sp} {i+1}/{N}] "
                          f"{time.time()-t0:.0f}s", flush=True)
        for t in thrs:
            otp, ofp, ofn, _ = objectwise_confusion(
                cat, per[t], 0.5, stack_fp=sfp, use_threads=False)
            ctp, cfp, cfn, _ = combined_objectwise_confusion_separate(
                cat, per[t], 0.5, stack_mask=sfp)
            lines.append(f"{sp:>12} {t:>5.2f} | {otp:>6} {ofp:>7} {ofn:>6} | "
                         f"{ctp:>5} {cfp:>7} {cfn:>5} {len(cat):>5}")
    rep = "\n".join(lines)
    print("\n" + rep)
    (res / f"bar_{tag}.txt").write_text(rep + "\n")
    print(f"\n[saved] {res}/bar_{tag}.txt")


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(
        "fp_analysis", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    de = sub.add_parser("dump-empty")
    de.add_argument("--data", required=True)
    de.add_argument("--model", required=True)
    de.add_argument("--rf", required=True)
    de.add_argument("--shard", type=int, default=0)
    de.add_argument("--nshards", type=int, default=1)
    de.add_argument("--tag", default="emp")
    de.add_argument("--results-dir", default=None)

    ds = sub.add_parser("dump-syn")
    ds.add_argument("--syn-dir", required=True)
    ds.add_argument("--model", required=True)
    ds.add_argument("--tag", required=True)
    ds.add_argument("--results-dir", required=True)

    sg = sub.add_parser("snr-gain")
    sg.add_argument("--data", required=True)
    sg.add_argument("--results-dir", default=None)
    sg.add_argument("--workers", type=int, default=32)

    sw = sub.add_parser("sweep")
    sw.add_argument("--data", required=True)
    sw.add_argument("--model", required=True)
    sw.add_argument("--rf", required=True)
    sw.add_argument("--shard", type=int, default=0)
    sw.add_argument("--nshards", type=int, default=1)
    sw.add_argument("--results-dir", default=None)

    swc = sub.add_parser("sweep-curve")
    swc.add_argument("--data", required=True)
    swc.add_argument("--results-dir", default=None)

    ff = sub.add_parser("fp-fix")
    ff.add_argument("--results-dir", required=True)
    ff.add_argument("--syn-cached-pkl", required=True,
                    help="ORIGINAL-model synthetic cand pkl; a dump-syn "
                         "syn5_<tag>.pkl (has label_v2) is reproducible")
    ff.add_argument("--syn-pp-npy", default=None,
                    help="legacy: original panel_probs npy, only needed when "
                         "--syn-cached-pkl lacks a label_v2 column")
    ff.add_argument("--syn-csv", required=True)
    ff.add_argument("--old-rf", required=True)
    ff.add_argument("--ft-syn-pkl", default=None)
    ff.add_argument("--old-tag", default="emp")
    ff.add_argument("--ft-tag", default="empft")
    ff.add_argument("--ckpt-out", default=None)

    br = sub.add_parser("bar")
    br.add_argument("--syn-root", required=True)
    br.add_argument("--model", required=True)
    br.add_argument("--rf", required=True)
    br.add_argument("--tag", required=True)
    br.add_argument("--splits", nargs="+", default=["test_5sigma"])
    br.add_argument("--results-dir", required=True)

    a = ap.parse_args()
    if a.cmd == "dump-empty":
        dump_empty(a.data, a.model, a.rf, shard=a.shard, nshards=a.nshards,
                   tag=a.tag, results_dir=a.results_dir)
    elif a.cmd == "dump-syn":
        dump_syn(a.syn_dir, a.model, tag=a.tag, results_dir=a.results_dir)
    elif a.cmd == "snr-gain":
        snr_gain(a.data, results_dir=a.results_dir, workers=a.workers)
    elif a.cmd == "sweep":
        sweep(a.data, a.model, a.rf, shard=a.shard, nshards=a.nshards,
              results_dir=a.results_dir)
    elif a.cmd == "sweep-curve":
        sweep_curve(a.data, results_dir=a.results_dir)
    elif a.cmd == "fp-fix":
        fp_fix(a.results_dir, a.syn_cached_pkl, a.syn_pp_npy, a.syn_csv,
               old_rf_path=a.old_rf, ft_syn_pkl=a.ft_syn_pkl,
               old_tag=a.old_tag, ft_tag=a.ft_tag, ckpt_out=a.ckpt_out)
    elif a.cmd == "bar":
        bar(a.syn_root, a.model, a.rf, a.tag, splits=tuple(a.splits),
            results_dir=a.results_dir)


if __name__ == "__main__":
    main()
