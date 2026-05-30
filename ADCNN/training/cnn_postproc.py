"""Train the stage-2 false-positive filter — the focal-loss cutout CNN.

Pipeline stage 2: the segmentation model emits many candidate components per panel (asteroid
trails + residual/artefact false positives). The cutout CNN scores a ``CUTOUT_K x CUTOUT_K x 3``
patch ``[diffim/sigma, seg_prob, seg_agg]`` per candidate and rejects FP while keeping trails.
Replaces the legacy 72-feature RandomForest.

Training is leakage-safe: cutouts are built by running the trained segmentation model on
held-out panels (never the test set); candidates are labelled by overlap with the injected
truth (:func:`ADCNN.inference.features.label_candidates_by_injection_overlap`); a focal CNN is
fit; the operating threshold is set by the COMBINED 5sigma+ADCNN FP-budget on val2 -- the
deployed system reports the deduplicated UNION of (5sigma-stack) ∪ (ADCNN), and the cut here
is the score at which that union's FP/panel == ``FPP_BUDGET``.

Default recipe (matches the deployed checkpoint, learned via the overnight filter-v2
investigation):
  - architecture: ``ADCNN.inference.cnn_postproc.build_net`` at width=40, depth=4, k=96.
  - 60 epochs, focal loss, cosine LR with 3-ep warmup, random rot/flip augmentation.
  - DataParallel across all visible GPUs for the train step.
  - Best-val-AUC checkpoint saved (NOT the last epoch — that one overfits and roughly doubles
    the FP/panel at the same recall).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ADCNN.inference.cnn_postproc import build_net, make_cutouts, CUTOUT_K, NET_WIDTH, NET_DEPTH

# ---- training recipe defaults (deployed) ----
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
HOLDOUT_FRAC = 0.10        # panel-disjoint holdout used for the diagnostic AUC
FP_CAP = 600               # max false-positive cutouts kept per panel (class balance)

# ---- operating point (the COMBINED detector) ----
# ADCNN complements the classical 5sigma stack: the deployed system reports the DEDUPLICATED
# UNION of (5sigma-stack) ∪ (ADCNN). FPP_BUDGET is the false-positives-per-panel that union is
# allowed; ``calibrate_combined_threshold`` sets ADCNN's score cut on val2 so the union hits it.
# val2 must carry ``real_labels_5sigma`` + ``stack_detection_5sigma`` -- build it with
# ``make_sim_data --multi-sigma-sets val2 --test-sigmas 5``.
FPP_BUDGET = 100.0
DEDUP_TOL_PX = 20.0
STACK_SIGMA = 5


# ---------------------------------------------------------------------------
# Cutout extraction (one panel)
# ---------------------------------------------------------------------------
def panel_cutouts(model, img, rl, panel_cat, *, pid: int = 0, k: int = CUTOUT_K,
                  fp_cap: int = 0, device="cuda"):
    """Run the seg model on one panel, extract candidates (lean), label by injection overlap,
    and return the per-candidate cutout stack. ``panel_cat`` is the truth catalog rows for this
    panel with ``image_id == pid``. Returns ``(X[N,3,k,k], y[N], xy[N,2], cids[N])``; empty
    arrays if none."""
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import extract_panel_candidates, label_candidates_by_injection_overlap

    prob, _sin, _cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=device)
    prob = prob.astype(np.float32); agg = np.asarray(agg, np.float32)
    cand, _ = extract_panel_candidates({pid: prob}, {pid: img}, real_labels={pid: rl.astype(np.uint16)})
    if not len(cand):
        z = np.zeros((0,), np.float32)
        return (np.zeros((0, 3, k, k), np.float32), z.astype(np.int8),
                np.zeros((0, 2), np.float32), z.astype(np.int32))
    lab = label_candidates_by_injection_overlap(cand, panel_cat, {pid: prob})
    keep = np.ones(len(cand), bool)
    if fp_cap > 0:
        fp_i = np.where(lab == 0)[0]
        if len(fp_i) > fp_cap:   # subsample FP (seeded by pid for reproducibility)
            drop = np.random.default_rng(pid).choice(fp_i, len(fp_i) - fp_cap, replace=False)
            keep[drop] = False
    cand = cand[keep].reset_index(drop=True); lab = lab[keep]
    X = make_cutouts(cand, img, prob, agg, k=k)
    xy = cand[["x_centroid", "y_centroid"]].to_numpy(np.float32)
    cids = cand["candidate_id"].astype(np.int32).to_numpy()
    return X, lab.astype(np.int8), xy, cids


# ---------------------------------------------------------------------------
# Focal-loss training (DataParallel, cosine LR, augmentation)
# ---------------------------------------------------------------------------
def _focal_loss_fn(pw, gamma: float = 2.0):
    """pos_weight-balanced focal BCE (the deployed recipe)."""
    import torch
    import torch.nn.functional as F

    def loss(logits, targets):
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        mod = (1 - p_t) ** gamma
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
        return (mod * ce).mean()
    return loss


def _aug_batch(x):
    """Per-sample random {0,90,180,270} rotation + h/v flip. Safe: trail TP/FP label is
    orientation-invariant. Inputs/outputs are (B, C, k, k) on the model device."""
    import torch
    rot = torch.randint(0, 4, (x.shape[0],), device=x.device)
    hf = torch.randint(0, 2, (x.shape[0],), device=x.device).bool()
    vf = torch.randint(0, 2, (x.shape[0],), device=x.device).bool()
    out = x.clone()
    for r in range(4):
        m = (rot == r)
        if m.any():
            out[m] = torch.rot90(out[m], r, dims=(2, 3))
    if hf.any():
        out[hf] = torch.flip(out[hf], dims=(3,))
    if vf.any():
        out[vf] = torch.flip(out[vf], dims=(2,))
    return out


def train_cnn(X, y, panel=None, *, width: int = NET_WIDTH, depth: int = NET_DEPTH,
              epochs: int = EPOCHS, lr: float = LR, weight_decay: float = WEIGHT_DECAY,
              batch_size: int = BATCH_SIZE, holdout_frac: float = HOLDOUT_FRAC,
              X_holdout=None, y_holdout=None, device: str = "cuda", seed: int = 7,
              gpus: int = 1, augment: bool = True, cosine_lr: bool = True):
    """Fit the focal cutout CNN on cutouts ``X(N,3,k,k)`` with labels ``y(N)``. The held-out
    set is used only for a diagnostic AUC (the operating threshold comes from
    :func:`calibrate_combined_threshold`). Pass an explicit ``X_holdout``/``y_holdout`` (the
    val2 cutouts) for AUC on dedicated panels; otherwise a panel-disjoint slice of the train
    cutouts is held out. ``gpus`` > 1 wraps in DataParallel. Returns ``(net, info)``."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    X = np.clip(np.asarray(X, np.float32), -20, 20)
    y = np.asarray(y, np.float32)
    in_ch, k = X.shape[1], X.shape[2]
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    rng = np.random.default_rng(seed)

    if X_holdout is not None and len(X_holdout):
        Xtr, ytr = X, y
        Xho = np.clip(np.asarray(X_holdout, np.float32), -20, 20)
        yho = np.asarray(y_holdout, np.float32)
        holdout_src = "val2"
    else:
        if panel is not None and len(np.unique(panel)) > 1:
            pans = np.unique(panel); rng.shuffle(pans)
            hp = set(pans[:max(1, int(len(pans) * holdout_frac))].tolist())
            m_h = np.isin(panel, list(hp))
        else:
            idx = rng.permutation(len(y)); n_h = max(1, int(len(y) * holdout_frac))
            m_h = np.zeros(len(y), bool); m_h[idx[:n_h]] = True
        m_t = ~m_h
        Xtr, ytr = X[m_t], y[m_t]
        Xho, yho = X[m_h], y[m_h]
        holdout_src = "internal"

    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr); N = len(yt)
    npos = float((yt == 1).sum()); nneg = float((yt == 0).sum())
    pw = torch.tensor([nneg / max(npos, 1.0)], device=dev)
    print(f"[cnn-train] train {N} (pos {int(npos)}) | holdout {len(yho)} ({holdout_src}) | "
          f"width={width} depth={depth} k={k} epochs={epochs} aug={augment} cos={cosine_lr} "
          f"gpus={gpus} pos_weight={float(pw):.1f}", flush=True)

    net = build_net(width=width, depth=depth, in_ch=in_ch, k=k).to(dev)
    if gpus > 1 and torch.cuda.device_count() >= gpus:
        net = nn.DataParallel(net, device_ids=list(range(gpus)))
        print(f"[cnn-train] DataParallel across {gpus} GPUs", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
    sched = None
    if cosine_lr:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warm = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=3)
        cos = CosineAnnealingLR(opt, T_max=max(1, epochs - 3), eta_min=lr * 0.01)
        sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[3])
    loss_fn = _focal_loss_fn(pw)

    def score(Xa):
        net.eval()
        T = torch.tensor(np.clip(Xa, -20, 20)).to(dev)
        with torch.no_grad():
            return torch.sigmoid(torch.cat([net(T[k:k + 512]) for k in range(0, len(T), 512)])).cpu().numpy()

    best_auc = -1.0
    best_state = None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(N)
        ep_loss = 0.0; nbatch = 0
        for kk in range(0, N, batch_size):
            b = perm[kk:kk + batch_size]
            opt.zero_grad()
            x = Xt[b].to(dev)
            if augment:
                x = _aug_batch(x)
            l = loss_fn(net(x), yt[b].to(dev))
            l.backward(); opt.step()
            ep_loss += float(l.detach()); nbatch += 1
        if sched is not None:
            sched.step()
        msg = f"  ep{ep+1:02d}  loss={ep_loss/max(nbatch,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}"
        if len(yho) and (yho == 0).any() and (yho == 1).any():
            auc = float(roc_auc_score(yho, score(Xho)))
            msg += f"  val_auc={auc:.4f}"
            if auc > best_auc:
                best_auc = auc
                raw_sd = (net.module.state_dict() if hasattr(net, "module") else net.state_dict())
                best_state = {kk: v.detach().cpu().clone() for kk, v in raw_sd.items()}
        print(msg, flush=True)

    # Use best-AUC checkpoint, falling back to the final state if no val was scored.
    final_sd = (net.module.state_dict() if hasattr(net, "module") else net.state_dict())
    state = best_state if best_state is not None else {kk: v.detach().cpu() for kk, v in final_sd.items()}
    info = {"width": int(width), "depth": int(depth), "in_ch": int(in_ch), "k": int(k),
            "epochs": int(epochs), "n_train": int(N), "n_pos_train": int(npos),
            "n_holdout": int(len(yho)), "holdout_src": holdout_src,
            "best_val_auc": round(float(best_auc), 4) if best_auc > 0 else None,
            "augment": bool(augment), "cosine_lr": bool(cosine_lr), "gpus": int(gpus)}
    return state, info


def save_cnn(state_dict, out_pt):
    """Persist the trained CNN state_dict (loadable by ``ADCNN.inference.cnn_postproc.load_cnn``)."""
    import torch
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(out_pt))


# ---------------------------------------------------------------------------
# Combined 5sigma-stack + ADCNN operating point
# ---------------------------------------------------------------------------
def combined_fpp_threshold(adcnn_cat: "pd.DataFrame", stack_cat: "pd.DataFrame",
                           truth, n_panels: int, *, budget: float = FPP_BUDGET,
                           tol_px: float = DEDUP_TOL_PX) -> tuple[float, dict]:
    """ADCNN score threshold at which the dedup'd union (stack ∪ ADCNN[score>=T]) has
    ``fp_per_panel == budget``. Binary search over T in [0, 1] (union FP is monotone decreasing
    in T). Returns ``(threshold, diag)`` with combined recall + FP/panel at the chosen T, the
    5sigma-alone baseline, and the recall ADCNN adds over the stack.

    ``adcnn_cat`` must have columns ``image_id, x, y, beta, length, score`` (ALL candidates --
    build with ``InferenceConfig(cnn_thr=0.0)``). ``stack_cat`` is from
    :func:`ADCNN.evaluation.catalog_match.stack_sigma_catalog`. ``truth`` is the truth catalog
    (path or DataFrame). ``n_panels`` is the panel-count denominator (``len(panel_ids)``).
    """
    from ADCNN.evaluation.catalog_match import evaluate_catalog, dedup_cross_catalog

    cols = ["image_id", "x", "y", "beta", "length"]
    stack = stack_cat[cols].copy()
    adcnn = adcnn_cat[[*cols, "score"]].copy()

    def union_metrics(thr: float):
        a = adcnn[adcnn["score"] >= thr][cols]
        u = dedup_cross_catalog(stack, a, tol_px=tol_px)
        m, _ = evaluate_catalog(u, truth, tol_px=tol_px, n_panels=n_panels)
        return m

    m_stack, _ = evaluate_catalog(stack, truth, tol_px=tol_px, n_panels=n_panels)
    m_lo, m_hi = union_metrics(0.0), union_metrics(1.0)
    warn = ""
    if m_hi["fp_per_panel"] > budget:
        thr = 1.0; m = m_hi
        warn = (f"stack-alone FP/panel {m_hi['fp_per_panel']:.1f} > budget {budget:.0f}; "
                f"ADCNN cut at 1.0 (off)")
    elif m_lo["fp_per_panel"] <= budget:
        thr = 0.0; m = m_lo
        warn = f"all ADCNN candidates under budget ({m_lo['fp_per_panel']:.1f} <= {budget:.0f})"
    else:
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if union_metrics(mid)["fp_per_panel"] > budget:
                lo = mid
            else:
                hi = mid
        thr = hi; m = union_metrics(hi)

    diag = {
        "threshold": round(float(thr), 4),
        "fpp_budget": float(budget),
        "dedup_tol_px": float(tol_px),
        "combined_recall": round(float(m["recall"]), 4),
        "combined_fp_per_panel": round(float(m["fp_per_panel"]), 2),
        f"stack{STACK_SIGMA}_recall": round(float(m_stack["recall"]), 4),
        f"stack{STACK_SIGMA}_fp_per_panel": round(float(m_stack["fp_per_panel"]), 2),
        "adcnn_added_recall": round(float(m["recall"] - m_stack["recall"]), 4),
        "n_panels": int(n_panels),
    }
    if warn:
        diag["warning"] = warn
    return float(thr), diag


def calibrate_combined_threshold(seg_ckpt, cnn_pt, h5_path, csv_path, panel_ids, *,
                                 budget: float = FPP_BUDGET, tol_px: float = DEDUP_TOL_PX,
                                 sigma: int = STACK_SIGMA, device: str = "cuda") -> tuple[float, dict]:
    """Run the two-stage detector at ``cnn_thr=0.0`` on a calibration set (val2), build the
    stack's per-sigma catalog from ``real_labels_<sigma>sigma`` + ``stack_detection_<sigma>sigma``,
    then call :func:`combined_fpp_threshold`. Returns ``(threshold, diag)``. Requires the set
    to carry the per-sigma stack plane + column (build with
    ``make_sim_data --multi-sigma-sets <set> --test-sigmas <sigma>``)."""
    from ADCNN.inference.catalog import (InferenceConfig, build_detection_catalog_multigpu)
    from ADCNN.evaluation.catalog_match import stack_sigma_catalog
    cfg = InferenceConfig(cnn_thr=0.0)
    print(f"[combined-fpp] building val ADCNN catalog (cnn_thr=0) on {len(panel_ids)} panels", flush=True)
    adcnn = build_detection_catalog_multigpu(h5_path, seg_ckpt, cnn_pt, config=cfg, panel_ids=list(panel_ids))
    stack = stack_sigma_catalog(h5_path, csv_path, list(panel_ids), sigma=sigma)
    print(f"[combined-fpp] stack {sigma}sigma catalog: {len(stack)} rows | ADCNN catalog: {len(adcnn)} rows", flush=True)
    return combined_fpp_threshold(adcnn, stack, csv_path, n_panels=len(panel_ids),
                                  budget=budget, tol_px=tol_px)


# ---------------------------------------------------------------------------
# Production stage-2 hook (called by train_end_to_end)
# ---------------------------------------------------------------------------
def _cutouts_for_panels(model, h5_path, csv_path, panel_ids, *, fp_cap, device, k=CUTOUT_K):
    """Build labelled cutouts ``(X, y, panel)`` for the given panels of an h5/csv. The truth
    catalog image_ids are remapped into the 0..N stacking order so the panel-id passed to
    ``panel_cutouts`` matches the injection-overlap labelling. Used by ``train_cnn_from_val``
    for both the training pool and the val2 AUC pool."""
    cat = pd.read_csv(csv_path)
    remap = {orig: i for i, orig in enumerate(panel_ids)}
    cat = cat[cat["image_id"].isin(remap)].copy()
    cat["image_id"] = cat["image_id"].map(remap)
    Xs, ys, pans = [], [], []
    with h5py.File(h5_path, "r") as f:
        for i, orig in enumerate(panel_ids):
            img = f["images"][orig][:].astype(np.float32)
            rl = f["real_labels"][orig][:].astype(np.uint16)
            panel_cat = cat[cat["image_id"] == i]
            X, y, _, _ = panel_cutouts(model, img, rl, panel_cat, pid=i, k=k,
                                       fp_cap=fp_cap, device=device)
            if len(X):
                Xs.append(X); ys.append(y); pans.append(np.full(len(X), i, np.int32))
    if not Xs:
        return (np.zeros((0, 3, k, k), np.float32),
                np.zeros((0,), np.float32), np.zeros((0,), np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(pans)


def train_cnn_from_val(seg_ckpt, train_h5, train_csv, train_panel_ids, out_pt, *,
                       thr_h5=None, thr_csv=None, thr_panel_ids=None,
                       fp_cap: int = FP_CAP, epochs: int = EPOCHS,
                       fpp_budget: float = FPP_BUDGET, gpus: int = 1,
                       device: str = "cuda"):
    """Full stage-2 CNN training: load the TorchScript segmentation model, build labelled
    cutouts on ``train_panel_ids`` (in-memory), fit the focal CNN, save the state_dict, then
    calibrate the operating threshold by the COMBINED 5sigma+ADCNN FP-budget on val2.

    When a threshold set is given (``thr_h5``/``thr_csv``/``thr_panel_ids``, the val2 set),
    :func:`calibrate_combined_threshold` runs the two-stage detector at ``cnn_thr=0`` on val2,
    builds the 5sigma-stack catalog, and binary-searches the ADCNN score cut so the
    deduplicated union has ``fp_per_panel == fpp_budget`` -- that is the threshold written to
    the sidecar. val2 MUST carry ``real_labels_5sigma`` + ``stack_detection_5sigma``
    (rebuild with ``make_sim_data --multi-sigma-sets val2 --test-sigmas 5``); without them the
    calibration is skipped and only the AUC diagnostic is reported."""
    import json
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(seg_ckpt), map_location=dev).eval()

    X, y, panel = _cutouts_for_panels(model, train_h5, train_csv, train_panel_ids,
                                      fp_cap=fp_cap, device=dev)
    if not len(X):
        raise RuntimeError("train_cnn_from_val: no candidates extracted on the train panels")
    print(f"[cnn-train] train pool: {len(y)} cutouts ({int((y == 1).sum())} pos) over "
          f"{len(np.unique(panel))} panels", flush=True)

    Xho = yho = None
    if thr_h5:
        Xho, yho, _ = _cutouts_for_panels(model, thr_h5, thr_csv, thr_panel_ids,
                                          fp_cap=fp_cap, device=dev)
        print(f"[cnn-train] val2 AUC pool: {len(yho)} cutouts ({int((yho == 1).sum())} pos)", flush=True)

    state, info = train_cnn(X, y, panel, epochs=epochs, device=dev,
                            X_holdout=Xho, y_holdout=yho, gpus=gpus)
    save_cnn(state, out_pt)
    sidecar = Path(out_pt).with_suffix(".json")
    sidecar.write_text(json.dumps(info, indent=2))

    # Combined-FPP operating point on val2 (sets the sidecar 'threshold').
    if thr_h5:
        del X, y, panel, Xho, yho   # free train-pool RAM before the calibration GPU pass.
        try:
            thr, diag = calibrate_combined_threshold(seg_ckpt, out_pt, thr_h5, thr_csv,
                                                     list(thr_panel_ids),
                                                     budget=fpp_budget, device=dev)
            info.update(diag)
            sidecar.write_text(json.dumps(info, indent=2))
            print(f"[cnn-train] op-point: thr={thr:.4f} combined recall={diag['combined_recall']:.4f} "
                  f"@ {diag['combined_fp_per_panel']:.1f} FP/panel "
                  f"(stack alone {diag[f'stack{STACK_SIGMA}_recall']:.4f} "
                  f"@ {diag[f'stack{STACK_SIGMA}_fp_per_panel']:.1f}, "
                  f"ADCNN adds +{diag['adcnn_added_recall']:.4f})", flush=True)
        except (ValueError, FileNotFoundError) as e:
            info["threshold_skipped"] = str(e)
            sidecar.write_text(json.dumps(info, indent=2))
            print(f"[cnn-train] WARNING combined-budget calibration skipped: {e}", flush=True)

    print(f"[cnn-train] saved -> {out_pt} | {info}", flush=True)
    return state, info
