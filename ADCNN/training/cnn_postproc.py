"""Train the stage-2 false-positive filter — the focal-loss cutout CNN.

Pipeline stage 2: the segmentation model segmentation emits many candidate components per panel (asteroid trails
+ residual/artefact false positives). The cutout CNN scores a 48x48x3 patch
``[diffim/sigma, seg_prob, seg_agg]`` per candidate and rejects false positives while keeping
trails. It replaces the legacy 72-feature RandomForest.

Training is leakage-safe: cutouts are built by running the trained segmentation model on held-out panels (never
the test set), candidates are labelled by overlap with the injected truth
(``label_candidates_by_injection_overlap``), and a focal-loss class-balanced CNN is fit.

Two entry points:
  - ``build_cutout_dataset`` : run segmentation model over a large h5 once and cache cutouts to chunked, resumable
    ``part_*.npz`` files (use for a dedicated training set, e.g. the SNR 2-8 ``train2`` set).
  - ``train_cnn_from_val``   : the production stage-2 hook (called by
    ``ADCNN.pipelines.train_end_to_end``) — build cutouts in-memory on the freshly trained segmentation model's
    held-out val panels, fit the CNN, save the state_dict.

The network architecture lives in ``ADCNN.inference.cnn_postproc.build_net`` so training and
inference share one definition.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ADCNN.inference.cnn_postproc import build_net, make_cutouts, CUTOUT_K, NET_WIDTH

# ---- training defaults (the recipe the deployed models/cnn_postproc.pt was trained with) ----
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
HOLDOUT_FRAC = 0.10        # panel-disjoint holdout used for the diagnostic AUC
FP_CAP = 600               # max false-positive cutouts kept per panel (class balance / disk)

# ---- operating point (the COMBINED detector) -----------------------------------------------
# ADCNN complements the classical 5sigma stack: the deployed system reports the DEDUPLICATED UNION
# of the stack's detections and ADCNN's. FPP_BUDGET is the false-positives-per-panel that union is
# allowed; calibrate_combined_threshold sets ADCNN's score cut so the union hits it on val2 (which
# must carry the stack's residual-FP plane, real_labels_5sigma -- build it with
# `make_sim_data --multi-sigma-sets val2 --test-sigmas 5`). A trail counts as recovered if EITHER
# detector finds it; FP both fire on (within DEDUP_TOL_PX) are counted once.
FPP_BUDGET = 200.0         # combined 5sigma-stack + ADCNN false positives per panel
DEDUP_TOL_PX = 20.0        # two detections this close (per panel) are the same source
STACK_SIGMA = 5            # stack-detection sigma the budget is reported against (real_labels_<s>sigma)


# ---------------------------------------------------------------------------
# Cutout extraction (one panel)
# ---------------------------------------------------------------------------
def panel_cutouts(model, img, rl, panel_cat, *, pid: int = 0, k: int = CUTOUT_K,
                  fp_cap: int = 0, device="cuda"):
    """Run segmentation model on one panel, extract candidates, label them by injection overlap, and return the
    per-candidate cutout stack. `panel_cat` is the truth catalog rows for this panel with its
    image_id set to `pid`. Returns (X[N,3,k,k], y[N], xy[N,2], cids[N]); empty arrays if none."""
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import compute_v2_features, label_candidates_by_injection_overlap

    prob, sn, cs, agg = predict_panel_overlap_3ch_full(model, img, rl, device=device)
    prob = prob.astype(np.float32); agg = np.asarray(agg, np.float32)
    cand, _ = compute_v2_features({pid: prob}, {pid: img}, {pid: sn}, {pid: cs}, {pid: agg},
                                  real_labels={pid: rl.astype(np.uint16)}, verbose=False)
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
# Cached cutout dataset (large training set, resumable)
# ---------------------------------------------------------------------------
def build_cutout_dataset(seg_ckpt, h5_path, csv_path, out_dir, *, k: int = CUTOUT_K,
                         fp_cap: int = FP_CAP, chunk: int = 40, device: str = "cuda"):
    """Run segmentation model over every panel of `h5_path`, build labelled cutouts, and cache them to chunked
    ``part_XXXX.npz`` files under `out_dir` (resumable via ``done.txt`` — survives preemption).
    Use this to materialise a dedicated training set (e.g. the SNR 2-8 train2 set)."""
    import torch
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    done_file = out / "done.txt"
    done = set(int(x) for x in done_file.read_text().split()) if done_file.exists() else set()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(seg_ckpt), map_location=dev).eval()
    cat = pd.read_csv(csv_path)

    buf = {kk: [] for kk in ("X", "y", "panel", "cid", "xy")}
    chunk_pids: list[int] = []

    def flush():
        if not buf["X"]:
            return
        fn = out / f"part_{chunk_pids[0]:04d}.npz"
        np.savez_compressed(fn, X=np.concatenate(buf["X"]).astype(np.float32),
                            y=np.concatenate(buf["y"]).astype(np.int8),
                            panel=np.array(buf["panel"], np.int32),
                            cid=np.concatenate(buf["cid"]).astype(np.int32),
                            xy=np.concatenate(buf["xy"]).astype(np.float32))
        done.update(chunk_pids); done_file.write_text(" ".join(map(str, sorted(done))))
        for kk in buf:
            buf[kk].clear()
        print(f"  [flush] {fn.name} ({len(done)} panels done)", flush=True)
        chunk_pids.clear()

    with h5py.File(h5_path, "r") as f:
        npan = int(f["images"].shape[0])
        for pid in range(npan):
            if pid in done:
                continue
            img = f["images"][pid][:].astype(np.float32)
            rl = f["real_labels"][pid][:].astype(np.uint16)
            X, y, xy, cids = panel_cutouts(model, img, rl, cat, pid=pid, k=k,
                                           fp_cap=fp_cap, device=dev)
            chunk_pids.append(pid)
            if len(X):
                buf["X"].append(X); buf["y"].append(y); buf["cid"].append(cids); buf["xy"].append(xy)
                buf["panel"].extend([pid] * len(X))
                print(f"  panel {pid}: {len(X)} cand, {int((y == 1).sum())} TP", flush=True)
            if len(chunk_pids) >= chunk:
                flush()
    flush()
    n_parts = len(list(out.glob("part_*.npz")))
    print(f"CUTOUTS DONE: {len(done)} panels -> {out}/ ({n_parts} parts)", flush=True)


# ---------------------------------------------------------------------------
# Focal-loss training
# ---------------------------------------------------------------------------
def _focal_loss_fn(pos_weight, gamma: float = 2.0, alpha: float | None = None):
    """Focal-modulated BCE-with-logits: ((1 - p_t)^gamma) * BCE.

    Default (alpha=None) keeps the class-balancing ``pos_weight`` INSIDE the BCE -- this is what the
    DEPLOYED ``models/cnn_postproc.pt`` was trained with, kept as the reproducible default. Note that
    combining ``pos_weight`` with the focal modulator is non-standard (it double-counts the class
    imbalance); to retrain with the canonical alpha-balanced focal loss instead, pass ``alpha`` (e.g.
    0.25), which drops ``pos_weight`` and applies a per-class alpha factor.
    """
    import torch
    import torch.nn.functional as F

    def loss(logits, targets):
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        mod = (1 - p_t) ** gamma
        if alpha is None:
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
            return (mod * ce).mean()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        a_t = torch.where(targets == 1, alpha, 1.0 - alpha)
        return (a_t * mod * ce).mean()
    return loss


def train_cnn(X, y, panel=None, *, width: int = NET_WIDTH, epochs: int = EPOCHS, lr: float = LR,
              weight_decay: float = WEIGHT_DECAY, batch_size: int = BATCH_SIZE,
              holdout_frac: float = HOLDOUT_FRAC,
              focal_alpha: float | None = None, device: str = "cuda", seed: int = 7,
              X_holdout=None, y_holdout=None):
    """Fit the focal-loss cutout CNN on cutouts `X`(N,3,k,k) with labels `y`(N).

    The held-out set is used only to report a diagnostic AUC — never to tune weights, and NOT to
    set the operating threshold (that is the combined 5sigma+ADCNN FP budget, set by
    `calibrate_combined_threshold` on val2). Pass an EXPLICIT `X_holdout`/`y_holdout` (the val2
    set) to score the AUC on dedicated panels; otherwise a PANEL-DISJOINT slice of the training
    cutouts (by `panel`, when given) is held out. Returns (net, info).
    """
    import torch
    from sklearn.metrics import roc_auc_score

    X = np.clip(np.asarray(X, np.float32), -20, 20)
    y = np.asarray(y, np.float32)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))  # deterministic weight init + minibatch shuffle (reproducible reruns)
    np.random.seed(int(seed))
    rng = np.random.default_rng(seed)

    if X_holdout is not None and len(X_holdout):
        # explicit threshold set (val2): train on ALL of (X, y); threshold/AUC on val2.
        Xtr, ytr = X, y
        Xho = np.clip(np.asarray(X_holdout, np.float32), -20, 20)
        yho = np.asarray(y_holdout, np.float32)
        holdout_src = "val2"
    else:
        # panel-disjoint (fallback row) holdout carved from the training cutouts
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
          f"epochs {epochs} pos_weight {float(pw):.1f}", flush=True)

    net = build_net(width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
    loss_fn = _focal_loss_fn(pw, alpha=focal_alpha)

    def score(Xa):
        net.eval()
        T = torch.tensor(np.clip(Xa, -20, 20)).to(dev)
        with torch.no_grad():
            return torch.sigmoid(torch.cat([net(T[k:k + 512]) for k in range(0, len(T), 512)])).cpu().numpy()

    for ep in range(epochs):
        net.train(); perm = torch.randperm(N)
        for k in range(0, N, batch_size):
            b = perm[k:k + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b].to(dev)), yt[b].to(dev))
            loss.backward(); opt.step()
        if ep % 5 == 4:
            print(f"  ep{ep} done", flush=True)

    # diagnostic AUC on the held-out cutouts (the operating threshold is set separately, by the
    # combined 5sigma+ADCNN FP budget in calibrate_combined_threshold)
    info = {"n_train": int(N), "n_pos_train": int(npos), "n_holdout": int(len(yho)), "holdout_src": holdout_src}
    if len(yho) and (yho == 0).any() and (yho == 1).any():
        info["holdout_auc"] = round(float(roc_auc_score(yho, score(Xho))), 4)
    return net, info


def save_cnn(net, out_pt):
    """Persist the trained CNN state_dict (loadable by ``ADCNN.inference.cnn_postproc.load_cnn``)."""
    import torch
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_pt))


# ---------------------------------------------------------------------------
# Combined 5sigma-stack + ADCNN operating point
# ---------------------------------------------------------------------------
def combined_fpp_threshold(adcnn_cat: "pd.DataFrame", stack_cat: "pd.DataFrame",
                           truth, n_panels: int, *, budget: float = FPP_BUDGET,
                           tol_px: float = DEDUP_TOL_PX) -> tuple[float, dict]:
    """Find the ADCNN score threshold at which the deduplicated union (stack ∪ ADCNN[score>=T])
    has ``fp_per_panel == budget``. Binary-search over T in [0, 1] (union FP is monotone
    decreasing in T). Returns (threshold, diagnostics): combined recall + FP/panel at the
    chosen T, the 5sigma-alone baseline, and the recall ADCNN adds over the stack.

    `adcnn_cat` : ADCNN catalog with at minimum ``image_id, x, y, beta, length, score`` (ALL
                  candidates kept -- the FP budget must see every false positive, so build with
                  ``InferenceConfig(cnn_thr=0.0)``).
    `stack_cat` : 5sigma-stack catalog from :func:`_stack_sigma_catalog`.
    `truth`     : truth catalog (path or DataFrame) for the recall denominator.
    `n_panels`  : panel count for the ``fp_per_panel`` denominator (== ``len(panel_ids)``).
    """
    from ADCNN.evaluation.catalog_match import evaluate_catalog, dedup_cross_catalog

    cols = ["image_id", "x", "y", "beta", "length"]
    stack = stack_cat[cols].copy()
    adcnn = adcnn_cat[[*cols, "score"]].copy()

    def union_metrics(thr: float):
        a = adcnn[adcnn["score"] >= thr][cols]
        # Cross-catalog dedup: a source both detectors fire on counts once (attributed to stack);
        # within-stack and within-ADCNN clusters are preserved (each detector reports its own).
        u = dedup_cross_catalog(stack, a, tol_px=tol_px)
        m, _ = evaluate_catalog(u, truth, tol_px=tol_px, n_panels=n_panels)
        return m

    m_stack, _ = evaluate_catalog(stack, truth, tol_px=tol_px, n_panels=n_panels)
    m_lo, m_hi = union_metrics(0.0), union_metrics(1.0)
    warn = ""
    if m_hi["fp_per_panel"] > budget:
        thr = 1.0; m = m_hi    # stack alone exceeds budget -- can't meet it even with ADCNN off
        warn = (f"stack-alone FP/panel {m_hi['fp_per_panel']:.1f} > budget {budget:.0f}; "
                f"ADCNN cut at 1.0 (off), combined FP/panel = {m['fp_per_panel']:.1f}")
    elif m_lo["fp_per_panel"] <= budget:
        thr = 0.0; m = m_lo    # even keeping all ADCNN candidates stays under budget
        warn = f"all ADCNN candidates under budget ({m_lo['fp_per_panel']:.1f} <= {budget:.0f}); cut at 0"
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
    """End-to-end combined-budget calibration on a held-out set (val2): run the two-stage
    detector at ``cnn_thr=0.0`` to get every candidate's score, build the stack's 5sigma
    catalog from the set's ``real_labels_<sigma>sigma`` plane + ``stack_detection_<sigma>sigma``
    column, then call :func:`combined_fpp_threshold`. Returns (threshold, diag).
    Requires `h5_path`/`csv_path` to carry the per-sigma stack plane + column (build the set
    with ``make_sim_data --multi-sigma-sets <set> --test-sigmas <sigma>``)."""
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
def _cutouts_for_panels(model, h5_path, csv_path, panel_ids, *, fp_cap, device):
    """Build labelled cutouts (X, y, panel) for the given panels of an h5/csv. The truth catalog
    image_ids are remapped into the 0..N stacking order before injection-overlap labelling (so it
    works even when the panels are a high-index slice of a shared shard h5)."""
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
            X, y, _, _ = panel_cutouts(model, img, rl, panel_cat, pid=i, fp_cap=fp_cap, device=device)
            if len(X):
                Xs.append(X); ys.append(y); pans.append(np.full(len(X), i, np.int32))
    if not Xs:
        return (np.zeros((0, 3, CUTOUT_K, CUTOUT_K), np.float32),
                np.zeros((0,), np.float32), np.zeros((0,), np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(pans)


def train_cnn_from_val(seg_ckpt, train_h5, train_csv, train_panel_ids, out_pt, *,
                       thr_h5=None, thr_csv=None, thr_panel_ids=None,
                       fp_cap: int = FP_CAP, epochs: int = EPOCHS,
                       fpp_budget: float = FPP_BUDGET, device: str = "cuda"):
    """Full stage-2 CNN training: load the TorchScript segmentation model, build labelled cutouts
    on `train_panel_ids` (in-memory), fit the focal CNN, save the state_dict to `out_pt`, then
    set its operating threshold by the COMBINED 5sigma+ADCNN FP budget on the val2 set.

    Operating point: when a threshold set is given (`thr_h5`/`thr_csv`/`thr_panel_ids`, the val2
    set), :func:`calibrate_combined_threshold` runs the two-stage detector at ``cnn_thr=0`` on
    val2, builds the 5sigma-stack catalog, and binary-searches the ADCNN score cut so the
    deduplicated union (stack ∪ ADCNN) has ``fp_per_panel == fpp_budget`` -- that is the
    threshold written to the sidecar. val2 must carry ``real_labels_<sigma>sigma`` +
    ``stack_detection_<sigma>sigma`` (rebuild with ``make_sim_data --multi-sigma-sets val2
    --test-sigmas 5``); without them the calibration is skipped and only the AUC diagnostic is
    reported. When no threshold set is given, a panel-disjoint slice of the train cutouts is
    held out for the AUC only.

    Leakage-safe: the segmentation model must not have trained on these panels."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(seg_ckpt), map_location=dev).eval()

    X, y, panel = _cutouts_for_panels(model, train_h5, train_csv, train_panel_ids, fp_cap=fp_cap, device=dev)
    if not len(X):
        raise RuntimeError("train_cnn_from_val: no candidates extracted on the train panels")
    print(f"[cnn-train] train pool: {len(y)} cutouts ({int((y == 1).sum())} pos) over "
          f"{len(np.unique(panel))} panels", flush=True)

    Xho = yho = None
    if thr_h5:
        Xho, yho, _ = _cutouts_for_panels(model, thr_h5, thr_csv, thr_panel_ids, fp_cap=fp_cap, device=dev)
        print(f"[cnn-train] val2 AUC pool: {len(yho)} cutouts ({int((yho == 1).sum())} pos)", flush=True)

    net, info = train_cnn(X, y, panel, epochs=epochs, device=dev, X_holdout=Xho, y_holdout=yho)
    save_cnn(net, out_pt)

    # combined operating point on val2 (sets sidecar 'threshold')
    if thr_h5:
        del X, y, panel, Xho, yho   # free training-pool memory before the calibration GPU pass
        try:
            thr, diag = calibrate_combined_threshold(seg_ckpt, out_pt, thr_h5, thr_csv,
                                                     list(thr_panel_ids), budget=fpp_budget, device=dev)
            info.update(diag)       # diag carries 'threshold' + combined-budget diagnostics
            print(f"[cnn-train] combined-budget op-point: thr={thr:.4f} "
                  f"combined recall={diag['combined_recall']:.4f} "
                  f"@ {diag['combined_fp_per_panel']:.1f} FP/panel "
                  f"(stack alone {diag[f'stack{STACK_SIGMA}_recall']:.4f} "
                  f"@ {diag[f'stack{STACK_SIGMA}_fp_per_panel']:.1f}, "
                  f"ADCNN adds +{diag['adcnn_added_recall']:.4f})", flush=True)
        except (ValueError, FileNotFoundError) as e:
            info["threshold_skipped"] = str(e)
            print(f"[cnn-train] WARNING combined-budget calibration skipped: {e}", flush=True)

    print(f"[cnn-train] saved -> {out_pt} | {info}", flush=True)
    return net, info
