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
HOLDOUT_FRAC = 0.10        # panel-disjoint holdout used to set the operating threshold
RECALL_TARGET = 0.95       # threshold reported at this trail recall on the holdout
FP_CAP = 600               # max false-positive cutouts kept per panel (class balance / disk)


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


def load_cutout_parts(path):
    """Load a cutout dataset: a directory of ``part_*.npz`` OR a single ``.npz``. Returns a dict
    with concatenated X, y, panel, xy arrays."""
    p = Path(path)
    files = sorted(p.glob("part_*.npz")) if p.is_dir() else [p]
    keys = ("X", "y", "panel", "xy")
    acc = {k: [] for k in keys}
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        for k in keys:
            acc[k].append(z[k])
    return {k: np.concatenate(v) for k, v in acc.items()}


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
              holdout_frac: float = HOLDOUT_FRAC, recall_target: float = RECALL_TARGET,
              focal_alpha: float | None = None, device: str = "cuda", seed: int = 7):
    """Fit the focal-loss cutout CNN on cutouts `X`(N,3,k,k) with labels `y`(N).

    A PANEL-DISJOINT holdout (by `panel`, when given) is used only to report the operating
    threshold at `recall_target` trail recall and the holdout AUC — never to tune weights.
    Returns (net, info) where info has the holdout threshold/auc and the class counts.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    X = np.clip(np.asarray(X, np.float32), -20, 20)
    y = np.asarray(y, np.float32)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # panel-disjoint train / holdout split (fallback to a row split if no panel ids)
    rng = np.random.default_rng(seed)
    if panel is not None and len(np.unique(panel)) > 1:
        pans = np.unique(panel); rng.shuffle(pans)
        hp = set(pans[:max(1, int(len(pans) * holdout_frac))].tolist())
        m_h = np.isin(panel, list(hp))
    else:
        idx = rng.permutation(len(y)); n_h = max(1, int(len(y) * holdout_frac))
        m_h = np.zeros(len(y), bool); m_h[idx[:n_h]] = True
    m_t = ~m_h

    Xt = torch.tensor(X[m_t]); yt = torch.tensor(y[m_t]); N = len(yt)
    npos = float((yt == 1).sum()); nneg = float((yt == 0).sum())
    pw = torch.tensor([nneg / max(npos, 1.0)], device=dev)
    print(f"[cnn-train] train {N} (pos {int(npos)}) | holdout {int(m_h.sum())} | "
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

    # operating threshold @ recall_target + AUC on the held-out panels
    info = {"n_train": int(N), "n_pos_train": int(npos), "n_holdout": int(m_h.sum())}
    if m_h.any() and (y[m_h] == 1).any():
        sh = score(X[m_h]); yh = y[m_h]
        info["threshold"] = round(float(np.quantile(sh[yh == 1], 1 - recall_target)), 4)
        if (yh == 0).any() and (yh == 1).any():
            info["holdout_auc"] = round(float(roc_auc_score(yh, sh)), 4)
    return net, info


def save_cnn(net, out_pt):
    """Persist the trained CNN state_dict (loadable by ``ADCNN.inference.cnn_postproc.load_cnn``)."""
    import torch
    Path(out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_pt))


# ---------------------------------------------------------------------------
# Production stage-2 hook (called by train_end_to_end)
# ---------------------------------------------------------------------------
def train_cnn_from_val(seg_ckpt, val_h5, val_csv, val_panel_ids, out_pt, *,
                       fp_cap: int = FP_CAP, epochs: int = EPOCHS, device: str = "cuda"):
    """Full stage-2 CNN training: load the TorchScript segmentation model, build labelled cutouts on the held-out
    val panels (in-memory), fit the focal CNN, and save the state_dict to `out_pt`.

    Mirrors the leakage-safe contract of the old ``train_rf_from_val``: the val panels' truth
    catalog image_ids are remapped into the 0..N stacking order before injection-overlap labelling
    (so it works even when the val panels are a high-index slice of a shared shard h5)."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(seg_ckpt), map_location=dev).eval()

    cat = pd.read_csv(val_csv)
    remap = {orig: i for i, orig in enumerate(val_panel_ids)}
    cat = cat[cat["image_id"].isin(remap)].copy()
    cat["image_id"] = cat["image_id"].map(remap)

    Xs, ys, pans = [], [], []
    with h5py.File(val_h5, "r") as f:
        for i, orig in enumerate(val_panel_ids):
            img = f["images"][orig][:].astype(np.float32)
            rl = f["real_labels"][orig][:].astype(np.uint16)
            panel_cat = cat[cat["image_id"] == i]
            X, y, _, _ = panel_cutouts(model, img, rl, panel_cat, pid=i, fp_cap=fp_cap, device=dev)
            if len(X):
                Xs.append(X); ys.append(y); pans.append(np.full(len(X), i, np.int32))
    if not Xs:
        raise RuntimeError("train_cnn_from_val: no candidates extracted on the val panels")
    X = np.concatenate(Xs); y = np.concatenate(ys); panel = np.concatenate(pans)
    print(f"[cnn-train] pool: {len(y)} cutouts ({int((y == 1).sum())} pos) over "
          f"{len(np.unique(panel))} val panels", flush=True)

    net, info = train_cnn(X, y, panel, epochs=epochs, device=dev)
    save_cnn(net, out_pt)
    print(f"[cnn-train] saved -> {out_pt} | holdout {info}", flush=True)
    return net, info
