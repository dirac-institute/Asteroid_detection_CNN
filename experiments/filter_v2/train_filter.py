"""Config-driven stage-2 filter trainer for the filter-v2 iteration loop.

Loads cached cutouts (built by build_cutouts.py), trains a small CNN, saves the state_dict.
Architecture knobs: width, depth, in_ch (= cutout channels), cutout_size (k -- requires the
matching cutouts cache for k != 48). Training knobs: epochs, batch_size, lr, weight_decay,
focal_alpha (None = pos_weight-balanced BCE focal, the deployed recipe), focal_gamma.
Optional hard-negative mining: a previously trained .pt path + a multiplier that oversamples
that model's high-score FP within the train pool.

Usage:
    python -m experiments.filter_v2.train_filter \
        --train-cuts experiments/filter_v2/cutouts/train2 \
        --val-cuts   experiments/filter_v2/cutouts/val \
        --out experiments/filter_v2/runs/iter01/cnn.pt \
        --width 40 --depth 3 --epochs 30
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Catalog features the aux MLP can see (indices into the cached `cand` array of build_cutouts.py).
# build_cutouts.CAT_COLS == [x_centroid, y_centroid, mf_beta, mf_length, mf_flux, mf_snr, area, max_p]
# Drop positional/orientation cols; keep the matched-filter + footprint shape signals.
DEFAULT_AUX_IDX = [3, 4, 5, 6, 7]   # [mf_length, mf_flux, mf_snr, area, max_p]


def build_net(width: int, depth: int = 3, in_ch: int = 3, k: int = 48, aux_dim: int = 0):
    """Flexible variant of ADCNN.inference.cnn_postproc.build_net.

    `depth` conv blocks (widths w, 2w, 4w, ...) -> AdaptiveAvgPool2d(1) -> Dropout -> Linear.
    If `aux_dim>0`, a small MLP on the catalog feature vector (mf_snr/length/area/max_p/...) is
    concatenated with the backbone output before the head -- the cutout CNN sees PIXELS; the aux
    MLP sees the MATCHED-FILTER SCORES, which is orthogonal information.
    """
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    layers = []
    c = in_ch
    for i in range(depth):
        w = width * (2 ** i)
        layers.append(blk(c, w))
        c = w
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    backbone = nn.Sequential(*layers)
    aux_head = None
    head_in = c
    if aux_dim > 0:
        aux_head = nn.Sequential(
            nn.Linear(aux_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32))
        head_in = c + 32
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(head_in, 1))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.backbone = backbone
            s.aux_head = aux_head
            s.head = head

        def forward(s, x, aux=None):
            z = s.backbone(x)
            if s.aux_head is not None:
                if aux is None:
                    raise ValueError("model was built with aux_dim>0 but aux was not provided")
                z = torch.cat([z, s.aux_head(aux)], dim=1)
            return s.head(z).squeeze(1)
    return Net()


def load_cache(cuts_dir: Path):
    """Load every part_*.npz under `cuts_dir` and concatenate. Returns (X, y, panel, cand)."""
    parts = sorted(cuts_dir.glob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no part_*.npz under {cuts_dir}")
    Xs, ys, ps, cs = [], [], [], []
    for p in parts:
        z = np.load(p)
        Xs.append(z["X"]); ys.append(z["y"]); ps.append(z["panel"]); cs.append(z["cand"])
    X = np.concatenate(Xs); y = np.concatenate(ys); panel = np.concatenate(ps); cand = np.concatenate(cs)
    print(f"  cache {cuts_dir}: N={len(y)}  TP={int((y == 1).sum())}  FP={int((y == 0).sum())}  panels={len(np.unique(panel))}")
    return X, y, panel, cand


def hnm_weights(net, X_train, y_train, batch_size, device, hnm_mult: float):
    """Hard-negative mining: score X_train with a prior `net`, give FP that score high `hnm_mult`x weight."""
    import torch
    net.eval()
    n = len(X_train); s = np.zeros(n, np.float32)
    with torch.no_grad():
        for k in range(0, n, batch_size):
            chunk = torch.tensor(np.clip(X_train[k:k + batch_size], -20, 20)).to(device)
            s[k:k + batch_size] = torch.sigmoid(net(chunk)).cpu().numpy()
    is_fp = y_train == 0
    fp_thr = np.quantile(s[is_fp], 0.75) if is_fp.any() else 1.0
    hard = is_fp & (s >= fp_thr)
    w = np.ones(n, np.float32); w[hard] = hnm_mult
    print(f"  HNM: hard FP = top quartile of FP scores (thr={fp_thr:.3f}); {int(hard.sum())} hard / {int(is_fp.sum())} FP; mult={hnm_mult}")
    return w


def focal_loss_fn(pw_or_alpha, gamma: float = 2.0, alpha=None):
    import torch, torch.nn.functional as F
    def loss(logits, targets, sample_w=None):
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        mod = (1 - p_t) ** gamma
        if alpha is None:
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none",
                                                     pos_weight=pw_or_alpha)
        else:
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            a_t = torch.where(targets == 1, alpha, 1.0 - alpha)
            ce = a_t * ce
        l = mod * ce
        if sample_w is not None:
            l = l * sample_w
        return l.mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cuts", required=True)
    ap.add_argument("--val-cuts", default=None, help="optional held-out cutouts for AUC")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--k", type=int, default=None,
                    help="center-crop cached cutouts to this side length before training (default = use cache size)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=None,
                    help="if set, use alpha-balanced focal instead of pos_weight focal")
    ap.add_argument("--hnm-from", default=None, help="prior .pt to mine hard FP from")
    ap.add_argument("--hnm-mult", type=float, default=3.0)
    ap.add_argument("--augment", action="store_true",
                    help="per-batch random {0,90,180,270} rotation + h/v flip "
                         "(safe: trail TP/FP label is orientation-invariant)")
    ap.add_argument("--aux", action="store_true",
                    help="add a small MLP head on the catalog features [mf_length, mf_flux, mf_snr, area, max_p]")
    ap.add_argument("--gpus", type=int, default=1,
                    help="wrap the net in DataParallel across the first N visible GPUs (1 = single-GPU)")
    ap.add_argument("--cosine-lr", action="store_true",
                    help="cosine LR schedule with warmup (3 ep) to 0 over the run")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    print(f"=== train_filter | out={a.out} ===", flush=True)
    print(f"config: width={a.width} depth={a.depth} epochs={a.epochs} lr={a.lr} "
          f"focal_gamma={a.focal_gamma} focal_alpha={a.focal_alpha} hnm_from={a.hnm_from}", flush=True)

    Xtr, ytr, _, ctr = load_cache(Path(a.train_cuts))
    Xtr = np.clip(Xtr, -20, 20).astype(np.float32)
    ytr = ytr.astype(np.float32)
    in_ch, cache_k = Xtr.shape[1], Xtr.shape[2]
    # Optional center crop (the cache is built at k=96; iters can subset to 48/64).
    k = a.k if a.k is not None else cache_k
    if k != cache_k:
        if k > cache_k:
            raise SystemExit(f"--k {k} exceeds cache k {cache_k}")
        o = (cache_k - k) // 2
        Xtr = Xtr[:, :, o:o + k, o:o + k]
    print(f"  cutout: in_ch={in_ch}  cache_k={cache_k}  training_k={k}", flush=True)

    # Optional aux catalog features.
    Atr = Ava = None; aux_mean = aux_std = None; aux_dim = 0
    if a.aux:
        # cand columns: see build_cutouts.CAT_COLS. Use DEFAULT_AUX_IDX = [3,4,5,6,7].
        raw = ctr[:, DEFAULT_AUX_IDX].astype(np.float32)
        # mf_snr + area can have huge dynamic range -> robust z-score on the training pool.
        aux_mean = np.nanmean(raw, axis=0)
        aux_std = np.nanstd(raw, axis=0) + 1e-6
        Atr = ((np.nan_to_num(raw, nan=0.0) - aux_mean) / aux_std).astype(np.float32)
        Atr = np.clip(Atr, -10, 10)
        aux_dim = Atr.shape[1]
        print(f"  aux: cols={DEFAULT_AUX_IDX}  mean={aux_mean.tolist()}  std={aux_std.tolist()}", flush=True)

    Xva = yva = None
    if a.val_cuts:
        Xva, yva, _, cva = load_cache(Path(a.val_cuts))
        Xva = np.clip(Xva, -20, 20).astype(np.float32); yva = yva.astype(np.float32)
        if k != cache_k:
            o = (cache_k - k) // 2
            Xva = Xva[:, :, o:o + k, o:o + k]
        if a.aux:
            rawv = cva[:, DEFAULT_AUX_IDX].astype(np.float32)
            Ava = np.clip(((np.nan_to_num(rawv, nan=0.0) - aux_mean) / aux_std), -10, 10).astype(np.float32)

    net = build_net(a.width, a.depth, in_ch=in_ch, k=k, aux_dim=aux_dim).to(dev)
    if a.gpus > 1 and torch.cuda.device_count() >= a.gpus:
        import torch.nn as nn
        net = nn.DataParallel(net, device_ids=list(range(a.gpus)))
        print(f"  DataParallel across {a.gpus} GPUs", flush=True)

    sample_w = None
    if a.hnm_from:
        prior = build_net(a.width, a.depth, in_ch=in_ch, k=k).to(dev)
        prior.load_state_dict(torch.load(a.hnm_from, map_location=dev, weights_only=True))
        sample_w = hnm_weights(prior, Xtr, ytr, a.batch_size, dev, a.hnm_mult)

    npos = float((ytr == 1).sum()); nneg = float((ytr == 0).sum())
    pw = torch.tensor([nneg / max(npos, 1.0)], device=dev)
    print(f"  train N={len(ytr)} pos={int(npos)} pos_weight={float(pw):.1f}", flush=True)
    loss_fn = focal_loss_fn(pw, gamma=a.focal_gamma, alpha=a.focal_alpha)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=a.weight_decay)
    # Optional cosine LR schedule with 3-epoch linear warmup; LR per-EPOCH (call .step() at epoch end).
    sched = None
    if a.cosine_lr:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warm = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=3)
        cos = CosineAnnealingLR(opt, T_max=max(1, a.epochs - 3), eta_min=a.lr * 0.01)
        sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[3])
        print(f"  cosine LR + 3-ep warmup ({a.lr:.1e} -> {a.lr*0.01:.1e})", flush=True)

    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    At = torch.tensor(Atr) if Atr is not None else None
    sw = torch.tensor(sample_w) if sample_w is not None else None
    N = len(yt)

    def aug_batch(x):
        """Random 90-rotations + flips, per-sample. Cutouts are square so dims stay (B,C,k,k)."""
        # rot in {0,1,2,3}, hflip {0,1}, vflip {0,1} -- chosen independently per sample
        rot = torch.randint(0, 4, (x.shape[0],), device=x.device)
        hf  = torch.randint(0, 2, (x.shape[0],), device=x.device).bool()
        vf  = torch.randint(0, 2, (x.shape[0],), device=x.device).bool()
        out = x.clone()
        # group by rotation (avoids per-sample python loop)
        for r in range(4):
            m = (rot == r)
            if m.any():
                out[m] = torch.rot90(out[m], r, dims=(2, 3))
        if hf.any():
            out[hf] = torch.flip(out[hf], dims=(3,))
        if vf.any():
            out[vf] = torch.flip(out[vf], dims=(2,))
        return out
    best_auc = -1.0; best_state = None
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(N)
        ep_loss = 0.0; nbatch = 0
        for bi in range(0, N, a.batch_size):
            idx = perm[bi:bi + a.batch_size]
            opt.zero_grad()
            x = Xt[idx].to(dev); y = yt[idx].to(dev)
            if a.augment:
                x = aug_batch(x)
            aux_b = At[idx].to(dev) if At is not None else None
            w = sw[idx].to(dev) if sw is not None else None
            loss = loss_fn(net(x, aux_b) if aux_b is not None else net(x), y, w)
            loss.backward(); opt.step()
            ep_loss += float(loss); nbatch += 1
        if sched is not None:
            sched.step()
        msg = f"  ep{ep+1:02d}  loss={ep_loss/max(nbatch,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}"
        if Xva is not None and (yva == 1).any() and (yva == 0).any():
            net.eval()
            Xv = torch.tensor(Xva); s = np.zeros(len(yva), np.float32)
            Av = torch.tensor(Ava) if Ava is not None else None
            with torch.no_grad():
                for bi in range(0, len(yva), 1024):
                    chunk = Xv[bi:bi + 1024].to(dev)
                    if Av is not None:
                        s[bi:bi + 1024] = torch.sigmoid(net(chunk, Av[bi:bi + 1024].to(dev))).cpu().numpy()
                    else:
                        s[bi:bi + 1024] = torch.sigmoid(net(chunk)).cpu().numpy()
            auc = float(roc_auc_score(yva, s))
            msg += f"  val_auc={auc:.4f}"
            if auc > best_auc:
                best_auc = auc
                best_state = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
        print(msg, flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    # strip DataParallel's "module." prefix so eval can load a fresh non-DP net.
    raw_state = (net.module.state_dict() if hasattr(net, "module") else net.state_dict())
    save_state = best_state if best_state is not None else {kk: v.detach().cpu() for kk, v in raw_state.items()}
    if best_state is not None:
        # best_state was captured WITH the prefix if net was DP -- strip it here too.
        save_state = {kk.removeprefix("module."): v for kk, v in save_state.items()}
    torch.save(save_state, out)
    info = {"out": str(out), "width": a.width, "depth": a.depth, "in_ch": in_ch, "k": k,
            "epochs": a.epochs, "best_val_auc": best_auc, "n_train": int(N), "n_pos": int(npos),
            "aux_dim": aux_dim,
            "aux_idx": DEFAULT_AUX_IDX if aux_dim > 0 else [],
            "aux_mean": aux_mean.tolist() if aux_mean is not None else [],
            "aux_std":  aux_std.tolist() if aux_std is not None else []}
    (out.with_suffix(".json")).write_text(json.dumps(info, indent=2))
    print(f"DONE -> {out}  best_val_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
