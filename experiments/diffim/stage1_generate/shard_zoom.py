"""Sample a handful of injected trails from a stage-1 shard and produce a
zoom PNG so the user can visually sanity-check the shard.

Shards lack the raw science/template images (only the diffim is kept), so
the columns shown are just the diffim/truth/channel views that actually
drive training.

Usage:
    python shard_zoom.py --shard path/to/shard.npz [--n 6] [--half 80]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def symmetric(arr):
    g = arr[np.isfinite(arr)]
    if g.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(g, [1, 99])
    v = max(abs(lo), abs(hi))
    return -v, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=6,
                    help="Number of injections to preview (random).")
    ap.add_argument("--half", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(args.shard)
    H, W = z["diffim_injected"].shape
    xs, ys = z["injection_x"], z["injection_y"]
    betas, lengths, mags = z["injection_beta"], z["injection_trail_length"], z["injection_mag"]
    recov = z["recovered_by_ap"]
    n_inj = len(xs)

    rng = np.random.default_rng(args.seed)
    # Sample across the AP-recovered and AP-missed slices.
    missed = np.where(recov == 0)[0]
    hit = np.where(recov == 1)[0]
    picks = []
    n_missed = min(args.n // 2, len(missed))
    n_hit = args.n - n_missed
    if n_missed:
        picks += list(rng.choice(missed, size=n_missed, replace=False))
    picks += list(rng.choice(hit, size=min(n_hit, len(hit)), replace=False))

    cols = [
        ("diffim_injected", "Diffim INJ", True),
        ("empirical_injected_only", "Empirical = inj - clean", True),
        ("truth", "Truth (injected geom in diffim)", False),
        ("ch_signed", "ch_signed (NN input)", True),
        ("ch_var", "ch_var (NN input)", False),
        ("ch_bad", "ch_bad (NN input)", False),
    ]
    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, len(cols), figsize=(2.2 * len(cols), 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for ri, idx in enumerate(picks):
        x0i, y0i = int(round(float(xs[idx]))), int(round(float(ys[idx])))
        hs = args.half
        r0, r1 = max(y0i - hs, 0), min(y0i + hs, H)
        c0, c1 = max(x0i - hs, 0), min(x0i + hs, W)
        for ci, (key, title, symm) in enumerate(cols):
            img = z[key][r0:r1, c0:c1]
            ax = axes[ri, ci]
            if symm:
                vmin, vmax = symmetric(img)
                cmap = "RdBu_r"
            elif key == "truth":
                vmin, vmax, cmap = 0, 1, "gray"
            else:
                fin = img[np.isfinite(img)]
                if fin.size:
                    vmin, vmax = np.percentile(fin, [1, 99])
                else:
                    vmin, vmax = 0, 1
                cmap = "viridis"
            ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=[c0, c1, r0, r1])
            ax.plot(float(xs[idx]), float(ys[idx]), "x",
                    color="yellow", markersize=8, markeredgewidth=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(title, fontsize=9)
            if ci == 0:
                tag = "AP-HIT" if recov[idx] else "AP-MISS"
                ax.set_ylabel(
                    f"{tag}\ni={int(idx)}\nL={float(lengths[idx]):.0f}px β={float(betas[idx]):.0f}°\nmag={float(mags[idx]):.2f}",
                    fontsize=8,
                )

    out = Path(args.out) if args.out else Path(args.shard).with_suffix(".zoom.png")
    fig.suptitle(f"shard zoom: {Path(args.shard).name} (n_inj={n_inj}, ap_recov={int(recov.sum())}/{n_inj})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
