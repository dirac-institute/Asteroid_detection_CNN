"""Per-trail zoom inspection for the diffim proof NPZ.

Reads a proof NPZ and produces a zoomed-in visualization around each injected
trail so that you can visually confirm each injected source actually appears
in the diffim. For each injected trail, draws a horizontal strip of
(science_clean, science_injected, template, diffim_clean, diffim_injected,
empirical, truth) all stretched in the LOCAL pixel statistics of a crop
centered on the injection.

Usage:
    python zoom_inspect.py --npz path/to/proof_vV_dD.npz [--half-size 80]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def local_stretch_symmetric(arr: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> tuple[float, float]:
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(good, [lo_q, hi_q])
    vmax = max(abs(lo), abs(hi))
    return -vmax, vmax


def local_stretch(arr: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> tuple[float, float]:
    good = arr[np.isfinite(arr)]
    if good.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(good, [lo_q, hi_q])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--half-size", type=int, default=80,
                    help="Half-size in pixels of each zoom crop.")
    ap.add_argument("--out", default=None,
                    help="Output PNG; defaults next to the NPZ.")
    args = ap.parse_args()

    npz = np.load(args.npz)
    H, W = npz["diffim_injected"].shape
    xs = npz["injection_x_hint"]
    ys = npz["injection_y_hint"]
    betas = npz["injection_beta"]
    lengths = npz["injection_trail_length"]
    mags = npz["injection_mag"]
    n = len(xs)
    hs = args.half_size

    panels = [
        ("science_clean", "Sci CLEAN", "viridis", False),
        ("science_injected", "Sci INJ", "viridis", False),
        ("template_warped", "Template", "viridis", False),
        ("diffim_clean", "Diffim CLEAN", "RdBu_r", True),
        ("diffim_injected", "Diffim INJ", "RdBu_r", True),
        ("empirical_injected_only", "Empirical (inj-clean)", "RdBu_r", True),
        ("truth_mask", "Truth", "gray", False),
    ]

    fig, axes = plt.subplots(n, len(panels), figsize=(2.2 * len(panels), 2.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row, (x0, y0, beta, L, mg) in enumerate(zip(xs, ys, betas, lengths, mags)):
        x0i, y0i = int(round(float(x0))), int(round(float(y0)))
        r0, r1 = max(y0i - hs, 0), min(y0i + hs, H)
        c0, c1 = max(x0i - hs, 0), min(x0i + hs, W)

        for col, (key, title, cmap, symm) in enumerate(panels):
            img = npz[key][r0:r1, c0:c1]
            ax = axes[row, col]
            if symm:
                vmin, vmax = local_stretch_symmetric(img)
            elif key == "truth_mask":
                vmin, vmax = 0, 1
            else:
                vmin, vmax = local_stretch(img)
            im = ax.imshow(img, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[c0, c1, r0, r1])
            ax.plot(float(x0), float(y0), "x", color="yellow", markersize=10, markeredgewidth=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"trail {row}\nx,y=({x0i},{y0i})\nL={float(L):.0f}px β={float(beta):.0f}°\nmag={float(mg):.1f}",
                    fontsize=8,
                )
        # Colorbar for the Empirical panel in each row
        cbar_ax = axes[row, -2]  # empirical column
        fig.colorbar(cbar_ax.images[0], ax=axes[row, -2], fraction=0.046)

    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".zoom.png")
    fig.suptitle(f"Per-trail zoom: {Path(args.npz).name}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
