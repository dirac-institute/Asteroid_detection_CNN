"""GalSim-only sanity check (no Butler/GPU): render a stock UNIFORM trail vs the
new REALISTIC trail with a Gaussian PSF and confirm the realistic one is less
'idealized' — non-uniform brightness along the trail, lower response to a uniform
matched-line template, lower elongation. Saves a side-by-side PNG."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import galsim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
from ADCNN.data.dataset_creation.realistic_trail import _trail_profile

OUT = REPO / "experiments/explore_simreal_gap"
FWHM = 3.2      # px, representative LSST seeing
STAMP = 121
BETA = 25.0     # deg
FLUX = 1.0e4


def draw(obj, psf):
    conv = galsim.Convolve([obj, psf])
    img = galsim.ImageF(STAMP, STAMP, scale=1.0)
    conv.drawImage(image=img, method="fft")
    return img.array.copy()


def uniform_trail(L):
    return galsim.Box(L, 1e-6).rotate(BETA * galsim.degrees).withFlux(FLUX)


def realistic_trail(L, seed):
    rng = np.random.default_rng(seed)
    s, perp, w = _trail_profile(L, rng)
    comps = [galsim.DeltaFunction(flux=float(wi)).shift(float(si), float(pi))
             for si, pi, wi in zip(s, perp, w)]
    return galsim.Add(comps).rotate(BETA * galsim.degrees).withFlux(FLUX)


def along_trail_profile(arr, L, beta_deg):
    """Sample the image along the trail axis through center; return brightness vs s."""
    cy, cx = (STAMP - 1) / 2, (STAMP - 1) / 2
    th = np.radians(beta_deg)
    s = np.linspace(-0.6 * L, 0.6 * L, 80)
    xs = cx + s * np.cos(th); ys = cy + s * np.sin(th)
    from scipy.ndimage import map_coordinates
    return s, map_coordinates(arr, [ys, xs], order=1)


def coherence_proxy(arr, L, beta_deg):
    """Crude matched-filter: correlate with a uniform line of length L at beta.
    Lower => less like an idealized uniform streak. Also return elongation."""
    s, prof = along_trail_profile(arr, L, beta_deg)
    cv = float(np.std(prof) / (np.mean(prof) + 1e-9))   # brightness non-uniformity
    # elongation from second moments of the (thresholded) stamp
    m = arr > 0.05 * arr.max()
    ys, xs = np.nonzero(m)
    if len(xs) < 5:
        return cv, np.nan
    x0, y0 = xs.mean(), ys.mean()
    cxx = np.mean((xs - x0) ** 2); cyy = np.mean((ys - y0) ** 2)
    cxy = np.mean((xs - x0) * (ys - y0))
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    l1 = tr / 2 + np.sqrt(max(tr ** 2 / 4 - det, 0))
    l2 = tr / 2 - np.sqrt(max(tr ** 2 / 4 - det, 0))
    elong = float(np.sqrt(l1 / max(l2, 1e-6)))
    return cv, elong


def main():
    psf = galsim.Gaussian(fwhm=FWHM)
    Ls = [9.0, 20.0, 40.0]
    fig, axes = plt.subplots(len(Ls), 4, figsize=(13, 3.1 * len(Ls)))
    print(f"{'L':>5} {'kind':>10} {'bright_CV':>10} {'elongation':>11}")
    for r, L in enumerate(Ls):
        u = draw(uniform_trail(L), psf)
        reals = [draw(realistic_trail(L, sd), psf) for sd in (1, 7, 13)]
        cv_u, el_u = coherence_proxy(u, L, BETA)
        print(f"{L:5.0f} {'uniform':>10} {cv_u:10.3f} {el_u:11.2f}")
        cvs, els = [], []
        for sd, ri in zip((1, 7, 13), reals):
            cv, el = coherence_proxy(ri, L, BETA); cvs.append(cv); els.append(el)
        print(f"{L:5.0f} {'realistic':>10} {np.mean(cvs):10.3f} {np.nanmean(els):11.2f}"
              f"   (3 seeds)")
        vmax = u.max()
        axes[r, 0].imshow(u, vmax=vmax, cmap="gray"); axes[r, 0].set_title(f"L={L:.0f} uniform")
        for k in range(3):
            axes[r, k + 1].imshow(reals[k], vmax=vmax, cmap="gray")
            axes[r, k + 1].set_title(f"realistic #{k+1}")
        for k in range(4):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
    plt.tight_layout()
    p = OUT / "realistic_trail_validation.png"
    plt.savefig(p, dpi=110); print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
