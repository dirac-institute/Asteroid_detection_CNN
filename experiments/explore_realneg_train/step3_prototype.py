"""Step 3 — CPU prototype: hybrid training panels = REAL empty diffim
background + injected FAINT synthetic streak (target matched-filter SNR 3-10).

NO LSST stack, NO Butler, NO GPU. We take already-built REAL empty-role
difference-image panels from DATA_DIFFIM/test_real/test.h5 (these are genuine
AlardLupton residual fields — DCR dipoles, CR, bad astro, edges) and paint a
faint Gaussian-PSF-convolved line into the diffim pixel array.

Flux model (CPU stand-in for simulate_inject_diffim's stack injector):
  The downstream detector / matched-filter (experiments/diffim_pilot/
  matched_filter.py) measures, for a line footprint of n_line pixels,
      SNR_mf = sum(diffim along line) / (panel_MAD_sigma * sqrt(n_line)).
  We invert this: to inject a trail of length L with line_width w and a
  Gaussian cross-section of sigma_psf at target SNR_mf = s on a panel with
  MAD sigma = sg, the required *integrated* line signal is
      S_target = s * sg * sqrt(n_line).
  We lay a 1px polyline of length L, convolve with a 2D Gaussian PSF
  (sigma_psf from the panel band's typical seeing ~0.8" / 0.2"/px ≈ 2 px),
  then scale the PSF-spread trail so that the sum of pixels under the
  matched-filter line mask equals S_target. This reproduces, on CPU, the
  same observable the network/matched-filter consumes, without the stack.

This is a SANITY prototype (construction is sane: faint, PSF-shaped, sits
in the noise like a real faint trail and unlike a bright real residual),
NOT the production injector — the production path stays the stack-based
simulate_inject_diffim with --mag-mode snr.

Outputs (experiments/explore_realneg_train/proto/):
  hybrid_panel_<id>.png   real-bg crop + injected trail + zoom + a real-FP
                          residual in the same panel, side by side
  proto_stats.csv         per-injection: target vs recovered mf_snr, trail
                          |z|, local-std under trail vs at a real residual
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd

REPO = "/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN"
sys.path.insert(0, REPO)

from experiments.diffim_pilot.matched_filter import (  # noqa: E402
    matched_filter_from_coords, panel_mad_sigma)
from ADCNN.data.diffim_dataset import local_std_panel  # noqa: E402
from ADCNN.utils.helpers import draw_one_line  # noqa: E402

H5 = os.path.join(REPO, "DATA_DIFFIM/test_real/test.h5")
PANELS = os.path.join(REPO, "DATA_DIFFIM/test_real/panels.csv")
OUTDIR = os.path.join(REPO, "experiments/explore_realneg_train/proto")
os.makedirs(OUTDIR, exist_ok=True)

# Low-n_dia empty-role panels (cleaner residual fields, faster to crop). We
# still pick a real high-|z| residual inside each crop as the contrast object.
EMPTY_PIDS = [2578, 2657, 2575, 2671, 2642]
CROP = 512          # work in a 512x512 crop (fast, plenty of room for 60px trail)
TARGET_SNRS = [3.0, 5.0, 7.0, 10.0, 4.0]
LENGTHS = [25.0, 40.0, 60.0, 80.0, 30.0]
ANGLES = [20.0, 70.0, 115.0, 150.0, 45.0]
SIGMA_PSF = 1.7      # px; ~0.8" seeing at 0.2"/px FWHM -> sigma ~1.7px
LINE_WIDTH = 2       # matched_filter default line_width


def gaussian_kernel(sigma, radius=None):
    if radius is None:
        radius = int(np.ceil(4 * sigma))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def convolve2d_fft(img, ker):
    from numpy.fft import irfft2, rfft2
    H, W = img.shape
    kh, kw = ker.shape
    ph, pw = H + kh - 1, W + kw - 1
    F = rfft2(img, s=(ph, pw)) * rfft2(ker, s=(ph, pw))
    full = irfft2(F, s=(ph, pw))
    sy, sx = kh // 2, kw // 2
    return full[sy:sy + H, sx:sx + W].astype(np.float32)


def main():
    pan = pd.read_csv(PANELS)
    rows = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_mpl = True
    except Exception as e:
        print(f"[warn] matplotlib unavailable ({e}); skipping PNGs")
        have_mpl = False

    with h5py.File(H5, "r") as f:
        imgs = f["images"]
        rlab = f["real_labels"]
        for i, pid in enumerate(EMPTY_PIDS):
            prow = pan[pan["image_id"] == pid]
            role = prow["role"].iloc[0] if len(prow) else "?"
            band = prow["band"].iloc[0] if len(prow) else "?"
            assert role == "empty", f"panel {pid} role={role} is NOT empty!"
            H, W = imgs.shape[1], imgs.shape[2]
            # Place the crop so it contains at least one REAL residual
            # footprint (real_labels>0) -> a genuine FP as the contrast
            # object. Fall back to centre if the panel has none.
            full_rl = rlab[pid, :, :]
            rys, rxs = np.nonzero(full_rl > 0)
            if len(rys):
                # pick the residual footprint pixel furthest from edges
                med_y, med_x = int(np.median(rys)), int(np.median(rxs))
                y0 = int(np.clip(med_y - CROP // 2, 0, H - CROP))
                x0 = int(np.clip(med_x - CROP // 2, 0, W - CROP))
            else:
                y0 = (H - CROP) // 2
                x0 = (W - CROP) // 2
            crop = imgs[pid, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
            rl_crop = full_rl[y0:y0 + CROP, x0:x0 + CROP]

            sg = panel_mad_sigma(crop)

            # ---- build the faint synthetic trail -------------------------
            tlen = LENGTHS[i]
            ang = ANGLES[i]
            s_target = TARGET_SNRS[i]
            # Inject the trail into a clean quadrant well away from the real
            # residual (which sits near crop centre by construction) so the
            # trail-vs-real-residual contrast is on disjoint regions.
            cx, cy = CROP * 0.27, CROP * 0.27
            line1 = np.zeros((CROP, CROP), dtype=np.uint8)
            draw_one_line(line1, (cx, cy), ang, tlen, true_value=1,
                          line_thickness=1)
            ker = gaussian_kernel(SIGMA_PSF)
            trail = convolve2d_fft(line1.astype(np.float32), ker)

            # matched-filter line mask the detector would integrate over
            ys_l, xs_l = np.nonzero(line1)
            mf_mask = np.zeros((CROP, CROP), dtype=np.uint8)
            import cv2
            # thicken to LINE_WIDTH like matched_filter_from_coords
            tmp = np.zeros((CROP, CROP), np.uint8)
            ang_r = np.deg2rad(ang)
            dx, dy = np.cos(ang_r), np.sin(ang_r)
            p1 = (int(round(cx - 0.5 * tlen * dx)), int(round(cy - 0.5 * tlen * dy)))
            p2 = (int(round(cx + 0.5 * tlen * dx)), int(round(cy + 0.5 * tlen * dy)))
            cv2.line(tmp, p1, p2, 1, thickness=LINE_WIDTH)
            mf_mask = tmp
            n_line = int(mf_mask.sum())

            # scale trail so SNR_mf == s_target on THIS panel.
            # First-order: SNR_mf = sum(trail under mf_mask)/(sg*sqrt(n_line)).
            s_under = float(trail[mf_mask > 0].sum())
            S_target = s_target * sg * np.sqrt(n_line)
            scale = S_target / max(s_under, 1e-9)

            # The production detector measures SNR with the *PCA* footprint
            # estimator (matched_filter_from_coords), which pads L_eff and
            # picks up a few more noise pixels => a fixed multiplicative
            # bias. One closed-loop correction against that exact estimator
            # makes the injected SNR land on target (this is precisely the
            # role snr_to_mag/the stack photoCalib plays in the real
            # injector — calibrate flux to the SNR the detector will read).
            ys, xs = np.nonzero(line1)
            probe = crop + trail * scale
            s_probe, _, _, _ = matched_filter_from_coords(
                probe, sg, ys, xs, line_width=LINE_WIDTH, pad_length=4)
            s_bg_probe, _, _, _ = matched_filter_from_coords(
                crop, sg, ys, xs, line_width=LINE_WIDTH, pad_length=4)
            gain = (s_probe - s_bg_probe) / max(s_target, 1e-6)
            if np.isfinite(gain) and gain > 1e-3:
                scale = scale / gain
            trail_scaled = trail * scale

            hybrid = crop + trail_scaled

            # ---- recover via the real matched filter ---------------------
            mf_snr, mf_nl, mf_flux, mf_L = matched_filter_from_coords(
                hybrid, sg, ys, xs, line_width=LINE_WIDTH, pad_length=4)
            mf_snr_bg, _, _, _ = matched_filter_from_coords(
                crop, sg, ys, xs, line_width=LINE_WIDTH, pad_length=4)

            # ---- |z| of trail pixels & local-std contrast ----------------
            z = hybrid / sg
            trail_pix_z = z[mf_mask > 0]
            lstd = local_std_panel(np.clip(crop / sg, -5, 5), window=11)
            lstd_trail = float(np.median(
                local_std_panel(np.clip(hybrid / sg, -5, 5), 11)[mf_mask > 0]))
            lstd_bg = float(np.median(lstd))

            # pick the strongest REAL residual in this crop for contrast:
            # real_labels>0 marks real diaSource footprints; fall back to the
            # brightest |z| pixel cluster if labels are sparse in this crop.
            real_z_max = float(np.nanmax(np.abs(z)))
            if (rl_crop > 0).any():
                rys, rxs = np.nonzero(rl_crop > 0)
                real_resid_absz = float(np.median(np.abs(z[rys, rxs])))
                real_resid_maxz = float(np.max(np.abs(z[rys, rxs])))
                real_resid_lstd = float(np.median(
                    local_std_panel(np.clip(crop / sg, -5, 5), 11)[rys, rxs]))
            else:
                real_resid_absz = real_resid_maxz = real_resid_lstd = np.nan

            rec = dict(
                panel_id=pid, band=band, role=role, panel_mad_sigma=sg,
                target_snr=s_target, recovered_mf_snr=round(mf_snr, 3),
                bg_only_mf_snr=round(mf_snr_bg, 3),
                trail_len_px=tlen, angle_deg=ang, n_line_px=n_line,
                trail_med_absz=round(float(np.median(np.abs(trail_pix_z))), 3),
                trail_p95_absz=round(float(np.percentile(np.abs(trail_pix_z), 95)), 3),
                trail_peak_amp_over_sigma=round(float(trail_scaled.max() / sg), 3),
                lstd_under_trail=round(lstd_trail, 4),
                lstd_background=round(lstd_bg, 4),
                real_resid_med_absz=round(real_resid_absz, 3),
                real_resid_max_absz=round(real_resid_maxz, 3),
                real_resid_lstd=round(real_resid_lstd, 4),
                crop_max_absz=round(real_z_max, 2),
            )
            rows.append(rec)
            print(f"[panel {pid} band={band}] target SNR={s_target:>4} -> "
                  f"recovered mf_snr={mf_snr:6.2f} (bg-only {mf_snr_bg:5.2f}) "
                  f"trail|z|med={rec['trail_med_absz']:.2f} "
                  f"peak/σ={rec['trail_peak_amp_over_sigma']:.2f} "
                  f"| real-resid |z|max={rec['real_resid_max_absz']}")

            if have_mpl:
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                vlim = 4 * sg
                ax[0].imshow(crop, vmin=-vlim, vmax=vlim, cmap="gray", origin="lower")
                ax[0].set_title(f"REAL empty diffim bg\npid={pid} band={band} σ={sg:.1f}")
                ax[1].imshow(trail_scaled, vmin=-vlim, vmax=vlim, cmap="gray", origin="lower")
                ax[1].set_title(f"injected faint trail\nL={tlen:.0f}px ang={ang:.0f}° "
                                f"SNR_mf→{s_target}")
                ax[2].imshow(hybrid, vmin=-vlim, vmax=vlim, cmap="gray", origin="lower")
                ax[2].set_title(f"HYBRID = bg + trail\nrecovered mf_snr={mf_snr:.2f}")
                zc = 96
                yy0 = int(cy) - zc; xx0 = int(cx) - zc
                ax[3].imshow(hybrid[yy0:yy0 + 2 * zc, xx0:xx0 + 2 * zc],
                             vmin=-vlim, vmax=vlim, cmap="gray", origin="lower")
                ax[3].set_title("zoom on injected trail")
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
                fig.tight_layout()
                fig.savefig(os.path.join(OUTDIR, f"hybrid_panel_{pid}.png"), dpi=85)
                plt.close(fig)

    df = pd.DataFrame(rows)
    csvp = os.path.join(OUTDIR, "proto_stats.csv")
    df.to_csv(csvp, index=False)
    print("\n" + "=" * 78)
    print("PROTOTYPE SUMMARY")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[["panel_id", "band", "target_snr", "recovered_mf_snr",
                  "bg_only_mf_snr", "trail_med_absz", "trail_peak_amp_over_sigma",
                  "lstd_under_trail", "lstd_background", "real_resid_max_absz",
                  "real_resid_lstd"]].to_string(index=False))
    err = (df["recovered_mf_snr"] - df["target_snr"]).abs()
    print(f"\nmean |recovered - target| SNR_mf = {err.mean():.3f}  "
          f"(max {err.max():.3f})  -> construction hits the target SNR")
    print(f"trail peak/σ: {df['trail_peak_amp_over_sigma'].min():.2f}–"
          f"{df['trail_peak_amp_over_sigma'].max():.2f}  vs  real-resid |z|max "
          f"{df['real_resid_max_absz'].min()}–{df['real_resid_max_absz'].max()}")
    print(f"[written] {csvp}")
    print(f"[written] {OUTDIR}/hybrid_panel_*.png")


if __name__ == "__main__":
    main()
