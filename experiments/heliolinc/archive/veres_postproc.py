"""Investigate the LSST Veres trailed-source fit as a POST-RF false-positive filter.

Hypothesis (user): a Veres trailed-PSF fit, run at each ADCNN+RF detection, AGREES with the
ADCNN-measured trail params for true positives (real trails) and DISAGREES for false positives
(noise / point-source artefacts the RF let through). If so, Veres-vs-ADCNN agreement is a cheap
post-processing discriminator.

We drive `VeresModel` directly (seeded with the ADCNN params) — NOT the Naive→Veres plugin chain,
which dies when SdssShape fails on long trails (Naive sets the shape flag; VeresPlugin then crashes
on `setValue(... NO_NAIVE)`). VeresModel re-fits robustly from any seed (validated), so we avoid that
path entirely. Runs on CPU in the lsst_distrib env (no GPU): reconstruct an Exposure per panel from
the sim diffim (image + Gaussian PSF + MAD variance), fit each detection, label TP/FP via truth
trail overlap, and report the discriminator separation.

    setup lsst_distrib
    python experiments/heliolinc/veres_postproc.py --rf-thr 0.5
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as sciOpt
import h5py

import lsst.afw.image as afwImage
from lsst.meas.algorithms import DoubleGaussianPsf
from lsst.meas.extensions.trailedSources import VeresModel
import lsst.geom as geom

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
PIXSCALE = 0.2
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"


def get_real_psfs(truth, pids):
    """Fetch the real per-panel PVI PSF from the Butler (the diffim PSF ≈ science PSF after
    AlardLupton matching). Returns {image_id: lsst.afw.detection.Psf}. Falls back silently."""
    from lsst.daf.butler import Butler
    b = Butler("dp2_prep", collections=[STAGE2])
    vd = truth.drop_duplicates("image_id").set_index("image_id")[["visit", "detector"]]
    psfs = {}
    for pid in pids:
        if pid not in vd.index:
            continue
        try:
            psf = b.get("preliminary_visit_image.psf",
                        dataId={"instrument": "LSSTCam", "visit": int(vd.loc[pid].visit),
                                "detector": int(vd.loc[pid].detector)})
            psfs[pid] = psf
        except Exception as e:
            print(f"  PSF fetch fail pid={pid}: {e}", flush=True)
    return psfs


def mad_sigma(a):
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def point_to_segments(px, py, segs):
    """Min distance from point (px,py) to each truth segment [[x0,y0],[x1,y1]] (N,2,2)."""
    a = segs[:, 0, :]; b = segs[:, 1, :]
    ab = b - a; ap = np.stack([px - a[:, 0], py - a[:, 1]], -1)
    denom = (ab * ab).sum(1)
    t = np.clip((ap * ab).sum(1) / np.where(denom > 1e-9, denom, 1.0), 0, 1)
    proj = a + t[:, None] * ab
    return np.hypot(px - proj[:, 0], py - proj[:, 1])


def truth_segments(truth_panel):
    th = np.radians(truth_panel.beta.to_numpy()); half = 0.5 * truth_panel.trail_length.to_numpy()
    x = truth_panel.x.to_numpy(); y = truth_panel.y.to_numpy()
    dx = np.cos(th) * half; dy = np.sin(th) * half
    return np.stack([np.stack([x - dx, y - dy], -1), np.stack([x + dx, y + dy], -1)], 1)


def ang_resid_deg(a_deg, b_deg):
    d = (a_deg - b_deg) % 180.0
    return d - 180.0 if d > 90 else d


def fit_veres(exp, x, y, flux, length, beta_deg, psf_sig):
    """VeresModel fit on a cutout around (x,y), seeded with ADCNN params. Returns dict."""
    half = int(max(length, 6) / 2 + 6 * psf_sig + 6)
    bb = geom.Box2I(geom.Point2I(int(x) - half, int(y) - half), geom.Extent2I(2 * half + 1, 2 * half + 1))
    bb.clip(exp.getBBox())
    if bb.getWidth() < 8 or bb.getHeight() < 8:
        return None
    cutout = exp.Factory(exp, bb)
    model = VeresModel(cutout)
    seed = np.array([float(x), float(y), float(flux if np.isfinite(flux) and flux > 0 else 1000.0),
                     float(max(length, 1.0)), float(np.radians(beta_deg))])
    try:
        r = sciOpt.minimize(model, seed, method="Nelder-Mead",
                            options=dict(maxiter=2000, xatol=1e-2, fatol=1e-2))
    except Exception:
        return None
    n = cutout.image.array.size
    return dict(L_fit=float(r.x[3]), th_fit=float(np.degrees(r.x[4]) % 180.0),
                flux_fit=float(r.x[2]), rChiSq=float(r.fun / max(n - 6, 1)), ok=bool(r.success))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", default=str(REPO / "Evaluation/catalogs_thr0/test_5sigma_detections.csv"))
    ap.add_argument("--h5", default=str(REPO / "DATA_DIFFIM/test_5sigma/test.h5"))
    ap.add_argument("--truth", default=str(REPO / "DATA_DIFFIM/test_5sigma/test.csv"))
    ap.add_argument("--rf-thr", type=float, default=0.5, help="post-RF operating point")
    ap.add_argument("--psf-fwhm-arcsec", type=float, default=0.8)
    ap.add_argument("--real-psf", action="store_true", help="fetch the real per-panel PVI PSF from the Butler")
    ap.add_argument("--tp-tol-px", type=float, default=5.0, help="centroid<->truth-trail dist for TP")
    ap.add_argument("--panels", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=str(REPO / "experiments/heliolinc/veres_postproc.csv"))
    a = ap.parse_args()

    psf_sig = a.psf_fwhm_arcsec / PIXSCALE / 2.355
    dets = pd.read_csv(a.dets)
    dets = dets[dets.score_rf >= a.rf_thr].copy()
    truth = pd.read_csv(a.truth)
    truth = truth[truth.source_type == "Trail"] if "source_type" in truth else truth
    pids = sorted(dets.image_id.unique())
    if a.panels:
        pids = pids[:a.panels]
    print(f"PSF sigma={psf_sig:.2f}px | {len(dets)} post-RF detections over {len(pids)} panels "
          f"(thr={a.rf_thr})", flush=True)

    real_psfs = get_real_psfs(truth, pids) if a.real_psf else {}
    if a.real_psf:
        print(f"fetched {len(real_psfs)}/{len(pids)} real PVI PSFs from Butler", flush=True)

    rows = []
    with h5py.File(a.h5, "r") as f:
        for pid in pids:
            img = f["images"][pid][:].astype(np.float32)
            sig = float(mad_sigma(img))
            exp = afwImage.ExposureF(img.shape[1], img.shape[0])
            exp.image.array[:] = img
            exp.variance.array[:] = sig ** 2
            if pid in real_psfs:
                exp.setPsf(real_psfs[pid])
                try:
                    panel_psf_sig = float(real_psfs[pid].computeShape(real_psfs[pid].getAveragePosition()).getDeterminantRadius())
                except Exception:
                    panel_psf_sig = psf_sig
            else:
                k = 2 * int(4 * psf_sig) + 1
                exp.setPsf(DoubleGaussianPsf(k, k, psf_sig))
                panel_psf_sig = psf_sig
            tp = truth[truth.image_id == pid]
            segs = truth_segments(tp) if len(tp) else np.empty((0, 2, 2))
            for _, d in dets[dets.image_id == pid].iterrows():
                res = fit_veres(exp, d.x, d.y, d.get("flux", np.nan), d.get("length", 10.0),
                                d.get("beta", 0.0), panel_psf_sig)
                if res is None:
                    continue
                truth_len = np.nan; is_tp = False
                if len(segs):
                    dd = point_to_segments(d.x, d.y, segs)
                    j = int(dd.argmin())
                    if dd[j] <= a.tp_tol_px:
                        is_tp = True
                        truth_len = float(tp.iloc[j].trail_length)
                res.update(image_id=int(pid), x=float(d.x), y=float(d.y),
                           adcnn_len=float(d.get("length", np.nan)), adcnn_beta=float(d.get("beta", np.nan)),
                           adcnn_flux=float(d.get("flux", np.nan)), score_rf=float(d.score_rf),
                           tp=is_tp, truth_len=truth_len)
                rows.append(res)
    df = pd.DataFrame(rows)
    df["dlen"] = (df.L_fit - df.adcnn_len).abs()
    df["dang"] = [abs(ang_resid_deg(r.th_fit, r.adcnn_beta)) for r in df.itertuples()]
    df["flux_ratio"] = df.flux_fit / df.adcnn_flux.replace(0, np.nan)
    df["flux_snr"] = df.flux_fit / df.L_fit.clip(lower=1).pow(0.5)  # rough; for relative comparison
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}\n", flush=True)

    tp, fp = df[df.tp], df[~df.tp]
    # FIT-QUALITY DIAGNOSTIC: does Veres (and ADCNN) recover the TRUE trail length on TPs?
    vt = (tp.L_fit - tp.truth_len).dropna(); at = (tp.adcnn_len - tp.truth_len).dropna()
    print(f"=== fit quality on TP (vs truth length) ===")
    print(f"  Veres L_fit - truth_len: med {vt.median():+.1f}px  MAD {(vt-vt.median()).abs().median():.1f}")
    print(f"  ADCNN len   - truth_len: med {at.median():+.1f}px  MAD {(at-at.median()).abs().median():.1f}")
    print(f"  (if Veres doesn't track truth, the PSF/variance approx is the problem, not the discriminator)\n")
    print(f"=== {len(tp)} TP vs {len(fp)} FP (post-RF, thr={a.rf_thr}) ===")
    def stat(col, lbl, fmt="{:7.2f}"):
        def q(s): return "  ".join(fmt.format(v) for v in (s.median(), s.quantile(.1), s.quantile(.9)))
        print(f"  {lbl:18s} TP[med/p10/p90]= {q(tp[col].dropna())}   |  FP= {q(fp[col].dropna())}")
    print("  (discriminators: smaller dlen/dang + flux agreement = trail-like)")
    stat("dlen", "|ΔL| px")
    stat("dang", "|Δangle| deg")
    stat("flux_fit", "Veres flux", "{:8.0f}")
    stat("flux_ratio", "flux_fit/adcnn")
    stat("L_fit", "Veres L px")
    stat("rChiSq", "rChiSq", "{:7.3f}")
    # simple separation: cut on |ΔL| <= thr ; report TP kept vs FP removed
    print("\n  cut: keep detections with |ΔL| <= T  (and Veres flux > 3·panel? no — relative):")
    for T in (3, 5, 8, 12):
        kt = (tp.dlen <= T).mean() * 100; kf = (fp.dlen <= T).mean() * 100
        print(f"    |ΔL|<= {T:2d}px:  TP kept {kt:5.1f}%   FP kept {kf:5.1f}%  (FP removed {100-kf:5.1f}%)")
    print("VERES POSTPROC DONE", flush=True)


if __name__ == "__main__":
    main()
