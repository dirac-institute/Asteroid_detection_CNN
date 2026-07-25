#!/usr/bin/env python3
"""Pixel-level miss audit: v2_D segmentation response to injected trails at truth locations.

Motivation (SLOWBAND_AUDIT.md): the slow band (1-2 deg/day) is 64% stage-1 loss, but the
catalog can't distinguish "net saw nothing" from "response below threshold", and length_raw
(= seg-footprint extent) is contrast-dependent, so the MF_LEN de-bias does not transfer
between datasets. This probe measures both directly on LIVE DM-53195 diffims (run_dev
manifests): inject a controlled SNR x length grid (same add_trails injector as all evals),
run the ACTIVE pipeline (v2_D) exactly as production (predict_panel_overlap_3ch_full ->
panel_to_catalog_rows, rl=zeros), and record per injected trail:

  nominal:  snr_target, trail_length, mag, beta
  realized: panel_sigma, peak_sig (max injected-flux pixel / sigma), flux_tot,
            mfsnr_true (flux on thin line / sigma*sqrt(n_line))
  response: pmax (max seg prob in 5px-dilated truth line), pmean_thin, frac_thin_ge05
  catalog:  matched det (10px): score, nn_pmax, length (de-biased), length_raw, mf_snr, dist

Run with a LOW cnn_thr (default 0.01, gate_pmax=0) so sub-threshold candidates are captured;
production visibility is re-derived offline (score >= sidecar thr & nn_pmax >= 0.10).

Usage (GPU node):
  python ADCNN/pipelines/heliolinc/probe_seg_response.py --panels 150 --out outputs/runs/run_probe_seg
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ADCNN_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
OUTPUTS = os.environ.get("ADCNN_OUTPUTS") or os.path.join(REPO, "outputs")

PIXSCALE = 0.2
PSF_FWHM_PX = 3.77
EDGE = 60
MIN_SEP = 100.0
L_GRID = [6.25, 8.0, 10.0, 12.5, 16.0, 25.0, 40.0]
SNR_GRID = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0]


def _mag_for_snr(snr, m5, trail_px):
    """Same SNR->mag model as sim_orbits / build_ft_dataset."""
    dil = np.sqrt(np.maximum(trail_px, PSF_FWHM_PX) / PSF_FWHM_PX)
    return m5 - 2.5 * np.log10(np.maximum(snr, 1e-3) * dil / 5.0)


def _place(rng, n):
    """Rejection-sample n anchor points with MIN_SEP spacing (LSSTCam 4072x4000)."""
    pts = []
    for _ in range(n * 40):
        p = (rng.uniform(EDGE, 4072 - EDGE), rng.uniform(EDGE, 4000 - EDGE))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > MIN_SEP ** 2 for q in pts):
            pts.append(p)
            if len(pts) == n:
                break
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=f"{OUTPUTS}/runs/run_dev", help="manifests with LIVE fits paths")
    ap.add_argument("--out", default=f"{OUTPUTS}/runs/run_probe_seg")
    ap.add_argument("--panels", type=int, default=150)
    ap.add_argument("--trails-per-panel", type=int, default=24)
    ap.add_argument("--m5", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cnn-thr", type=float, default=0.01)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    a = ap.parse_args()

    import torch
    from scipy.spatial import cKDTree
    from ADCNN.config import ACTIVE as PIPE
    from ADCNN.inference.diffim_io import open_diffim
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.catalog import _worker_init, _worker, InferenceConfig
    from ADCNN.inference import catalog as _cat
    from ADCNN.data.preprocessing import diffim_mad_sigma
    from ADCNN.pipelines.heliolinc.inject_trails import add_trails
    from ADCNN.utils.helpers import draw_one_line

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seg_ckpt = str(PIPE.seg_model); cnn_pt = str(PIPE.cnn_model)
    pname = getattr(PIPE, "name", "?")
    print(f"[probe] pipeline={pname} seg={seg_ckpt} cnn={cnn_pt} device={dev}", flush=True)
    model = torch.jit.load(seg_ckpt, map_location=dev).eval()
    _worker_init(cnn_pt, InferenceConfig(cnn_thr=a.cnn_thr, gate_pmax=0.0))

    mpaths = sorted(glob.glob(f"{a.run}/manifest_*.csv")) or sorted(glob.glob(f"{a.run}/manifest.csv"))
    mans = pd.concat([pd.read_csv(p) for p in mpaths], ignore_index=True)
    rng = np.random.default_rng(a.seed)
    mans = mans.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    os.makedirs(a.out, exist_ok=True)
    out_csv = f"{a.out}/probe_trails.csv"
    done_panels = 0
    rows_out = []
    wrote_header = False

    for _, mrow in mans.iterrows():
        if done_panels >= a.panels:
            break
        fp = mrow.fits_path
        if not str(fp).startswith("s3://") and not os.path.exists(fp):
            continue
        try:
            with open_diffim(fp, memmap=False) as hd:
                img0 = np.nan_to_num(hd[1].data.astype(np.float32))
        except Exception as e:
            print(f"[probe] skip unreadable {fp}: {e}", flush=True)
            continue
        H, W = img0.shape
        sigma = float(diffim_mad_sigma(img0))
        n_tr = a.trails_per_panel
        pts = _place(rng, n_tr)
        if len(pts) < n_tr:
            n_tr = len(pts)
        tr = []
        for i in range(n_tr):
            L = float(rng.choice(L_GRID)); snr = float(rng.choice(SNR_GRID))
            tr.append(dict(x=pts[i][0], y=pts[i][1], beta=float(rng.uniform(0, 360)),
                           trail_length=L, snr_target=snr,
                           mag=float(np.clip(_mag_for_snr(snr, a.m5, L), 16.0, 28.0))))
        img = add_trails(img0.copy(), tr, seed=int(rng.integers(1 << 31)))
        diffmap = img - img0
        rl = np.zeros(img.shape, dtype=np.uint16)         # production ch2 (detect_night convention)
        with torch.no_grad():
            prob, _sin, _cos, agg = predict_panel_overlap_3ch_full(model, img, rl, device=dev)
        cand = _worker((0, prob, img, agg, rl))
        ct = cKDTree(cand[["x", "y"]].to_numpy()) if cand is not None and len(cand) else None

        for t in tr:
            thin = np.zeros((H, W), np.uint8)
            draw_one_line(thin, (t["x"], t["y"]), t["beta"], t["trail_length"],
                          true_value=1, line_thickness=2)
            wide = np.zeros((H, W), np.uint8)
            draw_one_line(wide, (t["x"], t["y"]), t["beta"], t["trail_length"],
                          true_value=1, line_thickness=5)
            mthin = thin.astype(bool); mwide = wide.astype(bool)
            n_line = int(mthin.sum())
            flux = float(diffmap[mwide].sum())
            peak_sig = float(diffmap[mwide].max() / sigma) if mwide.any() else np.nan
            mfsnr_true = flux / (sigma * np.sqrt(max(n_line, 1)))
            pm = prob[mwide]; pt_ = prob[mthin]
            rec = dict(visit=int(mrow.visit), detector=int(mrow.detector),
                       band=str(mrow.get("band", "?")), panel_sigma=sigma,
                       x=t["x"], y=t["y"], beta=t["beta"], trail_length=t["trail_length"],
                       snr_target=t["snr_target"], mag=t["mag"],
                       flux_tot=flux, peak_sig=peak_sig, mfsnr_true=mfsnr_true, n_line=n_line,
                       pmax=float(pm.max()) if pm.size else 0.0,
                       pmean_thin=float(pt_.mean()) if pt_.size else 0.0,
                       frac_thin_ge05=float((pt_ >= 0.5).mean()) if pt_.size else 0.0,
                       det=False, det_dist=np.nan, score=np.nan, nn_pmax=np.nan,
                       length=np.nan, length_raw=np.nan, mf_snr=np.nan)
            if ct is not None:
                dist, idx = ct.query([t["x"], t["y"]],
                                     distance_upper_bound=max(10.0, t["trail_length"] / 2))
                if np.isfinite(dist):
                    c = cand.iloc[int(idx)]
                    rec.update(det=True, det_dist=float(dist), score=float(c.score),
                               nn_pmax=float(c.nn_pmax), length=float(c.length),
                               length_raw=float(c.get("length_raw", np.nan)),
                               mf_snr=float(c.mf_snr))
            rows_out.append(rec)
        done_panels += 1
        if done_panels % 10 == 0 or done_panels == a.panels:
            pd.DataFrame(rows_out).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
            wrote_header = True; rows_out = []
            print(f"[probe] {done_panels}/{a.panels} panels", flush=True)

    if rows_out:
        pd.DataFrame(rows_out).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
    with open(f"{a.out}/probe_meta.json", "w") as f:
        json.dump(dict(run=a.run, panels=done_panels, trails_per_panel=a.trails_per_panel,
                       m5=a.m5, seed=a.seed, cnn_thr=a.cnn_thr, pipeline=str(pname),
                       seg_model=seg_ckpt, cnn_model=cnn_pt,
                       L_grid=L_GRID, snr_grid=SNR_GRID), f, indent=2)
    print(f"[probe] DONE {done_panels} panels -> {out_csv}", flush=True)
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
