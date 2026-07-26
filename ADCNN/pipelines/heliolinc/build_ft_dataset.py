#!/usr/bin/env python3
"""ADCNN v2 fine-tune dataset builder (ADCNN_V2_SPRINT.md, Phase 2 data).

Replicates the canonical training-data contract on retained DM-53195 diffims (PVIs are purged, so
the ExposureInjectTask+re-subtraction chain is impossible; trails are injected post-subtraction via
inject_trails.add_trails — the same injector the dev/blind alert evals use):

  H5: images  (N,H,W) float32  add_trails-injected diffim (raw nJy-scale, loader normalizes)
      masks   (N,H,W) bool     draw_one_line truth (cv2.LINE_8, thickness 2 ~= PSF/2)
      real_labels (N,H,W) uint16  SourceDetectionTask 5sigma footprints on the CLEAN diffim (ch2)
  CSV: image_id, injection_id, visit, detector, x, y, beta, trail_length, mag, SNR(=snr_target),
       source_type, physical_filter, stack_detection (5sigma hit on the INJECTED panel, drives the
       loader's stratified anchor sampling + variant D oversampling)

Stages (deterministic; per-field):
  catalog  (any env)   : subsample panels per field, draw dense per-panel trail catalogs
  detect   (stack env) : per panel -- clean-diffim footprint plane (uint16 npz) + injected-panel
                         5sigma stack_detection flags per trail
  assemble (asteroid_cnn): re-inject identically -> images; draw masks; merge real_labels; write
                         train.h5/train.csv (night 20250723 fields) + val.h5/val.csv (20250704)

Usage: build_ft_dataset.py --stage catalog --run run_dev --out run_ft [--panels-train 1500 ...]
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
OUTPUTS = os.environ.get("ADCNN_OUTPUTS") or os.path.join(REPO, "outputs")  # all runtime OUTPUT goes here

TRAIN_NIGHT = 20250723
VAL_NIGHT = 20250704
PIXSCALE = 0.2
PSF_FWHM_PX = 3.77
LINE_THICK = 2            # int(psf_width/2) convention of the canonical mask (helpers.draw_one_line)
EDGE = 40                 # px margin for trail anchors


def _mag_for_snr(snr, m5, trail_px):
    """Same SNR->mag model as sim_orbits: point-source m5 + trail dilution sqrt(L/FWHM)."""
    dil = np.sqrt(np.maximum(trail_px, PSF_FWHM_PX) / PSF_FWHM_PX)
    return m5 - 2.5 * np.log10(np.maximum(snr, 1e-3) * dil / 5.0)


def stage_catalog(a):
    rng = np.random.default_rng(a.seed)
    fields = pd.read_csv(f"{a.run}/fields.csv")
    os.makedirs(a.out, exist_ok=True)
    excl = set()
    if a.exclude_catalog and os.path.exists(a.exclude_catalog):
        e = pd.read_csv(a.exclude_catalog)
        excl = set(zip(e.visit.astype(int), e.detector.astype(int)))
        print(f"[ft-catalog] excluding {len(excl)} panels already used (leakage-clean) from {a.exclude_catalog}")
    rows_all = []
    for split, night, n_panels in [("train", TRAIN_NIGHT, a.panels_train), ("val", VAL_NIGHT, a.panels_val)]:
        ks = fields[fields.night == night].field.astype(int).tolist()
        mans = pd.concat([pd.read_csv(f"{a.run}/manifest_{k}.csv").assign(field=k) for k in ks],
                         ignore_index=True)
        if excl:
            mans = mans[~mans.apply(lambda r: (int(r.visit), int(r.detector)) in excl, axis=1)]
        sel = mans.sample(n=min(n_panels, len(mans)), random_state=int(rng.integers(1 << 31)))
        for pid, r in enumerate(sel.itertuples()):
            n_tr = int(rng.integers(a.trails_min, a.trails_max + 1))
            x = rng.uniform(EDGE, 4072 - EDGE, n_tr)
            y = rng.uniform(EDGE, 4000 - EDGE, n_tr)
            beta = rng.uniform(0, 360, n_tr)
            length = np.exp(rng.uniform(np.log(a.len_min), np.log(a.len_max), n_tr))
            snr = np.exp(rng.uniform(np.log(a.snr_min), np.log(a.snr_max), n_tr))
            mag = np.clip(_mag_for_snr(snr, a.m5, length), 18.0, 27.0)
            for i in range(n_tr):
                rows_all.append(dict(split=split, field=int(r.field), visit=int(r.visit),
                                     detector=int(r.detector), fits_path=r.fits_path,
                                     wcs_json_present=hasattr(r, "wcs_json"),
                                     injection_id=i, x=float(x[i]), y=float(y[i]),
                                     beta=float(beta[i]), trail_length=float(length[i]),
                                     mag=float(mag[i]), SNR=float(snr[i]), source_type="Trail"))
    cat = pd.DataFrame(rows_all)
    cat.to_csv(f"{a.out}/ft_catalog.csv", index=False)
    pan = cat.groupby(["split", "visit", "detector"]).size()
    print(f"[ft-catalog] {len(cat)} trails over {len(pan)} panels "
          f"(train {len(cat[cat.split=='train'])}, val {len(cat[cat.split=='val'])}) -> {a.out}/ft_catalog.csv")
    print("FT_CATALOG_DONE")


def _inject_rows(g):
    """ft_catalog rows -> inject_trails.add_trails row dicts."""
    return [dict(x=r.x, y=r.y, trail_length=r.trail_length, beta=r.beta, mag=r.mag)
            for r in g.itertuples()]


def stage_detect(a):
    """Stack env: per panel, clean-diffim 5sigma footprint plane + injected-panel stack hits."""
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable
    from lsst.meas.algorithms import SourceDetectionTask
    from scipy.spatial import cKDTree
    from ADCNN.pipelines.heliolinc.inject_trails import add_trails
    cat = pd.read_csv(f"{a.out}/ft_catalog.csv")
    os.makedirs(f"{a.out}/reallabels", exist_ok=True)
    cfg = SourceDetectionTask.ConfigClass(); cfg.thresholdValue = 5.0; cfg.reEstimateBackground = False
    schema = afwTable.SourceTable.makeMinimalSchema()
    task = SourceDetectionTask(config=cfg, schema=schema)
    hits_col = np.zeros(len(cat), bool)
    panels = list(cat.groupby(["visit", "detector"]))
    for n, ((v, det), g) in enumerate(panels):
        out_npz = f"{a.out}/reallabels/{v}_{det}.npz"
        exp = afwImage.ExposureF.readFits(g.fits_path.iloc[0])
        if not os.path.exists(out_npz):
            tbl = afwTable.SourceTable.make(schema)
            res = task.run(tbl, exp.clone())
            lab = np.zeros(exp.image.array.shape, np.uint16)
            Hh, Ww = lab.shape
            x0p, y0p = exp.getXY0().getX(), exp.getXY0().getY()
            for i, src in enumerate(res.sources, start=1):
                fp = src.getFootprint()
                if fp is None:
                    continue
                val = int(min(i, 65535))
                for sp in fp.spans:                      # pure-numpy span fill (PARENT->array coords)
                    yy = sp.getY() - y0p
                    if 0 <= yy < Hh:
                        xa = max(sp.getX0() - x0p, 0); xb = min(sp.getX1() - x0p, Ww - 1)
                        if xb >= xa:
                            lab[yy, xa:xb + 1] = val
            np.savez_compressed(out_npz, real_labels=lab)
        # injected-panel stack detection -> per-trail hit flags
        expi = exp.clone()
        expi.image.array[:] = add_trails(np.array(expi.image.array, copy=True), _inject_rows(g))
        tbl = afwTable.SourceTable.make(schema)
        res = task.run(tbl, expi)
        pts = []
        for src in res.sources:
            fp = src.getFootprint()
            if fp is not None and fp.getPeaks():
                pk = fp.getPeaks()[0]; pts.append((pk.getFx(), pk.getFy()))
        if pts:
            d, _ = cKDTree(np.array(pts)).query(g[["x", "y"]].to_numpy(), distance_upper_bound=10.0)
            hits_col[g.index.to_numpy()] = np.isfinite(d)
        if (n + 1) % 100 == 0:
            print(f"  [ft-detect] {n+1}/{len(panels)} panels", flush=True)
    cat["stack_detection"] = hits_col
    cat.to_csv(f"{a.out}/ft_catalog.csv", index=False)
    print(f"[ft-detect] stack_detection: {int(hits_col.sum())}/{len(cat)} trails hit at 5sigma")
    print("FT_DETECT_DONE")


def stage_assemble(a):
    """asteroid_cnn env: images (re-injected) + masks (draw_one_line) + real_labels -> H5 + CSV."""
    import h5py
    from astropy.io import fits
    from ADCNN.pipelines.heliolinc.inject_trails import add_trails
    from ADCNN.utils.helpers import draw_one_line
    cat = pd.read_csv(f"{a.out}/ft_catalog.csv")
    assert "stack_detection" in cat.columns, "run --stage detect first"
    for split in ["train", "val"]:
        c = cat[cat.split == split]
        panels = list(c.groupby(["visit", "detector"]))
        H = W = None
        csv_rows = []
        h5p = f"{a.out}/{split}.h5"
        with h5py.File(h5p, "w") as h5:
            ims = msk = rl = None
            for pid, ((v, det), g) in enumerate(panels):
                from ADCNN.inference.diffim_io import open_diffim
                with open_diffim(g.fits_path.iloc[0], memmap=False) as hd:
                    img = np.nan_to_num(hd[1].data.astype(np.float32))
                if ims is None:
                    H, W = img.shape
                    ims = h5.create_dataset("images", (len(panels), H, W), dtype="f4",
                                            chunks=(1, H, W), compression="lzf")
                    msk = h5.create_dataset("masks", (len(panels), H, W), dtype="bool",
                                            chunks=(1, H, W), compression="lzf")
                    rl = h5.create_dataset("real_labels", (len(panels), H, W), dtype="u2",
                                           chunks=(1, H, W), compression="lzf")
                if img.shape != (H, W):     # mixed CCD geometry: pad/crop to first panel frame
                    tmp = np.zeros((H, W), np.float32); tmp[:img.shape[0], :img.shape[1]] = img[:H, :W]
                    img = tmp
                ims[pid] = add_trails(img.copy(), _inject_rows(g))
                m = np.zeros((H, W), np.uint8)
                for r in g.itertuples():
                    draw_one_line(m, (r.x, r.y), r.beta, r.trail_length,
                                  true_value=1, line_thickness=LINE_THICK)
                msk[pid] = m.astype(bool)
                z = np.load(f"{a.out}/reallabels/{v}_{det}.npz")["real_labels"]
                if z.shape != (H, W):
                    t2 = np.zeros((H, W), np.uint16); t2[:z.shape[0], :z.shape[1]] = z[:H, :W]
                    z = t2
                rl[pid] = z
                for r in g.itertuples():
                    csv_rows.append(dict(image_id=pid, injection_id=int(r.injection_id),
                                         visit=int(v), detector=int(det),
                                         x=int(round(r.x)), y=int(round(r.y)), beta=r.beta,
                                         trail_length=r.trail_length, mag=r.mag, SNR=r.SNR,
                                         source_type="Trail", physical_filter="?",
                                         stack_detection=bool(r.stack_detection)))
                if (pid + 1) % 100 == 0:
                    print(f"  [ft-assemble {split}] {pid+1}/{len(panels)}", flush=True)
        pd.DataFrame(csv_rows).to_csv(f"{a.out}/{split}.csv", index=False)
        print(f"[ft-assemble] {split}: {len(panels)} panels, {len(csv_rows)} trails -> {h5p}")
    print("FT_ASSEMBLE_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["catalog", "detect", "assemble"])
    ap.add_argument("--run", default=f"{OUTPUTS}/runs/run_dev")
    ap.add_argument("--out", default=f"{OUTPUTS}/runs/run_ft")
    ap.add_argument("--panels-train", type=int, default=1500)
    ap.add_argument("--panels-val", type=int, default=300)
    ap.add_argument("--trails-min", type=int, default=15)
    ap.add_argument("--trails-max", type=int, default=30)
    ap.add_argument("--m5", type=float, default=24.0)
    ap.add_argument("--len-min", type=float, default=6.0, help="trail length log-uniform range (px)")
    ap.add_argument("--len-max", type=float, default=50.0)
    ap.add_argument("--snr-min", type=float, default=2.0, help="snr_target log-uniform range")
    ap.add_argument("--snr-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--exclude-catalog", default=None,
                    help="csv with visit,detector to EXCLUDE (leakage-clean stage-2 set disjoint from stage-1)")
    a = ap.parse_args()
    {"catalog": stage_catalog, "detect": stage_detect, "assemble": stage_assemble}[a.stage](a)


if __name__ == "__main__":
    main()
