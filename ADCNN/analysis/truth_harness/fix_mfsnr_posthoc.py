#!/usr/bin/env python3
"""Repair mf_snr in EXISTING catalogues without re-detecting.

Majority-masked panels made median(|x|)=0, so diffim_mad_sigma returned its 1e-8 floor and
mf_snr = flux/(sigma*sqrt(n_line)) exploded -- up to 1.38e12 on 0706, 22,221 detections (0.58%)
across 770 panels. mfsnr_min_2v cannot reject them: they sit ABOVE the gate by twelve orders of
magnitude, so they enter the linkable set and manufacture chance links.

mf_snr is LINEAR in sigma, so the repair is exact without re-running detection:

    mf_snr_fixed = mf_snr_stored * (sigma_legacy / sigma_fixed)

Only affected panels are read. Trail length comes from the footprint and mf_flux does not involve
sigma, so nothing else in the catalogue needs touching.

Usage:  python fix_mfsnr_posthoc.py <dets.csv> [<dets.csv> ...]
"""
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ADCNN.data.preprocessing import diffim_mad_sigma
from ADCNN.inference.diffim_io import open_diffim

FLAG = 1e4          # no real nJy diffim detection reaches this; a marker for degenerate sigma


def legacy_sigma(a):
    good = a[np.isfinite(a)]
    return 1.0 if good.size == 0 else float(1.4826 * np.median(np.abs(good)) + 1e-8)


def fix(path, manifest):
    d = pd.read_csv(path)
    if "mf_snr" not in d.columns:
        print(f"  {path}: no mf_snr column, skipped"); return
    man = pd.read_csv(manifest)
    fp = man.set_index(["visit", "detector"])["fits_path"].to_dict()
    bad = d[d.mf_snr > FLAG]
    panels = sorted({(int(v), int(t)) for v, t in zip(bad.visit, bad.detector)})
    print(f"  {path}\n    {len(d):,} detections | {len(bad):,} above mf_snr {FLAG:g} "
          f"({100*len(bad)/len(d):.3f}%) on {len(panels)} panels", flush=True)
    if not panels:
        print("    nothing to repair"); return
    scale = {}
    for i, key in enumerate(panels):
        p = fp.get(key)
        if p is None:
            continue
        try:
            with open_diffim(p, memmap=False) as h:
                a = np.nan_to_num(h[1].data.astype(np.float32))
        except Exception:
            continue
        s_old, s_new = legacy_sigma(a), diffim_mad_sigma(a)
        if np.isfinite(s_old) and np.isfinite(s_new) and s_new > 0:
            scale[key] = s_old / s_new
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(panels)} panels", flush=True)
    key = list(zip(d.visit.astype(int), d.detector.astype(int)))
    f = np.array([scale.get(k, 1.0) for k in key], float)
    n_ch = int((f != 1.0).sum())
    before = d.mf_snr.max()
    d["mf_snr"] = d.mf_snr.to_numpy(float) * f
    shutil.copy2(path, path + ".pre_mfsnr_fix")
    d.to_csv(path, index=False)
    print(f"    rescaled {n_ch:,} detections on {len(scale)} panels | "
          f"max mf_snr {before:.3g} -> {d.mf_snr.max():.3g} | backup at {path}.pre_mfsnr_fix", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        night = p.rsplit("/", 2)[-2]
        fix(p, f"outputs/runs/10k_cadence/{night}/manifest.csv")
