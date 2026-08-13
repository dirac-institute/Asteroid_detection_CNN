#!/usr/bin/env python3
"""Measure the stack's detections with ADCNN's trail estimator, so both measurements are available.

WHY. The DPDD `trailLength` on `dia_source_detector` is ABSENT (NaN) for ~31% of sources -- the trailed
fit is not available for most of them -- and where present it tops out well short of the fast-mover
regime. Only 1.12% of production stack rows clear the op's `len_db_min: 6.0`, against 82.09% of
ADCNN's, which is why the stack contributes 0.33% of the linked stream and ZERO delivered alerts. The
stack nonetheless FINDS real movers ADCNN misses -- 154 of them at both epochs on the injection truth
set, the gap between a 53.8% union ceiling and ADCNN's 49.8% -- and they are currently wasted because
the linker is built on trail-vs-chord agreement and they carry no trail.

The estimator needs only a POSITION and PIXELS, not an ADCNN detection, so it can measure a trail for
a source only the stack found. Verified prerequisites: the two catalogues share a pixel frame (median
x/y offset 0.02/0.11 px on coincident sources, 94.6% within 3 px), and the 96 px stamp is in-bounds
for 97% of stack detections.

IT MUST BE GATED, and that is the whole difficulty. Run indiscriminately the template bank SATURATES:
re-measuring ADCNN's own rows on real panels returned a median 31.19 px where the catalogue says
14.31, and stack rows returned 68.46 px with 32.1% pinned at the 79 px ceiling. The reason ADCNN's own
lengths are trustworthy is that ADCNN only measures what its SEGMENTATION already judged trail-like;
a DIASource table is dominated by stars and artifacts, on which a long line template happily
accumulates flux. Agreement with the catalogue is a steep function of trail-likeness -- Spearman 0.493
over everything, 0.772 at score>=0.7, 0.924 at score>=0.9 where the medians agree to 0.01 px.

*** STATUS: THE APPROACH DOES NOT DELIVER WHAT IT WAS BUILT FOR. Read this before investing further.
    The gate and the reason a detection is stack-only are THE SAME THING, so any gate strong enough to
    stop the estimator saturating also removes almost everything there was to rescue. ***

    Validated on real inputs (prob AND agg regenerated for 6 real 0706 panels, 992 stack detections,
    no placeholder arrays), gate = seg prob >= 0.45 AND stage-2 CNN >= 0.5886:

        ungated                              median 69.01 px   33.9% at the 79 px ceiling
        seg probability only                 median  6.20 px   24.0% at ceiling
        seg + stage-2 (this module)          median  5.32 px    9.5% at ceiling   <- TRUSTWORTHY
                                                          (ADCNN's own rows sit at 4.7-10%)

    So the two-stage gate SOLVES saturation. But of the 84 detections that pass it, only **2** were
    not already reported by ADCNN -- and both of those saturate. The rescuable population is 2 of 992.
    That is the risk flagged before this was built, and it materialised: a detection is stack-only
    PRECISELY BECAUSE ADCNN's segmentation did not fire on it, so gating on ADCNN's segmentation
    selects almost exactly the sources ADCNN already has. Every weaker gate saturates instead.

    CONSEQUENCE FOR THE MERGE. The stack's unique contribution -- 154 both-epoch movers ADCNN misses,
    the gap between a 53.8% union ceiling and ADCNN's 49.8% -- CANNOT be unlocked by re-measuring its
    trails with our estimator. Unlocking it needs either a trail measurement that does not depend on
    our segmentation, or the stack's own trailed fit to work on that population (it does not: those
    rows are the ones with NaN or ~2 px trailLength). Sample is 6 panels; the mechanism is not
    sample-limited, but the exact 2/992 is.

*** The stack-native gate (trailFlux/psfFlux + extendedness) is separately NOT sufficient either. *** Validated on 2,738 real DIASource rows from
    LSSTCam/runs/prompt/20260706/ApPipe (embargo repo) against the 64 cached 0706 panels:

        ungated                 n=2,666   median 50.25 px   27.2% AT THE 79 px CEILING
        gated (trailFlux/psfFlux>=1.15 AND extendedness<=0.98)
                                n=   43   median 69.47 px   18.6% AT THE CEILING

    The gate keeps only 1.6% of rows and the survivors saturate WORSE, not better -- the median rises.
    trailFlux/psfFlux (1.267 trailed vs 1.052 point) and extendedness (0.867 vs 0.999) are simply too
    weak a discriminator. What DOES work is ADCNN's CNN score, where agreement with the catalogue runs
    Spearman 0.493 over everything, 0.772 at score>=0.7 and 0.924 at score>=0.9 -- so the gate has to
    come from running ADCNN's scorer at stack positions, which is a larger piece of work than this
    file. The self-check at the end of measure() enforces this: it REFUSES to write a catalogue whose
    ceiling fraction indicates saturation.

The gate attempted here is the stack's OWN trail-likeness evidence, which costs nothing extra:
  trailFlux/psfFlux  -- measured median 1.267 on len>=6 px sources vs 1.052 below
  extendedness       -- 0.867 vs 0.999
Both are carried through by ingest_diasource. A row failing the gate keeps whatever it had; nothing is
overwritten in place.

NEITHER MEASUREMENT IS UNIVERSALLY BETTER. Paired against injected truth on 1,273 movers both
estimators measured, the crossover is at ~20 px:

    true trail    n     ADCNN |frac err|   LSST trailed fit |frac err|
    0-10 px     522        0.272                 0.216
    10-20 px    526        0.288                 0.201
    20-35 px    173        0.142                 0.250
    35-99 px     52        0.112                 0.333

So this writes `len_mf`/`beta_mf` ALONGSIDE the existing `len_db`/`beta` rather than replacing them,
and `len_db` is only filled where it was absent. Choosing per-regime is a separate decision that must
be judged on DELIVERED-at-1k completeness, not on estimator error.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

PIXEL_SCALE = 0.2
STAMP = 96
TRAILFLUX_MIN = 1.15      # trailFlux/psfFlux: between the measured 1.052 (point) and 1.267 (trailed)
EXTENDEDNESS_MAX = 0.98   # measured 0.867 on trailed vs 0.999 on point sources
MAX_CEILING_FRAC = 0.05   # above this the bank is saturating, not measuring (ungated real data: 0.27)


def _gate(d):
    """Trail-likeness from the stack's own columns. Returns (mask, reason-counts)."""
    n = len(d)
    have_tf = {"trail_flux", "psfFlux"} <= set(d.columns) or {"trail_flux", "psf_flux"} <= set(d.columns)
    psf = d.get("psfFlux", d.get("psf_flux"))
    ok = np.ones(n, bool)
    why = {}
    if have_tf and psf is not None:
        r = (d["trail_flux"].to_numpy(float) / np.where(np.abs(psf.to_numpy(float)) > 0,
                                                        psf.to_numpy(float), np.nan))
        m = np.isfinite(r) & (r >= TRAILFLUX_MIN)
        why["trailFlux/psfFlux"] = int((~m).sum()); ok &= m
    if "extendedness" in d.columns:
        # An UNMEASURED extendedness must not pass while an unmeasured trailFlux/psfFlux rejects --
        # that asymmetry let 5.15% of gate-passers through on a value nobody measured (extendedness is
        # NaN on 31.6% of rows). Absent evidence of trail-likeness is not evidence of it.
        e = d["extendedness"].to_numpy(float)
        m = np.isfinite(e) & (e <= EXTENDEDNESS_MAX)
        why["extendedness"] = int((~m).sum()); ok &= m
    if not why:
        raise SystemExit(
            "[stack-trails] REFUSING: the stack catalogue carries neither trailFlux/psfFlux nor "
            "extendedness, so there is no trail-likeness gate. Running the template bank ungated "
            "SATURATES (measured: median 68.46 px with 32.1% at the 79 px ceiling) and would write "
            "confident nonsense over the whole catalogue. Re-run ingest_diasource so these columns "
            "are carried through.")
    return ok, why


def measure_panel(x, y, img, prob, agg, cnn, *, t_low, cnn_thr, panel_sigma=None, device="cpu"):
    """THE GATE THAT WORKS: ADCNN's own two stages, applied at externally-supplied positions.

    Designed to be called from inside detect_night, where `prob`, `agg` and the stage-2 `cnn` are
    ALREADY in hand for the panel -- so the expensive part (2,962 ms of seg inference) is already
    paid and this adds only a cutout batch and one template-bank call.

    Why both stages are needed, measured on real 0706 panels against stack detections:
      * ungated, the bank saturates: median 69.01 px, 33.9% pinned at the 79 px ceiling
      * gating on the SEG PROBABILITY alone fixes the bulk -- median 69.01 -> 6.20 px -- but leaves
        24% at the ceiling, because a bright star's diffraction spike has genuine seg response and is
        genuinely linear
      * stack rows ADCNN ITSELF reported sit at 10%, and ADCNN's own detections at 4.7%; the thing
        that separates a trail from a spike is STAGE 2, which is exactly what ADCNN applies

    The rescue is real: of 349 stack detections with seg prob >= 0.4 on four panels, 278 (80%) were
    never reported by ADCNN -- that is the "stack finds movers we miss" population, and it is
    currently unlinkable because 21% of them have no DPDD trailLength at all and the rest read a
    median 2.35 px.

    Returns (length_px, beta_deg, score, passed) arrays, NaN/False where the gate rejected.
    """
    import pandas as _pd
    from ADCNN.inference.cnn_postproc import apply_cnn
    from ADCNN.inference.mf_trail_length import refine_trail_length
    from ADCNN.data.preprocessing import diffim_mad_sigma

    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    L = np.full(n, np.nan); B = np.full(n, np.nan); S = np.full(n, np.nan)
    if not n:
        return L, B, S, np.zeros(0, bool)
    H, W = prob.shape
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    # STAGE 1: local MAX of the seg probability. A trail's peak response need not sit exactly under
    # the external centroid, so a single-pixel read under-selects.
    segp = np.array([prob[max(0, b - 3):b + 4, max(0, a - 3):a + 4].max() for b, a in zip(yi, xi)])
    g1 = segp >= t_low
    if not g1.any():
        return L, B, S, np.zeros(n, bool)
    sig = float(panel_sigma) if panel_sigma is not None else diffim_mad_sigma(img)
    # STAGE 2: the focal cutout CNN, the discriminator that rejects spikes.
    idx = np.flatnonzero(g1)
    cand = _pd.DataFrame({"y_centroid": y[idx], "x_centroid": x[idx],
                          "panel_sigma": np.full(len(idx), sig)})
    scored = apply_cnn(cand, cnn, img, prob, agg, thr=None, device=device)
    sc = scored["score"].to_numpy(float)
    S[idx] = sc
    g2 = sc >= cnn_thr
    keep = idx[g2]
    if not len(keep):
        return L, B, S, np.zeros(n, bool)
    Lk, Bk = refine_trail_length(x[keep], y[keep], img,
                                 np.zeros(len(keep)), np.zeros(len(keep)), sigma=sig)
    ran = Lk != 0                                  # 0 means the stamp fell off the panel edge
    L[keep[ran]] = Lk[ran]; B[keep[ran]] = Bk[ran]
    passed = np.zeros(n, bool); passed[keep[ran]] = True
    return L, B, S, passed


def measure(dets_path, out_path, limit_panels=None, verbose=True):
    from ADCNN.inference.diffim_io import open_diffim
    from ADCNN.inference.mf_trail_length import refine_trail_length
    from ADCNN.data.preprocessing import diffim_mad_sigma
    from ADCNN.linking.ingest_diasource import _endpoints

    d = pd.read_csv(dets_path, low_memory=False)
    if "src" in d.columns:
        sel = (d.src == "stack").to_numpy()
    else:
        sel = np.ones(len(d), bool)
    if not sel.any():
        raise SystemExit(f"[stack-trails] no stack rows in {dets_path}")
    gate_ok, why = _gate(d)
    todo = sel & gate_ok & d.x.notna().to_numpy() & d.y.notna().to_numpy()
    if verbose:
        print(f"[stack-trails] {int(sel.sum()):,} stack rows; trail-likeness gate rejects "
              + ", ".join(f"{k}:{v:,}" for k, v in why.items())
              + f" -> {int(todo.sum()):,} to measure", flush=True)

    L_out = np.full(len(d), np.nan); B_out = np.full(len(d), np.nan)
    if "fits_path" not in d.columns:
        raise SystemExit("[stack-trails] no fits_path column -- run merge_dets first (it backfills "
                         "fits_path from the ADCNN catalogue's visit/detector map)")
    groups = list(d[todo].groupby(["visit", "detector"]))
    if limit_panels:
        groups = groups[:limit_panels]
    n_edge = 0
    for i, ((v, det), g) in enumerate(groups):
        fp = g.fits_path.dropna()
        if not len(fp):
            continue
        try:
            with open_diffim(str(fp.iloc[0]), memmap=False) as h:
                img = np.nan_to_num(h[1].data.astype(np.float32))
        except Exception as e:
            print(f"[stack-trails]   skip visit {v} det {det}: {type(e).__name__}", flush=True)
            continue
        sig = diffim_mad_sigma(img)
        inc = g.len_db.fillna(0.0).to_numpy(float)
        inb = g.beta.fillna(0.0).to_numpy(float)
        L, B = refine_trail_length(g.x.to_numpy(float), g.y.to_numpy(float), img, inc, inb, sigma=sig)
        moved = (L != inc)                       # the estimator only writes where it actually ran
        n_edge += int((~moved).sum())
        idx = g.index.to_numpy()
        L_out[idx[moved]] = L[moved]; B_out[idx[moved]] = B[moved]
        if verbose and (i + 1) % 200 == 0:
            print(f"[stack-trails]   {i+1}/{len(groups)} panels", flush=True)

    d["len_mf"] = L_out
    d["beta_mf"] = B_out
    got = np.isfinite(L_out)
    # FILL ONLY WHERE ABSENT. The two estimators cross over at ~20 px (ours better above, the stack's
    # trailed fit better below), so overwriting a present measurement is a judgement call that has to
    # be made on delivered completeness -- not silently here.
    fill = got & (~np.isfinite(d.len_db.to_numpy(float)))
    if fill.any():
        ra = d.ra.to_numpy(float); dec = d.dec.to_numpy(float)
        r0, d0, r1, d1 = _endpoints(ra[fill], dec[fill], L_out[fill], B_out[fill])
        d.loc[fill, ["len_db", "length", "beta"]] = np.c_[L_out[fill], L_out[fill], B_out[fill]]
        d.loc[fill, ["ra0", "dec0", "ra1", "dec1"]] = np.c_[r0, d0, r1, d1]
    if verbose:
        print(f"[stack-trails] measured {int(got.sum()):,} rows "
              f"({n_edge:,} skipped: too close to a panel edge for a {STAMP}px stamp)", flush=True)
        print(f"[stack-trails] filled len_db on {int(fill.sum()):,} rows that had NO trail "
              f"measurement; {int((got & ~fill).sum()):,} already had one and were LEFT ALONE "
              f"(len_mf carries the alternative)", flush=True)
        if got.any():
            print(f"[stack-trails] len_mf median {np.nanmedian(L_out[got]):.2f} px, "
                  f">=6px on {100*np.mean(L_out[got] >= 6):.1f}%; at the "
                  f"bank ceiling on {100*np.mean(L_out[got] >= 75):.1f}% (a high ceiling fraction "
                  f"means the gate is too loose -- the bank is latching onto structure, not trails)",
                  flush=True)
    # SELF-CHECK: refuse to write a saturated catalogue. The template bank is calibrated on injected
    # TRAILS; at arbitrary positions on a real panel it latches onto stellar structure and pins at the
    # top of MF_L. A high ceiling fraction is the signature, and writing it would put confident
    # nonsense into len_db for exactly the rows a length floor then admits.
    if got.any():
        ceil_frac = float(np.mean(L_out[got] >= 0.95 * 79))
        if ceil_frac > MAX_CEILING_FRAC and not os.environ.get("ADCNN_ALLOW_SATURATED_TRAILS"):
            raise SystemExit(
                f"[stack-trails] REFUSING to write: {100*ceil_frac:.1f}% of measured rows are at the "
                f"template-bank ceiling (limit {100*MAX_CEILING_FRAC:.0f}%), which means the estimator "
                f"is saturating on structure rather than measuring trails, not that these are long "
                f"movers. The trail-likeness gate is too loose. MEASURED on real 0706 panels: the "
                f"stack-native gate (trailFlux/psfFlux + extendedness) gives 18.6% here against 27.2% "
                f"ungated -- it does NOT solve this. A gate built on ADCNN's CNN score does "
                f"(Spearman 0.924 at score>=0.9). Set ADCNN_ALLOW_SATURATED_TRAILS=1 only to study "
                f"the output, never to build a product.")
    d.to_csv(out_path, index=False)
    return int(got.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="merged catalogue (needs src + fits_path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-panels", type=int, default=None, help="for a quick trial run")
    a = ap.parse_args(argv)
    measure(a.dets, a.out, limit_panels=a.limit_panels)


if __name__ == "__main__":
    sys.exit(main())
