#!/usr/bin/env python3
"""Threshold-selection figures for the same-night 2-visit alert product (paper figures).

Reads the EXACT per-pair evidence tables produced by
``ADCNN/pipelines/heliolinc/exact_lowS_pairs.py`` (full chord-pair enumeration at score floor 0.60,
82 injection-on-real validation fields, no FP subsampling; validated bit-identical to the python
physical_check chain and cross-checked against an independent run) and regenerates:

  1. threshold_selection_2x2.png      -- the 2x2 dual-axis selection figure:
        (a) completeness+purity vs ADCNN score S at the shipped mfsnr>=5  -> S=0.80 selected
        (b) completeness+purity vs mfsnr at the shipped S>=0.80           -> mfsnr=5 selected
        (c) completeness+purity vs S with NO photometric cut (why S alone cannot fix purity)
        (d) completeness+purity vs mfsnr at low floor S>=0.60 (why mfsnr alone cannot either)
  2. completeness_vs_threshold.png    -- single-panel total completeness vs S (3 mfsnr lines)
  3. purity_vs_threshold_insample.png -- single-panel purity vs S (3 mfsnr lines, linear axis)

Definitions (faint-fast science bin): completeness = distinct injected faint-fast NEOs
(detection-SNR 2-10, rate 1-8 deg/day; denominator = all such objects injected into >=2 same-night
visits) with >=1 accepted 2-visit pair; purity = TP pairs / (TP+FP pairs) at the INJECTED truth
density (the real-sky base-rate-corrected purity is quoted separately in THRESHOLD_PROTOCOL.md).

Usage (from the repo root):
    PYTHONPATH=. python -m ADCNN.qa.plots_thresholds \
        [--cache-dir ADCNN/pipelines/heliolinc/run_lambda/_nomfsnr_cache] [--out Evaluation/figures]
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The canonical curve computation + the decision rule live in ONE place (the selection stage); this
# figure script imports them so the plotted operating point is the REGENERATED selection, never a
# hardcoded constant (acceptance D). Run as a module: PYTHONPATH=. python -m ADCNN.qa.plots_thresholds
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (ADCNN/qa/ -> root)
from ADCNN.calibration.threshold_selection import (
    load_pairs, make_metrics as _make_metrics, select_operating_point, DEFAULT_CACHE_DIR)

CB, CR = "#1f77b4", "#c0392b"          # completeness blue / purity red
RATE_LO, RATE_HI = 1.0, 8.0            # faint-fast rate band (deg/day)
OP_S, OP_MF = None, None               # set in main() from the REGENERATED selection (not hardcoded)


def make_metrics(R, allrec, ff_tot, n_boot=300, seed=42):
    """Thin adapter: the selection stage returns (C, P, band, stats); the figures use (C, P, band)."""
    C, P, band, _stats = _make_metrics(R, allrec, ff_tot, n_boot=n_boot, seed=seed)
    return C, P, band


def fig_2x2(C, P, band, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    def panel(ax, xs, getC, getP, getB, xlab, vline, title):
        bands = [getB(x) for x in xs]
        cl = [b[0][0] for b in bands]; ch = [b[0][1] for b in bands]
        pl = [b[1][0] for b in bands]; ph = [b[1][1] for b in bands]
        ax.fill_between(xs, cl, ch, color=CB, alpha=0.18, lw=0)
        ax.plot(xs, [getC(x) for x in xs], "o-", color=CB, ms=5)
        ax.set_ylabel("completeness [%]", color=CB); ax.tick_params(axis="y", labelcolor=CB)
        ax.set_ylim(0, 9); ax.set_xlabel(xlab); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.fill_between(xs, pl, ph, color=CR, alpha=0.15, lw=0)
        ax2.plot(xs, [getP(x) for x in xs], "s-", color=CR, ms=5)
        ax2.set_ylabel("validation injected-truth fraction [%]", color=CR)
        ax2.tick_params(axis="y", labelcolor=CR); ax2.set_ylim(0, 103)
        ax.axvline(vline, color="k", ls="--", lw=1.2); ax.set_title(title, fontsize=10.5)

    Ss = list(np.round(np.arange(0.60, 0.901, 0.025), 4)); Ms = list(range(0, 11))
    panel(axes[0, 0], Ss, lambda S: C(S, OP_MF), lambda S: P(S, OP_MF), lambda S: band(S, OP_MF),
          "ADCNN score threshold S", OP_S,
          "(a) sweep S at the adopted mfsnr$\\geq$5  $\\rightarrow$  S=0.80 adopted")
    panel(axes[0, 1], Ms, lambda m: C(OP_S, m), lambda m: P(OP_S, m), lambda m: band(OP_S, m),
          "mfsnr threshold", OP_MF,
          "(b) sweep mfsnr at the adopted S$\\geq$0.80  $\\rightarrow$  mfsnr=5 adopted")
    panel(axes[1, 0], Ss, lambda S: C(S, 0), lambda S: P(S, 0), lambda S: band(S, 0),
          "ADCNN score threshold S", OP_S,
          "(c) sweep S with NO photometric cut: injected-truth fraction collapses at low S")
    panel(axes[1, 1], Ms, lambda m: C(0.60, m), lambda m: P(0.60, m), lambda m: band(0.60, m),
          "mfsnr threshold", OP_MF,
          "(d) sweep mfsnr at low floor S$\\geq$0.60: the photometric cut carries the cleaning")
    fig.suptitle("Threshold selection for the same-night 2-visit alert product — completeness (blue, left)"
                 " and validation injected-truth fraction (red, right)\nall points measured by full pair"
                 " enumeration (82 injection-on-real validation fields); shaded = field-bootstrap 16-84%;"
                 " dashed = adopted default alert operating point", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out}/threshold_selection_2x2.png", dpi=140); plt.close(fig)


def fig_singles(C, P, band, out):
    Ss = list(np.round(np.arange(0.60, 0.901, 0.025), 4))
    lines = [(0, "#36b", "no photometric cut (mfsnr$\\geq$0)"),
             (5, "#283", "adopted: mfsnr$\\geq$5"), (10, "#c44", "mfsnr$\\geq$10")]
    for name, getter, bidx, ylab, ymax in [
            ("completeness_vs_threshold", C, 0, "faint-fast NEO completeness [%]", 9),
            ("purity_vs_threshold_insample", P, 1, "validation injected-truth fraction [%]", 103)]:
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for mf, c, lab in lines:
            bs = [band(S, mf)[bidx] for S in Ss]
            ax.fill_between(Ss, [b[0] for b in bs], [b[1] for b in bs], color=c, alpha=0.15, lw=0)
            ax.plot(Ss, [getter(S, mf) for S in Ss], "o-", color=c, label=lab, ms=4)
        sel = getter(OP_S, OP_MF)
        ax.plot([OP_S], [sel], "k*", ms=16, zorder=6)
        ax.annotate(f"adopted default alert operating point\nS$\\geq$0.80, mfsnr$\\geq$5  ({sel:.1f}%)",
                    xy=(OP_S, sel), xytext=(0.62, 0.85 * ymax), fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.set_xlabel("ADCNN score threshold S"); ax.set_ylabel(ylab)
        ax.set_title("all points measured by full pair enumeration; shaded = field-bootstrap 16-84%",
                     fontsize=9.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0, ymax)
        fig.tight_layout(); fig.savefig(f"{out}/{name}.png", dpi=140); plt.close(fig)


def main():
    global OP_S, OP_MF
    import os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out", default="Evaluation/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R, allrec, ff_tot = load_pairs(a.cache_dir)
    C, P, band = make_metrics(R, allrec, ff_tot)
    # the marked operating point is the REGENERATED selection (purity-floor + retention rule),
    # not a hardcoded literal -- so the figure always reflects what the decision rule produces.
    sel = select_operating_point(C, P)
    OP_S, OP_MF = sel["score_min"], int(sel["mfsnr_min"])
    print(f"loaded {len(R)} rate-banded exact pairs; faint-fast denominator {ff_tot}")
    print(f"regenerated operating point: S>={OP_S}, mfsnr>={OP_MF} "
          f"(C={sel['at_op']['faint_fast_completeness_pct']}%, "
          f"P={sel['at_op']['in_sample_purity_pct']}%)")
    fig_2x2(C, P, band, a.out)
    fig_singles(C, P, band, a.out)
    print(f"  -> figures in {a.out}/")


if __name__ == "__main__":
    main()
