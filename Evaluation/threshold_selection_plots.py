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
    PYTHONPATH=. python Evaluation/threshold_selection_plots.py \
        [--cache-dir ADCNN/pipelines/heliolinc/run_lambda/_nomfsnr_cache] [--out Evaluation/figures]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CB, CR = "#1f77b4", "#c0392b"          # completeness blue / purity red
OP_S, OP_MF = 0.80, 5                   # the frozen operating point (THRESHOLD_PROTOCOL.md)
RATE_LO, RATE_HI = 1.0, 8.0             # faint-fast rate band (deg/day)


def load_pairs(cache_dir):
    """Load the exact per-pair rows + the recoverable-object truth map."""
    rows, allrec = [], {}
    files = sorted(glob.glob(f"{cache_dir}/*_smin0.6_v3exact.json"))
    if not files:
        raise SystemExit(f"no *_smin0.6_v3exact.json caches under {cache_dir} -- run exact_lowS_pairs first")
    for cp in files:
        k = os.path.basename(cp).split("_smin")[0]
        c = json.load(open(cp))
        for r in c["rows"]:
            rows.append((k, *r))
        for o, s in c["rec"].items():
            allrec[f"{k}_{o}"] = float(s)
    R = [(k, mn, mf, rate, label, obj)
         for (k, mn, mf, rate, label, nfp, obj, mx, ln, c2, dpa, dsp, perp) in rows
         if RATE_LO <= rate <= RATE_HI]
    ff_tot = sum(1 for s in allrec.values() if 2 <= s < 10)
    return R, allrec, ff_tot


def make_metrics(R, allrec, ff_tot):
    def C(S, mf):
        objs = {f"{k}_{obj}" for (k, mn, mfv, rate, label, obj) in R
                if label == "tp" and mn >= S and mfv >= mf and 2 <= allrec.get(f"{k}_{obj}", -1) < 10}
        return 100 * len(objs) / ff_tot

    def P(S, mf):
        tp = sum(1 for (k, mn, mfv, rate, label, obj) in R if label == "tp" and mn >= S and mfv >= mf)
        fp = sum(1 for (k, mn, mfv, rate, label, obj) in R if label == "fp" and mn >= S and mfv >= mf)
        return 100 * tp / (tp + fp) if (tp + fp) else np.nan
    return C, P


def fig_2x2(C, P, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    def panel(ax, xs, getC, getP, xlab, vline, title):
        ax.plot(xs, [getC(x) for x in xs], "o-", color=CB, ms=5)
        ax.set_ylabel("completeness [%]", color=CB); ax.tick_params(axis="y", labelcolor=CB)
        ax.set_ylim(0, 9); ax.set_xlabel(xlab); ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, [getP(x) for x in xs], "s-", color=CR, ms=5)
        ax2.set_ylabel("purity [%] (injected-truth density)", color=CR)
        ax2.tick_params(axis="y", labelcolor=CR); ax2.set_ylim(0, 103)
        ax.axvline(vline, color="k", ls="--", lw=1.2); ax.set_title(title, fontsize=10.5)

    Ss = list(np.round(np.arange(0.60, 0.901, 0.025), 4)); Ms = list(range(0, 11))
    panel(axes[0, 0], Ss, lambda S: C(S, OP_MF), lambda S: P(S, OP_MF), "ADCNN score threshold S", OP_S,
          "(a) sweep S at shipped mfsnr$\\geq$5  $\\rightarrow$  S=0.80 selected")
    panel(axes[0, 1], Ms, lambda m: C(OP_S, m), lambda m: P(OP_S, m), "mfsnr threshold", OP_MF,
          "(b) sweep mfsnr at shipped S$\\geq$0.80  $\\rightarrow$  mfsnr=5 selected")
    panel(axes[1, 0], Ss, lambda S: C(S, 0), lambda S: P(S, 0), "ADCNN score threshold S", OP_S,
          "(c) sweep S with NO photometric cut: purity collapses at low S")
    panel(axes[1, 1], Ms, lambda m: C(0.60, m), lambda m: P(0.60, m), "mfsnr threshold", OP_MF,
          "(d) sweep mfsnr at low floor S$\\geq$0.60: purity needs mfsnr, completeness flat")
    fig.suptitle("Threshold selection for the same-night 2-visit alert product — completeness (blue, left)"
                 " and purity (red, right)\nALL points exactly measured (82 injection-on-real fields, full"
                 " pair enumeration, validated vs independent chain)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{out}/threshold_selection_2x2.png", dpi=140); plt.close(fig)


def fig_singles(C, P, out):
    Ss = list(np.round(np.arange(0.60, 0.901, 0.025), 4))
    lines = [(0, "#36b", "no photometric cut (mfsnr$\\geq$0)"),
             (5, "#283", "shipped: mfsnr$\\geq$5"), (10, "#c44", "mfsnr$\\geq$10")]
    for name, getter, ylab, ymax in [("completeness_vs_threshold", C, "faint-fast NEO completeness [%]", 9),
                                     ("purity_vs_threshold_insample", P,
                                      "alert purity on the validation fields [%]", 103)]:
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        for mf, c, lab in lines:
            ax.plot(Ss, [getter(S, mf) for S in Ss], "o-", color=c, label=lab, ms=4)
        sel = getter(OP_S, OP_MF)
        ax.plot([OP_S], [sel], "k*", ms=16, zorder=6)
        ax.annotate(f"selected operating point\nS$\\geq$0.80, mfsnr$\\geq$5  ({sel:.1f}%)",
                    xy=(OP_S, sel), xytext=(0.62, 0.85 * ymax), fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.set_xlabel("ADCNN score threshold S"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0, ymax)
        fig.tight_layout(); fig.savefig(f"{out}/{name}.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="ADCNN/pipelines/heliolinc/run_lambda/_nomfsnr_cache")
    ap.add_argument("--out", default="Evaluation/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R, allrec, ff_tot = load_pairs(a.cache_dir)
    C, P = make_metrics(R, allrec, ff_tot)
    print(f"loaded {len(R)} rate-banded exact pairs; faint-fast denominator {ff_tot}")
    fig_2x2(C, P, a.out)
    fig_singles(C, P, a.out)
    print(f"selected op: C={C(OP_S, OP_MF):.2f}%  P={P(OP_S, OP_MF):.1f}%  -> figures in {a.out}/")


if __name__ == "__main__":
    main()
