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


def make_metrics(R, allrec, ff_tot, n_boot=300, seed=42):
    """Point estimates + field-bootstrap 16-84% bands (fields are the independent unit)."""
    from collections import defaultdict
    fields = sorted({k for (k, *_r) in R}); fidx = {k: i for i, k in enumerate(fields)}; NF = len(fields)
    F = np.array([fidx[k] for (k, mn, mf, rate, label, obj) in R])
    MN = np.array([mn for (k, mn, mf, rate, label, obj) in R])
    MF = np.array([mf for (k, mn, mf, rate, label, obj) in R])
    TP = np.array([label == "tp" for (k, mn, mf, rate, label, obj) in R])
    OBJ = np.array([f"{k}_{obj}" if obj else "" for (k, mn, mf, rate, label, obj) in R], dtype=object)
    FFO = np.array([bool(obj) and 2 <= allrec.get(f"{k}_{obj}", -1) < 10
                    for (k, mn, mf, rate, label, obj) in R])
    denom_f = np.zeros(NF)
    for o, s in allrec.items():
        if 2 <= s < 10:
            denom_f[fidx[o.split("_", 1)[0]]] += 1

    def stats(S, mf):
        m = (MN >= S) & (MF >= mf)
        tp_f = np.bincount(F[m & TP], minlength=NF).astype(float)
        fp_f = np.bincount(F[m & ~TP], minlength=NF).astype(float)
        per = defaultdict(set)
        for i in np.where(m & TP & FFO)[0]:
            per[F[i]].add(OBJ[i])
        obj_f = np.zeros(NF)
        for f, s_ in per.items():
            obj_f[f] = len(s_)
        return obj_f, tp_f, fp_f

    def C(S, mf):
        return 100 * stats(S, mf)[0].sum() / ff_tot

    def P(S, mf):
        _, tp, fp = stats(S, mf); t = tp.sum() + fp.sum()
        return 100 * tp.sum() / t if t else np.nan

    rng = np.random.default_rng(seed)
    BOOT = [rng.integers(0, NF, NF) for _ in range(n_boot)]

    def band(S, mf):
        obj_f, tp_f, fp_f = stats(S, mf)
        Cs, Ps = [], []
        for b in BOOT:
            Cs.append(100 * obj_f[b].sum() / max(denom_f[b].sum(), 1))
            t = tp_f[b].sum() + fp_f[b].sum()
            Ps.append(100 * tp_f[b].sum() / t if t else np.nan)
        return ((np.percentile(Cs, 16), np.percentile(Cs, 84)),
                (np.nanpercentile(Ps, 16), np.nanpercentile(Ps, 84)))
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="ADCNN/pipelines/heliolinc/run_lambda/_nomfsnr_cache")
    ap.add_argument("--out", default="Evaluation/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R, allrec, ff_tot = load_pairs(a.cache_dir)
    C, P, band = make_metrics(R, allrec, ff_tot)
    print(f"loaded {len(R)} rate-banded exact pairs; faint-fast denominator {ff_tot}")
    fig_2x2(C, P, band, a.out)
    fig_singles(C, P, band, a.out)
    (cb, _), (pb, _) = band(OP_S, OP_MF)[0], band(OP_S, OP_MF)[1]
    print(f"adopted op: C={C(OP_S, OP_MF):.2f}%  injected-truth fraction={P(OP_S, OP_MF):.1f}%  "
          f"-> figures in {a.out}/")


if __name__ == "__main__":
    main()
