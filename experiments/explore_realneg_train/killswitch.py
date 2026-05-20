#!/usr/bin/env python
"""Realneg fine-tune GO / NO-GO per the design kill-switch (RESULTS.md §10).

Inputs (produced by slurm_eval_realneg.sh):
  <eval-dir>/fp_fix.txt   — OLD=promoted-ft vs FT=realneg, same held-out
                            empties: 'thr | OLD_FPgen FT_FPgen | OLD_posR FT_posR'
  <eval-dir>/bar_realneg.txt  — realneg synthetic objectwise bar
  <baseline-bar> (bar_ft.txt) — promoted-ft synthetic bar

KILL (NO-GO) if EITHER:
  (a) synthetic recall regressed: realneg posR < 0.95 at a thr where the
      promoted-ft posR >= 0.95  (or bar cTP/NN_TP < 0.95x baseline);
  (b) held-out real-empty genuine FP/CCD did NOT improve > 2x over the
      promoted ft at a matched-posR operating point (need realneg_FP <=
      ft_FP/2, trending toward a 2nd-stage-usable <= ~1-2/CCD).
GO only if recall preserved AND FP > 2x better at a usable operating point.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FLOAT = r"[-+]?\d+(?:\.\d+)?"


def parse_fpfix(p: Path):
    """Return [(thr, old_fp, ft_fp, old_posR, ft_posR), ...] from the
    'ORIGINAL vs FINE-TUNED' table (5 numeric fields per row)."""
    rows, in_tbl = [], False
    for ln in p.read_text().splitlines():
        if "ORIGINAL vs FINE-TUNED" in ln or "OLD FPgen" in ln:
            in_tbl = True
            continue
        if in_tbl:
            nums = re.findall(FLOAT, ln)
            if len(nums) >= 5 and "|" in ln:
                t, ofp, ffp, opr, fpr = (float(nums[0]), float(nums[1]),
                                         float(nums[2]), float(nums[3]),
                                         float(nums[4]))
                rows.append((t, ofp, ffp, opr, fpr))
    return rows


def parse_bar(p: Path):
    """Return {thr: (NN_TP, cTP, cFP)} from a BAR txt.

    Line: '{split} {thr} | {NN_TP} {NN_FP} {NN_FN} | {cTP} {cFP} {cFN} {nObj}'
    Split on '|' so a digit inside the split name (test_5sigma) can't shift
    column indices.
    """
    out = {}
    for ln in p.read_text().splitlines():
        if ln.count("|") < 2 or "split" in ln or "BAR" in ln:
            continue
        seg = ln.split("|")
        head = re.findall(FLOAT, seg[0])   # [...maybe '5'..., thr]
        nn = re.findall(FLOAT, seg[1])     # NN_TP NN_FP NN_FN
        cc = re.findall(FLOAT, seg[2])     # cTP cFP cFN nObj
        if not head or len(nn) < 1 or len(cc) < 2:
            continue
        thr = float(head[-1])              # thr is the LAST float before '|'
        out[round(thr, 2)] = (int(float(nn[0])),
                              int(float(cc[0])), int(float(cc[1])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--baseline-bar", required=True)
    a = ap.parse_args()
    ed = Path(a.eval_dir)
    fpfix = parse_fpfix(ed / "fp_fix.txt")
    bar_rn = parse_bar(ed / "bar_realneg.txt")
    bar_ft = parse_bar(Path(a.baseline_bar))

    L = ["=" * 64, "REALNEG FINE-TUNE — KILL-SWITCH DECISION", "=" * 64, ""]
    L.append("Held-out real-empty genuine FP/CCD  (same panels, both models)")
    L.append(f"{'thr':>5} {'ft_FP':>9} {'rn_FP':>9} {'x_better':>9} "
             f"{'ft_posR':>8} {'rn_posR':>8}")
    best_factor, gate_b_pt = 0.0, None
    recall_ok = True
    for t, ofp, ffp, opr, fpr in fpfix:
        fac = (ofp / ffp) if ffp > 1e-9 else float("inf")
        L.append(f"{t:>5.2f} {ofp:>9.1f} {ffp:>9.1f} {fac:>8.2f}x "
                 f"{opr:>8.3f} {fpr:>8.3f}")
        # recall gate (a): where ft kept the trail, realneg must too
        if opr >= 0.95 and fpr < 0.95:
            recall_ok = False
        # FP gate (b): >2x better at a matched-posR, usable point
        if opr >= 0.95 and fpr >= 0.95 * opr and fac > best_factor:
            best_factor = fac
            gate_b_pt = (t, ofp, ffp, fpr)

    # synthetic objectwise bar regression check
    bar_ok, bar_lines = True, ["", "Synthetic objectwise BAR (must not regress)"]
    bar_lines.append(f"{'thr':>5} {'ft_cTP':>7} {'rn_cTP':>7} "
                     f"{'ft_NNTP':>7} {'rn_NNTP':>7} {'ft_cFP':>8} {'rn_cFP':>8}")
    for thr in sorted(set(bar_ft) & set(bar_rn)):
        f_nn, f_ctp, f_cfp = bar_ft[thr]
        r_nn, r_ctp, r_cfp = bar_rn[thr]
        bar_lines.append(f"{thr:>5.2f} {f_ctp:>7} {r_ctp:>7} {f_nn:>7} "
                         f"{r_nn:>7} {f_cfp:>8} {r_cfp:>8}")
        if r_ctp < 0.95 * f_ctp or r_nn < 0.95 * f_nn:
            bar_ok = False
    L += bar_lines

    # 2nd-stage-usable: best realneg FP at a recall-preserving point
    usable_fp = min([ffp for _, _, ffp, opr, fpr in fpfix
                     if opr >= 0.95 and fpr >= 0.95 * opr], default=float("inf"))

    decision = "GO"
    reasons = []
    if not recall_ok:
        decision = "NO-GO"
        reasons.append("(a) synthetic recall regressed >5% at a "
                       "trail-preserving threshold")
    if not bar_ok:
        decision = "NO-GO"
        reasons.append("(a) synthetic objectwise BAR cTP/NN_TP dropped "
                       ">5% vs promoted ft")
    if best_factor < 2.0:
        decision = "NO-GO"
        reasons.append(f"(b) held-out FP/CCD not >2x better than promoted "
                       f"ft at matched posR (best {best_factor:.2f}x)")
    if best_factor >= 2.0 and usable_fp > 2.0:
        reasons.append(f"NOTE: >2x better ({best_factor:.2f}x) but best "
                       f"usable FP/CCD={usable_fp:.1f} (>~1-2 target — "
                       f"improvement real but not yet deployable)")

    L += ["", "-" * 64,
          f"best FP improvement at matched posR : {best_factor:.2f}x"
          + (f"  @thr {gate_b_pt[0]:.2f} "
             f"(ft {gate_b_pt[1]:.1f} -> rn {gate_b_pt[2]:.1f}/CCD, "
             f"rn_posR {gate_b_pt[3]:.3f})" if gate_b_pt else ""),
          f"best recall-preserving realneg FP/CCD: {usable_fp:.1f}",
          f"synthetic recall preserved          : {recall_ok and bar_ok}",
          "", f"DECISION: {decision}"]
    for r in reasons:
        L.append(f"  - {r}")
    if decision == "GO":
        L.append("  -> realneg fine-tune beats the promoted ft on the "
                 "kill-switch criteria; candidate for a promote path "
                 "(requires test_real re-score to confirm objects-gained).")
    else:
        L.append("  -> per design §10 this bounded experiment stops here; "
                 "v7-as-2nd-stage at a usable FP rate is not reached. "
                 "Bank the evidence; do not promote.")
    out = "\n".join(L) + "\n"
    print(out)
    (ed / "DECISION.txt").write_text(out)
    print(f"[saved] {ed}/DECISION.txt")


if __name__ == "__main__":
    main()
