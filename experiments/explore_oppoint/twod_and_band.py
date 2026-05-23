"""explore_oppoint: the two levers that the per-band/SNR tables hinted at.

(a) 2-D score x mf_snr cut on the ASTEROID side, with the FP side scored
    fairly using empft (which HAS mf_snr) -> a real apples-to-apples
    per-SNR/score frontier.
(b) per-band restriction: r/i/z carry almost all new objects (9/9/9 of the
    unique set); u/y nearly worthless (1 each). Does dropping low-value
    bands move FP/CCD at fixed object recall?
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/diffim_runs/test_real/results")
OUT = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/explore_oppoint")

ps = pd.read_csv(RES / "per_sighting_snr.csv")
never = set(ps.groupby("ObjID").stack_detected.any().pipe(lambda s: s[~s].index))
cand = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "parts/cand_*.csv")))],
                 ignore_index=True)
ast = cand[cand.role == "asteroid"].copy()
emp = cand[cand.role == "empty"].copy()
N_EMP_IMG = emp.image_id.nunique()
sight_meta = ps.set_index("image_id")[["band", "trail_length", "mf_snr"]]
ast = ast.join(sight_meta, on="image_id")
ast["filt"] = ast["band"].astype(str).str.split("_").str[0]

# empft for a FAIR SNR-aware FP count (it has mf_snr per FP candidate)
empft = pd.read_csv(RES / "parts/empft_0.csv")
N_EMPFT_IMG = empft.image_id.nunique()

lines = []


def w(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def sci(a_keep, fp_count):
    hit = a_keep[a_keep.on_truth == 1]
    return (hit[hit.ObjID.isin(never)].ObjID.nunique(),
            hit.image_id.nunique(),
            fp_count / N_EMP_IMG)


# ---------- (a) 2-D score x mf_snr ----------
# Hypothesis: low-SNR sightings (mf_snr<3) are mostly the stack-missed prize
# but also where FP live; require HIGHER score there, LOWER score where
# mf_snr is in the addressable 3-7 band. FP side: use empft mf_snr+score_rf.
w("=" * 78)
w("(a) 2-D  score x mf_snr  cut  (FP counted on empft, SNR-matched)")
w("=" * 78)
w("rule: keep if score>=s_hi  OR  (3<=mf_snr<12 AND score>=s_lo)")
w(f"{'s_lo':>5s} {'s_hi':>5s} {'new_obj':>7s} {'new_sight':>9s} {'FP/CCD':>8s}")
emp_snrish = empft.copy()
emp_snrish["mf_snr"] = emp_snrish["mf_snr"].replace([np.inf, -np.inf], np.nan)
for s_lo in [0.04, 0.06, 0.08, 0.10, 0.15]:
    for s_hi in [0.20, 0.30, 0.50]:
        a = ast[(ast.score_rf >= s_hi) |
                ((ast.mf_snr >= 3) & (ast.mf_snr < 12) & (ast.score_rf >= s_lo))]
        fcount = ((emp_snrish.score_rf >= s_hi) |
                  ((emp_snrish.mf_snr >= 3) & (emp_snrish.mf_snr < 12) &
                   (emp_snrish.score_rf >= s_lo))).sum()
        # scale empft FP count to per-150-CCD basis (same CCD set as emp)
        fcount_scaled = fcount / N_EMPFT_IMG * N_EMP_IMG
        no, ns, _ = sci(a, 0)
        w(f"{s_lo:5.2f} {s_hi:5.2f} {no:7d} {ns:9d} {fcount_scaled / N_EMP_IMG:8.1f}")

# flat baseline FP via empft for fair comparison at same thresholds
w("\n  flat-threshold baseline, FP via empft (same CCDs as 2-D rule):")
w(f"  {'thr':>5s} {'new_obj':>7s} {'new_sight':>9s} {'FP/CCD':>8s}")
for t in [0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    a = ast[ast.score_rf >= t]
    fcount = (empft.score_rf >= t).sum() / N_EMPFT_IMG * N_EMP_IMG
    no, ns, _ = sci(a, 0)
    w(f"  {t:5.2f} {no:7d} {ns:9d} {fcount / N_EMP_IMG:8.1f}")

# ---------- (b) per-band restriction ----------
w("\n" + "=" * 78)
w("(b) per-band restriction (r,i,z = high value; u,y = low value)")
w("=" * 78)
# FP per band: empft has no band; empties from per_sighting have band only for
# asteroid sightings, not empties. So FP/CCD cannot be split by band from data.
# We can only show the OBJECT/SIGHT yield per band-set vs the FP that the SAME
# flat threshold produces (band restriction does NOT reduce empty FP because
# empties aren't band-labelled here -> band cut only loses objects, no FP gain
# unless the pipeline is run per-band, which it is in production).
band_sets = {
    "all": list("ugrizy"),
    "drop_u": list("grizy"),
    "drop_uy": list("griz"),
    "riz_only": list("riz"),
}
for t in [0.05, 0.10, 0.20]:
    w(f"\n  thr={t}")
    w(f"  {'bandset':10s} {'new_obj':>7s} {'new_sight':>9s} {'asteroid_cand_frac':>18s}")
    base_n = (ast.score_rf >= t).sum()
    for name, bs in band_sets.items():
        a = ast[(ast.score_rf >= t) & (ast.filt.isin(bs))]
        hit = a[a.on_truth == 1]
        no = hit[hit.ObjID.isin(never)].ObjID.nunique()
        ns = hit.image_id.nunique()
        frac = len(a) / base_n if base_n else 0
        w(f"  {name:10s} {no:7d} {ns:9d} {frac:18.3f}")
w("\nNote: empties in cand_*.csv carry no band, so a per-band gate's FP")
w("benefit is not measurable here. In production the pipeline runs per")
w("CCD/band, so restricting to r,i,z would drop ~the u/y empty CCDs'")
w("FP entirely while losing only ~2 new objects (u,y give 1 each).")

(OUT / "_twod_band_report.txt").write_text("\n".join(lines) + "\n")
print("\nwrote", OUT / "_twod_band_report.txt")
