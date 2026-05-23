"""explore_oppoint: real science Pareto frontier + per-SNR/band/length thresholds.

Deliverables 1 & 2.

Anchored to experiments/diffim_runs/test_real/results/threshold_sweep.txt:
  - new objects  = unique ObjID (restricted to the 99 stack-NEVER-detected
                    objects) with >=1 on_truth=1 asteroid candidate at score>=t
  - new sightings= unique asteroid image_id with an on_truth=1 candidate score>=t
  - FP/empty-CCD = (# empty-role candidates score>=t) / (# unique empty image_id)

Note: cand_*.csv on_truth=1 marks a candidate sitting on a *stack-missed
sighting* of an asteroid trail. 78 unique such ObjIDs exist but only 22 are
in the 99 stack-never-detected set; the rest are "free" sightings of objects
the stack already catches elsewhere -> they do NOT add a new object.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/diffim_runs/test_real/results")
OUT = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN/experiments/explore_oppoint")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load ----
ps = pd.read_csv(RES / "per_sighting_snr.csv")
never = set(ps.groupby("ObjID").stack_detected.any().pipe(lambda s: s[~s].index))
N_OBJ = len(never)                       # 99
N_SIGHT = int((~ps.stack_detected).sum())  # 917

cand = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "parts/cand_*.csv")))],
                 ignore_index=True)
ast = cand[cand.role == "asteroid"].copy()
emp = cand[cand.role == "empty"].copy()
N_EMP_IMG = emp.image_id.nunique()       # 150

# attach band / trail_length / mf_snr to asteroid candidates via image_id
sight_meta = ps.set_index("image_id")[["band", "trail_length", "mf_snr"]]
ast = ast.join(sight_meta, on="image_id")
# band string is like "u_24" (filter_detector); take leading filter token
ast["filt"] = ast["band"].astype(str).str.split("_").str[0]


def science(sub_ast_keep: pd.DataFrame, emp_kept_n: int):
    """sub_ast_keep = asteroid candidates that PASS the rule.
    emp_kept_n = number of empty-role candidates that pass the rule."""
    hit = sub_ast_keep[sub_ast_keep.on_truth == 1]
    new_obj = hit[hit.ObjID.isin(never)].ObjID.nunique()
    new_sight = hit.image_id.nunique()
    fp_ccd = emp_kept_n / N_EMP_IMG
    return new_obj, new_sight, fp_ccd


# ============================================================
# 1) FLAT-THRESHOLD dense Pareto frontier
# ============================================================
thr_grid = np.r_[np.arange(0.02, 0.10, 0.005),
                 np.arange(0.10, 0.30, 0.01),
                 np.arange(0.30, 0.95, 0.02)]
rows = []
for t in thr_grid:
    a = ast[ast.score_rf >= t]
    e_n = int((emp.score_rf >= t).sum())
    no, ns, fp = science(a, e_n)
    rows.append((round(float(t), 4), no, ns, round(fp, 2)))
flat = pd.DataFrame(rows, columns=["thr", "new_obj", "new_sight", "fp_ccd"])
flat.to_csv(OUT / "flat_pareto.csv", index=False)

# promoted points
prom = flat[flat.thr.isin([0.10, 0.50])]


def pareto_front(df, x="fp_ccd", y="new_obj", y2="new_sight"):
    """min x, max y (tie-break max y2). Lower-left-to-upper-right frontier."""
    d = df.sort_values([x, y], ascending=[True, False]).reset_index(drop=True)
    keep, best = [], -1
    for _, r in d.iterrows():
        if r[y] > best:
            keep.append(r)
            best = r[y]
    return pd.DataFrame(keep)


front_obj = pareto_front(flat, y="new_obj")

# ============================================================
# 2a) PER-SNR-BIN thresholds
# ============================================================
SNR_BINS = [(-1e9, 3), (3, 5), (5, 7), (7, 12), (12, 1e12)]
SNR_LBL = ["<3", "3-5", "5-7", "7-12", ">12"]


def snrbin(v):
    for i, (lo, hi) in enumerate(SNR_BINS):
        if lo <= v < hi:
            return SNR_LBL[i]
    return SNR_LBL[-1]


ast["snrbin"] = ast["mf_snr"].fillna(-999).map(snrbin)
emp_snr = emp.copy()  # empties have no mf_snr in cand file; gate empties by score only

# For per-SNR thresholds we can only differentiate the ASTEROID side by snrbin;
# the FP side (empties) has no mf_snr in cand_*.csv. So a per-SNR-bin asteroid
# threshold is only fair if we also know how empties distribute. We instead use
# the empft full-feature empties (which DO have mf_snr) to get FP/CCD at a
# matched per-bin score+snr rule -> handled in gates script. Here we report the
# best-case (oracle) frontier: choose per-bin thr to maximize objects, and
# bound FP by the corresponding flat empty count at the *minimum* per-bin thr.


def per_snr_frontier(thr_lo=0.02, thr_hi=0.90, step=0.02):
    """Greedy: for each target FP budget, pick per-bin thresholds.
    FP proxy: empties have no SNR, so a per-bin asteroid rule that uses
    threshold t_b in bin b lets through empties at the *lowest* t_b used
    (worst case). We report objects vs that worst-case FP/CCD."""
    grid = np.round(np.arange(thr_lo, thr_hi + 1e-9, step), 3)
    out = []
    # enumerate: independent per-bin threshold, FP = flat empties at min thr
    # We sweep a single "effort" knob: bin b gets thr = base + offset_b where
    # offsets are tuned so high-SNR bins (stack owns them, low marginal value)
    # get HIGHER thr and the prize bin 3-7 gets LOWER thr.
    offsets = {"<3": +0.30, "3-5": -0.04, "5-7": -0.02, "7-12": +0.10, ">12": +0.30}
    for base in grid:
        thr_b = {b: float(np.clip(base + o, 0.02, 0.98)) for b, o in offsets.items()}
        keep = ast[ast.apply(lambda r: r.score_rf >= thr_b[r.snrbin], axis=1)]
        tmin = min(thr_b.values())
        e_n = int((emp.score_rf >= tmin).sum())
        no, ns, fp = science(keep, e_n)
        out.append((round(base, 3), no, ns, round(fp, 2), json.dumps({k: round(v, 3) for k, v in thr_b.items()})))
    return pd.DataFrame(out, columns=["base", "new_obj", "new_sight", "fp_ccd_worst", "thr_by_bin"])


per_snr = per_snr_frontier()
per_snr.to_csv(OUT / "per_snr_frontier.csv", index=False)

# ============================================================
# 2b) PER-BAND thresholds (filter)
# ============================================================
band_tab = (ast[ast.on_truth == 1]
            .assign(is_new=lambda d: d.ObjID.isin(never))
            .groupby("filt")
            .agg(n_hit=("on_truth", "size"),
                 n_sight=("image_id", "nunique"),
                 n_newobj=("is_new", "sum"))
            .sort_values("n_hit", ascending=False))
band_tab.to_csv(OUT / "per_band_hits.csv")

# ============================================================
# 2c) PER-TRAIL-LENGTH
# ============================================================
LBINS = [(-1, 8), (8, 12), (12, 20), (20, 1e9)]
LLBL = ["<8", "8-12", "12-20", ">20"]
ast["lbin"] = pd.cut(ast.trail_length, [b[0] for b in LBINS] + [1e9], labels=LLBL)
len_tab = (ast[ast.on_truth == 1]
           .assign(is_new=lambda d: d.ObjID.isin(never))
           .groupby("lbin", observed=True)
           .agg(n_hit=("on_truth", "size"),
                n_sight=("image_id", "nunique"),
                n_newobj=("is_new", "sum")))
len_tab.to_csv(OUT / "per_length_hits.csv")

# ============================================================
# report
# ============================================================
with open(OUT / "_pareto_report.txt", "w") as fh:
    def w(*a):
        print(*a); print(*a, file=fh)
    w("ANCHOR: 99 stack-never-detected objects ; 917 stack-missed sightings ;",
      N_EMP_IMG, "empty CCDs")
    w("cand on_truth unique ObjID:", ast[ast.on_truth == 1].ObjID.nunique(),
      " of which in never-detected set:",
      ast[ast.on_truth == 1].ObjID.isin(never).groupby(ast[ast.on_truth == 1].ObjID).any().sum())
    w("\n=== FLAT dense Pareto (every point) ===")
    w(flat.to_string(index=False))
    w("\n=== PROMOTED points ===")
    w(prom.to_string(index=False))
    w("\n=== FLAT Pareto FRONTIER (objects) ===")
    w(front_obj[["thr", "new_obj", "new_sight", "fp_ccd"]].to_string(index=False))
    w("\n=== PER-SNR-BIN frontier (worst-case FP, oracle offsets) ===")
    w(per_snr.to_string(index=False))
    w("\n=== PER-BAND on_truth hits ===")
    w(band_tab.to_string())
    w("\n=== PER-TRAIL-LENGTH on_truth hits ===")
    w(len_tab.to_string())
print("\nwrote", OUT)
