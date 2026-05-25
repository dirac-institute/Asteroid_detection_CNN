"""Stage 3 — FALSE-POSITIVE CLEANING of the trailed-detection catalog (the step the pipeline
still needs to make linking productive).

On real diffims the trailed-detection set is FP-dominated (~18k/night of subtraction residuals /
instrumental artefacts that the Veres model happily fits as trails). HelioLinC then drowns in
chance-alignment links. This stage cuts the catalog down to plausible real trails BEFORE linking.

Currently applies the cheap, defensible cuts (fast, vectorised pandas):
  - RF score              >= score_min        (detector confidence)
  - Veres reduced chi^2   <= rchisq_max        (a real trail fits the trailed-PSF model; junk doesn't)
  - de-biased length      >= lendb_min         (fast movers; sub-px trails carry no velocity)

>>> FP-CLEANING HOOK (the real missing work) <<<
The cheap cuts above are necessary but NOT sufficient — see notes/simreal-gap. Add here, in order
of expected impact:
  1. diaSource real/bogus RELIABILITY join: match each detection to the Rubin diaSource catalog and
     keep only high-reliability (real/bogus score) sources — the single biggest FP reducer on DP2.
  2. trail-COHERENCE cut: require the flux to be coherent along the fitted line (e.g. fraction of the
     line's pixels above N·sigma, or low residual structure) — rejects dipoles / star-halo residuals
     that fit a line but aren't trails.
  3. de-duplication / merge of multiple detections of the same trail on a panel.
Each is a pure post-processing filter on this catalog; wire it as another vectorised mask below.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame, *, score_min: float, rchisq_max: float, lendb_min: float) -> pd.DataFrame:
    n0 = len(df)
    m = np.ones(n0, bool)
    if "score_rf" in df:    m &= df.score_rf.to_numpy() >= score_min
    if "veres_rchi" in df:  m &= df.veres_rchi.to_numpy() <= rchisq_max
    if "len_db" in df:      m &= df.len_db.to_numpy() >= lendb_min
    # --- FP-CLEANING HOOK: add reliability / coherence / dedup masks here (see module docstring) ---
    out = df[m].copy()
    print(f"[clean] {n0} -> {len(out)} detections "
          f"(score>={score_min}, rChiSq<={rchisq_max}, len_db>={lendb_min}px); "
          f"per-night now {len(out)//max(out['mjd'].astype(int).nunique(),1) if len(out) else 0}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="Veres-measured detection catalog")
    ap.add_argument("--out", required=True, help="cleaned catalog (input to linking)")
    ap.add_argument("--score-min", type=float, default=0.5)
    ap.add_argument("--rchisq-max", type=float, default=1.5)
    ap.add_argument("--lendb-min", type=float, default=6.0)
    a = ap.parse_args()
    df = pd.read_csv(a.inp)
    out = clean(df, score_min=a.score_min, rchisq_max=a.rchisq_max, lendb_min=a.lendb_min)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[clean] wrote {len(out)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
