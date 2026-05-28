"""Stage 3 — FALSE-POSITIVE CLEANING + source merge, tuned to NOT lose low-SNR trails.

Two detection streams feed the linker, cleaned DIFFERENTLY (validated on truth):
  • diaSources (5σ stack): apply Rubin's real/bogus RELIABILITY cut (≥thr & !isNegative).
      On truth this keeps 95.7% of TP while removing 93.6% of FP — a clean, TP-safe FP cleaner
      for the stack stream.
  • ADCNN detections: DO NOT apply any FP/score cut here. The stage-2 false-positive cut already
      happened ONCE upstream (the CNN score at detect); re-thresholding the same score here would be
      a redundant, contradictory second filter. Nor do we apply real/bogus (tested,
      rb_synthetic_test.py: its SNR floor + >30px trail ceiling would discard exactly ADCNN's
      faint/fast trails). ADCNN's residual FP are left for multi-epoch ORBITAL LINKING to reject (a
      faint real trail recurs on a consistent orbit; faint noise does not). See
      [[realbogus-fp-filter-limits]], [[linking-needs-recall]].

Both streams are restricted to fast/trailed candidates by the **Veres-measured** trail length
(len_db ≥ lendb_min ~6px ≈ 1 deg/day — the accurate forward-model length, not the ADCNN estimate)
and merged into one catalog with sky endpoints for trail_tracklets.

    python clean_fp.py --adcnn adcnn_dets_veres.csv --dia diasources.csv --out dets_clean.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["detid", "mjd", "ra", "dec", "ra0", "dec0", "ra1", "dec1", "len_db",
        "flux", "snr", "mag", "mag_err", "band", "obscode", "score", "source"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adcnn", required=True, help="ADCNN Veres-measured catalog (sky endpoints, len_db)")
    ap.add_argument("--dia", default=None, help="stack diaSource catalog (reliability, trailLength)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dia-reliability-min", type=float, default=0.5, help="real/bogus cut (diaSources ONLY)")
    ap.add_argument("--lendb-min", type=float, default=6.0,
                    help="Veres-measured trail length px (~1 deg/day) — the fast-mover cut (accurate length)")
    a = ap.parse_args()

    parts = []
    # --- ADCNN stream: keep fast/trailed by the ACCURATE Veres length; NO score cut (done once at
    #     detect), NO real/bogus (preserve low-SNR trails) -> linking rejects residual FP. ---
    ad = pd.read_csv(a.adcnn)
    n0 = len(ad)
    if "len_db" in ad:                       # len_db here = Veres-fit length (adcnn_dets_veres.csv)
        ad = ad[ad.len_db >= a.lendb_min]
    ad["source"] = "adcnn"
    print(f"[clean] ADCNN {n0} -> {len(ad)} (Veres len_db>={a.lendb_min}px; no score cut, no real/bogus)", flush=True)
    parts.append(ad)

    # --- diaSource stream: real/bogus reliability cut (TP-safe FP cleaner), trailed ---
    if a.dia and Path(a.dia).exists():
        dia = pd.read_csv(a.dia); m0 = len(dia)
        if "reliability" in dia:
            keep = (dia.reliability >= a.dia_reliability_min)
        else:                                        # no reliability column -> keep all (don't crash)
            keep = pd.Series(True, index=dia.index)
        if "isNegative" in dia:
            keep &= ~dia.isNegative.astype(bool)
        if "trailLength" in dia:
            keep &= dia.trailLength >= a.lendb_min
        dia = dia[keep].copy()
        # diaSource carries trailLength but not endpoints -> trail_tracklets/Veres adds them; map names
        if "len_db" not in dia and "trailLength" in dia:
            dia["len_db"] = dia.trailLength
        dia["source"] = "dia"
        dia["score"] = dia.get("reliability", np.nan)
        print(f"[clean] diaSource {m0} -> {len(dia)} (reliability>={a.dia_reliability_min} & !neg & trailLen>={a.lendb_min})", flush=True)
        parts.append(dia)

    out = pd.concat(parts, ignore_index=True, sort=False)
    keep_cols = [c for c in COLS if c in out.columns]
    out = out[keep_cols]
    nnight = int(np.floor(out["mjd"] - 0.5).nunique()) if len(out) else 1   # floor-to-night, not trunc
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[clean] merged -> {len(out)} detections ({len(out)//max(nnight,1)}/night) "
          f"[ADCNN={(out.source=='adcnn').sum()}, dia={(out.source=='dia').sum()}] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
