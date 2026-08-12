#!/usr/bin/env python3
"""Ingest the STACK's DIA sources (`dia_source_detector`) into the ADCNN detection schema, so the
linker can consume ADCNN and stack detections together.

WHY: the two detectors are complementary, not redundant. The stack's diaSource+reliability finds MORE
catalogued objects than ADCNN and is ~6x cleaner, but it misses the sub-5sigma faint end that ADCNN
exists for (measured: union recall 62.1% vs stack 58.9 / ADCNN 52.4). Merging lifts the BRIGHT end and
-- because chance-link rate scales as n1*n2 -- a cleaner combined list also cuts the noise chance-links
that dominate the residual 2-visit contamination once rings are removed.

Emitted columns match the ADCNN catalogue the linker reads:
  mjd ra dec visit detector x y score length len_db mf_snr beta ra0 dec0 ra1 dec1 + provenance `src`
  score   <- `reliability` (the stack's real/bogus; same [0,1] sense as the ADCNN CNN score)
  mf_snr  <- `snr`
  length  <- `trailLength` converted arcsec -> pixels (PIXEL_SCALE); beta <- `trailAngle`
  ra0/1   <- trail endpoints from (trailLength, trailAngle), the same convention detect_night writes
RING SAFETY: the stack measures dipoles itself (dipoleFluxDiff / dipoleMeanFlux / dipoleLength). We
drop sources whose dipole flux difference dominates -- the stack's own statement that the source is a
subtraction dipole -- so the merge cannot re-import the ring population ADCNN was just cleaned of.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

PIXEL_SCALE = 0.2          # arcsec/pixel (LSSTCam)


def _reliability(rel, keep, n0):
    """The stack's real/bogus score, or a REFUSAL -- never a silent 1.0.

    This was `rel[...] if len(rel) == n0 else 1.0`, i.e. one absent or mis-sized `reliability` column
    away from putting EVERY stack detection at the maximum possible score. That is not a benign
    default: P(real) is a logistic fit on (score_min, chi2, mfsnr_min), `--claim-order preal` claims
    the highest P(real) first, and a detection can only be claimed once -- so max-scored stack pairs
    outrank real ADCNN pairs and take their detections. MEASURED in the injection harness, where
    score=1.0 IS hard-coded: stack-containing alerts are 27.8% of the merged stream but 97.3% of the
    top 2000 by priorityScore, with WORSE orbit chi2 (16.68 vs 14.43), and the merged arm links 84
    fewer true movers than ADCNN alone. Delivered nights are NOT in this state (0706 stack score is a
    real distribution, median 0.6228) -- this guard is so they cannot silently enter it.
    """
    if len(rel) == n0:
        return rel[keep.values].reset_index(drop=True)
    raise SystemExit(
        f"[ingest_diasource] the stack table has no usable `reliability` column "
        f"({len(rel)} values for {n0} rows). Defaulting it to 1.0 would put every stack detection at "
        f"the maximum P(real) input and let stack pairs win the claim competition against real ADCNN "
        f"pairs. Supply reliability, or pass --no-stack-score to link with the stack demoted.")


def _endpoints(ra, dec, length_px, beta_deg):
    """Trail endpoints: half-length along the trail PA. Matches detect_night's convention so the
    linker's chord/trail geometry is identical for both detectors."""
    L_deg = np.clip(length_px, 0, None) * PIXEL_SCALE / 3600.0
    b = np.radians(beta_deg)
    cd = np.cos(np.radians(dec))
    # trailAngle is a SKY position angle measured NORTH->EAST. Mapping it as if measured from EAST
    # mirrors every trail about the NE diagonal (PA -> 90-PA), randomising its direction.
    # MEASURED on 1,127 stack<->ADCNN matched detections: as shipped the trail-PA disagreement with
    # ADCNN was median 50.8 deg (frac>20deg 0.826, vs 45.0/0.556 for an unrelated reference);
    # with sin/cos swapped it is median 4.4 deg (frac>20deg 0.115). 83% of stack trails were
    # therefore beyond pair_chi2's hardcoded dpa_tm>20 kill, and because the PA was RANDOMISED
    # rather than absent, the ~17% that passed did so by chance -- the chi2 PA term carried no
    # information for stack members. N->E means the East component goes with sin, North with cos.
    hdx = 0.5 * L_deg * np.sin(b) / np.maximum(cd, 1e-6)
    hdy = 0.5 * L_deg * np.cos(b)
    return ra - hdx, dec - hdy, ra + hdx, dec + hdy


def ingest(butler_repo, collection, night, out_path, reliability_min=0.5, snr_min=0.0,
           drop_dipoles=True, dipole_frac_max=0.6, visits=None):
    from lsst.daf.butler import Butler
    b = Butler(butler_repo)
    # `day_obs` depends on `instrument`, so the registry requires the instrument constrained too.
    kw = {}
    if night:
        kw = dict(where=f"instrument='LSSTCam' AND exposure.day_obs = {int(night)}")
    try:
        refs = list(b.registry.queryDatasets("dia_source_detector", collections=collection,
                                             findFirst=True, **kw))
    except Exception as e:
        print(f"[diasrc] day_obs query failed ({type(e).__name__}); falling back to the whole "
              f"collection (it is a single night's prompt run anyway)", flush=True)
        refs = list(b.registry.queryDatasets("dia_source_detector", collections=collection,
                                             findFirst=True))
    if visits:
        vs = set(int(v) for v in visits)
        refs = [r for r in refs if int(r.dataId["visit"]) in vs]
    print(f"[diasrc] {len(refs)} detector-visit tables to read", flush=True)
    parts = []
    for i, r in enumerate(refs):
        try:
            t = b.get(r)
        except Exception as e:
            print(f"[diasrc]   skip {r.dataId}: {type(e).__name__}", flush=True); continue
        if not len(t):
            continue
        d = pd.DataFrame({c: np.asarray(t[c]) for c in t.columns
                          if c in ("ra", "dec", "midpointMjdTai", "snr", "reliability", "trailLength",
                                   "trailAngle", "ssObjectId", "dipoleMeanFlux", "dipoleFluxDiff",
                                   "x", "y", "psfFlux")})
        d["visit"] = int(r.dataId["visit"]); d["detector"] = int(r.dataId["detector"])
        parts.append(d)
        if (i + 1) % 200 == 0:
            print(f"[diasrc]   {i+1}/{len(refs)} tables, {sum(len(p) for p in parts):,} sources", flush=True)
    if not parts:
        raise SystemExit("[diasrc] no dia_source_detector rows found -- wrong collection/night?")
    d = pd.concat(parts, ignore_index=True)
    n0 = len(d)
    # quality + ring cuts ------------------------------------------------------------------
    rel = d["reliability"].fillna(0.0) if "reliability" in d else pd.Series(1.0, index=d.index)
    keep = (rel >= reliability_min)
    if snr_min > 0 and "snr" in d:
        keep &= d["snr"].fillna(0) >= snr_min
    n_rel = int((~keep).sum())
    n_dip = 0
    if drop_dipoles and {"dipoleFluxDiff", "dipoleMeanFlux"} <= set(d.columns):
        # the stack's own dipole statement: |flux difference| comparable to the mean lobe flux
        frac = (d["dipoleFluxDiff"].abs() /
                d["dipoleMeanFlux"].abs().replace(0, np.nan)).fillna(0.0)
        isdip = frac > dipole_frac_max
        n_dip = int((keep & isdip).sum())
        keep &= ~isdip
    d = d[keep].reset_index(drop=True)
    print(f"[diasrc] {n0:,} sources -> {len(d):,} kept "
          f"(reliability<{reliability_min} or snr cut: {n_rel:,}; stack-dipole: {n_dip:,})", flush=True)
    # -> ADCNN schema ----------------------------------------------------------------------
    L_px = (d["trailLength"].fillna(0.0) / PIXEL_SCALE) if "trailLength" in d else pd.Series(0.0, index=d.index)
    beta = d["trailAngle"].fillna(0.0) if "trailAngle" in d else pd.Series(0.0, index=d.index)
    ra0, dec0, ra1, dec1 = _endpoints(d.ra.to_numpy(), d.dec.to_numpy(), L_px.to_numpy(), beta.to_numpy())
    out = pd.DataFrame(dict(
        mjd=d["midpointMjdTai"], ra=d["ra"], dec=d["dec"], mag=np.nan, band="r", obscode="I11",
        visit=d["visit"], detector=d["detector"],
        x=d.get("x", pd.Series(np.nan, index=d.index)), y=d.get("y", pd.Series(np.nan, index=d.index)),
        score=_reliability(rel, keep, n0),
        length=L_px, len_db=L_px, mf_snr=d.get("snr", pd.Series(np.nan, index=d.index)),
        ra0=ra0, dec0=dec0, ra1=ra1, dec1=dec1, beta=beta,
        ssObjectId=d.get("ssObjectId", pd.Series(0, index=d.index)),
        is_dipole=False, art_frac=0.0, src="stack"))
    out.to_csv(out_path, index=False)
    known = int((out.ssObjectId.fillna(0) > 0).sum())
    print(f"[diasrc] wrote {len(out):,} stack detections -> {out_path} "
          f"({known:,} already associated to a known solar-system object)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--butler-repo", default="embargo")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--night", type=int, default=None, help="day_obs, e.g. 20260706")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reliability-min", type=float, default=0.5,
                    help="stack real/bogus floor (its reliability is well calibrated; 0.5 is the "
                         "documented split between low/high reliability products)")
    ap.add_argument("--snr-min", type=float, default=0.0, help="0 = keep the faint end too")
    ap.add_argument("--no-drop-dipoles", action="store_true")
    a = ap.parse_args(argv)
    ingest(a.butler_repo, a.collection, a.night, a.out, a.reliability_min, a.snr_min,
           drop_dipoles=not a.no_drop_dipoles)


if __name__ == "__main__":
    main()
