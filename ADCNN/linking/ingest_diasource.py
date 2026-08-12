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
RING SAFETY -- CURRENTLY ABSENT, READ THIS BEFORE TRUSTING THE MERGE. The intent was to drop sources
whose dipole flux difference dominates, using the stack's own dipoleFluxDiff/dipoleMeanFlux. On the
DRP output we actually consume those are NaN on 72.4% of rows and EXACTLY 0 on the rest, so the rule
flags nothing: nine delivered nights, 3,880,041 sources, zero dropped. The stack side is therefore NOT
ring-cleaned, while the ADCNN side is (25.5% of its rows dropped as rings before the union). Measured
counterfactually from pixels, 41.9% of stack rows would be flagged is_dipole and 60.6% would fail
--art-frac-max. `is_dipole` and `art_frac` are consequently written as NaN, not False/0.0, so no
downstream gate mistakes "never measured" for "measured clean".
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
                                   "x", "y", "psfFlux",
                                   # trail QUALITY + trail-likeness, previously discarded. trailFlux
                                   # over psfFlux separates trailed from point sources on the stack's
                                   # own terms (measured median 1.267 at len>=6 px vs 1.052 below),
                                   # and extendedness likewise (0.867 vs 0.999) -- a native gate for
                                   # deciding which unmeasured rows are worth re-measuring.
                                   "trailFlux", "trailFluxErr", "trail_flag_edge", "extendedness")})
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
        # THIS DROP CANNOT FIRE ON THE CURRENT DRP OUTPUT, and printing "stack-dipole: 0" makes that
        # indistinguishable from "no dipoles present". MEASURED on 17,097 real 0706 rows:
        # dipoleMeanFlux/dipoleFluxDiff are NaN on 72.4%, and on the 27.6% that ARE present
        # dipoleFluxDiff is EXACTLY 0 on 100.0%. Across nine delivered nights the log reads
        # "stack-dipole: 0" every time -- 3,880,041 sources, zero dropped. The module docstring's
        # "RING SAFETY" claim rests on this, so it is currently fiction.
        _dm = d["dipoleMeanFlux"]; _dd = d["dipoleFluxDiff"]
        _usable = (_dm.notna() & _dd.notna() & (_dd.abs() > 0)).sum()
        if _usable == 0:
            print(f"[diasrc] WARNING: the stack-dipole drop is INERT -- dipoleFluxDiff is absent or "
                  f"exactly zero on all {len(d):,} rows, so it can flag nothing. The stack side is "
                  f"therefore NOT ring-cleaned by this path; is_dipole must be measured from pixels "
                  f"(see ADCNN.linking.measure_stack_trails) before these rows are made linkable.",
                  flush=True)
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
    # DO NOT fillna(0). `trailLength` is NaN on ~31% of dia_source_detector rows -- MEASURED on the
    # collection the nights actually use, LSSTCam/runs/prompt/20260706/ApPipe in the `embargo` repo
    # (30.9% of 2,738 rows; 25.7% of delivered stack rows carry the len_db==0 & beta==0 signature).
    # An earlier 69% figure quoted here came from dp2_prep/DM-53881 stage4, whose visits are 2025-04
    # and which the night pipeline does not read -- do not requote it. The DPDD trailed
    # fit is simply not available for most sources -- and mapping that to 0.0 makes "never measured"
    # indistinguishable from "measured as a point source". Both then fail the op's len_db_min of 6.0,
    # so two thirds of the stack's contribution is discarded for a measurement that was never
    # attempted rather than for being short. Keeping NaN lets a consumer route those rows to a
    # different estimator instead of silently killing them.
    #
    # NB the plugin columns are NOT an option here: dia_source_detector in this collection carries
    # trailFlux/trailFluxErr/trailRa/trailDec/trailLength/trailAngle/trail_flag_edge, and NO
    # ext_trailedSources_Naive_* or _Veres_* -- those plugins are not run in the DRP that produces it.
    # Where trailLength IS present it behaves correctly: median 0.54" with 84% sub-PSF, which is what
    # a trailed fit should return for the point sources that dominate a DIASource table, and it
    # reaches 48.6 px on genuine trails.
    if "trailLength" in d:
        L_px = d["trailLength"] / PIXEL_SCALE
        n_unmeasured = int(L_px.isna().sum())
        if n_unmeasured:
            print(f"[diasrc] {n_unmeasured:,} of {len(d):,} sources ({100*n_unmeasured/len(d):.1f}%) have NO "
                  f"trailLength -- kept as NaN, NOT 0. They cannot pass a len_db floor and need a "
                  f"separate trail measurement to become linkable.", flush=True)
    else:
        L_px = pd.Series(np.nan, index=d.index)
    beta = d["trailAngle"] if "trailAngle" in d else pd.Series(np.nan, index=d.index)
    # endpoints are NaN wherever the length is: a row with no trail measurement has no trail, and the
    # linker's trail-vs-chord terms must see that rather than a zero-length segment at the centroid
    ra0, dec0, ra1, dec1 = _endpoints(d.ra.to_numpy(), d.dec.to_numpy(),
                                      L_px.to_numpy(), beta.fillna(0.0).to_numpy())
    out = pd.DataFrame(dict(
        mjd=d["midpointMjdTai"], ra=d["ra"], dec=d["dec"], mag=np.nan, band="r", obscode="I11",
        visit=d["visit"], detector=d["detector"],
        x=d.get("x", pd.Series(np.nan, index=d.index)), y=d.get("y", pd.Series(np.nan, index=d.index)),
        score=_reliability(rel, keep, n0),
        length=L_px, len_db=L_px, mf_snr=d.get("snr", pd.Series(np.nan, index=d.index)),
        ra0=ra0, dec0=dec0, ra1=ra1, dec1=dec1, beta=beta,
        ssObjectId=d.get("ssObjectId", pd.Series(0, index=d.index)),
        trail_flux=d.get("trailFlux", pd.Series(np.nan, index=d.index)),
        trail_flux_err=d.get("trailFluxErr", pd.Series(np.nan, index=d.index)),
        trail_flag_edge=d.get("trail_flag_edge", pd.Series(np.nan, index=d.index)),
        extendedness=d.get("extendedness", pd.Series(np.nan, index=d.index)),
        # NOT MEASURED -- do not write False/0.0. Those are the values that mean "clean", and they
        # disarm two gates that demonstrably fire for ADCNN rows: link_2visit's --art-frac-max (0.3)
        # and its pre-seed dipole veto (which removes 113,207 of 441,651 ADCNN dets on a real night).
        # COUNTERFACTUAL on real pixels, recomputed with the SAME code the ADCNN side uses and
        # validated to reproduce the ADCNN catalogue's art_frac exactly (max |diff| 0.000 on 4,598
        # rows): of 1,200 random stack rows, 60.6% would FAIL --art-frac-max and 41.9% would be
        # flagged is_dipole. On the merged 0706 stream, 8 of the 33 stack-member alerts fail the
        # measured art_frac and 4 of those sit inside the 1000-alert budget, one at rank 24.
        # That exposure scales with any fix that makes more stack rows linkable, so these must be
        # MEASURED (measure_stack_trails has the panel open already), not defaulted.
        is_dipole=np.nan, art_frac=np.nan, src="stack"))
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
