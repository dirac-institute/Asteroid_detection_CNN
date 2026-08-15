#!/usr/bin/env python3
"""INJECTION HARNESS v2 -- unbiased placement, magnitude grid, and a LOSS CASCADE.

Fixes vs v1:
  1. RANDOM SKY placement inside each panel. v1 seeded movers at existing DETECTION positions; 45.5%
     of ADCNN detections sit on stars (they are ring residuals) vs 1.8% of random sky, which made every
     star-based veto look ~10x more costly than it is.
  2. MAGNITUDE GRID (19.5-23.5). v1 used 21.5-23.2 only -- the extreme faint end -- so the ~5%
     completeness it reported was a corner of the curve, not the curve.
  3. Per-injection LOSS CASCADE: for every mover we record whether it was detected in visit A, in
     visit B, survived the linker's input cuts, and ended up in an alert. That localises WHERE the
     recall is lost instead of only reporting that it is.

Trail length is set by rate (rate * exptime), so slow movers are short-trailed by construction.
Trails are defined by SKY endpoints and mapped through each panel's WCS, so PA/length are correct.
"""
import os, sys, numpy as np, pandas as pd, torch
from astropy.wcs import WCS

SOLARDAY = 86400.0; EXPTIME = 30.0; PIX = 0.2
# TRAIL LENGTH is the primary axis (it is what the detector actually measures); the rate follows from
# it, since the trail IS the motion smeared over the exposure:  rate[deg/day] = L_px * PIX/3600 * 86400/EXPTIME.
# All of these are >1 deg/day, the scoped faint-FAST band.
TRAIL_PX = [7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0]
SNR_LO, SNR_HI = 2.0, 10.0                 # UNIFORM in SNR, the scoped faint band
M5 = 24.0                                  # same default as build_ft_dataset
PSF_FWHM_PX = 3.0


def mag_for_snr(snr, m5, trail_px):
    """Project SNR->mag model (build_ft_dataset._mag_for_snr / sim_orbits): point-source m5 with the
    sqrt(L/FWHM) TRAIL-DILUTION term -- a trailed source spreads its flux, so reaching a given SNR
    needs more flux the longer the trail. Using SNR (not a flat mag grid) is what makes completeness
    comparable across rates, since rate SETS the trail length."""
    dil = np.sqrt(np.maximum(trail_px, PSF_FWHM_PX) / PSF_FWHM_PX)
    return m5 - 2.5 * np.log10(np.maximum(snr, 1e-3) * dil / 5.0)


FOV_DEG = 1.75          # LSSTCam field radius


def visit_groups(vc, n_epochs=2, win_min=52.0, dt_min_min=10.0, sep_deg=0.3,
                 positions=None, fov_deg=FOV_DEG, limit=None):
    """Visit groups to inject a mover into: PAIRS (n_epochs=2) or common-footprint TRIPLES (3).

    n_epochs=2 keeps the original criterion exactly -- two visits whose BORESIGHTS sit within
    `sep_deg`, separated by (dt_min_min, win_min] minutes -- so the validated 2-epoch truth sets
    reproduce unchanged.

    n_epochs=3 CANNOT use that criterion, and this is the whole reason the 3+visit tier had never
    been truth-validated. Requiring three boresights within 0.3 deg of each other yields ZERO triples
    on every night measured (0 at 1.0 deg too), yet the pipeline finds 23 real 3+visit tracks across
    the nine embargo nights -- because three ~3.5 deg-wide fields can share a footprint with their
    BORESIGHTS up to 3.5 deg apart. Boresight proximity is simply the wrong test for a triple.

    The right test is a COMMON FOOTPRINT: three visits that all cover the same sky, with the widest
    leg still inside the linking window. Measured that way the same nights offer 353-850 triples
    each, over 44-60% of detections, with arcs 4.4-45 min -- matching the real tracks' 4.1-51.7 min.

    `positions` is an (N,2) array of real (ra, dec) used to sample where the sky was actually
    observed; a triple counts only if it co-covers at least one such position. Returns a list of
    (visits_tuple, widest_dt_min), shortest-arc first.
    """
    import itertools
    vs = vc.index.to_numpy()
    ra, dec, mjd = vc.ra.to_numpy(), vc.dec.to_numpy(), vc.mjd.to_numpy()

    def _dt(i, j):
        return abs(mjd[j] - mjd[i]) * 1440.0

    if n_epochs == 2:
        out = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                dt = _dt(i, j)
                if not (dt_min_min < dt <= win_min):
                    continue
                if np.hypot((ra[i] - ra[j]) * np.cos(np.radians(dec[i])), dec[i] - dec[j]) < sep_deg:
                    out.append(((vs[i], vs[j]), dt))
        out.sort(key=lambda t: t[1])
        return out[:limit] if limit else out

    if positions is None or not len(positions):
        raise ValueError("n_epochs=3 needs `positions` (observed sky) to find common footprints")
    seen = {}
    for pra, pdec in np.asarray(positions, float):
        d = np.hypot((ra - pra) * np.cos(np.radians(pdec)), dec - pdec)
        m = np.flatnonzero(d < fov_deg)
        if len(m) < 3:
            continue
        for c in itertools.combinations(m.tolist(), 3):
            span = (mjd[list(c)].max() - mjd[list(c)].min()) * 1440.0
            if span <= win_min:
                seen.setdefault(tuple(vs[list(c)]), span)
    out = sorted(seen.items(), key=lambda t: t[1])
    return out[:limit] if limit else out


def sky_trail_to_pixel(w, ra, dec, L_deg, pa_deg):
    cd = np.cos(np.radians(dec))
    dra = 0.5 * L_deg * np.cos(np.radians(pa_deg)) / max(cd, 1e-6)
    ddec = 0.5 * L_deg * np.sin(np.radians(pa_deg))
    (x0, y0), (x1, y1) = w.all_world2pix([[ra - dra, dec - ddec], [ra + dra, dec + ddec]], 0)
    return 0.5*(x0+x1), 0.5*(y0+y1), float(np.hypot(x1-x0, y1-y0)), float(np.degrees(np.arctan2(y1-y0, x1-x0)))


def main():
    V = "outputs/runs/pa_validate"
    RUN = os.environ.get("INJ_RUN", "outputs/runs/ringpipe_0706")
    TAG = os.environ.get("INJ_TAG", "v2")
    print(f"[inj] run={RUN} tag={TAG}", flush=True)
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_dets = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    per_panel = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    from ADCNN.inference.diffim_io import open_diffim
    from ADCNN.pipelines.heliolinc.inject_trails import add_trails
    from ADCNN.inference.catalog import panel_to_catalog_rows, InferenceConfig
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.cnn_postproc import load_cnn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = torch.jit.load("models/v2_D/segmentation_scripted.pt", map_location=dev).eval()
    cnn = load_cnn("models/v2_D/cnn_postproc.pt", device=("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = InferenceConfig(cnn_thr=0.5, gate_pmax=0.10)
    man = pd.read_csv(f"{RUN}/manifest.csv")
    dets = pd.read_csv(f"{RUN}/adcnn_dets_masked.csv",
                       usecols=["visit", "detector", "ra", "dec", "mjd"])
    vc = dets.groupby("visit").agg(ra=("ra", "median"), dec=("dec", "median"), mjd=("mjd", "median"))
    # N_EPOCHS=3 injects a mover into a COMMON-FOOTPRINT visit TRIPLE, which is what makes the
    # 3+visit tier truth-testable at all (see visit_groups: the boresight rule finds zero triples).
    N_EPOCHS = int(os.environ.get("INJ_EPOCHS", "2"))
    _pos = dets[["ra", "dec"]].sample(min(30000, len(dets)), random_state=1).to_numpy() \
        if N_EPOCHS >= 3 else None
    groups = visit_groups(vc, N_EPOCHS, positions=_pos, limit=n_pairs)
    print(f"[v2] {len(groups)} visit group(s), {N_EPOCHS} epochs each", flush=True)
    rng = np.random.default_rng(12345)
    truth, cats = [], []
    oid = 0
    for (_grp, _span) in groups:
        vA, followups = _grp[0], list(_grp[1:])
        # TAGS follow the existing truth-column convention: detB_* for the 2nd epoch, detC_* for the
        # 3rd. Epoch A is always the earliest of the group.
        _tags = ["B", "C", "D"][:len(followups)]
        _mjdA = float(vc.loc[vA].mjd)
        dA = dets[dets.visit == vA]
        for det in dA.detector.value_counts().head(n_dets).index.tolist():
            rowA = man[(man.visit == vA) & (man.detector == det)]
            if not len(rowA): continue
            try:
                with open_diffim(rowA.fits_path.iloc[0], memmap=False) as h:
                    imgA = np.nan_to_num(h[1].data.astype(np.float32)); wA = WCS(h[1].header)
            except Exception: continue
            H, W = imgA.shape
            injA, plan = [], []
            for k in range(per_panel):
                L_target = TRAIL_PX[k % len(TRAIL_PX)]
                rate = L_target * PIX / 3600.0 * (SOLARDAY / EXPTIME)   # deg/day implied by the trail
                snr_t = float(rng.uniform(SNR_LO, SNR_HI))
                pa = float(rng.uniform(0, 360))
                # *** RANDOM position in the panel (the v1 bias fix) ***
                xr, yr = float(rng.uniform(200, W-200)), float(rng.uniform(200, H-200))
                ra0, dec0 = [float(v) for v in wA.all_pix2world([[xr, yr]], 0)[0]]
                L_deg = rate*(EXPTIME/SOLARDAY)
                x, y, Lpx, beta = sky_trail_to_pixel(wA, ra0, dec0, L_deg, pa)
                mag = float(np.clip(mag_for_snr(snr_t, M5, Lpx), 16.0, 28.0))
                cd = np.cos(np.radians(dec0))
                injA.append(dict(x=x, y=y, trail_length=Lpx, beta=beta, mag=mag))
                _p = dict(oid=oid, rate=rate, L_target=L_target, mag=mag, snr_t=snr_t, pa=pa,
                          raA=ra0, decA=dec0, L_px=Lpx, visitA=int(vA), detA=int(det))
                for _tg, _vf in zip(_tags, followups):
                    _dtd = float(vc.loc[_vf].mjd) - _mjdA
                    _p[f"ra{_tg}"] = ra0 + rate*_dtd*np.cos(np.radians(pa))/cd
                    _p[f"dec{_tg}"] = dec0 + rate*_dtd*np.sin(np.radians(pa))
                    _p[f"visit{_tg}"] = int(_vf)
                plan.append(_p)
                oid += 1
            imgA2 = add_trails(np.array(imgA, copy=True), injA)
            prob, _, _, agg = predict_panel_overlap_3ch_full(seg, imgA2, np.zeros(imgA.shape, np.uint16), device=dev)
            cA = panel_to_catalog_rows(0, prob, imgA2, agg, np.zeros(imgA.shape, np.uint16), cnn, cfg)
            if cA is not None and len(cA):
                sky = wA.all_pix2world(cA[["x", "y"]].to_numpy(), 0)
                cA["ra"], cA["dec"] = sky[:, 0], sky[:, 1]
                cA["visit"] = int(vA); cA["detector"] = int(det); cA["mjd"] = float(dA.mjd.median())
                cats.append(cA)
                # CASCADE: was each injection DETECTED in A (and did it pass the linker input cuts)?
                for p, ij in zip(plan, injA):
                    d2 = (cA["x"].to_numpy()-ij["x"])**2 + (cA["y"].to_numpy()-ij["y"])**2
                    kk = int(np.argmin(d2))
                    p["detA_ok"] = bool(d2[kk] <= 25.0)
                    p["detA_len"] = float(cA["length"].to_numpy()[kk]) if p["detA_ok"] else np.nan
                    p["detA_score"] = float(cA["score"].to_numpy()[kk]) if p["detA_ok"] else np.nan
                    p["detA_snr"] = float(cA["mf_snr"].to_numpy()[kk]) if p["detA_ok"] else np.nan
            else:
                for p in plan: p["detA_ok"] = False
            # FOLLOW-UP EPOCHS. Identical logic per epoch (propagate -> find the covering detector ->
            # inject -> detect -> record), so it runs once per tag instead of being duplicated. With
            # INJ_EPOCHS=2 this executes exactly the original single B pass.
            _seen_this_panel = set()
            for _tg, _vf in zip(_tags, followups):
                dF = dets[dets.visit == _vf]
                for detF in dF.detector.unique():
                    gF = dF[dF.detector == detF]
                    if len(gF) < 50: continue
                    inF = [p for p in plan
                           if gF.ra.min() < p[f"ra{_tg}"] < gF.ra.max()
                           and gF.dec.min() < p[f"dec{_tg}"] < gF.dec.max()]
                    if not inF: continue
                    rowF = man[(man.visit == _vf) & (man.detector == detF)]
                    if not len(rowF): continue
                    try:
                        with open_diffim(rowF.fits_path.iloc[0], memmap=False) as h:
                            imgF = np.nan_to_num(h[1].data.astype(np.float32)); wF = WCS(h[1].header)
                    except Exception: continue
                    injF, keptF = [], []
                    for p in inF:
                        x, y, Lpx, beta = sky_trail_to_pixel(wF, p[f"ra{_tg}"], p[f"dec{_tg}"],
                                                             p["rate"]*(EXPTIME/SOLARDAY), p["pa"])
                        if not (200 < x < imgF.shape[1]-200 and 200 < y < imgF.shape[0]-200): continue
                        injF.append(dict(x=x, y=y, trail_length=Lpx, beta=beta, mag=p["mag"]))
                        p[f"det{_tg}_x"], p[f"det{_tg}_y"] = x, y
                        p[f"det{_tg}"] = int(detF); keptF.append(p)
                    if not injF: continue
                    imgF2 = add_trails(np.array(imgF, copy=True), injF)
                    prob, _, _, agg = predict_panel_overlap_3ch_full(seg, imgF2, np.zeros(imgF.shape, np.uint16), device=dev)
                    cF = panel_to_catalog_rows(0, prob, imgF2, agg, np.zeros(imgF.shape, np.uint16), cnn, cfg)
                    if cF is not None and len(cF):
                        sky = wF.all_pix2world(cF[["x", "y"]].to_numpy(), 0)
                        cF["ra"], cF["dec"] = sky[:, 0], sky[:, 1]
                        cF["visit"] = int(_vf); cF["detector"] = int(detF); cF["mjd"] = float(gF.mjd.median())
                        cats.append(cF)
                        for p in keptF:
                            d2 = (cF["x"].to_numpy()-p[f"det{_tg}_x"])**2 + (cF["y"].to_numpy()-p[f"det{_tg}_y"])**2
                            kk = int(np.argmin(d2))
                            ok = bool(d2[kk] <= 25.0)
                            p[f"det{_tg}_ok"] = ok
                            p[f"det{_tg}_len"] = float(cF["length"].to_numpy()[kk]) if ok else np.nan
                            p[f"det{_tg}_score"] = float(cF["score"].to_numpy()[kk]) if ok else np.nan
                            p[f"det{_tg}_snr"] = float(cF["mf_snr"].to_numpy()[kk]) if ok else np.nan
                    else:
                        for p in keptF: p[f"det{_tg}_ok"] = False
                    # a mover is recorded ONCE per panel-group, not once per follow-up epoch --
                    # otherwise a 3-epoch injection would appear twice in truth and double-count.
                    for p in keptF:
                        if p["oid"] not in _seen_this_panel:
                            _seen_this_panel.add(p["oid"]); truth.append(p)
        print(f"[v2] group {_grp} (arc {_span:.1f} min): truth {len(truth)}", flush=True)
    T = pd.DataFrame(truth)
    # One column per epoch actually injected. The fillna was OUTSIDE this loop, so only the LAST
    # column was ever filled -- detA_ok kept its NaNs. Latent (both A branches assign it) but wrong,
    # and with a third epoch it would stop being latent.
    _okcols = ["detA_ok"] + [f"det{t}_ok" for t in ["B", "C", "D"][:max(0, N_EPOCHS - 1)]]
    for c in _okcols:
        if c not in T:
            T[c] = False
        T[c] = T[c].fillna(False).astype(bool)
    T.to_csv(f"{V}/truth_{TAG}.csv", index=False)
    C = pd.concat(cats, ignore_index=True)
    # TRAIL ENDPOINTS: `beta` is the IMAGE-frame angle, so the half-length offset must be applied in
    # PIXELS and pushed through that panel's WCS -- exactly as detect_night does. Treating beta as a
    # SKY position angle (the v2 bug) rotates every trail by its detector's orientation, which made
    # trail-vs-motion PA disagree by ~48 deg while trail-vs-trail agreed to ~2 deg, and the
    # dpa_tm>20 hard gate then discarded 81% of TRUE pairs -- a harness artefact, not a pipeline fault.
    C["ra0"] = np.nan; C["dec0"] = np.nan; C["ra1"] = np.nan; C["dec1"] = np.nan
    for (vv, dd_), gg in C.groupby(["visit", "detector"]):
        rr = man[(man.visit == vv) & (man.detector == dd_)]
        if not len(rr):
            continue
        try:
            with open_diffim(rr.fits_path.iloc[0], memmap=False) as h:
                ww = WCS(h[1].header)
        except Exception:
            continue
        br = np.radians(gg["beta"].to_numpy()); Lp = np.clip(gg["length"].to_numpy(), 0, None)
        hdx = 0.5*Lp*np.cos(br); hdy = 0.5*Lp*np.sin(br)
        xy = gg[["x", "y"]].to_numpy()
        s0 = ww.all_pix2world(np.stack([xy[:, 0]-hdx, xy[:, 1]-hdy], 1), 0)
        s1 = ww.all_pix2world(np.stack([xy[:, 0]+hdx, xy[:, 1]+hdy], 1), 0)
        C.loc[gg.index, "ra0"] = s0[:, 0]; C.loc[gg.index, "dec0"] = s0[:, 1]
        C.loc[gg.index, "ra1"] = s1[:, 0]; C.loc[gg.index, "dec1"] = s1[:, 1]
    C = C[np.isfinite(C.ra0)].reset_index(drop=True)
    C["len_db"] = C["length"]; C["mag"] = np.nan; C["band"] = "r"; C["obscode"] = "I11"; C["art_frac"] = 0.0
    C.to_csv(f"{V}/inj_dets_{TAG}.csv", index=False)
    print(f"[v2] TRUTH {len(T)} | DETS {len(C)}", flush=True)
    print("\nDETECTION cascade (fraction of injected movers):")
    # ALL-epoch recovery is the number that matters for the 3+visit tier: the tier is rare because
    # it needs the SAME faint object found in every epoch, so its yield goes as (per-epoch p)^N.
    hdr = "".join(f"{c[:-3]+'%':>8}" for c in _okcols)
    print(f"{'SNR bin':>10} {'rate':>5} {'n':>5}{hdr}{'ALL%':>8}")
    T["snr_bin"] = pd.cut(T.snr_t, [2,4,6,8,10], right=False)
    for mag in sorted(T.snr_bin.dropna().unique()):
        for rate in sorted(T.L_target.unique()):
            m = (T.snr_bin == mag) & (T.L_target == rate)
            if m.sum() < 5: continue
            cells = "".join(f"{100*T[c][m].mean():7.1f}%" for c in _okcols)
            alls = np.logical_and.reduce([T[c][m].to_numpy() for c in _okcols])
            print(f"{str(mag):>10} {rate:5.1f} {int(m.sum()):5d}{cells}{100*alls.mean():7.1f}%")


if __name__ == "__main__":
    main()
