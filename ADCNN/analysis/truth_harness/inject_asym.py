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
    # ASYMMETRIC THRESHOLDS. Every co-pointed revisit in this survey is CROSS-BAND, and the shallow
    # epoch discards 84-87% of the faint-fast movers the deep epoch found (m5 gap 0.45-0.62 mag).
    # Requiring an independent blind 5-sigma detection in the shallow epoch is the wrong statistical
    # test: epoch A already tells us where, when, and along what PA the object must appear, so the
    # trials factor collapses. Rather than hand-build a targeted search, go DEEPER on the shallow
    # epoch and let the linker's existing chi2 (trail PA + length vs chord) reject the extra FPs.
    # Chance links then grow LINEARLY in n2, not quadratically -- only one side is loosened.
    THR_B = float(os.environ.get("INJ_THR_B", "0.15"))
    cfgB = InferenceConfig(cnn_thr=THR_B, gate_pmax=float(os.environ.get("INJ_GATE_B", "0.02")))
    print(f"[inj] epoch A cnn_thr=0.50 (deep band) | epoch B cnn_thr={THR_B} (shallow band)", flush=True)
    man = pd.read_csv(f"{RUN}/manifest.csv")
    dets = pd.read_csv(f"{RUN}/adcnn_dets_masked.csv",
                       usecols=["visit", "detector", "ra", "dec", "mjd"])
    vc = dets.groupby("visit").agg(ra=("ra", "median"), dec=("dec", "median"), mjd=("mjd", "median"))
    vs = vc.index.to_numpy(); pairs = []
    for i in range(len(vs)):
        for j in range(i+1, len(vs)):
            a, b = vc.loc[vs[i]], vc.loc[vs[j]]
            dt = abs(b.mjd - a.mjd)*1440.0
            if not (10 < dt <= 52): continue
            if np.hypot((a.ra-b.ra)*np.cos(np.radians(a.dec)), a.dec-b.dec) < 0.3:
                pairs.append((vs[i], vs[j], dt))
    pairs.sort(key=lambda t: t[2]); pairs = pairs[:n_pairs]
    print(f"[v2] {len(pairs)} visit pairs", flush=True)
    rng = np.random.default_rng(12345)
    truth, cats = [], []
    oid = 0
    for (vA, vB, dtmin) in pairs:
        dt_day = dtmin/1440.0
        dA = dets[dets.visit == vA]; dB = dets[dets.visit == vB]
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
                raB = ra0 + rate*dt_day*np.cos(np.radians(pa))/cd
                decB = dec0 + rate*dt_day*np.sin(np.radians(pa))
                injA.append(dict(x=x, y=y, trail_length=Lpx, beta=beta, mag=mag))
                plan.append(dict(oid=oid, rate=rate, L_target=L_target, mag=mag, snr_t=snr_t, pa=pa, raA=ra0, decA=dec0, raB=raB,
                                 decB=decB, L_px=Lpx, visitA=int(vA), visitB=int(vB), detA=int(det)))
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
            for detB in dB.detector.unique():
                gB = dB[dB.detector == detB]
                if len(gB) < 50: continue
                inB = [p for p in plan if gB.ra.min() < p["raB"] < gB.ra.max() and gB.dec.min() < p["decB"] < gB.dec.max()]
                if not inB: continue
                rowB = man[(man.visit == vB) & (man.detector == detB)]
                if not len(rowB): continue
                try:
                    with open_diffim(rowB.fits_path.iloc[0], memmap=False) as h:
                        imgB = np.nan_to_num(h[1].data.astype(np.float32)); wB = WCS(h[1].header)
                except Exception: continue
                injB, keptB = [], []
                for p in inB:
                    x, y, Lpx, beta = sky_trail_to_pixel(wB, p["raB"], p["decB"], p["rate"]*(EXPTIME/SOLARDAY), p["pa"])
                    if not (200 < x < imgB.shape[1]-200 and 200 < y < imgB.shape[0]-200): continue
                    injB.append(dict(x=x, y=y, trail_length=Lpx, beta=beta, mag=p["mag"]))
                    p["detB_x"], p["detB_y"] = x, y; p["detB"] = int(detB); keptB.append(p)
                if not injB: continue
                imgB2 = add_trails(np.array(imgB, copy=True), injB)
                prob, _, _, agg = predict_panel_overlap_3ch_full(seg, imgB2, np.zeros(imgB.shape, np.uint16), device=dev)
                cB = panel_to_catalog_rows(0, prob, imgB2, agg, np.zeros(imgB.shape, np.uint16), cnn, cfgB)
                if cB is not None and len(cB):
                    sky = wB.all_pix2world(cB[["x", "y"]].to_numpy(), 0)
                    cB["ra"], cB["dec"] = sky[:, 0], sky[:, 1]
                    cB["visit"] = int(vB); cB["detector"] = int(detB); cB["mjd"] = float(gB.mjd.median())
                    cats.append(cB)
                    for p in keptB:
                        d2 = (cB["x"].to_numpy()-p["detB_x"])**2 + (cB["y"].to_numpy()-p["detB_y"])**2
                        kk = int(np.argmin(d2))
                        p["detB_ok"] = bool(d2[kk] <= 25.0)
                        p["detB_len"] = float(cB["length"].to_numpy()[kk]) if p["detB_ok"] else np.nan
                        p["detB_score"] = float(cB["score"].to_numpy()[kk]) if p["detB_ok"] else np.nan
                        p["detB_snr"] = float(cB["mf_snr"].to_numpy()[kk]) if p["detB_ok"] else np.nan
                else:
                    for p in keptB: p["detB_ok"] = False
                truth.extend(keptB)
        print(f"[v2] pair {vA}/{vB}: truth {len(truth)}", flush=True)
    T = pd.DataFrame(truth)
    for c in ("detA_ok", "detB_ok"):
        if c not in T: T[c] = False
    T[c] = T[c].fillna(False)
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
    print(f"{'SNR bin':>10} {'rate':>5} {'n':>5} {'detA%':>7} {'detB%':>7} {'both%':>7}")
    T["snr_bin"] = pd.cut(T.snr_t, [2,4,6,8,10], right=False)
    for mag in sorted(T.snr_bin.dropna().unique()):
        for rate in sorted(T.L_target.unique()):
            m = (T.snr_bin == mag) & (T.L_target == rate)
            if m.sum() < 5: continue
            a = 100*T.detA_ok[m].mean(); bb = 100*T.detB_ok[m].fillna(False).mean()
            both = 100*(T.detA_ok[m] & T.detB_ok[m].fillna(False)).mean()
            print(f"{str(mag):>10} {rate:5.1f} {int(m.sum()):5d} {a:6.1f}% {bb:6.1f}% {both:6.1f}%")


if __name__ == "__main__":
    main()
