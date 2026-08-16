#!/usr/bin/env python3
"""Measure the TRAIL-PA measurement precision as a function of trail LENGTH, against injected truth.

Why this and not the alert population: the observed dpa spread in a linked stream is ~33 deg at every
trail length, which is just 1.4826*MAD of a UNIFORM [0,90] distribution -- i.e. it measures chance
links, not PA precision. Only injected trails with a KNOWN position angle can measure the real thing.

The 2-visit chi2 divides the trail-vs-motion PA disagreement by a FIXED sigma
(CHI2_SIG_2V['dpa_tm'] = 4.869 deg) at all trail lengths. If the true precision degrades as ~1/L,
short-trailed (slow / stack-contributed) movers are penalised for a measurement limitation rather than
for being bad links -- which would explain why only 3 of the 45 stack-contributed alerts survive
chi2<=9 despite sitting inside the rate band.

Output: sigma(beta_measured - beta_true) per injected-trail-length bin -> the empirical sigma_PA(L).
"""
import sys, numpy as np, pandas as pd, torch

def main():
    from ADCNN.inference.diffim_io import open_diffim
    from ADCNN.pipelines.heliolinc.inject_trails import add_trails
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.features import extract_panel_candidates
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = torch.jit.load("models/v2_D/segmentation_scripted.pt", map_location=dev).eval()
    man = pd.read_csv("outputs/runs/ringpipe_0706/manifest.csv")
    n_panels = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    rng = np.random.default_rng(0)
    LENS = [6, 8, 10, 13, 17, 22, 30]          # injected trail lengths (px)
    rows = []
    panels = man.sample(n=min(n_panels, len(man)), random_state=0)
    for pi, r in enumerate(panels.itertuples()):
        try:
            with open_diffim(r.fits_path, memmap=False) as h:
                img = np.nan_to_num(h[1].data.astype(np.float32))
        except Exception as e:
            print(f"  panel {pi} load failed: {type(e).__name__}", flush=True); continue
        H, W = img.shape
        inj = []
        for L in LENS:
            for _ in range(12):                 # 12 injections per length per panel
                inj.append(dict(x=float(rng.uniform(200, W-200)), y=float(rng.uniform(200, H-200)),
                                trail_length=float(L), beta=float(rng.uniform(0, 180)),
                                mag=float(rng.uniform(21.5, 23.5))))
        img2 = add_trails(np.array(img, copy=True), inj)
        prob, _, _, agg = predict_panel_overlap_3ch_full(seg, img2, np.zeros(img.shape, np.uint16), device=dev)
        cand, _ = extract_panel_candidates(prob[None], img2[None], real_labels=None, gate_pmax=0.10)
        if not len(cand):
            continue
        cx = cand["x_centroid"].to_numpy(); cy = cand["y_centroid"].to_numpy()
        cb = cand["mf_beta"].to_numpy(); cl = cand["mf_length"].to_numpy()
        for j in inj:                            # match each injection to its nearest detection
            d2 = (cx - j["x"])**2 + (cy - j["y"])**2
            k = int(np.argmin(d2))
            if d2[k] > 5.0**2:                   # unrecovered
                continue
            dpa = abs(((cb[k] % 180) - (j["beta"] % 180) + 90) % 180 - 90)
            rows.append((j["trail_length"], dpa, float(cl[k])))
        if (pi + 1) % 5 == 0:
            print(f"  {pi+1}/{len(panels)} panels, {len(rows)} recovered injections", flush=True)
    df = pd.DataFrame(rows, columns=["L_true", "dpa", "L_meas"])
    df.to_csv("outputs/runs/pa_precision/pa_vs_length.csv", index=False)
    print(f"\nrecovered {len(df)} injections")
    print(f"\n{'L_true':>7} {'n':>5} {'sigma_PA(deg)':>14} {'median|dpa|':>12} {'1/L model':>10}")
    ref = None
    for L in LENS:
        m = df.L_true == L
        if m.sum() < 20: continue
        d = df.dpa[m].to_numpy()
        s = 1.4826*np.median(np.abs(d - np.median(d)))
        med = np.median(d)
        if ref is None: ref = (L, s)
        print(f"{L:7.0f} {int(m.sum()):5d} {s:14.2f} {med:12.2f} {ref[1]*ref[0]/L:10.2f}")
    print(f"\ncurrent FIXED CHI2_SIG_2V['dpa_tm'] = 4.869 deg at ALL lengths")

if __name__ == "__main__":
    main()
