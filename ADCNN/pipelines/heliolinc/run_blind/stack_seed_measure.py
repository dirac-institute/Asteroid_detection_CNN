#!/usr/bin/env python3
"""#250 experiment A: seeded matched-filter trail state for STACK-ONLY peaks (STACK_SEEDED_DESIGN.md).

For each stack-5sigma peak farther than TOL px from every ADCNN detection (the stack-only additions),
re-create the injected panel deterministically, cut a KxK stamp, and scan a trailed-PSF template bank
(PA x L) -> mf_snr, best PA, best L. Label tp/fp against injected sightings; tp rows carry truth
(beta, trail_length, snr_target) so the PA/length errors are measurable. Segmentation-independent:
answers whether stack-only TP additions carry usable 2v trail geometry at all (design step A).
Measurement only -- nothing is gated or tuned here.

Run: python run_blind/stack_seed_measure.py --fields 2 3   (repo root on PYTHONPATH)
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, REPO)

K = 64                 # stamp side (px)
TOL = 10.0             # stack<->ADCNN / truth match radius (px)
PSF_SIG = 3.77 / 2.355  # px
ANGLES = np.arange(0.0, 180.0, 7.5)          # deg
LENGTHS = np.arange(6.0, 41.0, 2.0)          # px


def template_bank():
    """(N_t, K*K) normalized trailed-PSF templates centred in the stamp."""
    yy, xx = np.mgrid[0:K, 0:K].astype(np.float64)
    cx = cy = (K - 1) / 2.0
    bank, meta = [], []
    for L in LENGTHS:
        for a in ANGLES:
            th = np.radians(a)
            n = max(int(L * 2), 2)
            ts = np.linspace(-L / 2, L / 2, n)
            T = np.zeros((K, K))
            for t in ts:
                px, py = cx + t * np.cos(th), cy + t * np.sin(th)
                T += np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * PSF_SIG ** 2)))
            T /= np.linalg.norm(T)
            bank.append(T.ravel()); meta.append((a, L))
    return np.array(bank), meta


def stamps_at(img, xs, ys):
    """(N, K*K) stamps (zero-padded at edges)."""
    H, W = img.shape
    out = np.zeros((len(xs), K * K), np.float64)
    h = K // 2
    for i, (x, y) in enumerate(zip(xs, ys)):
        x0, y0 = int(round(x)) - h, int(round(y)) - h
        sx0, sy0 = max(0, -x0), max(0, -y0)
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(W, x0 + K), min(H, y0 + K)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        st = np.zeros((K, K))
        st[sy0:sy0 + (iy1 - iy0), sx0:sx0 + (ix1 - ix0)] = img[iy0:iy1, ix0:ix1]
        out[i] = st.ravel()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, nargs="+", required=True)
    a = ap.parse_args()
    from ADCNN.pipelines.heliolinc.inject_trails import load_inject_map, add_trails
    B, meta = template_bank()
    rows = []
    for k in a.fields:
        man = pd.read_csv(f"{HERE}/manifest_{k}.csv")
        peaks = pd.read_csv(f"{HERE}/stack_full_s5_{k}_peaks.csv")
        ad = pd.read_csv(f"{HERE}/adcnn_dets_masked_{k}.csv", usecols=["visit", "detector", "x", "y"])
        inj = pd.read_csv(f"{HERE}/inject_{k}.csv")
        t = pd.read_csv(f"{HERE}/truth_{k}.csv").set_index("objID")
        imap = load_inject_map(f"{HERE}/inject_{k}.csv")
        gp = dict(tuple(peaks.groupby(["visit", "detector"])))
        ga = dict(tuple(ad.groupby(["visit", "detector"])))
        gi = dict(tuple(inj.groupby(["visit", "detector"])))
        n_pan = 0
        for r in man.itertuples():
            key = (int(r.visit), int(r.detector))
            pk = gp.get(key)
            if pk is None or key not in imap:
                continue   # peaks exist only on inject panels (inject-panels-only run)
            # stack-only = peaks >TOL from every ADCNN det on this panel
            adp = ga.get(key)
            pxy = pk[["x", "y"]].to_numpy()
            if adp is not None and len(adp):
                d, _ = cKDTree(adp[["x", "y"]].to_numpy()).query(pxy, distance_upper_bound=TOL)
                only = ~np.isfinite(d)
            else:
                only = np.ones(len(pk), bool)
            if not only.any():
                continue
            pko = pk[only]
            with fits.open(r.fits_path, memmap=False) as h:
                img = np.nan_to_num(h[1].data.astype(np.float32))
            img = add_trails(img, imap[key])
            S = stamps_at(img, pko.x.to_numpy(), pko.y.to_numpy())
            sig = 1.4826 * np.median(np.abs(S - np.median(S, axis=1, keepdims=True)), axis=1)
            sig[sig <= 0] = 1.0
            resp = (S @ B.T) / sig[:, None]          # (N, N_templates) matched-filter S/N
            best = resp.argmax(axis=1)
            mf_snr = resp.max(axis=1)
            # truth labels
            inj_p = gi.get(key)
            if inj_p is not None and len(inj_p):
                dt, it = cKDTree(inj_p[["x", "y"]].to_numpy()).query(
                    pko[["x", "y"]].to_numpy(), distance_upper_bound=TOL)
                lab = np.isfinite(dt)
            else:
                lab = np.zeros(len(pko), bool); it = np.zeros(len(pko), int)
            for i, (pr, br) in enumerate(zip(pko.itertuples(), best)):
                pa, L = meta[br]
                row = dict(field=k, visit=key[0], detector=key[1], x=float(pr.x), y=float(pr.y),
                           stack_snr=float(pr.snr), mf_snr=float(mf_snr[i]),
                           mf_pa=float(pa), mf_len=float(L), label="fp", objID="",
                           true_beta=np.nan, true_len=np.nan, snr_target=np.nan)
                if lab[i]:
                    ir = inj_p.iloc[int(it[i])]
                    row.update(label="tp", objID=str(ir.objID),
                               true_beta=float(ir.beta) % 180.0,
                               true_len=float(ir.trail_length),
                               snr_target=float(t.loc[ir.objID].snr_target))
                rows.append(row)
            n_pan += 1
        print(f"[seedmf] field {k}: {n_pan} panels done, rows so far {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    out = f"{HERE}/stack_seed_mf_{'_'.join(map(str, a.fields))}.csv"
    df.to_csv(out, index=False)
    tp = df[df.label == "tp"]; fp = df[df.label == "fp"]
    print(f"[seedmf] TOTAL {len(df)} stack-only peaks: tp={len(tp)} fp={len(fp)}")
    if len(tp):
        dpa = np.abs(((tp.mf_pa - tp.true_beta) + 90) % 180 - 90)
        print(f"[seedmf] tp PA error: median {dpa.median():.1f} deg, <=15deg {100*(dpa<=15).mean():.0f}%")
        print(f"[seedmf] tp mf_snr median {tp.mf_snr.median():.1f} | fp mf_snr median {fp.mf_snr.median():.1f}")
        print(f"[seedmf] tp true_len median {tp.true_len.median():.1f}px | tp snr_target median {tp.snr_target.median():.1f}")
    print("SEEDMF_DONE")


if __name__ == "__main__":
    main()
