#!/usr/bin/env python3
"""Per-stage timing of the REAL detection path on a GPU, to locate the 10.6 GPU-s/panel.

sn_detect.slurm's own header says a full-cadence night (~15k panels) is ~11 h on 4 GPUs = ~10.6
GPU-seconds/panel, while stage-1 inference was optimised to ~1.4 s/panel in earlier work. That gap is
what this measures. CPU-only profiling cannot answer it -- seg inference alone takes 472 s/panel
without a GPU, which swamps everything.

torch.cuda.synchronize() around each GPU stage, or the async launches make inference look free and
the next CPU stage absorb its cost.
"""
import argparse, time
import numpy as np, pandas as pd, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", default="20260706")
    ap.add_argument("--panels", type=int, default=40)
    a = ap.parse_args()
    run = f"outputs/runs/10k_cadence/run_night_{a.night}"
    man = pd.read_csv(f"{run}/manifest.csv").sample(a.panels, random_state=0)
    from ADCNN.inference.diffim_io import open_diffim
    from ADCNN.inference.predict import predict_panel_overlap_3ch_full
    from ADCNN.inference.cnn_postproc import load_cnn, apply_cnn
    from ADCNN.inference.features import extract_panel_candidates
    import ADCNN.inference.catalog as C
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prof] device={dev}  panels={a.panels}", flush=True)
    seg = torch.jit.load("models/v2_D/segmentation_scripted.pt", map_location=dev).eval()
    cnn = load_cnn("models/v2_D/cnn_postproc.pt", device=str(dev))

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    acc, ndet = {}, []
    for i, row in enumerate(man.itertuples()):
        t0 = time.perf_counter()
        try:
            with open_diffim(row.fits_path, memmap=False) as h:
                img = np.nan_to_num(h[1].data.astype(np.float32))
        except Exception:
            continue
        acc["1 pixel read + nan_to_num"] = acc.get("1 pixel read + nan_to_num", 0) + time.perf_counter() - t0
        rl = np.zeros(img.shape, np.uint16)
        sync(); t0 = time.perf_counter()
        prob, _, _, agg = predict_panel_overlap_3ch_full(seg, img, np.zeros(img.shape, np.uint16), device=dev)
        sync(); acc["2 seg inference (GPU)"] = acc.get("2 seg inference (GPU)", 0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        cand, _ = extract_panel_candidates(prob[None], img[None], real_labels=rl[None], gate_pmax=0.10)
        acc["3 candidates + matched filter"] = acc.get("3 candidates + matched filter", 0) + time.perf_counter() - t0
        if not len(cand):
            continue
        sync(); t0 = time.perf_counter()
        cand = apply_cnn(cand, cnn, img, prob, agg, device=dev)
        sync(); acc["4 cutout CNN scoring (GPU)"] = acc.get("4 cutout CNN scoring (GPU)", 0) + time.perf_counter() - t0
        cand = cand[cand["score"] >= 0.5].copy()
        ndet.append(len(cand))
        if not len(cand):
            continue
        t0 = time.perf_counter()
        from ADCNN.inference.mf_trail_length import refine_trail_length
        sig = float(cand["panel_sigma"].iloc[0]) if "panel_sigma" in cand.columns else None
        refine_trail_length(cand["x_centroid"].to_numpy(), cand["y_centroid"].to_numpy(), img,
                            cand["mf_length"].to_numpy(), cand["mf_beta"].to_numpy(), sigma=sig)
        acc["5 template trail length"] = acc.get("5 template trail length", 0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        C._attach_dipole_morphology(cand, img)
        acc["6 dipole morphology"] = acc.get("6 dipole morphology", 0) + time.perf_counter() - t0
        if (i + 1) % 10 == 0:
            print(f"[prof] {i+1}/{a.panels}", flush=True)
    n = max(len(ndet), 1)
    tot = sum(acc.values())
    print(f"\nPER-PANEL MEAN over {n} panels ({np.mean(ndet):.0f} scored detections/panel)\n")
    for k in sorted(acc):
        print(f"  {k:<34}{1000*acc[k]/n:>9.1f} ms  ({100*acc[k]/tot:>5.1f}%)")
    print(f"  {'TOTAL':<34}{1000*tot/n:>9.1f} ms")
    print(f"\n  sn_detect.slurm implies ~10,600 ms/panel of GPU time (11 h, 15k panels, 4 GPUs).")
    print(f"  A full-cadence night at this rate: {16937*tot/n/3600:.1f} GPU-hours "
          f"=> {16937*tot/n/3600/4:.1f} h on 4 GPUs, {16937*tot/n/3600/10:.1f} h on 10.")


if __name__ == "__main__":
    main()
