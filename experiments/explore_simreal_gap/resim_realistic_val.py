"""Re-simulate the 50 RF-training VAL panels of train.h5 with the REALISTIC trail
renderer (light-curve / tapered / curved), reproducing the SAME injection catalog
(same per-pair seed=123 formula) so only the trail MORPHOLOGY changes. Output is a
fresh leak-free synthetic set (val panels are disjoint from test_5sigma) used to
retrain the stage-2 RF. With --uniform it reproduces the stock renderer (control).

Reuses the production one_detector_injection so the inject->AlardLupton-subtract->
detect path is identical to the original build. Read-only on test_real (untouched).
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
import h5py, numpy as np, pandas as pd
import lsst.geom as geom

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ADCNN/data/dataset_creation"))
import simulate_inject_diffim as sid
from simulate_inject_diffim import one_detector_injection, catalog_to_pandas
from ADCNN.data.dataset_creation import realistic_trail

REPO_BUTLER = "dp2_prep"
STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
COLL = [STAGE3, STAGE2]
SKYMAP = "lsst_cells_v2"
GLOBAL_SEED = 123
NX, NY = 4096, 4004
ARGS = dict(n_inject=20, trail_length=(6, 60), mag=(2, 8), beta=(0, 180),
            mag_mode="snr", psf_template="image", detection_threshold=5.0)


def val_pairs():
    import json
    val = set(json.load(open(REPO / "experiments/diffim_runs/pilot_v7/split.json"))["val_panels"])
    tr = pd.read_csv(REPO / "DATA_DIFFIM/train.csv")
    vp = (tr[tr.image_id.isin(val)][["image_id", "visit", "detector"]]
          .drop_duplicates("image_id").sort_values("image_id"))
    return list(vp.itertuples(index=False, name=None))  # (image_id, visit, detector)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "DATA_DIFFIM/train_realistic_val"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uniform", action="store_true", help="stock renderer (control)")
    args = ap.parse_args()
    if not args.uniform:
        realistic_trail.install()
    else:
        print("[resim] UNIFORM control (stock renderer)", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pairs = val_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
    N = len(pairs)
    h5p = out / "train.h5"; csvp = out / "train.csv"
    if csvp.exists():
        csvp.unlink()
    with h5py.File(h5p, "w") as f:
        for name, dt in [("images", "f4"), ("masks", "bool"), ("real_labels", "u2")]:
            f.create_dataset(name, shape=(N, NY, NX), dtype=dt, chunks=(1, 128, NX))
    dims = geom.Extent2I(NX, NY)
    print(f"[resim] {N} val panels -> {out}  (realistic={not args.uniform})", flush=True)

    ok = 0; t0 = time.time()
    for i, (image_id, visit, detector) in enumerate(pairs):
        seed = (GLOBAL_SEED * 1_000_003 + int(visit) * 1_003 + int(detector)) & 0xFFFFFFFF
        ref_dataId = {"instrument": "LSSTCam", "visit": int(visit), "detector": int(detector)}
        try:
            res = one_detector_injection(
                ARGS["n_inject"], ARGS["trail_length"], ARGS["mag"], ARGS["beta"],
                REPO_BUTLER, COLL, dims, "preliminary_visit_image", ref_dataId,
                skymap=SKYMAP, stage3_collection=STAGE3, seed=seed,
                mag_mode=ARGS["mag_mode"], psf_template=ARGS["psf_template"],
                detection_threshold=ARGS["detection_threshold"])
            if res[0] is False:
                print(f"[{i+1}/{N}] SKIP id={image_id} v={visit} d={detector}: {res[1]}", flush=True)
                continue
            _, img, mask, real_labels, catalog = res
            with h5py.File(h5p, "a") as f:
                f["images"][i] = img; f["masks"][i] = mask; f["real_labels"][i] = real_labels
            df = catalog_to_pandas(catalog); df["image_id"] = i
            df["orig_image_id"] = image_id; df["visit"] = visit; df["detector"] = detector
            df.to_csv(csvp, mode=("a" if csvp.exists() else "w"),
                      header=(not csvp.exists()), index=False)
            ok += 1
            el = time.time() - t0
            print(f"[{i+1}/{N}] OK id={image_id} v={visit} d={detector} "
                  f"inj={len(df)} ({el:.0f}s, {el/(i+1):.0f}s/panel)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{N}] FAIL id={image_id}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"[resim] done: {ok}/{N} panels -> {h5p}", flush=True)
    print("RESIM DONE", flush=True)


if __name__ == "__main__":
    main()
