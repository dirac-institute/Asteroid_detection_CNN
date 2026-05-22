"""Streaming data producer: continuously generate FULL 4004x4096 injected diffim
panels (the exact validated one_detector_injection pipeline -- lossless, same
statistics) into a ROLLING BUFFER on shared scratch, so training can stream from an
unbounded supply of fresh panels without ever storing the whole dataset.

Each panel -> <buffer>/panel_<seq>.h5 (images/masks/real_labels, one panel) + an
atomically-updated manifest.json listing READY panels. Keeps the newest --buffer
panels, deletes older ones. Run alongside the trainer (DiffimStreamDataset consumes
the manifest). Parallel workers via --parallel.
"""
from __future__ import annotations
import argparse, json, os, sys, time, tempfile
from pathlib import Path
import concurrent.futures as cf
import h5py, numpy as np, pandas as pd
import lsst.geom as geom

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ADCNN/data/dataset_creation"))
import simulate_inject_diffim as sid
from simulate_inject_diffim import one_detector_injection, catalog_to_pandas
import realistic_trail

STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
COLL = [STAGE3, STAGE2]
NX, NY = 4096, 4004
ARGS = dict(n_inject=20, trail_length=(6, 60), mag=(2, 8), beta=(0, 180),
            mag_mode="snr", psf_template="image", detection_threshold=5.0)


def gen_one(visit, detector, seq, buf, realistic):
    if realistic:
        realistic_trail.install(verbose=False)
    seed = (456 * 1_000_003 + int(visit) * 1_003 + int(detector) + seq * 7919) & 0xFFFFFFFF
    res = one_detector_injection(
        ARGS["n_inject"], ARGS["trail_length"], ARGS["mag"], ARGS["beta"],
        "dp2_prep", COLL, geom.Extent2I(NX, NY), "preliminary_visit_image",
        {"instrument": "LSSTCam", "visit": int(visit), "detector": int(detector)},
        skymap="lsst_cells_v2", stage3_collection=STAGE3, seed=seed,
        mag_mode=ARGS["mag_mode"], psf_template=ARGS["psf_template"],
        detection_threshold=ARGS["detection_threshold"])
    if res[0] is False:
        return None
    _, img, mask, real_labels, catalog = res
    cat = catalog_to_pandas(catalog)
    tmp = Path(buf) / f".tmp_{seq}.h5"
    final = Path(buf) / f"panel_{seq:07d}.h5"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("image", data=img.astype(np.float32))
        f.create_dataset("mask", data=mask.astype(bool))
        f.create_dataset("real_labels", data=real_labels.astype(np.uint16))
        f.attrs["visit"] = int(visit); f.attrs["detector"] = int(detector)
        # per-injection geometry for the orientation maps (x,y,beta,trail_length)
        for c in ["x", "y", "beta", "trail_length"]:
            f.create_dataset(f"inj_{c}", data=cat[c].to_numpy(np.float32))
    os.replace(tmp, final)   # atomic publish
    return final.name


def write_manifest(buf):
    files = sorted(p.name for p in Path(buf).glob("panel_*.h5"))
    tmp = Path(buf) / ".manifest.tmp"
    tmp.write_text(json.dumps({"panels": files, "t": time.time()}))
    os.replace(tmp, Path(buf) / "manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-dir", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--buffer", type=int, default=200, help="max panels kept on disk")
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--max-panels", type=int, default=0, help="0 = run until stopfile")
    ap.add_argument("--realistic", action="store_true")
    args = ap.parse_args()
    buf = Path(args.buffer_dir); buf.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs_csv)[["visit", "detector"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(0)
    stop = buf / "STOP"
    print(f"[producer] buffer={args.buffer} parallel={args.parallel} pairs={len(pairs)} -> {buf}", flush=True)

    seq = 0; done = 0; t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.parallel) as ex:
        inflight = {}
        def submit():
            nonlocal seq
            v, d = pairs[rng.integers(len(pairs))]
            fut = ex.submit(gen_one, int(v), int(d), seq, str(buf), args.realistic); seq += 1
            inflight[fut] = True
        for _ in range(args.parallel):
            submit()
        while True:
            if stop.exists() or (args.max_panels and done >= args.max_panels):
                break
            fdone = next(cf.as_completed(inflight))
            inflight.pop(fdone)
            try:
                name = fdone.result()
            except Exception as e:
                name = None; print(f"[producer] gen fail: {type(e).__name__}: {e}", flush=True)
            if name:
                done += 1
                # rolling delete: keep newest --buffer panels
                files = sorted(Path(buf).glob("panel_*.h5"))
                for old in files[:-args.buffer]:
                    try: old.unlink()
                    except OSError: pass
                write_manifest(buf)
                if done % 10 == 0:
                    el = time.time() - t0
                    print(f"[producer] {done} panels ({el/done:.0f}s/panel, "
                          f"{done/el*60:.1f}/min, on-disk={len(files[-args.buffer:])})", flush=True)
            submit()
    print(f"[producer] stopped after {done} panels", flush=True)


if __name__ == "__main__":
    main()
