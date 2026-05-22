"""Storage-efficient data generation: drive the SAME inject->subtract pipeline as
the full-panel generator, but save only TILES (cutouts around each injected trail +
sampled background/residual negatives) instead of the 4004x4096 panel. Training only
ever uses ~128px tiles, so this is ~10x smaller per panel -> ~10x more panels for the
same storage. Realistic renderer + test-pair exclusion supported.

Tile store: <out>/tiles.h5 with datasets img/mask/real_label of shape (N, T, T) and
<out>/tiles.csv with per-tile metadata (panel_key, is_pos, beta, trail_length, snr).
Reconstruct orientation (sin2b/cos2b) maps in the dataset from mask + beta.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
import lsst.geom as geom

REPO = Path("/sdf/data/rubin/user/mrakovci/Projects/Asteroid_detection_CNN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "ADCNN/data/dataset_creation"))
import simulate_inject_diffim as sid
from simulate_inject_diffim import one_detector_injection, catalog_to_pandas
from ADCNN.data.dataset_creation import realistic_trail

STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
COLL = [STAGE3, STAGE2]
NX, NY = 4096, 4004
ARGS = dict(n_inject=20, trail_length=(6, 60), mag=(2, 8), beta=(0, 180),
            mag_mode="snr", psf_template="image", detection_threshold=5.0)


def extract_tiles(img, mask, real_labels, cat_df, T, n_neg, rng):
    """Return lists of (img,mask,real_label) tiles + metadata rows. Positives:
    one tile centered on each injection (x,y). Negatives: n_neg random background
    tiles not overlapping any injection footprint."""
    H, W = img.shape; half = T // 2
    pos_xy = cat_df[["x", "y"]].to_numpy()
    imgs, masks, rls, meta = [], [], [], []

    def crop(cy, cx):
        y0 = int(np.clip(cy - half, 0, H - T)); x0 = int(np.clip(cx - half, 0, W - T))
        sl = (slice(y0, y0 + T), slice(x0, x0 + T))
        return img[sl], mask[sl], real_labels[sl]

    for _, r in cat_df.iterrows():
        ti, tm, tr = crop(r.y, r.x)
        imgs.append(ti.astype(np.float32)); masks.append(tm.astype(bool)); rls.append(tr.astype(np.uint16))
        meta.append(dict(is_pos=1, beta=float(r.beta), trail_length=float(r.trail_length),
                         snr=float(r.get("SNR_estimation", np.nan))))
    # negatives: random centers >= T away from any injection
    tries = 0
    while sum(m["is_pos"] == 0 for m in meta) < n_neg and tries < n_neg * 20:
        tries += 1
        cy = rng.integers(half, H - half); cx = rng.integers(half, W - half)
        if len(pos_xy) and np.any(np.hypot(pos_xy[:, 0] - cx, pos_xy[:, 1] - cy) < T):
            continue
        ti, tm, tr = crop(cy, cx)
        imgs.append(ti.astype(np.float32)); masks.append(tm.astype(bool)); rls.append(tr.astype(np.uint16))
        meta.append(dict(is_pos=0, beta=np.nan, trail_length=np.nan, snr=np.nan))
    return imgs, masks, rls, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True, help="csv with visit,detector to inject")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=176)
    ap.add_argument("--n-neg", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--realistic", action="store_true")
    args = ap.parse_args()
    if args.realistic:
        realistic_trail.install()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    T = args.tile
    pairs = pd.read_csv(args.pairs_csv)[["visit", "detector"]].drop_duplicates()
    if args.limit:
        pairs = pairs.head(args.limit)
    dims = geom.Extent2I(NX, NY)

    h5p = out / "tiles.h5"; csvp = out / "tiles.csv"
    with h5py.File(h5p, "w") as f:
        f.create_dataset("img", shape=(0, T, T), maxshape=(None, T, T), dtype="f4",
                         chunks=(32, T, T), compression="gzip", compression_opts=4)
        f.create_dataset("mask", shape=(0, T, T), maxshape=(None, T, T), dtype="bool",
                         chunks=(32, T, T), compression="gzip", compression_opts=4)
        f.create_dataset("real_label", shape=(0, T, T), maxshape=(None, T, T), dtype="u2",
                         chunks=(32, T, T), compression="gzip", compression_opts=4)
    meta_rows = []; ntiles = 0; ok = 0; t0 = time.time()
    for i, (visit, detector) in enumerate(pairs.itertuples(index=False, name=None)):
        seed = (args.seed * 1_000_003 + int(visit) * 1_003 + int(detector)) & 0xFFFFFFFF
        try:
            res = one_detector_injection(
                ARGS["n_inject"], ARGS["trail_length"], ARGS["mag"], ARGS["beta"],
                "dp2_prep", COLL, dims, "preliminary_visit_image",
                {"instrument": "LSSTCam", "visit": int(visit), "detector": int(detector)},
                skymap="lsst_cells_v2", stage3_collection=STAGE3, seed=seed,
                mag_mode=ARGS["mag_mode"], psf_template=ARGS["psf_template"],
                detection_threshold=ARGS["detection_threshold"])
            if res[0] is False:
                print(f"[{i+1}/{len(pairs)}] SKIP v={visit} d={detector}: {res[1]}", flush=True); continue
            _, img, mask, real_labels, catalog = res
            cat_df = catalog_to_pandas(catalog)
            imgs, masks, rls, meta = extract_tiles(img, mask, real_labels, cat_df, T, args.n_neg,
                                                   np.random.default_rng(seed))
            n = len(imgs)
            with h5py.File(h5p, "a") as f:
                for ds, arr in [("img", imgs), ("mask", masks), ("real_label", rls)]:
                    d = f[ds]; d.resize(ntiles + n, axis=0); d[ntiles:ntiles + n] = np.stack(arr)
            for m in meta:
                m.update(panel_key=f"{visit}_{detector}", tile_idx=ntiles); meta_rows.append(m); ntiles += 1
            ok += 1
            el = time.time() - t0
            print(f"[{i+1}/{len(pairs)}] OK v={visit} d={detector} +{n} tiles "
                  f"(pos={sum(x['is_pos'] for x in meta)}) total={ntiles} ({el:.0f}s)", flush=True)
        except Exception as e:
            import traceback; print(f"[{i+1}] FAIL: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
    pd.DataFrame(meta_rows).to_csv(csvp, index=False)
    sz = h5p.stat().st_size / 1e6
    print(f"[done] {ok} panels -> {ntiles} tiles, {sz:.0f} MB ({sz/max(ok,1):.2f} MB/panel "
          f"vs ~117 MB/panel full) -> {117/max(sz/max(ok,1),1e-6):.0f}x smaller", flush=True)
    print("TILE-GEN DONE", flush=True)


if __name__ == "__main__":
    main()
