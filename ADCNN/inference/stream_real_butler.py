"""Butler-side producer for the streaming test_real pipeline.

Runs in the LSST stack env (loadLSST + setup lsst_distrib). For each (visit, detector) in the
fast-mover seed catalog, FIRST tries to fetch the PRE-BUILT diffim ``difference_image`` from
the DP2 reprocessing (stored in stage4 collections, ~2 s/panel). If that's missing for a
panel, falls back to AlardLupton subtract from PVI+template (~113 s/panel). Either way, emits
one pickled message ``{visit, detector, image, t_butler_s, t_subtract_s}`` to stdout.

Pipe into ``stream_real_inference.py`` running in the asteroid_cnn env.
"""
from __future__ import annotations
import argparse
import logging
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

for _n in ("lsst", "lsst.ip.diffim", "lsst.detectAndMeasure",
           "lsst.meas.algorithms", "ip_diffim_DipoleFit"):
    logging.getLogger(_n).setLevel(logging.ERROR)
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEF_REPO = "dp2_prep"
DEF_DIFFIM_COLLECTION = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage4"
DEF_DIFFIM_TYPE = "difference_image"          # DP2 renamed the AlardLupton output to this.
DEF_STAGE2 = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
DEF_STAGE3 = "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3"
DEF_SKYMAP = "lsst_cells_v2"


def _emit(msg, stream):
    body = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(len(body).to_bytes(8, "little"))
    stream.write(body); stream.flush()


def _try_prebuilt(butler, dataId, dt_name):
    """Fetch the DRP-prebuilt diffim. Returns (image, t_s) or (None, nan) if missing/err."""
    t0 = time.perf_counter()
    try:
        exp = butler.get(dt_name, dataId=dataId)
        img = exp.image.array.astype(np.float32)
        return img, time.perf_counter() - t0
    except Exception:
        return None, float("nan")


def _fallback_alard(butler_stage2, dataId, skymap, stage3):
    """AlardLupton on the fly when prebuilt is missing. Slow path."""
    from ADCNN.data.dataset_creation.butler_tasks import fetch_diffim_inputs, run_subtract
    t0 = time.perf_counter()
    pvi, sources, template, pf, _ = fetch_diffim_inputs(butler_stage2, dataId, skymap, stage3_collection=stage3)
    t_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    sub = run_subtract(template, pvi, sources)
    t_s = time.perf_counter() - t0
    return sub.difference.image.array.astype(np.float32), t_b, t_s


# Per-worker Butler cache: one connection per (repo, collection) per worker process. Butler
# init costs ~10-30 s — prohibitive per-panel, negligible when reused.
_BUTLER_CACHE = {}


def _get_butler(repo, collection):
    from lsst.daf.butler import Butler
    key = (repo, collection)
    b = _BUTLER_CACHE.get(key)
    if b is None:
        b = Butler(repo, collections=[collection])
        _BUTLER_CACHE[key] = b
    return b


def _run_one(args):
    visit, detector, repo, diffim_coll, diffim_type, stage2, stage3, skymap, allow_fallback = args
    dataId = {"instrument": "LSSTCam", "visit": int(visit), "detector": int(detector)}
    # 1) prebuilt diffim fast path
    img, t_fetch = _try_prebuilt(_get_butler(repo, diffim_coll), dataId, diffim_type)
    if img is not None:
        return {"visit": int(visit), "detector": int(detector), "image": img,
                "t_butler_s": float(t_fetch), "t_subtract_s": 0.0,
                "source": "prebuilt", "err": None}
    if not allow_fallback:
        return {"visit": int(visit), "detector": int(detector), "image": None,
                "t_butler_s": float("nan"), "t_subtract_s": float("nan"),
                "source": "prebuilt-missing", "err": "no prebuilt difference_image"}
    # 2) slow AlardLupton fallback -- catch BaseException because LSST's NoWorkFound /
    # UnprocessableDataError are deliberately NOT subclasses of Exception (so quantum-graph
    # pruning can tell them apart from regular bugs). We treat them as panel-level skips.
    try:
        img, t_b, t_s = _fallback_alard(_get_butler(repo, stage2), dataId, skymap, stage3)
        return {"visit": int(visit), "detector": int(detector), "image": img,
                "t_butler_s": float(t_b), "t_subtract_s": float(t_s),
                "source": "alard", "err": None}
    except BaseException as e:
        return {"visit": int(visit), "detector": int(detector), "image": None,
                "t_butler_s": float("nan"), "t_subtract_s": float("nan"),
                "source": "alard-fail", "err": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv", default=str(REPO / "DATA/sv_fast_movers_for_karlo_fast_with_pixels_rerun.csv"))
    ap.add_argument("--repo", default=DEF_REPO)
    ap.add_argument("--diffim-collection", default=DEF_DIFFIM_COLLECTION)
    ap.add_argument("--diffim-type", default=DEF_DIFFIM_TYPE)
    ap.add_argument("--stage2", default=DEF_STAGE2)
    ap.add_argument("--stage3", default=DEF_STAGE3)
    ap.add_argument("--skymap", default=DEF_SKYMAP)
    ap.add_argument("--no-fallback", action="store_true",
                    help="skip AlardLupton fallback for panels missing the prebuilt diffim")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", default=None,
                    help="X/N — this process handles the panels with index % N == X (multi-process sharding)")
    a = ap.parse_args()

    df = pd.read_csv(a.real_csv).drop_duplicates(["FieldID", "detector"])
    panels = list(zip(df["FieldID"].astype(int), df["detector"].astype(int)))
    if a.shard:
        x_str, n_str = a.shard.split("/")
        x = int(x_str); n = int(n_str)
        panels = panels[x::n]    # strided slice -> disjoint shards, even coverage
        shard_tag = f" shard={x}/{n}"
    else:
        shard_tag = ""
    if a.limit:
        panels = panels[: a.limit]
    print(f"[butler] {len(panels)} panels, workers={a.workers}, fallback={not a.no_fallback}{shard_tag}", file=sys.stderr, flush=True)

    out = sys.stdout.buffer
    args = [(v, d, a.repo, a.diffim_collection, a.diffim_type, a.stage2, a.stage3, a.skymap, not a.no_fallback)
            for v, d in panels]
    if a.workers <= 1:
        for x in args:
            _emit(_run_one(x), out)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_run_one, x) for x in args]
            for fut in as_completed(futs):
                try:
                    msg = fut.result()
                except BaseException as e:
                    # last-ditch safety net: even if a worker escaped its own try/except, don't
                    # take the whole producer down -- emit an error placeholder and keep going.
                    msg = {"visit": -1, "detector": -1, "image": None,
                           "t_butler_s": float("nan"), "t_subtract_s": float("nan"),
                           "source": "worker-crash", "err": f"{type(e).__name__}: {str(e)[:200]}"}
                _emit(msg, out)
    _emit({"_eof": True}, out)
    print("[butler] done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
