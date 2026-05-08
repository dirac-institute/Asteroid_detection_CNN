"""Stage-1 diffim generator: one (visit, detector) → one NPZ shard.

Generalization of experiments/diffim/proof/single_visit_proof.py. Key changes
vs. the proof:

- Takes a manifest + task index (or explicit --visit/--detector) so it slots
  into a SLURM array.
- Randomizes injection parameters per task (count, mag, length, angle) with
  a deterministic per-task seed so reruns are reproducible.
- Writes a metadata CSV row per injected object and a summary JSON per shard.
- Diffim input channels are built with the signed, zero-centered, variance-
  aware normalization from ADCNN/data/diffim_norm.py so the NN training-time
  view is already the right representation.
- Truth mask is built ONLY from the injected geometry projected into diffim
  space (empirical_inj_only > k·σ_diff). AP detections are logged as a
  baseline but are never used to define the label.

Usage:
    # Manifest-driven (SLURM array):
    python driver.py --manifest manifests/pilot5.json --task-index 0 --out-root /path/to/DATA/diffim

    # Ad-hoc single-pair:
    python driver.py --visit 2025042400200 --detector 1 --out-root /tmp/adhoc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_DEFAULT = "dp2_prep"
COLLECTIONS_DEFAULT = [
    "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3",
    "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
]
SKYMAP_DEFAULT = "lsst_cells_v2"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--collections", nargs="+", default=COLLECTIONS_DEFAULT)
    p.add_argument("--skymap", default=SKYMAP_DEFAULT)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", type=str, help="JSON manifest with 'pairs' list.")
    g.add_argument("--visit", type=int, help="Ad-hoc visit id (no manifest).")

    p.add_argument("--task-index", type=int, default=0,
                   help="Array task index into manifest['pairs'].")
    p.add_argument("--detector", type=int, default=None,
                   help="Required when --visit is used without a manifest.")
    p.add_argument("--out-root", required=True,
                   help="Output root directory. A subdir per (visit, detector)"
                   " is created beneath it.")
    p.add_argument("--n-trails-min", type=int, default=30)
    p.add_argument("--n-trails-max", type=int, default=100)
    p.add_argument("--mag-min", type=float, default=22.5)
    p.add_argument("--mag-max", type=float, default=25.0)
    p.add_argument("--trail-length-min", type=float, default=6.0)
    p.add_argument("--trail-length-max", type=float, default=60.0)
    p.add_argument("--truth-sigma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=None,
                   help="Master RNG seed; if None, derived from (visit, detector).")
    return p.parse_args()


def resolve_pair(args: argparse.Namespace) -> tuple[int, int, dict | None]:
    """Return (visit, detector, manifest_row_or_None)."""
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        pairs = manifest["pairs"]
        if not (0 <= args.task_index < len(pairs)):
            raise SystemExit(
                f"task-index {args.task_index} out of range [0, {len(pairs)-1}]"
            )
        row = pairs[args.task_index]
        return int(row["visit"]), int(row["detector"]), row
    if args.detector is None:
        raise SystemExit("--detector is required when --visit is used without --manifest.")
    return int(args.visit), int(args.detector), None


# ---------------------------------------------------------------------------
# Butler helpers (proof-identical)
# ---------------------------------------------------------------------------
def get_butler(repo: str, collections: list[str]):
    from lsst.daf.butler import Butler
    return Butler(repo, collections=collections)


def get_pvi_bundle(butler, visit: int, detector: int):
    did = dict(instrument="LSSTCam", visit=visit, detector=detector)
    pvi = butler.get("preliminary_visit_image", dataId=did)
    try:
        sources = butler.get("single_visit_star_footprints", dataId=did)
    except Exception as e:
        raise RuntimeError(f"single_visit_star_footprints missing for {did}: {e}") from e
    return did, pvi, sources


def get_template(butler, pvi, skymap: str, physical_filter: str):
    import lsst.geom
    from lsst.ip.diffim.getTemplate import GetTemplateTask, GetTemplateConfig
    from lsst.sphgeom import ConvexPolygon

    band = pvi.filter.bandLabel
    wcs = pvi.wcs
    bbox = pvi.getBBox()
    corners_pix = [lsst.geom.Point2D(c) for c in bbox.getCorners()]
    corners_sky = [wcs.pixelToSky(p) for p in corners_pix]
    region = ConvexPolygon([c.getVector() for c in corners_sky])

    all_refs = list(
        butler.registry.queryDatasets(
            "template_coadd",
            where="skymap = :skymap AND patch.region OVERLAPS :region",
            bind={"skymap": skymap, "region": region},
            findFirst=True,
        )
    )
    refs = [r for r in all_refs if r.dataId.get("band") == band]
    if not refs:
        bands_seen = sorted({r.dataId.get("band") for r in all_refs})
        raise RuntimeError(
            f"No {band}-band template_coadd overlapping visit={pvi.visitInfo.id} "
            f"(raw_overlap={len(all_refs)}, bands_seen={bands_seen})"
        )

    by_tract: dict[int, list[Any]] = {}
    dataIds: dict[int, list[Any]] = {}
    for ref in refs:
        tract = int(ref.dataId["tract"])
        by_tract.setdefault(tract, []).append(butler.getDeferred(ref))
        dataIds.setdefault(tract, []).append(ref.dataId)

    cfg = GetTemplateConfig()
    task = GetTemplateTask(config=cfg)
    return task.run(
        coaddExposureHandles=by_tract,
        bbox=bbox,
        wcs=wcs,
        dataIds=dataIds,
        physical_filter=physical_filter,
    ).template, len(refs)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def build_injection_catalog(
    pvi,
    n_trails: int,
    mag_range: tuple[float, float],
    length_range: tuple[float, float],
    seed: int,
    *,
    edge_pad: int = 200,
):
    """Random (not gridded) injections — matches what the production stage-1
    generator will do."""
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    W = pvi.getDimensions().x
    H = pvi.getDimensions().y
    wcs = pvi.wcs

    xs = rng.uniform(edge_pad, W - edge_pad, n_trails)
    ys = rng.uniform(edge_pad, H - edge_pad, n_trails)
    betas = rng.uniform(0, 180, n_trails)
    mags = rng.uniform(mag_range[0], mag_range[1], n_trails)
    lengths = rng.uniform(length_range[0], length_range[1], n_trails)

    rows = []
    for i, (x, y, beta, mg, L) in enumerate(zip(xs, ys, betas, mags, lengths)):
        sky = wcs.pixelToSky(float(x), float(y))
        rows.append({
            "injection_id": int(i),
            "ra": float(sky.getRa().asDegrees()),
            "dec": float(sky.getDec().asDegrees()),
            "source_type": "Trail",
            "trail_length": float(L),
            "mag": float(mg),
            "beta": float(beta),
            "x_hint": float(x),
            "y_hint": float(y),
        })
    return Table(rows=rows)


def inject_into_pvi(pvi_copy, catalog):
    from lsst.source.injection.inject_exposure import ExposureInjectTask
    task = ExposureInjectTask()
    res = task.run([catalog], pvi_copy, pvi_copy.psf, pvi_copy.photoCalib, pvi_copy.wcs)
    return res.output_exposure


def run_subtract(template, science, sources):
    from lsst.ip.diffim.subtractImages import AlardLuptonSubtractTask, AlardLuptonSubtractConfig
    cfg = AlardLuptonSubtractConfig()
    cfg.doApplyExternalCalibrations = False
    task = AlardLuptonSubtractTask(config=cfg)
    return task.run(template=template, science=science, sources=sources)


def run_detect(science, matchedTemplate, difference):
    from lsst.ip.diffim.detectAndMeasure import DetectAndMeasureTask, DetectAndMeasureConfig
    cfg = DetectAndMeasureConfig()
    cfg.doSkySources = False
    task = DetectAndMeasureTask(config=cfg)
    return task.run(science=science, matchedTemplate=matchedTemplate, difference=difference)


# ---------------------------------------------------------------------------
# Label and normalization
# ---------------------------------------------------------------------------
def build_truth_from_injection_geometry(
    diffim_variance: np.ndarray,
    empirical_injected_only: np.ndarray,
    *,
    sigma_thresh: float,
) -> np.ndarray:
    v = diffim_variance.astype(np.float32, copy=False)
    finite = np.isfinite(v) & (v > 0)
    floor = float(np.quantile(v[finite], 0.02)) if finite.any() else 1.0
    v_eff = np.maximum(v, floor)
    snr = empirical_injected_only.astype(np.float32, copy=False) / np.sqrt(v_eff + 1e-12)
    truth = ((snr >= sigma_thresh) & (empirical_injected_only > 0)).astype(np.uint8)
    return truth


def _load_diffim_norm():
    """Load ADCNN/data/diffim_norm.py as a bare module WITHOUT triggering
    ADCNN/__init__.py (which imports torch — not present in the LSST env)."""
    import importlib.util
    norm_path = Path(__file__).resolve().parents[3] / "ADCNN" / "data" / "diffim_norm.py"
    spec = importlib.util.spec_from_file_location("diffim_norm", norm_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_channels(diffim_img, diffim_var, bad_mask):
    """Three-channel NN input, already in the form the training dataset will
    consume. See ADCNN/data/diffim_norm.py for the math."""
    m = _load_diffim_norm()
    ch_signed = m.normalize_diffim_variance(
        diffim_img, diffim_var, bad_mask=bad_mask, clip=5.0,
    )
    ch_var = m.normalize_variance_channel(diffim_var, bad_mask=bad_mask)
    ch_bad = bad_mask.astype(np.float32, copy=False)
    return ch_signed, ch_var, ch_bad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    visit, detector, manifest_row = resolve_pair(args)

    out_dir = Path(args.out_root) / f"v{visit}_d{detector}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "driver.log"
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)

    # Per-task seed derived from (visit, detector) if not specified, so retries
    # are identical but different tasks get different injection realizations.
    if args.seed is None:
        seed = int((visit * 100003 + detector * 7919) & 0x7fffffff)
    else:
        seed = int(args.seed)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    report: dict[str, Any] = dict(
        visit=visit, detector=detector, seed=seed, status="running",
    )

    try:
        butler = get_butler(args.repo, args.collections)
        did, pvi, sources = get_pvi_bundle(butler, visit, detector)
        band = pvi.filter.bandLabel
        physical_filter = pvi.filter.physicalLabel
        print(f"[info] PVI band={band} phys={physical_filter} sources={len(sources)} "
              f"elapsed={time.time()-t0:.1f}s")

        template, n_tmpl_refs = get_template(butler, pvi, args.skymap, physical_filter)
        print(f"[info] template refs={n_tmpl_refs}, elapsed={time.time()-t0:.1f}s")

        # Draw randomized injection parameters for this task.
        n_trails = int(rng.integers(args.n_trails_min, args.n_trails_max + 1))
        catalog = build_injection_catalog(
            pvi,
            n_trails=n_trails,
            mag_range=(args.mag_min, args.mag_max),
            length_range=(args.trail_length_min, args.trail_length_max),
            seed=seed,
        )
        print(f"[info] injection catalog: {len(catalog)} trails")

        # ExposureInjectTask mutates the passed catalog (adds/renames cols),
        # so snapshot the columns we'll need for metadata writes BEFORE the
        # call — otherwise KeyError('injection_id') downstream.
        inj_ids = np.array(catalog["injection_id"], dtype=np.int32)
        inj_x = np.array(catalog["x_hint"], dtype=np.float32)
        inj_y = np.array(catalog["y_hint"], dtype=np.float32)
        inj_beta = np.array(catalog["beta"], dtype=np.float32)
        inj_length = np.array(catalog["trail_length"], dtype=np.float32)
        inj_mag = np.array(catalog["mag"], dtype=np.float32)

        pvi_clean = pvi
        pvi_injected = inject_into_pvi(pvi.clone(), catalog)
        print(f"[info] injected, elapsed={time.time()-t0:.1f}s")

        sub_clean = run_subtract(template=template, science=pvi_clean, sources=sources)
        sub_inj = run_subtract(template=template, science=pvi_injected, sources=sources)
        print(f"[info] subtractions done, elapsed={time.time()-t0:.1f}s")

        diffim_clean_img = sub_clean.difference.image.array.astype(np.float32, copy=True)
        diffim_inj_img = sub_inj.difference.image.array.astype(np.float32, copy=True)
        diffim_inj_var = sub_inj.difference.variance.array.astype(np.float32, copy=True)
        diffim_inj_mask = sub_inj.difference.mask.array.copy()
        matched_template_inj = sub_inj.matchedTemplate.image.array.astype(np.float32, copy=True)

        # bad-pixel mask from diffim mask plane.
        mask_planes = sub_inj.difference.mask.getMaskPlaneDict()
        bad_bits = 0
        for nm in ("EDGE", "NO_DATA", "BAD", "SAT", "CR"):
            if nm in mask_planes:
                bad_bits |= 1 << mask_planes[nm]
        bad_mask = ((diffim_inj_mask & bad_bits) != 0)

        empirical = (diffim_inj_img - diffim_clean_img).astype(np.float32)
        truth = build_truth_from_injection_geometry(
            diffim_inj_var, empirical, sigma_thresh=args.truth_sigma,
        )

        # Pre-compute the three NN input channels so downstream pack step is
        # dead simple and the training-time representation is fixed at
        # generation time.
        ch_signed, ch_var, ch_bad = build_channels(diffim_inj_img, diffim_inj_var, bad_mask)

        # AP baseline (logged only, not a label).
        try:
            det = run_detect(
                science=pvi_injected,
                matchedTemplate=sub_inj.matchedTemplate,
                difference=sub_inj.difference,
            )
            dia_xy = np.array(
                [[s.getCentroid().getX(), s.getCentroid().getY()] for s in det.diaSources],
                dtype=np.float32,
            )
        except Exception as e:
            print(f"[warn] DetectAndMeasureTask failed: {e}")
            dia_xy = np.zeros((0, 2), dtype=np.float32)

        # Per-trail recovery: for each injection, did AP find anything within
        # a small radius? This is baseline info, not a label.
        recovered_by_ap = np.zeros(len(inj_ids), dtype=np.uint8)
        if dia_xy.size > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(dia_xy)
            for i in range(len(inj_ids)):
                hits = tree.query_ball_point([float(inj_x[i]), float(inj_y[i])], r=15.0)
                if hits:
                    recovered_by_ap[i] = 1

        shard_path = out_dir / f"shard_v{visit}_d{detector}.npz"
        np.savez_compressed(
            shard_path,
            # NN-ready channels (float32, clipped to ±5).
            ch_signed=ch_signed,
            ch_var=ch_var,
            ch_bad=ch_bad,
            # Binary label.
            truth=truth,
            # Raw arrays for offline inspection / regeneration of labels.
            diffim_clean=diffim_clean_img,
            diffim_injected=diffim_inj_img,
            diffim_variance=diffim_inj_var,
            diffim_bad_mask=bad_mask.astype(np.uint8),
            empirical_injected_only=empirical,
            matched_template_inj=matched_template_inj,
            # AP baseline
            dia_sources_xy=dia_xy,
            recovered_by_ap=recovered_by_ap,
            # Metadata arrays (pre-injection snapshot — ExposureInjectTask
            # mutates the catalog columns in place).
            injection_ids=inj_ids,
            injection_x=inj_x,
            injection_y=inj_y,
            injection_beta=inj_beta,
            injection_trail_length=inj_length,
            injection_mag=inj_mag,
        )
        print(f"[info] shard written: {shard_path}")

        # Per-trail CSV metadata row.
        import csv
        csv_path = out_dir / f"meta_v{visit}_d{detector}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "visit", "detector", "band", "physical_filter",
                "injection_id", "x", "y", "beta", "trail_length", "mag",
                "recovered_by_ap_within_15px", "seed",
            ])
            for i in range(len(inj_ids)):
                w.writerow([
                    visit, detector, band, physical_filter,
                    int(inj_ids[i]), float(inj_x[i]), float(inj_y[i]),
                    float(inj_beta[i]), float(inj_length[i]), float(inj_mag[i]),
                    int(recovered_by_ap[i]), seed,
                ])
        print(f"[info] meta csv written: {csv_path}")

        report.update(
            status="ok",
            band=band,
            physical_filter=physical_filter,
            n_trails=int(len(inj_ids)),
            n_truth_positive_pixels=int(truth.sum()),
            n_dia_sources=int(dia_xy.shape[0]),
            n_recovered_by_ap=int(recovered_by_ap.sum()),
            diffim_sigma_mad=float(
                1.4826 * np.median(np.abs(diffim_inj_img[np.isfinite(diffim_inj_img)]))
            ),
            empirical_min=float(empirical.min()),
            empirical_max=float(empirical.max()),
            wall_clock_seconds=time.time() - t0,
            shard_path=str(shard_path),
        )

    except Exception as e:
        report.update(status="fail", error=str(e))
        traceback.print_exc()

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[done]", json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
