"""Single-visit / single-detector end-to-end diffim proof.

This script is the hard gate before any wider diffim scaffolding. It runs, in
order:

  1. fetch a preliminary_visit_image + star footprints from the DP2 butler,
  2. fetch overlapping template_coadd patches,
  3. warp the template to the PVI via GetTemplateTask,
  4. inject a small number of very bright synthetic trails into a copy of the
     PVI (so they are unambiguously visible),
  5. run AlardLuptonSubtractTask twice: once on the CLEAN PVI, once on the
     INJECTED PVI. The clean run uses exactly the same `sources` catalog as
     the injected run, so the fitted PSF-matching kernel is as close as
     possible between them.
  6. derive the empirical injected-only residual as
     (diffim_injected - diffim_clean),
  7. build the training truth mask from the INJECTED GEOMETRY projected into
     diffim space (rendering the injection catalog as a per-trail
     psf-convolved image, thresholded by the diffim variance). AP detections
     are *not* used for truth.
  8. run DetectAndMeasureTask on the injected diffim to see what AP would do
     on this case (baseline only; not a label),
  9. dump all arrays to an NPZ and produce PNG overlays for visual inspection.

The script is deliberately small, single-file, and driven by CLI args. It is
runnable either on a login node for quick debugging or via SLURM (see
slurm_proof.sh).
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

# LSST stack imports are deferred behind functions so that `--help` works
# without the stack being set up.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default="dp2_prep")
    p.add_argument(
        "--collections",
        nargs="+",
        default=[
            "LSSTCam/runs/DRP/DP2/v30_0_6_rc1/DM-53881/stage3",
            "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2",
        ],
        help="Butler collection chain, earlier wins. Stage3 typically carries"
        " template_coadd, stage2 carries preliminary_visit_image.",
    )
    p.add_argument("--visit", type=int, required=True)
    p.add_argument("--detector", type=int, required=True)
    p.add_argument(
        "--n-trails",
        type=int,
        default=5,
        help="Number of synthetic trails to inject. Keep small (< 20) for the"
        " proof so they are obvious in the output images.",
    )
    p.add_argument(
        "--trail-mag",
        type=float,
        default=22.0,
        help="Integrated magnitude of the injected trails. 22 is bright enough"
        " to be unambiguous on most DP2 detectors.",
    )
    p.add_argument(
        "--trail-length-px",
        type=float,
        default=30.0,
        help="Length of each injected trail in pixels.",
    )
    p.add_argument(
        "--truth-sigma",
        type=float,
        default=2.0,
        help="Empirical truth threshold in units of diffim σ. A pixel is"
        " positive iff the injected-only residual exceeds this many σ.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skymap",
        default="lsst_cells_v2",
        help="Which skymap the template_coadd is registered under.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Butler helpers
# ---------------------------------------------------------------------------
def get_butler(repo: str, collections: list[str]):
    from lsst.daf.butler import Butler
    return Butler(repo, collections=collections)


def get_pvi_bundle(butler, visit: int, detector: int):
    """Fetch all direct inputs for subtraction: PVI, its background, and the
    single_visit_star_footprints used as kernel candidate sources."""
    did = dict(instrument="LSSTCam", visit=visit, detector=detector)
    pvi = butler.get("preliminary_visit_image", dataId=did)
    background = None
    try:
        background = butler.get("preliminary_visit_image_background", dataId=did)
    except Exception as e:
        print(f"[warn] preliminary_visit_image_background missing: {e}")
    try:
        sources = butler.get("single_visit_star_footprints", dataId=did)
    except Exception as e:
        print(f"[warn] single_visit_star_footprints missing: {e}")
        sources = None
    return did, pvi, background, sources


def get_template(butler, pvi, skymap: str, physical_filter: str):
    """Find overlapping template_coadd patches and warp them to the PVI.

    Returns the template exposure produced by GetTemplateTask.
    """
    import lsst.geom
    from lsst.daf.butler import DatasetRef
    from lsst.ip.diffim.getTemplate import GetTemplateTask, GetTemplateConfig
    from lsst.sphgeom import ConvexPolygon

    band = pvi.filter.bandLabel

    # Polygon of the PVI on sky → find overlapping tract/patch of template_coadd.
    wcs = pvi.wcs
    bbox = pvi.getBBox()
    corners_pix = [lsst.geom.Point2D(c) for c in bbox.getCorners()]
    corners_sky = [wcs.pixelToSky(p) for p in corners_pix]
    region = ConvexPolygon([c.getVector() for c in corners_sky])

    # Query overlapping template_coadd patches. In the modern butler query
    # language the spatial region of a coadd lives on its skypix dimension
    # (patch), not on the dataset type itself. We filter by band in Python
    # (the bind-based band filter proved unreliable for the coadd query on
    # this stack version).
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
            f"No overlapping template_coadd in {skymap}/{band} for visit={pvi.visitInfo.id} "
            f"(raw overlap={len(all_refs)}, bands_seen={bands_seen})"
        )
    print(f"[info] template_coadd overlapping refs: {len(refs)} (raw={len(all_refs)}) "
          f"(first: band={refs[0].dataId.get('band')} tract={refs[0].dataId.get('tract')} "
          f"patch={refs[0].dataId.get('patch')})")

    # Group deferred handles and dataIds by tract. GetTemplateTask expects
    # BOTH coaddExposureHandles and dataIds to be dict[int (tract), list[...]],
    # paired element-wise within each tract. dataIds must be hashable
    # DataCoordinates (used as dict keys inside the task), NOT plain dicts.
    by_tract: dict[int, list[Any]] = {}
    dataIds: dict[int, list[Any]] = {}
    for ref in refs:
        tract = int(ref.dataId["tract"])
        by_tract.setdefault(tract, []).append(butler.getDeferred(ref))
        dataIds.setdefault(tract, []).append(ref.dataId)

    # Run GetTemplateTask.
    cfg = GetTemplateConfig()
    task = GetTemplateTask(config=cfg)
    result = task.run(
        coaddExposureHandles=by_tract,
        bbox=bbox,
        wcs=wcs,
        dataIds=dataIds,
        physical_filter=physical_filter,
    )
    return result.template


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def build_injection_catalog(
    pvi,
    n_trails: int,
    mag: float,
    length_px: float,
    seed: int,
) -> "astropy.table.Table":
    """Build a tiny injection catalog with N bright trails on a regular grid.

    Grid placement is deliberate so the trails are easy to find visually.
    Uses WCS to convert pixel positions to RA/Dec.
    """
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    H, W = pvi.getDimensions().y, pvi.getDimensions().x
    wcs = pvi.wcs

    # Grid of positions, with some jitter, well away from the edges.
    pad = 400
    xs = np.linspace(pad, W - pad, n_trails) + rng.uniform(-20, 20, n_trails)
    ys = np.linspace(pad, H - pad, n_trails) + rng.uniform(-20, 20, n_trails)
    betas = rng.uniform(0, 180, n_trails)

    rows = []
    for i, (x, y, beta) in enumerate(zip(xs, ys, betas)):
        sky = wcs.pixelToSky(float(x), float(y))
        rows.append({
            "injection_id": int(i),
            "ra": float(sky.getRa().asDegrees()),
            "dec": float(sky.getDec().asDegrees()),
            "source_type": "Trail",
            "trail_length": float(length_px),
            "mag": float(mag),
            "beta": float(beta),
            "x_hint": float(x),
            "y_hint": float(y),
        })
    cat = Table(rows=rows)
    return cat


def inject_into_pvi(pvi_copy, catalog):
    """Run ExposureInjectTask on a COPY of the PVI. Returns a new exposure."""
    from lsst.source.injection.inject_exposure import ExposureInjectTask
    task = ExposureInjectTask()
    res = task.run([catalog], pvi_copy, pvi_copy.psf, pvi_copy.photoCalib, pvi_copy.wcs)
    return res.output_exposure


# ---------------------------------------------------------------------------
# Subtraction & detection
# ---------------------------------------------------------------------------
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
# Truth from injected geometry in diffim space
# ---------------------------------------------------------------------------
def build_truth_from_injection_geometry(
    injection_catalog,
    pvi,
    diffim_variance: np.ndarray,
    empirical_injected_only: np.ndarray,
    *,
    sigma_thresh: float,
) -> np.ndarray:
    """Build a binary truth mask by thresholding the empirical injected-only
    residual (diffim_injected - diffim_clean) at `sigma_thresh` * per-pixel σ.

    The geometry is implicitly carried by the injection catalog: the residual
    is nonzero only where the injection actually added flux (up to PSF-
    matching imperfections). This is strictly "injected geometry projected
    into diffim space" — no AP detection touches this mask.
    """
    # Per-pixel sigma from the diffim variance plane, floored at the 2%
    # quantile to avoid division by tiny numbers near the edge.
    v = diffim_variance.astype(np.float32, copy=False)
    finite = np.isfinite(v) & (v > 0)
    if finite.any():
        floor = float(np.quantile(v[finite], 0.02))
    else:
        floor = 1.0
    v_eff = np.maximum(v, floor)
    snr = empirical_injected_only.astype(np.float32, copy=False) / np.sqrt(v_eff + 1e-12)
    truth = (snr >= sigma_thresh).astype(np.uint8)
    # Require the injected component to be POSITIVE: the trail adds flux in
    # the diffim, so negative excursions (template artefacts) must not count.
    truth[empirical_injected_only <= 0] = 0
    return truth


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def percentile_clip(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def save_png_overview(out_png: Path, arrays: dict[str, np.ndarray], catalog):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("science_clean", "Science (PVI, clean)"),
        ("science_injected", "Science (PVI + injections)"),
        ("template_warped", "Template (warped to PVI)"),
        ("diffim_clean", "Diffim CLEAN = science - template"),
        ("diffim_injected", "Diffim INJECTED"),
        ("empirical_injected_only", "Empirical = injected - clean"),
        ("truth_mask", "Truth (injected geometry in diffim space)"),
        ("diffim_variance", "Diffim variance plane (log)"),
        ("diffim_bad_mask", "Diffim bad-pixel mask"),
    ]
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, panels):
        if key not in arrays:
            ax.axis("off")
            ax.set_title(f"{title} (missing)")
            continue
        img = arrays[key]
        if key == "diffim_variance":
            img_show = np.log1p(np.maximum(img, 0.0))
            lo, hi = percentile_clip(img_show, 1, 99)
        elif key in ("truth_mask", "diffim_bad_mask"):
            img_show = img.astype(np.float32)
            lo, hi = 0, 1
        elif key.startswith("diffim") or key == "empirical_injected_only":
            lo_a, hi_a = percentile_clip(img, 1, 99)
            vmax = max(abs(lo_a), abs(hi_a))
            lo, hi = -vmax, vmax
            img_show = img
        else:
            lo, hi = percentile_clip(img, 1, 99)
            img_show = img
        im = ax.imshow(img_show, origin="lower", cmap="RdBu_r" if (key.startswith("diffim") or key == "empirical_injected_only") else "viridis", vmin=lo, vmax=hi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        # Overlay injection positions
        if catalog is not None and "x_hint" in catalog.colnames:
            for r in catalog:
                ax.plot(float(r["x_hint"]), float(r["y_hint"]), "x", color="yellow", markersize=8, markeredgewidth=1.2)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"proof_v{args.visit}_d{args.detector}.log"
    sys.stdout = sys.stderr = open(log_path, "w", buffering=1)
    print(f"[info] args: {vars(args)}")

    try:
        butler = get_butler(args.repo, args.collections)
        print(f"[info] butler ready, elapsed={time.time()-t0:.1f}s")

        did, pvi, background, sources = get_pvi_bundle(butler, args.visit, args.detector)
        if sources is None:
            raise RuntimeError("Cannot proceed without single_visit_star_footprints.")
        print(f"[info] PVI loaded, shape={pvi.getDimensions()} band={pvi.filter.bandLabel} "
              f"filter={pvi.filter.physicalLabel} sources={len(sources)}, elapsed={time.time()-t0:.1f}s")

        template = get_template(
            butler, pvi, skymap=args.skymap, physical_filter=pvi.filter.physicalLabel
        )
        print(f"[info] template built, shape={template.getDimensions()}, elapsed={time.time()-t0:.1f}s")

        # Clean PVI stays pristine; make a deep copy for the injected run.
        import copy
        pvi_clean = pvi
        pvi_for_injection = pvi.clone()

        catalog = build_injection_catalog(
            pvi, args.n_trails, args.trail_mag, args.trail_length_px, args.seed
        )
        print(f"[info] injection catalog: {len(catalog)} trails")
        print(catalog)

        pvi_injected = inject_into_pvi(pvi_for_injection, catalog)
        print(f"[info] injected, elapsed={time.time()-t0:.1f}s")

        # Run subtraction twice with the SAME `sources` to keep the fitted
        # kernel as close as possible between the two runs.
        sub_clean = run_subtract(template=template, science=pvi_clean, sources=sources)
        print(f"[info] sub CLEAN done, elapsed={time.time()-t0:.1f}s")
        sub_inj = run_subtract(template=template, science=pvi_injected, sources=sources)
        print(f"[info] sub INJECTED done, elapsed={time.time()-t0:.1f}s")

        # Extract numpy arrays.
        diffim_clean_img = sub_clean.difference.image.array.astype(np.float32, copy=True)
        diffim_clean_var = sub_clean.difference.variance.array.astype(np.float32, copy=True)
        diffim_clean_mask = sub_clean.difference.mask.array.copy()

        diffim_inj_img = sub_inj.difference.image.array.astype(np.float32, copy=True)
        diffim_inj_var = sub_inj.difference.variance.array.astype(np.float32, copy=True)
        diffim_inj_mask = sub_inj.difference.mask.array.copy()

        matched_template_clean = sub_clean.matchedTemplate.image.array.astype(np.float32, copy=True)
        matched_template_inj = sub_inj.matchedTemplate.image.array.astype(np.float32, copy=True)

        empirical = (diffim_inj_img - diffim_clean_img).astype(np.float32)

        # Build a bad-mask (any bit set in the union of a few planes).
        mask_planes = sub_inj.difference.mask.getMaskPlaneDict()
        bad_bits = 0
        for name in ("EDGE", "NO_DATA", "BAD", "SAT", "CR"):
            if name in mask_planes:
                bad_bits |= 1 << mask_planes[name]
        bad_mask = (diffim_inj_mask & bad_bits) != 0

        truth_mask = build_truth_from_injection_geometry(
            catalog,
            pvi_injected,
            diffim_inj_var,
            empirical,
            sigma_thresh=args.truth_sigma,
        )

        # Detection on the INJECTED diffim (BASELINE, not training label).
        try:
            det = run_detect(
                science=pvi_injected,
                matchedTemplate=sub_inj.matchedTemplate,
                difference=sub_inj.difference,
            )
            dia_sources = det.diaSources
            n_dia = len(dia_sources)
            dia_xy = np.array([[s.getCentroid().getX(), s.getCentroid().getY()] for s in dia_sources])
            print(f"[info] DetectAndMeasureTask produced {n_dia} diaSources on the injected diffim")
        except Exception as e:
            print(f"[warn] DetectAndMeasureTask failed: {e}")
            traceback.print_exc()
            dia_xy = np.zeros((0, 2), dtype=np.float32)

        # ------------------------------------------------------------------
        # Save outputs.
        # ------------------------------------------------------------------
        tag = f"v{args.visit}_d{args.detector}"
        npz_path = out_dir / f"proof_{tag}.npz"
        np.savez_compressed(
            npz_path,
            science_clean=pvi_clean.image.array.astype(np.float32),
            science_injected=pvi_injected.image.array.astype(np.float32),
            template_warped=template.image.array.astype(np.float32),
            matched_template_clean=matched_template_clean,
            matched_template_inj=matched_template_inj,
            diffim_clean=diffim_clean_img,
            diffim_injected=diffim_inj_img,
            diffim_variance=diffim_inj_var,
            diffim_bad_mask=bad_mask.astype(np.uint8),
            empirical_injected_only=empirical,
            truth_mask=truth_mask,
            dia_sources_xy=dia_xy,
            injection_x_hint=np.array(catalog["x_hint"]),
            injection_y_hint=np.array(catalog["y_hint"]),
            injection_beta=np.array(catalog["beta"]),
            injection_trail_length=np.array(catalog["trail_length"]),
            injection_mag=np.array(catalog["mag"]),
        )
        print(f"[info] NPZ saved: {npz_path}")

        save_png_overview(
            out_dir / f"proof_{tag}.png",
            dict(
                science_clean=pvi_clean.image.array,
                science_injected=pvi_injected.image.array,
                template_warped=template.image.array,
                diffim_clean=diffim_clean_img,
                diffim_injected=diffim_inj_img,
                empirical_injected_only=empirical,
                truth_mask=truth_mask,
                diffim_variance=diffim_inj_var,
                diffim_bad_mask=bad_mask.astype(np.uint8),
            ),
            catalog,
        )

        # ------------------------------------------------------------------
        # Tiny quantitative sanity report.
        # ------------------------------------------------------------------
        report = {
            "visit": args.visit,
            "detector": args.detector,
            "band": pvi.filter.bandLabel,
            "n_trails_injected": int(args.n_trails),
            "n_truth_positive_pixels": int(truth_mask.sum()),
            "n_dia_sources_on_injected_diffim": int(dia_xy.shape[0]),
            "empirical_max": float(np.max(empirical)),
            "empirical_min": float(np.min(empirical)),
            "diffim_inj_sigma_mad": float(
                1.4826 * np.median(np.abs(diffim_inj_img[np.isfinite(diffim_inj_img)]))
            ),
            "wall_clock_seconds": time.time() - t0,
        }
        with open(out_dir / f"proof_{tag}.json", "w") as f:
            json.dump(report, f, indent=2)
        print("[done]", json.dumps(report, indent=2))
        return 0

    except Exception as e:
        print(f"[fatal] {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
