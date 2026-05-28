"""Butler helpers for diffim dataset generation: source-catalog conversion plus the
difference-image primitives (template fetch, AlardLupton subtraction, DIA detect).

Used by ``simulate`` (injected-trail training/test sets) and ``build_real`` (real-asteroid
test set)."""


def _get_schema_names(catalog):
    try:
        return list(catalog.schema.getNames())
    except Exception:
        return [item.field.getName() for item in catalog.schema]

def catalog_to_pandas(catalog, measueTrails=False):
    df = catalog.to_pandas()
    if measueTrails:
        trail_fields = [
            name for name in _get_schema_names(catalog)
            if name.startswith("ext_trailedSources_Naive_") or name.startswith("ext_trailedSources_Veres_")
        ]
        for name in trail_fields:
            if name not in df.columns:
                df[name] = [record[name] for record in catalog]
    return df


# ======================================================================================
# Diffim primitives (template fetch + AlardLupton subtract + DIA detect)
# ======================================================================================
#
# Used by simulate.py / build_real.py. Two stack-specific quirks are baked in:
#   1) `band=:band` bind on template_coadd queries silently no-ops on this
#      stack — must filter Python-side via `r.dataId.get("band") == band`.
#   2) The OVERLAPS clause must use `patch.region`, NOT `template_coadd.region`
#      (region lives on the skypix dim, not the dataset type).

def get_template(butler, pvi, skymap, physical_filter, stage3_collection=None):
    """Fetch all overlapping same-band template_coadd patches and assemble a
    template Exposure aligned to the science PVI bbox/wcs.

    Returns:
        (template_exposure, n_template_refs)
    """
    import lsst.geom
    from lsst.ip.diffim.getTemplate import GetTemplateTask, GetTemplateConfig
    from lsst.sphgeom import ConvexPolygon

    band = pvi.filter.bandLabel
    wcs = pvi.wcs
    bbox = pvi.getBBox()
    corners_pix = [lsst.geom.Point2D(c) for c in bbox.getCorners()]
    corners_sky = [wcs.pixelToSky(p) for p in corners_pix]
    region = ConvexPolygon([c.getVector() for c in corners_sky])

    query_kwargs = dict(
        where="skymap = :skymap AND patch.region OVERLAPS :region",
        bind={"skymap": skymap, "region": region},
        findFirst=True,
    )
    if stage3_collection is not None:
        query_kwargs["collections"] = (
            [stage3_collection] if isinstance(stage3_collection, str) else list(stage3_collection)
        )

    all_refs = list(butler.registry.queryDatasets("template_coadd", **query_kwargs))
    refs = [r for r in all_refs if r.dataId.get("band") == band]
    if not refs:
        bands_seen = sorted({r.dataId.get("band") for r in all_refs})
        raise RuntimeError(
            f"No {band}-band template_coadd overlapping science visit "
            f"(raw_overlap={len(all_refs)}, bands_seen={bands_seen})"
        )

    by_tract: dict = {}
    dataIds: dict = {}
    for ref in refs:
        tract = int(ref.dataId["tract"])
        by_tract.setdefault(tract, []).append(butler.getDeferred(ref))
        dataIds.setdefault(tract, []).append(ref.dataId)

    cfg = GetTemplateConfig()
    task = GetTemplateTask(config=cfg)
    out = task.run(
        coaddExposureHandles=by_tract,
        bbox=bbox,
        wcs=wcs,
        dataIds=dataIds,
        physical_filter=physical_filter,
    )
    return out.template, len(refs)


def run_subtract(template, science, sources):
    """AlardLupton PSF-matched subtraction. `sources` should be the
    single_visit_star_footprints catalog (kernel candidates).

    Returns the SubtractTask Struct with: difference, matchedTemplate, ...
    """
    from lsst.ip.diffim.subtractImages import (
        AlardLuptonSubtractTask,
        AlardLuptonSubtractConfig,
    )
    cfg = AlardLuptonSubtractConfig()
    cfg.doApplyExternalCalibrations = False
    task = AlardLuptonSubtractTask(config=cfg)
    return task.run(template=template, science=science, sources=sources)


def run_detect_diffim(science, matchedTemplate, difference, threshold=5.0, measueTrails=False):
    """DetectAndMeasure on the difference image. Returns the task Struct;
    use `.diaSources` for the catalog. Schema is API-compatible with
    SingleFrameDetectAndMeasureTask outputs (footprints, centroids, fluxes),
    so existing code that calls `.getFootprint()` / `.getCentroid()` works
    unchanged.
    """
    if measueTrails:
        try:
            import lsst.meas.extensions.trailedSources  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "measueTrails=True requires lsst.meas.extensions.trailedSources to be set up."
            ) from e

    from lsst.ip.diffim.detectAndMeasure import (
        DetectAndMeasureTask,
        DetectAndMeasureConfig,
    )
    cfg = DetectAndMeasureConfig()
    cfg.doSkySources = False
    cfg.detection.thresholdValue = float(threshold)
    if measueTrails:
        for plugin_name in (
            "base_SdssCentroid",
            "base_SdssShape",
            "ext_trailedSources_Naive",
            "ext_trailedSources_Veres",
        ):
            cfg.measurement.plugins.names.add(plugin_name)
        cfg.measurement.slots.centroid = "base_SdssCentroid"
        cfg.measurement.slots.shape = "base_SdssShape"

    task = DetectAndMeasureTask(config=cfg)
    return task.run(science=science, matchedTemplate=matchedTemplate, difference=difference)


def fetch_diffim_inputs(butler, dataId, skymap, stage3_collection=None):
    """Fetch the science PVI, the kernel-candidate source list, and the
    template Exposure for one (visit, detector). Caller runs subtract +
    detect themselves (typically twice: clean and injected).

    Returns:
        (pvi, sources, template, physical_filter, n_template_refs)
    """
    pvi = butler.get("preliminary_visit_image", dataId=dataId)
    sources = butler.get("single_visit_star_footprints", dataId=dataId)
    physical_filter = pvi.filter.physicalLabel
    template, n_refs = get_template(butler, pvi, skymap, physical_filter, stage3_collection=stage3_collection)
    return pvi, sources, template, physical_filter, n_refs
