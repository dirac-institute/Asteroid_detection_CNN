"""Single source of truth for the ACTIVE ADCNN detection pipeline.

The deployed default is the promoted current detector described by
``models/current/pipeline.json``. A pipeline bundles, as one immutable unit:

  * the stage-1 segmentation + stage-2 cutout-CNN model files,
  * the **MF_LEN trail-length de-bias** (``offset``/``slope``), and
  * the stage-2 detection-retention floor + a pointer to the frozen alert op-point.

Why bundle the MF_LEN de-bias with the models instead of hardcoding a global default:
a domain-adapted stage-1 has a *different* matched-filter "ends-bloom", so it needs its
OWN de-bias. Mixing one model with another model's de-bias silently corrupts ``len_db``
(the linker length gate then deletes real detections — the "0-pairs" failure we hit during
the v2_D sprint). Keeping them together in one config makes them travel together.

Selecting a pipeline (in priority order):
  1. an explicit ``name_or_path`` argument to :func:`load_pipeline`,
  2. the ``ADCNN_PIPELINE`` env var — a name (``current`` / ``legacy_v1``) or a path to a
     ``pipeline.json``,
  3. the built-in default ``models/current/pipeline.json``.

Individual values stay overridable for advanced/standalone use: ``ADCNN_MF_LEN_OFFSET`` /
``ADCNN_MF_LEN_SLOPE`` override the de-bias (e.g. set both so length is emitted raw when
*fitting* a new de-bias). Prefer selecting a whole pipeline over overriding pieces.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE = REPO / "models" / "current" / "pipeline.json"

# The ONE runtime output location (layout since 2026-07-14): repo-root outputs/ holds
# runs/ (night + campaign run dirs), logs/ (slurm logs), training_runs/, query_snapshots/.
# Overridable via ADCNN_OUTPUTS (e.g. a scratch filesystem). Nothing may write into the
# package tree at runtime -- outputs/ is gitignored, the package tree is code + frozen calib.
OUTPUTS = Path(os.environ.get("ADCNN_OUTPUTS", str(REPO / "outputs")))


def outputs_dir(*parts: str) -> Path:
    """Resolve (and mkdir -p) a directory under the outputs root."""
    p = OUTPUTS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Pipeline:
    """The resolved active pipeline. Model paths are absolute; de-bias is model-specific."""
    name: str
    provenance: str
    seg_model: Path
    cnn_model: Path
    cnn_sidecar: Optional[Path]
    mf_len_offset: float
    mf_len_slope: float
    cnn_thr_floor: float
    alert_op_point: Optional[Path]
    source: Path


def _resolve(rel: Optional[str]) -> Optional[Path]:
    if rel is None:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (REPO / p)


def _config_path(name_or_path: Optional[Union[str, Path]]) -> Path:
    """Resolve a pipeline selector (arg > env > default) to a pipeline.json path.

    Accepts a full path to a ``pipeline.json``, a directory under ``models/`` containing one,
    or a bare name (``current``/``legacy_v1``) -> ``models/<name>/pipeline.json``.
    """
    sel = name_or_path or os.environ.get("ADCNN_PIPELINE") or DEFAULT_PIPELINE
    p = Path(sel)
    if p.is_dir():
        return p / "pipeline.json"
    if p.suffix == ".json" or p.name == "pipeline.json":
        return p
    # bare name -> models/<name>/pipeline.json
    return REPO / "models" / p.name / "pipeline.json"


def load_pipeline(name_or_path: Optional[Union[str, Path]] = None) -> Pipeline:
    """Load and validate the active pipeline config. Fails loud on a missing/garbled file."""
    cfg = _config_path(name_or_path)
    if not cfg.exists():
        raise FileNotFoundError(
            f"ADCNN pipeline config not found: {cfg}. Expected models/current/pipeline.json "
            f"(or set ADCNN_PIPELINE to a valid pipeline.json / name)."
        )
    d = json.loads(cfg.read_text())
    try:
        models = d["models"]
        deb = d["mf_len_debias"]
        off = float(os.environ.get("ADCNN_MF_LEN_OFFSET", deb["offset"]))
        slope = float(os.environ.get("ADCNN_MF_LEN_SLOPE", deb["slope"]))
        return Pipeline(
            name=d.get("name", cfg.parent.name),
            provenance=d.get("provenance", ""),
            seg_model=_resolve(models["segmentation"]),
            cnn_model=_resolve(models["cnn_postproc"]),
            cnn_sidecar=_resolve(models.get("cnn_sidecar")),
            mf_len_offset=off,
            mf_len_slope=slope,
            cnn_thr_floor=float(d.get("cnn_thr_floor", 0.5)),
            alert_op_point=_resolve(d.get("alert_op_point")),
            source=cfg,
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed ADCNN pipeline config {cfg}: {e}") from e


# Resolved once at import for callers that just want the default model/de-bias (e.g. catalog.py).
# Cheap (one small JSON read); callers that need a non-default pipeline call load_pipeline() directly.
try:
    ACTIVE = load_pipeline()
except FileNotFoundError:
    ACTIVE = None  # repo without a current/ config (e.g. mid-bootstrap); callers fall back.
