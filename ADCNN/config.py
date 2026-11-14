"""Single source of truth for the ACTIVE ADCNN detection pipeline.

The active pipeline is the ``pipeline.json`` named by ``$ADCNN_PIPELINE`` (the Slurm stages set it),
else ``models/v2_D2/pipeline.json``. A pipeline bundles, as one immutable unit:

  * the stage-1 segmentation + stage-2 cutout-CNN model files,
  * the **MF_LEN trail-length de-bias** (``offset``/``slope``), and
  * the stage-2 detection-retention floor.

Why bundle the MF_LEN de-bias with the models instead of hardcoding a global default:
a domain-adapted stage-1 has a *different* matched-filter "ends-bloom", so it needs its
OWN de-bias. Mixing one model with another model's de-bias silently corrupts ``len_db``.
Keeping them together in one config makes them travel together.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE = REPO / "models" / "v2_D2" / "pipeline.json"


@dataclass(frozen=True)
class Pipeline:
    """The resolved active pipeline. Model paths are absolute; de-bias is model-specific."""
    name: str
    seg_model: Path
    cnn_model: Path
    mf_len_offset: float
    mf_len_slope: float
    cnn_thr_floor: float


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (REPO / p)


def load_pipeline() -> Pipeline:
    """Load and validate the active pipeline config. Fails loud on a missing/garbled file."""
    cfg = Path(os.environ.get("ADCNN_PIPELINE") or DEFAULT_PIPELINE)
    if not cfg.exists():
        raise FileNotFoundError(
            f"ADCNN pipeline config not found: {cfg}. Expected models/v2_D2/pipeline.json "
            f"(or set ADCNN_PIPELINE to a valid pipeline.json)."
        )
    d = json.loads(cfg.read_text())
    try:
        models = d["models"]
        deb = d["mf_len_debias"]
        return Pipeline(
            name=d.get("name", cfg.parent.name),
            seg_model=_resolve(models["segmentation"]),
            cnn_model=_resolve(models["cnn_postproc"]),
            mf_len_offset=float(deb["offset"]),
            mf_len_slope=float(deb["slope"]),
            cnn_thr_floor=float(d.get("cnn_thr_floor", 0.5)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed ADCNN pipeline config {cfg}: {e}") from e


# Resolved once at import (one small JSON read). None without a config: detect_night refuses to run.
try:
    ACTIVE = load_pipeline()
except FileNotFoundError:
    ACTIVE = None
