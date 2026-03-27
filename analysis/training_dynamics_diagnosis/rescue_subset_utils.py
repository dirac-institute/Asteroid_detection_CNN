from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from ADCNN.utils.helpers import split_indices


def select_harder_val_panels(
    *,
    train_h5: str,
    train_csv: str,
    val_frac: float,
    seed: int,
    max_images: int,
    mode: str = "missed_count",
) -> list[int]:
    idx_tr, idx_va = split_indices(train_h5, val_frac=float(val_frac), seed=int(seed))
    val_ids = set(map(int, idx_va.tolist()))

    cat = pd.read_csv(train_csv).copy()
    cat = cat[cat["image_id"].isin(val_ids)].copy()
    cat["stack_detection"] = cat["stack_detection"].fillna(False).astype(bool)

    if mode == "missed_count":
        score = (~cat["stack_detection"]).groupby(cat["image_id"]).sum()
    elif mode == "missed_fraction":
        grouped = cat.groupby("image_id")["stack_detection"]
        score = 1.0 - grouped.mean()
    else:
        raise ValueError(f"Unknown harder-subset mode: {mode!r}")

    ordered = score.sort_values(ascending=False).index.astype(int).tolist()
    if max_images > 0:
        ordered = ordered[: int(max_images)]
    return ordered


def save_panel_subset(path: str | Path, panel_ids: Sequence[int], *, meta: dict | None = None) -> None:
    rec = {"panel_ids": [int(x) for x in panel_ids], "meta": meta or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
