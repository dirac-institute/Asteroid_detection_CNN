from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ADCNN.data.datasets import H5TiledDataset
from ADCNN.inference.postprocess import postprocess_predictions
from ADCNN.utils.helpers import draw_one_line


class _PanelTileDataset(Dataset):
    """Tile view over a fixed list of panel ids."""

    def __init__(self, base_ds: H5TiledDataset, panel_ids: list[int]):
        self.base = base_ds
        self.panel_ids = list(map(int, panel_ids))
        self.panel_id_set = set(self.panel_ids)
        self.indices = [k for k, (pid, _r, _c) in enumerate(self.base.indices) if int(pid) in self.panel_id_set]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[int(idx)]]


@dataclass
class RescueValidationResult:
    budget: int
    n_candidates: int
    new_tp: int
    missed_recall: float
    union_recall: float
    added_fp: int


class RescueValidator:
    """
    Small fixed-subset rescue validation for training-time monitoring.

    The subset is deterministic from the validation panel ids and seed, so the same
    subset is reused across epochs and repeated runs unless the config changes.
    """

    def __init__(
        self,
        *,
        h5_path: str,
        csv_path: str,
        val_panel_ids: list[int],
        seed: int,
        max_images: int,
        batch_size: int,
        num_workers: int,
        rescue_budget_primary: int,
        rescue_budget_secondary: Optional[int],
        rescue_overlap_policy: str,
        psf_width: int,
        threshold: float,
        pixel_gap: int,
        min_area: int,
        max_area: Optional[int],
        min_score: float,
        min_peak_probability: float,
        score_method: str,
        topk_fraction: float,
    ):
        self.h5_path = str(h5_path)
        self.csv_path = str(csv_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.primary_budget = int(rescue_budget_primary)
        self.secondary_budget = int(rescue_budget_secondary) if rescue_budget_secondary else None
        self.overlap_policy = str(rescue_overlap_policy)
        self.psf_width = int(psf_width)
        self.post_cfg = {
            "threshold": float(threshold),
            "pixel_gap": int(pixel_gap),
            "min_area": int(min_area),
            "max_area": None if max_area is None else int(max_area),
            "min_score": float(min_score),
            "min_peak_probability": float(min_peak_probability),
            "score_method": str(score_method),
            "topk_fraction": float(topk_fraction),
            "return_label_mask": True,
        }

        self.base_ds = H5TiledDataset(self.h5_path, tile=128, k_sigma=5.0)
        self.panel_ids = self._select_panel_ids(val_panel_ids, seed=seed, max_images=max_images)
        self.local_by_global = {pid: i for i, pid in enumerate(self.panel_ids)}

        self.pred_ds = _PanelTileDataset(self.base_ds, self.panel_ids)
        self.pred_loader = DataLoader(
            self.pred_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.catalog = self._build_catalog()
        self.total_objects = int(len(self.catalog))
        self.baseline_detected = int(self.catalog["stack_detection"].sum())
        self.missed_total = int((~self.catalog["stack_detection"]).sum())
        self.missed_maps, self.detected_maps = self._build_truth_maps()

    @staticmethod
    def _select_panel_ids(val_panel_ids: list[int], *, seed: int, max_images: int) -> list[int]:
        ids = sorted(set(map(int, val_panel_ids)))
        if max_images <= 0 or len(ids) <= max_images:
            return ids
        rng = np.random.default_rng(int(seed))
        pick = np.sort(rng.choice(np.asarray(ids, dtype=np.int64), size=int(max_images), replace=False))
        return pick.astype(int).tolist()

    def _build_catalog(self) -> pd.DataFrame:
        cat = pd.read_csv(self.csv_path)
        cat = cat[cat["image_id"].isin(self.panel_ids)].copy()
        cat["stack_detection"] = cat["stack_detection"].fillna(False).astype(bool)
        cat["global_image_id"] = cat["image_id"].astype(int)
        cat["image_id"] = cat["global_image_id"].map(self.local_by_global).astype(int)
        cat["object_id"] = np.arange(len(cat), dtype=np.int64)
        return cat

    def _build_truth_maps(self) -> tuple[np.ndarray, np.ndarray]:
        H, W = int(self.base_ds.H), int(self.base_ds.W)
        missed = np.zeros((len(self.panel_ids), H, W), dtype=np.int32)
        detected = np.zeros((len(self.panel_ids), H, W), dtype=np.int32)
        half = max(1, int(self.psf_width / 2))

        for row in self.catalog.itertuples(index=False):
            target = detected if bool(row.stack_detection) else missed
            lid = int(row.object_id) + 1
            img = int(row.image_id)

            mask = draw_one_line(
                np.zeros((H, W), dtype=np.uint8),
                (float(row.x), float(row.y)),
                float(row.beta),
                float(row.trail_length),
                true_value=1,
                line_thickness=half,
            ).astype(bool)
            target[img][mask] = lid

        return missed, detected

    @torch.no_grad()
    def _predict_subset(self, model: torch.nn.Module, device: torch.device) -> np.ndarray:
        model.eval()
        H, W = int(self.base_ds.H), int(self.base_ds.W)
        tile = int(self.base_ds.tile)
        preds = np.zeros((len(self.panel_ids), H, W), dtype=np.float32)

        for batch_idx, batch in enumerate(self.pred_loader):
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device, non_blocking=True)
            probs = torch.sigmoid(model(xb)).detach().cpu().numpy()[:, 0]

            start = batch_idx * self.pred_loader.batch_size
            for j, prob in enumerate(probs):
                ds_idx = self.pred_ds.indices[start + j]
                global_pid, r, c = self.base_ds.indices[ds_idx]
                local_pid = self.local_by_global[int(global_pid)]
                r0, c0 = int(r) * tile, int(c) * tile
                r1, c1 = min(r0 + tile, H), min(c0 + tile, W)
                preds[local_pid, r0:r1, c0:c1] = prob[: r1 - r0, : c1 - c0]

        return preds

    def _score_budget(self, candidates: list[dict], budget: int) -> RescueValidationResult:
        seen_missed: set[int] = set()
        new_tp = 0
        added_fp = 0

        top = candidates[: max(0, int(budget))]
        for cand in top:
            missed_ids = cand["missed_ids"]
            if missed_ids:
                fresh = missed_ids - seen_missed
                if fresh:
                    seen_missed.update(fresh)
                    new_tp += len(fresh)
            elif cand["is_fp"]:
                added_fp += 1

        missed_recall = float(new_tp / max(self.missed_total, 1))
        union_recall = float((self.baseline_detected + new_tp) / max(self.total_objects, 1))
        return RescueValidationResult(
            budget=int(budget),
            n_candidates=len(candidates),
            new_tp=int(new_tp),
            missed_recall=missed_recall,
            union_recall=union_recall,
            added_fp=int(added_fp),
        )

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
        t0 = time.time()
        preds = self._predict_subset(model, device=device)
        label_masks, detections_per_image = postprocess_predictions(preds, **self.post_cfg)

        all_candidates: list[dict] = []
        for img_id, detections in enumerate(detections_per_image):
            labels = label_masks[img_id]
            missed_map = self.missed_maps[img_id]
            detected_map = self.detected_maps[img_id]

            for det in detections:
                lid = int(det["label_id"])
                comp = labels == lid
                if not bool(comp.any()):
                    continue

                missed_ids = set(np.unique(missed_map[comp]).tolist())
                missed_ids.discard(0)

                detected_ids = set(np.unique(detected_map[comp]).tolist())
                detected_ids.discard(0)

                if self.overlap_policy == "ignore_baseline_duplicates" and detected_ids and not missed_ids:
                    continue

                all_candidates.append(
                    {
                        "score": float(det["score"]),
                        "missed_ids": missed_ids,
                        "is_fp": (len(missed_ids) == 0 and len(detected_ids) == 0),
                    }
                )

        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        primary = self._score_budget(all_candidates, self.primary_budget)
        secondary = None
        if self.secondary_budget is not None and self.secondary_budget != self.primary_budget:
            secondary = self._score_budget(all_candidates, self.secondary_budget)

        return {
            "subset_images": len(self.panel_ids),
            "subset_objects": self.total_objects,
            "baseline_detected": self.baseline_detected,
            "baseline_missed": self.missed_total,
            "primary": primary,
            "secondary": secondary,
            "elapsed_s": float(time.time() - t0),
        }
