from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ADCNN.inference.postprocess import RECOMMENDED_POSTPROCESS_CONFIG


@dataclass
class DataConfig:
    train_h5: Optional[str] = None  # required
    test_h5: Optional[str] = "../DATA/test.h5"
    train_csv: Optional[str] = "../DATA/train.csv"
    test_csv: Optional[str] = "../DATA/test.csv"

    tile: int = 128
    val_frac: float = 0.10

    margin_pix: float = 0.0
    stack_col: str = "stack_detection"

    # Pixels to ignore in loss/metrics (e.g. real sources)
    real_labels_key: str = "real_labels"

    # Target mask source
    target_mask_mode: str = "hard"  # "hard" or "soft"
    soft_mask_sigma_pix: float = 2.0
    soft_mask_line_width: int = 1
    soft_mask_truncate: float = 4.0
    soft_mask_cache_dir: Optional[str] = None
    soft_mask_cache_size: int = 64
    soft_mask_cache_dtype: str = "float16"


@dataclass
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class ModelConfig:
    # Used for checkpoint validation metadata
    model_name: str = "UNetResSEASPP"
    model_hparams: Dict[str, Any] = field(default_factory=lambda: {"in_ch": 1, "out_ch": 1})
    norm_name: Optional[str] = "medmad_clip_k5"  # set to None if you don't use named normalization


@dataclass
class TrainConfig:
    seed: int = 1337
    deterministic: bool = False  # optional default; can still override via CLI flag

    # Mixture fractions (train sampler stream).
    # Default is still rescue-aware, but less extreme than the previous setup.
    frac_missed: float = 0.40
    frac_detected: float = 0.35
    frac_background: float = 0.25
    epoch_size: int = 0  # per-rank samples per epoch (0 -> ceil(N/world_size))

    fixed_thr: float = 0.5  # kept for compatibility with downstream evaluation

    init_head_prior: float = 0.70

    # Optional short stabilization phase before the main run.
    warmup_epochs: int = 3
    warmup_batches: int = 0
    warmup_lr: float = 1.5e-4
    warmup_pos_weight: float = 12.0

    # Main training schedule
    max_epochs: int = 24
    val_every: int = 3
    main_lr: float = 1.5e-4
    min_lr_ratio: float = 0.20
    lr_schedule: str = "cosine"  # "cosine" or "constant"
    weight_decay: float = 1e-4
    main_batches: int = 0  # 0 = full loader

    # Loss family:
    # - "bce": BCE only
    # - "blend"/"bce_ft": BCE + focal-Tversky
    # - "bce_dice": BCE + soft Dice
    # The lam ramp controls the auxiliary overlap-term weight in the non-BCE modes.
    loss_mode: str = "blend"
    ramp_kind: str = "linear"  # "linear" or "sigmoid"
    ramp_start_epoch: int = 8
    ramp_end_epoch: int = 18
    sigmoid_k: float = 8.0
    lam_max: float = 0.20

    # Loss params
    bce_pos_weight_main: float = 8.0
    ft_alpha: float = 0.45
    ft_gamma: float = 1.3

    # Real-label handling.
    # ignore: old behavior (remove from loss/metric)
    # downweight: keep them with reduced weight
    # full: treat them like all other pixels
    train_real_label_mode: str = "downweight"
    train_real_label_weight: float = 0.25
    val_real_label_mode: str = "ignore"
    val_real_label_weight: float = 0.0

    # Validation AUC control
    auc_batches: int = 0
    auc_bins: int = 256

    # Rescue-oriented validation on a small fixed validation subset.
    enable_rescue_val: bool = True
    rescue_val_every: int = 3
    rescue_val_every_early: int = 1
    rescue_val_early_epochs: int = 8
    rescue_val_max_images: int = 8
    rescue_val_seed_offset: int = 50_000
    rescue_budget_primary: int = 50
    rescue_budget_secondary: int = 15000
    rescue_overlap_policy: str = "ignore_baseline_duplicates"
    rescue_val_num_workers: int = 0
    rescue_val_psf_width: int = 40
    rescue_post_threshold: float = RECOMMENDED_POSTPROCESS_CONFIG["threshold"]
    rescue_post_pixel_gap: int = RECOMMENDED_POSTPROCESS_CONFIG["pixel_gap"]
    rescue_post_min_area: int = RECOMMENDED_POSTPROCESS_CONFIG["min_area"]
    rescue_post_max_area: Optional[int] = RECOMMENDED_POSTPROCESS_CONFIG["max_area"]
    rescue_post_min_score: float = RECOMMENDED_POSTPROCESS_CONFIG["min_score"]
    rescue_post_min_peak_probability: float = RECOMMENDED_POSTPROCESS_CONFIG["min_peak_probability"]
    rescue_post_score_method: str = RECOMMENDED_POSTPROCESS_CONFIG["score_method"]
    rescue_post_topk_fraction: float = RECOMMENDED_POSTPROCESS_CONFIG["topk_fraction"]

    # EMA defaults (can be overridden via CLI)
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_eval: bool = False

    # Checkpoints. The primary artifact is the last checkpoint.
    save_best_to: Optional[str] = None
    save_last_to: str = "../checkpoints/ckpt_last.pt"

    verbose: int = 2


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if self.data.train_h5 is None:
            raise ValueError("data.train_h5 is required (set via --train-h5).")

        if self.data.tile <= 0:
            raise ValueError(f"data.tile must be positive, got {self.data.tile}")
        if not (0.0 <= float(self.data.val_frac) <= 1.0):
            raise ValueError(f"data.val_frac must be in [0,1], got {self.data.val_frac}")
        if self.data.margin_pix < 0.0:
            raise ValueError(f"data.margin_pix must be non-negative, got {self.data.margin_pix}")
        if self.data.target_mask_mode not in ("hard", "soft"):
            raise ValueError(
                f"data.target_mask_mode must be 'hard' or 'soft', got {self.data.target_mask_mode}"
            )

        if self.loader.batch_size <= 0:
            raise ValueError(f"loader.batch_size must be positive, got {self.loader.batch_size}")
        if self.loader.num_workers < 0:
            raise ValueError(f"loader.num_workers must be non-negative, got {self.loader.num_workers}")

        frac_sum = float(self.train.frac_missed + self.train.frac_detected + self.train.frac_background)
        if frac_sum <= 0.0:
            raise ValueError("At least one mixture fraction must be positive.")
        if not (0.0 <= float(self.train.fixed_thr) <= 1.0):
            raise ValueError(f"train.fixed_thr must be in [0,1], got {self.train.fixed_thr}")

        if self.train.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.train.max_epochs <= 0:
            raise ValueError(f"train.max_epochs must be positive, got {self.train.max_epochs}")

        if self.train.ramp_kind not in ("linear", "sigmoid"):
            raise ValueError(f"train.ramp_kind must be 'linear' or 'sigmoid', got {self.train.ramp_kind}")
        if self.train.lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"train.lr_schedule must be 'cosine' or 'constant', got {self.train.lr_schedule}")
        if self.train.loss_mode not in ("blend", "bce", "bce_ft", "bce_dice"):
            raise ValueError(
                "train.loss_mode must be 'blend', 'bce', 'bce_ft', or 'bce_dice', "
                f"got {self.train.loss_mode}"
            )
        if not (0.0 <= float(self.train.ft_alpha) <= 1.0):
            raise ValueError(f"train.ft_alpha must be in [0,1], got {self.train.ft_alpha}")
        if not (0.0 <= float(self.train.lam_max) <= 1.0):
            raise ValueError(f"train.lam_max must be in [0,1], got {self.train.lam_max}")
        if self.train.train_real_label_mode not in ("ignore", "downweight", "full"):
            raise ValueError(
                f"train.train_real_label_mode must be 'ignore', 'downweight', or 'full', got {self.train.train_real_label_mode}"
            )
        if self.train.val_real_label_mode not in ("ignore", "downweight", "full"):
            raise ValueError(
                f"train.val_real_label_mode must be 'ignore', 'downweight', or 'full', got {self.train.val_real_label_mode}"
            )
        if self.train.rescue_overlap_policy not in ("ignore_baseline_duplicates", "keep_all"):
            raise ValueError(
                "train.rescue_overlap_policy must be 'ignore_baseline_duplicates' or 'keep_all', "
                f"got {self.train.rescue_overlap_policy}"
            )

        # Model metadata sanity
        hp = dict(self.model.model_hparams)
        if "in_ch" in hp and int(hp["in_ch"]) <= 0:
            raise ValueError(f"model.model_hparams['in_ch'] must be positive, got {hp['in_ch']}")
        if "out_ch" in hp and int(hp["out_ch"]) <= 0:
            raise ValueError(f"model.model_hparams['out_ch'] must be positive, got {hp['out_ch']}")
