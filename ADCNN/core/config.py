from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


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

    # Mixture fractions (train sampler stream)
    frac_missed: float = 0.60
    frac_detected: float = 0.25
    frac_background: float = 0.15
    epoch_size: int = 0  # per-rank samples per epoch (0 -> ceil(N/world_size))

    fixed_thr: float = 0.5  # kept for compatibility; selection is by AUC

    init_head_prior: float = 0.70

    warmup_epochs: int = 5
    warmup_batches: int = 800
    warmup_lr: float = 2e-4
    warmup_pos_weight: float = 40.0

    head_epochs: int = 10
    head_batches: int = 2000
    head_lr: float = 3e-5
    head_pos_weight: float = 5.0

    tail_epochs: int = 6
    tail_batches: int = 2500
    tail_lr: float = 1.5e-4
    tail_pos_weight: float = 2.0

    # Long training schedule
    max_epochs: int = 60
    val_every: int = 3
    base_lrs: Tuple[float, float, float] = (3e-4, 2e-4, 1e-4)
    weight_decay: float = 1e-4
    long_batches: int = 0  # 0 = full loader

    # Ramp: BCE -> focal-tversky
    ramp_kind: str = "linear"  # "linear" or "sigmoid"
    ramp_start_epoch: int = 11
    ramp_end_epoch: int = 40
    sigmoid_k: float = 8.0

    # Loss params (long)
    bce_pos_weight_long: float = 8.0
    ft_alpha: float = 0.45
    ft_gamma: float = 1.3

    # Validation AUC control
    auc_batches: int = 12
    auc_bins: int = 256

    # EMA defaults (can be overridden via CLI)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_eval: bool = True

    # Checkpoints
    save_best_to: str = "../checkpoints/ckpt_best.pt"
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

        if self.loader.batch_size <= 0:
            raise ValueError(f"loader.batch_size must be positive, got {self.loader.batch_size}")
        if self.loader.num_workers < 0:
            raise ValueError(f"loader.num_workers must be non-negative, got {self.loader.num_workers}")

        frac_sum = float(self.train.frac_missed + self.train.frac_detected + self.train.frac_background)
        if frac_sum <= 0.0:
            raise ValueError("At least one mixture fraction must be positive.")
        if not (0.0 <= float(self.train.fixed_thr) <= 1.0):
            raise ValueError(f"train.fixed_thr must be in [0,1], got {self.train.fixed_thr}")

        if self.train.warmup_epochs < 0 or self.train.head_epochs < 0 or self.train.tail_epochs < 0:
            raise ValueError("warmup/head/tail epochs must be non-negative")
        if self.train.max_epochs <= 0:
            raise ValueError(f"train.max_epochs must be positive, got {self.train.max_epochs}")

        if self.train.ramp_kind not in ("linear", "sigmoid"):
            raise ValueError(f"train.ramp_kind must be 'linear' or 'sigmoid', got {self.train.ramp_kind}")
        if not (0.0 <= float(self.train.ft_alpha) <= 1.0):
            raise ValueError(f"train.ft_alpha must be in [0,1], got {self.train.ft_alpha}")

        # Model metadata sanity
        hp = dict(self.model.model_hparams)
        if "in_ch" in hp and int(hp["in_ch"]) <= 0:
            raise ValueError(f"model.model_hparams['in_ch'] must be positive, got {hp['in_ch']}")
        if "out_ch" in hp and int(hp["out_ch"]) <= 0:
            raise ValueError(f"model.model_hparams['out_ch'] must be positive, got {hp['out_ch']}")