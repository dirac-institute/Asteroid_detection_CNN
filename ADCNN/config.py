# config.py
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    train_h5: str = "/home/karlo/train_chunked.h5"
    test_h5:  str | None = "../DATA/test.h5"
    train_csv: str | None = "../DATA/train.csv"
    test_csv:  str | None = "../DATA/test.csv"
    tile: int = 128

@dataclass
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 10
    pin_memory: bool = True

@dataclass
class TrainConfig:
    seed: int = 1337
    warmup_epochs:   int   = 5
    warmup_batches:  int   = 800
    warmup_lr:       float = 2e-4
    warmup_pos_weight: float = 40.0

    head_epochs:   int   = 10
    head_batches:  int   = 2000
    head_lr:       float = 3e-5
    head_pos_weight: float = 5.0

    tail_epochs:   int   = 6
    tail_batches:  int   = 2500
    tail_lr:       float = 1.5e-4
    tail_pos_weight: float = 2.0

    max_epochs:   int   = 60
    val_every:    int   = 3
    base_lrs:     tuple[float, float, float] = (3e-4, 2e-4, 1e-4)
    weight_decay: float = 1e-4

    thr_beta: float = 1.0
    thr_pos_rate_early: tuple[float, float] = (0.03, 0.10)
    thr_pos_rate_late:  tuple[float, float] = (0.08, 0.12)

    save_best_to: str = "../checkpoints/ckpt_best.pt"
    init_head_prior: float = 0.70

@dataclass
class Config:
    # Use default_factory for nested dataclasses
    data:   DataConfig   = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    train:  TrainConfig  = field(default_factory=TrainConfig)
