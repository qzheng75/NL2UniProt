from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class MetricConfig:
    metric_type: Literal["TopKAcc"]
    metric_args: dict[str, Any]


@dataclass
class LossConfig:
    loss_type: Literal["CLIPLoss", "TorchLossWrapper"]
    loss_args: dict[str, Any]


@dataclass
class ModelConfig:
    model_type: Literal["faesm_bert.FAEsmBertEncoder"]
    model_args: dict[str, Any]


@dataclass
class OptimizerConfig:
    optimizer_type: Literal["AdamW"]
    optimizer_args: dict[str, dict[str, Any]]


@dataclass
class SchedulerConfig:
    scheduler_type: Literal["ReduceLROnPlateau", "CosineAnnealingLR", "PolynomialDecay"]
    scheduler_args: dict[str, Any]


@dataclass
class DatasetConfig:
    dataset_type: Literal["RawDescSeqDataset", "EmbeddedDescDataset"]
    dataset_args: dict[str, dict[str, Any]]


@dataclass
class DataloaderConfig:
    batch_size: int
    split: Literal["train", "val", "test"]
    collate_fn: Literal["default", "dual_encoder_collate_fn"]
    collate_args: dict[str, Any]
    sampler_type: Literal["RandomSampler", "DistributedSampler"]
    sampler_args: dict[str, Any]
    num_workers: int = 0


@dataclass
class SaveModelConfig:
    save_model: bool
    mode: str = "min"
    monitor: str = "val/loss"
    save_model_dir: str = "trained_models/"
    identifier: str = "test"
    filename: str = "model-{epoch:02d}"
    save_top_k: int = 1
    every_n_epoch: int = 1


@dataclass
class LoggerConfig:
    logger_type: Literal["tensorboard", "wandb", "stdout"]
    logger_args: dict[str, Any]
    # monitor: list[Literal['train/loss', 'val/loss', 'val/evaluator',\
    #     'train/grad_norm', 'train/epoch_loss', 'val/epoch_loss']]


@dataclass
class TrainerConfig:
    max_epochs: int
    save_model_config: SaveModelConfig
    resume_from_checkpoint: str | None = None
    use_amp: bool = False
    gradient_clip_val: float = 0.0
    log_every_n_steps: int = 50
    device: str | None = None
    strategy: Literal["ddp", "auto"] = "auto"
    enable_progress_bar: bool = True
