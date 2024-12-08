from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class MetricConfig:
    metric_type: Literal["TopKAcc"]
    metric_args: dict[str, Any]


@dataclass
class LossConfig:
    loss_type: Literal["CLIPLoss", "TorchLossWrapper"] = "CLIPLoss"
    loss_args: dict[str, Any] = {}


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
    dataset_args: dict[str, Any]


@dataclass
class DataloaderConfig:
    batch_size: int
    collate_fn: Literal["default", "dual_encoder_collate_fn"]
    collate_args: dict[str, Any]
    sampler_type: Literal["RandomSampler"]
    sampler_args: dict[str, Any]


@dataclass
class SaveModelConfig:
    save_model: bool
    save_model_path: str = "trained_models/"
    filename: str = "model-{epoch:02d}"
    mode: str = "max"
    monitor: str = "val_avg_k_acc"
    save_top_k: int = 1
    every_n_epoch: int = 1


@dataclass
class LoggerConfig:
    logger_type: Literal["TensorBoardLogger", "WandbLogger"]
    logger_args: dict[str, Any]


@dataclass
class TrainerConfig:
    max_epochs: int
    save_model_config: SaveModelConfig
    logger_config: LoggerConfig
    precision: str = "bf16-mixed"
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.0
    accelerator: Literal["gpu", "auto"] = "auto"
    devices: list[int] | None = None
    strategy: Literal["ddp"] | None = None
    enable_progress_bar: bool = True
