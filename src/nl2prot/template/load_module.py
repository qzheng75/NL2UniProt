from __future__ import annotations

import importlib
from typing import Any, Literal

import pytorch_lightning as pl
import yaml
from nl2prot.data import dataset
from nl2prot.data.collate import get_dataloader
from nl2prot.data.utils import split_dataset
from nl2prot.models.base_model import BaseModel
from nl2prot.modules import loss
from nl2prot.modules.scheduler import LRScheduler
from nl2prot.template.module_configs import (
    DataloaderConfig,
    DatasetConfig,
    LoggerConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    SaveModelConfig,
    SchedulerConfig,
    TrainerConfig,
)
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def load_loss(config: LossConfig) -> nn.Module:
    loss_type = config.loss_type
    loss_args = config.loss_args
    return getattr(loss, loss_type)(**loss_args)


# def load_evaluator(config: MetricConfig) -> Evaluator:
#     metric_type = config.metric_type
#     metric_args = config.metric_args
#     return Evaluator(metric=metric_type, **metric_args)


def load_model(config: ModelConfig) -> BaseModel:
    model_type = config.model_type
    module_name, class_name = model_type.rsplit(".", 1)
    module = importlib.import_module(f"nl2prot.models.{module_name}")
    model_class = getattr(module, class_name)

    model_args = config.model_args
    return model_class(**model_args)


def load_optimizer(config: OptimizerConfig, model: BaseModel) -> optim.Optimizer:
    optimizer_type = config.optimizer_type
    optimizer_args = config.optimizer_args
    optimizer_cls = getattr(optim, optimizer_type)

    param_groups = model.trainable_parameters()
    optim_groups = []

    for group_name, params in param_groups.items():
        config_for_group = optimizer_args.get(group_name, {})
        assert "lr" in config_for_group, f"Learning rate not provided for {group_name}"
        optim_groups.append({"name": group_name, "params": params, **config_for_group})

    return optimizer_cls(optim_groups)


def load_scheduler(config: SchedulerConfig, optimizer: optim.Optimizer) -> LRScheduler:
    return LRScheduler(optimizer, config)


def load_dataset(
    config: DatasetConfig,
) -> dict[Literal["train", "val", "test"], Dataset]:
    dataset_type = config.dataset_type
    dataset_cls = getattr(dataset, dataset_type)

    dataset_args = config.dataset_args
    datasets: dict[Literal["train", "val", "test"], Dataset] = {}
    common_args = dataset_args.pop("common", {})

    for key, args in dataset_args.items():
        if key in ("train", "val", "test"):
            use_ratio = args.pop("use_ratio", None)
            ds: Dataset = dataset_cls(**args, **common_args)
            if use_ratio:
                ds, _, _ = split_dataset(
                    ds, train_size=use_ratio, valid_size=0.0, test_size=0.0
                )
            datasets[key] = ds

    return datasets


def load_dataloader(config: DataloaderConfig, dataset: Dataset) -> DataLoader:
    return get_dataloader(dataset, config)


def load_pl_logger(config: LoggerConfig) -> loggers.Logger:
    logger_cls = getattr(loggers, config.logger_type)
    return logger_cls(**config.logger_args)


def load_pl_model_save(config: SaveModelConfig) -> ModelCheckpoint | None:
    if not config.save_model:
        return None

    return ModelCheckpoint(
        dirpath=config.save_model_path,
        filename=config.filename,
        monitor=config.monitor,
        mode=config.mode,
        save_top_k=config.save_top_k,
        every_n_epochs=config.every_n_epoch,
    )


def load_trainer(config: TrainerConfig) -> pl.Trainer:
    if config.logger_config is not None:
        logger = load_pl_logger(config.logger_config)
    else:
        logger = None

    if config.save_model_config is not None:
        model_save = load_pl_model_save(config.save_model_config)
    else:
        model_save = None

    devices = config.devices if config.devices is not None else -1
    callbacks = [model_save] if model_save else None

    return pl.Trainer(
        max_epochs=config.max_epochs,
        logger=False if logger is None else logger,
        callbacks=callbacks,  # type: ignore
        precision=config.precision,  # type: ignore
        log_every_n_steps=config.log_every_n_steps,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        accelerator=config.accelerator,
        devices=devices,
        strategy=str(config.strategy),
        enable_progress_bar=config.enable_progress_bar,
    )


def load_everything(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loss_config = LossConfig(**config["loss"])
    loss = load_loss(loss_config)

    model_config = ModelConfig(**config["model"])
    model = load_model(model_config)

    optimizer_config = OptimizerConfig(**config["optimizer"])
    optimizer = load_optimizer(optimizer_config, model)

    scheduler_config = SchedulerConfig(**config["scheduler"])
    scheduler = load_scheduler(scheduler_config, optimizer)

    dataset_config = DatasetConfig(**config["dataset"])
    datasets = load_dataset(dataset_config)

    dataloader_configs: dict[Literal["train", "val", "test"], DataloaderConfig] = {
        k: DataloaderConfig(split=k, **config["dataloader"]) for k in datasets.keys()
    }
    dataloaders = {
        k: load_dataloader(v, datasets[k]) for k, v in dataloader_configs.items()
    }

    pl_module = config["pl_module"]
    if pl_module == "DualEncoder":
        from nl2prot.models.base_model import BaseDualEncoder
        from nl2prot.trainer.dual_encoder_pl import DualEncoderPl

        assert isinstance(
            model, BaseDualEncoder
        ), "Model must be a Dual Encoder model to use DualEncoderPl"

        pl_module = DualEncoderPl(
            model=model, loss_fn=loss, optimizer=optimizer, scheduler=scheduler
        )
    else:
        raise ValueError(f"Unknown pl_module: {pl_module}")

    if "logger" in config:
        logger_config = LoggerConfig(**config["logger"])
    else:
        logger_config = None

    if "save_model" in config:
        save_model_config = SaveModelConfig(**config["save_model"])
    else:
        save_model_config = None

    trainer_config = TrainerConfig(
        logger_config=logger_config,
        save_model_config=save_model_config,
        **config["trainer"],
    )

    trainer = load_trainer(trainer_config)

    return {
        "pl_module": pl_module,
        "checkpoint": trainer_config.resume_from_checkpoint,
        "trainer": trainer,
        **dataloaders,
    }
