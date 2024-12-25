from __future__ import annotations

import importlib
import warnings
from typing import Any, Literal

import yaml
from nl2prot.data import dataset
from nl2prot.data.collate import get_dataloader
from nl2prot.data.utils import split_dataset
from nl2prot.models.base_model import BaseModel
from nl2prot.modules import loss
from nl2prot.modules.evaluator import Evaluator
from nl2prot.modules.scheduler import LRScheduler
from nl2prot.template.module_configs import (
    DataloaderConfig,
    DatasetConfig,
    LoggerConfig,
    LossConfig,
    MetricConfig,
    ModelConfig,
    OptimizerConfig,
    SaveModelConfig,
    SchedulerConfig,
    TrainerConfig,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def load_loss(config: LossConfig) -> nn.Module:
    loss_type = config.loss_type
    loss_args = config.loss_args
    return getattr(loss, loss_type)(**loss_args)


def load_evaluator(config: MetricConfig) -> Evaluator:
    metric_type = config.metric_type
    metric_args = config.metric_args
    return Evaluator(metric=metric_type, **metric_args)


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


def load_everything(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loss_config = LossConfig(**config["loss"])
    loss = load_loss(loss_config)

    model_config = ModelConfig(
        model_type=config["model"]["model_type"],
        model_args=config["model"]["model_args"],
    )
    model = load_model(model_config)

    optimizer_config = OptimizerConfig(**config["optimizer"])
    optimizer = load_optimizer(optimizer_config, model)

    if "scheduler" in config:
        scheduler_config = SchedulerConfig(**config["scheduler"])
        scheduler = load_scheduler(scheduler_config, optimizer)
    else:
        scheduler = None
        warnings.warn("No scheduler provided in config", UserWarning)

    if "evaluator" in config:
        evaluator_config = MetricConfig(**config["evaluator"])
        evaluator = load_evaluator(evaluator_config)
    else:
        evaluator = None
        warnings.warn("No evaluator provided in config", UserWarning)

    dataset_config = DatasetConfig(**config["dataset"])
    datasets = load_dataset(dataset_config)

    dataloader_configs: dict[Literal["train", "val", "test"], DataloaderConfig] = {
        k: DataloaderConfig(split=k, **config["dataloader"]) for k in datasets.keys()
    }
    dataloaders = {
        k: load_dataloader(v, datasets[k]) for k, v in dataloader_configs.items()
    }

    if "logger" in config:
        logger_config = LoggerConfig(**config["logger"])
    else:
        logger_config = LoggerConfig(logger_type="stdout", logger_args={})
        warnings.warn("No logger provided in config. Default to stdout.", UserWarning)

    if "save_model" in config:
        save_model_config = SaveModelConfig(**config["save_model"])
    else:
        save_model_config = SaveModelConfig(save_model=False)
        warnings.warn(
            "No save_model provided in config.\
            Default not to save model.",
            UserWarning,
        )

    trainer_type = config["trainer"]["trainer_type"]
    trainer_config = TrainerConfig(
        logger_config=logger_config,
        save_model_config=save_model_config,
        **config["trainer"]["trainer_args"],
    )
    module_name, class_name = trainer_type.rsplit(".", 1)
    module = importlib.import_module(f"nl2prot.trainer.{module_name}")
    trainer_cls = getattr(module, class_name)
    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        loss=loss,
        trainer_config=trainer_config,
        evaluator=evaluator,
        save_model_config=save_model_config,
        logger_config=logger_config,
        lr_scheduler=scheduler,
    )

    return {
        "trainer": trainer,
        **dataloaders,
    }
