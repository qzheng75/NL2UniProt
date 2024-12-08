from __future__ import annotations

import importlib

from nl2prot.models.base_model import BaseModel
from nl2prot.modules import loss
from nl2prot.modules.evaluator import Evaluator
from nl2prot.template.module_configs import (
    LossConfig,
    MetricConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from torch import nn, optim


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
        optim_groups.append({"params": params, **config_for_group})

    return optimizer_cls(optim_groups)


def load_scheduler(
    config: SchedulerConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler._LRScheduler:
    scheduler_type = config.scheduler_type
    scheduler_args = config.scheduler_args
    scheduler_cls = getattr(optim.lr_scheduler, scheduler_type)

    return scheduler_cls(optimizer, **scheduler_args)
