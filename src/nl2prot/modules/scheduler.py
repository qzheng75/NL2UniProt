from __future__ import annotations

from typing import override

import torch
from nl2prot.template.module_configs import SchedulerConfig
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(
        self, optimizer: torch.optim.Optimizer, scheduler_config: SchedulerConfig
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_config.scheduler_type

        self.scheduler: _LRScheduler = getattr(
            torch.optim.lr_scheduler, self.scheduler_type
        )(optimizer, **scheduler_config.scheduler_args)

        self.lr: dict[str, float] = {
            group["name"]: group["lr"] for group in optimizer.param_groups
        }

    @override
    def step(self, epoch=None, metrics=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

        # update the learning rate attribute to current lr
        self.update_lr()

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            self.lr[param_group["name"]] = param_group["lr"]
