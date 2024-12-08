from __future__ import annotations

import torch
from nl2prot.template.module_configs import SchedulerConfig
from torch import nn


class LRScheduler:
    """wrapper around torch.optim.lr_scheduler._LRScheduler"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        model_parameters: dict[str, nn.Parameter],
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)(
            optimizer, **model_parameters
        )

        self.lr = self.optimizer.param_groups[0]["lr"]

    @classmethod
    def from_config(
        cls, optimizer: torch.optim.Optimizer, optim_config: SchedulerConfig
    ) -> LRScheduler:
        scheduler_type = optim_config.scheduler_type
        scheduler_args = optim_config.scheduler_args
        return cls(optimizer, scheduler_type, **scheduler_args)

    def step(self, metrics=None, epoch=None):
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
            self.lr = param_group["lr"]
