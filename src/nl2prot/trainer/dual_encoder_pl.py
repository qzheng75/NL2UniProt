from __future__ import annotations

from typing import Any, override

import pytorch_lightning as pl
import torch
from nl2prot.models.base_model import BaseDualEncoder
from nl2prot.modules.scheduler import LRScheduler
from torch import nn


class DualEncoderPl(pl.LightningModule):
    def __init__(
        self,
        model: BaseDualEncoder,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Define your model architecture here
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    @override
    def forward(self, x, return_embeddings=False, **kwargs):
        # Define forward pass
        return self.model(x, return_embeddings=return_embeddings, **kwargs)

    @override
    def training_step(self, batch, batch_idx):
        out = self(batch, return_embeddings=False)
        loss = self.loss_fn(out)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        for param_group in self.optimizer.param_groups:
            self.log(
                f'train/lr_{param_group["name"]}',
                param_group["lr"],
                on_step=True,
                on_epoch=False,
            )

        return loss

    @override
    def validation_step(self, batch, batch_idx):
        out = self(batch, return_embeddings=False)
        loss = self.loss_fn(out)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @override
    def test_step(self, batch, batch_idx):
        out = self(batch, return_embeddings=True)
        loss = self.loss_fn(out)
        return loss

    @override
    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | tuple[list[torch.optim.Optimizer], list[Any]]:
        optimizer = self.optimizer

        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler.scheduler
        if self.scheduler.scheduler_type == "ReduceLROnPlateau":
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                }
            ]
        return [optimizer], [scheduler]
