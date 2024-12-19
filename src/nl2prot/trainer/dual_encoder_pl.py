from __future__ import annotations

import warnings
from typing import Any, override

import numpy as np
import pytorch_lightning as pl
import torch
from nl2prot.models.base_model import BaseDualEncoder
from nl2prot.modules.evaluator import Evaluator
from nl2prot.modules.scheduler import LRScheduler
from torch import nn


class DualEncoderPl(pl.LightningModule):
    def __init__(
        self,
        model: BaseDualEncoder,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None = None,
        evaluator: Evaluator | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # Define your model architecture here
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator

        # Use to compute metric at validation
        self.name2idx_dict = {}
        self.names = []
        self.seq_embs = []

        # Use at validation time
        self.ground_truth = []
        self.desc_embs = []

    @override
    def forward(self, x, return_embeddings=False, **kwargs):
        return self.model(x, return_embeddings=return_embeddings, **kwargs)

    @override
    def training_step(self, batch, batch_idx):
        if self.evaluator is not None:
            out = self(batch, return_embeddings=True)
            names, desc_emb, seq_emb = (
                out["names"],
                out["desc_embeddings"],
                out["prot_embeddings"],
            )
            similarity = self.model.compute_similarity(desc_emb, seq_emb)
            loss = self.loss_fn(similarity)

            seq_emb = seq_emb.detach().cpu().numpy()
            for i in range(len(names)):
                if names[i] in self.name2idx_dict:
                    continue
                name = names[i]
                self.name2idx_dict[name] = len(self.names)
                self.names.append(name)
                self.seq_embs.append(seq_emb[i])
        else:
            out = self(batch, return_embeddings=False)
            loss = self.loss_fn(out)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @override
    def on_train_epoch_start(self):
        self.name2idx_dict = {}
        self.names = []
        self.seq_embs = []
        self.ground_truth = []
        self.desc_embs = []

    @override
    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(
                f'train/lr_{param_group["name"]}',
                param_group["lr"],
                on_epoch=True,
            )
        self.log("epoch", self.current_epoch)

    @override
    def validation_step(self, batch, batch_idx):
        if self.evaluator is not None:
            out = self(batch, return_embeddings=True)
            names, desc_emb, seq_emb = (
                out["names"],
                out["desc_embeddings"],
                out["prot_embeddings"],
            )
            similarity = self.model.compute_similarity(desc_emb, seq_emb)
            loss = self.loss_fn(similarity)

            desc_to_validate, idx_to_validate = [], []

            for i in range(len(names)):
                if names[i] not in self.name2idx_dict:
                    continue
                desc_to_validate.append(desc_emb[i].detach().cpu().numpy())
                idx_to_validate.append(self.name2idx_dict[names[i]])

            self.ground_truth.extend(idx_to_validate)
            self.desc_embs.extend(desc_to_validate)
        else:
            out = self(batch, return_embeddings=False)
            loss = self.loss_fn(out)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @override
    def on_validation_epoch_end(self):
        if self.evaluator is None:
            return
        if len(self.seq_embs) == 0 or len(self.desc_embs) == 0:
            warnings.warn("No data to evaluate. Skipped evaluation.")
            return

        desc_embs = np.array(self.desc_embs)
        ground_truth = np.array(self.ground_truth)
        seq_embs = np.array(self.seq_embs)
        results = self.evaluator.evaluate(desc_embs, seq_embs, ground_truth)
        for key, value in results.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)

    @override
    def test_step(self, batch, batch_idx):
        out = self(batch, return_embeddings=False)
        loss = self.loss_fn(out)
        return loss

    @override
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        assert "batch_type" in batch.keys(), "batch_type must be present in batch"
        batch_type = batch["batch_type"]

        if batch_type == "sequence":
            assert self.model.prot_encoder is not None, "Protein encoder not set"
            encoder = self.model.prot_encoder
        else:
            assert self.model.desc_encoder is not None, "Description encoder not set"
            encoder = self.model.desc_encoder
        embeddings = encoder(**batch["tokens"])
        return {"accessions": batch["accessions"], "embeddings": embeddings}

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
