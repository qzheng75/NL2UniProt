from __future__ import annotations

import warnings
from typing import override

import numpy as np
from nl2prot.models.base_model import BaseDualEncoder, BaseModel
from nl2prot.modules.evaluator import Evaluator
from nl2prot.modules.scheduler import LRScheduler
from nl2prot.template.module_configs import LoggerConfig, SaveModelConfig, TrainerConfig
from nl2prot.trainer.base_trainer import BaseTrainer
from torch import nn
from torch.optim import Optimizer


class CLIPTrainer(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        loss: nn.Module,
        trainer_config: TrainerConfig,
        save_model_config: SaveModelConfig,
        logger_config: LoggerConfig,
        evaluator: Evaluator | None = None,
        lr_scheduler: LRScheduler | None = None,
        device: str | None = None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss=loss,
            trainer_config=trainer_config,
            evaluator=evaluator,
            save_model_config=save_model_config,
            logger_config=logger_config,
            lr_scheduler=lr_scheduler,
            device=device,
        )

        # Use to compute metric at validation
        self.name2idx_dict = {}
        self.names = []
        self.seq_embs = []

        # Use at validation time
        self.ground_truth = []
        self.desc_embs = []

    @override
    def init_metrics(self):
        if self.evaluator is not None:
            for metric in self.evaluator.metrics:
                self.metrics[f"val/{metric}"] = []
        self.metrics["train/loss"] = []
        self.metrics["val/loss"] = []

    @override
    def training_step(self, batch, batch_idx):
        assert isinstance(
            self.model, BaseDualEncoder
        ), "Model must be a dual encoder model"
        if self.evaluator is not None:
            out = self.model(batch, return_embeddings=True)
            names, desc_emb, seq_emb = (
                out["names"],
                out["desc_embeddings"],
                out["prot_embeddings"],
            )
            similarity = self.model.compute_similarity(desc_emb, seq_emb)
            loss = self.loss(similarity)

            seq_emb = seq_emb.detach().cpu().numpy()
            for i in range(len(names)):
                if names[i] in self.name2idx_dict:
                    continue
                name = names[i]
                self.name2idx_dict[name] = len(self.names)
                self.names.append(name)
                self.seq_embs.append(seq_emb[i])
        else:
            out = self.model(batch, return_embeddings=False)
            loss = self.loss(out)
        return loss

    @override
    def before_train_epoch(self):
        self.name2idx_dict = {}
        self.names = []
        self.seq_embs = []
        self.ground_truth = []
        self.desc_embs = []

    @override
    def after_train_epoch(self):
        for param_group in self.optimizer.param_groups:
            self.logger.log(
                param_group["lr"],
                f'train/lr_{param_group["name"]}',
                self.curr_epoch,
                commit=False,
            )
        self.logger.log(self.curr_epoch, "epoch", self.curr_epoch, commit=False)

    @override
    def validation_step(self, batch, batch_idx):
        assert isinstance(
            self.model, BaseDualEncoder
        ), "Model must be a dual encoder model"
        if self.evaluator is not None:
            out = self.model(batch, return_embeddings=True)
            names, desc_emb, seq_emb = (
                out["names"],
                out["desc_embeddings"],
                out["prot_embeddings"],
            )
            similarity = self.model.compute_similarity(desc_emb, seq_emb)
            loss = self.loss(similarity)

            desc_to_validate, idx_to_validate = [], []

            for i in range(len(names)):
                if names[i] not in self.name2idx_dict:
                    continue
                desc_to_validate.append(desc_emb[i].detach().cpu().numpy())
                idx_to_validate.append(self.name2idx_dict[names[i]])

            self.ground_truth.extend(idx_to_validate)
            self.desc_embs.extend(desc_to_validate)
        else:
            out = self.model(batch, return_embeddings=False)
            loss = self.loss(out)

        return loss

    @override
    def after_val_epoch(self):
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
            key = f"val/{key}"
            self.metrics[key].append(value)
            self.logger.log(value, key, self.curr_epoch, commit=False)
