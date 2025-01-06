from __future__ import annotations

import logging
import os
from typing import override

import torch
from nl2prot.models.base_model import BaseDualEncoder, BaseModel
from nl2prot.modules.evaluator import Evaluator
from nl2prot.modules.misc import Logger
from nl2prot.modules.scheduler import LRScheduler
from nl2prot.template.module_configs import SaveModelConfig, TrainerConfig
from nl2prot.trainer.clip_trainer import CLIPTrainer
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


class DDPCLIPTrainer(CLIPTrainer):
    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        loss: nn.Module,
        trainer_config: TrainerConfig,
        save_model_config: SaveModelConfig,
        logger: Logger,
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
            logger=logger,
            lr_scheduler=lr_scheduler,
            device=device,
        )
        self.device = torch.cuda.current_device()
        self.model = DistributedDataParallel(
            self.model.to(device), device_ids=[device], find_unused_parameters=True
        )

    @override
    def check_model(self) -> None:
        assert isinstance(
            self.model, DistributedDataParallel
        ), "Model must be ddp for ddp training"
        assert isinstance(
            self.model.module, BaseDualEncoder
        ), "Model must be a dual encoder model"

    @override
    def save_state(self) -> None:
        if (
            self.save_model_config is None
            or not self.save_model_config.save_model
            or self.device != 0
        ):
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        state = {
            "epoch": self.curr_epoch,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "best_metric": self.best_metric,
            "global_training_step": self.global_training_step,
        }
        if self.lr_scheduler is not None:
            state["scheduler"] = self.lr_scheduler.state_dict()

        if self.curr_epoch % self.save_model_config.every_n_epoch == 0:
            logging.info(f"Saving model at epoch {self.curr_epoch}")
            save_path = os.path.join(self.save_dir, "state.pt")
            torch.save(state, save_path)

        monitor_metric = self.metrics[self.save_model_config.monitor][-1]
        if self.save_model_config.mode == "max":
            if monitor_metric > self.best_metric:
                self.best_metric = monitor_metric
            else:
                return
        elif self.save_model_config.mode == "min":
            if monitor_metric < self.best_metric:
                self.best_metric = monitor_metric
            else:
                return

        logging.info(f"Saving best model at epoch {self.curr_epoch}")
        state["metrics"] = self.metrics
        save_path = os.path.join(self.save_dir, "best_state.pt")
        torch.save(state, save_path)
