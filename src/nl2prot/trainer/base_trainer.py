from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import torch
from nl2prot.models.base_model import BaseModel
from nl2prot.modules.evaluator import Evaluator
from nl2prot.modules.misc import Logger
from nl2prot.modules.scheduler import LRScheduler
from nl2prot.template.module_configs import LoggerConfig, SaveModelConfig, TrainerConfig
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer(ABC):
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
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.save_model_config = save_model_config

        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_folder = current_time + "-" + self.save_model_config.identifier
        self.save_dir = os.path.join(self.save_model_config.save_model_dir, save_folder)
        self.best_metric = (
            float("inf") if self.save_model_config.mode == "min" else float("-inf")
        )

        self.logger_config = logger_config
        self.logger = Logger(logger_config)

        self.evaluator = evaluator
        self.trainer_config = trainer_config

        self.metrics: dict[str, list[float]] = {}
        self.init_metrics()

        self.lr_scheduler = lr_scheduler
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)

        self.grad_scaler = GradScaler(
            enabled=trainer_config.use_amp and torch.cuda.is_available()
        )
        self.curr_epoch = 0
        self.resume_from_checkpoint()
        self.epochs_to_train = trainer_config.max_epochs - self.curr_epoch

    def save_state(self) -> None:
        if self.save_model_config is None or not self.save_model_config.save_model:
            return

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        state = {
            "epoch": self.curr_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "best_metric": self.best_metric,
        }
        if self.lr_scheduler is not None:
            state["scheduler"] = self.lr_scheduler.state_dict()

        # Save checkpoint.pt
        if self.curr_epoch % self.save_model_config.every_n_epoch == 0:
            logging.info(f"Saving model at epoch {self.curr_epoch}")
            save_path = os.path.join(self.save_dir, "state.pt")
            torch.save(state, save_path)

        # Save best model
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

    def resume_from_checkpoint(self):
        if self.trainer_config.resume_from_checkpoint is None:
            return

        checkpoint = torch.load(
            self.trainer_config.resume_from_checkpoint, map_location=self.device
        )
        self.curr_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.metrics = checkpoint["metrics"]
        self.best_metric = checkpoint["best_metric"]

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        logging.info(f"Resuming training from epoch {self.curr_epoch}")

    @abstractmethod
    def init_metrics(self) -> None:
        pass

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        pass

    def before_train_epoch(self) -> None:
        pass

    def after_train_epoch(self) -> None:
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        pass

    def before_val_epoch(self) -> None:
        pass

    def after_val_epoch(self) -> None:
        pass

    def __backward(self, loss: Tensor) -> float:
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        if self.trainer_config.gradient_clip_val > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.trainer_config.gradient_clip_val
            ).item()
        else:
            grad_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.grad.detach(), 2)
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                ),
                2,
            ).item()

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return grad_norm

    def __batch_to_device(self, batch: Any) -> dict:
        if isinstance(batch, dict):
            for key in batch.keys():
                try:
                    batch[key] = batch[key].to(self.device)
                except AttributeError:
                    pass
        elif isinstance(batch, Tensor):
            batch = batch.to(self.device)
        else:
            try:
                batch = batch.to(self.device)
            except AttributeError:
                raise ValueError("Batch must be a dict, Tensor or a subclass of Tensor")
        return batch

    def __scheduler_step(self, val_metric: float) -> None:
        if self.lr_scheduler is not None:
            if self.lr_scheduler.scheduler_type == "ReduceLROnPlateau":
                self.lr_scheduler.step(epoch=self.curr_epoch, metrics=val_metric)
            else:
                self.lr_scheduler.step(epoch=self.curr_epoch)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        logging.info(
            f"Starting training {self.epochs_to_train}\
            epochs with {self.model.__class__.__name__} model."
        )

        while self.curr_epoch < self.trainer_config.max_epochs:
            self.model.train()
            self.before_train_epoch()

            epoch_train_total_loss = 0.0
            pbar = tqdm(
                train_loader, disable=not self.trainer_config.enable_progress_bar
            )
            for batch_idx, batch in enumerate(pbar):
                batch = self.__batch_to_device(batch)

                with autocast(
                    enabled=self.trainer_config.use_amp and torch.cuda.is_available(),
                    device_type=self.device,
                ):
                    loss = self.training_step(batch, batch_idx)

                grad_norm = self.__backward(loss)
                pbar.set_description(
                    f"Batch {batch_idx} | Loss: {loss.item():.4f}\
                    | Grad Norm: {grad_norm:.4f}"
                )
                epoch_train_total_loss += loss.item()

            self.after_train_epoch()
            epoch_loss = epoch_train_total_loss / len(train_loader)
            self.logger.log(epoch_loss, "train/loss", self.curr_epoch, commit=False)

            self.metrics["train/loss"].append(epoch_loss)

            if val_loader is not None:
                self.validate(val_loader)
                monitor_metric = self.metrics[self.save_model_config.monitor][-1]
                self.__scheduler_step(monitor_metric)

            # Flush all logs
            self.logger.log(None, None, self.curr_epoch, True)

            self.save_state()
            self.curr_epoch += 1

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        self.before_val_epoch()

        epoch_val_total_loss = 0.0
        pbar = tqdm(loader, disable=not self.trainer_config.enable_progress_bar)
        for batch_idx, batch in enumerate(pbar):
            batch = self.__batch_to_device(batch)
            with autocast(
                enabled=self.trainer_config.use_amp and torch.cuda.is_available(),
                device_type=self.device,
            ):
                val_loss = self.validation_step(batch, batch_idx)

            pbar.set_description(f"Batch {batch_idx} | Loss: {val_loss.item():.4f}")
            epoch_val_total_loss += val_loss.item()

        self.after_val_epoch()
        val_loss = epoch_val_total_loss / len(loader)
        self.logger.log(val_loss, "val/epoch_loss", self.curr_epoch, commit=False)

        if "val/loss" in self.metrics:
            self.metrics["val/loss"].append(val_loss)
