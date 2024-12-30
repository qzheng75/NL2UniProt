from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import Any

import torch
from nl2prot.models.base_model import print_model_parameters
from nl2prot.modules.evaluator import Evaluator
from nl2prot.modules.misc import Logger
from nl2prot.modules.scheduler import LRScheduler
from nl2prot.template.module_configs import SaveModelConfig, TrainerConfig
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss: nn.Module,
        trainer_config: TrainerConfig,
        save_model_config: SaveModelConfig,
        logger: Logger,
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

        self.log_per = trainer_config.log_every_n_steps
        self.logger = logger

        self.evaluator = evaluator
        self.trainer_config = trainer_config

        self.metrics: dict[str, list[float]] = {}
        self.init_metrics()

        self.global_training_step = 0

        self.lr_scheduler = lr_scheduler
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logging.info(f"Using {self.device} as device.")
        self.model = self.model.to(self.device)

        self.grad_scaler = GradScaler(
            enabled=trainer_config.use_amp and torch.cuda.is_available()
        )
        self.curr_epoch = 0
        self.resume_from_checkpoint(self.trainer_config.resume_from_checkpoint)
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
            "global_training_step": self.global_training_step,
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

    def resume_from_checkpoint(self, checkpoint_path: str | None = None) -> None:
        if checkpoint_path is None:
            return

        device = self.device if isinstance(self.device, str) else f"cuda:{self.device}"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.curr_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.metrics = checkpoint["metrics"]
        self.best_metric = checkpoint["best_metric"]

        try:
            self.global_training_step = checkpoint["global_training_step"]
        except KeyError:
            self.global_training_step = 0
            warnings.warn(
                "Global training step not found in checkpoint. "
                + "Maybe you're using an older version of this codebase. "
                + "Report this issue if you're using the latest version."
            )

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])

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

    @abstractmethod
    def predict_step(self, batch, batch_idx, **kwargs) -> Any:
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
                self.lr_scheduler.step(
                    epoch=self.global_training_step, metrics=val_metric
                )
            else:
                self.lr_scheduler.step(epoch=self.global_training_step)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        print_model_parameters(self.model)
        if self.global_training_step != 0:
            logging.info(
                f"Already training {self.global_training_step} steps. "
                + f"Resuming training from epoch {self.epochs_to_train} "
                + f"epochs with {self.model.__class__.__name__} model."
            )
        else:
            logging.info(
                f"Starting training {self.epochs_to_train} "
                + f"epochs with {self.model.__class__.__name__} model."
            )

        while self.curr_epoch < self.trainer_config.max_epochs:
            epoch_start_time = time()
            self.model.train()
            self.before_train_epoch()

            # DDP
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(self.curr_epoch)

            train_loader_iter = iter(train_loader)
            pbar = tqdm(
                range(0, len(train_loader_iter)),
                disable=not self.trainer_config.enable_progress_bar,
            )

            epoch_train_total_loss = 0.0
            accumulate_train_loss = 0.0
            accumulate_grad_norm = 0.0

            for batch_idx in pbar:
                batch = next(train_loader_iter)
                device = (
                    self.device
                    if isinstance(self.device, str)
                    else f"cuda:{self.device}"
                )

                with autocast(
                    enabled=self.trainer_config.use_amp and torch.cuda.is_available(),
                    device_type=device,
                ):
                    batch = self.__batch_to_device(batch)
                    loss = self.training_step(batch, batch_idx)

                grad_norm = self.__backward(loss)
                pbar.set_description(
                    f"Batch {batch_idx} | Loss: {loss.item():.4f}"
                    + f", Grad Norm: {grad_norm:.4f}"
                )
                epoch_train_total_loss += loss.item()
                accumulate_train_loss += loss.item()
                accumulate_grad_norm += grad_norm

                if (self.global_training_step + 1) % self.log_per == 0:
                    step_loss = accumulate_train_loss / self.log_per
                    step_grad_norm = accumulate_grad_norm / self.log_per
                    self.logger.log(
                        {
                            "train/step_loss": step_loss,
                            "train/step_grad_norm": step_grad_norm,
                        },
                        self.global_training_step,
                    )
                    accumulate_train_loss, accumulate_grad_norm = 0.0, 0.0

                    if batch_idx != len(train_loader) - 1:
                        # Flush logs
                        self.logger.log(None, self.global_training_step, True)

                self.global_training_step += 1

            self.after_train_epoch()
            epoch_loss = epoch_train_total_loss / len(train_loader)

            # Final training log, flush
            self.logger.log({"train/epoch_loss": epoch_loss}, self.global_training_step)
            self.metrics["train/epoch_loss"].append(epoch_loss)

            if val_loader is not None:
                self.validate(val_loader)
                monitor_metric = self.metrics[self.save_model_config.monitor][-1]
                self.__scheduler_step(monitor_metric)

            self.save_state()
            epoch_end_time = time()
            self.logger.log(
                {"epoch_time": epoch_end_time - epoch_start_time},
                self.global_training_step,
                commit=True,
            )
            self.curr_epoch += 1

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        self.before_val_epoch()

        epoch_val_total_loss = 0.0

        pbar = tqdm(loader, disable=not self.trainer_config.enable_progress_bar)
        device = self.device if isinstance(self.device, str) else f"cuda:{self.device}"

        for batch_idx, batch in enumerate(pbar):
            batch = self.__batch_to_device(batch)
            with autocast(
                enabled=self.trainer_config.use_amp and torch.cuda.is_available(),
                device_type=device,
            ):
                val_loss = self.validation_step(batch, batch_idx)

            pbar.set_description(f"Batch {batch_idx} | Loss: {val_loss.item():.4f}")
            epoch_val_total_loss += val_loss.item()

        self.after_val_epoch()
        val_loss = epoch_val_total_loss / len(loader)

        # Final log, flush
        self.logger.log({"val/epoch_loss": val_loss}, self.global_training_step)
        self.metrics["val/epoch_loss"].append(val_loss)
