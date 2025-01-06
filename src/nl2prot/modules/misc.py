from __future__ import annotations

import logging
import warnings
from typing import Any

from nl2prot.template.module_configs import LoggerConfig
from torch.utils.tensorboard import SummaryWriter

import wandb


class Logger:
    def __init__(self, logger_config: LoggerConfig | None = None):
        self.logger_config = logger_config
        self._load_logger()

    def _load_logger(self):
        if self.logger_config is not None:
            logger_type = self.logger_config.logger_type
            logger_args = self.logger_config.logger_args

            if logger_type == "tensorboard":
                # Personally never used. May contain bugs
                self.logger = SummaryWriter(**logger_args)
            elif logger_type == "wandb":
                # print("Init wandb")
                wandb.init(**logger_args)
            elif logger_type == "stdout":
                logging.basicConfig(level=logging.INFO)
            else:
                raise ValueError(f"Invalid logger type: {logger_type}")
        else:
            warnings.warn(
                "No logger configuration found.\
                No logs will be saved.",
                UserWarning,
            )

    def log(
        self,
        metrics: dict[str, Any] | None,
        step: int,
        commit: bool = False,
    ) -> None:
        if self.logger_config is None:
            return
        if self.logger_config.logger_type == "tensorboard":
            # TODO: Implement tensorboard logging
            raise NotImplementedError("Tensorboard logging not implemented")
        elif self.logger_config.logger_type == "wandb":
            if metrics is None and commit:
                wandb.log(data={}, step=step, commit=True)
            else:
                assert metrics is not None
                wandb.log(data=metrics, step=step, commit=commit)
        elif self.logger_config.logger_type == "stdout":
            if metrics is None:
                return
            for metric_name, metric in metrics.items():
                logging.info(f"Step: {step} | {metric_name}: {metric}")
        else:
            raise ValueError(f"Invalid logger type: {self.logger_config.logger_type}")
