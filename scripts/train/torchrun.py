from __future__ import annotations

import argparse
import datetime
import logging
import os
import warnings
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from nl2prot.modules.misc import Logger, LoggerConfig
from nl2prot.template.load_module import load_everything
from nl2prot.trainer.ddp_clip_trainer import DDPCLIPTrainer
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=30))


def get_logger(config: dict[str, Any]) -> Logger | None:
    if os.environ["RANK"] == "0":
        if "logger" in config:
            logger_config = LoggerConfig(**config["logger"])
            logger = Logger(logger_config)
        else:
            logger_config = LoggerConfig(logger_type="stdout", logger_args={})
            logger = Logger(logger_config)
            warnings.warn("No logger provided in config. Default to stdout.")
    else:
        logger = Logger(None)

    return logger


def main(args):
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    ddp_setup()

    status = load_dotenv()
    assert status, "Failed to load .env file"

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = get_logger(config)
    out = load_everything(config=config, logger=logger)
    logging.info("Successfully loaded all modules!")

    assert isinstance(out["trainer"], DDPCLIPTrainer), "Trainer must be DDPCLIPTrainer"
    trainer: DDPCLIPTrainer = out["trainer"]
    train_dataloader: DataLoader = out["train"]
    val_dataloader: DataLoader = out["val"]
    trainer.train(train_dataloader, val_dataloader)

    destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args)
