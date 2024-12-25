from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv
from nl2prot.template.load_module import load_everything
from nl2prot.trainer.base_trainer import BaseTrainer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()


def main(args):
    status = load_dotenv()
    assert status, "Failed to load .env file"
    out = load_everything(args.config_path)
    logging.info("Successfully loaded all modules!")

    trainer: BaseTrainer = out["trainer"]
    train_dataloader: DataLoader = out["train"]
    val_dataloader: DataLoader = out["val"]

    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
