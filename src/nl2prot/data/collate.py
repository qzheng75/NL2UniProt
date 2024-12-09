from __future__ import annotations

import os
from typing import Any

from nl2prot.template.module_configs import DataloaderConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def dual_encoder_collate_fn(
    batch: list[tuple[str, str, list[str]]],
    seq_tokenizer_args: dict[str, Any],
    desc_tokenizer_args: dict[str, Any],
):
    desc_tokenizer = AutoTokenizer.from_pretrained(
        cache_dir=os.environ["MODEL_CACHE"], **desc_tokenizer_args
    )
    seq_tokenizer = AutoTokenizer.from_pretrained(
        cache_dir=os.environ["MODEL_CACHE"], **seq_tokenizer_args
    )
    names, sequences, descriptions_list = zip(*batch)

    selected_descriptions = descriptions_list

    # Tokenize sequences
    sequence_tokens = seq_tokenizer(
        list(sequences),
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )

    # Tokenize descriptions
    description_tokens = desc_tokenizer(
        selected_descriptions,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    return {
        "names": names,
        "sequences": sequence_tokens,
        "descriptions": description_tokens,
    }


def get_dataloader(dataset: Dataset, config: DataloaderConfig):
    batch_size = config.batch_size
    shuffle = config.split == "train"

    if config.collate_fn == "default":
        collate_fn = None
    else:
        fn = globals()[config.collate_fn]
        collate_fn = lambda batch: fn(batch, **config.collate_args)

    sampler = None
    if config.sampler_type == "RandomSampler":
        pass
    # else:
    #     raise ValueError(f"Sampler type {config['sampler_type']} not supported")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        sampler=sampler,
    )
