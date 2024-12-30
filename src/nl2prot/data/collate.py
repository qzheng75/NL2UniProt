from __future__ import annotations

import os
from typing import Any, Literal

from nl2prot.template.module_configs import DataloaderConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def tokenize(to_tokenize: list[str], tokenizer_init_args: dict[str, Any], max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(
        cache_dir=os.environ["MODEL_CACHE"], **tokenizer_init_args
    )
    return tokenizer(
        to_tokenize,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )


def single_encoder_collate_fn(
    batch: list[tuple[str, str]],
    tokenizer_args: dict[str, Any],
    batch_type: Literal["sequence", "description"],
):
    accessions, to_tokenize = zip(*batch)
    if batch_type == "sequence":
        tokens = tokenize(to_tokenize, tokenizer_args, max_len=1024)
    else:
        tokens = tokenize(to_tokenize, tokenizer_args, max_len=512)
    return {"batch_type": batch_type, "accessions": accessions, "tokens": tokens}


def dual_encoder_collate_fn(
    batch: list[tuple[str, str, list[str]]],
    seq_tokenizer_args: dict[str, Any],
    desc_tokenizer_args: dict[str, Any],
):
    names, sequences, descriptions_list = zip(*batch)

    sequence_tokens = tokenize(sequences, seq_tokenizer_args, max_len=1024)
    description_tokens = tokenize(descriptions_list, desc_tokenizer_args, max_len=512)

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
    elif config.sampler_type == "DistributedSampler":
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset)
        shuffle = None
    else:
        raise ValueError(f"Sampler type {config.sampler_type} not supported")

    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        sampler=sampler,
    )
