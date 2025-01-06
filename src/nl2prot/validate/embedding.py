from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scanpy as sc
import torch
from Bio import SeqIO
from nl2prot.data.collate import single_encoder_collate_fn
from nl2prot.data.dataset import DescDataset, ProtDataset
from nl2prot.template.load_module import load_everything
from nl2prot.trainer.base_trainer import BaseTrainer
from nl2prot.trainer.clip_trainer import CLIPTrainer
from tqdm import tqdm


def compute_embeddings(
    trainer: BaseTrainer,
    tokenizer_args: dict[str, Any],
    input_type: Literal["sequence", "description"],
    to_embed: list[str],
    accessions: list[str] | None = None,
    batch_size: int = 16,
    device: str = "cuda",
    enable_progress_bar: bool = True,
) -> tuple[list[str] | None, np.ndarray]:
    if input_type == "sequence":
        assert (
            accessions is not None
        ), "Accessions must be provided for sequence embeddings"
        dataset = ProtDataset(to_embed, accessions)
        collate_fn = lambda batch: single_encoder_collate_fn(
            batch, tokenizer_args, "sequence"
        )
    else:
        dataset = DescDataset(to_embed, accessions)
        collate_fn = lambda batch: single_encoder_collate_fn(
            batch, tokenizer_args, "description"
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    embeddings = []
    for idx, batch in enumerate(tqdm(dataloader, disable=not enable_progress_bar)):
        batch["tokens"] = {k: v.to(device) for k, v in batch["tokens"].items()}
        out = trainer.predict_step(batch, idx)
        embeddings.append(out["embeddings"].float().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return accessions, embeddings


def get_trainer(
    module_config_path: str,
    model_ckpt_path: str,
    quantize_model: bool = False,
    device: str = "cuda",
) -> BaseTrainer:
    assert (
        module_config_path is not None and model_ckpt_path is not None
    ), "You must provide a module config and a model checkpoint"
    all_modules = load_everything(config_path=module_config_path)
    trainer: BaseTrainer = all_modules["trainer"]

    if isinstance(trainer, CLIPTrainer):
        trainer.resume_from_checkpoint(model_ckpt_path)
    else:
        raise NotImplementedError("Only CLIPTrainer is supported for now")

    trainer.model = trainer.model.to(device)
    trainer.device = device
    if quantize_model:
        from torch.quantization import quantize_dynamic

        trainer.model = quantize_dynamic(
            trainer.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    return trainer


def embed_sequences(
    module_config_path: str,
    model_ckpt_path: str,
    tokenizer_args: dict[str, Any],
    fasta_file: str | None = None,
    raw_sequences: list[str] | None = None,
    accessions: list[str] | None = None,
    output_file: str | None = None,
    **kwargs,
) -> tuple[list[str], np.ndarray]:
    assert tokenizer_args is not None, "You must provide tokenizer arguments"
    assert output_file is None or output_file.endswith(
        ".h5ad"
    ), "Temporary output only supports h5ad format"

    trainer = get_trainer(module_config_path, model_ckpt_path)
    if raw_sequences is None:
        assert (
            fasta_file is not None
        ), "You must provide input sequences as a list or a fasta file"
        seqs: list[str] = []
        names: list[str] = []
        with open(fasta_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                names.append(str(record.id))
                seqs.append(str(record.seq))
        acc, embeddings = compute_embeddings(
            trainer, tokenizer_args, "sequence", seqs, names, **kwargs
        )
    else:
        assert (
            accessions is not None
        ), "You must provide accessions for raw sequences input type"
        acc, embeddings = compute_embeddings(
            trainer, tokenizer_args, "sequence", raw_sequences, accessions, **kwargs
        )

    assert acc is not None, "Accessions must be returned. Report this issue."
    if output_file is not None:
        adata = sc.AnnData(X=embeddings)
        adata.obs["accession"] = acc
        sc.write(output_file, adata)

    return acc, embeddings


def embed_descriptions(
    module_config_path: str,
    model_ckpt_path: str,
    tokenizer_args: dict[str, Any],
    descriptions: list[str],
    accessions: list[str] | None = None,
    **kwargs,
) -> tuple[list[str] | None, np.ndarray]:
    assert tokenizer_args is not None, "You must provide tokenizer arguments"

    trainer = get_trainer(module_config_path, model_ckpt_path)
    acc, embeddings = compute_embeddings(
        trainer, tokenizer_args, "description", descriptions, accessions, **kwargs
    )

    return acc, embeddings
