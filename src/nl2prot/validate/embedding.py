from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from Bio import SeqIO
from nl2prot.data.collate import single_encoder_collate_fn
from nl2prot.data.dataset import DescDataset, ProtDataset
from nl2prot.template.load_module import load_everything
from nl2prot.trainer.dual_encoder_pl import DualEncoderPl
from tqdm import tqdm


def compute_embeddings(
    pl_module: pl.LightningModule,
    tokenizer_args: dict[str, Any],
    input_type: Literal["sequence", "description"],
    to_embed: list[str],
    accessions: list[str] | None = None,
    batch_size: int = 16,
    device: str = "cuda",
    precision: Literal["16", "32", "bfloat16"] = "bfloat16",
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

    dtype_dict = {"16": torch.float16, "32": torch.float32, "bfloat16": torch.bfloat16}

    pl_module.eval()
    with torch.autocast(device_type="cuda", dtype=dtype_dict[precision]):
        with torch.no_grad():
            embeddings = []
            for idx, batch in enumerate(
                tqdm(dataloader, disable=not enable_progress_bar)
            ):
                batch["tokens"] = {k: v.to(device) for k, v in batch["tokens"].items()}
                out = pl_module.predict_step(batch, idx)
                embeddings.append(out["embeddings"].float().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return accessions, embeddings


def get_pl_module(
    module_config_path: str,
    model_ckpt_path: str,
) -> pl.LightningModule:
    assert (
        module_config_path is not None and model_ckpt_path is not None
    ), "You must provide a module config and a model checkpoint"
    all_modules = load_everything(module_config_path)
    pl_module: pl.LightningModule = all_modules["pl_module"]

    if isinstance(pl_module, DualEncoderPl):
        pl_module = DualEncoderPl.load_from_checkpoint(
            model_ckpt_path, model=pl_module.model, loss_fn=pl_module.loss_fn
        )
    else:
        raise NotImplementedError("Only DualEncoderPl is supported for now")

    return pl_module


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

    pl_module = get_pl_module(module_config_path, model_ckpt_path)
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
            pl_module, tokenizer_args, "sequence", seqs, names, **kwargs
        )
    else:
        assert (
            accessions is not None
        ), "You must provide accessions for raw sequences input type"
        acc, embeddings = compute_embeddings(
            pl_module, tokenizer_args, "sequence", raw_sequences, accessions, **kwargs
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

    pl_module = get_pl_module(module_config_path, model_ckpt_path)
    acc, embeddings = compute_embeddings(
        pl_module, tokenizer_args, "description", descriptions, accessions, **kwargs
    )

    return acc, embeddings
