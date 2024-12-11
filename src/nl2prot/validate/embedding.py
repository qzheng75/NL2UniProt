from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from nl2prot.data.collate import single_encoder_collate_fn
from nl2prot.data.dataset import DescDataset, ProtDataset


def compute_embeddings(
    pl_module: pl.LightningModule,
    tokenizer_args: dict[str, Any],
    input_type: Literal["sequence", "description"],
    to_embed: list[str],
    accessions: list[str] | None = None,
    batch_size: int = 16,
    device: str = "cuda",
    precision: Literal["16", "32", "bfloat16"] = "bfloat16",
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
        dataset = DescDataset(to_embed)
        collate_fn = lambda batch: single_encoder_collate_fn(
            batch, tokenizer_args, "description"
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    dtype_dict = {"16": torch.float16, "32": torch.float32, "bfloat16": torch.bfloat16}
    for param in pl_module.parameters():
        param.data = param.data.to(dtype_dict[precision])

    pl_module.eval()
    with torch.no_grad():
        embeddings = []
        for idx, batch in enumerate(dataloader):
            batch["tokens"] = {k: v.to(device) for k, v in batch["tokens"].items()}
            out = pl_module.predict_step(batch, idx)
            embeddings.append(out["embeddings"].float().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return accessions, embeddings
