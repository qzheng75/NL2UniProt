from __future__ import annotations

import os
import pathlib

import yaml
from scanpy import read
from sklearn.neighbors import KDTree

from nl2prot.validate.embedding import compute_embeddings, get_trainer


class NL2ProtAPI:
    def __init__(self, config: str | None = None):
        nl2prot_path = os.environ.get("NL2PROT_PATH", ".")

        if config is None:
            config = os.path.join(nl2prot_path, "config/deploy/config.yml")
            assert os.path.exists(config), "You must provide a configuration file"

        with open(config, "r") as f:
            self.config = yaml.safe_load(f)

        ckpt_path = pathlib.Path(
            os.path.join(
                nl2prot_path,
                self.config["trainer"]["trainer_args"]["resume_from_checkpoint"],
            )
        )

        folder, _ = ckpt_path.parent, ckpt_path.name

        seq_embed_path = os.path.join(folder, "seq_embed.h5ad")
        assert os.path.exists(
            seq_embed_path
        ), f"Sequence embedding file not found: {seq_embed_path}"

        self.tokenizer_args = self.config["dataloader"]["collate_args"][
            "desc_tokenizer_args"
        ]
        adata = read(seq_embed_path)
        self.accessions: list[str] = adata.obs["accession"].tolist()
        self.vector_db = KDTree(adata.X)

        quantize_model = self.config["model"].get("quantize", False)
        self.device = self.config["trainer"]["trainer_args"].get("device", "cpu")
        self.trainer = get_trainer(config, str(ckpt_path), quantize_model, self.device)

    def recommend_sequences(
        self, description: str | list[str], top_k: int = 20
    ) -> dict[str, list[str] | list[float]] | list[dict[str, list[str] | list[float]]]:
        if isinstance(description, str):
            description = [description]

        _, embeddings = compute_embeddings(
            self.trainer,
            self.tokenizer_args,
            "description",
            description,
            device=self.device,
            enable_progress_bar=False,
        )
        dist, idx = self.vector_db.query(embeddings, k=top_k)

        out = [
            {"accession": [self.accessions[i] for i in indices], "distance": d}
            for indices, d in zip(idx, dist)
        ]

        if len(out) == 1:
            return out[0]

        return out
