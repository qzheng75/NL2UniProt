from __future__ import annotations

import os
from typing import Any, Literal, override

import numpy as np
import torch
from nl2prot.models.base_model import BaseDualEncoder
from torch import Tensor, nn
from transformers import BertModel, EsmModel


class ProjectionEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )

    @override
    def forward(self, **x: dict[str, Tensor]) -> Tensor:
        outputs = self.encoder(**x)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        features = self.projection(embeddings)
        return nn.functional.normalize(features, dim=-1)


class EsmBertEncoder(BaseDualEncoder):
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        esm_model_name: str = "facebook/esm1b_t33_650M_UR50S",
        projection_dim: int = 256,
        dropout: float = 0.1,
        num_unfrozen_bert_layers: int = 2,
        num_unfrozen_esm_layers: int = 2,
        init_method: Literal["normal", "xavier"] = "normal",
    ):
        super(EsmBertEncoder, self).__init__()

        bert = BertModel.from_pretrained(
            bert_model_name, cache_dir=os.environ["MODEL_CACHE"]
        )
        esm = EsmModel.from_pretrained(
            esm_model_name, cache_dir=os.environ["MODEL_CACHE"]
        )

        self._freeze_bert_layers(bert, num_unfrozen_bert_layers)
        self._freeze_esm_layers(esm, num_unfrozen_esm_layers)

        self.desc_encoder = ProjectionEncoder(bert, projection_dim, dropout)
        self.prot_encoder = ProjectionEncoder(esm, projection_dim, dropout)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if init_method == "xavier":
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if "weight" in name and param.ndimension() >= 2:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    @staticmethod
    def _freeze_bert_layers(bert_model: nn.Module, num_unfrozen_layers: int):
        """Freeze all but the bottom N layers of BERT"""
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False

        total_layers = len(bert_model.encoder.layer)
        layers_to_freeze = total_layers - num_unfrozen_layers

        for layer in bert_model.encoder.layer[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    @staticmethod
    def _freeze_esm_layers(esm_model: nn.Module, num_unfrozen_layers: int):
        """Freeze all but the bottom N layers of ESM"""
        for param in esm_model.embeddings.parameters():
            param.requires_grad = False

        total_layers = len(esm_model.encoder.layer)
        layers_to_freeze = total_layers - num_unfrozen_layers

        for layer in esm_model.encoder.layer[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    @override
    def trainable_parameters(self):
        """Get trainable parameters grouped by component"""
        assert self.desc_encoder is not None, "Description encoder not set"
        assert self.prot_encoder is not None, "Protein encoder not set"
        assert self.logit_scale is not None, "Model must have a logit_scale"

        desc_params = [p for p in self.desc_encoder.parameters() if p.requires_grad]
        prot_params = [p for p in self.prot_encoder.parameters() if p.requires_grad]

        return {
            "desc": desc_params,
            "prot": prot_params,
            "other": [self.logit_scale],
        }

    @override
    def forward(
        self, batch: Any, return_embeddings=False, **kwargs
    ) -> dict[str, Tensor] | Tensor:
        names = batch["names"]
        prot_emb = batch["sequences"]
        desc_emb = batch["descriptions"]

        assert self.desc_encoder is not None, "Description encoder not set"
        assert self.prot_encoder is not None, "Protein encoder not set"
        protein_features = self.prot_encoder(
            input_ids=prot_emb["input_ids"], attention_mask=prot_emb["attention_mask"]
        )
        desc_features = self.desc_encoder(
            input_ids=desc_emb["input_ids"],
            token_type_ids=desc_emb["token_type_ids"],
            attention_mask=desc_emb["attention_mask"],
        )

        if return_embeddings:
            return {
                "names": names,
                "prot_embeddings": protein_features,
                "desc_embeddings": desc_features,
            }
        return self.compute_similarity(desc_features, protein_features, **kwargs)
