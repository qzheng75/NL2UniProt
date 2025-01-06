from __future__ import annotations

import os
from typing import Any, override

import numpy as np
import torch
from faesm.esm import FAEsmForMaskedLM
from nl2prot.models.base_model import BaseDualEncoder
from torch import Tensor, nn
from transformers import BertModel


class ProjectionEncoder(nn.Module):
    def __init__(
        self, encoder: nn.Module, projection_dim: int, dropout: float = 0.1
    ) -> None:
        super(ProjectionEncoder, self).__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )

    @override
    def forward(self, **x: dict[str, Tensor]) -> Tensor:
        outputs = self.encoder(**x)

        if isinstance(outputs, dict):
            assert "last_hidden_state" in outputs
            embeddings = outputs["last_hidden_state"][:, 0, :]
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        features = self.projection(embeddings)
        return torch.nn.functional.normalize(features, dim=-1)


class FAEsmBertEncoder(BaseDualEncoder):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        esm_model_name="facebook/esm2b_t33_650M_UR50S",
        projection_dim=256,
        dropout=0.1,
        num_unfrozen_bert_layers=-1,
        num_unfrozen_esm_layers=-1,
    ) -> None:
        super(FAEsmBertEncoder, self).__init__()

        # config = {'cache_dir': os.environ["MODEL_CACHE"]}
        bert = BertModel.from_pretrained(
            bert_model_name,
            attn_implementation="sdpa",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            cache_dir=os.environ["MODEL_CACHE"],
        )
        esm = FAEsmForMaskedLM.from_pretrained(
            esm_model_name, dropout=dropout, use_fa=True
        )

        if num_unfrozen_bert_layers > 0:
            self._freeze_bert_layers(bert, num_unfrozen_bert_layers)
        if num_unfrozen_esm_layers > 0:
            self._freeze_esm_layers(esm, num_unfrozen_esm_layers)

        self.desc_encoder = ProjectionEncoder(bert, projection_dim, dropout)
        self.prot_encoder = ProjectionEncoder(esm, projection_dim, dropout)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @staticmethod
    def _freeze_bert_layers(bert_model: BertModel, num_unfrozen_layers: int) -> None:
        """Freeze all but the top N layers of BERT"""
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False

        total_layers = len(bert_model.encoder.layer)
        layers_to_freeze = total_layers - num_unfrozen_layers

        for layer in bert_model.encoder.layer[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    @staticmethod
    def _freeze_esm_layers(
        esm_model: FAEsmForMaskedLM, num_unfrozen_layers: int
    ) -> None:
        """Freeze all but the top N layers of ESM"""
        for param in esm_model.esm.embeddings.parameters():
            param.requires_grad = False

        total_layers = len(esm_model.esm.encoder.layer)
        layers_to_freeze = total_layers - num_unfrozen_layers

        for layer in esm_model.esm.encoder.layer[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    @override
    def trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Get trainable parameters grouped by component"""
        assert self.prot_encoder is not None, "Protein encoder not set"
        assert self.desc_encoder is not None, "Description encoder not set"
        assert self.logit_scale is not None, "Model must have a logit_scale"

        desc_params = [p for p in self.desc_encoder.parameters() if p.requires_grad]
        prot_params = [p for p in self.prot_encoder.parameters() if p.requires_grad]

        return {
            "desc": desc_params,
            "prot": prot_params,
            "other": [self.logit_scale],
        }

    @override
    def forward(self, batch: Any, only_embeddings=False, **kwargs) -> dict[str, Tensor]:
        names = batch["names"]
        prot_emb = batch["sequences"]
        desc_emb = batch["descriptions"]

        assert self.prot_encoder is not None, "Protein encoder not set"
        assert self.desc_encoder is not None, "Description encoder not set"
        assert self.logit_scale is not None, "Model must have a logit_scale"

        protein_features = self.prot_encoder(**prot_emb)
        desc_features = self.desc_encoder(**desc_emb)

        out_dict = {
            "names": names,
            "prot_embeddings": protein_features,
            "desc_embeddings": desc_features,
        }

        if only_embeddings:
            return out_dict

        out_dict["similarity"] = self.compute_similarity(
            desc_features, protein_features, **kwargs
        )
        return out_dict
