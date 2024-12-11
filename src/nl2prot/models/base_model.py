from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, override

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        pass


class BaseDualEncoder(BaseModel, ABC):
    def __init__(self):
        super().__init__()
        self.desc_encoder: nn.Module | None = None
        self.prot_encoder: nn.Module | None = None

    @abstractmethod
    @override
    def forward(
        self, batch: Any, return_embeddings=False, **kwargs
    ) -> dict[str, Tensor] | Tensor:
        pass

    @property
    def description_encoder(self) -> nn.Module:
        assert self.desc_encoder is not None, "Description encoder not set"
        return self.desc_encoder

    @property
    def protein_encoder(self) -> nn.Module:
        assert self.prot_encoder is not None, "Protein encoder not set"
        return self.prot_encoder
