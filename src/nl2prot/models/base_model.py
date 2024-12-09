from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, override

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        pass


class BaseDualEncoder(BaseModel, ABC):
    @abstractmethod
    @override
    def forward(
        self, batch: Any, return_embeddings=False, **kwargs
    ) -> dict[str, Tensor] | Tensor:
        pass
