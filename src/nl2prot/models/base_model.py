from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        pass
