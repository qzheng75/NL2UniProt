from __future__ import annotations

from typing import override

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy


class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()

    @override
    def forward(self, logits: Tensor) -> Tensor:
        n = logits.size(0)

        labels = torch.arange(n).to(logits.device)
        loss_i = cross_entropy(logits, labels)
        loss_t = cross_entropy(logits.T, labels)

        loss = (loss_i + loss_t) / 2
        return loss


class TorchLossWrapper(torch.nn.Module):
    def __init__(self, loss_name: str, **loss_args):
        super(TorchLossWrapper, self).__init__()
        self.loss = getattr(torch.nn, loss_name)(**loss_args)

    @override
    def forward(self, out: dict[str, Tensor]) -> Tensor:
        return self.loss(out["out"], out["target"])
