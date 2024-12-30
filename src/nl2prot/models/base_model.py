from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, override

from prettytable import PrettyTable
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
        self.logit_scale: nn.Parameter | None = None

    @abstractmethod
    @override
    def forward(self, batch: Any, only_embeddings=False, **kwargs) -> dict[str, Tensor]:
        pass

    def compute_similarity(
        self, desc_emb: Tensor, prot_emb: Tensor, **kwargs
    ) -> Tensor:
        assert self.logit_scale is not None, "Model must have a logit_scale"
        logit_scale = self.logit_scale.exp()
        return logit_scale * prot_emb @ desc_emb.T

    @property
    def description_encoder(self) -> nn.Module:
        assert self.desc_encoder is not None, "Description encoder not set"
        return self.desc_encoder

    @property
    def protein_encoder(self) -> nn.Module:
        assert self.prot_encoder is not None, "Protein encoder not set"
        return self.prot_encoder


def print_model_parameters(
    model: nn.Module, show_non_trainable: bool = False, include_grad_flag: bool = True
) -> None:
    """
    Print model parameters in a well-formatted table.

    Args:
        model: PyTorch model
        show_non_trainable: If True, also show non-trainable parameters
        include_grad_flag: If True, include requires_grad status in table
    """
    table = PrettyTable()
    if include_grad_flag:
        table.field_names = ["Layer", "Shape", "Params", "Trainable"]
    else:
        table.field_names = ["Layer", "Shape", "Params"]

    table.align["Layer"] = "l"  # Left align layer names
    table.align["Shape"] = "r"  # Right align shapes
    table.align["Params"] = "r"  # Right align param counts
    if include_grad_flag:
        table.align["Trainable"] = "c"  # Center align trainable status

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if show_non_trainable or param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count

            shape_str = str(tuple(param.shape)).replace(" ", "")

            if include_grad_flag:
                table.add_row(
                    [
                        name,
                        shape_str,
                        f"{param_count:,}",
                        "✓" if param.requires_grad else "✗",
                    ]
                )
            else:
                table.add_row([name, shape_str, f"{param_count:,}"])

    logging.info("Model Parameters:")
    logging.info("\n" + str(table))
    logging.info(f"Total Parameters: {total_params:,}")
    if show_non_trainable:
        logging.info(f"Trainable Parameters: {trainable_params:,}")
        logging.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
