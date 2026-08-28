"""Checkpoint-independent executable tensor program for the complete bilin18 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_preserving_attention import PROJECTION_NAMES, TensorAttentionBank
from tensor_preserving_mlp import TensorMLPBank


LAYERS = 18
WIDTH = 1152
VOCAB = 50_304


@dataclass(frozen=True)
class ShellCostReceipt:
    token_embedding_values: int
    residual_lambda_values: int
    unembedding_values: int
    parameter_free_rmsnorm_calls: int
    parameter_free_softcap_calls: int
    total_shell_stored_values: int


class TensorBilin18Program(nn.Module):
    """Complete owned bilin18 forward with an explicit contextual tensor core.

    Construction may read the checkpoint once.  Execution retains no checkpoint model,
    block, embedding, or linear-module reference.  The only cross-position primitive is
    the owned causal squared-bilinear attention contraction in ``attention_bank``.
    """

    def __init__(
        self,
        *,
        token_embedding: torch.Tensor,
        residual_lambdas: torch.Tensor,
        unembedding: torch.Tensor,
        attention_bank: TensorAttentionBank,
        mlp_bank: TensorMLPBank,
    ) -> None:
        super().__init__()
        if not isinstance(attention_bank, TensorAttentionBank) or not isinstance(
            mlp_bank, TensorMLPBank,
        ):
            raise ValueError("standalone program requires owned attention and MLP banks")
        if len(attention_bank.programs) != LAYERS or len(mlp_bank.programs) != LAYERS:
            raise ValueError("standalone program requires exactly 18 component pairs")
        width = attention_bank.programs[0].width
        if width != mlp_bank.programs[0].width or any(
            program.width != width for program in attention_bank.programs
        ) or any(program.width != width for program in mlp_bank.programs):
            raise ValueError("standalone component-bank widths disagree")
        if token_embedding.ndim != 2 or token_embedding.shape[1] != width:
            raise ValueError("token embedding has the wrong shape")
        if unembedding.ndim != 2 or unembedding.shape[1] != width:
            raise ValueError("unembedding has the wrong shape")
        if residual_lambdas.shape != (LAYERS, 2):
            raise ValueError("residual lambda tensor has the wrong shape")
        shell = (token_embedding, residual_lambdas, unembedding)
        if any(not value.is_floating_point() for value in shell) or len({
            value.device for value in shell
        }) != 1 or len({value.dtype for value in shell}) != 1 or any(
            not bool(torch.isfinite(value.detach()).all()) for value in shell
        ):
            raise ValueError("standalone shell tensors are malformed")
        bank_buffers = tuple(attention_bank.buffers()) + tuple(mlp_bank.buffers())
        if any(value.device != token_embedding.device for value in bank_buffers):
            raise ValueError("standalone shell and component banks occupy different devices")

        self.attention_bank = attention_bank
        self.mlp_bank = mlp_bank
        self.register_buffer("token_embedding", token_embedding.detach().clone())
        self.register_buffer("residual_lambdas", residual_lambdas.detach().clone())
        self.register_buffer("unembedding", unembedding.detach().clone())
        self.width = int(width)
        self.vocab_size = int(token_embedding.shape[0])
        self.logit_vocab = int(unembedding.shape[0])

    @classmethod
    def from_model(cls, model: nn.Module) -> "TensorBilin18Program":
        blocks = tuple(model.transformer.h)
        if len(blocks) != LAYERS:
            raise ValueError("bilin18 checkpoint block count changed")
        attention_bank = TensorAttentionBank.from_model(
            model, ranks={name: None for name in PROJECTION_NAMES},
        )
        mlp_bank = TensorMLPBank.from_model(model)
        return cls(
            token_embedding=model.transformer.wte.weight.detach(),
            residual_lambdas=torch.stack([
                block.lambdas.detach() for block in blocks
            ]),
            unembedding=model.lm_head.weight.detach(),
            attention_bank=attention_bank,
            mlp_bank=mlp_bank,
        )

    def validate_tokens(self, tokens: torch.Tensor) -> None:
        if not torch.is_tensor(tokens) or tokens.ndim != 2 or tokens.numel() == 0:
            raise ValueError("tokens must be a nonempty rank-two tensor")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must have torch.long dtype")
        if tokens.device != self.token_embedding.device:
            raise ValueError("tokens and standalone program occupy different devices")
        low, high = int(tokens.min()), int(tokens.max())
        if low < 0 or high >= self.vocab_size:
            raise ValueError("token ID is outside standalone embedding support")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.validate_tokens(tokens)
        state = F.embedding(tokens, self.token_embedding)
        state = F.rms_norm(state, (self.width,))
        initial = state
        first_value = None
        for site in range(LAYERS):
            lambdas = self.residual_lambdas[site].to(state.dtype)
            state = lambdas[0] * state + lambdas[1] * initial
            attention_state = F.rms_norm(state, (self.width,))
            attention_write, first_value = self.attention_bank.programs[site](
                attention_state, first_value,
            )
            state = state + attention_write
            mlp_state = F.rms_norm(state, (self.width,))
            state = state + self.mlp_bank.programs[site](mlp_state)
        final_state = F.rms_norm(state, (self.width,))
        logits = F.linear(final_state, self.unembedding.to(final_state.dtype))
        return (30.0 * torch.tanh(logits / 30.0)).float()

    def shell_cost_receipt(self) -> ShellCostReceipt:
        total = (
            self.token_embedding.numel()
            + self.residual_lambdas.numel()
            + self.unembedding.numel()
        )
        return ShellCostReceipt(
            token_embedding_values=self.token_embedding.numel(),
            residual_lambda_values=self.residual_lambdas.numel(),
            unembedding_values=self.unembedding.numel(),
            parameter_free_rmsnorm_calls=2 * LAYERS + 2,
            parameter_free_softcap_calls=1,
            total_shell_stored_values=total,
        )

    def cost_receipt(self) -> dict[str, Any]:
        shell = self.shell_cost_receipt()
        attention = self.attention_bank.cost_receipt()
        mlp = self.mlp_bank.cost_receipt()
        total = (
            shell.total_shell_stored_values
            + int(attention["total_stored_values"])
            + int(mlp["total_stored_values"])
        )
        return {
            "shell": shell.__dict__,
            "attention": attention,
            "mlp": mlp,
            "total_stored_values": total,
            "fitted_lookup_table_values": 0,
            "native_calls_per_forward": 0,
            "total_input_support": True,
            "sequence_primitive": "causal squared-bilinear tensor attention",
        }

    def operation_receipt(self, *, batch: int, sequence: int) -> dict[str, int]:
        if min(batch, sequence) <= 0:
            raise ValueError("batch and sequence must be positive")
        attention_mas = sum(program.multiply_adds(
            batch=batch, sequence=sequence,
        ) for program in self.attention_bank.programs)
        mlp_rows = [program.multiply_adds(
            batch=batch, sequence=sequence,
        ) for program in self.mlp_bank.programs]
        positions = batch * sequence
        return {
            "embedding_lookups": positions,
            "attention_multiply_adds": attention_mas,
            "mlp_linear_multiply_adds": sum(
                row["linear_multiply_adds"] for row in mlp_rows
            ),
            "mlp_bilinear_multiplies": sum(
                row["bilinear_multiplies"] for row in mlp_rows
            ),
            "unembedding_multiply_adds": positions * self.unembedding.numel(),
            "residual_scalar_multiplies": 2 * LAYERS * positions * self.width,
            "residual_additions": 3 * LAYERS * positions * self.width,
            "whole_state_rmsnorm_elements": (2 * LAYERS + 2) * positions * self.width,
            "softcap_elements": positions * self.logit_vocab,
        }
