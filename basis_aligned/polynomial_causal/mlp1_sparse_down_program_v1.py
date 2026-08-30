"""Standalone hard-TopK program for bilin18 MLP1's bias-free Down action."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


GATE_DIM = 4608
OUTPUT_DIM = 1152
DICTIONARY_SIZE = 512
ACTIVE_ATOMS = 32
STATE_SHAPES = {
    "encoder": (DICTIONARY_SIZE, GATE_DIM),
    "decoder": (OUTPUT_DIM, DICTIONARY_SIZE),
    "intercept": (OUTPUT_DIM,),
}


def topk_relu(scores: torch.Tensor, k: int = ACTIVE_ATOMS) -> torch.Tensor:
    """Keep the k largest positive scores, matching the established MLP1 assay."""

    if scores.ndim != 2 or not 0 < k <= scores.shape[1]:
        raise ValueError("TopK score shape or k is invalid")
    values, indices = torch.topk(scores, k, dim=1, largest=True, sorted=False)
    output = torch.zeros_like(scores)
    output.scatter_(1, indices, torch.relu(values))
    return output


def validate_state(value: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != set(STATE_SHAPES):
        raise RuntimeError("sparse-Down program state schema changed")
    output = {}
    for key, shape in STATE_SHAPES.items():
        tensor = value[key]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.float32 \
                or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"sparse-Down state {key} changed")
        output[key] = tensor.detach().cpu().contiguous().clone()
    norms = output["encoder"].double().norm(dim=1)
    if float((norms - 1.0).abs().max()) > 2e-4:
        raise RuntimeError("sparse-Down encoder rows lost unit norm")
    return output


class SparseDownProgram(nn.Module):
    def __init__(self, state: Mapping[str, Any], device: torch.device | str) -> None:
        super().__init__()
        checked = validate_state(state)
        self.register_buffer("encoder", checked["encoder"].to(device))
        self.register_buffer("decoder", checked["decoder"].to(device))
        self.register_buffer("intercept", checked["intercept"].to(device))

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-1] != GATE_DIM:
            raise RuntimeError("sparse-Down gate width changed")
        shape = gate.shape[:-1]
        flat = gate.float().reshape(-1, GATE_DIM)
        codes = topk_relu(flat @ self.encoder.T)
        output = codes @ self.decoder.T + self.intercept
        return output.reshape(*shape, OUTPUT_DIM).to(gate.dtype)

    @staticmethod
    def price() -> dict[str, Any]:
        encoder = DICTIONARY_SIZE * GATE_DIM
        decoder = OUTPUT_DIM * DICTIONARY_SIZE
        intercept = OUTPUT_DIM
        native_down = GATE_DIM * OUTPUT_DIM
        return {
            "stored_float32_reals": encoder + decoder + intercept,
            "encoder_reals": encoder,
            "decoder_reals": decoder,
            "intercept_reals": intercept,
            "active_atoms_per_token": ACTIVE_ATOMS,
            "score_multiplies_per_token": encoder,
            "sparse_decode_multiplies_per_token": ACTIVE_ATOMS * OUTPUT_DIM,
            "native_down_reals": native_down,
            "fraction_of_native_down_storage": (encoder + decoder + intercept) / native_down,
            "native_full_mlp1_reals": 15_926_400,
            "fraction_of_native_full_mlp1_storage_saved": (
                native_down - (encoder + decoder + intercept)
            ) / 15_926_400,
            "topk_comparisons_and_indices_charged_separately": True,
        }


def cpu_state(program: SparseDownProgram) -> dict[str, torch.Tensor]:
    return {
        key: getattr(program, key).detach().cpu().float().contiguous().clone()
        for key in STATE_SHAPES
    }
