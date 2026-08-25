"""TheseusBench contract (spec v0.2 section 3, verbatim semantics).

Manifest declares everything the harness needs to price and wire a replacement;
Replacement is the base class submissions implement. The harness computes all
scores (Invariant 1); replacements only see declared inputs (Invariant 2).
"""
from dataclasses import dataclass, field
from typing import Literal, Optional
import torch
import torch.nn as nn


@dataclass
class Manifest:
    model: str                        # e.g. "bilinear-546m"
    module: str                       # canonical id, e.g. "blocks.3.mlp"
    granularity: Literal["module", "head", "neuron_group"]
    inputs: list                      # canonical activation names; edges = len(inputs)
    library_refs: list = field(default_factory=list)
    constants: list = field(default_factory=list)
    state_cardinality: Optional[int] = None
    positionwise: bool = False


class Replacement(nn.Module):
    manifest: Manifest

    def forward(self, *declared_inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, dev_batch_iterator) -> None:
        pass
