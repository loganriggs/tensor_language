"""Pure analysis for the four-arm hybrid tensor-class oracle."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


ARM_NAMES = (
    "both_compiled",
    "attention_native",
    "mlp_native",
    "both_native",
)


@dataclass(frozen=True)
class HybridClassAnalysis:
    live_ce: float
    ce: Mapping[str, float]
    harm: Mapping[str, float]
    attention_restoration_gain: float
    mlp_restoration_gain: float
    interaction_harm: float
    dominant_missing_contraction: str

    def __post_init__(self) -> None:
        if set(self.ce) != set(ARM_NAMES) or set(self.harm) != set(ARM_NAMES) or not all(
            math.isfinite(value) for value in (
                self.live_ce,
                *self.ce.values(),
                *self.harm.values(),
                self.attention_restoration_gain,
                self.mlp_restoration_gain,
                self.interaction_harm,
            )
        ) or self.dominant_missing_contraction not in {
            "attention", "mlp", "tied",
        }:
            raise ValueError("hybrid tensor-class analysis is malformed")
        object.__setattr__(self, "ce", MappingProxyType(dict(self.ce)))
        object.__setattr__(self, "harm", MappingProxyType(dict(self.harm)))


def analyze_hybrid_losses(
    ce: Mapping[str, float], *, live_ce: float, native_atol: float = 1e-6,
    dominance_atol: float = 1e-6,
) -> HybridClassAnalysis:
    """Attribute compiled-program harm to missing typed contractions and interaction.

    ``attention_native`` compiles only MLPs, while ``mlp_native`` compiles only
    attention.  Hence the reduction from ``both_compiled`` harm after making attention
    native is the attention-restoration gain, and conversely for MLP.
    """

    if set(ce) != set(ARM_NAMES) or not math.isfinite(live_ce) or native_atol < 0 or (
        dominance_atol < 0
    ) or any(not math.isfinite(value) for value in ce.values()):
        raise ValueError("hybrid arm losses or tolerances are malformed")
    values = {name: float(ce[name]) for name in ARM_NAMES}
    if abs(values["both_native"] - live_ce) > native_atol:
        raise ValueError("both-native arm does not reproduce live CE")
    harm = {name: values[name] - live_ce for name in ARM_NAMES}
    attention_gain = harm["both_compiled"] - harm["attention_native"]
    mlp_gain = harm["both_compiled"] - harm["mlp_native"]
    interaction = (
        harm["both_compiled"]
        - harm["attention_native"]
        - harm["mlp_native"]
    )
    if attention_gain > mlp_gain + dominance_atol:
        dominant = "attention"
    elif mlp_gain > attention_gain + dominance_atol:
        dominant = "mlp"
    else:
        dominant = "tied"
    return HybridClassAnalysis(
        live_ce=float(live_ce), ce=values, harm=harm,
        attention_restoration_gain=attention_gain,
        mlp_restoration_gain=mlp_gain,
        interaction_harm=interaction,
        dominant_missing_contraction=dominant,
    )
