"""Denominator-safe storage/execution accounting for table-plus-factor compilers.

The discovery compiler currently uses output hooks.  Every hook first executes the
native module, then substitutes a token table plus a low-rank correction for covered
token IDs, and falls back to the native output elsewhere.  Consequently four prices
must not be conflated:

1. trainable/incremental factor reals;
2. conditional covered-support table+factor description;
3. tensors allocated by the current hook implementation, plus retained native model;
4. a zero-native-call standalone program, which does not yet exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CompilerProgramCost:
    site_count: int
    active_token_types: int
    vocabulary_size: int
    input_blocks: int
    rank: int
    input_dim: int
    output_dim: int
    coefficient_bits: int = 32
    native_parameter_reals: int | None = None
    native_forward_executed: bool = True
    native_fallback_for_uncovered: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            self.site_count, self.active_token_types, self.vocabulary_size,
            self.input_blocks, self.rank, self.input_dim, self.output_dim,
            self.coefficient_bits,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields) or (
            self.active_token_types > self.vocabulary_size
        ) or self.native_parameter_reals is not None and (
            type(self.native_parameter_reals) is not int
            or self.native_parameter_reals <= 0
        ):
            raise ValueError("compiler cost dimensions are malformed")

    @property
    def factor_reals(self) -> int:
        return self.site_count * self.rank * (
            self.input_blocks * self.input_dim + self.output_dim
        )

    @property
    def active_table_value_reals(self) -> int:
        return self.site_count * self.active_token_types * self.output_dim

    @property
    def active_token_index_bits(self) -> int:
        # One shared ordered support list suffices because all sites use the same mask.
        return self.active_token_types * math.ceil(math.log2(self.vocabulary_size))

    @property
    def conditional_value_reals(self) -> int:
        """Values needed only on the registered covered-token support."""

        return self.factor_reals + self.active_table_value_reals

    @property
    def conditional_description_bits(self) -> int:
        return self.conditional_value_reals * self.coefficient_bits + (
            self.active_token_index_bits
        )

    @property
    def allocated_dense_table_reals(self) -> int:
        """Table values materialized by the present 50,257-row implementation."""

        return self.site_count * self.vocabulary_size * self.output_dim

    @property
    def hook_added_reals(self) -> int:
        return self.allocated_dense_table_reals + self.factor_reals

    @property
    def current_hook_parameter_reals(self) -> int | None:
        if self.native_parameter_reals is None:
            return None
        return self.native_parameter_reals + self.hook_added_reals

    @property
    def standalone_admissible(self) -> bool:
        return not self.native_forward_executed and not self.native_fallback_for_uncovered

    @property
    def standalone_parameter_reals(self) -> int | None:
        return self.conditional_value_reals if self.standalone_admissible else None

    def nats_per_million(self, recovered_nats: float, *, denominator: str) -> float:
        if not math.isfinite(recovered_nats) or denominator not in {
            "factor_only", "conditional_values", "hook_added", "standalone",
        }:
            raise ValueError("unknown or malformed compiler efficiency request")
        if denominator == "factor_only":
            count = self.factor_reals
        elif denominator == "conditional_values":
            count = self.conditional_value_reals
        elif denominator == "hook_added":
            count = self.hook_added_reals
        else:
            count = self.standalone_parameter_reals
            if count is None:
                raise RuntimeError("no zero-native-call standalone program exists")
        return recovered_nats * 1_000_000.0 / count

    def receipt(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update({
            "factor_reals": self.factor_reals,
            "active_table_value_reals": self.active_table_value_reals,
            "active_token_index_bits": self.active_token_index_bits,
            "conditional_value_reals": self.conditional_value_reals,
            "conditional_description_bits": self.conditional_description_bits,
            "allocated_dense_table_reals": self.allocated_dense_table_reals,
            "hook_added_reals": self.hook_added_reals,
            "current_hook_parameter_reals": self.current_hook_parameter_reals,
            "standalone_admissible": self.standalone_admissible,
            "standalone_parameter_reals": self.standalone_parameter_reals,
        })
        return payload
