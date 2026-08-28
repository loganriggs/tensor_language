from __future__ import annotations

import pytest

from compiler_program_cost import CompilerProgramCost


def current_cost(*, blocks: int = 1, rank: int = 8) -> CompilerProgramCost:
    return CompilerProgramCost(
        site_count=36, active_token_types=5419, vocabulary_size=50257,
        input_blocks=blocks, rank=rank, input_dim=1152, output_dim=1152,
    )


def test_s1751_rank8_exact_factor_and_table_counts() -> None:
    cost = current_cost()
    assert cost.factor_reals == 663_552
    assert cost.active_table_value_reals == 224_736_768
    assert cost.conditional_value_reals == 225_400_320
    assert cost.active_token_index_bits == 86_704
    assert cost.allocated_dense_table_reals == 2_084_258_304
    assert cost.hook_added_reals == 2_084_921_856


def test_factor_only_efficiency_is_not_whole_program_efficiency() -> None:
    cost = current_cost()
    factor = cost.nats_per_million(0.60060, denominator="factor_only")
    conditional = cost.nats_per_million(0.60060, denominator="conditional_values")
    hook_added = cost.nats_per_million(0.60060, denominator="hook_added")
    assert factor == pytest.approx(0.90512, rel=1e-4)
    assert conditional == pytest.approx(0.0026646, rel=1e-4)
    assert hook_added == pytest.approx(0.00028807, rel=1e-4)
    assert factor / conditional > 300


def test_nonlocal_feature_blocks_change_factors_not_shared_table_price() -> None:
    one = current_cost(blocks=1)
    three = current_cost(blocks=3)
    assert one.factor_reals == 663_552
    assert three.factor_reals == 1_327_104
    assert one.active_table_value_reals == three.active_table_value_reals


def test_current_hook_has_no_standalone_price_and_retains_native_parameters() -> None:
    cost = CompilerProgramCost(
        site_count=36, active_token_types=5419, vocabulary_size=50257,
        input_blocks=1, rank=8, input_dim=1152, output_dim=1152,
        native_parameter_reals=546_000_000,
    )
    assert not cost.standalone_admissible
    assert cost.standalone_parameter_reals is None
    assert cost.current_hook_parameter_reals == 546_000_000 + cost.hook_added_reals
    with pytest.raises(RuntimeError, match="standalone"):
        cost.nats_per_million(0.6, denominator="standalone")


def test_hypothetical_total_support_zero_native_program_can_be_priced() -> None:
    cost = CompilerProgramCost(
        site_count=1, active_token_types=10, vocabulary_size=10,
        input_blocks=2, rank=3, input_dim=5, output_dim=7,
        coefficient_bits=16, native_forward_executed=False,
        native_fallback_for_uncovered=False,
    )
    assert cost.standalone_admissible
    assert cost.factor_reals == 51
    assert cost.active_table_value_reals == 70
    assert cost.standalone_parameter_reals == 121
    assert cost.conditional_description_bits == 121 * 16 + 40


def test_bad_dimensions_and_efficiency_requests_fail_closed() -> None:
    with pytest.raises(ValueError, match="malformed"):
        CompilerProgramCost(
            site_count=1, active_token_types=11, vocabulary_size=10,
            input_blocks=1, rank=1, input_dim=1, output_dim=1,
        )
    with pytest.raises(ValueError, match="unknown"):
        current_cost().nats_per_million(float("nan"), denominator="factor_only")
