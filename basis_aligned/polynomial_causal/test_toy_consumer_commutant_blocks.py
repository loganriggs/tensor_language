from __future__ import annotations

import toy_consumer_commutant_blocks as toy


def test_planted_blocks_gauge_null_and_edit_controls() -> None:
    result = toy.run_checks()
    assert result["all_passed"], result
    assert result["planted_commutant_dimension"] == 3
    assert result["recovered_block_sizes"] == [2, 2, 3]
    assert result["dense_null_commutant_dimension"] == 1
    assert result["recovered_offblock_energy_fraction"] < 1e-20
    assert max(abs(value) for value in result["planted_cross_block_edit_interactions"]) < 1e-12
    assert max(abs(value) for value in result["dense_cross_block_edit_interactions"]) > 1e-2
