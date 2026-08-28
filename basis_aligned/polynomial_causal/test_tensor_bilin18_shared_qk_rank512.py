from __future__ import annotations

import tensor_bilin18_shared_qk_rank512 as rank512


def test_rank512_price_formula_and_saving() -> None:
    attention = 18 * (5 * 1152 * 512 + 2 * 1152 * 1152 + 65)
    total = 115_900_452 + 286_675_200 + attention
    assert total == rank512.EXPECTED_TOTAL == 503_436_726
    assert rank512.DENSE_TOTAL - total == 42_467_328


def test_rank512_protocol_binds_rank384_and_causal_discriminator() -> None:
    source = rank512.Path(rank512.__file__).read_text()
    prereg = rank512.PREREG.read_text()
    for fragment in (
        "base.RANK = RANK", "context_delta_recovery", "recovery_gain_vs_rank384",
        "covered_harm_minus_rank384", "model_reference() is not None", "os.O_EXCL",
    ):
        assert fragment in source
    assert "activation-weighted basis" in prereg
    assert "0.90" in prereg and "0.95" in prereg
    assert rank512.RANK384_PARENT.name == "tensor_bilin18_shared_qk_whole_program_results.json"
