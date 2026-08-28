from __future__ import annotations

import tensor_component_bank_composition_identity as composition


def test_composition_protocol_binds_both_parents_and_remaining_interfaces() -> None:
    source = composition.Path(composition.__file__).read_text()
    prereg = composition.PREREG.read_text()
    assert "AttentionNativePoison" in source and "MLPNativePoison" in source
    assert "430_003_602" in source
    assert "literal_native_attention_calls" in source
    assert "literal_native_mlp_calls" in source
    assert "mutually_disjoint_storage" in source
    assert "tensor_preserving_attention_identity_results.json" in source
    assert "tensor_preserving_mlp_identity_results.json" in source
    assert "token embedding" in prereg and "unembedding" in prereg
    assert len(composition.UNOWNED_EXACT_INTERFACES) == 5


def test_pointer_set_deduplicates_aliases_without_losing_distinct_storage() -> None:
    import torch

    first = torch.randn(4)
    alias = first.view(2, 2)
    second = first.clone()
    observed = composition.pointer_set((first, alias, second))
    assert len(observed) == 2
