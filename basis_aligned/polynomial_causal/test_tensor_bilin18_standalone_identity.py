from __future__ import annotations

import tensor_bilin18_standalone_identity as identity


def test_standalone_gate_binds_independence_context_and_complete_cost() -> None:
    source = identity.Path(identity.__file__).read_text()
    prereg = identity.PREREG.read_text()
    for fragment in (
        "model_reference() is not None",
        "native_context_max_abs",
        "changed_program_logits",
        "545_904_054",
        "115_900_452",
        "430_003_602",
        "native_module_references",
        "native_program_storage_disjoint",
        "os.O_EXCL",
    ):
        assert fragment in source
    assert "checkpoint model is garbage-collected" in prereg
    assert "No native checkpoint object" in prereg
    assert identity.PARENT.name == "tensor_component_bank_composition_identity_results.json"


def test_source_manifest_binds_every_complete_program_boundary() -> None:
    names = {path.name for path in identity.SOURCES}
    assert {
        "tensor_bilin18_program.py",
        "tensor_bilin18_standalone_identity.py",
        "test_tensor_bilin18_program.py",
        "test_tensor_bilin18_standalone_identity.py",
        "tensor_preserving_attention.py",
        "tensor_preserving_mlp.py",
        "bilin18_observed_model_facade.py",
        "TENSOR_BILIN18_STANDALONE_IDENTITY_PREREGISTRATION.md",
    } <= names
