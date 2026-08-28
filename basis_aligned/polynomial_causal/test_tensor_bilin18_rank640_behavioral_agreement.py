from pathlib import Path

import torch

import tensor_bilin18_rank640_behavioral_agreement as agreement


def test_metric_accumulation_has_exact_known_answer() -> None:
    native = torch.tensor([[[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]]])
    program = native.clone()
    targets = torch.tensor([[0, 1]])
    frequencies = torch.tensor([[0, 130]])
    accumulator = {name: agreement.empty_row() for name in (
        "all", *(row[0] for row in agreement.BUCKETS),
    )}
    agreement.add_metrics(accumulator, native, program, targets, frequencies)
    result = agreement.finalize(accumulator)
    assert result["all"]["top1_agreement"] == 1
    assert result["all"]["native_top1_accuracy"] == 1
    assert result["all"]["program_top1_accuracy"] == 1
    assert result["all"]["kl_live_program"] == 0
    assert result["0"]["n"] == 1 and result["125+"]["n"] == 1


def test_tail_combination_is_count_weighted() -> None:
    result = {
        "0": {"n": 1, "native_top1_accuracy": 1.0, "program_top1_accuracy": 0.0},
        "1-4": {"n": 3, "native_top1_accuracy": 1 / 3, "program_top1_accuracy": 1 / 3},
    }
    combined = agreement.combine_tail(result)
    assert combined["n"] == 4
    assert combined["native_top1_accuracy"] == 0.5
    assert combined["program_top1_accuracy"] == 0.25
    assert combined["accuracy_retained_fraction"] == 0.5


def test_gate_accepts_exact_registered_thresholds(monkeypatch) -> None:
    role = {
        "all": {
            "native_ce": 3.0, "program_ce": 3.005,
            "top1_agreement": 0.98, "accuracy_difference": -0.005,
            "kl_live_program": 0.01,
        },
        "target_frequency_0_4": {
            "accuracy_difference": -0.005, "accuracy_retained_fraction": 0.97,
        },
    }
    results = {name: dict(role) for name in ("a", "b")}
    parent = {"roles": {
        name: {"native": {"all": {"ce": 3.0}}, "program": {"all": {"ce": 3.005}}}
        for name in results
    }}
    hashes = {
        agreement.PREDICTIVE_PARENT: agreement.EXPECTED_PREDICTIVE_SHA256,
        agreement.CAUSAL_PARENT: agreement.EXPECTED_CAUSAL_SHA256,
    }
    monkeypatch.setattr(agreement, "file_sha256", lambda path: hashes[path])
    cost = {
        "total_stored_values": agreement.EXPECTED_TOTAL,
        "native_calls_per_forward": 0, "fitted_lookup_table_values": 0,
        "total_input_support": True,
    }
    ownership = {"storage_disjoint": True, "native_module_references": []}
    assert all(agreement.gates(results, cost, ownership, parent).values())


def test_sources_and_create_only_namespace_are_frozen() -> None:
    names = {Path(path).name for path in agreement.SOURCES}
    assert agreement.PREREG.name in names
    assert Path(__file__).name in names
    assert "tensor_bilin18_program.py" in names
    assert agreement.OUTPUT.name == "tensor_bilin18_rank640_behavioral_agreement_results.json"
    assert agreement.EXPECTED_TOTAL == 516_707_766

