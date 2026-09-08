import json

import run_temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v2 as run


def test_v2_is_bound_to_exact_invalid_v1_and_amended_executor():
    assert {name: run.sha(path) for name, path in {
        "base_runner": run.BASE_RUNNER, "prior": run.PRIOR, "v1_result": run.V1_RESULT,
    }.items()} == run.EXPECTED
    invalid = json.loads(run.V1_RESULT.read_text())
    assert invalid["terminal"] == "invalid_instrument"
    assert not invalid["predictions"][
        "pred_a_authority_formula_replay_hook_coverage_finiteness_and_exact_price"]


def test_v2_changes_only_precision_gate_and_output_identity():
    assert run.experiment.BARS["output_relative_squared"] == 1e-8
    assert run.experiment.PRICE["model_forwards"] == 56
    assert run.OUT.name.endswith("_v2_result.json")
