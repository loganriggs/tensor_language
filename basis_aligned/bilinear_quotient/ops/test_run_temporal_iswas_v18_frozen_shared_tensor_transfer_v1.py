import json

import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as run


def test_frozen_program_and_price_are_literal():
    prior = json.loads(run.PRIOR.read_text())
    assert run.RANK == 8 and run.H3 == 3 and run.TOP == 32
    assert prior["frozen_design"]["price"] == run.PRICE
    assert run.PRICE["model_forwards"] == 4 * (3 + run.PRICE["intervention_arms_per_panel"])
    assert run.PRICE["sequence_evaluations"] == 44 * 16


def test_authorities_are_bound_and_capability_only():
    assert {name: run.sha(path) for name, path in {
        "prior": run.PRIOR, "tensor_result": run.TENSOR_RESULT,
        "head_result": run.HEAD_RESULT, "capability_result": run.CAPABILITY_RESULT,
        "capability_runner": run.CAPABILITY_RUNNER, "builder": run.BUILDER,
    }.items()} == run.EXPECTED
    capability = json.loads(run.CAPABILITY_RESULT.read_text())
    assert capability["causal_outcomes_opened"] is False
    assert all(len(capability["jointly_capable_row_ids"][family]) == 16
               for family in run.TARGETS)


def test_control_normalization_is_registered_separately():
    prior = json.loads(run.PRIOR.read_text())
    assert run.CONTROLS == ("P", "C")
    assert run.BARS["control_ratio"] == .25
    assert "For P and C separately" in prior["frozen_design"]["controls"]


def test_no_fit_or_optimizer_is_present():
    source = run.Path(run.__file__).read_text()
    assert "torch.linalg.svd" not in source
    assert "Adam" not in source
    assert run.PRICE["fit_parameters"] == run.PRICE["model_updates"] == 0
