import json

import run_temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1 as run


def test_authority_and_prefixes_are_frozen():
    assert run.PREFIXES == (1, 2, 4, 8, 16, 32, 64, 128)
    assert {name: run.sha(path) for name, path in {
        "prior": run.PRIOR, "transfer_result": run.TRANSFER_RESULT,
        "tensor_result": run.TENSOR_RESULT, "capability_result": run.CAPABILITY_RESULT,
        "builder": run.BUILDER, "transfer_runner": run.TRANSFER_RUNNER,
    }.items()} == run.EXPECTED


def test_price_is_exact_for_four_panels():
    assert run.PRICE == json.loads(run.PRIOR.read_text())["frozen_design"]["price"]
    assert run.PRICE["model_forwards"] == 4 * (3 + run.PRICE["intervention_arms_per_panel"])
    assert run.PRICE["sequence_evaluations"] == 56 * 16


def test_no_u8_or_fitting_path_exists():
    source = run.Path(run.__file__).read_text()
    assert "project_reader" not in source and "linalg.svd" not in source and "Adam" not in source
    assert run.PRICE["fit_parameters"] == run.PRICE["model_updates"] == 0


def test_controls_are_separate_and_frozen():
    assert run.CONTROLS == ("P", "C")
    assert run.BARS["control_ratio"] == .25
    prior = json.loads(run.PRIOR.read_text())
    assert "top-32 and top-128" in prior["frozen_design"]["measurements"]
