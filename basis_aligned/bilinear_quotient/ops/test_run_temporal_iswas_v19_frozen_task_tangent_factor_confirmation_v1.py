import json

import run_temporal_iswas_v19_frozen_task_tangent_factor_confirmation_v1 as run


def test_authority_and_frozen_order_are_exact():
    assert {name: run.sha(path) for name, path in {
        "prior": run.PRIOR, "task_result": run.TASK_RESULT, "task_runner": run.TASK_RUNNER,
        "capability_result": run.CAPABILITY_RESULT, "builder": run.BUILDER,
        "executor": run.EXECUTOR}.items()} == run.EXPECTED
    result = json.loads(run.TASK_RESULT.read_text())
    order = result["stable_factor_order_first256"]
    assert len(order) == len(set(order)) == 256


def test_price_and_panels_are_frozen():
    assert run.PRICE == json.loads(run.PRIOR.read_text())["frozen_design"]["price"]
    assert run.PRICE["model_forwards"] == 4 * (3 + run.PRICE["intervention_arms_per_panel"])
    assert run.PANELS == ("A1", "A2", "P", "C")


def test_no_v19_fit_rank_or_dose_path():
    source = run.Path(run.__file__).read_text()
    assert "stable_order(" not in source and "backward(" not in source
    assert "Adam" not in source and "linalg.svd" not in source
    assert run.PRICE["fit_parameters"] == run.PRICE["model_updates"] == 0
