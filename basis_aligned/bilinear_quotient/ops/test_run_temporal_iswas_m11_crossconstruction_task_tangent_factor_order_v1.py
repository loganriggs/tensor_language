import json
import torch

import run_temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1 as run


def test_stable_order_uses_minimum_normalized_signed_contribution():
    first = torch.tensor([4.0, 2.0, 1.0, -1.0])
    second = torch.tensor([1.0, 3.0, 2.0, -2.0])
    order, score = run.stable_order((first, second))
    assert order.tolist() == [1, 0, 2, 3]
    assert score[1] > score[0] == score[2] > score[3]


def test_authorities_price_and_v19_seal_are_exact():
    assert {name: run.sha(path) for name, path in {
        "prior": run.PRIOR, "greedy_result": run.GREEDY_RESULT,
        "factor_result": run.FACTOR_RESULT, "v19_capability": run.V19_CAPABILITY,
        "greedy_runner": run.GREEDY_RUNNER, "builder17": run.BUILDER17,
        "builder18": run.BUILDER18}.items()} == run.EXPECTED
    assert run.PRICE == json.loads(run.PRIOR.read_text())["frozen_design"]["price"]
    assert json.loads(run.V19_CAPABILITY.read_text())["causal_outcomes_opened"] is False


def test_task_tangent_is_not_output_reconstruction_or_parameter_fit():
    source = run.Path(run.__file__).read_text()
    assert "factor_gram" not in source and "exact_greedy" not in source
    assert "Adam" not in source and "linalg.svd" not in source
    assert run.PRICE["fit_parameters"] == run.PRICE["model_updates"] == 0
