import json
import torch

import run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1 as run


def test_exact_greedy_prefers_largest_fixed_reduction_and_ties_by_index():
    gram = torch.diag(torch.tensor([.1, .4, .4, .1]))
    assert run.exact_greedy([gram], 4) == (1, 2, 0, 3)


def test_gram_residual_matches_direct_orthogonal_atoms():
    hidden = torch.eye(3)
    down = torch.eye(3)
    gram = run.factor_gram(hidden, down)
    assert abs(run.residual_from_gram(gram, (0,)) - run.direct_residual(
        hidden, hidden @ down.T, down, (0,))) < 1e-6


def test_authority_price_and_pristine_v19_are_frozen():
    assert {name: run.sha(path) for name, path in {
        "prior": run.PRIOR, "factor_result": run.FACTOR_RESULT,
        "transfer_result": run.TRANSFER_RESULT, "tensor_result": run.TENSOR_RESULT,
        "v19_capability": run.V19_CAPABILITY, "builder17": run.BUILDER17,
        "builder18": run.BUILDER18}.items()} == run.EXPECTED
    assert run.PRICE == json.loads(run.PRIOR.read_text())["frozen_design"]["price"]
    assert json.loads(run.V19_CAPABILITY.read_text())["causal_outcomes_opened"] is False


def test_no_coefficient_fit_or_v19_builder_import():
    source = run.Path(run.__file__).read_text()
    assert "Adam" not in source and "linalg.svd" not in source
    assert "fresh_lexicon_v19 as" not in source
    assert run.PRICE["fit_parameters"] == run.PRICE["model_updates"] == 0
