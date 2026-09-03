import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp0_one_circuit_interaction_atlas_rung519.py")
SPEC = importlib.util.spec_from_file_location("r519", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class Linear:
    def __init__(self, weight):
        self.weight = weight


class ToyMLP:
    def __init__(self, width=7, hidden=11):
        generator = torch.Generator().manual_seed(519)
        self.Left = Linear(torch.randn(hidden, width, generator=generator))
        self.Right = Linear(torch.randn(hidden, width, generator=generator))
        self.Down = Linear(torch.randn(width, hidden, generator=generator))
        self.Down_bias = torch.randn(width, generator=generator)

    def deployed(self, state):
        return R._float_mlp(self, state)


def test_term_names_cover_47_bilinear_and_two_closing_terms():
    atoms = tuple(f"H{head}.{group}" for head in range(9)
                  for group in ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER"))
    names = R.term_names(atoms)
    assert len(names) == 49
    assert names[1 + R.SELECTED_ATOM] == "SELF"
    assert names[R.NORMALIZATION_TERM] == "NORMALIZATION"
    assert names[R.DEPLOYMENT_ROUNDING_TERM] == "DEPLOYMENT_ROUNDING"


def test_normalized_sources_close_with_numerical_residual():
    generator = torch.Generator().manual_seed(520)
    token = torch.randn(2, 3, 7, generator=generator)
    atoms = torch.randn(45, 2, 3, 7, generator=generator)
    normalized = torch.randn(2, 3, 7, generator=generator)
    sources = R.normalized_sources(token, atoms, normalized)
    assert sources.shape == (47, 2, 3, 7)
    assert torch.allclose(sources.sum(0), normalized)


def test_interaction_terms_exactly_close_fixed_gain_and_deployed_drop():
    generator = torch.Generator().manual_seed(521)
    mlp = ToyMLP()
    sources = torch.randn(47, 2, 3, 7, generator=generator)
    full = sources.sum(0)
    selected = 1 + R.SELECTED_ATOM
    raw_drop = full - 1.17 * sources[selected]
    deployed_full = mlp.deployed(full)
    deployed_drop = mlp.deployed(raw_drop) + .001
    result = R.interaction_terms(
        mlp, sources, raw_drop, deployed_full, deployed_drop)
    assert result["terms"].shape == (49, 2, 3, 7)
    assert result["fixed_gain_relative_squared"] < 1e-12
    assert result["deployed_relative_squared"] < 1e-12
    torch.testing.assert_close(
        result["terms"].sum(0), result["target"], rtol=5e-5, atol=5e-4)


def test_discovery_recovers_only_semantic_planted_terms():
    effects, whole, expected = R.planted_problem(51900)
    found = [row["term"] for row in R.discover_terms(effects, whole, 0)]
    assert found == expected
    assert R.NUMERICAL_SOURCE not in found
    assert R.NORMALIZATION_TERM not in found
    assert R.DEPLOYMENT_ROUNDING_TERM not in found


def test_permutations_destroy_planted_target_alignment():
    effects, whole, _ = R.planted_problem(51901)
    assert R.permutation_control_counts(effects, whole, 0) == [0] * 16


def test_mobius_recovers_known_interactions():
    coefficients = torch.tensor([0., .2, -.1, .4, .3, 0., 0., -.25])
    table = torch.zeros(8)
    for mask in range(8):
        table[mask] = sum(coefficients[sub]
                          for sub in range(8) if sub & ~mask == 0)
    assert torch.allclose(R.mobius(table), coefficients.double(), atol=1e-7)


def test_r518_selection_is_hash_pinned_and_mechanical():
    validated = R.validate_inputs()
    assert validated["selected"]["atom"] == R.SELECTED_ATOM
    assert validated["selected"]["name"] == R.SELECTED_ATOM_NAME


def test_dry_run_opens_no_model_or_outcome():
    result = R.dry_run()
    assert result["model_loaded"] is False
    assert result["model_outcomes_opened"] is False
    assert result["planted_recovery"]["all_eight_exact"]
