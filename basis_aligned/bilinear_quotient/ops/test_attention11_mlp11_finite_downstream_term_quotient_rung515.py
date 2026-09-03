import importlib.util
from pathlib import Path
import sys

import torch


OPS = Path(__file__).resolve().parent
for path in (OPS, OPS.parent, OPS.parent.parent / "polynomial_causal", OPS.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location(
    "rung515", OPS / "attention11_mlp11_finite_downstream_term_quotient_rung515.py")
rung = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rung)


def test_registered_node_pair_and_price_counts():
    assert rung.N_SUBSETS * rung.N_ACTIONS * rung.N_TERMS == 816
    assert rung.PAIR_COUNT == 6 * 3 * (31 ** 2 + 3 ** 2) == 17460
    assert len(rung.CONTROL_SEEDS) == 16
    assert 2 * 52452 + 1860 + 124 * 16 == 108748


def test_pair_space_allows_different_terms_but_not_different_sites():
    assert "A11{Q}" in rung.pair_name(0, 0, "a11", 0, 1)
    assert "A11{K}" in rung.pair_name(0, 0, "a11", 0, 1)
    assert sum(len(names) ** 2 for names in rung.SITE_TERMS.values()) == 970


def test_all_eight_planted_pairs_are_uniquely_recovered():
    result = rung.planted_recovery_suite()
    assert result["all_exact_unique_recoveries"]
    assert len(result["cases"]) == 8
    assert all(len(row["observed"]) == 1 and row["holds"] for row in result["cases"])
    assert any(row["expected"][3] != row["expected"][4] for row in result["cases"])


def test_fitted_scalar_and_bidirectional_metrics_recover_exact_pair():
    generator = torch.Generator().manual_seed(515)
    matrices = {}
    for window in ("half0", "half1", "pooled"):
        matrices[window] = {
            "circuit": .002 * torch.randn(
                rung.N_SUBSETS, rung.N_ACTIONS, rung.N_TERMS, 32,
                generator=generator, dtype=torch.float64),
            "task": .002 * torch.randn(
                rung.N_SUBSETS, rung.N_ACTIONS, rung.N_TERMS, 4,
                generator=generator, dtype=torch.float64),
        }
    left_action, right_action = rung.r513.RELATION_ACTIONS[0]
    for window in matrices:
        for kind in ("circuit", "task"):
            donor = matrices[window][kind][0, right_action, 1]
            matrices[window][kind][0, left_action, 0] = -2 * donor
    pairs, _summary = rung.discover_pairs(matrices)
    match = [row for row in pairs if (row["subset"], row["relation"], row["site"],
                                      row["left_term"], row["right_term"])
             == (0, 0, "a11", 0, 1)]
    assert len(match) == 1
    assert abs(match[0]["beta_left_from_right"] + 2) < 1e-10
    assert match[0]["holds"]


def test_control_permutation_is_shared_within_an_action():
    generator = torch.Generator().manual_seed(516)
    values = torch.randn(rung.N_SUBSETS, rung.N_TERMS, 32, generator=generator)
    order = torch.randperm(32, generator=torch.Generator().manual_seed(51511 * 10 + 2))
    permuted = values[:, :, order]
    original_gram = values[0, :5] @ values[0, :5].T
    permuted_gram = permuted[0, :5] @ permuted[0, :5].T
    torch.testing.assert_close(original_gram, permuted_gram)


def test_all_frozen_hashes_and_parent_route_are_pinned():
    for path, expected in rung.HASHES.items():
        assert rung.sha256(path) == expected
    result = __import__("json").loads(rung.R514_RESULT.read_text())
    assert result["pred_b_constrained_multi_term_discovery"] is False
    assert result["analysis"]["discovery_summary"]["real"]["counts"]["accepted"] == 0


def test_dry_run_opens_no_model_outcome(capsys):
    rung.dry_run()
    output = capsys.readouterr().out
    assert '"model_loaded": false' in output
    assert '"outcomes_opened": false' in output
    assert '"all_planted_pairs_uniquely_recovered": true' in output
