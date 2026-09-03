import importlib.util
from pathlib import Path
import sys

import torch


OPS = Path(__file__).resolve().parent
for path in (OPS, OPS.parent, OPS.parent.parent / "polynomial_causal", OPS.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location(
    "rung513", OPS / "attention11_mlp11_exact_factor_interactions_rung513.py")
rung = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rung)


def test_fixed_vocabulary_relations_and_price():
    assert rung.FACTOR_NAMES == ("Q", "K", "Q2", "K2", "V")
    assert len(rung.ATTENTION_TERMS) == 31
    assert rung.MLP_TERMS == ("M11{L}", "M11{R}", "M11{L,R}")
    assert len(rung.TERM_NAMES) == 34
    assert len(rung.SELECTED_SUBSETS) == 6
    assert rung.RELATION_NAMES == ("N-Z7", "N-Z8", "P-Z7")
    assert 6 * 34 == 204
    assert 6 * 34 * 3 == 612
    assert 4216 + 1798 + 620 * 204 == 132494


def test_boolean_mobius_inversion_is_exact_on_five_linear_toy():
    baseline = (2., 3., 5., 7., 11.)
    intact = (13., 17., 19., 23., 29.)
    corners = {
        mask: torch.tensor([__import__("math").prod(
            intact[index] if mask & (1 << index) else baseline[index]
            for index in range(5))])
        for mask in range(32)
    }
    terms = rung.mobius_terms(corners)
    assert len(terms) == 31
    torch.testing.assert_close(sum(terms), corners[31] - corners[0], rtol=1e-6, atol=1e-2)


def test_planted_group_requires_all_three_relations():
    candidates, summary = rung.discover_groups(rung._toy_collection())
    planted = [row for row in candidates
               if row["selected_subset"] == 0 and row["term_index"] == 0]
    assert summary["fixed_groups"] == 204
    assert summary["relation_term_tests"] == 612
    assert len(planted) == 1
    assert set(planted[0]["relations"]) == set(rung.RELATION_NAMES)
    assert all(row["holds"] for row in planted[0]["relations"].values())


def test_confirmation_keeps_discovery_scales_frozen():
    toy = rung._toy_collection()
    candidates, _ = rung.discover_groups(toy)
    planted = next(row for row in candidates
                   if row["selected_subset"] == 0 and row["term_index"] == 0)
    frozen = {name: row["beta_left_from_right"]
              for name, row in planted["relations"].items()}
    confirmed, checks = rung.confirm_groups(toy, [planted])
    assert len(confirmed) == 1
    key = f"{planted['subset_name']} @ {planted['term_name']}"
    for name, beta in frozen.items():
        assert checks[key]["relations"][name]["metrics"]["beta_left_from_right"] == beta


def test_consumer_term_removal_and_substitution_algebra():
    target = {"a11": torch.tensor([3., 4.]), "m11": torch.tensor([5., 6.])}
    removed = rung._patch_write(target, torch.tensor([1., 2.]), "a11")
    replaced = rung._substitution_write(
        target, torch.tensor([1., 2.]), torch.tensor([2., 1.]), "a11", .5)
    torch.testing.assert_close(removed, torch.tensor([2., 2.]))
    torch.testing.assert_close(replaced, torch.tensor([3., 2.5]))


def test_all_frozen_hashes_and_parent_route_are_pinned():
    for path, expected in rung.HASHES.items():
        assert rung.sha256(path) == expected
    result = __import__("json").loads(rung.R512_RESULT.read_text())
    assert result["analysis"]["discovery_summary"]["candidate_count"] == 0
    assert result["next_step"] == (
        "split_attention11_q_k_q2_k2_value_and_mlp11_left_right_product_finitely")


def test_dry_run_opens_no_model_outcome(capsys):
    rung.dry_run()
    output = capsys.readouterr().out
    assert '"model_loaded": false' in output
    assert '"outcomes_opened": false' in output
    assert '"planted_group_recovered": true' in output
