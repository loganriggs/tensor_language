import importlib.util
from pathlib import Path
import sys

import torch


OPS = Path(__file__).resolve().parent
for path in (OPS, OPS.parent, OPS.parent.parent / "polynomial_causal", OPS.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location(
    "rung514", OPS / "attention11_mlp11_constrained_multi_term_programs_rung514.py")
rung = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(rung)


def test_registered_bank_counts_and_price():
    assert len(rung.fixed_factor_programs("a11")) == 6
    assert len(rung.fixed_factor_programs("m11")) == 2
    assert len(rung.canonical_sparse_programs("a11")) == 18910
    assert len(rung.canonical_sparse_programs("m11")) == 10
    assert sum(len(rows) for rows in rung.PROGRAMS.values()) * 6 == 113568
    assert 4216 + 1798 + 620 * 32 == 25854


def test_fixed_shapley_allocations_close_exactly():
    attention = rung.fixed_factor_programs("a11")
    torch.testing.assert_close(
        sum(row["coefficient"] for row in attention[:5]),
        torch.ones(31, dtype=torch.float64))
    torch.testing.assert_close(
        attention[-1]["coefficient"],
        attention[0]["coefficient"] + attention[2]["coefficient"]
        + attention[4]["coefficient"])
    mlp = rung.fixed_factor_programs("m11")
    torch.testing.assert_close(
        sum(row["coefficient"] for row in mlp), torch.ones(3, dtype=torch.float64))


def test_sparse_signs_are_canonical_and_support_is_two_or_three():
    for site in ("a11", "m11"):
        for row in rung.canonical_sparse_programs(site):
            assert len(row["support"]) in (2, 3)
            assert row["signs"][0] == 1
            assert set(row["signs"]) <= {-1, 1}
            assert row["support"] == sorted(row["support"])


def test_sparse_program_gram_matches_direct_construction():
    generator = torch.Generator().manual_seed(514)
    values = torch.randn(4, 3, 40, generator=generator).double()
    joint = values.reshape(12, 40) @ values.reshape(12, 40).T
    programs = rung.canonical_sparse_programs("m11")
    got = rung.bank_program_grams(joint, programs, 3)
    for index, program in enumerate(programs):
        response = torch.einsum("t,atd->ad", program["coefficient"], values)
        expected = response @ response.T
        torch.testing.assert_close(got[index], expected)


def test_control_permutation_preserves_within_action_gram():
    generator = torch.Generator().manual_seed(515)
    values = torch.randn(12, 50, generator=generator)
    permuted = rung._permuted_site_vectors(values, 3, 51410, 500)
    for action in range(4):
        block = slice(3 * action, 3 * (action + 1))
        torch.testing.assert_close(values[block] @ values[block].T,
                                   permuted[block] @ permuted[block].T)
    assert not torch.allclose(values[:3] @ values[3:6].T,
                              permuted[:3] @ permuted[3:6].T)


def test_all_eight_planted_programs_are_uniquely_recovered():
    result = rung.planted_recovery_suite()
    assert result["all_exact_unique_recoveries"]
    assert len(result["cases"]) == 8
    assert all(row["recovered_program_indices"] == [row["expected_program_index"]]
               for row in result["cases"])


def test_program_tensor_uses_exact_registered_coefficients():
    candidate = rung._toy_physical_candidate()
    terms = tuple(torch.tensor([float(index)]) for index in range(34))
    expected = sum(terms[index] * candidate["coefficient"][index]
                   for index in range(31) if candidate["coefficient"][index] != 0)
    torch.testing.assert_close(rung.program_tensor(terms, candidate), expected)


def test_all_frozen_hashes_and_parent_routes_are_pinned():
    for path, expected in rung.HASHES.items():
        assert rung.sha256(path) == expected
    result = __import__("json").loads(rung.R513_RESULT.read_text())
    assert result["analysis"]["discovery_summary"]["candidate_count"] == 0
    mismatch = __import__("json").loads(rung.MISMATCH_RESULT.read_text())
    assert mismatch["pred_c_stable_dominant_factor_subspace"] is True


def test_dry_run_opens_no_model_outcome(capsys):
    rung.dry_run()
    output = capsys.readouterr().out
    assert '"model_loaded": false' in output
    assert '"outcomes_opened": false' in output
    assert '"all_planted_supports_uniquely_recovered": true' in output
