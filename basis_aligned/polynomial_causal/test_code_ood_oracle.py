import importlib.util
import json
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("code_ood_oracle.py")
SPEC = importlib.util.spec_from_file_location("code_ood_oracle", PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORACLE)


def test_fineweb_license_is_fail_closed_and_gate_coherent(tmp_path):
    path = tmp_path / "fineweb.json"
    with pytest.raises(RuntimeError, match="absent"):
        ORACLE.load_fineweb_license(path)

    gates = {
        "full_oracle_ci95_lower_gt_zero": True,
        "content_positive_both_splits": True,
        "content_beats_matched_null95_heldout": True,
    }
    path.write_text(json.dumps({
        "config": {"status": "authoritative_frozen_ship_v2",
                   "ship_realization_sha256": "a" * 64},
        "training_license_sites": [1], "site_decisions": {"1": gates}
    }))
    sites, _ = ORACLE.load_fineweb_license(path)
    assert sites == [1]

    gates["content_positive_both_splits"] = False
    path.write_text(json.dumps({
        "config": {"status": "authoritative_frozen_ship_v2",
                   "ship_realization_sha256": "a" * 64},
        "training_license_sites": [1], "site_decisions": {"1": gates}
    }))
    with pytest.raises(RuntimeError, match="disagrees"):
        ORACLE.load_fineweb_license(path)


def test_token_conditional_basis_removes_token_means():
    tokens = torch.tensor([[10, 10, 11, 11], [10, 10, 11, 11]])
    sign = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    direction = torch.tensor([0.0, 1.0, 0.0, 0.0])
    token_means = {10: torch.tensor([100.0, 0.0, 0.0, 0.0]),
                   11: torch.tensor([-100.0, 0.0, 0.0, 0.0])}
    base = torch.stack([
        token_means[int(token)] + sign[index] * direction
        for index, token in enumerate(tokens.reshape(-1))
    ]).view(2, 4, 4)
    layers = {
        layer: base + torch.tensor([0.0, 0.0, float(offset), 0.0])
        for offset, layer in enumerate(ORACLE.CONTENT_LAYERS)
    }
    basis = ORACLE.token_conditional_content_basis(layers, tokens, rank=1)
    assert abs(float(basis[:, 0] @ direction)) > 0.999
    assert abs(float(basis[0, 0])) < 1e-5


def test_ship_tree_hash_is_key_order_invariant_and_value_sensitive():
    tensor = torch.arange(6).view(2, 3)
    left = {"b": [tensor, 2], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [tensor.clone(), 2]}
    changed = {"a": {"x": True}, "b": [tensor + 1, 2]}
    assert ORACLE.tensor_tree_sha256(left) == ORACLE.tensor_tree_sha256(right)
    assert ORACLE.tensor_tree_sha256(left) != ORACLE.tensor_tree_sha256(changed)


def test_projection_nulls_share_directions_and_match_each_content_rms():
    generator = torch.Generator().manual_seed(7)
    residual = torch.randn(80, 8, generator=generator)
    prose = torch.linalg.qr(torch.randn(8, 2, generator=generator)).Q
    code = torch.linalg.qr(torch.randn(8, 2, generator=generator)).Q
    arms = ORACLE.build_projection_arms(
        residual, prose, code, site=0, rank=2, support_rank=6, nulls=3
    )
    for index in range(3):
        prose_null = arms[f"prose_content_null_{index:02d}"]
        code_null = arms[f"code_content_null_{index:02d}"]
        assert torch.equal(prose_null["basis"], code_null["basis"])
        gram = prose_null["basis"].T @ prose_null["basis"]
        assert torch.allclose(gram, torch.eye(2), atol=1e-5, rtol=1e-5)
        assert prose_null["fit_correction_rms"] == pytest.approx(
            arms["prose_content"]["fit_correction_rms"], rel=1e-6
        )
        assert code_null["fit_correction_rms"] == pytest.approx(
            arms["code_content"]["fit_correction_rms"], rel=1e-6
        )


def test_lexical_table_uses_token_means_and_global_unseen_fallback():
    residual = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 0.0]])
    tokens = torch.tensor([2, 2, 4])
    table = ORACLE.fit_lexical_residual_table(residual, tokens, vocab=6)
    assert torch.equal(table[2], torch.tensor([2.0, 4.0]))
    assert torch.equal(table[4], torch.tensor([10.0, 0.0]))
    assert torch.allclose(table[0], residual.mean(0))


def score(row_sums, counts):
    return {
        "ce": {"global": sum(row_sums) / sum(counts)},
        "row_sums": {"global": row_sums},
        "row_counts": {"global": counts},
    }


def test_file_cluster_bootstrap_preserves_signed_paired_gain():
    baseline = score([10.0, 10.0, 20.0], [10, 10, 10])
    better = score([9.0, 9.0, 18.0], [10, 10, 10])
    worse = score([11.0, 11.0, 22.0], [10, 10, 10])
    clusters = ["a.py", "a.py", "b.py"]
    positive = ORACLE.paired_cell_gain(
        baseline, better, "global", clusters, seed=1, draws=200
    )
    negative = ORACLE.paired_cell_gain(
        baseline, worse, "global", clusters, seed=1, draws=200
    )
    assert positive["mean"] == pytest.approx(4 / 30)
    assert negative["mean"] == pytest.approx(-4 / 30)
    assert positive["clusters"] == 2
    assert positive["ci95"][0] > 0


def test_fraction_of_full_is_undefined_when_cluster_ci_crosses_zero():
    baseline = score([10.0, 10.0], [10, 10])
    arm = score([9.0, 9.0], [10, 10])
    unstable_full = score([8.0, 12.0], [10, 10])
    result = ORACLE.bootstrap_fraction_of_full(
        baseline, arm, unstable_full, ["a.py", "b.py"], seed=2, draws=500
    )
    assert result["status"] == "undefined_full_gain_ci_includes_zero"
    assert result["ci95"] == [None, None]


def test_exact_twenty_null_test_and_classification_truth_table():
    passed = ORACLE.exact_null_test(1.0, [0.0] * 20)
    tied = ORACLE.exact_null_test(1.0, [0.0] * 19 + [1.0])
    assert passed == {
        "content_gain": 1.0,
        "null_gains": [0.0] * 20,
        "nulls_at_least_content": 0,
        "exact_one_sided_p": pytest.approx(1 / 21),
        "passes_5pct": True,
    }
    assert tied["passes_5pct"] is False
    assert tied["exact_one_sided_p"] == pytest.approx(2 / 21)

    def arm(mean, lower=None):
        return {"global": {"mean": mean, "ci95": [mean - 0.01 if lower is None else lower,
                                                   mean + 0.01]}}

    gains = {
        split: {
            "full": arm(0.30, 0.20),
            "prose_content": arm(0.08),
            "code_content": arm(0.09),
            "local_pca": arm(0.15, 0.10),
            "lexical_mean": arm(0.05, 0.03),
        }
        for split in ("discovery", "heldout")
    }
    nulls = {name: ORACLE.exact_null_test(gains["heldout"][name]["global"]["mean"],
                                          [0.0] * 20)
             for name in ("prose_content", "code_content")}
    shared = ORACLE.classify_site(
        gains, nulls, {"mean": 0.01, "ci95": [0.0, 0.019]}
    )
    typed = ORACLE.classify_site(
        gains, nulls, {"mean": 0.04, "ci95": [0.021, 0.06]}
    )
    ambiguous = ORACLE.classify_site(
        gains, nulls, {"mean": 0.02, "ci95": [-0.01, 0.05]}
    )
    assert shared["classification"] == "shared prose coordinate"
    assert typed["classification"] == "domain-typed coordinate"
    assert ambiguous["classification"] == "inconclusive content coordinate"


def test_standalone_entry_refuses_ship_reconstruction():
    with pytest.raises(SystemExit, match="Independent execution is forbidden"):
        ORACLE.main()
