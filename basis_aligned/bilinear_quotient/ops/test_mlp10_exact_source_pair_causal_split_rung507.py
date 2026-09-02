from types import SimpleNamespace

import torch

import mlp10_exact_source_pair_causal_split_rung507 as rung


def test_named_sources_and_unordered_pairs_are_complete():
    assert rung.NAMED_SOURCES == (
        "E", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "A9", "A10", "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
        "M8", "M9",
    )
    assert len(rung.SOURCE_PAIRS) == 22 * 23 // 2 == 253
    assert len(set(rung.SOURCE_PAIRS)) == 253
    assert all(left <= right for left, right in rung.SOURCE_PAIRS)


def test_residual_coefficients_match_direct_mlp10_recurrence():
    blocks = [SimpleNamespace(lambdas=torch.tensor([.81 + site / 100, .17]))
              for site in range(rung.TARGET + 1)]
    model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
    embedding, writes = rung._source_coefficients(model)
    coefficients = {"E": torch.tensor(1.0)}
    for site, block in enumerate(blocks):
        coefficients = {name: block.lambdas[0] * value
                        for name, value in coefficients.items()}
        coefficients["E"] += block.lambdas[1]
        coefficients[f"A{site}"] = torch.tensor(1.0)
        if site < rung.TARGET:
            coefficients[f"M{site}"] = torch.tensor(1.0)
    torch.testing.assert_close(embedding, coefficients["E"])
    for site in range(rung.TARGET + 1):
        torch.testing.assert_close(writes[site], coefficients[f"A{site}"])
        if site < rung.TARGET:
            torch.testing.assert_close(writes[site], coefficients[f"M{site}"])


def test_unordered_pair_terms_sum_to_the_full_named_bilinear_product():
    torch.manual_seed(507)
    factors = {
        "left": torch.randn(2, 3, len(rung.NAMED_SOURCES), 7),
        "right": torch.randn(2, 3, len(rung.NAMED_SOURCES), 7),
    }
    observed = rung._sum_unordered_pair_hidden(factors)
    expected = factors["left"].sum(2) * factors["right"].sum(2)
    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)


def test_unordered_gradient_contractions_match_explicit_terms():
    torch.manual_seed(508)
    factors = {
        "left": torch.randn(2, 3, len(rung.NAMED_SOURCES), 7),
        "right": torch.randn(2, 3, len(rung.NAMED_SOURCES), 7),
    }
    reader = torch.randn(2, 3, 7)
    observed = rung._unordered_contraction(reader, factors)
    expected = torch.stack([
        (reader * rung._pair_hidden(factors, index)).sum()
        for index in range(len(rung.SOURCE_PAIRS))
    ]).double()
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-5)


def _finite_fixture():
    arms = ("intact", rung.PAIR_NAMES[0], rung.PAIR_NAMES[1],
            f"{rung.PAIR_NAMES[0]}+{rung.PAIR_NAMES[1]}")
    task = torch.zeros(len(rung.SOURCES), len(arms), 4, len(rung.TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.ones(4, len(rung.TASK_CELLS), dtype=torch.float64)
    left = torch.tensor([.004, .003, .005, .002], dtype=torch.float64)
    right = torch.tensor([.002, .004, .003, .005], dtype=torch.float64)
    context_indices = [rung.TASK_CELLS.index(cell) for cell in rung.GRAD_CELLS[:4]]
    all_index = rung.TASK_CELLS.index("all_positive")
    off_index = rung.TASK_CELLS.index("off_target")
    for source_index in range(len(rung.SOURCES)):
        for document in range(4):
            task[source_index, 1, document, context_indices] = left
            task[source_index, 2, document, context_indices] = right
            task[source_index, 3, document, context_indices] = left + right
            task[source_index, 1, document, all_index] = .004
            task[source_index, 2, document, all_index] = .004
            task[source_index, 3, document, all_index] = .008
            task[source_index, 1:, document, off_index] = .0001
    return {
        "bounds": (0, 4, 2), "arms": arms, "task": task,
        "task_counts": counts,
    }


def test_additive_finite_composition_is_frozen_then_predicted():
    data = _finite_fixture()
    left, right = rung.PAIR_NAMES[:2]
    rule = rung.fit_composition(data, data, left, right)
    assert rule["identified"] is True
    assert rule["kind"] == "additive"
    scored = rung.score_composition(data, left, right, rule)
    assert scored["holds"] is True


def test_instrument_rejects_even_one_zero_term_edit():
    diagnostics = {
        "calls_exact": True,
        "factor_reconstruction_max": 0.0,
        "raw_source_relative_squared": 0.0,
        "normalized_closure_relative_squared": 0.0,
        "normalized_numerical_rms_ratio": 0.0,
        "float32_mlp10_closure": 0.0,
        "deployed_mlp10_relative_squared": 0.0,
        "score_delta_float32_closure": 0.0,
        "score_delta_predeployment_relative_squared": 0.016,
        "score_delta_deployed_closure_relative_squared": 0.0,
        "minimum_nonzero_score_edit_rms": 1.0,
        "term_patches_exact": True,
        "term_patches": 2,
        "minimum_nonzero_term_edit_rms": 1.0,
        "zero_term_edits": 1,
    }
    assert rung._phase_instrument({"diagnostics": diagnostics}) is False


def test_output_rounding_remainder_closes_deployed_score_change_exactly():
    torch.manual_seed(509)
    width, hidden = 5, 7
    mlp = SimpleNamespace(
        Left=SimpleNamespace(weight=torch.randn(hidden, width)),
        Right=SimpleNamespace(weight=torch.randn(hidden, width)),
        Down=SimpleNamespace(weight=torch.randn(width, hidden)),
        Down_bias=torch.randn(width),
    )

    def capture():
        sources = torch.randn(1, 2, len(rung.NAMED_SOURCES), width)
        numerical = .001 * torch.randn(1, 2, width)
        left = torch.nn.functional.linear(sources.sum(2) + numerical, mlp.Left.weight)
        right = torch.nn.functional.linear(sources.sum(2) + numerical, mlp.Right.weight)
        independent = torch.nn.functional.linear(left * right, mlp.Down.weight) + mlp.Down_bias
        deployed = independent + .01 * torch.randn_like(independent)
        factors = rung._source_factors(mlp, sources, numerical, deployed)
        return {
            "factors": factors, "deployed_write": deployed,
            "numerical_output": factors["numerical_output"],
        }

    absent, current = capture(), capture()
    diagnostics = rung._empty_diagnostics()
    rung._score_delta_closure(diagnostics, current, absent)
    assert diagnostics["score_delta_float32_closure"] < 1e-12
    assert diagnostics["score_delta_deployed_closure_relative_squared"] < 1e-12
    assert diagnostics["score_delta_predeployment_relative_squared"] > 0


def test_registered_maximum_price_is_literal():
    assert 1369 + 248 * 8 + 500 * 8 + 748 * 28 == 28297
