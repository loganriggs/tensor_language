import numpy as np

import broad_mlp_suffix_dealias_v1 as assay


def _role(cost, role="skip7000", documents=5):
    values = {
        name: np.tile(np.asarray(vector, dtype=np.float64), (documents, 1))
        for name, vector in cost.items()
    }
    return assay.RoleArrays(
        role=role, token_count=np.ones(documents), **values,
    )


def test_registry_is_exact_mlp3_through_mlp8_without_attention():
    assert assay.MLP_SUFFIX == tuple(("mlp", layer) for layer in range(3, 9))
    assert len(assay.REQUEST_MASKS) == 8
    assert all(not any(kind == "attn" for kind, _ in mask) for mask in assay.REQUEST_MASKS)


def test_large_prefix_invariant_suffix_synergy_does_not_break_law():
    # e is arbitrary prefix cost.  a and m have early interactions da/dm.  Adding a
    # constant A×M suffix synergy R=9 to every prefix background makes raw synergy
    # huge but Q=R_i-R_0 exactly zero, so the registered prediction must be exact.
    e = np.arange(8, dtype=np.float64)
    da = np.asarray([0, 1, -2, 3, -4, 5, -6, 7], dtype=np.float64)
    dm = np.asarray([0, -2, 4, -6, 8, -10, 12, -14], dtype=np.float64)
    a = e + 2.0 + da
    m = e + 3.0 + dm
    am = a + m - e + 9.0
    contrast = assay.contrasts({"e": e, "a": a, "m": m, "am": am})
    assert np.allclose(contrast["r"], 9.0)
    assert np.allclose(contrast["q"], 0.0)
    assert np.allclose(contrast["prediction"], contrast["d_m"])
    assert assay.decision_metrics(contrast)["nre"] == 0.0


def test_prediction_error_is_exact_three_way_contrast():
    rng = np.random.default_rng(3)
    cost = {name: rng.normal(size=8) for name in ("e", "a", "m", "am")}
    contrast = assay.contrasts(cost)
    assert np.allclose(
        np.asarray(contrast["prediction"]) - np.asarray(contrast["d_m"]),
        contrast["q"],
    )


def test_descriptive_r_metrics_exclude_the_baseline_cell():
    d_m = np.arange(8, dtype=np.float64)
    contrast = {
        "d_m": d_m,
        "prediction": d_m.copy(),
        "r": np.asarray([10_000.0, *([3.0] * 7)]),
        "q": np.zeros(8),
    }
    metrics = assay.decision_metrics(contrast)
    assert np.isclose(metrics["norms"]["r"], np.sqrt(7 * 9))
    assert metrics["cosines"]["d_m_r"] == assay.descriptive_cosine(
        d_m[1:], np.full(7, 3.0),
    )


def test_zero_q_cosine_is_descriptive_undefined_not_decision_failure():
    e = np.arange(8, dtype=np.float64)
    a = e + np.asarray([1, 2, 4, 8, 16, 32, 64, 128])
    m = e + np.asarray([3, 5, 7, 11, 13, 17, 19, 23])
    am = a + m - e + 6
    metrics = assay.decision_metrics(assay.contrasts({"e": e, "a": a, "m": m, "am": am}))
    assert metrics["cosines"]["d_m_q"] == "undefined_zero_norm"
    assert metrics["nre"] == 0


def test_document_bootstrap_is_deterministic_and_passes_exact_law():
    e = np.arange(8, dtype=np.float64)
    a = e + np.asarray([1, 2, 4, 8, 16, 32, 64, 128])
    m = e + np.asarray([3, 5, 7, 11, 13, 17, 19, 23])
    am = a + m - e + 6
    data = _role({"e": e, "a": a, "m": m, "am": am})
    first = assay.score_role(data, draws=20)
    second = assay.score_role(data, draws=20)
    assert first == second
    assert first["useful_pass"]


def test_zero_interaction_denominator_fails_closed():
    zero = np.zeros(8)
    data = _role({"e": zero, "a": zero, "m": zero, "am": zero})
    score = assay.score_role(data, draws=5)
    assert not score["useful_pass"]
    assert not score["gates"]["finite_decision_metrics"]


def test_cross_role_bootstrap_conditions_on_source_point():
    base = np.arange(8, dtype=np.float64)
    cost = {"e": base, "a": base + 2, "m": base + np.arange(8), "am": base + 2 + np.arange(8)}
    source = _role(cost, "skip7000", documents=4)
    target = _role(cost, "skip11000", documents=7)
    result = assay.score_cross_role(source, target, draws=10)
    assert result["source_prediction_fixed_at_point"] is True
    assert result["bootstrap"]["draw_count"] == 10
    assert result["useful_pass"]
