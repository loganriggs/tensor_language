import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factorial


def test_intervention_clamps_absolute_factor_responses(monkeypatch):
    import torch

    changed8 = torch.arange(2 * 3 * 9 * 4).reshape(2, 3, 9, 4)
    changed15 = changed8 + 1000
    observed = {}

    def run_patch(backend, batch, cache, support):
        observed.update(backend=backend, batch=batch, cache=cache, support=support)
        return "captured-output"

    monkeypatch.setattr(factorial.greedy, "run_patch", run_patch)
    output = factorial.intervene_with_state_capture("backend", "batch", (
        {"layer": 8, "selected_heads": (1,), "changed_capture": {"head_output": changed8}},
        {"layer": 15, "selected_heads": tuple(range(9)),
         "changed_capture": {"head_output": changed15}},
    ))

    assert output == "captured-output"
    assert observed["support"] == ["L8H1", "attn:15"]
    assert observed["cache"]["head_layer:8"].shape == (2, 3, 36)
    assert observed["cache"]["attn:15"].shape == (2, 3, 36)
    assert torch.equal(observed["cache"]["head_layer:8"], changed8.reshape(2, 3, 36))
    assert torch.equal(observed["cache"]["attn:15"], changed15.reshape(2, 3, 36))


def test_registered_parent_report_drops_only_legacy_pooled_control():
    report = {
        "targets": {"A1": {"behavior": 1}, "A2": {"behavior": 2}},
        "controls": {"P": {"kl": 3}, "C": {"kl": 4}, "pooled": {"kl": 5}},
    }

    assert factorial.registered_parent_report(report) == {
        "targets": report["targets"],
        "controls": {"P": {"kl": 3}, "C": {"kl": 4}},
    }


def test_replay_comparison_keeps_categorical_mismatch_finite_and_fail_closed():
    comparison = factorial.replay_comparison(
        {"score": 1.0, "flipped": ["row-a"]},
        {"score": 1.000003, "flipped": ["row-b"]},
    )

    assert comparison["numeric_schema_match"] is True
    assert comparison["categorical_match"] is False
    assert 0.0 < comparison["numeric_max_abs_error"] < 1e-4
