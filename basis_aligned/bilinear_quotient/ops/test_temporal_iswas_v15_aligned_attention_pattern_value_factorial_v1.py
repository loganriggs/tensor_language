from types import SimpleNamespace

import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factorial


def test_intervention_requests_state_capture_and_restores_native(monkeypatch):
    calls = []

    def native(batch, *, capture):
        calls.append((batch, capture))
        return SimpleNamespace(captured={"final": True})

    backend = SimpleNamespace(native=native)

    def intervention(observed_backend, batch, specs):
        assert specs == ("frozen",)
        return observed_backend.native(batch, capture=False)

    monkeypatch.setattr(
        factorial.attention_eval,
        "intervene_ordered_head_output_deltas",
        intervention,
    )
    original = backend.native
    output = factorial.intervene_with_state_capture(
        backend, "batch", ("frozen",)
    )

    assert output.captured == {"final": True}
    assert calls == [("batch", True)]
    assert backend.native is original


def test_registered_parent_report_drops_only_legacy_pooled_control():
    report = {
        "targets": {"A1": {"behavior": 1}, "A2": {"behavior": 2}},
        "controls": {"P": {"kl": 3}, "C": {"kl": 4}, "pooled": {"kl": 5}},
    }

    assert factorial.registered_parent_report(report) == {
        "targets": report["targets"],
        "controls": {"P": {"kl": 3}, "C": {"kl": 4}},
    }
