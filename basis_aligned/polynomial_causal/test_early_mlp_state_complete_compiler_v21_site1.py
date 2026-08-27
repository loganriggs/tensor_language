from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import early_mlp_state_complete_compiler_v21_site1 as site1


def test_site1_both_complete_banks_freeze_before_selector(monkeypatch) -> None:
    events = []
    ledgers = {
        name: {f"c{i}": {} for i in range(108)}
        for name in ("true_site1", "shuffle_site1")
    }

    def freeze(stage, banks, controls, diagnostics, *, launch_state):
        assert stage == "site1"
        assert {name: len(bank) for name, bank in banks.items()} == {
            "true_site1": 108, "shuffle_site1": 108,
        }
        events.append("freeze")

    def select(stage):
        assert stage == "site1"
        events.append("select")
        return {"selected": {}}

    monkeypatch.setattr(site1.lifecycle, "freeze_preselector_stage", freeze)
    monkeypatch.setattr(site1.lifecycle, "select_frozen_stage", select)
    assert site1.freeze_and_select_site1(
        ledgers, {}, {}, launch_state=object(),
    ) == {"selected": {}}
    assert events == ["freeze", "select"]


def test_site1_rejects_incomplete_bank_before_freeze(monkeypatch) -> None:
    monkeypatch.setattr(
        site1.lifecycle, "freeze_preselector_stage",
        lambda *args, **kwargs: pytest.fail("freeze must not run"),
    )
    with pytest.raises(RuntimeError, match="banks are incomplete"):
        site1.freeze_and_select_site1(
            {"true_site1": {}, "shuffle_site1": {}}, {}, {},
            launch_state=object(),
        )


def test_full_native_control_binds_distinct_parent_context() -> None:
    state = {"grammar": "native", "weight": torch.ones(1)}
    gate = {
        "passed": True,
        "physical_tolerance": 8e-6,
        "physical_max_abs_error": 1e-7,
        "poison_calls": {0: 0, 1: 0, 2: 0},
        "row_ce_max_abs_drift": 1e-7,
    }
    control = site1.full_native_control(
        state, gate, {0: 0, 1: 24, 2: 0}, context="true_site0",
        upstream_state_sha256="a" * 64, validation_identity="b" * 64,
        physical_reference_scale=2.0,
    )
    assert control["context"] == "true_site0"
    assert control["upstream_state_sha256"] == "a" * 64
    assert control["observed"]["capture_call_counters"] == {0: 0, 1: 24, 2: 0}
    assert control["observed"]["scored_arm_call_counters"] == {0: 0, 1: 0, 2: 0}


def test_full_native_control_rejects_failed_gate() -> None:
    with pytest.raises(RuntimeError, match="did not pass"):
        site1.full_native_control(
            {}, {"passed": False}, {}, context="true_site0",
            upstream_state_sha256="a" * 64, validation_identity="b" * 64,
            physical_reference_scale=1.0,
        )


def test_omission_hook_zeros_post_subtraction_c_not_projected_p() -> None:
    # One active basis direction.  p=5 and live mo coefficient=2, so full c=3.
    # Omitting c must return mo unchanged (2 physical units), not mo-2 (p=0 error).
    basis = torch.zeros(1152, 64)
    basis[0, 0] = 1.0
    state0 = {
        "grammar": "constant", "interface": "state_complete_p",
        "family": "fit_mean_control", "bias": torch.zeros(64),
    }
    state1 = {
        "grammar": "constant", "interface": "state_complete_p",
        "family": "fit_mean_control", "bias": torch.tensor([5.0] + [0.0] * 63),
    }
    hook = site1.OmissionAuditHook(
        {0: basis, 1: basis}, {"audit": {0: state0, 1: state1}},
    )
    hook.configure({1: "Q"}, program_name="audit")
    z = torch.zeros(1, 1, 1152)
    mo = torch.zeros_like(z)
    mo[..., 0] = 2.0
    full = hook(1, object(), z, mo)
    hook.omitted_direction = 0
    omitted = hook(1, object(), z, mo)
    assert full[..., 0].item() == 5.0
    assert omitted[..., 0].item() == 2.0


def test_site1_source_never_requests_final_role() -> None:
    source = Path(site1.__file__).read_text()
    tree = ast.parse(source)
    requested = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (
            node.func.attr == "load_roles_and_validate"
        ):
            requested.append(ast.unparse(node))
    assert requested
    assert all("compiler_final_v21" not in call for call in requested)


def test_direction_prediction_uses_average_ranks_for_ties() -> None:
    losses = torch.tensor([1.0, 1.0, 2.0] + list(range(3, 64)), dtype=torch.float64)
    moments = torch.ones(64, dtype=torch.float64)
    errors = losses.clone() * site1.authority.CAUSAL_CAPTURE_COUNT
    result = site1.authority.derive_direction_prediction(
        losses, moments, errors, site1.authority.CAUSAL_CAPTURE_COUNT,
    )
    assert result["spearman_average_rank"] == pytest.approx(1.0)
    assert result["registered_prediction_positive"] is True
