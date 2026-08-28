import pytest
import torch

import early_mlp_suffix_transport_v1_final_actions as actions
import early_mlp_suffix_transport_v1_final_capability as capability
import early_mlp_suffix_transport_v1_runtime as runtime


def _program(route: str, marker: float) -> runtime.JointAffineProgram:
    def site(value: float) -> runtime.AffineCodeProgram:
        weight = torch.zeros(runtime.D_MODEL, runtime.CODE_DIM)
        weight[0, 0] = value
        return runtime.AffineCodeProgram(
            mean=torch.zeros(runtime.D_MODEL), scale=torch.ones(runtime.D_MODEL),
            weight=weight, bias=torch.full((runtime.CODE_DIM,), value / 100),
        )

    value = runtime.JointAffineProgram(site(marker), site(marker + 0.5), route=route)
    if route == "T":
        with torch.no_grad():
            value.cross.fill_(marker / 1000)
    return value


def _sources() -> tuple[actions.FinalProgramSourceBank, dict[str, float]]:
    expected_routes = {
        "inherited_q": "L", "true/L": "L", "true/R": "R",
        "true/S0": "S0", "true/S1": "S1", "true/T": "T",
        "mapped/document_shuffle/L": "L", "mapped/document_shuffle/R": "R",
        **{f"mapped/A_null_{index:02d}/T": "T" for index in range(20)},
        "new_fit_mean": "L",
    }
    markers = {
        key: float(index + 1) for index, key in enumerate(actions.SOURCE_PROGRAM_KEYS)
    }
    programs = {
        key: _program(expected_routes[key], markers[key])
        for key in actions.SOURCE_PROGRAM_KEYS
    }
    return actions.FinalProgramSourceBank(programs), markers


def test_action_plan_is_complete_ordered_and_shared_with_capability() -> None:
    assert len(actions.BASE_ARMS) == 34
    assert len(actions.CANONICAL_ARM_PLANS) == 34
    assert len(actions.CANONICAL_ACTION_PLANS) == 68
    assert actions.CANONICAL_ACTION_KEYS == capability.CANONICAL_ACTION_KEYS
    assert tuple(
        f"{action.arm}/{action.background}" for action in capability.CANONICAL_ACTIONS
    ) == actions.CANONICAL_ACTION_KEYS


def test_composite_program_sources_are_not_silently_joint_programs() -> None:
    expected = {
        "s0_l1": ("true_s0", "true_l1", "hybrid_s0_l1"),
        "l0_s1": ("true_l0", "true_s1", "hybrid_l0_s1"),
        "r0_l1": ("true_r0", "true_l1", "hybrid_r0_l1"),
        "l0_r1": ("true_l0", "true_r1", "hybrid_l0_r1"),
    }
    for arm, sources in expected.items():
        plan = actions.plan_for(arm, "N").arm_plan
        assert (plan.site0_source, plan.site1_source, plan.identity_control) == sources
        assert plan.execution_kind == "projected_program"


def test_transport_true_zero_and_null_cross_sources_are_distinct() -> None:
    assert actions.plan_for("lt", "N").arm_plan.cross_source == "true_t_cross"
    assert actions.plan_for("zero_a", "N").arm_plan.cross_source == "zero_cross"
    for index in range(20):
        arm = f"a_null_{index:02d}"
        plan = actions.plan_for(arm, "N").arm_plan
        assert plan.cross_source == f"mapped_A_null_{index:02d}_cross"
        assert plan.identity_control == f"A_null_{index:02d}"


def test_baselines_cannot_acquire_projected_program_components() -> None:
    for arm, kind in (("n_n", "deployed_baseline"), ("o_o", "native_baseline")):
        plan = actions.plan_for(arm, "N").arm_plan
        assert plan.execution_kind == kind
        assert all(value is None for value in (
            plan.site0_source, plan.site1_source, plan.cross_source,
            plan.identity_route, plan.identity_control,
        ))


def test_background_controls_primary_and_response_permissions() -> None:
    for arm in actions.BASE_ARMS:
        native = actions.plan_for(arm, "N")
        exact = actions.plan_for(arm, "E")
        assert native.mlp2_source == "deployed_mlp2"
        assert exact.mlp2_source == "exact_mlp2"
        assert native.emits_registered_primary
        assert not exact.emits_registered_primary
        assert not exact.emits_logit_response and not exact.emits_code_response
        assert native.emits_logit_response == (arm in actions.RESPONSE_ARMS)
        assert native.emits_code_response == (arm in actions.CODE_RESPONSE_ARMS)


def test_action_plan_rejects_aliases_and_malformed_baselines() -> None:
    with pytest.raises(ValueError, match="outside the registered lattice"):
        actions.plan_for("RR", "N")
    with pytest.raises(ValueError, match="outside the registered lattice"):
        actions.plan_for("rr", "native")
    with pytest.raises(ValueError, match="acquired a projected program"):
        actions.FinalArmPlan(
            arm="n_n", execution_kind="deployed_baseline", site0_source="true_l0",
            site1_source=None, cross_source=None, identity_route=None,
            identity_control=None,
        )


def test_source_bank_requires_every_source_in_canonical_order() -> None:
    bank, _ = _sources()
    assert len(bank.sha256) == 64
    programs = {key: bank.clone(key) for key in reversed(actions.SOURCE_PROGRAM_KEYS)}
    with pytest.raises(ValueError, match="incomplete or reordered"):
        actions.FinalProgramSourceBank(programs)
    programs = {key: bank.clone(key) for key in actions.SOURCE_PROGRAM_KEYS[:-1]}
    with pytest.raises(ValueError, match="incomplete or reordered"):
        actions.FinalProgramSourceBank(programs)


def test_hybrid_materialization_uses_the_named_site_sources() -> None:
    bank, marker = _sources()
    cases = {
        "s0_l1": ("true/S0", "true/L"),
        "l0_s1": ("true/L", "true/S1"),
        "r0_l1": ("true/R", "true/L"),
        "l0_r1": ("true/L", "true/R"),
    }
    for arm, (site0_key, site1_key) in cases.items():
        materialized = actions.materialize(actions.plan_for(arm, "N"), bank)
        program = materialized.make_program()
        assert float(program.site0.weight[0, 0].detach()) == marker[site0_key]
        assert float(program.site1.weight[0, 0].detach()) == marker[site1_key] + 0.5
        assert materialized.program_sha256 == runtime.program_snapshot_sha256(program)
        assert materialized.plan.key == f"{arm}/N"


def test_transport_materialization_distinguishes_true_zero_and_each_null() -> None:
    bank, marker = _sources()
    true = actions.materialize(actions.plan_for("lt", "N"), bank).make_program()
    zero = actions.materialize(actions.plan_for("zero_a", "N"), bank).make_program()
    assert true.cross is not None and zero.cross is not None
    assert torch.count_nonzero(true.cross) and not torch.count_nonzero(zero.cross)
    assert float(true.cross[0, 0].detach()) == pytest.approx(marker["true/T"] / 1000)
    identities = set()
    for index in range(20):
        arm = f"a_null_{index:02d}"
        value = actions.materialize(actions.plan_for(arm, "N"), bank)
        program = value.make_program()
        key = f"mapped/A_null_{index:02d}/T"
        assert float(program.cross[0, 0].detach()) == pytest.approx(marker[key] / 1000)
        identities.add(value.program_sha256)
    assert len(identities) == 20


def test_baseline_materialization_has_no_program_and_is_action_distinct() -> None:
    bank, _ = _sources()
    deployed = actions.materialize(actions.plan_for("n_n", "N"), bank)
    native = actions.materialize(actions.plan_for("o_o", "N"), bank)
    assert deployed.program_sha256 is None and native.program_sha256 is None
    assert deployed.sha256 != native.sha256
    with pytest.raises(RuntimeError, match="has no projected program"):
        deployed.make_program()


def test_final_action_batch_identity_binds_action_materialization_rows_and_support() -> None:
    bank, _ = _sources()
    rr = actions.materialize(actions.plan_for("rr", "E"), bank)
    rows = torch.arange(4 * 513, dtype=torch.long).view(4, 513).contiguous()
    identity = actions.FinalActionBatchIdentity.from_role_rows(
        materialized=rr, role_rows=rows, ordered_batch_indices=(8, 9, 10, 11),
        batch_ordinal=2, source_commit="1" * 40,
        inherited_snapshot_sha256="2" * 64, rows_receipt_sha256="3" * 64,
        final_role_tensor_sha256="4" * 64, program_payload_sha256="5" * 64,
        common_support_sha256="6" * 64,
    )
    identity.require_role_rows(
        materialized=rr, role_rows=rows, ordered_batch_indices=(8, 9, 10, 11),
    )
    with pytest.raises(RuntimeError, match="differs from its sealed identity"):
        identity.require_role_rows(
            materialized=actions.materialize(actions.plan_for("ll", "E"), bank),
            role_rows=rows, ordered_batch_indices=(8, 9, 10, 11),
        )
    changed_target = rows.clone()
    changed_target[0, 256] += 1
    assert torch.equal(changed_target[:, :256], rows[:, :256])
    with pytest.raises(RuntimeError, match="differs from its sealed identity"):
        identity.require_role_rows(
            materialized=rr, role_rows=changed_target,
            ordered_batch_indices=(8, 9, 10, 11),
        )
    with pytest.raises(ValueError, match="not canonical"):
        actions.FinalActionBatchIdentity.from_role_rows(
            materialized=rr, role_rows=rows, ordered_batch_indices=(9, 8, 10, 11),
            batch_ordinal=2, source_commit="1" * 40,
            inherited_snapshot_sha256="2" * 64, rows_receipt_sha256="3" * 64,
            final_role_tensor_sha256="4" * 64, program_payload_sha256="5" * 64,
            common_support_sha256="6" * 64,
        )


def test_all_68_observational_call_ledgers_match_their_physical_paths() -> None:
    ledgers = actions.expected_observational_action_call_ledgers()
    assert tuple(ledgers) == actions.CANONICAL_ACTION_KEYS
    assert len(ledgers) == 68
    batches = actions.OBSERVATIONAL_BATCH_COUNT
    assert batches == 48

    assert ledgers["rr/N"] == {
        "outer_forward_count": batches,
        "deployed_n_calls": {"0": batches, "1": batches, "2": batches},
        "correction_calls": {"0": batches, "1": batches, "2": 0},
        "literal_early_mlp_calls": {"0": 0, "1": 0, "2": 0},
    }
    assert ledgers["rr/E"]["literal_early_mlp_calls"] == {
        "0": 0, "1": 0, "2": batches,
    }
    assert ledgers["n_n/E"]["correction_calls"] == {"0": 0, "1": 0, "2": 0}
    assert ledgers["o_o/N"]["deployed_n_calls"] == {
        "0": 0, "1": 0, "2": batches,
    }
    assert ledgers["o_o/E"]["literal_early_mlp_calls"] == {
        "0": batches, "1": batches, "2": batches,
    }
