import pytest

import early_mlp_suffix_transport_v1_final_actions as actions
import early_mlp_suffix_transport_v1_final_capability as capability


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
