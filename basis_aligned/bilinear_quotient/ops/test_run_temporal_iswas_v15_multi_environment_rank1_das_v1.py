import run_temporal_iswas_v15_multi_environment_rank1_das_v1 as runner


def _pair(initialization, violation, control=0.01, maximum=0.02, flips=0):
    score = {
        "worst_target_violation": violation, "control_objective": control,
        "worst_control_max_kl": maximum, "total_control_flips": flips,
    }
    return {
        "initialization": initialization,
        "fits": {0: {"best": {"score": score}}, 1: {"best": {"score": dict(score)}}},
    }


def test_pair_key_never_trades_target_feasibility_for_control_quality():
    feasible = _pair("factor_svd_a1", 0.0, control=0.2)
    infeasible = _pair("joint_construction_dim", 0.01, control=0.0)
    assert runner.pair_key(feasible) < runner.pair_key(infeasible)


def test_pair_key_uses_frozen_initialization_order_only_as_final_tie_break():
    factor = _pair("factor_svd_a1", 0.0)
    frozen = _pair("frozen_a1_only_projector", 0.0)
    assert runner.pair_key(factor) < runner.pair_key(frozen)


def test_v16_c_is_explicitly_outside_patch_panels():
    assert runner.V16_PROJECTION_MIN == 0.65
    assert runner.V16_DIRECTION_MIN == 0.875
    assert runner.V16_IMPROVEMENT_MIN == 0.08
