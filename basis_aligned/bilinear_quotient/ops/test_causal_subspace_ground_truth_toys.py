import numpy as np

import causal_subspace_ground_truth_toys as target


def test_exact_minimum_attains_rank_lower_bound_and_zero_error():
    for scenario in target.make_scenarios().values():
        basis, certificate = target.exact_minimum_subspace(scenario.deltas, scenario.readers)
        assert certificate["attained_rank"] == certificate["minimum_rank_lower_bound"]
        assert target.preservation_error(scenario.deltas, scenario.readers, basis) < 1e-10


def test_ladder_has_expected_ground_truth_ranks_and_dim_failure():
    reports = target.evaluate_ladder()["scenarios"]
    assert reports["axis_aligned_rank1"]["exact_certificate"]["attained_rank"] == 1
    assert reports["axis_aligned_rank1"]["dim_preservation_error"] < 1e-10
    assert reports["rotated_rank3"]["exact_certificate"]["attained_rank"] == 3
    assert reports["rotated_rank3"]["dim_preservation_error"] > .5


def test_response_regression_recovers_exact_subspace_without_reader_weights():
    for scenario in target.make_scenarios().values():
        reader = target.stack_readers(scenario.readers)
        responses = scenario.deltas @ reader.T
        recovered = target.response_regression_subspace(scenario.deltas, responses)
        exact, certificate = target.exact_minimum_subspace(scenario.deltas, scenario.readers)
        assert recovered.shape[1] == certificate["attained_rank"]
        assert target.projector_distance(recovered, exact) < 1e-10
        assert target.preservation_error(scenario.deltas, scenario.readers, recovered) < 1e-10


def test_shared_private_tasks_have_rank_two_each_and_one_shared_direction():
    report = target.evaluate_ladder()["scenarios"]["shared_plus_private"]
    assert report["task_a_rank"] == 2
    assert report["task_b_rank"] == 2
    assert report["exact_certificate"]["attained_rank"] == 3
    assert report["shared_dimension"] == 1


def test_subset_only_training_has_zero_training_error_but_fails_full_population():
    scenario = target.make_scenarios()["subset_gated"]
    common, certificate = target.exact_minimum_subspace(
        scenario.deltas, scenario.readers, environments=["common_rows"])
    assert certificate["attained_rank"] == 1
    assert target.preservation_error(
        scenario.deltas, scenario.readers, common, environments=["common_rows"]) < 1e-10
    assert target.preservation_error(scenario.deltas, scenario.readers, common) > .5


def test_rotated_solution_is_gauge_invariant():
    scenario = target.make_scenarios()["rotated_rank3"]
    basis, _ = target.exact_minimum_subspace(scenario.deltas, scenario.readers)
    q, _ = np.linalg.qr(np.asarray([[1.0, 2.0, 3.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0]]))
    assert target.projector_distance(basis, basis @ q) < 1e-10


def test_environment_augmentation_breaks_exact_causal_nuisance_alias():
    report = target.evaluate_environment_confounding()
    assert report["exact_certificate"]["attained_rank"] == 2
    assert report["one_environment_rank"] == 2
    assert report["one_environment_training_error"] < 1e-10
    assert report["one_environment_full_error"] > .4
    assert report["one_environment_projector_distance"] > .9
    assert report["augmented_rank"] == 2
    assert report["augmented_full_error"] < 1e-10
    assert report["augmented_projector_distance"] < 1e-10


def test_bilinear_context_model_recovers_rare_subset_direction():
    report = target.evaluate_bilinear_context_gate()
    assert report["rare_fraction"] == .125
    assert report["exact_certificate"]["attained_rank"] == 2
    assert report["pooled_linear_rank"] == 1
    assert report["pooled_linear_full_error"] > .5
    assert report["contextual_bilinear_rank"] == 2
    assert report["contextual_bilinear_full_error"] < 1e-10
    assert report["contextual_bilinear_projector_distance"] < 1e-10
