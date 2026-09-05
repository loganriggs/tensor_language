import run_task14_pristine_split_mlp6_7_absolute_composition_transfer as run


def test_derived_price_matches_complete_lattice():
    assert run.derive_price() == {"physical_model_forwards": 22,
        "example_evaluations": 5360, "causal_installations": 2560,
        "maximum_patch_chunk_rows": 256, "patch_chunks": 10,
        "backwards": 0, "parameter_updates": 0}


def test_five_scalar_law_recovers_additive_set_function():
    weights = {"E": -.4, "A": -.3, "U": -.2, "W": -.1}
    q = {s: .7 + sum(weights[f] for f in s) for s in run.SUBSETS}
    law = run._five_scalar(q); predicted = run._predict(law)
    assert law["closure_error"] < 1e-12
    assert all(abs(predicted[s] - q[s]) < 1e-12 for s in run.SUBSETS)


def test_plan_binds_pristine_split_and_license():
    plan = run.compile_plan()
    assert plan["fit_rows"] == 32 and plan["holdout_rows"] == 8
    assert plan["deployed_gate_scalars"] == 10
    assert plan["capability_license_sha256"] == run.CAPABILITY_LICENSE_SHA256
    assert plan["price"] == run.derive_price()


def test_error_scorer_reports_absolute_and_normalized_values():
    observed = {s: float(len(s)) for s in run.SUBSETS}
    predicted = {s: observed[s] + .1 for s in run.SUBSETS}
    score = run._errors(predicted, observed, run.SUBSETS, 2.0)
    assert abs(score["absolute_mae"] - .1) < 1e-12
    assert abs(score["normalized_mae"] - .05) < 1e-12
