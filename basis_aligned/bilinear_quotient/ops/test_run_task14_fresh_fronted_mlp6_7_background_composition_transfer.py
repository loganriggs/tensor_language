from collections import Counter

import run_task14_fresh_fronted_mlp6_7_background_composition_transfer as run


def test_adapter_has_four_balanced_cells_and_three_roles():
    rows = run.build_rows()
    counts = Counter(f"{row['direction_id']}__{row['template_id']}" for row in rows)
    assert len(rows) == 32 and len(counts) == 4 and set(counts.values()) == {8}
    assert all(tuple(row["endpoints"]) == (
        "recipient", "opposite_same_lemma", "same_number_different_lemma") for row in rows)


def test_adapter_donor_counterfactual_changes_only_subject_token():
    for row in run.build_rows():
        opposite = row["endpoints"]["opposite_same_lemma"]["ids"]
        lexical = row["endpoints"]["same_number_different_lemma"]["ids"]
        assert [i for i, pair in enumerate(zip(opposite, lexical))
                if pair[0] != pair[1]] == [8]
        assert row["endpoints"]["recipient"]["ids"][:8] != opposite[:8]


def test_plan_binds_license_and_corrected_price():
    plan = run.compile_plan()
    assert plan["capability_license_sha256"] == run.CAPABILITY_LICENSE_SHA256
    assert plan["price"]["physical_model_forwards"] == 18
    assert plan["price"]["example_evaluations"] == 4288
    assert plan["price"]["causal_installations"] == 2048


def test_cosine_identity():
    profile = {f: i + 1.0 for i, f in enumerate(run.parent.BACKGROUND_FACTORS)}
    assert abs(run._cosine(profile, profile) - 1.0) < 1e-12
