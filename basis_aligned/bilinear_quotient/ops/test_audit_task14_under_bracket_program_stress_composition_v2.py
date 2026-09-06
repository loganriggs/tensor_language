import audit_task14_under_bracket_program_stress_composition_v2 as audit

def test_audit_is_zero_execution_and_one_sided():
    plan=audit.compile_plan()
    assert not any(plan["price"].values())
    assert plan["only_promoted_panel"]=="task14 under bracket-program stress"
    assert plan["excluded_panel"]=="bracket under Task14 stress"

def test_task14_component_passes_without_reverse_upgrade():
    score=audit.evaluate()
    assert all(score["predictions"].values())
    assert score["terminal"]=="screen"
    assert score["task14"]["foreign_stress_to_own_norm_ratio"]>2
    assert score["task14"]["interaction_to_own_norm_ratio"]<.02
    assert "two-sided composition" in score["explicitly_not_exported"]
