import run_bracket_l13h8_ordered_pair_program_ood_capability as capability


def test_capability_plan_never_installs_or_opens_causal_program():
    plan = capability.compile_plan()
    assert plan["split"] == "OOD"
    assert plan["rows"] == 180 and plan["endpoints"] == 360
    assert plan["causal_interventions"] == 0
    assert plan["program_vectors_installed"] == 0
    assert plan["price"]["model_forwards"] == 1


def test_score_requires_every_target_and_control_cell():
    evidence = []
    families = capability.authority.TARGET_FAMILIES
    for family in families:
        for a, b in capability.authority.ORDERED_PAIRS:
            for index in range(6):
                evidence.append({"cell_id": f"target|{family}|{a}->{b}",
                                 "correct": True, "closer_margin": 1.0})
    for family in capability.authority.CONTROL_FAMILIES:
        for side in ("base", "donor"):
            for index in range(36):
                evidence.append({"cell_id": f"control|{family}|{side}",
                                 "correct": True, "closer_margin": 1.0})
    scored = capability.score(evidence)
    assert len(evidence) == 360
    assert all(scored["predictions"].values())
    assert scored["terminal"] == "capability_pass"
