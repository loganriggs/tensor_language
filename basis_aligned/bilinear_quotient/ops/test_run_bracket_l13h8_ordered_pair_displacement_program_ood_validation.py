import run_bracket_l13h8_ordered_pair_displacement_program_ood_validation as program


def test_plan_binds_committed_artifact_and_capability():
    plan = program.compile_plan()
    assert plan["artifact_sha256"] == program.ARTIFACT_SHA256
    assert plan["capability_sha256"] == program.CAPABILITY_SHA256
    assert plan["target_installations"] == 144
    assert plan["control_zero_dispatches"] == 216
    assert plan["total_export_capability_causal_price"]["model_forwards"] == 5


def test_synthetic_exact_program_passes_all_gates():
    records = []
    for family in program.authority.TARGET_FAMILIES:
        for a, b in program.authority.ORDERED_PAIRS:
            for index in range(12):
                records.append({"program_role": "target", "family_id": family,
                                "ordered_pair": f"{a}->{b}", "dispatch": f"{a}->{b}",
                                "native_margin_replay_absolute_error": 0.0,
                                "program_max_absolute_logit_change": 1.0,
                                "program_recipient_correct": False,
                                "exact_donorward_effect": 1.0,
                                "program_donorward_effect": 1.0})
    for family in program.authority.CONTROL_FAMILIES:
        for index in range(72):
            records.append({"program_role": "control", "family_id": family,
                            "ordered_pair": "1->1", "dispatch": "zero",
                            "native_margin_replay_absolute_error": 0.0,
                            "program_max_absolute_logit_change": 0.0,
                            "program_recipient_correct": True})
    scored = program.score(records)
    assert len(records) == 360
    assert all(scored["predictions"].values())
    assert scored["terminal"] == "program_screen"
