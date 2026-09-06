import audit_task14_direction_cardinality_program_literal_price_v1 as audit


def test_plan_is_cpu_only_and_hash_closed():
    plan = audit.compile_plan()
    assert plan["price"] == {
        "gpu_model_forwards": 0,
        "cpu_model_forwards": 0,
        "backwards": 0,
        "parameter_updates": 0,
    }
    assert len(plan["inputs"]) == 7


def test_literal_counts_and_dependency_verdict():
    score = audit.evaluate()
    assert score["artifact_accounting"]["selected_executable_scalars"] == 13_824
    assert score["artifact_accounting"]["selected_executable_fp32_bytes"] == 55_296
    assert score["artifact_accounting"]["excluded_direction_only_control_scalars"] == 2_304
    assert score["interface_runtime"]["causal_plus_reader_scalar_arithmetic"] == 3_455
    assert score["native_reference"]["two_mlp6_7_parameters"] == 31_852_800
    assert score["native_reference"]["full_loaded_model_parameters"] == 545_902_902
    assert score["native_reference"]["native_blocks_eliminated_by_current_harness"] == 0
    assert score["classification"] == "interface_simple_not_end_to_end"
    assert all(score["predictions"].values())
    assert score["terminal"] == "screen"
