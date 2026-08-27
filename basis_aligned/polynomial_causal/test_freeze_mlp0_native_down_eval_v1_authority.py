from freeze_mlp0_native_down_eval_v1_authority import SOURCE_CLOSURE


def test_eval_authority_binds_rows_scorer_poison_collector_and_decoder():
    names = {path.name for path in SOURCE_CLOSURE}
    assert {
        "evaluate_mlp0_native_down_hierarchy_v1.py",
        "score_mlp0_native_down_hierarchy_v1.py",
        "mlp0_native_down_program.py",
        "prepare_mlp0_native_down_hierarchy_v1_rows.py",
    }.issubset(names)
    assert all(path.is_file() for path in SOURCE_CLOSURE)
    assert "freeze_mlp0_native_down_eval_v1_authority.py" in names
