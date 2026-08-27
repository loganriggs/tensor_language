from freeze_mlp0_native_down_fit_v1_authority import SOURCE_CLOSURE


def test_fit_authority_source_closure_exists_and_excludes_eval_row_loader():
    assert all(path.is_file() for path in SOURCE_CLOSURE)
    names = {path.name for path in SOURCE_CLOSURE}
    assert "prepare_mlp0_native_down_hierarchy_v1_rows.py" not in names
    compiler = next(path for path in SOURCE_CLOSURE if path.name == "compile_mlp0_native_down_fit_v1.py")
    source = compiler.read_text()
    assert "mlp0_native_down_hierarchy_v1_rows_receipt" not in source
    assert "load_frozen_role(\"fit\")" in source
