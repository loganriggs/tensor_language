import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("freeze_mlp0_quotient_stage0_v1_authority.py")
SPEC = importlib.util.spec_from_file_location("freeze_mlp0_authority", PATH)
AUTH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUTH)


def test_source_closure_contains_math_rows_model_and_amendment():
    names = {path.name for path in AUTH.SOURCE_CLOSURE}
    assert "mlp0_quotient_worst_cell.py" in names
    assert "causal_response_quotient.py" in names
    assert "prepare_mlp0_quotient_stage0_v1_rows.py" in names
    assert "MLP0_QUOTIENT_STAGE0_V1_AMENDMENT.md" in names
    assert "bilin18_joint_removal.py" in names
    assert "tier2_model.py" in names
    assert "tt_model.py" in names


def test_result_failure_and_authority_namespaces_are_distinct():
    assert len({AUTH.AUTHORITY, AUTH.RESULT, AUTH.FAILURE, AUTH.FIT_RECEIPT}) == 4
    assert "stage0_v1" in AUTH.RESULT.name
