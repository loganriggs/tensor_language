import ast
from pathlib import Path

import run_temporal_iswas_dual_command_ood_native_capability_v1 as target


def test_wrapper_authority_and_sources_exist():
    paths = {"prior": target.PRIOR, "builder": target.BUILDER,
             "temporal_capability": target.TEMPORAL_CAPABILITY,
             "iswas_capability": target.ISWAS_CAPABILITY, "base_runner": target.BASE_RUNNER}
    assert {name: target.sha(path) for name, path in paths.items()} == target.EXPECTED_WRAPPER


def test_fresh_population_is_exact_and_balanced():
    rows = target.candidate.build_rows()
    assert len(rows) == 32
    assert target.candidate.validate_rows(rows) == target.candidate.EXPECTED_AUTHORITY_SHA256
    assert len([(row, cell) for row in rows for cell in target.candidate.CELLS]) == 128


def test_prediction_schema_and_gate_are_ood_specific():
    assert len(target.PREDICTION_KEYS) == 4
    assert target.PREDICTION_KEYS[-1] == "pred_d_dual_command_ood_license_is_issued"
    source = Path(target.__file__).read_text(); ast.parse(source)
    assert "# BQGATE: EXPERIMENT" in source


def test_output_is_distinct_and_absent_before_run():
    assert target.OUT.name == "temporal_iswas_dual_command_ood_native_capability_v1_result.json"
    assert not target.OUT.exists()
