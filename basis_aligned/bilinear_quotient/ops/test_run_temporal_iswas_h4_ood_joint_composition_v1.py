import ast
from pathlib import Path

import run_temporal_iswas_h4_ood_joint_composition_v1 as target


def test_authority_hashes_and_ood_population():
    paths = {"prior": target.PRIOR, "builder": target.BUILDER, "capability": target.CAPABILITY,
             "h4_result": target.H4_RESULT, "parent_runner": target.PARENT_RUNNER,
             "joint": target.JOINT, "accounting": target.ACCOUNTING, "producer": target.PRODUCER}
    assert {name: target.sha(path) for name, path in paths.items()} == target.EXPECTED
    assert target.candidate.validate_rows(target.candidate.build_rows()) == target.candidate.EXPECTED_AUTHORITY_SHA256


def test_union_and_literal_price_are_frozen():
    assert target.UNION == {9: (1, 4), 11: (3,), 15: (5,)}
    assert target.PRICE == {"checkpoint_loads": 1, "model_forwards": 4,
        "sequence_evaluations": 512, "scored_token_positions": 1024,
        "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}


def test_prediction_schema_and_atomic_write():
    assert len(target.PREDICTION_KEYS) == 5
    source = Path(target.__file__).read_text(); tree = ast.parse(source)
    assert "# BQGATE: EXPERIMENT" in source
    assert any(isinstance(node, ast.Call) and getattr(node.func, "id", "") == "atomic_create_json"
               for node in ast.walk(tree))


def test_result_absent_before_run():
    assert not target.OUT.exists()
