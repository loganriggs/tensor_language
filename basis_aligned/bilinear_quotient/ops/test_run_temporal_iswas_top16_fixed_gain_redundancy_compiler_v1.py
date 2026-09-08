import json

import run_temporal_iswas_top16_fixed_gain_redundancy_compiler_v1 as run


def test_authority_gain_and_top_are_frozen():
    assert {name: run.sha(path) for name, path in {
        "prior": run.PRIOR, "task_result": run.TASK_RESULT, "v19_result": run.V19_RESULT,
        "builder18": run.BUILDER18, "builder19": run.BUILDER19,
        "executor": run.EXECUTOR, "transfer": run.TRANSFER}.items()} == run.EXPECTED
    assert run.GAIN == 1.25 and run.TOP == 16


def test_price_covers_seven_eight_forward_panels():
    assert run.PRICE == json.loads(run.PRIOR.read_text())["frozen_design"]["price"]
    assert run.PRICE["model_forwards"] == 7 * 8 == 56


def test_receipt_must_remain_retrospective():
    prior = json.loads(run.PRIOR.read_text())
    assert prior["evidence_status"].startswith("Retrospective")
    source = run.Path(run.__file__).read_text()
    assert "retrospective_calibration_only_not_confirmation" in source
    assert "Adam" not in source and "linalg.svd" not in source
