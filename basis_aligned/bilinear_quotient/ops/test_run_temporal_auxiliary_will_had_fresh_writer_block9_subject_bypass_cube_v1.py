import json

import run_temporal_auxiliary_will_had_fresh_writer_block9_subject_bypass_cube_v1 as runner


def test_static_authorities_and_factorial():
    rows, capability = runner.validate_static()
    assert len(rows) == 64
    assert len(capability) == 8
    assert len(runner.SUBSETS) == 8


def test_dryrun_exact_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 20
    assert receipt["example_evaluations"] == 640
    assert receipt["intervention_records"] == 512
    assert receipt["gpu_accessed"] is False
