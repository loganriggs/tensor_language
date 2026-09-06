import json

import run_temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v1 as runner


def test_static_authority_and_dryrun(monkeypatch, capsys):
    rows, writer = runner.validate_static()
    assert len(rows) == 64
    assert writer["terminal"] == "screen"
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["factorial_arms"] == 8
    assert receipt["model_forwards"] == 24
    assert receipt["example_evaluations"] == 768
    assert receipt["factorial_records"] == 512
    assert receipt["gpu_accessed"] is False
