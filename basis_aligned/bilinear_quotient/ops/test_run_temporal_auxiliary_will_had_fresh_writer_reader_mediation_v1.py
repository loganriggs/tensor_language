import json

import run_temporal_auxiliary_will_had_fresh_writer_reader_mediation_v1 as runner


def test_static_authorities_and_arms():
    rows, capability = runner.validate_static()
    assert len(rows) == 64
    assert len(capability) == 8
    assert len(runner.ARMS) == 5


def test_dryrun_exact_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 16
    assert receipt["example_evaluations"] == 512
    assert receipt["writer_arm_records"] == 256
    assert receipt["self_clamp_controls"] == 64
    assert receipt["gpu_accessed"] is False
