import json

import run_temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1 as runner


def test_static_authority_and_dryrun(monkeypatch, capsys):
    rows, writer = runner.validate_static()
    assert len(rows) == 64 and writer["terminal"] == "screen"
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["arms"] == list(runner.ARMS)
    assert receipt["model_forwards"] == 16
    assert receipt["example_evaluations"] == 512
    assert receipt["intervention_records"] == 256
    assert receipt["gpu_accessed"] is False
