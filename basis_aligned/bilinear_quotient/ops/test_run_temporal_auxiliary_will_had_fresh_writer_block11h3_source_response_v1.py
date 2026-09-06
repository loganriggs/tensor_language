import json

import run_temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_v1 as runner


def test_static_and_dryrun(monkeypatch, capsys):
    assert len(runner.validate_static()) == 64
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert len(receipt["arms"]) == 8
    assert receipt["model_forwards"] == 26
    assert receipt["example_evaluations"] == 832
    assert receipt["intervention_records"] == 512
    assert receipt["gpu_accessed"] is False
