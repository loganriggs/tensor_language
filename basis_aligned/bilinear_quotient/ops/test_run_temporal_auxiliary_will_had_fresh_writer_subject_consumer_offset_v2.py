import json

import run_temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_v2 as runner


def test_static_authority_and_price(monkeypatch, capsys):
    rows, writer, bypass = runner.validate_static()
    assert len(rows) == 64
    assert writer["terminal"] == "screen"
    assert bypass["terminal"] == "null"
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 24
    assert receipt["example_evaluations"] == 768
    assert receipt["intervention_records"] == 576
    assert receipt["boundaries"] == list(range(10, 19))
    assert receipt["gpu_accessed"] is False
