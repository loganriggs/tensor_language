import json

import run_temporal_auxiliary_will_had_block8_subject_attention_heads_v2 as runner


def test_repaired_inventory_is_exact():
    assert runner.HEADS == tuple(range(9))
    assert len(runner.ARMS) == 11


def test_v2_dryrun_exact_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["candidate_id"].endswith("attention_heads_v2")
    assert receipt["heads"] == list(range(9))
    assert receipt["model_forwards"] == 26
    assert receipt["example_evaluations"] == 832
    assert receipt["intervention_records"] == 704
    assert receipt["gpu_accessed"] is False
