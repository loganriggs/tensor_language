import json

import run_temporal_auxiliary_will_had_block8_subject_attention_heads_v1 as runner


def test_static_population_and_arms():
    rows, fast_screen, capability = runner.validate_static()
    assert len(rows) == 64
    assert fast_screen["terminal"] == "screen"
    assert len(capability) == 4
    assert len(runner.HEADS) == 12
    assert len(runner.ARMS) == 14


def test_dryrun_exact_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 32
    assert receipt["example_evaluations"] == 1024
    assert receipt["intervention_records"] == 896
    assert receipt["gpu_accessed"] is False


def test_arm_inventory_is_nonadaptive():
    assert runner.ARMS[:2] == ("head:00", "head:01")
    assert runner.ARMS[-2:] == ("full_heads", "direct_output")
