import json

import run_temporal_auxiliary_will_had_subject_onset_block8_component_cube_v1 as runner


def test_static_authorities_and_factorial():
    rows, fast_screen, capability = runner.validate_static()
    assert len(rows) == 64
    assert fast_screen["terminal"] == "screen"
    assert len(capability) == 4
    assert len(runner.SUBSETS) == 8


def test_dryrun_has_exact_registered_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 20
    assert receipt["example_evaluations"] == 640
    assert receipt["intervention_records"] == 512
    assert receipt["gpu_accessed"] is False


def test_arm_ids_cover_empty_and_full():
    assert runner.arm_id(()) == "empty"
    assert runner.arm_id(runner.BRANCHES) == "+".join(runner.BRANCHES)
