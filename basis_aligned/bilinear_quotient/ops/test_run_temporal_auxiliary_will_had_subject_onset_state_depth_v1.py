import json

import run_temporal_auxiliary_will_had_subject_onset_state_depth_v1 as runner


def test_static_authorities_and_population():
    rows, fast_screen, capability = runner.validate_static()
    assert len(rows) == 64
    assert fast_screen["terminal"] == "screen"
    assert len(capability) == 4
    assert all(cell["passed"] for cell in capability)


def test_dryrun_has_exact_registered_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 44
    assert receipt["example_evaluations"] == 1408
    assert receipt["intervention_records"] == 1280
    assert receipt["gpu_accessed"] is False


def test_pair_error_contract():
    class Output:
        answer_foil = ((1.0, 2.0), (3.0, 4.0))

    assert runner.pair_error(Output(), ((1.0, 2.0), (3.0, 4.0))) == 0.0
