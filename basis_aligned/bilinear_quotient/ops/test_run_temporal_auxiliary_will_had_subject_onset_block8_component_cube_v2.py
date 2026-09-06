import json

import run_temporal_auxiliary_will_had_subject_onset_block8_component_cube_v2 as runner


def test_native_sum_preserves_registered_operation_order():
    class Piece:
        def __init__(self, text):
            self.text = text
        def __add__(self, other):
            return Piece(f"({self.text}+{other.text})")
    assert runner.native_sum((Piece("z"), Piece("a"), Piece("m"))).text == "((z+a)+m)"


def test_v2_dryrun(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["candidate_id"].endswith("component_cube_v2")
    assert receipt["model_forwards"] == 20
    assert receipt["gpu_accessed"] is False
