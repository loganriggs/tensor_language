import json

import run_temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v2 as runner


def test_repair_static_dryrun(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["candidate_id"] == runner.CANDIDATE_ID
    assert receipt["model_forwards"] == 24
    assert receipt["factorial_records"] == 512
    assert receipt["gpu_accessed"] is False


def test_repair_result_requires_entry_recurrence():
    result = {
        "summaries": {"entry": {family: {"mean_recovery": value} for family, value in runner.ENTRY_TARGET.items()}},
        "predictions": {
            "pred_a_authority_capability_exact_full_sequence_cube": True,
            "pred_b_boundary11_direct_ceiling_recurrence": True,
            "pred_c_full_sequence_cube_recovers_crossing": True,
            "pred_d_attention_is_dominant_transfer": True,
            "pred_e_exact_zero_fit_price": True,
        },
        "dryrun": {"candidate_id": "old"},
    }
    assert runner.repair_result(result)["terminal"] == "screen"
