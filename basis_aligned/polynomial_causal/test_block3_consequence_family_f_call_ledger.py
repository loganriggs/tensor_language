import copy
import pathlib

import pytest

import block3_consequence_family_f_call_ledger as calls


def complete_ledger():
    ledger = calls.FamilyFCallLedger()
    calls.record_frozen_schedule(ledger)
    return ledger


def test_exact_schedule_closes_with_registered_totals_and_sites():
    receipt = complete_ledger().close()
    assert receipt["totals"] == calls.EXPECTED_TOTALS
    assert receipt["totals"]["optimizer_steps"] == 2400
    assert receipt["totals"]["two_row_backwards"] == 9600
    assert receipt["totals"]["prefixes"] == 2940
    assert receipt["totals"]["teacher_suffixes"] == 2460
    assert receipt["totals"]["student_suffixes"] == 10680
    assert receipt["totals"]["suffix_returns"] == 13140
    assert receipt["totals"]["student_native_mlp3_calls"] == 0
    assert receipt["totals"]["donor_prefixes"] == 480
    assert receipt["totals"]["projections"] == 1440
    assert receipt["totals"]["outer_model_calls"] == 1
    assert receipt["totals"]["outer_model_returns"] == 1
    assert receipt["totals"]["raw_logit_returns"] == 13141
    assert receipt["totals"]["attention_calls_by_site"] == {
        str(site): 2941 if site <= 3 else 13141 for site in range(18)
    }
    assert receipt["totals"]["mlp_calls_by_site"] == {
        str(site): 2941 if site <= 3 else 13141 for site in range(18)
    }


def test_cell_schedule_distinguishes_phases_and_every_reporting_arm():
    expected = calls.expected_cells()
    assert len(expected) == 3 + 4 + 1 + 18 + 1
    assert expected[("score_fit", "teacher")].optimizer_steps == 480
    assert expected[("score_fit", "teacher_document_derangement")].prefixes == 960
    assert expected[("affine_fit", "family_A_k512")].two_row_backwards == 960
    assert expected[("postfit_report", calls.REPORT_SHARED_ARM)].teacher_suffixes == 60
    for arm in calls.REPORT_STUDENT_ARMS:
        assert expected[("postfit_report", arm)].student_suffixes == 60
    assert expected[("known_answer", calls.OUTER_REPLAY_ARM)].outer_model_calls == 1


def test_incremental_logical_steps_accumulate_before_close():
    ledger = calls.FamilyFCallLedger()
    for _ in range(480):
        ledger.record_prefix("score_fit", "teacher")
        ledger.record_teacher_suffix("score_fit", "teacher")
        for _ in range(4):
            ledger.record_student_suffix("score_fit", "teacher")
        ledger.record_optimizer_step(
            "score_fit", "teacher", backwards=4, projection=True,
        )
    assert ledger.partial_receipt()["cells"]["score_fit/teacher"] == (
        calls.expected_cells()[("score_fit", "teacher")].as_dict()
    )
    with pytest.raises(RuntimeError, match="incomplete or misattributed"):
        ledger.close()


def test_equal_global_totals_cannot_hide_cross_arm_misattribution():
    ledger = complete_ledger()
    ledger._cells[("score_fit", "teacher")] = calls.CellCounts(
        optimizer_steps=479, two_row_backwards=1916, prefixes=479,
        teacher_suffixes=479, student_suffixes=1916, projections=479,
    )
    ledger._cells[("score_fit", "teacher_row_reversal")] = calls.CellCounts(
        optimizer_steps=481, two_row_backwards=1924, prefixes=481,
        teacher_suffixes=481, student_suffixes=1924, projections=481,
    )
    assert calls._derived_totals(ledger._cells) == calls.EXPECTED_TOTALS
    with pytest.raises(RuntimeError, match="teacher"):
        ledger.close()


def test_student_native_mlp3_is_rejected_even_if_added_to_known_arm():
    ledger = calls.FamilyFCallLedger()
    with pytest.raises(RuntimeError, match="exceeded protocol"):
        ledger.record(
            "score_fit", "teacher", student_native_mlp3_calls=1,
        )


def test_outer_replay_is_exactly_one_and_is_required_for_close():
    ledger = complete_ledger()
    ledger._cells[("known_answer", calls.OUTER_REPLAY_ARM)] = calls.CellCounts()
    with pytest.raises(RuntimeError, match="known_answer"):
        ledger.validate_exact()
    ledger.record_outer_replay()
    ledger.validate_exact()
    with pytest.raises(RuntimeError, match="already closed"):
        ledger.record_outer_replay()


def test_donor_prefix_and_projection_are_phase_bound():
    ledger = calls.FamilyFCallLedger()
    with pytest.raises(RuntimeError, match="derangement"):
        ledger.record_prefix("score_fit", "teacher", donor=True)
    ledger.record_prefix("score_fit", "teacher_document_derangement", donor=True)
    with pytest.raises(RuntimeError, match="exceeded protocol"):
        ledger.record_optimizer_step(
            "affine_fit", "teacher_F_k512", projection=True,
        )


def test_unknown_arm_and_nonliteral_or_negative_counts_are_rejected():
    ledger = calls.FamilyFCallLedger()
    with pytest.raises(ValueError, match="unregistered"):
        ledger.record("score_fit", "teacher_typo", optimizer_steps=1)
    with pytest.raises(ValueError, match="literal integer"):
        ledger.record("score_fit", "teacher", optimizer_steps=True)
    with pytest.raises(ValueError, match="literal integer"):
        ledger.record("score_fit", "teacher", prefixes=-1)


def test_closed_ledger_is_inert_and_complete_receipt_replays():
    ledger = complete_ledger()
    receipt = ledger.validate_exact()
    replayed = calls.FamilyFCallLedger.replay_complete_receipt(receipt)
    assert replayed.closed
    assert replayed.receipt() == receipt
    with pytest.raises(RuntimeError, match="already closed"):
        ledger.record("score_fit", "teacher")


@pytest.mark.parametrize("corruption", ("cell", "total", "status"))
def test_complete_receipt_corruption_is_rejected(corruption):
    receipt = copy.deepcopy(complete_ledger().close())
    if corruption == "cell":
        receipt["cells"]["score_fit/teacher"]["prefixes"] -= 1
    elif corruption == "total":
        receipt["totals"]["prefixes"] -= 1
    else:
        receipt["status"] = "partial_nonauthoritative"
    with pytest.raises(RuntimeError):
        calls.FamilyFCallLedger.replay_complete_receipt(receipt)


def test_module_is_pure_and_has_no_torch_model_or_file_io_imports():
    source = pathlib.Path(calls.__file__).read_text()
    assert "import torch" not in source
    assert "facade" not in source
    assert "torch.load" not in source
    assert "open(" not in source


def test_mutating_convenience_expected_totals_cannot_weaken_validation(monkeypatch):
    monkeypatch.setitem(calls.EXPECTED_TOTALS, "prefixes", -1)
    receipt = complete_ledger().validate_exact()
    assert receipt["totals"]["prefixes"] == 2940
