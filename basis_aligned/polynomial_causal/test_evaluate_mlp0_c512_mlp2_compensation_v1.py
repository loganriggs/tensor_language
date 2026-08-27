from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import evaluate_mlp0_c512_mlp2_compensation_v1 as evaluate
from mlp0_c512_mlp2_evaluator_contract import expected_call_contract


def fake_blocks():
    return [SimpleNamespace(mlp=nn.Identity()) for _ in range(18)]


def test_phase_counter_realizes_exact_registered_algebra():
    blocks = fake_blocks()
    expected = expected_call_contract(4)["exact_phase_site_call_counts"]
    counter = evaluate.PhaseSiteCounter(blocks, expected)
    value = torch.ones(1)
    try:
        with counter.phase("mlp1_teacher_capture"):
            for _ in range(4):
                blocks[0].mlp(value)  # intentional uncounted upstream work
                blocks[1].mlp(value)
        with counter.phase("mlp2_teacher_capture"):
            for _ in range(4):
                blocks[2].mlp(value)
        with counter.phase("parent_replay_mlp_sites"):
            for site in range(18):
                repeats = 2 if site == 2 else 4
                for _ in range(repeats):
                    blocks[site].mlp(value)
        with counter.phase("crossed_suffix_replay"):
            for site in range(3, 18):
                for _ in range(8):
                    blocks[site].mlp(value)
        assert counter.counts == expected
    finally:
        counter.close()


def test_phase_counter_records_forbidden_crossed_teacher_and_rejects_other_drift():
    blocks = fake_blocks()
    expected = expected_call_contract(4)["exact_phase_site_call_counts"]
    counter = evaluate.PhaseSiteCounter(blocks, expected)
    try:
        with counter.phase("crossed_suffix_replay"):
            blocks[2].mlp(torch.ones(1))
        assert counter.counts["crossed_forbidden_teacher"]["2"] == 1
        with pytest.raises(RuntimeError, match="unexpected site 3"):
            with counter.phase("mlp1_teacher_capture"):
                blocks[3].mlp(torch.ones(1))
    finally:
        counter.close()


def test_inherited_currency_contract_is_complete_and_deterministic():
    first, first_hash = evaluate.inherited_currency_contract()
    second, second_hash = evaluate.inherited_currency_contract()
    assert first == second
    assert first_hash == second_hash
    assert first["centered_capped_logit_rms"] == 2.584135756182981
    assert first["fit_rows_shape"] == [960, 257]
    assert set(first) >= {
        "prior_result_sha256", "prior_authority_sha256",
        "stage0_row_receipt_sha256", "stage0_fit_receipt_sha256",
        "fit_rows_tensor_sha256", "token_count_tensor_sha256",
        "frequency_median", "pre_mlp0_raw_residual_norm_median",
        "punctuation_table_sha256", "valid_mask_definition",
        "logit_cap_definition", "nrmse_definition",
    }
    assert len(first_hash) == 64


def test_v2_repair_amendment_is_complete_deterministic_and_outcome_blind():
    first, first_hash = evaluate.repair_amendment_contract()
    second, second_hash = evaluate.repair_amendment_contract()
    assert first == second and first_hash == second_hash
    assert first["v1_result_absent"] is True
    assert first["v1_exposed_scientific_outcomes"] == []
    assert first["v1_sufficient_statistics_serialized"] is False
    assert first["preexisting_core_norm_contract"] == (
        evaluate.NATIVE_CONTROL_NORM_CONTRACT
    )
    assert not evaluate.V1_RESULT.exists()
    assert len(first_hash) == 64


def test_native_control_norm_diagnostic_uses_scale_aware_coordinatewise_bound():
    target = torch.tensor([[[100.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    within = torch.tensor([[[100.0009, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    report = evaluate.native_control_norm_diagnostics(
        target, within, evaluate.NATIVE_CONTROL_NORM_CONTRACT
    )
    assert report["native_control_norm_max_abs_error"] > 1e-6
    assert report["native_control_norm_max_allowance_ratio"] < 1
    assert report["native_control_norm_all_positions_within_bound"] is True

    outside = torch.tensor([[[100.0011, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    report = evaluate.native_control_norm_diagnostics(
        target, outside, evaluate.NATIVE_CONTROL_NORM_CONTRACT
    )
    assert report["native_control_norm_max_allowance_ratio"] > 1
    assert report["native_control_norm_all_positions_within_bound"] is False


def test_frozen_domain_rebuilds_exact_1256_window_identity():
    domain, identity, rows = evaluate.load_domain()
    assert tuple(rows.shape) == (628, 513)
    assert tuple(domain["rows"].shape) == (1256, 257)
    assert len(identity["ordered_ids"]) == 384
    assert domain["unit_ids"].tolist() == identity["row_to_unit"]


def test_replay_error_tracks_raw_capped_and_mean_ce():
    reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[0, 1]])
    replay = {
        "raw_logits_max_abs": 0.0,
        "capped_logits_max_abs": 0.0,
        "ce_abs": 0.0,
    }
    evaluate.replay_error(
        replay, reference, reference, reference.clone(), reference.clone(), target
    )
    assert replay == {
        "raw_logits_max_abs": 0.0,
        "capped_logits_max_abs": 0.0,
        "ce_abs": 0.0,
    }


def test_empty_ledgers_are_exact_registered_shape():
    ledgers = evaluate.empty_ledgers()
    assert set(ledgers) == set(evaluate.CONTRASTS)
    for metrics in ledgers.values():
        assert set(metrics) == set(evaluate.MARGINS)
        for ledger in metrics.values():
            assert ledger["sums"].shape == (384, 16)
            assert ledger["counts"].shape == (384, 16)


def test_lock_conflict_preserves_the_live_owner_and_emits_no_failure(tmp_path, monkeypatch):
    lock = tmp_path / "owned.lock"
    failure = tmp_path / "failure.json"
    output = tmp_path / "result.json"
    monkeypatch.setattr(evaluate, "LOCK", lock)
    monkeypatch.setattr(evaluate, "FAILURE", failure)
    monkeypatch.setattr(evaluate, "OUT", output)
    owner = evaluate.acquire_lock()
    inode = lock.stat().st_ino
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            evaluate.authoritative_entry()
        assert lock.exists() and lock.stat().st_ino == inode
        assert not failure.exists() and not output.exists()
    finally:
        evaluate.release_owned_lock(owner)
    assert not lock.exists()


def test_create_only_failure_writer_never_replaces_existing_namespace(tmp_path):
    path = tmp_path / "failure.json"
    evaluate.write_json_create_only({"first": 1}, path)
    with pytest.raises(FileExistsError):
        evaluate.write_json_create_only({"second": 2}, path)
    assert path.read_text().strip() == '{\n  "first": 1\n}'
