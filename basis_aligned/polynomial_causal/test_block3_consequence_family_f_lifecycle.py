import json

import pytest

import block3_consequence_family_f_lifecycle as life


def test_protocol_matches_frozen_schedule_and_denies_evaluation():
    protocol = life.protocol()
    assert protocol["score_arms"] == [
        "teacher", "teacher_row_reversal", "teacher_document_derangement",
    ]
    assert protocol["total_logical_optimizer_steps"] == 2400
    assert protocol["total_two_row_backwards"] == 9600
    assert protocol["affine_fitted_coordinates_total"] == 4 * 1153
    assert protocol["affine_parameter_adam_dtype"] == "torch.float64"
    assert protocol["postfit_reporting_student_arms_per_batch"] == 18
    assert protocol["total_prefix_calls"] == 2940
    assert protocol["total_teacher_suffix_calls"] == 2460
    assert protocol["total_student_suffix_calls"] == 10680
    assert protocol["total_suffix_returns"] == 13140
    assert protocol["outer_full_model_replays"] == 1
    assert protocol["total_raw_logit_returns"] == 13141
    assert protocol["total_attention_mlp_calls_sites_0_3_each"] == 2941
    assert protocol["total_attention_mlp_calls_sites_4_17_each"] == 13141
    assert protocol["promotive_family"] == "uncalibrated_real_teacher_F"
    assert protocol["authorized_for_validation"] is False
    assert protocol["authorized_for_final"] is False
    assert protocol["authorized_for_global_ledger_credit"] is False
    assert protocol["authorized_for_fit_execution"] is False


def test_row_binding_uses_only_receipt_metadata_and_records_derangement_reuse(monkeypatch):
    opened = []
    original_load = life.torch.load

    def forbidden_load(*args, **kwargs):
        opened.append(args[0] if args else None)
        raise AssertionError("row tensor must not load while authority is constructed")

    monkeypatch.setattr(life.torch, "load", forbidden_load)
    binding = life.row_binding()
    monkeypatch.setattr(life.torch, "load", original_load)
    assert opened == []
    assert binding["row_count"] == 480
    assert len(binding["ordered_document_ids"]) == 209
    assert len(binding["row_to_document"]) == 480
    assert len(binding["document_deranged_donor_rows"]) == 480
    assert binding["logical_batch_reversal_same_document_count"] == 132
    assert len(binding["donor_row_reuse_multiplicities"]) == 480
    assert sum(binding["donor_row_reuse_multiplicities"]) == 480
    assert all(
        binding["row_to_document"][donor] != binding["row_to_document"][target]
        for target, donor in enumerate(binding["document_deranged_donor_rows"])
    )


def test_prior_binding_replays_failed_family_a_branch_without_final_access():
    binding = life.prior_artifact_binding()
    assert binding["registered_branch"] == {
        "kind": "stop_activation_family_and_preregister_finite_suffix_family",
        "budget": None,
    }
    life.verify_prior_artifact_binding(binding)


def test_verified_draft_is_explicitly_nonauthorizing(monkeypatch):
    source = {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64}
    prior = {"sha256": "c" * 64}
    rows = {"sha256": "d" * 64}
    checkpoint = life.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="e" * 64,
        weights_sha256="f" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=10,
    )
    monkeypatch.setattr(life, "require_pristine_namespace", lambda: None)
    monkeypatch.setattr(life, "source_closure", lambda: source)
    monkeypatch.setattr(life, "prior_artifact_binding", lambda: prior)
    monkeypatch.setattr(life, "row_binding", lambda: rows)
    monkeypatch.setattr(life.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    verified = []
    monkeypatch.setattr(
        life, "verify_frozen_inputs",
        lambda *args: verified.append(args),
    )
    observed = life.verified_draft_authority()
    assert len(verified) == 1
    assert observed["schema"].endswith("_draft")
    assert observed["authorized_for_fit_execution"] is False
    assert observed["status"].startswith("nonauthoritative_lifecycle_scaffold")
    assert observed["authority_sha256"] == life.logical_sha256({
        key: value for key, value in observed.items() if key != "authority_sha256"
    })


def test_pristine_namespace_rejects_any_spent_output(tmp_path):
    paths = tuple(tmp_path / name for name in ("authority", "programs", "result"))
    life.require_pristine_namespace(paths)
    paths[1].write_text(json.dumps({"spent": True}))
    with pytest.raises(RuntimeError, match="namespace is spent"):
        life.require_pristine_namespace(paths)
