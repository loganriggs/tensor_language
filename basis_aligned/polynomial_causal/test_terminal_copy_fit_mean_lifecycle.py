import hashlib
import json

import pytest
import torch

import terminal_copy_fit_mean_lifecycle as life
from terminal_copy_fit_head_means import FitHeadMeanBank, NAMED_HEADS_BY_LAYER, _mean_bank_digest
from terminal_copy_fit_mean_owner import FitMeanOwnerClosure


def _write_audit(path):
    path.write_text(json.dumps({
        "schema": "terminal_copy_fit_mean_lifecycle_independent_audit_v1",
        "status": "approved_outcome_blind_infrastructure",
        "approved": True,
        "outcome_access": False,
        "reviewer": "independent_artifact_audit_agent",
        "reviewed_source_sha256s": {
            "basis_aligned/polynomial_causal/terminal_copy_fit_mean_lifecycle.py":
                life.file_sha256(life.Path(life.__file__).resolve()),
            "basis_aligned/polynomial_causal/test_terminal_copy_fit_mean_lifecycle.py":
                life.file_sha256(life.HERE / "test_terminal_copy_fit_mean_lifecycle.py"),
        },
        "focused_tests": {"passed": True, "count": 1},
        "remaining_launch_blockers": [],
    }))


def _install_synthetic_fit_input(monkeypatch, tmp_path):
    tokens = torch.arange(192 * 256, dtype=torch.long).reshape(192, 256)
    documents = tuple(f"doc-{index}" for index in range(192))
    authority_body = {
        "status": "frozen_before_any_E4_fit_model_forward_after_disclosed_parent_container_access",
        "parent_binding": {"receipt_sha256": life.ROW_RECEIPT_SHA256},
    }
    projection_authority = {
        **authority_body,
        "authority_sha256": life.logical_sha256(authority_body),
    }
    projection_authority_path = tmp_path / "fit_input_authority.json"
    projection_authority_path.write_text(json.dumps(projection_authority))
    input_path = tmp_path / "fit_inputs.pt"
    payload = {
        "schema": "terminal_copy_fit_inputs_v1_payload",
        "authority_sha256": projection_authority["authority_sha256"],
        "tokens": tokens,
        "ordered_document_ids": documents,
        "tokens_sha256": life.tensor_sha256(tokens),
        "ordered_document_ids_sha256": life._document_digest(documents),
    }
    torch.save(payload, input_path)
    projection_receipt = {
        "status": "complete_receipt_last_input_only_no_model_access",
        "parent_row_receipt_sha256": life.ROW_RECEIPT_SHA256,
        "erratum_sha256": life.file_sha256(life.FIT_INPUT_ERRATUM),
        "authority_file_sha256": life.file_sha256(projection_authority_path),
        "authority_sha256": projection_authority["authority_sha256"],
        "tokens_shape": [192, 256],
        "payload_fields": sorted(payload),
        "label_copy_cell_synthetic_fields_absent": True,
        "E4_fit_model_forward_calls": 0,
        "scientific_outcomes_read": False,
        "authorized_for_fit_mean_input_only": True,
        "authorized_for_candidate_selection": False,
        "inputs_path": str(input_path),
        "inputs_file_sha256": life.file_sha256(input_path),
        "tokens_sha256": payload["tokens_sha256"],
        "ordered_document_ids_sha256": payload["ordered_document_ids_sha256"],
    }
    projection_manifest = {
        "inputs_file_sha256": projection_receipt["inputs_file_sha256"],
    }
    projection_manifest_path = tmp_path / "fit_input_manifest.json"
    projection_manifest_path.write_text(json.dumps(projection_manifest))
    projection_receipt["manifest_file_sha256"] = life.file_sha256(projection_manifest_path)
    projection_receipt_path = tmp_path / "fit_input_receipt.json"
    projection_receipt_path.write_text(json.dumps(projection_receipt))
    monkeypatch.setattr(life, "FIT_INPUT_AUTHORITY", projection_authority_path)
    monkeypatch.setattr(life, "FIT_INPUT_RECEIPT", projection_receipt_path)
    monkeypatch.setattr(life, "FIT_INPUT_MANIFEST", projection_manifest_path)
    monkeypatch.setattr(life, "FIT_INPUTS", input_path)
    monkeypatch.setattr(
        life.fit_input_projection,
        "validate_published_metadata",
        lambda: {
            "authority": projection_authority,
            "manifest": projection_manifest,
            "receipt": projection_receipt,
            "input_file_sha256": projection_receipt["inputs_file_sha256"],
        },
    )
    return payload


def _bank(sequence_length=3, width=4):
    master = {
        layer: torch.arange(
            sequence_length * len(NAMED_HEADS_BY_LAYER[layer]) * width,
            dtype=torch.float64,
        ).reshape(sequence_length, len(NAMED_HEADS_BY_LAYER[layer]), width) / 7
        for layer in life.NAMED_LAYERS
    }
    runtime = {layer: value.float() for layer, value in master.items()}
    document_hash = "d" * 64
    return FitHeadMeanBank(
        per_head_position_means=runtime,
        master_per_head_position_means=master,
        document_count=2,
        ordered_document_ids_sha256=document_hash,
        runtime_means_sha256=_mean_bank_digest(document_hash, runtime),
        master_means_sha256=_mean_bank_digest(document_hash, master),
        accumulator_dtype="torch.float64",
        published_dtype="torch.float32",
        source_dtype="torch.bfloat16",
    )


def _closure(batches=2):
    return FitMeanOwnerClosure(
        batch_calls=batches,
        document_calls=192,
        native_attention_calls=(batches,) * 18,
        adapter_decomposition_calls=tuple(
            batches if layer in life.NAMED_LAYERS else 0 for layer in range(18)
        ),
        native_mlp_calls=(batches,) * 18,
        native_unembedding_calls=0,
        maximum_full_write_abs_error=0.0,
        maximum_head_recomposition_abs_error=0.5,
        maximum_head_recomposition_relative_error=0.0027,
        final_state_sha256s=tuple(hashlib.sha256(str(i).encode()).hexdigest() for i in range(batches)),
        closed=True,
    )


def test_row_binding_is_fit_only_and_never_loads_tensor(monkeypatch, tmp_path):
    _install_synthetic_fit_input(monkeypatch, tmp_path)
    monkeypatch.setattr(life.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("row tensor opened while binding fit authority")
    ))
    binding = life.row_binding()
    assert binding["row_count"] == 192
    assert binding["authorized_use"] == "fit_per_position_head_write_means_only"
    assert binding["labels_copy_cells_synthetic_present"] is False


def test_adapter_binding_is_engineering_only_and_exact_checkpoint():
    binding = life.adapter_binding()
    assert binding["layers"] == [5, 7, 8, 13, 14]
    assert binding["checkpoint_weights_sha256"] == life.facade.WEIGHTS_SHA256
    assert binding["scientific_claim_authorized_by_parent"] is False


def test_protocol_has_no_logits_labels_or_selection():
    protocol = life.protocol()
    assert protocol["heads_by_layer"] == {
        "5": [5], "7": [3], "8": [3, 4], "13": [0], "14": [7],
    }
    assert protocol["unembedding_calls"] == 0
    assert protocol["loss_or_logit_reads"] == 0
    assert protocol["label_or_copy_cell_reads"] == 0
    assert protocol["authorized_for_candidate_selection"] is False
    assert protocol["authorized_for_E4_evidence"] is False


def test_semantic_reload_reconstructs_hash_verified_bank(tmp_path):
    bank = _bank()
    authority_sha = "a" * 64
    path = tmp_path / "bank.pt"
    torch.save(life._bank_payload(bank, authority_sha), path)
    replay = life.load_bank_semantically(path, authority_sha, require_production=False)
    assert replay.verify_hashes()
    assert replay.master_means_sha256 == bank.master_means_sha256
    assert replay.runtime_means_sha256 == bank.runtime_means_sha256


def test_semantic_reload_rejects_tensor_tampering(tmp_path):
    bank = _bank()
    authority_sha = "a" * 64
    payload = life._bank_payload(bank, authority_sha)
    payload["runtime"]["8"][0, 0, 0] += 1
    path = tmp_path / "bank.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="tensor semantics|hash replay"):
        life.load_bank_semantically(path, authority_sha, require_production=False)


def test_validate_closure_enforces_exact_physical_census():
    observed = life.validate_closure(_closure())
    assert observed["native_unembedding_calls"] == 0
    broken = _closure()
    object.__setattr__(broken, "document_calls", 191)
    with pytest.raises(RuntimeError, match="call census"):
        life.validate_closure(broken)


def test_execution_authority_replays_every_parent(monkeypatch, tmp_path):
    audit = tmp_path / "audit.json"
    _write_audit(audit)
    checkpoint = life.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="c" * 64,
        weights_sha256="w" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=11,
    )
    monkeypatch.setattr(life, "AUTHORITY", tmp_path / "authority.json")
    monkeypatch.setattr(life, "AUDIT", audit)
    monkeypatch.setattr(life, "PROTECTED_PATHS", ())
    monkeypatch.setattr(life, "row_binding", lambda: {"sha256": "r" * 64})
    monkeypatch.setattr(life, "adapter_binding", lambda: {"sha256": "p" * 64})
    monkeypatch.setattr(life, "protocol", lambda: {"fit": True})
    monkeypatch.setattr(life, "verify_source_closure", lambda value: None)
    monkeypatch.setattr(life.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    outputs = {name: str(path) for name, path in {
        "authority": life.AUTHORITY, "bank": life.BANK, "result": life.RESULT,
        "manifest": life.MANIFEST, "receipt": life.RECEIPT,
        "failure": life.FAILURE, "lock": life.LOCK,
    }.items()}
    body = {
        "schema": "terminal_copy_fit_means_v1_authority",
        "status": "frozen_before_any_fit_model_forward_after_disclosed_parent_container_access",
        "source_closure": {"closed": True},
        "row_binding": life.row_binding(),
        "adapter_binding": life.adapter_binding(),
        "checkpoint": life.asdict(checkpoint),
        "protocol": life.protocol(),
        "outputs": outputs,
        "protected_paths": [],
        "independent_audit": {
            "approved": True, "outcome_access": False,
            "path": str(audit), "sha256": life.file_sha256(audit),
        },
        "authorized_for_fit_execution": True,
        "authorized_for_candidate_selection": False,
        "authorized_for_scored_experiments": False,
    }
    authority = {**body, "authority_sha256": life.logical_sha256(body)}
    life.create_only_json(life.AUTHORITY, authority)
    life.validate_execution_authority(authority)
    changed = dict(authority)
    changed["authorized_for_candidate_selection"] = True
    with pytest.raises(RuntimeError, match="authority identity"):
        life.validate_execution_authority(changed)


def test_freeze_authority_is_create_only_and_fit_only(monkeypatch, tmp_path):
    audit = tmp_path / "audit.json"
    _write_audit(audit)
    authority_path = tmp_path / "authority.json"
    checkpoint = life.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="c" * 64,
        weights_sha256="w" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=11,
    )
    monkeypatch.setattr(life, "AUTHORITY", authority_path)
    monkeypatch.setattr(life, "AUDIT", audit)
    monkeypatch.setattr(life, "PROTECTED_PATHS", ())
    monkeypatch.setattr(life, "require_pristine_namespace", lambda: None)
    monkeypatch.setattr(life, "source_closure", lambda: {"closed": True})
    monkeypatch.setattr(life, "row_binding", lambda: {"sha256": "r" * 64})
    monkeypatch.setattr(life, "adapter_binding", lambda: {"sha256": "p" * 64})
    monkeypatch.setattr(life, "protocol", lambda: {"fit": True})
    monkeypatch.setattr(life.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    validated = []
    monkeypatch.setattr(life, "validate_execution_authority", lambda value: validated.append(value))
    authority = life.freeze_execution_authority(audit)
    assert validated == [authority]
    assert authority["authorized_for_fit_execution"] is True
    assert authority["authorized_for_candidate_selection"] is False
    assert authority["authorized_for_scored_experiments"] is False
    assert json.loads(authority_path.read_text()) == authority


def test_create_only_json_never_overwrites(tmp_path):
    path = tmp_path / "x.json"
    life.create_only_json(path, {"first": 1})
    with pytest.raises(FileExistsError):
        life.create_only_json(path, {"second": 2})
    assert json.loads(path.read_text()) == {"first": 1}
    assert not list(tmp_path.glob(".x.json.tmp-*"))


def test_execution_lock_is_exclusive_and_detects_replacement(tmp_path):
    lock = tmp_path / "lock"
    claim = life.acquire_claim(lock)
    try:
        with pytest.raises(RuntimeError, match="locked"):
            life.acquire_claim(lock)
        life.require_claim(claim, lock)
        lock.unlink()
        lock.write_text(claim.nonce + "\n")
        with pytest.raises(RuntimeError, match="ownership changed"):
            life.require_claim(claim, lock)
    finally:
        life.release_claim(claim, lock)


def test_fit_input_loader_joins_synthetic_input_only_bytes_to_document_order(monkeypatch, tmp_path):
    _install_synthetic_fit_input(monkeypatch, tmp_path)
    lock = tmp_path / "lock"
    monkeypatch.setattr(life, "LOCK", lock)
    monkeypatch.setattr(life, "validate_execution_authority", lambda value: None)
    binding = life.row_binding()
    claim = life.acquire_claim()
    try:
        loaded = life._load_fit_role_inputs({"row_binding": binding}, claim)
    finally:
        life.release_claim(claim)
    assert loaded.tokens.shape == (192, 256)
    assert len(loaded.ordered_document_ids) == 192
    assert len(set(loaded.ordered_document_ids)) == 192
    assert loaded.ordered_document_ids_sha256 == life._document_digest(
        loaded.ordered_document_ids
    )
    assert loaded.row_file_sha256 == binding["input_file_sha256"]


def test_collection_capability_requires_private_seal():
    with pytest.raises(RuntimeError, match="cannot be constructed externally"):
        life._CollectedFitTransaction(
            object(), bank=_bank(), closure=_closure(),
            ordered_document_ids_sha256="d" * 64,
            row_file_sha256="f" * 64,
            checkpoint_weights_sha256_before="w" * 64,
            checkpoint_weights_sha256_after="w" * 64,
            claim=life.RunClaim(-1, -1, "n"), authority_sha256="a" * 64,
        )


def test_publish_failure_is_terminal_and_never_creates_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(life, "BANK", tmp_path / "bank.pt")
    monkeypatch.setattr(life, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(life, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(life, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(life, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(life, "LOCK", tmp_path / "lock")
    claim = life.acquire_claim()
    try:
        life._publish_failure(claim, "a" * 64, ValueError("expected"))
        assert not life.RECEIPT.exists()
        assert json.loads(life.FAILURE.read_text())["status"] == "terminal_failure_no_success_receipt"
    finally:
        life.release_claim(claim)


def test_pristine_namespace_rejects_partial_transaction(tmp_path):
    paths = tuple(tmp_path / name for name in ("authority", "bank", "receipt"))
    life.require_pristine_namespace(paths)
    paths[1].write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="namespace is spent"):
        life.require_pristine_namespace(paths)


def _install_publish_transaction(monkeypatch, tmp_path):
    monkeypatch.setattr(life, "AUTHORITY", tmp_path / "authority.json")
    monkeypatch.setattr(life, "BANK", tmp_path / "bank.pt")
    monkeypatch.setattr(life, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(life, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(life, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(life, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(life, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(life, "validate_execution_authority", lambda value: None)
    stable_protected = {"parent": "p" * 64}
    monkeypatch.setattr(life, "protected_snapshot", lambda: dict(stable_protected))
    original_load = life.load_bank_semantically
    monkeypatch.setattr(
        life, "load_bank_semantically",
        lambda path, authority_sha256, require_production=True: original_load(
            path, authority_sha256, require_production=False,
        ),
    )
    authority_body = {"row_binding": {"input_file_sha256": "f" * 64}}
    authority = {
        **authority_body,
        "authority_sha256": life.logical_sha256(authority_body),
    }
    life.create_only_json(life.AUTHORITY, authority)
    claim = life.acquire_claim()
    bank = _bank()
    collected = life._CollectedFitTransaction(
        life._COLLECTION_SEAL,
        bank=bank,
        closure=_closure(),
        ordered_document_ids_sha256=bank.ordered_document_ids_sha256,
        row_file_sha256="f" * 64,
        checkpoint_weights_sha256_before=life.facade.WEIGHTS_SHA256,
        checkpoint_weights_sha256_after=life.facade.WEIGHTS_SHA256,
        claim=claim,
        authority_sha256=authority["authority_sha256"],
    )
    return authority, claim, collected, stable_protected


def _publish(monkeypatch, tmp_path):
    authority, claim, collected, protected = _install_publish_transaction(
        monkeypatch, tmp_path,
    )
    return authority, claim, collected, protected


def test_full_publish_transaction_is_receipt_last(monkeypatch, tmp_path):
    authority, claim, collected, protected = _publish(monkeypatch, tmp_path)
    try:
        receipt = life._publish_fit_mean_bundle(
            authority=authority, claim=claim, collected=collected,
            protected_before=protected, protected_after=protected,
        )
        assert life.RECEIPT.exists()
        assert not life.FAILURE.exists()
        assert receipt["status"] == "complete_receipt_last_fit_only"
        assert receipt["authorized_for_candidate_selection_parent"] is False
    finally:
        life.release_claim(claim)


def test_full_publish_detects_late_protected_mutation(monkeypatch, tmp_path):
    authority, claim, collected, protected = _publish(monkeypatch, tmp_path)
    calls = 0

    def changing_snapshot():
        nonlocal calls
        calls += 1
        return protected if calls < 5 else {"parent": "changed"}

    monkeypatch.setattr(life, "protected_snapshot", changing_snapshot)
    try:
        with pytest.raises(RuntimeError, match="adjacent receipt gate"):
            life._publish_fit_mean_bundle(
                authority=authority, claim=claim, collected=collected,
                protected_before=protected, protected_after=protected,
            )
        assert not life.RECEIPT.exists()
    finally:
        life.release_claim(claim)


def test_full_publish_detects_late_lock_replacement(monkeypatch, tmp_path):
    authority, claim, collected, protected = _publish(monkeypatch, tmp_path)
    original = life.load_bank_semantically
    calls = 0

    def replace_on_adjacent_reload(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 3:
            life.LOCK.unlink()
            life.LOCK.write_text("replacement\n")
        return result

    monkeypatch.setattr(life, "load_bank_semantically", replace_on_adjacent_reload)
    try:
        with pytest.raises(RuntimeError, match="lock ownership changed"):
            life._publish_fit_mean_bundle(
                authority=authority, claim=claim, collected=collected,
                protected_before=protected, protected_after=protected,
            )
        assert not life.RECEIPT.exists()
    finally:
        life.release_claim(claim)


def test_full_publish_detects_bank_reserialization_after_manifest(monkeypatch, tmp_path):
    authority, claim, collected, protected = _publish(monkeypatch, tmp_path)
    original = life.create_only_json

    def reserialize_after_manifest(path, value):
        original(path, value)
        if path == life.MANIFEST:
            payload = torch.load(life.BANK, map_location="cpu", weights_only=True)
            torch.save(payload, life.BANK)

    monkeypatch.setattr(life, "create_only_json", reserialize_after_manifest)
    try:
        with pytest.raises(RuntimeError, match="terminal publication recheck"):
            life._publish_fit_mean_bundle(
                authority=authority, claim=claim, collected=collected,
                protected_before=protected, protected_after=protected,
            )
        assert not life.RECEIPT.exists()
    finally:
        life.release_claim(claim)


def test_full_publish_injected_post_manifest_failure_has_no_receipt(monkeypatch, tmp_path):
    authority, claim, collected, protected = _publish(monkeypatch, tmp_path)
    original = life.create_only_json

    def fail_after_manifest(path, value):
        original(path, value)
        if path == life.MANIFEST:
            raise RuntimeError("injected post-manifest failure")

    monkeypatch.setattr(life, "create_only_json", fail_after_manifest)
    try:
        with pytest.raises(RuntimeError, match="injected post-manifest"):
            life._publish_fit_mean_bundle(
                authority=authority, claim=claim, collected=collected,
                protected_before=protected, protected_after=protected,
            )
        assert not life.RECEIPT.exists()
    finally:
        life.release_claim(claim)
