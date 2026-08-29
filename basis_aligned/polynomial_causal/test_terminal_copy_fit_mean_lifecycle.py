import hashlib
import json

import pytest
import torch

import terminal_copy_fit_mean_lifecycle as life
from terminal_copy_fit_head_means import FitHeadMeanBank, NAMED_HEADS_BY_LAYER, _mean_bank_digest
from terminal_copy_fit_mean_owner import FitMeanOwnerClosure


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


def test_row_binding_is_fit_only_and_never_loads_tensor(monkeypatch):
    monkeypatch.setattr(life.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("row tensor opened while binding fit authority")
    ))
    binding = life.row_binding()
    assert binding["row_count"] == 192
    assert binding["authorized_use"] == "fit_per_position_head_write_means_only"
    assert binding["labels_or_copy_cells_authorized"] is False


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


def test_create_only_json_never_overwrites(tmp_path):
    path = tmp_path / "x.json"
    life.create_only_json(path, {"first": 1})
    with pytest.raises(FileExistsError):
        life.create_only_json(path, {"second": 2})
    assert json.loads(path.read_text()) == {"first": 1}
    assert not list(tmp_path.glob(".x.json.tmp-*"))


def test_publish_failure_is_terminal_and_never_creates_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(life, "BANK", tmp_path / "bank.pt")
    monkeypatch.setattr(life, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(life, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(life, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(life, "FAILURE", tmp_path / "failure.json")
    life.publish_failure("a" * 64, ValueError("expected"))
    assert not life.RECEIPT.exists()
    assert json.loads(life.FAILURE.read_text())["status"] == "terminal_failure_no_success_receipt"


def test_pristine_namespace_rejects_partial_transaction(tmp_path):
    paths = tuple(tmp_path / name for name in ("authority", "bank", "receipt"))
    life.require_pristine_namespace(paths)
    paths[1].write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="namespace is spent"):
        life.require_pristine_namespace(paths)
