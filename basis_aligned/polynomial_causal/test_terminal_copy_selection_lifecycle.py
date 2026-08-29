from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import terminal_copy_induction_v1 as contract
import terminal_copy_selection_lifecycle as life
from terminal_copy_attention_owner import CandidateOwnerClosure
from terminal_copy_selection_owner import (
    MergedSelectionBatches,
    MergedSyntheticBatches,
    SelectionBatchClosure,
    SyntheticBatchClosure,
    SyntheticPairEffect,
)
from terminal_copy_streaming_statistics import (
    CELL_NAMES,
    FROZEN_CANDIDATES,
    DocumentCellSums,
    SelectionResult,
)


def _candidate_closure(candidate: str, document_calls: int) -> dict:
    plan = life.PhysicalCandidateDispatcher.plan(candidate)
    selected = dict(plan)
    return {
        "candidate": candidate,
        "attempted_batch_calls": 1,
        "batch_calls": 1,
        "document_calls": document_calls,
        "native_attention_calls": [
            0 if layer in selected else 1 for layer in range(18)
        ],
        "adapter_attention_calls": [
            1 if layer in selected else 0 for layer in range(18)
        ],
        "native_mlp_calls": [1] * 18,
        "selected_layer_heads": [[layer, list(heads)] for layer, heads in plan],
        "maximum_head_recomposition_abs_error": 0.0,
        "maximum_head_recomposition_relative_error": 0.0,
        "closed": True,
    }


def _ledger_payload() -> dict:
    documents = tuple(f"d{index:03d}" for index in range(life.NATURAL_DOCUMENTS))
    candidates = {}
    for candidate_index, candidate in enumerate(FROZEN_CANDIDATES):
        candidates[candidate] = {}
        for document in documents:
            candidates[candidate][document] = {}
            for cell_index, cell in enumerate(CELL_NAMES):
                support = hashlib.sha256(f"{document}:{cell}".encode()).hexdigest()
                delta = (candidate_index + 1) * (1.0 if cell == "positive" else -0.1)
                candidates[candidate][document][cell] = {
                    "n": 1,
                    "native_nll_sum": 2.0 + cell_index,
                    "ablated_nll_sum": 2.0 + cell_index + delta,
                    "native_correct_count": 1,
                    "ablated_correct_count": 0,
                    "native_to_ablated_kl_sum": 0.1,
                    "support_sha256": support,
                }
    natural_closures = []
    for start in range(0, len(documents), life.NATURAL_BATCH_SIZE):
        natural_closures.append({
            "document_ids": list(documents[start:start + life.NATURAL_BATCH_SIZE]),
            "native_attention_calls": [1] * 18,
            "native_mlp_calls": [1] * 18,
            "native_unembedding_calls": 1,
            "candidate_unembedding_calls": [1] * len(FROZEN_CANDIDATES),
            "candidate_closures": [
                _candidate_closure(candidate, life.NATURAL_BATCH_SIZE)
                for candidate in FROZEN_CANDIDATES
            ],
            "raw_logits_returned": False,
            "closed": True,
        })
    item_ids = tuple(f"selection_synthetic_{index:03d}" for index in range(life.SYNTHETIC_PAIRS))
    synthetic_effects = {
        candidate: [
            {
                "item_id": item,
                "native_did": float(index),
                "candidate_did": float(index) + 0.5,
                "candidate_minus_native_did": 0.5,
            }
            for index, item in enumerate(item_ids)
        ]
        for candidate in FROZEN_CANDIDATES
    }
    synthetic_closures = []
    for start in range(0, len(item_ids), 2):
        synthetic_closures.append({
            "item_ids": list(item_ids[start:start + 2]),
            "native_attention_calls": [1] * 18,
            "native_mlp_calls": [1] * 18,
            "native_unembedding_calls": 1,
            "candidate_unembedding_calls": [1] * len(FROZEN_CANDIDATES),
            "candidate_closures": [
                _candidate_closure(candidate, life.SYNTHETIC_BATCH_SIZE)
                for candidate in FROZEN_CANDIDATES
            ],
            "raw_logits_returned": False,
            "closed": True,
        })
    return {
        "schema": "terminal_copy_selection_v1_ledger",
        "authority_sha256": "a" * 64,
        "ordered_document_ids": list(documents),
        "ordered_document_ids_sha256": life._document_digest(documents),
        "candidates": candidates,
        "natural_batch_closures": natural_closures,
        "synthetic_item_ids": list(item_ids),
        "synthetic_effects": synthetic_effects,
        "synthetic_batch_closures": synthetic_closures,
        "raw_logits_published": False,
    }


def _candidate_closure_object(candidate: str, document_calls: int) -> CandidateOwnerClosure:
    plan = life.PhysicalCandidateDispatcher.plan(candidate)
    selected = dict(plan)
    return CandidateOwnerClosure(
        candidate=candidate,
        attempted_batch_calls=1,
        batch_calls=1,
        document_calls=document_calls,
        native_attention_calls=tuple(
            0 if layer in selected else 1 for layer in range(18)
        ),
        adapter_attention_calls=tuple(
            1 if layer in selected else 0 for layer in range(18)
        ),
        native_mlp_calls=(1,) * 18,
        selected_layer_heads=plan,
        maximum_head_recomposition_abs_error=0.0,
        maximum_head_recomposition_relative_error=0.0,
        closed=True,
    )


def _materialized_transaction(selected_candidate: str | None):
    payload = _ledger_payload()
    documents = tuple(payload["ordered_document_ids"])
    ledgers = {
        candidate: {
            document: {
                cell: DocumentCellSums(**payload["candidates"][candidate][document][cell])
                for cell in CELL_NAMES
            }
            for document in documents
        }
        for candidate in FROZEN_CANDIDATES
    }
    natural_closures = []
    for start in range(0, len(documents), life.NATURAL_BATCH_SIZE):
        batch_documents = documents[start:start + life.NATURAL_BATCH_SIZE]
        natural_closures.append(SelectionBatchClosure(
            document_ids=batch_documents,
            native_attention_calls=(1,) * 18,
            native_mlp_calls=(1,) * 18,
            native_unembedding_calls=1,
            candidate_unembedding_calls=(1,) * len(FROZEN_CANDIDATES),
            candidate_closures=tuple(
                _candidate_closure_object(candidate, life.NATURAL_BATCH_SIZE)
                for candidate in FROZEN_CANDIDATES
            ),
            raw_logits_returned=False,
            closed=True,
        ))
    natural = MergedSelectionBatches(
        ledgers=ledgers,
        batch_closures=tuple(natural_closures),
        ordered_document_ids=documents,
    )
    item_ids = tuple(payload["synthetic_item_ids"])
    synthetic_effects = {
        candidate: tuple(
            SyntheticPairEffect(**value)
            for value in payload["synthetic_effects"][candidate]
        )
        for candidate in FROZEN_CANDIDATES
    }
    synthetic_closures = []
    for start in range(0, len(item_ids), 2):
        synthetic_closures.append(SyntheticBatchClosure(
            item_ids=item_ids[start:start + 2],
            native_attention_calls=(1,) * 18,
            native_mlp_calls=(1,) * 18,
            native_unembedding_calls=1,
            candidate_unembedding_calls=(1,) * len(FROZEN_CANDIDATES),
            candidate_closures=tuple(
                _candidate_closure_object(candidate, life.SYNTHETIC_BATCH_SIZE)
                for candidate in FROZEN_CANDIDATES
            ),
            raw_logits_returned=False,
            closed=True,
        ))
    synthetic = MergedSyntheticBatches(
        effects=synthetic_effects,
        batch_closures=tuple(synthetic_closures),
        ordered_item_ids=item_ids,
    )
    candidates = tuple(sorted(FROZEN_CANDIDATES))
    coordinate_names = tuple(
        f"{candidate}:{name}"
        for candidate in candidates
        for name in ("tau_positive", "specificity", "collateral_margin")
    )
    lower = torch.ones(24, dtype=torch.float64)
    if selected_candidate is None:
        lower.fill_(-1.0)
    selection = SelectionResult(
        candidates=candidates,
        coordinate_names=coordinate_names,
        point_estimates=torch.zeros(24, dtype=torch.float64),
        simultaneous_lower_bounds=lower,
        critical_value=0.0,
        selected_candidate=selected_candidate,
    )
    return natural, synthetic, selection


def _patch_output_namespace(monkeypatch, directory: Path):
    paths = {
        "AUTHORITY": directory / "authority.json",
        "LEDGER": directory / "ledger.json",
        "RESULT": directory / "result.json",
        "MANIFEST": directory / "manifest.json",
        "PASSER_RECEIPT": directory / "passer.json",
        "NEGATIVE_RECEIPT": directory / "negative.json",
        "FAILURE": directory / "failure.json",
        "LOCK": directory / "selection.lock",
        "AUDIT": directory / "audit.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(life, name, path)
    return paths


class _FakeTensor:
    def __getitem__(self, _key):
        return self

    def to(self, *args, **kwargs):
        return self


class _FakeDispatcher:
    def assert_matches_native(self, _attentions):
        return None


def _run_mocked_execute(
    tmp_path: Path, monkeypatch, selected_candidate: str | None,
    *, fail_during_natural: bool = False,
):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    weights = tmp_path / "snapshot" / "pytorch_model.bin"
    weights.parent.mkdir()
    weights.write_bytes(b"mock weights")
    weights_sha = life.file_sha256(weights)
    monkeypatch.setattr(life.facade, "DEFAULT_SNAPSHOT", weights.parent)
    monkeypatch.setattr(life.facade, "WEIGHTS_SHA256", weights_sha)
    checkpoint = life.facade.CheckpointReceipt(
        revision="mock", snapshot=str(weights.parent), config_sha256="c" * 64,
        weights_sha256=weights_sha, weights_bytes=len(b"mock weights"),
        tokenizer_vocab=50_257, logit_vocab=50_304,
    )
    authority = {
        "authority_sha256": "a" * 64,
        "checkpoint": life.asdict(checkpoint),
        "source_closure": {},
        "protected_snapshot": {"protected": "fixed"},
    }
    life.create_only_json(paths["AUTHORITY"], authority)
    natural, synthetic, selection = _materialized_transaction(selected_candidate)
    inputs = life.SelectionInputs(
        rows=_FakeTensor(), masks={cell: torch.zeros(192, 256, dtype=torch.bool) for cell in CELL_NAMES},
        ordered_document_ids=natural.ordered_document_ids,
        ordered_document_ids_sha256=life._document_digest(natural.ordered_document_ids),
        selection_file_sha256=life.SELECTION_PAYLOAD_SHA256,
        frequencies_file_sha256=life.FIT_FREQUENCIES_SHA256,
        synthetic_rows=_FakeTensor(),
        synthetic_item_ids=synthetic.ordered_item_ids,
        synthetic_query_positions=tuple(80 for _ in range(32)),
        synthetic_successor_y=tuple(1 for _ in range(32)),
        synthetic_successor_z=tuple(2 for _ in range(32)),
        expected_support_sha256s={},
    )
    fake_model = SimpleNamespace(
        transformer=SimpleNamespace(h=[SimpleNamespace(attn=object()) for _ in range(18)])
    )
    monkeypatch.setattr(life, "validate_execution_authority", lambda _authority: None)
    monkeypatch.setattr(life, "verify_source_closure", lambda _binding: None)
    monkeypatch.setattr(life, "protected_snapshot", lambda *args, **kwargs: {"protected": "fixed"})
    monkeypatch.setattr(life, "_load_selection_inputs", lambda _authority, _claim: inputs)
    monkeypatch.setattr(
        life, "_load_fit_bank",
        lambda _authority: SimpleNamespace(
            per_head_position_means={layer: _FakeTensor() for layer in life.NAMED_LAYERS}
        ),
    )
    monkeypatch.setattr(life.facade, "load_bilin18", lambda **kwargs: (fake_model, checkpoint))
    monkeypatch.setattr(life.facade, "validate_production_model", lambda _model: None)
    monkeypatch.setattr(life, "model_state_sha256", lambda _model: "m" * 64)
    monkeypatch.setattr(
        life.PhysicalCandidateDispatcher, "from_native", lambda **kwargs: _FakeDispatcher(),
    )
    monkeypatch.setattr(life, "_validate_exact_support", lambda *_args, **_kwargs: None)
    counters = {"natural": 0, "synthetic": 0}

    class FakeNaturalOwner:
        def __init__(self, _dispatcher):
            pass

        def run(self, *_args, **_kwargs):
            counters["natural"] += 1
            if fail_during_natural and counters["natural"] == 2:
                raise RuntimeError("injected partial natural forward")
            return None

    class FakeSyntheticOwner:
        def __init__(self, _dispatcher):
            pass

        def run(self, *_args, **_kwargs):
            counters["synthetic"] += 1
            return None

    monkeypatch.setattr(life, "SelectionBatchOwner", FakeNaturalOwner)
    monkeypatch.setattr(life, "SyntheticSelectionBatchOwner", FakeSyntheticOwner)
    monkeypatch.setattr(life, "merge_selection_batches", lambda *_args, **_kwargs: natural)
    monkeypatch.setattr(life, "merge_synthetic_batches", lambda *_args, **_kwargs: synthetic)
    monkeypatch.setattr(life, "simultaneous_selection_bootstrap", lambda *_args, **_kwargs: selection)
    return paths, counters, life.execute_selection


def test_protocol_freezes_literal_selection_and_call_census():
    protocol = life.protocol()
    assert protocol["natural_batches"] == 48
    assert protocol["synthetic_batches"] == 16
    assert protocol["total_outer_forwards"] == 576
    assert protocol["bootstrap_repetitions"] == 10_000
    assert protocol["bootstrap_seed"] == "terminal-copy-v1-document-bootstrap:0"
    assert protocol["critical_index_zero_based"] == 9_499
    assert protocol["candidates"] == list(FROZEN_CANDIDATES)
    assert protocol["all_mlps_native"] is True
    assert protocol["late_mlp_screen_omitted"] is True
    assert protocol["synthetic_role"] == "descriptive_only_no_selection_credit"


def test_serialized_ledger_replays_bootstrap_and_all_closures():
    payload = _ledger_payload()
    ledgers, documents = life._deserialize_ledger(payload, "a" * 64)
    assert len(documents) == 192
    assert set(ledgers) == set(FROZEN_CANDIDATES)
    assert all(len(ledgers[candidate]) == 192 for candidate in FROZEN_CANDIDATES)


def test_serialized_ledger_rejects_candidate_specific_native_baseline():
    payload = _ledger_payload()
    candidate = FROZEN_CANDIDATES[1]
    payload["candidates"][candidate]["d000"]["positive"]["native_nll_sum"] += 1.0
    with pytest.raises(RuntimeError, match="one native baseline"):
        life._deserialize_ledger(payload, "a" * 64)


def _write_selection_fixture(
    tmp_path: Path, *, corrupt_published_mask: bool = False,
    duplicate_bank_across_items: bool = False, bank_token_in_base_row: bool = False,
):
    generator = torch.Generator().manual_seed(4)
    rows = torch.randint(0, 101, (life.NATURAL_DOCUMENTS, 257), generator=generator)
    if bank_token_in_base_row:
        # Position 8 is one of the overwritten crossover slots.  The pure crossover
        # constructor permits this, but the amendment's token-bank contract does not.
        rows[0, life.SYNTHETIC_POSITION_TEMPLATES[0][0]] = 49_000
    documents = tuple(f"document-{index:03d}" for index in range(life.NATURAL_DOCUMENTS))
    records = [
        {"role": "selection_natural", "role_row_index": index, "document_id": document}
        for index, document in enumerate(documents)
    ]
    frequencies = contract.FitTokenFrequencies.from_rows(rows)
    cells = contract.build_copy_cells(rows, frequencies, documents)
    copy_cells = life._copy_cells_payload(cells)
    if corrupt_published_mask:
        copy_cells["positive"] = copy_cells["positive"].clone()
        copy_cells["positive"][0, 64] = ~copy_cells["positive"][0, 64]
    query_to_y, query_to_z, banks = [], [], []
    for index in range(life.SYNTHETIC_PAIRS):
        bank = [49_000 + 4 * index + offset for offset in range(4)]
        if duplicate_bank_across_items and index == 1:
            bank = [49_000 + offset for offset in range(4)]
        first, reciprocal, query = life.SYNTHETIC_POSITION_TEMPLATES[index % 4]
        crossover = contract.build_synthetic_association_crossover(
            tuple(int(value) for value in rows[index]),
            first_query_position=first, reciprocal_position=reciprocal,
            query_position=query, query_token=bank[0], reciprocal_query=bank[1],
            successor_y=bank[2], successor_z=bank[3],
        )
        query_to_y.append(crossover.query_to_y)
        query_to_z.append(crossover.query_to_z)
        banks.append(bank)
    synthetic = {
        "query_to_y": torch.stack(query_to_y),
        "query_to_z": torch.stack(query_to_z),
    }
    selection_path = tmp_path / "selection.pt"
    frequency_path = tmp_path / "frequencies.pt"
    torch.save({
        "rows": rows, "records": records, "synthetic": synthetic,
        "synthetic_token_banks": banks,
        "synthetic_position_templates": life.SYNTHETIC_POSITION_TEMPLATES,
        "copy_cells": copy_cells,
    }, selection_path)
    torch.save({"query": frequencies.query, "target": frequencies.target}, frequency_path)
    return rows, cells, synthetic, selection_path, frequency_path, documents


def test_loader_reconstructs_masks_and_synthetic_rows_independently(tmp_path, monkeypatch):
    rows, cells, synthetic, selection_path, frequency_path, documents = (
        _write_selection_fixture(tmp_path)
    )
    monkeypatch.setattr(life, "SELECTION_PAYLOAD_SHA256", life.file_sha256(selection_path))
    monkeypatch.setattr(life, "FIT_FREQUENCIES_SHA256", life.file_sha256(frequency_path))
    monkeypatch.setattr(life, "SELECTION_ROWS_SHA256", life.tensor_sha256(rows))
    monkeypatch.setattr(life, "SELECTION_POSITIVE_SHA256", life.tensor_sha256(cells.positive))
    monkeypatch.setattr(life, "SELECTION_NEGATIVE_SHA256", life.tensor_sha256(cells.matched_negative))
    monkeypatch.setattr(
        life, "SELECTION_SYNTHETIC_Y_SHA256", life.tensor_sha256(synthetic["query_to_y"]),
    )
    monkeypatch.setattr(
        life, "SELECTION_SYNTHETIC_Z_SHA256", life.tensor_sha256(synthetic["query_to_z"]),
    )
    frequencies = torch.load(frequency_path, weights_only=True)
    monkeypatch.setattr(
        life, "FIT_QUERY_FREQUENCY_SHA256", life.tensor_sha256(frequencies["query"]),
    )
    monkeypatch.setattr(
        life, "FIT_TARGET_FREQUENCY_SHA256", life.tensor_sha256(frequencies["target"]),
    )
    monkeypatch.setattr(life, "require_claim", lambda _claim: None)
    monkeypatch.setattr(life, "validate_execution_authority", lambda _authority: None)
    authority = {"row_binding": {
        "container_path": str(selection_path),
        "fit_frequencies_path": str(frequency_path),
        "support_census": life._support_census(cells),
    }}
    loaded = life._load_selection_inputs(authority, None)  # type: ignore[arg-type]
    assert torch.equal(loaded.rows, rows)
    assert loaded.ordered_document_ids == documents
    assert torch.equal(loaded.masks["positive"], cells.positive)
    assert torch.equal(loaded.masks["matched_negative"], cells.matched_negative)
    assert tuple(loaded.synthetic_rows.shape) == (64, 257)
    assert loaded.synthetic_item_ids[0] == "selection_synthetic_000"


def test_loader_rejects_published_mask_that_disagrees_with_independent_replay(
    tmp_path, monkeypatch,
):
    rows, cells, synthetic, selection_path, frequency_path, _ = _write_selection_fixture(
        tmp_path, corrupt_published_mask=True,
    )
    for name, value in {
        "SELECTION_PAYLOAD_SHA256": life.file_sha256(selection_path),
        "FIT_FREQUENCIES_SHA256": life.file_sha256(frequency_path),
        "SELECTION_ROWS_SHA256": life.tensor_sha256(rows),
        "SELECTION_POSITIVE_SHA256": life.tensor_sha256(cells.positive),
        "SELECTION_NEGATIVE_SHA256": life.tensor_sha256(cells.matched_negative),
        "SELECTION_SYNTHETIC_Y_SHA256": life.tensor_sha256(synthetic["query_to_y"]),
        "SELECTION_SYNTHETIC_Z_SHA256": life.tensor_sha256(synthetic["query_to_z"]),
    }.items():
        monkeypatch.setattr(life, name, value)
    frequencies = torch.load(frequency_path, weights_only=True)
    monkeypatch.setattr(life, "FIT_QUERY_FREQUENCY_SHA256", life.tensor_sha256(frequencies["query"]))
    monkeypatch.setattr(life, "FIT_TARGET_FREQUENCY_SHA256", life.tensor_sha256(frequencies["target"]))
    monkeypatch.setattr(life, "require_claim", lambda _claim: None)
    monkeypatch.setattr(life, "validate_execution_authority", lambda _authority: None)
    authority = {"row_binding": {
        "container_path": str(selection_path),
        "fit_frequencies_path": str(frequency_path),
        "support_census": life._support_census(cells),
    }}
    with pytest.raises(RuntimeError, match="mask reconstruction"):
        life._load_selection_inputs(authority, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fixture_options",
    [
        {"duplicate_bank_across_items": True},
        {"bank_token_in_base_row": True},
    ],
)
def test_loader_rejects_nonunique_or_base_row_present_synthetic_bank(
    tmp_path, monkeypatch, fixture_options,
):
    rows, cells, synthetic, selection_path, frequency_path, _ = _write_selection_fixture(
        tmp_path, **fixture_options,
    )
    for name, value in {
        "SELECTION_PAYLOAD_SHA256": life.file_sha256(selection_path),
        "FIT_FREQUENCIES_SHA256": life.file_sha256(frequency_path),
        "SELECTION_ROWS_SHA256": life.tensor_sha256(rows),
        "SELECTION_POSITIVE_SHA256": life.tensor_sha256(cells.positive),
        "SELECTION_NEGATIVE_SHA256": life.tensor_sha256(cells.matched_negative),
        "SELECTION_SYNTHETIC_Y_SHA256": life.tensor_sha256(synthetic["query_to_y"]),
        "SELECTION_SYNTHETIC_Z_SHA256": life.tensor_sha256(synthetic["query_to_z"]),
    }.items():
        monkeypatch.setattr(life, name, value)
    frequencies = torch.load(frequency_path, weights_only=True)
    monkeypatch.setattr(life, "FIT_QUERY_FREQUENCY_SHA256", life.tensor_sha256(frequencies["query"]))
    monkeypatch.setattr(life, "FIT_TARGET_FREQUENCY_SHA256", life.tensor_sha256(frequencies["target"]))
    monkeypatch.setattr(life, "require_claim", lambda _claim: None)
    monkeypatch.setattr(life, "validate_execution_authority", lambda _authority: None)
    authority = {"row_binding": {
        "container_path": str(selection_path),
        "fit_frequencies_path": str(frequency_path),
        "support_census": life._support_census(cells),
    }}
    with pytest.raises(RuntimeError, match="cross-item unique and base-row absent"):
        life._load_selection_inputs(authority, None)  # type: ignore[arg-type]


def test_model_state_hash_handles_bfloat16_and_detects_mutation():
    model = torch.nn.Linear(4, 3, bias=True).to(dtype=torch.bfloat16)
    model.register_buffer("scalar_bfloat16", torch.tensor(1.25, dtype=torch.bfloat16))
    before = life.model_state_sha256(model)
    with torch.no_grad():
        model.bias[0] += 1
    assert life.model_state_sha256(model) != before


def test_tensor_hash_handles_scalar_bfloat16_and_encodes_shape():
    scalar = torch.tensor(1.25, dtype=torch.bfloat16)
    vector = scalar.reshape(1)
    assert life.tensor_sha256(scalar) != life.tensor_sha256(vector)


def test_create_only_json_never_overwrites(tmp_path):
    path = tmp_path / "artifact.json"
    life.create_only_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        life.create_only_json(path, {"a": 2})
    assert life.stable_json(path) == {"a": 1}


def test_mocked_authority_freeze_is_create_only_and_validates(tmp_path, monkeypatch):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    paths["AUDIT"].write_text("{}\n")
    audit = {"approved": True, "outcome_access": False}
    source = {"commit": "c" * 40, "paths": {}, "sha256": "s" * 64}
    row = {"row": "bound"}
    fit = {"fit": "bound"}
    adapter = {"adapter": "bound"}
    checkpoint = life.facade.CheckpointReceipt(
        revision="mock", snapshot="/mock", config_sha256="c" * 64,
        weights_sha256="w" * 64, weights_bytes=1,
        tokenizer_vocab=50_257, logit_vocab=50_304,
    )
    monkeypatch.setattr(life, "validate_canonical_audit", lambda _path=paths["AUDIT"]: audit)
    monkeypatch.setattr(life, "source_closure", lambda: source)
    monkeypatch.setattr(life, "verify_source_closure", lambda binding: binding == source or None)
    monkeypatch.setattr(life, "row_binding", lambda: row)
    monkeypatch.setattr(life.fit_parent, "replay_fit_parent", lambda: fit)
    monkeypatch.setattr(life, "adapter_binding", lambda: adapter)
    monkeypatch.setattr(life.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"frozen")
    monkeypatch.setattr(life, "PROTECTED_PATHS", (protected,))
    authority = life.freeze_execution_authority(paths["AUDIT"])
    assert authority["authorized_for_selection_execution"] is True
    assert authority["authorized_for_final_ood"] is False
    assert authority["fit_receipt_self_authorizes_selection"] is False
    assert authority["amendment_governs_conflicts"] is True
    assert authority["protected_snapshot"] == {str(protected): life.file_sha256(protected)}
    assert life.stable_json(paths["AUTHORITY"]) == authority
    with pytest.raises(RuntimeError, match="spent"):
        life.freeze_execution_authority(paths["AUDIT"])


def test_authority_rejects_protected_mutation_between_freeze_and_execute(
    tmp_path, monkeypatch,
):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    paths["AUDIT"].write_text("{}\n")
    protected = tmp_path / "final_ood_payload.pt"
    protected.write_bytes(b"before")
    source = {"commit": "c" * 40, "paths": {}, "sha256": "s" * 64}
    checkpoint = life.facade.CheckpointReceipt(
        revision="mock", snapshot="/mock", config_sha256="c" * 64,
        weights_sha256="w" * 64, weights_bytes=1,
        tokenizer_vocab=50_257, logit_vocab=50_304,
    )
    monkeypatch.setattr(life, "PROTECTED_PATHS", (protected,))
    monkeypatch.setattr(
        life, "validate_canonical_audit",
        lambda _path=paths["AUDIT"]: {"approved": True, "outcome_access": False},
    )
    monkeypatch.setattr(life, "source_closure", lambda: source)
    monkeypatch.setattr(life, "verify_source_closure", lambda binding: binding == source or None)
    monkeypatch.setattr(life, "row_binding", lambda: {"row": "bound"})
    monkeypatch.setattr(life.fit_parent, "replay_fit_parent", lambda: {"fit": "bound"})
    monkeypatch.setattr(life, "adapter_binding", lambda: {"adapter": "bound"})
    monkeypatch.setattr(life.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    authority = life.freeze_execution_authority(paths["AUDIT"])
    protected.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="authority identity changed"):
        life.validate_execution_authority(authority)


@pytest.mark.parametrize(
    ("selected_candidate", "receipt_key", "other_key", "authorized"),
    [
        ("L5H5", "PASSER_RECEIPT", "NEGATIVE_RECEIPT", True),
        (None, "NEGATIVE_RECEIPT", "PASSER_RECEIPT", False),
    ],
)
def test_mocked_full_execute_publishes_exactly_one_receipt_last(
    tmp_path, monkeypatch, selected_candidate, receipt_key, other_key, authorized,
):
    paths, counters, execute = _run_mocked_execute(
        tmp_path, monkeypatch, selected_candidate,
    )
    receipt = execute()
    assert counters == {"natural": 48, "synthetic": 16}
    assert paths["LEDGER"].is_file()
    assert paths["RESULT"].is_file()
    assert paths["MANIFEST"].is_file()
    assert paths[receipt_key].is_file()
    assert not paths[other_key].exists()
    assert not paths["FAILURE"].exists()
    assert not paths["LOCK"].exists()
    assert receipt["selected_candidate"] == selected_candidate
    assert receipt["final_ood_opening_authorized"] is authorized
    assert receipt["negative_forbids_final_ood_opening"] is (not authorized)
    assert receipt["ledger_file_sha256"] == life.file_sha256(paths["LEDGER"])
    assert receipt["result_file_sha256"] == life.file_sha256(paths["RESULT"])
    assert receipt["manifest_file_sha256"] == life.file_sha256(paths["MANIFEST"])


def test_partial_forward_publishes_failure_not_scientific_negative(tmp_path, monkeypatch):
    paths, counters, execute = _run_mocked_execute(
        tmp_path, monkeypatch, None, fail_during_natural=True,
    )
    with pytest.raises(RuntimeError, match="injected partial"):
        execute()
    failure = life.stable_json(paths["FAILURE"])
    assert counters == {"natural": 2, "synthetic": 0}
    assert failure["status"] == "terminal_integrity_or_execution_failure_no_decision_receipt"
    assert failure["same_authority_retry_authorized"] is False
    assert failure["decision_receipts_mutually_absent"] is True
    assert not paths["PASSER_RECEIPT"].exists()
    assert not paths["NEGATIVE_RECEIPT"].exists()
    assert not paths["LOCK"].exists()
    with pytest.raises(RuntimeError, match="spent"):
        execute()


def test_late_protected_mutation_preserves_hashed_partial_outputs_and_no_decision(
    tmp_path, monkeypatch,
):
    paths, _, execute = _run_mocked_execute(tmp_path, monkeypatch, "L5H5")
    calls = {"count": 0}

    def changing_snapshot(*_args, **_kwargs):
        calls["count"] += 1
        return {"protected": "fixed" if calls["count"] <= 4 else "mutated"}

    monkeypatch.setattr(life, "protected_snapshot", changing_snapshot)
    with pytest.raises(RuntimeError, match="terminal publication replay"):
        execute()
    failure = life.stable_json(paths["FAILURE"])
    assert all(paths[name].is_file() for name in ("LEDGER", "RESULT", "MANIFEST"))
    assert not paths["PASSER_RECEIPT"].exists()
    assert not paths["NEGATIVE_RECEIPT"].exists()
    for name in ("LEDGER", "RESULT", "MANIFEST"):
        entry = failure["partial_artifacts"][str(paths[name])]
        assert entry["file_sha256"] == life.file_sha256(paths[name])
        assert entry["stable_read"] is True
        assert entry["joins_failed_authority"] is True
    assert failure["protected_at_failure"] == {"protected": "mutated"}


def test_lock_inode_replacement_is_detected(tmp_path, monkeypatch):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    claim = life.acquire_claim()
    paths["LOCK"].unlink()
    paths["LOCK"].write_text(claim.nonce + "\n")
    with pytest.raises(RuntimeError, match="ownership changed"):
        life.require_claim(claim)
    life.release_claim(claim)
    paths["LOCK"].unlink()


def test_fit_bank_load_rejects_parent_binding_reserialization(monkeypatch):
    monkeypatch.setattr(life.fit_parent, "replay_fit_parent", lambda: {"fit": "changed"})
    with pytest.raises(RuntimeError, match="binding changed"):
        life._load_fit_bank({"fit_parent_binding": {"fit": "authorized"}})


def test_failure_cannot_coexist_with_a_decision_receipt(tmp_path, monkeypatch):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    life.create_only_json(paths["AUTHORITY"], {"authority_sha256": "a" * 64})
    life.create_only_json(paths["PASSER_RECEIPT"], {"status": "already complete"})
    claim = life.acquire_claim()
    try:
        with pytest.raises(RuntimeError, match="after a decision receipt"):
            life._publish_failure(claim, "a" * 64, RuntimeError("late"))
        assert not paths["FAILURE"].exists()
    finally:
        life.release_claim(claim)


def test_failure_rejects_receipt_race_after_initial_gate(tmp_path, monkeypatch):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    life.create_only_json(paths["AUTHORITY"], {"authority_sha256": "a" * 64})
    claim = life.acquire_claim()
    original = life._failure_input_snapshot
    calls = {"count": 0}

    def racing_snapshot(authority_sha256):
        value = original(authority_sha256)
        calls["count"] += 1
        if calls["count"] == 2:
            life.create_only_json(paths["NEGATIVE_RECEIPT"], {"status": "late race"})
        return value

    monkeypatch.setattr(life, "_failure_input_snapshot", racing_snapshot)
    try:
        with pytest.raises(RuntimeError, match="exclusivity changed"):
            life._publish_failure(claim, "a" * 64, RuntimeError("late"))
        assert not paths["FAILURE"].exists()
    finally:
        life.release_claim(claim)


def test_failure_rejects_partial_artifact_race_during_terminal_rehash(
    tmp_path, monkeypatch,
):
    paths = _patch_output_namespace(monkeypatch, tmp_path)
    life.create_only_json(paths["AUTHORITY"], {"authority_sha256": "a" * 64})
    life.create_only_json(paths["LEDGER"], {
        "schema": "terminal_copy_selection_v1_ledger", "authority_sha256": "a" * 64,
    })
    claim = life.acquire_claim()
    original = life._failure_input_snapshot
    calls = {"count": 0}

    def racing_snapshot(authority_sha256):
        calls["count"] += 1
        if calls["count"] == 2:
            paths["LEDGER"].write_text('{"mutated":true}\n')
        return original(authority_sha256)

    monkeypatch.setattr(life, "_failure_input_snapshot", racing_snapshot)
    try:
        with pytest.raises(RuntimeError, match="terminal inputs"):
            life._publish_failure(claim, "a" * 64, RuntimeError("late"))
        assert not paths["FAILURE"].exists()
    finally:
        life.release_claim(claim)
