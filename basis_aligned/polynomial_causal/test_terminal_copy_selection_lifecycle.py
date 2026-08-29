from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import terminal_copy_induction_v1 as contract
import terminal_copy_selection_lifecycle as life
from terminal_copy_streaming_statistics import CELL_NAMES, FROZEN_CANDIDATES


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


def _write_selection_fixture(tmp_path: Path, *, corrupt_published_mask: bool = False):
    generator = torch.Generator().manual_seed(4)
    rows = torch.randint(0, 101, (life.NATURAL_DOCUMENTS, 257), generator=generator)
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


def test_model_state_hash_handles_bfloat16_and_detects_mutation():
    model = torch.nn.Linear(4, 3, bias=True).to(dtype=torch.bfloat16)
    before = life.model_state_sha256(model)
    with torch.no_grad():
        model.bias[0] += 1
    assert life.model_state_sha256(model) != before


def test_create_only_json_never_overwrites(tmp_path):
    path = tmp_path / "artifact.json"
    life.create_only_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        life.create_only_json(path, {"a": 2})
    assert life.stable_json(path) == {"a": 1}
