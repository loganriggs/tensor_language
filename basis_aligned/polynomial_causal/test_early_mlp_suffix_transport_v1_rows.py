import copy
import hashlib
import json

import pytest
import torch

import early_mlp_suffix_transport_v1_rows as rows


def candidate(index=0):
    triple = rows.candidate_triple(index)
    counts = {
        "fit": triple.fit_n,
        "validation": triple.validation_n,
        "final": triple.final_n,
    }
    starts = {"fit": 0, "validation": 1000, "final": 2000}
    tensors = {}
    records = {}
    for role in rows.ROLES:
        n = counts[role]
        start = starts[role]
        base = torch.arange(start, start + n, dtype=torch.long)[:, None]
        position = torch.arange(rows.TOKEN_LENGTH, dtype=torch.long)[None, :]
        tensors[role] = base * 1000 + position
        records[role] = [
            {
                "document_id": f"doc-{start + offset}",
                "dataset_document_index": start + offset,
                "chunk_id": 0,
                "token_start": 0,
            }
            for offset in range(n)
        ]
    return tensors, records


def test_clean_candidate_is_accepted_and_reports_hashes_only():
    tensors, records = candidate()
    report = rows.adjudicate_candidate(
        candidate_index=0,
        rows_by_role=tensors,
        records_by_role=records,
        prior=rows.IdentitySets.empty(),
    )
    assert report["accepted"] is True and report["collision_evidence_count"] == 0
    assert report["role_identity_counts"]["fit"]["full_rows"] == 384
    encoded = repr(report)
    assert "doc-" not in encoded
    assert str(tensors["final"][0, 0].item()) not in encoded
    assert all(len(value) == 64 for value in report["role_identity_hashes"]["fit"].values())


def test_prior_and_cross_role_collisions_are_rejected():
    tensors, records = candidate()
    tensors["final"][0] = tensors["fit"][0]
    records["validation"][0] = dict(records["fit"][0])
    prior = rows.IdentitySets(
        documents=frozenset({records["fit"][1]["document_id"]}),
        dataset_indices=frozenset(),
        full_rows=frozenset(),
        prefix32=frozenset(),
    )
    report = rows.adjudicate_candidate(
        candidate_index=0,
        rows_by_role=tensors,
        records_by_role=records,
        prior=prior,
    )
    assert report["accepted"] is False
    assert report["prior_collision_counts"]["fit"]["documents"] == 1
    assert report["cross_role_collision_counts"]["fit__validation"]["documents"] == 1
    assert report["cross_role_collision_counts"]["fit__final"]["full_rows"] == 1


def test_internal_duplicate_prefix_is_a_collision():
    tensors, records = candidate()
    tensors["fit"][1, :32] = tensors["fit"][0, :32]
    report = rows.adjudicate_candidate(
        candidate_index=0,
        rows_by_role=tensors,
        records_by_role=records,
        prior=rows.IdentitySets.empty(),
    )
    assert report["accepted"] is False
    assert report["internal_duplicate_counts"]["fit"]["prefix32"] == 1


@pytest.mark.parametrize("field,value", [
    ("document_id", ""),
    ("dataset_document_index", True),
    ("chunk_id", -1),
    ("token_start", 1),
])
def test_bad_provenance_fails_closed(field, value):
    tensors, records = candidate()
    records["fit"][0][field] = value
    with pytest.raises(RuntimeError):
        rows.adjudicate_candidate(
            candidate_index=0,
            rows_by_role=tensors,
            records_by_role=records,
            prior=rows.IdentitySets.empty(),
        )


def test_document_index_mapping_must_be_bijective_within_role():
    tensors, records = candidate()
    records["fit"][1]["document_id"] = records["fit"][0]["document_id"]
    with pytest.raises(RuntimeError, match="one document_id to multiple"):
        rows.adjudicate_candidate(
            candidate_index=0,
            rows_by_role=tensors,
            records_by_role=records,
            prior=rows.IdentitySets.empty(),
        )


def test_exact_role_keys_shapes_and_counts_are_required():
    tensors, records = candidate()
    bad_tensors = dict(tensors)
    bad_tensors["fit"] = bad_tensors["fit"][:-1]
    with pytest.raises(RuntimeError, match="long"):
        rows.adjudicate_candidate(
            candidate_index=0,
            rows_by_role=bad_tensors,
            records_by_role=records,
            prior=rows.IdentitySets.empty(),
        )
    bad_records = dict(records)
    bad_records["extra"] = []
    with pytest.raises(RuntimeError, match="exactly"):
        rows.adjudicate_candidate(
            candidate_index=0,
            rows_by_role=tensors,
            records_by_role=bad_records,
            prior=rows.IdentitySets.empty(),
        )


def test_collision_history_is_contiguous_first_accepted_and_literal_bool():
    tensors, records = candidate()
    accepted = rows.adjudicate_candidate(
        candidate_index=0,
        rows_by_role=tensors,
        records_by_role=records,
        prior=rows.IdentitySets.empty(),
    )
    rejected = copy.deepcopy(accepted)
    rejected["accepted"] = False
    rejected["collision_evidence_count"] = 1
    rejected["prior_collision_counts"]["fit"]["documents"] = 1
    tensors1, records1 = candidate(1)
    chosen = rows.adjudicate_candidate(
        candidate_index=1,
        rows_by_role=tensors1,
        records_by_role=records1,
        prior=rows.IdentitySets.empty(),
    )
    rows.validate_collision_history([rejected, chosen], 1)
    assert len(rows.collision_history_hash([rejected, chosen])) == 64

    with pytest.raises(RuntimeError, match="skipped"):
        rows.validate_collision_history([accepted, chosen], 1)
    chosen["accepted"] = "true"
    with pytest.raises(RuntimeError, match="literal boolean"):
        rows.validate_collision_history([rejected, chosen], 1)


def test_ordered_provenance_to_row_mapping_is_bound():
    tensors, records = candidate()
    original = rows.adjudicate_candidate(
        candidate_index=0, rows_by_role=tensors, records_by_role=records,
        prior=rows.IdentitySets.empty(),
    )
    permuted = copy.deepcopy(records)
    permuted["fit"][0], permuted["fit"][1] = permuted["fit"][1], permuted["fit"][0]
    changed = rows.adjudicate_candidate(
        candidate_index=0, rows_by_role=tensors, records_by_role=permuted,
        prior=rows.IdentitySets.empty(),
    )
    assert original["role_identity_hashes"]["fit"]["ordered_tensor_raw"] == \
        changed["role_identity_hashes"]["fit"]["ordered_tensor_raw"]
    assert original["role_identity_hashes"]["fit"]["ordered_provenance"] != \
        changed["role_identity_hashes"]["fit"]["ordered_provenance"]
    assert original["role_identity_hashes"]["fit"]["ordered_row_provenance_binding"] != \
        changed["role_identity_hashes"]["fit"]["ordered_row_provenance_binding"]


@pytest.mark.parametrize("mutation,message", [
    ("extra", "schema changed"),
    ("schedule", "schedule changed"),
    ("count", "does not recompute"),
    ("decision", "disagrees"),
    ("hash", "lowercase SHA256"),
])
def test_collision_report_validator_rejects_mutation_and_leakage(mutation, message):
    tensors, records = candidate()
    report = rows.adjudicate_candidate(
        candidate_index=0, rows_by_role=tensors, records_by_role=records,
        prior=rows.IdentitySets.empty(),
    )
    if mutation == "extra":
        report["raw_document_id"] = "leak"
    elif mutation == "schedule":
        report["candidate"]["fit"]["skip"] += 1
    elif mutation == "count":
        report["collision_evidence_count"] = 1
    elif mutation == "decision":
        report["accepted"] = False
    else:
        report["role_identity_hashes"]["fit"]["documents"] = "A" * 64
    with pytest.raises(RuntimeError, match=message):
        rows.validate_collision_report(report, 0)


def test_collision_history_hash_validates_before_hashing():
    tensors, records = candidate()
    report = rows.adjudicate_candidate(
        candidate_index=0, rows_by_role=tensors, records_by_role=records,
        prior=rows.IdentitySets.empty(),
    )
    report["raw_tokens"] = [1, 2, 3]
    with pytest.raises(RuntimeError, match="schema changed"):
        rows.collision_history_hash([report])


def test_exact_allowlist_census_loads_provenance_rows_and_protected_files(tmp_path):
    prior_rows = torch.arange(2 * 40, dtype=torch.long).reshape(2, 40)
    tensor_path = tmp_path / "prior.pt"
    torch.save(prior_rows, tensor_path)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "document_provenance": {"sets": {"fit": [{
            "document_id": "prior-doc", "dataset_document_index": 17,
        }]}},
    }))
    protected_path = tmp_path / "ship.pt"
    protected_path.write_bytes(b"protected")
    identities, census = rows.load_canonical_prior(
        registry_files={registry_path: rows.file_sha256(registry_path)},
        row_tensors={tensor_path: (
            rows.file_sha256(tensor_path), rows.tensor_raw_sha256(prior_rows), None,
        )},
        protected_files={protected_path: rows.file_sha256(protected_path)},
    )
    assert identities.documents == {"prior-doc"}
    assert identities.dataset_indices == {17}
    assert tuple(prior_rows[0].tolist()) in identities.full_rows
    assert tuple(prior_rows[0, :32].tolist()) in identities.prefix32
    assert census["discovery_rule"] == \
        "exact_prospective_allowlist_no_recursive_discovery"


@pytest.mark.parametrize("kind", ["registry", "tensor_file", "tensor_raw", "protected"])
def test_exact_allowlist_census_fails_on_any_identity_drift(tmp_path, kind):
    value = torch.arange(40, dtype=torch.long).reshape(1, 40)
    tensor_path = tmp_path / "prior.pt"
    torch.save(value, tensor_path)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")
    protected_path = tmp_path / "ship.pt"
    protected_path.write_bytes(b"ship")
    registry_hash = rows.file_sha256(registry_path)
    tensor_file_hash = rows.file_sha256(tensor_path)
    tensor_raw_hash = rows.tensor_raw_sha256(value)
    protected_hash = rows.file_sha256(protected_path)
    if kind == "registry":
        registry_hash = "0" * 64
    elif kind == "tensor_file":
        tensor_file_hash = "0" * 64
    elif kind == "tensor_raw":
        tensor_raw_hash = "0" * 64
    else:
        protected_hash = "0" * 64
    with pytest.raises(RuntimeError, match="hash changed"):
        rows.load_canonical_prior(
            registry_files={registry_path: registry_hash},
            row_tensors={tensor_path: (tensor_file_hash, tensor_raw_hash, None)},
            protected_files={protected_path: protected_hash},
        )


def test_code_style_keyed_tensor_is_censused(tmp_path):
    value = torch.arange(40, dtype=torch.long).reshape(1, 40)
    tensor_path = tmp_path / "code.pt"
    torch.save({"rows": value, "manifest": {}}, tensor_path)
    identities, _ = rows.load_canonical_prior(
        registry_files={},
        row_tensors={tensor_path: (
            rows.file_sha256(tensor_path), rows.tensor_raw_sha256(value), "rows",
        )},
        protected_files={},
    )
    assert tuple(value[0, :32].tolist()) in identities.prefix32


def test_missing_keyed_tensor_payload_fails_closed(tmp_path):
    tensor_path = tmp_path / "code.pt"
    torch.save({"other": torch.arange(40, dtype=torch.long).reshape(1, 40)}, tensor_path)
    with pytest.raises(RuntimeError, match="invalid shape or dtype"):
        rows.load_canonical_prior(
            registry_files={},
            row_tensors={tensor_path: (
                rows.file_sha256(tensor_path), "0" * 64, "rows",
            )},
            protected_files={},
        )
