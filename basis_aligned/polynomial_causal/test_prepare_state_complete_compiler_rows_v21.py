from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess

import pytest
import torch

import prepare_state_complete_compiler_rows_v21 as rows


def test_v21_protocol_freezes_total_nondeployable_null_and_fresh_currency() -> None:
    protocol = json.loads(rows.PROTOCOL.read_text())
    null = protocol["selection_permissive_nondeployable_shuffle_null"]
    assert "Copy collateral is never eligibility" in null["total_selector"]
    assert "r*<=0" in null["total_selector"]
    assert "nondeployable" in null["total_selector"]
    final_gate = protocol["final_control_gate"]
    assert "gain(A)=CE(NNN)-CE(A)" in final_gate["arms_and_gain"]
    assert "KL(OON||S0S1N)-KL(OON||T0T1N)" in final_gate["label_alignment"]
    assert "P_T=P(T0)+P(T1)" in final_gate["price"]
    fresh = protocol["fresh_currency"]
    assert "permanently forbidden" in fresh["spent_validation"]
    assert "old compiler_final" in fresh["validation"]
    assert "skip 39000" in fresh["final"]


def test_v21_old_role_identities_and_failed_absences_are_pinned() -> None:
    receipt = json.loads(rows.OLD_RECEIPT.read_text())
    for role, identities in rows.OLD_IDENTITIES.items():
        entry = receipt["entries"][role]
        assert rows.file_sha256(Path(entry["cache_path"])) == identities[
            "cache_file_sha256"
        ]
        assert entry["tensor_full_raw_sha256"] == identities["tensor_full_raw_sha256"]
        assert entry["tensor_prefix257_raw_sha256"] == identities[
            "tensor_prefix257_raw_sha256"
        ]
        records = receipt["document_provenance"]["sets"][role]
        assert rows.logical_json_sha256(records) == identities[
            "provenance_records_sha256"
        ]
    assert not any(path.exists() for path in rows.ORIGINAL_ABSENT)


def test_v21_source_closure_contains_behavior_and_tests() -> None:
    names = {path.name for path in rows.SOURCE_CLOSURE}
    assert "prepare_state_complete_compiler_rows_v21.py" in names
    assert "test_prepare_state_complete_compiler_rows_v21.py" in names
    assert "prepare_state_complete_compiler_rows_v2.py" in names
    assert "test_prepare_state_complete_compiler_rows_v2.py" in names


def test_v21_remap_uses_fit_and_old_final_never_spent_validation() -> None:
    receipt = json.loads(rows.OLD_RECEIPT.read_text())
    fit = rows.remapped_entry(receipt, "compiler_fit", "fit_reuse")
    validation = rows.remapped_entry(
        receipt, "compiler_final", "prospective_validation_remap"
    )
    assert fit["source_role"] == "compiler_fit"
    assert validation["source_role"] == "compiler_final"
    assert fit["cache_path"] != receipt["entries"]["compiler_validation"]["cache_path"]
    assert validation["cache_path"] != receipt["entries"]["compiler_validation"][
        "cache_path"
    ]


def _fake_entry(path, tensor, records, *, source_role=None):
    entry = {
        "cache_path": str(path),
        "cache_file_sha256": rows.file_sha256(path),
        "tensor_full_raw_sha256": rows.tensor_sha256(tensor),
        "tensor_prefix257_raw_sha256": rows.tensor_sha256(tensor[:, :rows.MODEL_LEN]),
        "provenance_records_sha256": rows.logical_json_sha256(records),
    }
    if source_role is not None:
        entry["source_role"] = source_role
    return entry


def test_v21_loader_deserializes_only_requested_roles(monkeypatch, tmp_path) -> None:
    tensors = {
        "compiler_fit_v21": torch.zeros(480, rows.T_LEN, dtype=torch.long),
        "compiler_validation_v21": torch.ones(192, rows.T_LEN, dtype=torch.long),
        "compiler_final_v21": torch.full((192, rows.T_LEN), 2, dtype=torch.long),
    }
    records = {
        role: [{"document_id": f"{role}:{index}"} for index in range(len(tensor))]
        for role, tensor in tensors.items()
    }
    paths = {}
    for role, tensor in tensors.items():
        path = tmp_path / f"{role}.pt"
        torch.save(tensor, path)
        paths[role] = path
    entries = {
        "compiler_fit_v21": _fake_entry(
            paths["compiler_fit_v21"], tensors["compiler_fit_v21"],
            records["compiler_fit_v21"], source_role="compiler_fit",
        ),
        "compiler_validation_v21": _fake_entry(
            paths["compiler_validation_v21"], tensors["compiler_validation_v21"],
            records["compiler_validation_v21"], source_role="compiler_final",
        ),
        "compiler_final_v21": _fake_entry(
            paths["compiler_final_v21"], tensors["compiler_final_v21"],
            records["compiler_final_v21"],
        ),
    }
    old_receipt = {
        "entries": {
            "compiler_fit": {"cache_path": str(paths["compiler_fit_v21"])},
            "compiler_final": {"cache_path": str(paths["compiler_validation_v21"])},
        }
    }
    receipt = {
        "status": "frozen_before_any_v21_validation_model_forward",
        "authority": "compiler_v21_prospective_role_designation",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "role_licenses": {
            "compiler_fit_v21": {
                "training": True, "selection": False, "final_scoring": False
            },
            "compiler_validation_v21": {
                "training": False, "selection": True, "final_scoring": False
            },
            "compiler_final_v21": {
                "training": False, "selection": False, "final_scoring": True,
                "requires_final_unlock_authority": True
            },
            "old_compiler_validation": "forbidden"
        },
        "protocol_sha256": rows.PINS[rows.PROTOCOL],
        "old_rows_receipt_sha256": rows.PINS[rows.OLD_RECEIPT],
        "retry1_failure_manifest_sha256": rows.PINS[rows.RETRY1_FAILURE],
        "entries": entries,
        "document_provenance": {"sets": records},
        "disjointness_gates": {"all": True},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(rows, "RECEIPT", receipt_path)
    monkeypatch.setattr(rows, "verify_inputs", lambda: (old_receipt, "head", {}))
    original_load = torch.load
    loaded = []

    def recording_load(path, *args, **kwargs):
        loaded.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    _, selected = rows.load_roles_and_validate(
        ("compiler_fit_v21", "compiler_validation_v21")
    )
    assert set(selected) == {"compiler_fit_v21", "compiler_validation_v21"}
    assert paths["compiler_final_v21"] not in loaded
    with pytest.raises(RuntimeError, match="final is locked"):
        rows.load_roles_and_validate(("compiler_final_v21",))
    with pytest.raises(ValueError, match="invalid"):
        rows.load_roles_and_validate(("compiler_validation",))


def _write_final_unlock(monkeypatch, tmp_path):
    row_receipt = tmp_path / "rows_receipt.json"
    validation_records = [
        {"document_id": f"doc-{index}"} for index in range(192)
    ]
    validation_identity = rows.logical_json_sha256([
        record["document_id"] for record in validation_records
    ])
    row_receipt.write_text(json.dumps({
        "entries": {
            "compiler_validation_v21": {
                "document_ids_sha256": validation_identity,
                "shape_model_prefix": [192, rows.MODEL_LEN],
            },
        },
        "document_provenance": {
            "sets": {"compiler_validation_v21": validation_records},
        },
    }))
    programs = tmp_path / "programs.pt"
    source = rows.HERE / "state_complete_compiler_selection_v2.py"
    source_relative = str(source.resolve().relative_to(rows.ROOT.resolve()))
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rows.ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    monkeypatch.setattr(rows, "PROGRAM_SOURCE_CLOSURE", (source,))
    scalar = torch.zeros(1)

    def candidate_state(spec):
        family, interface, ridge, size = spec
        if ridge is not None:
            return {
                "grammar": "affine", "interface": interface, "family": family,
                "mean": scalar.expand(rows.compiler.D_MODEL),
                "scale": torch.ones(1).expand(rows.compiler.D_MODEL),
                "bias": scalar.expand(rows.compiler.COEFFICIENT_DIM),
                "left": scalar.expand(rows.compiler.D_MODEL, size),
                "right": scalar.expand(size, rows.compiler.COEFFICIENT_DIM),
                "lambda": ridge, "rank": size,
            }
        return {
            "grammar": "native", "interface": interface, "family": family, "k": size,
            "left": scalar.expand(size, rows.compiler.D_MODEL),
            "right": scalar.expand(size, rows.compiler.D_MODEL),
            "projected_decoder": scalar.expand(size, rows.compiler.COEFFICIENT_DIM),
            "beta": scalar.expand(rows.compiler.COEFFICIENT_DIM),
            "indices": torch.arange(size, dtype=torch.long),
        }

    specs = rows._candidate_specs()
    ledger_names = ("true_site0", "true_site1", "shuffle_site0", "shuffle_site1")
    ledgers = {}
    for ledger_name in ledger_names:
        ledger = {}
        for candidate_name, spec in specs.items():
            state = candidate_state(spec)
            recovery = 0.0 if candidate_name.startswith("A_") else 0.1
            ledger[candidate_name] = {
                "state": state,
                "metrics": {
                    "recovery": recovery,
                    "copy_worsening": 0.0,
                    "price": rows.selection.state_price(state),
                },
            }
        ledgers[ledger_name] = ledger
    receipts = {
        name: (
            rows.selection.freeze_validation_selection(ledgers[name])
            if name.startswith("true_") else rows._total_shuffle_selection(ledgers[name])
        )
        for name in ledger_names
    }
    states = {
        arm: {
            site: ledgers[f"{arm}_site{site}"][receipts[f"{arm}_site{site}"]["selected"]][
                "state"
            ]
            for site in (0, 1)
        }
        for arm in ("true", "shuffle")
    }
    states["mean"] = {
        site: {
            "grammar": "constant", "interface": "state_complete_p",
            "family": "fit_mean_control",
            "bias": scalar.expand(rows.compiler.COEFFICIENT_DIM),
        }
        for site in (0, 1)
    }
    full_native = candidate_state((
        "full_native_ceiling_control", "state_complete_p", None,
        rows.compiler.NATIVE_PRODUCTS,
    ))
    controls = {}
    for site in (0, 1):
        controls[f"full_native_site{site}"] = {
            "state": full_native,
            "integrity_gates": {
                "algebra_identity": True, "physical_identity": True,
                "poison_zero_original_calls": True, "row_ce_identity": True,
            },
            "observed": {
                "physical_max_abs_error": 0.0,
                "physical_reference_scale": 1000.0,
                "physical_tolerance": 0.004,
                "original_mlp_calls": 0,
                "max_row_ce_abs_error": 0.0,
            },
        }
        controls[f"copy_constrained_shuffle_sensitivity_site{site}"] = {
            "status": "selected",
            "selection": rows.selection.freeze_control_selection(
                ledgers[f"shuffle_site{site}"]
            ),
        }
    prices = {
        "true": rows._pipeline_price(states["true"][0], states["true"][1]),
        "shuffle": rows._pipeline_price(states["shuffle"][0], states["shuffle"][1]),
        "mean": {
            "site0": rows._constant_price(), "site1": rows._constant_price(),
            "total_reals": 2 * rows._constant_price()["total_reals"],
        },
    }
    bundle = {
        "schema_version": 1,
        "status": "frozen_v21_program_bundle_pending_final_unlock",
        "authority": "compiler_v21_program_bundle",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": rows.PINS[rows.PROTOCOL],
        "rows_receipt_sha256": rows.file_sha256(row_receipt),
        "programs": states,
        "pipeline_contexts": {
            "true": {0: "baseline", 1: "true_site0"},
            "shuffle": {0: "baseline", 1: "shuffle_site0"},
            "mean": {0: "baseline", 1: "mean_site0"},
        },
        "candidate_ledgers": ledgers,
        "selection_receipts": receipts,
        "controls": controls,
        "strata": {
            "source": "compiler_validation_v21",
            "validation_document_ids_sha256": validation_identity,
            "token_frequency": {"boundaries": [1], "counts": [1, 36863]},
            "causal_weights_site1": torch.ones(rows.compiler.COEFFICIENT_DIM),
        },
        "prices": prices,
    }
    torch.save(bundle, programs)
    authority = tmp_path / "programs_receipt.json"
    monkeypatch.setattr(rows, "RECEIPT", row_receipt)
    monkeypatch.setattr(rows, "PROGRAMS_ARTIFACT", programs)
    monkeypatch.setattr(rows, "PROGRAMS_RECEIPT", authority)
    payload = {
        "status": "frozen_v21_programs_controls_strata_prices_before_final",
        "authority": "compiler_v21_final_unlock",
        "authorized_for_training": False,
        "authorized_for_final_scoring": True,
        "protocol_sha256": rows.PINS[rows.PROTOCOL],
        "rows_receipt_path": str(row_receipt.resolve()),
        "rows_receipt_sha256": rows.file_sha256(row_receipt),
        "programs_artifact_path": str(programs.resolve()),
        "programs_artifact_sha256": rows.file_sha256(programs),
        "programs_artifact_bytes": programs.stat().st_size,
        "frozen_contents": {
            "true_program_sites": [0, 1],
            "shuffle_program_sites": [0, 1],
            "mean_program_sites": [0, 1],
            "candidate_ledgers_frozen": True,
            "controls_frozen": True,
            "strata_frozen": True,
            "standalone_prices_frozen": True,
        },
        "source_commit": source_commit,
        "source_hashes": {source_relative: rows.file_sha256(source)},
    }
    authority.write_text(json.dumps(payload))
    return authority, programs, payload, bundle


def test_v21_final_unlock_validates_exact_program_freeze(monkeypatch, tmp_path) -> None:
    authority, _, _, _ = _write_final_unlock(monkeypatch, tmp_path)
    validated = rows.validate_final_unlock(authority)
    assert validated["authorized_for_final_scoring"] is True


@pytest.mark.parametrize("failure", [
    "arbitrary_hash", "missing_program", "protocol", "rows",
    "empty_programs", "incomplete_sources", "nonexistent_commit",
    "malformed_state", "incomplete_grid", "selected_mismatch",
    "dummy_controls", "dummy_strata", "wrong_price",
    "wrong_dtype", "wrong_document_identity", "wrong_physical_scale_gate",
])
def test_v21_final_unlock_rejects_broken_bindings(monkeypatch, tmp_path, failure) -> None:
    authority, programs, payload, bundle = _write_final_unlock(monkeypatch, tmp_path)
    bundle_changed = False
    if failure == "arbitrary_hash":
        payload["programs_artifact_sha256"] = "0" * 64
    elif failure == "missing_program":
        programs.unlink()
    elif failure == "protocol":
        payload["protocol_sha256"] = "0" * 64
    elif failure == "rows":
        payload["rows_receipt_sha256"] = "0" * 64
    elif failure == "empty_programs":
        torch.save({"programs": {}}, programs)
        payload["programs_artifact_sha256"] = rows.file_sha256(programs)
        payload["programs_artifact_bytes"] = programs.stat().st_size
    elif failure == "incomplete_sources":
        payload["source_hashes"] = {}
    elif failure == "nonexistent_commit":
        payload["source_commit"] = "a" * 40
    elif failure == "malformed_state":
        bundle["candidate_ledgers"]["true_site0"]["B_l0_r8"]["state"]["left"] = torch.zeros(1)
        bundle_changed = True
    elif failure == "incomplete_grid":
        del bundle["candidate_ledgers"]["true_site0"]["E_k256"]
        bundle_changed = True
    elif failure == "selected_mismatch":
        bundle["programs"]["true"][0] = bundle["candidate_ledgers"]["true_site0"][
            "B_l0_r16"
        ]["state"]
        bundle_changed = True
    elif failure == "dummy_controls":
        bundle["controls"] = {"passed": True}
        bundle_changed = True
    elif failure == "dummy_strata":
        bundle["strata"] = {"all": {"rows": 1}}
        bundle_changed = True
    elif failure == "wrong_price":
        bundle["prices"]["true"]["total_reals"] += 1
        bundle_changed = True
    elif failure == "wrong_dtype":
        bundle["candidate_ledgers"]["true_site0"]["B_l0_r8"]["state"]["left"] = (
            bundle["candidate_ledgers"]["true_site0"]["B_l0_r8"]["state"]["left"].double()
        )
        bundle_changed = True
    elif failure == "wrong_document_identity":
        bundle["strata"]["validation_document_ids_sha256"] = "f" * 64
        bundle_changed = True
    elif failure == "wrong_physical_scale_gate":
        observed = bundle["controls"]["full_native_site0"]["observed"]
        observed["physical_reference_scale"] = 1.0
        observed["physical_tolerance"] = 4e-6
        observed["physical_max_abs_error"] = 1e-3
        bundle_changed = True
    if bundle_changed:
        torch.save(bundle, programs)
        payload["programs_artifact_sha256"] = rows.file_sha256(programs)
        payload["programs_artifact_bytes"] = programs.stat().st_size
    authority.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        rows.validate_final_unlock(authority)


def test_v21_program_closure_inherits_retry1_transitive_dependencies() -> None:
    closure = {path.resolve() for path in rows.PROGRAM_SOURCE_CLOSURE}
    required = {
        rows.HERE / "early_mlp_state_complete_compiler_v2_site0.py",
        rows.HERE / "frozen_ship_oracle_v2.py",
        rows.HERE / "code_ood_oracle.py",
        rows.BQ / "ship_error_attrib.py",
        rows.ROOT / "basis_aligned/qk_mdl/tier2_model.py",
        rows.ROOT / "jacclust/tt_model.py",
    }
    assert {path.resolve() for path in required} <= closure


def test_v21_build_request_is_fixed_without_fallback() -> None:
    assert rows.FINAL_SPEC == (192, 39000)


def test_v21_row_receipt_is_last_build_content_write() -> None:
    source = inspect.getsource(rows.build)
    authority = source.rfind("write_json_atomic(receipt, RECEIPT)")
    assert authority >= 0
    assert "write_" not in source[authority + len(
        "write_json_atomic(receipt, RECEIPT)"
    ):]
