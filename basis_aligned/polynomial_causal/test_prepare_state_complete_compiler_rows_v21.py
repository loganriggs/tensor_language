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
    monkeypatch.setattr(rows, "_validate_historical_row_authority", lambda _: None)
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
    source = rows.HERE / "state_complete_compiler_selection_v2.py"
    source_relative = str(source.resolve().relative_to(rows.ROOT.resolve()))
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rows.ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    source_hashes = {source_relative: rows.file_sha256(source)}
    row_receipt = tmp_path / "rows_receipt.json"
    row_manifest = tmp_path / "rows_manifest.json"
    fit_cache = tmp_path / "fit.pt"
    validation_cache = tmp_path / "validation.pt"
    fit_rows = torch.zeros((480, rows.T_LEN), dtype=torch.long)
    validation_rows = torch.zeros((192, rows.T_LEN), dtype=torch.long)
    torch.save(fit_rows, fit_cache)
    torch.save(validation_rows, validation_cache)
    validation_records = [
        {"document_id": f"doc-{index}"} for index in range(192)
    ]
    fit_records = [
        {"document_id": f"fit-doc-{index}"} for index in range(480)
    ]
    validation_identity = rows.logical_json_sha256([
        record["document_id"] for record in validation_records
    ])
    row_manifest.write_text(json.dumps({
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "protected_before": {},
        "protected_after": {},
    }))
    row_receipt.write_text(json.dumps({
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "manifest_sha256": rows.file_sha256(row_manifest),
        "entries": {
            "compiler_fit_v21": {
                "cache_path": str(fit_cache.resolve()),
                "cache_file_sha256": rows.file_sha256(fit_cache),
            },
            "compiler_validation_v21": {
                "cache_path": str(validation_cache.resolve()),
                "cache_file_sha256": rows.file_sha256(validation_cache),
                "document_ids_sha256": validation_identity,
                "shape_model_prefix": [192, rows.MODEL_LEN],
            },
        },
        "document_provenance": {
            "sets": {
                "compiler_fit_v21": fit_records,
                "compiler_validation_v21": validation_records,
            },
        },
    }))
    programs = tmp_path / "programs.pt"
    monkeypatch.setattr(rows, "PROGRAM_SOURCE_CLOSURE", (source,))
    monkeypatch.setattr(rows, "MANIFEST", row_manifest)
    monkeypatch.setattr(rows, "ROWS_RECEIPT_SHA256", rows.file_sha256(row_receipt))
    monkeypatch.setattr(rows, "ROWS_MANIFEST_SHA256", rows.file_sha256(row_manifest))
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
            candidate_kl = 1.0 - recovery
            candidate_kl_sum = candidate_kl * rows.VALIDATION_TOKEN_COUNT
            candidate_kl = candidate_kl_sum / rows.VALIDATION_TOKEN_COUNT
            recovery = 1.0 - candidate_kl
            raw = {
                "candidate_teacher_kl_sum": candidate_kl_sum,
                "candidate_teacher_kl_count": rows.VALIDATION_TOKEN_COUNT,
                "global_ce_sum": float(rows.VALIDATION_TOKEN_COUNT),
                "global_ce_count": rows.VALIDATION_TOKEN_COUNT,
                "copy_ce_sum": 2.0,
                "copy_ce_count": 2,
            }
            ledger[candidate_name] = {
                "state": state,
                "metrics": {
                    "candidate_teacher_kl": candidate_kl,
                    "oracle_denominator_kl": 1.0,
                    "remaining_kl_ratio": candidate_kl,
                    "recovery": recovery,
                    "global_ce": 1.0,
                    "copy_ce": 1.0,
                    "copy_count": 2,
                    "copy_worsening": 0.0,
                    "price": rows.selection.state_price(state),
                    "raw_sufficient_statistics": raw,
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
    def full_native_control(site, context, upstream_hash):
        capture_calls = ({0: 24, 1: 0, 2: 0} if site == 0 else {
            0: 0, 1: 24, 2: 0,
        })
        scored_calls = ({0: 0, 1: 24, 2: 0} if site == 0 else {
            0: 0, 1: 0, 2: 0,
        })
        observed = {
            "physical_max_abs_error": 0.0,
            "physical_reference_scale": 1000.0,
            "physical_tolerance": 0.004,
            "target_original_mlp_calls": 0,
            "capture_call_counters": capture_calls,
            "scored_arm_call_counters": scored_calls,
            "max_row_ce_abs_error": 0.0,
        }
        gates = {
            "algebra_identity": True, "physical_identity": True,
            "poison_zero_original_calls": True, "row_ce_identity": True,
        }
        value = {
            "state": full_native,
            "context": context,
            "upstream_state_sha256": upstream_hash,
            "validation_document_ids_sha256": validation_identity,
            "scorer": "CUDA float32 per-token; float64 row/aggregate",
            "integrity_gates": gates,
            "observed": observed,
        }
        value["measurement_sha256"] = rows.logical_json_sha256({
            "context": context,
            "upstream_state_sha256": upstream_hash,
            "validation_document_ids_sha256": validation_identity,
            "scorer": value["scorer"],
            "state_sha256": rows.state_logical_sha256(full_native),
            "integrity_gates": gates,
            "observed": observed,
        })
        return value
    controls = {
        "full_native_site0": full_native_control(0, "baseline", "baseline"),
        "full_native_site1_true_context": full_native_control(
            1, "true_site0", rows.state_logical_sha256(states["true"][0]),
        ),
        "full_native_site1_shuffle_context": full_native_control(
            1, "shuffle_site0", rows.state_logical_sha256(states["shuffle"][0]),
        ),
    }
    for site in (0, 1):
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
    stage_bindings = {}
    for stage, names in {
        "site0": ("true_site0", "shuffle_site0"),
        "site1": ("true_site1", "shuffle_site1"),
    }.items():
        artifact = tmp_path / f"{stage}_ledger.pt"
        receipt = tmp_path / f"{stage}_ledger_receipt.json"
        stage_controls = ({
            "mean_site0": states["mean"][0],
            "full_native_site0": controls["full_native_site0"],
        } if stage == "site0" else {
            "mean_site1": states["mean"][1],
            "full_native_site1_true_context": controls[
                "full_native_site1_true_context"
            ],
            "full_native_site1_shuffle_context": controls[
                "full_native_site1_shuffle_context"
            ],
        })
        upstream = ({name: "baseline" for name in names} if stage == "site0" else {
            "true_site1": rows.state_logical_sha256(states["true"][0]),
            "shuffle_site1": rows.state_logical_sha256(states["shuffle"][0]),
        })
        capture_keys = ({
            "fit_original", "fit_shuffled", "validation_site0",
        } if stage == "site0" else {
            "true_fit_site1", "shuffle_fit_site1",
            "true_validation_site1", "shuffle_validation_site1", "mean_fit_site1",
        })
        if stage == "site0":
            expected_calls = {
                "fit_capture": {0: 60, 1: 60, 2: 0},
                "validation_capture": {0: 24, 1: 0, 2: 0},
                "teacher": {0: 24, 1: 24, 2: 0},
                "copy_baseline": {0: 0, 1: 24, 2: 0},
                "candidate": {0: 0, 1: 24, 2: 0},
            }
        else:
            expected_calls = {
                "fit_capture": {0: 0, 1: 60, 2: 0},
                "validation_capture": {0: 0, 1: 24, 2: 0},
                "teacher": {0: 0, 1: 24, 2: 0},
                "copy_baseline": {0: 0, 1: 0, 2: 0},
                "candidate": {0: 0, 1: 0, 2: 0},
            }
        diagnostics = {
            "fit_permutation_sha256": rows.expected_fit_permutation_sha256(
                json.loads(row_receipt.read_text())
            ),
            "capture_hashes": {
                name: rows.logical_json_sha256([stage, name])
                for name in capture_keys
            },
            "contexts": {
                name: {
                    "upstream_state_sha256": upstream[name],
                    "scorer": "CUDA float32 per-token; float64 row/aggregate",
                    "teacher_denominator": 1.0,
                    "teacher_kl_sum": float(rows.VALIDATION_TOKEN_COUNT),
                    "teacher_token_count": rows.VALIDATION_TOKEN_COUNT,
                    "copy_baseline": 1.0,
                    "copy_ce_sum": 2.0,
                    "copy_token_count": 2,
                    "call_counters": {
                        "fit_capture": expected_calls["fit_capture"],
                        "validation_capture": expected_calls[
                            "validation_capture"
                        ],
                        "teacher": expected_calls["teacher"],
                        "copy_baseline": expected_calls["copy_baseline"],
                        "candidates": {
                            candidate_name: expected_calls["candidate"]
                            for candidate_name in ledgers[name]
                        },
                    },
                }
                for name in names
            },
        }
        if stage == "site1":
            p_sum = torch.zeros(rows.compiler.COEFFICIENT_DIM, dtype=torch.float64)
            diagnostics["mean_control"] = {
                "context": "mean_site0",
                "upstream_state_sha256": rows.state_logical_sha256(states["mean"][0]),
                "scorer": "CUDA float32 capture; float64 coefficient sums",
                "p_sum": p_sum,
                "p_sum_sha256": rows.tensor_sha256(p_sum),
                "p_count": rows.FIT_CAPTURE_COUNT,
                "capture_call_counter": {0: 0, 1: 60, 2: 0},
            }
        torch.save({
            "schema_version": 1,
            "status": f"pending_v21_{stage}_preselector_ledger",
            "authorized_for_training": False,
            "authorized_for_final_scoring": False,
            "candidate_ledgers": {name: ledgers[name] for name in names},
            "controls": stage_controls,
            "diagnostics": diagnostics,
        }, artifact)
        receipt.write_text(json.dumps({
            "status": f"frozen_v21_{stage}_preselector_ledger",
            "authority": f"compiler_v21_{stage}_preselector_ledger",
            "authorized_for_training": False,
            "authorized_for_final_scoring": False,
            "protocol_sha256": rows.PINS[rows.PROTOCOL],
            "implementation_amendment_sha256": rows.IMPLEMENTATION_AMENDMENT_SHA256,
            "rows_receipt_sha256": rows.file_sha256(row_receipt),
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": rows.file_sha256(artifact),
            "artifact_bytes": artifact.stat().st_size,
        }))
        stage_bindings[stage] = {
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": rows.file_sha256(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "receipt_path": str(receipt.resolve()),
            "receipt_sha256": rows.file_sha256(receipt),
            "receipt_bytes": receipt.stat().st_size,
        }
        monkeypatch.setattr(rows, f"{stage.upper()}_LEDGER_ARTIFACT", artifact)
        monkeypatch.setattr(rows, f"{stage.upper()}_LEDGER_RECEIPT", receipt)
    site0_training_receipt = tmp_path / "site0_training_receipt.json"
    site0_training_payload = {
        "status": "frozen_v21_site0_programs_after_outer_return",
        "authority": "compiler_v21_site0_to_site1_training_unlock",
        "authorized_for_training": True,
        "training_license_sites": [1],
        "authorized_for_final_scoring": False,
        "selected_state_sha256": {
            arm: rows.state_logical_sha256(states[arm][0])
            for arm in ("true", "shuffle")
        },
        "mean_state_sha256": rows.state_logical_sha256(states["mean"][0]),
        "stage_binding": stage_bindings["site0"],
        "component_tree_sha256": "component-tree",
        "outer_model_returned": True,
        "hook_restored_and_inert": True,
    }
    site0_training_receipt.write_text(json.dumps(site0_training_payload))
    monkeypatch.setattr(rows, "SITE0_TRAINING_RECEIPT", site0_training_receipt)
    token_frequency = rows.derive_token_frequency_strata(
        fit_rows, validation_rows, rows.TOKEN_FREQUENCY_BOUNDARIES,
    )
    full_oracle_row_ce = torch.zeros(192, dtype=torch.float64)
    omit_row_ce = torch.ones(
        rows.compiler.COEFFICIENT_DIM, 192, dtype=torch.float64,
    )
    target_p_count = 192 * 64
    target_p_square_sums = torch.full(
        (rows.compiler.COEFFICIENT_DIM,), float(target_p_count), dtype=torch.float64,
    )
    causal = rows.derive_causal_audit(
        full_oracle_row_ce, omit_row_ce, target_p_square_sums, target_p_count,
    )
    bundle = {
        "schema_version": 1,
        "status": "frozen_v21_program_bundle_pending_final_unlock",
        "authority": "compiler_v21_program_bundle",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": rows.PINS[rows.PROTOCOL],
        "implementation_amendment_sha256": rows.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": rows.file_sha256(row_receipt),
        "programs": states,
        "pipeline_contexts": {
            "true": {0: "baseline", 1: "true_site0"},
            "shuffle": {0: "baseline", 1: "shuffle_site0"},
            "mean": {0: "baseline", 1: "mean_site0"},
        },
        "candidate_ledgers": ledgers,
        "selection_receipts": receipts,
        "stage_bindings": stage_bindings,
        "site0_training_authorization": {
            "path": str(site0_training_receipt.resolve()),
            "sha256": rows.file_sha256(site0_training_receipt),
            "bytes": site0_training_receipt.stat().st_size,
            "receipt": site0_training_payload,
        },
        "controls": controls,
        "strata": {
            "source": "compiler_validation_v21",
            "validation_document_ids_sha256": validation_identity,
            "token_frequency": token_frequency,
            "causal_omission_audit": {
                "context": "true_site0",
                "upstream_state_sha256": rows.state_logical_sha256(states["true"][0]),
                "validation_document_ids_sha256": validation_identity,
                "scorer": "CUDA float32 per-token; float64 row/aggregate",
                "quantile_currency": "torch.float64 q=0.05 interpolation=linear",
                "rule": "abs(loss)/max(second_moment,1e-12); positive 5pct floor; mean-one",
                "full_oracle_row_ce": full_oracle_row_ce,
                "omit_row_ce": omit_row_ce,
                "full_oracle_row_ce_sha256": rows.tensor_sha256(full_oracle_row_ce),
                "omit_row_ce_sha256": rows.tensor_sha256(omit_row_ce),
                "target_p_square_sums": target_p_square_sums,
                "target_p_square_sums_sha256": rows.tensor_sha256(target_p_square_sums),
                "target_p_count": target_p_count,
                **causal,
                "call_counters": {
                    "full_oracle": {0: 0, 1: 0, 2: 0},
                    "omissions": {0: 0, 1: 0, 2: 0},
                },
            },
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
        "implementation_amendment_sha256": rows.IMPLEMENTATION_AMENDMENT_SHA256,
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
            "preselector_stage_receipts_bound": True,
            "strata_derivations_recomputed": True,
            "site1_full_native_contexts": ["true", "shuffle"],
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
    "permutation", "capture_keys", "forbidden_candidate_call", "denominator_stats",
    "mean_site1_stats",
])
def test_v21_stage_diagnostics_fail_closed(monkeypatch, tmp_path, failure) -> None:
    authority, programs, authority_payload, bundle = _write_final_unlock(
        monkeypatch, tmp_path
    )
    stage = "site1" if failure == "mean_site1_stats" else "site0"
    artifact_path = getattr(rows, f"{stage.upper()}_LEDGER_ARTIFACT")
    receipt_path = getattr(rows, f"{stage.upper()}_LEDGER_RECEIPT")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    diagnostics = artifact["diagnostics"]
    if failure == "permutation":
        diagnostics["fit_permutation_sha256"] = "f" * 64
    elif failure == "capture_keys":
        diagnostics["capture_hashes"] = {"placeholder": "f" * 64}
    elif failure == "forbidden_candidate_call":
        name = next(iter(diagnostics["contexts"]["true_site0"][
            "call_counters"
        ]["candidates"]))
        diagnostics["contexts"]["true_site0"]["call_counters"][
            "candidates"
        ][name][2] = 1
    elif failure == "denominator_stats":
        diagnostics["contexts"]["true_site0"]["teacher_kl_sum"] += 1.0
    elif failure == "mean_site1_stats":
        diagnostics["mean_control"]["p_sum"][0] = 1.0
        diagnostics["mean_control"]["p_sum_sha256"] = rows.tensor_sha256(
            diagnostics["mean_control"]["p_sum"]
        )
    torch.save(artifact, artifact_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_sha256"] = rows.file_sha256(artifact_path)
    receipt["artifact_bytes"] = artifact_path.stat().st_size
    receipt_path.write_text(json.dumps(receipt))
    bundle["stage_bindings"][stage].update({
        "artifact_sha256": rows.file_sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "receipt_sha256": rows.file_sha256(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
    })
    torch.save(bundle, programs)
    authority_payload["programs_artifact_sha256"] = rows.file_sha256(programs)
    authority_payload["programs_artifact_bytes"] = programs.stat().st_size
    authority.write_text(json.dumps(authority_payload))
    with pytest.raises(RuntimeError):
        rows.validate_final_unlock(authority)


def test_v21_causal_currency_is_exact_float64_and_fixed_count() -> None:
    full = torch.zeros(192, dtype=torch.float64)
    omitted = torch.ones(rows.compiler.COEFFICIENT_DIM, 192, dtype=torch.float64)
    sums = torch.full(
        (rows.compiler.COEFFICIENT_DIM,), float(rows.CAUSAL_CAPTURE_COUNT),
        dtype=torch.float64,
    )
    result = rows.derive_causal_audit(
        full, omitted, sums, rows.CAUSAL_CAPTURE_COUNT,
    )
    assert result["omission_losses"].dtype == torch.float64
    assert result["target_second_moments"].dtype == torch.float64
    assert result["weights"].dtype == torch.float32
    with pytest.raises(RuntimeError, match="sufficient statistics"):
        rows.derive_causal_audit(full, omitted, sums, rows.CAUSAL_CAPTURE_COUNT + 1)
    with pytest.raises(RuntimeError, match="causal omission losses"):
        rows.derive_causal_weights(
            result["omission_losses"].float(), result["target_second_moments"],
        )


def test_v21_sha256_syntax_is_strict() -> None:
    assert rows._is_sha256("0" * 64)
    assert not rows._is_sha256("g" * 64)
    assert not rows._is_sha256("A" * 64)
    assert not rows._is_sha256("0" * 63)


@pytest.mark.parametrize("failure", [
    "arbitrary_hash", "missing_program", "protocol", "rows",
    "empty_programs", "incomplete_sources", "nonexistent_commit",
    "malformed_state", "incomplete_grid", "selected_mismatch",
    "dummy_controls", "dummy_strata", "wrong_price",
    "wrong_dtype", "wrong_document_identity", "wrong_physical_scale_gate",
    "missing_site1_context", "wrong_token_derivation", "wrong_causal_weights",
    "unbound_stage_receipt", "amendment",
    "duplicate_site1_control", "wrong_receipt_bytes", "unbound_causal_rows",
    "full_native_mlp2_call",
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
    elif failure == "missing_site1_context":
        del bundle["controls"]["full_native_site1_shuffle_context"]
        bundle_changed = True
    elif failure == "wrong_token_derivation":
        bundle["strata"]["token_frequency"]["validation_assignment_sha256"] = "f" * 64
        bundle_changed = True
    elif failure == "wrong_causal_weights":
        bundle["strata"]["causal_omission_audit"]["weights"][0] = 2.0
        bundle_changed = True
    elif failure == "unbound_stage_receipt":
        bundle["stage_bindings"]["site0"]["receipt_sha256"] = "f" * 64
        bundle_changed = True
    elif failure == "amendment":
        payload["implementation_amendment_sha256"] = "f" * 64
    elif failure == "duplicate_site1_control":
        bundle["controls"]["full_native_site1_shuffle_context"] = bundle[
            "controls"
        ]["full_native_site1_true_context"]
        bundle_changed = True
    elif failure == "wrong_receipt_bytes":
        bundle["stage_bindings"]["site0"]["receipt_bytes"] += 1
        bundle_changed = True
    elif failure == "unbound_causal_rows":
        bundle["strata"]["causal_omission_audit"]["omit_row_ce"][0, 0] += 1.0
        bundle_changed = True
    elif failure == "full_native_mlp2_call":
        control = bundle["controls"]["full_native_site1_true_context"]
        control["observed"]["capture_call_counters"][2] = 1
        control["measurement_sha256"] = rows.logical_json_sha256({
            "context": control["context"],
            "upstream_state_sha256": control["upstream_state_sha256"],
            "validation_document_ids_sha256": control[
                "validation_document_ids_sha256"
            ],
            "scorer": control["scorer"],
            "state_sha256": rows.state_logical_sha256(control["state"]),
            "integrity_gates": control["integrity_gates"],
            "observed": control["observed"],
        })
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


def test_realized_v21_row_authority_remains_commit_bound() -> None:
    receipt = json.loads(rows.RECEIPT.read_text())
    rows._validate_historical_row_authority(receipt)


def test_token_frequency_positions_and_boundary_sentinel() -> None:
    fit = torch.full((1, rows.T_LEN), 9, dtype=torch.long)
    validation = torch.full((1, rows.T_LEN), 9, dtype=torch.long)
    fit[0, 64], fit[0, 65], fit[0, 256], fit[0, 257] = 1, 2, 3, 4
    validation[0, 64], validation[0, 65], validation[0, 256], validation[0, 257] = (
        1, 2, 3, 4
    )
    result = rows.derive_token_frequency_strata(
        fit, validation, rows.TOKEN_FREQUENCY_BOUNDARIES,
    )
    fit_targets = fit[:, 65:257]
    counts = torch.bincount(fit_targets.flatten(), minlength=rows.TOKEN_VOCAB).long()
    assert counts[1] == 0 and counts[2] == 1 and counts[3] == 1 and counts[4] == 0
    assignments = torch.bucketize(
        counts.index_select(0, validation[:, 65:257].flatten()),
        torch.tensor(rows.TOKEN_FREQUENCY_BOUNDARIES), right=True,
    ).view(1, -1)
    assert assignments[0, 0] == 1 and assignments[0, -1] == 1
    assert result["fit_token_counts_sha256"] == rows.tensor_sha256(counts)
    assert result["validation_assignment_sha256"] == rows.tensor_sha256(assignments)
    with pytest.raises(RuntimeError, match="boundaries changed"):
        rows.derive_token_frequency_strata(fit, validation, [1, 2, 4])


def test_v21_row_receipt_is_last_build_content_write() -> None:
    source = inspect.getsource(rows.build)
    authority = source.rfind("write_json_atomic(receipt, RECEIPT)")
    assert authority >= 0
    assert "write_" not in source[authority + len(
        "write_json_atomic(receipt, RECEIPT)"
    ):]
