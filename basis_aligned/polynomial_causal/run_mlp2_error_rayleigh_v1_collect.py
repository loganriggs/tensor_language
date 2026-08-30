#!/usr/bin/env python3
"""Source-closed two-role physical collector for the MLP2 error-Rayleigh pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for source_root in (ROOT, HERE, BQ):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import mlp2_error_rayleigh_collector_core as core
import mlp2_error_rayleigh_predictor as predictor
import prepare_mlp2_error_rayleigh_v1_rows as row_life
import run_mlp0_c512_mlp2_full512_composition_v1 as base
import run_mlp2_rank512_refit_v1 as refit
import run_mlp2_trajectory_robust_r512_v1_physical_eval as prior
from mlp0_native_down_program import load_program


PREREG = HERE / "MLP2_ERROR_RAYLEIGH_VALIDITY_PILOT_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP2_ERROR_RAYLEIGH_V1_EXECUTION_ADDENDUM.md"
RUNNER = Path(__file__).resolve()
TEST = HERE / "test_run_mlp2_error_rayleigh_v1_collect.py"
CORE = HERE / "mlp2_error_rayleigh_collector_core.py"
CORE_TEST = HERE / "test_mlp2_error_rayleigh_collector_core.py"
AUDIT = HERE / "mlp2_error_rayleigh_v2_collector_independent_audit.json"
ROWS_RECEIPT = BQ / "mlp2_error_rayleigh_v1_rows_receipt.json"
PREDICTOR_RECEIPT = HERE / "mlp2_error_rayleigh_v2_design_predictor_receipt.json"
PREDICTOR_BUNDLE = HERE / "mlp2_error_rayleigh_v2_design_predictor_bundle.pt"
PREDICTOR_AUTHORITY = HERE / "mlp2_error_rayleigh_v2_design_predictor_authority.json"
PREDICTOR_AUDIT = HERE / "mlp2_error_rayleigh_v2_design_scorer_independent_audit.json"
RECOVERY_AMENDMENT = HERE / "MLP2_ERROR_RAYLEIGH_DESIGN_V2_RECOVERY_AMENDMENT.md"
V1_DESIGN_AUTHORITY = HERE / "mlp2_error_rayleigh_v1_design_authority.json"
V1_DESIGN_FAILURE = HERE / "mlp2_error_rayleigh_v1_design_failure.json"
V1_DESIGN_LEDGER = HERE / "mlp2_error_rayleigh_v1_design_ledger.pt"
V1_DESIGN_RECEIPT = HERE / "mlp2_error_rayleigh_v1_design_receipt.json"
V1_ABSENT_PATHS = (
    V1_DESIGN_LEDGER,
    V1_DESIGN_RECEIPT,
    Path("/workspace/runs/.mlp2_error_rayleigh_v1_design.lock"),
    HERE / "mlp2_error_rayleigh_v1_heldout_authority.json",
    HERE / "mlp2_error_rayleigh_v1_heldout_ledger.pt",
    HERE / "mlp2_error_rayleigh_v1_heldout_receipt.json",
    HERE / "mlp2_error_rayleigh_v1_heldout_failure.json",
    Path("/workspace/runs/.mlp2_error_rayleigh_v1_heldout.lock"),
    HERE / "mlp2_error_rayleigh_v1_design_predictor_authority.json",
    HERE / "mlp2_error_rayleigh_v1_design_predictor_bundle.pt",
    HERE / "mlp2_error_rayleigh_v1_design_predictor_receipt.json",
    HERE / "mlp2_error_rayleigh_v1_design_predictor_failure.json",
    Path("/workspace/runs/.mlp2_error_rayleigh_v1_design_predictor.lock"),
)
V1_AUTHORITY_SHA = "d5d6f785a61568ed1aa6979af1eeea76183d1ffb6f080415cc294a68252ae8db"
V1_FAILURE_SHA = "a8b6a88d342db2f2b2e3720cf87bb40caac4333d240dc27e04498d078585bbba"
SOURCE_PATHS = tuple(dict.fromkeys((
    PREREG, ADDENDUM, RECOVERY_AMENDMENT, RUNNER, TEST, CORE, CORE_TEST,
    HERE / "mlp2_error_rayleigh_metrics.py",
    HERE / "test_mlp2_error_rayleigh_metrics.py",
    HERE / "mlp2_error_rayleigh_predictor.py",
    HERE / "test_mlp2_error_rayleigh_predictor.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "run_mlp2_trajectory_robust_r512_v1_physical_eval.py",
    HERE / "test_mlp2_trajectory_robust_r512_v1_physical_eval.py",
    HERE / "run_mlp0_c512_mlp2_full512_composition_v1.py",
    HERE / "run_mlp2_rank512_refit_v1.py",
    HERE / "mlp0_native_down_program.py",
    HERE / "prepare_mlp2_trajectory_robust_r512_v1_eval_rows.py",
    HERE / "prepare_mlp0_c512_mlp2_full512_composition_v2_rows.py",
    HERE / "prepare_mlp0_c512_mlp2_full512_composition_v1_rows.py",
    HERE / "mlp2_cmr_v1_physical_program.py",
    HERE / "MLP2_TRAJECTORY_ROBUST_R512_V1_PHYSICAL_EVALUATION_ADDENDUM.md",
    HERE / "MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md",
    HERE / "MLP0_C512_MLP2_FULL512_COMPOSITION_V1_PREREGISTRATION.md",
    HERE / "MLP0_C512_MLP2_FULL512_COMPOSITION_V2_ROW_RECOVERY_AMENDMENT.md",
    HERE / "MLP2_RANK512_REFIT_V1_PREREGISTRATION.md",
    HERE / "test_bilin18_observed_model_facade.py",
    HERE / "test_mlp0_c512_mlp2_full512_composition_v1.py",
    HERE / "test_mlp0_native_down_program.py",
    HERE / "test_mlp2_cmr_v1_physical_program.py",
    HERE / "test_mlp2_rank512_refit_v1.py",
    HERE / "run_mlp2_error_rayleigh_v1_score_design.py",
    HERE / "test_run_mlp2_error_rayleigh_v1_score_design.py",
    *prior.row_life.SOURCE_PATHS, *base.SOURCE_PATHS, *refit.SOURCE_PATHS,
    *row_life.SOURCE_PATHS,
    ROOT / "jacclust/__init__.py", ROOT / "jacclust/tt_model.py",
)))

PROGRAM_NAMES = ("FULL512", "CONTINUE512", "ROBUST512")
BACKGROUND_NAMES = ("NATIVE", "C512")
ROLE_NAMES = ("DESIGN", "HELDOUT")
SCORING = slice(64, 256)
BATCH_SIZE = 4
DOCUMENTS = 32
CONTROL_SEED = 2026082951


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(commit: str) -> dict[str, str]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)):
        raise RuntimeError("collector source closure contains duplicates")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted collector source: {relative}")
        output[relative] = digest
    return output


def validate_audit(sources: Mapping[str, str]) -> tuple[dict[str, Any], str]:
    raw = AUDIT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if file_sha256(AUDIT) != digest:
        raise RuntimeError("collector audit changed while reading")
    value = json.loads(raw)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if set(value) != required or value.get("schema") != (
        "mlp2_error_rayleigh_v2_collector_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) \
            or value["tests_passed"] < 1 or not value.get("reviewer"):
        raise RuntimeError("collector audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("collector audit commit binding changed")
    return value, digest


def role_paths(role: str) -> dict[str, Path]:
    if role not in ROLE_NAMES:
        raise ValueError("collector role changed")
    stem = f"mlp2_error_rayleigh_v2_{role.lower()}"
    return {
        "authority": HERE / f"{stem}_authority.json",
        "ledger": HERE / f"{stem}_ledger.pt",
        "receipt": HERE / f"{stem}_receipt.json",
        "failure": HERE / f"{stem}_failure.json",
        "lock": Path(f"/workspace/runs/.{stem}.lock"),
    }


def v1_absence_state() -> dict[str, bool]:
    if len(V1_ABSENT_PATHS) != len(set(V1_ABSENT_PATHS)):
        raise RuntimeError("spent Rayleigh v1 absence set contains duplicates")
    return {str(path): path.exists() for path in V1_ABSENT_PATHS}


def validate_spent_v1_design() -> dict[str, Any]:
    authority, authority_sha = stable_json(V1_DESIGN_AUTHORITY, V1_AUTHORITY_SHA)
    failure, failure_sha = stable_json(V1_DESIGN_FAILURE, V1_FAILURE_SHA)
    absences = v1_absence_state()
    if authority_sha != V1_AUTHORITY_SHA or failure_sha != V1_FAILURE_SHA \
            or authority.get("schema") != "mlp2_error_rayleigh_v1_collector_authority" \
            or authority.get("role") != "DESIGN" \
            or failure != {
                "artifact_hashes": {"authority": V1_AUTHORITY_SHA},
                "authority_exists": True,
                "error": "TypeError('Got unsupported ScalarType BFloat16')",
                "model_or_response_may_have_opened": True,
                "protected_observation": {"status": "matches"},
                "role": "DESIGN",
                "schema": "mlp2_error_rayleigh_v1_collector_failure",
                "status": "terminal_failure_no_receipt",
            } or any(absences.values()):
        raise RuntimeError("spent Rayleigh DESIGN v1 failure chain changed")
    # Close the cross-file read window: no authority/failure mutation or late v1
    # terminal may occur between the first joins and the published snapshot.
    authority_replay, authority_replay_sha = stable_json(
        V1_DESIGN_AUTHORITY, V1_AUTHORITY_SHA,
    )
    failure_replay, failure_replay_sha = stable_json(
        V1_DESIGN_FAILURE, V1_FAILURE_SHA,
    )
    absences_replay = v1_absence_state()
    if authority_replay != authority or authority_replay_sha != authority_sha \
            or failure_replay != failure or failure_replay_sha != failure_sha \
            or absences_replay != absences or any(absences_replay.values()):
        raise RuntimeError("spent Rayleigh v1 aggregate lineage raced validation")
    return {
        "authority_sha256": authority_sha,
        "failure_sha256": failure_sha,
        "absent_paths": absences,
    }


def stable_json(path: Path, expected: str | None = None):
    return base.stable_json(path, expected)


def stable_torch(path: Path, expected: str | None = None):
    return base.stable_torch(path, expected)


def validate_row_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != "mlp2_error_rayleigh_v1_rows" \
            or value.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" \
            or value.get("selection") != {
                "start_document_index": 121000, "documents_per_role": 32,
                "token_length": 257, "scored_slice": [64, 256],
            } or value.get("roles") != {
                "DESIGN": {"authorized_for_training": True, "authorized_for_evaluation": False},
                "HELDOUT": {"authorized_for_training": False, "authorized_for_evaluation": True},
            } or value.get("outcome_access") != {
                "model_loaded": False, "training_run": False,
            } or set(value.get("entries", {})) != set(ROLE_NAMES) \
            or not all(value.get("disjointness", {}).values()):
        raise RuntimeError("Rayleigh row receipt semantics changed")
    if len(value.get("provenance", {}).get("DESIGN", [])) != DOCUMENTS \
            or len(value.get("provenance", {}).get("HELDOUT", [])) != DOCUMENTS:
        raise RuntimeError("Rayleigh row provenance changed")
    for role, entry in value["entries"].items():
        path = Path(entry["path"])
        if entry.get("shape") != [DOCUMENTS, 257] or entry.get("dtype") != "torch.int64" \
                or not path.is_file() or file_sha256(path) != entry.get("file_sha256"):
            raise RuntimeError(f"Rayleigh {role} row bytes changed")
    commit = value.get("source_commit")
    source_values = value.get("source_hashes")
    if not isinstance(commit, str) or not isinstance(source_values, dict):
        raise RuntimeError("Rayleigh row source binding is absent")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    for relative, expected in source_values.items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError("Rayleigh row committed source binding changed")
    return value


def validate_predictor_unlock() -> tuple[dict[str, Any], str]:
    value, digest = stable_json(PREDICTOR_RECEIPT)
    required = {
        "schema", "status", "design_ledger_sha256", "design_receipt_sha256",
        "predictor_authority_sha256", "scorer_audit_sha256",
        "predictor_bundle_sha256", "heldout_unlocked",
    }
    design = role_paths("DESIGN")
    if set(value) != required or value.get("schema") != (
        "mlp2_error_rayleigh_v1_design_predictor_receipt"
    ) or value.get("status") != "design_predictor_frozen_receipt_last" \
            or value.get("heldout_unlocked") is not True \
            or value.get("design_ledger_sha256") != file_sha256(design["ledger"]) \
            or value.get("design_receipt_sha256") != file_sha256(design["receipt"]):
        raise RuntimeError("HELDOUT predictor unlock changed")
    design_receipt, _ = stable_json(design["receipt"], value["design_receipt_sha256"])
    design_ledger, ledger_sha = stable_torch(design["ledger"], value["design_ledger_sha256"])
    required_receipt = {
        "schema", "status", "role", "authority_sha256", "ledger_sha256",
        "runtime_s", "model_responses_opened", "heldout_predictor_was_frozen",
    }
    if set(design_receipt) != required_receipt \
            or design_receipt.get("schema") != "mlp2_error_rayleigh_v1_collector_receipt" \
            or design_receipt.get("status") != "role_measurements_complete_receipt_last" \
            or design_receipt.get("role") != "DESIGN" \
            or design_receipt.get("ledger_sha256") != ledger_sha \
            or design_receipt.get("model_responses_opened") is not True \
            or design_receipt.get("heldout_predictor_was_frozen") is not False:
        raise RuntimeError("HELDOUT DESIGN receipt authority chain changed")
    design_authority, authority_sha = stable_json(
        design["authority"], design_receipt["authority_sha256"],
    )
    required_authority = {
        "schema", "status", "role", "source_commit", "source_hashes",
        "audit_sha256", "audit_reviewer", "row_receipt_sha256", "row_file_sha256",
        "parent_snapshot", "predictor_unlock_sha256", "programs", "backgrounds",
        "controls", "amplitudes", "control_seed", "scored_slice",
        "attention_capture_sites", "outcome_access",
    }
    if set(design_authority) != required_authority \
            or design_authority.get("schema") != "mlp2_error_rayleigh_v1_collector_authority" \
            or design_authority.get("status") != "frozen_before_role_response_open" \
            or design_authority.get("role") != "DESIGN" \
            or design_authority.get("predictor_unlock_sha256") is not None \
            or design_authority.get("outcome_access") is not False \
            or authority_sha != design_receipt["authority_sha256"]:
        raise RuntimeError("HELDOUT DESIGN authority semantics changed")
    # This is deliberately a full semantic replay, not only a hash join.  It binds
    # the DESIGN ledger to the checkpoint, rows, programs, audit, and sources that
    # were frozen before the DESIGN responses were opened.
    protected_snapshot(design_authority)
    validate_ledger(design_ledger, design_receipt["authority_sha256"], "DESIGN",
                    design_authority["parent_snapshot"]["checkpoint"])
    bundle, bundle_sha = stable_torch(PREDICTOR_BUNDLE, value["predictor_bundle_sha256"])
    predictor.validate_frozen_bundle(bundle)
    if bundle_sha != value["predictor_bundle_sha256"]:
        raise RuntimeError("HELDOUT predictor bundle hash changed")
    scorer_authority, scorer_authority_sha = stable_json(
        PREDICTOR_AUTHORITY, value["predictor_authority_sha256"],
    )
    scorer_required = {
        "schema", "status", "source_commit", "source_hashes", "audit_sha256",
        "audit_reviewer", "design_receipt_sha256", "design_ledger_sha256",
        "design_authority_sha256", "ridge_grid", "families", "heldout_opened",
    }
    if set(scorer_authority) != scorer_required \
            or scorer_authority.get("schema") != "mlp2_error_rayleigh_v1_design_predictor_authority" \
            or scorer_authority.get("status") != "frozen_before_design_ledger_open" \
            or scorer_authority.get("heldout_opened") is not False \
            or scorer_authority.get("design_receipt_sha256") != value["design_receipt_sha256"] \
            or scorer_authority.get("design_ledger_sha256") != value["design_ledger_sha256"] \
            or scorer_authority.get("design_authority_sha256") != design_receipt["authority_sha256"] \
            or scorer_authority.get("audit_sha256") != value["scorer_audit_sha256"] \
            or scorer_authority.get("ridge_grid") != list(predictor.RIDGE_GRID) \
            or scorer_authority.get("families") != {
                name: list(features) for name, features in predictor.FAMILIES.items()
            } \
            or scorer_authority_sha != value["predictor_authority_sha256"]:
        raise RuntimeError("HELDOUT scorer authority chain changed")
    if committed_hash_map(scorer_authority["source_commit"], scorer_authority["source_hashes"]) \
            != scorer_authority["source_hashes"]:
        raise RuntimeError("HELDOUT scorer source closure changed")
    scorer_audit, scorer_audit_sha = stable_json(PREDICTOR_AUDIT, value["scorer_audit_sha256"])
    audit_required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if set(scorer_audit) != audit_required \
            or scorer_audit.get("schema") != "mlp2_error_rayleigh_v2_design_scorer_independent_audit" \
            or scorer_audit.get("status") != "GO" or scorer_audit.get("outcome_access") is not False \
            or scorer_audit.get("audited_source_hashes") != scorer_authority["source_hashes"] \
            or scorer_audit.get("reviewer") != scorer_authority["audit_reviewer"] \
            or not isinstance(scorer_audit.get("tests_passed"), int) \
            or scorer_audit["tests_passed"] < 1 \
            or scorer_audit_sha != scorer_authority["audit_sha256"]:
        raise RuntimeError("HELDOUT scorer audit chain changed")
    if committed_hash_map(
        scorer_audit["audited_source_commit"], scorer_audit["audited_source_hashes"],
    ) != scorer_authority["source_hashes"]:
        raise RuntimeError("HELDOUT scorer audit source closure changed")
    recomputed = predictor.serialize_fit(
        predictor.fit_design(design_ledger["features"], design_ledger["finite"]),
    )
    predictor.validate_frozen_bundle(recomputed)
    if not predictor.exact_nested_equal(recomputed, bundle):
        raise RuntimeError("HELDOUT predictor does not reproduce from DESIGN ledger")
    return value, digest


def committed_hash_map(commit: str, values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(commit, str) or not isinstance(values, Mapping):
        raise RuntimeError("committed source map changed")
    expected_relatives = {str(path.relative_to(ROOT)) for path in SOURCE_PATHS}
    if set(values) != expected_relatives:
        raise RuntimeError("committed scorer source closure is not the exact canonical set")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    observed = {}
    for relative, expected in values.items():
        path = ROOT / relative
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected or not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError("committed source bytes changed")
        observed[relative] = digest
    return observed


def parent_snapshot() -> dict[str, Any]:
    return {
        "parents": prior.validate_parents(),
        "program_integrity": prior.expected_program_integrity(),
        "row_receipt_sha256": file_sha256(ROWS_RECEIPT),
        "checkpoint": facade.validate_snapshot().__dict__,
        "spent_design_v1": validate_spent_v1_design(),
    }


def protected_snapshot(authority: Mapping[str, Any]) -> dict[str, Any]:
    if source_hashes(authority["source_commit"]) != authority["source_hashes"]:
        raise RuntimeError("collector sources changed")
    _, audit_sha = validate_audit(authority["source_hashes"])
    rows, row_sha = stable_json(ROWS_RECEIPT, authority["row_receipt_sha256"])
    validate_row_receipt(rows)
    entry = rows["entries"][authority["role"]]
    if entry["file_sha256"] != authority["row_file_sha256"]:
        raise RuntimeError("collector role row changed")
    output = parent_snapshot()
    if output != authority["parent_snapshot"] or audit_sha != authority["audit_sha256"] \
            or row_sha != authority["row_receipt_sha256"]:
        raise RuntimeError("collector protected parent changed")
    if authority["role"] == "HELDOUT":
        _, unlock_sha = validate_predictor_unlock()
        if unlock_sha != authority["predictor_unlock_sha256"]:
            raise RuntimeError("collector predictor unlock changed")
    return output


def verify_protected(expected: Mapping[str, Any], authority: Mapping[str, Any], claim,
                     paths: Mapping[str, Path]) -> None:
    row_life.base.require_claim(claim, paths["lock"])
    if protected_snapshot(authority) != expected:
        raise RuntimeError("collector protected snapshot changed")
    row_life.base.require_claim(claim, paths["lock"])


def load_programs(device: torch.device):
    old_bundle, _ = stable_torch(base.FULL_BUNDLE, base.FULL_BUNDLE_SHA)
    robust_bundle, _ = stable_torch(prior.ROBUST_BUNDLE, prior.ROBUST_BUNDLE_SHA)
    states = {
        "FULL512": old_bundle["programs"]["FULL512"],
        "CONTINUE512": robust_bundle["programs"]["CONTINUE512"],
        "ROBUST512": robust_bundle["programs"]["ROBUST512"],
    }
    return {name: refit.build_from_state(value, device).eval()
            for name, value in states.items()}


def c512_tensors(device: torch.device):
    value = load_program(base.C512_PATH)
    return {key: value[key].to(device) for key in ("intercept", "left", "right")}


def forward_capture(
    model, tokens: torch.Tensor, background: str, mode: str,
    c512: Mapping[str, torch.Tensor], program=None, candidate=None, error=None,
    alpha: float = 0.0, calls: dict[str, int] | None = None,
):
    capture: dict[str, torch.Tensor] = {}
    calls = {} if calls is None else calls

    def count(name: str, amount: int = 1):
        calls[name] = calls.get(name, 0) + amount

    def attention(event: facade.AttentionEvent):
        write, next_v1 = event.block.attn(event.state, event.first_value)
        count("attention_calls")
        if event.site in (5, 6):
            capture[f"attention{event.site}"] = write
        return write, next_v1

    def mlp(event: facade.EarlyMLPEvent):
        if event.site == 0 and background == "C512":
            count("c512_calls")
            return base.c512_write(event, c512)
        if event.site != 2:
            count("native_mlp_calls")
            return event.block.mlp(event.state)
        capture["mlp2_state"] = event.state
        if mode == "DIRECT":
            count("direct_program_calls")
            return program(event.state)
        native = event.block.mlp(event.state)
        count("native_mlp_calls"); count("native_mlp2_calls")
        capture["native_mlp2"] = native
        if mode == "BASELINE":
            return native
        if mode == "ACTUAL":
            count("injected_calls")
            return core.actual_write(native, candidate, alpha)
        if mode == "CONTROL":
            count("injected_calls")
            return core.control_write(native, error, alpha)
        raise ValueError("collector forward mode changed")

    count("outer_forwards")
    logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
    count("outer_returns")
    if set(capture).issuperset({"attention5", "attention6"}) is False:
        raise RuntimeError("collector consumer capture is incomplete")
    return {"logits": logits, **capture}


def capture_error_banks(model, rows: torch.Tensor, programs, c512, device, calls):
    banks = {}
    for background in BACKGROUND_NAMES:
        parts = {name: [] for name in ("logits", "attention5", "attention6",
                                       "state", "native")}
        candidate_parts = {program: [] for program in PROGRAM_NAMES}
        for start in range(0, DOCUMENTS, BATCH_SIZE):
            tokens = rows[start:start+BATCH_SIZE, :-1].to(device)
            captured = forward_capture(
                model, tokens, background, "BASELINE", c512, calls=calls,
            )
            parts["logits"].append(captured["logits"][:, SCORING].detach())
            parts["attention5"].append(captured["attention5"].detach())
            parts["attention6"].append(captured["attention6"].detach())
            parts["state"].append(captured["mlp2_state"].detach())
            parts["native"].append(captured["native_mlp2"].detach())
            for name, program in programs.items():
                calls["offline_program_calls"] = calls.get("offline_program_calls", 0) + 1
                candidate_parts[name].append(program(captured["mlp2_state"]).detach())
        bank = {name: torch.cat(value) for name, value in parts.items()}
        bank["candidate"] = {name: torch.cat(value) for name, value in candidate_parts.items()}
        bank["errors"] = {
            name: (bank["candidate"][name].float() - bank["native"].float()).cpu()
            for name in PROGRAM_NAMES
        }
        banks[background] = bank
    return banks


def control_seed(role: str, program_index: int, background_index: int) -> int:
    if role not in ROLE_NAMES or program_index not in range(3) \
            or background_index not in range(2):
        raise ValueError("control seed coordinates changed")
    return CONTROL_SEED + 10_000 * ROLE_NAMES.index(role) + 100*program_index \
        + 10*background_index


def tensor_sha256_raw(value: torch.Tensor) -> str:
    """Hash dtype, shape, and exact raw bytes without NumPy dtype conversion."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def expected_calls() -> dict[str, int]:
    batches = DOCUMENTS // BATCH_SIZE
    outer = 2*batches + 3*2*3*4*batches + 3*2*2*batches
    native_mlp2 = 2*batches + 3*2*3*4*batches + 3*2*batches
    direct = 3*2*batches
    return {
        "outer_forwards": outer, "outer_returns": outer,
        "attention_calls": outer * 18,
        "native_mlp_calls": outer * 18 - direct - outer // 2,
        "native_mlp2_calls": native_mlp2,
        "c512_calls": outer // 2,
        "injected_calls": 3*2*3*4*batches + 3*2*batches,
        "direct_program_calls": direct,
        "offline_program_calls": 3*2*batches,
    }


def collect(model, rows: torch.Tensor, programs, c512, device, role: str):
    calls: dict[str, int] = {}
    banks = capture_error_banks(model, rows, programs, c512, device, calls)
    features = torch.empty(
        len(PROGRAM_NAMES), len(BACKGROUND_NAMES), len(core.CONTROL_NAMES),
        DOCUMENTS, len(core.FEATURE_NAMES), dtype=torch.float64,
    )
    finite = torch.empty(
        len(PROGRAM_NAMES), len(BACKGROUND_NAMES), DOCUMENTS,
        len(core.FINITE_NAMES), dtype=torch.float64,
    )
    control_hashes = {}
    targets_all = rows[:, 1:][:, SCORING]
    for pi, program_name in enumerate(PROGRAM_NAMES):
        for bi, background in enumerate(BACKGROUND_NAMES):
            bank = banks[background]
            seed = control_seed(role, pi, bi)
            controls = core.control_error_bank(bank["errors"][program_name], seed)
            control_hashes[f"{program_name}|{background}"] = {
                "seed": seed,
                "bindings": {
                    "mlp2_state": tensor_sha256_raw(bank["state"]),
                    "native_write": tensor_sha256_raw(bank["native"]),
                    "candidate_write": tensor_sha256_raw(
                        bank["candidate"][program_name]
                    ),
                },
                "errors": {
                    name: tensor_sha256_raw(value) for name, value in controls.items()
                },
            }
            for ci, control_name in enumerate(core.CONTROL_NAMES):
                chunks = []
                for start in range(0, DOCUMENTS, BATCH_SIZE):
                    stop = start + BATCH_SIZE
                    tokens = rows[start:stop, :-1].to(device)
                    targets = targets_all[start:stop].to(device)
                    by_amplitude = {}
                    for amplitude in core.AMPLITUDES:
                        changed = {}
                        for sign in (-1.0, 1.0):
                            alpha = sign * amplitude
                            if control_name == "ACTUAL":
                                changed[sign] = forward_capture(
                                    model, tokens, background, "ACTUAL", c512,
                                    candidate=bank["candidate"][program_name][start:stop],
                                    alpha=alpha, calls=calls,
                                )
                            else:
                                changed[sign] = forward_capture(
                                    model, tokens, background, "CONTROL", c512,
                                    error=controls[control_name][start:stop].to(device),
                                    alpha=alpha, calls=calls,
                                )
                        by_amplitude[amplitude] = core.response_statistics(
                            bank["logits"][start:stop], changed[1.0]["logits"][:, SCORING],
                            changed[-1.0]["logits"][:, SCORING],
                            bank["attention5"][start:stop], changed[1.0]["attention5"],
                            changed[-1.0]["attention5"], bank["attention6"][start:stop],
                            changed[1.0]["attention6"], changed[-1.0]["attention6"],
                            targets, amplitude,
                        )
                    local = controls[control_name][start:stop].flatten(1).square().mean(1)
                    chunks.append(core.pack_features(local, by_amplitude))
                features[pi, bi, ci] = torch.cat(chunks)

            replay_chunks = []
            for start in range(0, DOCUMENTS, BATCH_SIZE):
                stop = start + BATCH_SIZE
                tokens = rows[start:stop, :-1].to(device)
                targets = targets_all[start:stop].to(device)
                direct = forward_capture(
                    model, tokens, background, "DIRECT", c512,
                    program=programs[program_name], calls=calls,
                )
                injected = forward_capture(
                    model, tokens, background, "ACTUAL", c512,
                    candidate=bank["candidate"][program_name][start:stop],
                    alpha=1.0, calls=calls,
                )
                replay_chunks.append(core.replay_statistics(
                    bank["logits"][start:stop], direct["logits"][:, SCORING],
                    injected["logits"][:, SCORING], bank["attention5"][start:stop],
                    direct["attention5"], injected["attention5"],
                    bank["attention6"][start:stop], direct["attention6"],
                    injected["attention6"], targets,
                ))
            finite[pi, bi] = torch.cat(replay_chunks)
    if calls != expected_calls():
        raise RuntimeError(f"collector call census changed: {calls} != {expected_calls()}")
    if not bool((finite[..., 5:] == 1).all()) or not bool((finite[..., 2:5] == 0).all()):
        raise RuntimeError("alpha=1 physical replay is not exact")
    return features, finite, control_hashes, calls


def validate_ledger(value: Any, authority_sha: str, role: str,
                    expected_checkpoint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = {
        "schema", "role", "features", "finite", "axes", "control_hashes", "calls",
        "authority_sha256", "checkpoint",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema") != "mlp2_error_rayleigh_v1_role_ledger" \
            or value.get("role") != role or value.get("authority_sha256") != authority_sha \
            or value.get("calls") != expected_calls() or value.get("axes") != {
                "programs": list(PROGRAM_NAMES), "backgrounds": list(BACKGROUND_NAMES),
                "controls": list(core.CONTROL_NAMES),
                "features": list(core.FEATURE_NAMES), "finite": list(core.FINITE_NAMES),
                "documents": DOCUMENTS,
            }:
        raise RuntimeError("collector ledger metadata changed")
    if expected_checkpoint is not None and value.get("checkpoint") != dict(expected_checkpoint):
        raise RuntimeError("collector ledger checkpoint changed")
    expected_keys = {
        f"{program}|{background}"
        for program in PROGRAM_NAMES for background in BACKGROUND_NAMES
    }
    if set(value.get("control_hashes", {})) != expected_keys:
        raise RuntimeError("collector control-hash cells changed")
    for pi, program in enumerate(PROGRAM_NAMES):
        for bi, background in enumerate(BACKGROUND_NAMES):
            cell = value["control_hashes"][f"{program}|{background}"]
            digests = (*cell.get("bindings", {}).values(), *cell.get("errors", {}).values()) \
                if isinstance(cell, dict) and isinstance(cell.get("bindings"), dict) \
                and isinstance(cell.get("errors"), dict) else ()
            if not isinstance(cell, dict) or set(cell) != {"seed", "bindings", "errors"} \
                    or cell["seed"] != control_seed(role, pi, bi) \
                    or set(cell["bindings"]) != {
                        "mlp2_state", "native_write", "candidate_write",
                    } or set(cell["errors"]) != set(core.CONTROL_NAMES) \
                    or any(not isinstance(digest, str) or len(digest) != 64
                           or any(character not in "0123456789abcdef" for character in digest)
                           for digest in digests):
                raise RuntimeError("collector control-hash schema changed")
    expected_feature_shape = (3, 2, 3, DOCUMENTS, len(core.FEATURE_NAMES))
    expected_finite_shape = (3, 2, DOCUMENTS, len(core.FINITE_NAMES))
    if not isinstance(value["features"], torch.Tensor) \
            or value["features"].dtype != torch.float64 \
            or tuple(value["features"].shape) != expected_feature_shape \
            or not torch.isfinite(value["features"]).all() \
            or not isinstance(value["finite"], torch.Tensor) \
            or value["finite"].dtype != torch.float64 \
            or tuple(value["finite"].shape) != expected_finite_shape \
            or not torch.isfinite(value["finite"]).all() \
            or not bool((value["finite"][..., 5:] == 1).all()) \
            or not bool((value["finite"][..., 2:5] == 0).all()):
        raise RuntimeError("collector ledger tensors changed")
    return value


def artifact_snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        name: file_sha256(path) for name, path in paths.items()
        if name not in ("lock", "failure", "receipt") and path.is_file()
    }


def publish_failure(paths, claim, exc: BaseException, authority,
                    protected: Mapping[str, Any] | None, opened: bool):
    def observe_protected() -> dict[str, Any]:
        if authority is None or protected is None or not paths["authority"].is_file():
            return {"status": "not_available"}
        try:
            current = protected_snapshot(authority)
            return {"status": "matches" if current == protected else "mismatch"}
        except BaseException as replay_error:
            return {"status": "replay_error", "error": repr(replay_error)}

    frozen_artifacts = artifact_snapshot(paths)
    frozen_protected_observation = observe_protected()
    failure = {
        "schema": "mlp2_error_rayleigh_v1_collector_failure",
        "status": "terminal_failure_no_receipt", "role": authority.get("role") if authority else None,
        "error": repr(exc), "authority_exists": paths["authority"].exists(),
        "model_or_response_may_have_opened": opened,
        "artifact_hashes": frozen_artifacts,
        "protected_observation": frozen_protected_observation,
    }

    def failure_guard():
        row_life.base.require_claim(claim, paths["lock"])
        if paths["receipt"].exists() or paths["failure"].exists() \
                or artifact_snapshot(paths) != frozen_artifacts:
            raise RuntimeError("collector failure terminal or artifacts raced")
        if authority is not None and paths["authority"].is_file():
            observed, observed_sha = stable_json(
                paths["authority"], frozen_artifacts.get("authority"),
            )
            if observed != authority or observed_sha != frozen_artifacts.get("authority"):
                raise RuntimeError("collector failure authority semantic join changed")
            # Protected input drift is itself a failure class that must remain
            # publishable.  The observation above records it, while the authority
            # and any partial owned artifacts remain hash- and semantics-bound.
        if paths["receipt"].exists() or paths["failure"].exists() \
                or artifact_snapshot(paths) != frozen_artifacts:
            raise RuntimeError("collector failure terminal raced protected replay")
        row_life.base.require_claim(claim, paths["lock"])

    if paths["receipt"].exists() or paths["failure"].exists():
        return
    # Failure publication is the final fallible action on the failure path.
    base.atomic_json(paths["failure"], failure, pre_link_check=failure_guard)


def run(role: str) -> None:
    paths = role_paths(role)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{role} collector namespace already exists")
    if role == "HELDOUT" and not PREDICTOR_RECEIPT.is_file():
        raise RuntimeError("HELDOUT remains locked until DESIGN predictor receipt")
    claim = row_life.base.acquire_claim(paths["lock"])
    authority = None
    protected = None
    opened = False
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        sources = source_hashes(commit)
        audit, audit_sha = validate_audit(sources)
        rows_receipt, rows_sha = stable_json(ROWS_RECEIPT)
        validate_row_receipt(rows_receipt)
        entry = rows_receipt["entries"][role]
        parents = parent_snapshot()
        predictor_sha = validate_predictor_unlock()[1] if role == "HELDOUT" else None
        authority = {
            "schema": "mlp2_error_rayleigh_v1_collector_authority",
            "status": "frozen_before_role_response_open", "role": role,
            "source_commit": commit, "source_hashes": sources,
            "audit_sha256": audit_sha, "audit_reviewer": audit["reviewer"],
            "row_receipt_sha256": rows_sha, "row_file_sha256": entry["file_sha256"],
            "parent_snapshot": parents, "predictor_unlock_sha256": predictor_sha,
            "programs": list(PROGRAM_NAMES), "backgrounds": list(BACKGROUND_NAMES),
            "controls": list(core.CONTROL_NAMES), "amplitudes": list(core.AMPLITUDES),
            "control_seed": CONTROL_SEED, "scored_slice": [64, 256],
            "attention_capture_sites": [5, 6], "outcome_access": False,
        }
        protected = protected_snapshot(authority)

        def authority_guard():
            row_life.base.require_claim(claim, paths["lock"])
            if any(paths[name].exists() for name in ("authority", "ledger", "receipt", "failure")):
                raise RuntimeError("collector authority terminal appeared")
            if protected_snapshot(authority) != protected:
                raise RuntimeError("collector authority inputs changed")
            if any(paths[name].exists() for name in ("authority", "ledger", "receipt", "failure")):
                raise RuntimeError("collector authority terminal raced protected replay")
            row_life.base.require_claim(claim, paths["lock"])

        base.atomic_json(paths["authority"], authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(paths["authority"])
        observed_authority, observed_authority_sha = stable_json(
            paths["authority"], authority_sha,
        )
        if observed_authority != authority or observed_authority_sha != authority_sha:
            raise RuntimeError("collector authority changed before role access")
        opened = True; started = time.time()
        rows, observed_sha = stable_torch(Path(entry["path"]), entry["file_sha256"])
        if observed_sha != entry["file_sha256"] or tuple(rows.shape) != (DOCUMENTS, 257) \
                or row_life.base.tensor_sha256(rows) != entry["tensor_sha256"]:
            raise RuntimeError(f"{role} row tensor changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        if checkpoint.__dict__ != authority["parent_snapshot"]["checkpoint"]:
            raise RuntimeError("collector loaded checkpoint differs from frozen authority")
        verify_protected(protected, authority, claim, paths)
        programs = load_programs(device)
        c512 = c512_tensors(device)

        def collection_guard():
            row_life.base.require_claim(claim, paths["lock"])
            observed, observed_sha = stable_json(paths["authority"], authority_sha)
            if observed != authority or observed_sha != authority_sha:
                raise RuntimeError("collector authority changed at final collection boundary")
            if any(paths[name].exists() for name in ("ledger", "receipt", "failure")):
                raise RuntimeError("collector terminal appeared before collection")
            if protected_snapshot(authority) != protected:
                raise RuntimeError("collector protected state changed before collection")
            if any(paths[name].exists() for name in ("ledger", "receipt", "failure")):
                raise RuntimeError("collector terminal raced final collection guard")
            observed, observed_sha = stable_json(paths["authority"], authority_sha)
            if observed != authority or observed_sha != authority_sha:
                raise RuntimeError("collector authority raced final collection guard")
            if any(paths[name].exists() for name in ("ledger", "receipt", "failure")):
                raise RuntimeError("collector terminal raced final authority reload")
            row_life.base.require_claim(claim, paths["lock"])

        collection_guard()
        with torch.inference_mode():
            features, finite, control_hashes, calls = collect(
                model, rows, programs, c512, device, role,
            )
        ledger = {
            "schema": "mlp2_error_rayleigh_v1_role_ledger", "role": role,
            "features": features, "finite": finite,
            "axes": {"programs": list(PROGRAM_NAMES),
                     "backgrounds": list(BACKGROUND_NAMES),
                     "controls": list(core.CONTROL_NAMES),
                     "features": list(core.FEATURE_NAMES),
                     "finite": list(core.FINITE_NAMES), "documents": DOCUMENTS},
            "control_hashes": control_hashes, "calls": calls,
            "authority_sha256": authority_sha, "checkpoint": checkpoint.__dict__,
        }

        def ledger_guard():
            verify_protected(protected, authority, claim, paths)
            observed_authority, observed_sha = stable_json(paths["authority"], authority_sha)
            if observed_authority != authority or observed_sha != authority_sha:
                raise RuntimeError("collector ledger authority semantic join changed")
            if any(paths[name].exists() for name in ("ledger", "receipt", "failure")):
                raise RuntimeError("collector terminal raced ledger")
            row_life.base.require_claim(claim, paths["lock"])

        base.atomic_torch(paths["ledger"], ledger, pre_link_check=ledger_guard)
        replay, ledger_sha = stable_torch(paths["ledger"])
        validate_ledger(replay, authority_sha, role, authority["parent_snapshot"]["checkpoint"])
        receipt = {
            "schema": "mlp2_error_rayleigh_v1_collector_receipt",
            "status": "role_measurements_complete_receipt_last", "role": role,
            "authority_sha256": authority_sha, "ledger_sha256": ledger_sha,
            "runtime_s": time.time() - started, "model_responses_opened": True,
            "heldout_predictor_was_frozen": role == "HELDOUT",
        }

        def receipt_guard():
            verify_protected(protected, authority, claim, paths)
            stable_json(paths["authority"], authority_sha)
            stable_torch(paths["ledger"], ledger_sha)
            if paths["receipt"].exists() or paths["failure"].exists():
                raise RuntimeError("collector terminal raced receipt")
            row_life.base.require_claim(claim, paths["lock"])

        rendered_receipt = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
        print(rendered_receipt)
        # Receipt publication is the final fallible action in the success path.
        base.atomic_json(paths["receipt"], receipt, pre_link_check=receipt_guard)
    except BaseException as exc:
        try:
            publish_failure(paths, claim, exc, authority, protected, opened)
        except BaseException:
            # A lost claim or semantically mutated authority must fail closed, but a
            # secondary publication error must not replace the scientific root cause.
            pass
        raise
    finally:
        row_life.base.release_claim(claim, paths["lock"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=ROLE_NAMES, required=True)
    run(parser.parse_args().role)


if __name__ == "__main__":
    main()
