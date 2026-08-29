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
AUDIT = HERE / "mlp2_error_rayleigh_v1_collector_independent_audit.json"
ROWS_RECEIPT = BQ / "mlp2_error_rayleigh_v1_rows_receipt.json"
PREDICTOR_RECEIPT = HERE / "mlp2_error_rayleigh_v1_design_predictor_receipt.json"
SOURCE_PATHS = tuple(dict.fromkeys((
    PREREG, ADDENDUM, RUNNER, TEST, CORE, CORE_TEST,
    HERE / "mlp2_error_rayleigh_metrics.py",
    HERE / "test_mlp2_error_rayleigh_metrics.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "run_mlp2_trajectory_robust_r512_v1_physical_eval.py",
    HERE / "test_mlp2_trajectory_robust_r512_v1_physical_eval.py",
    HERE / "run_mlp0_c512_mlp2_full512_composition_v1.py",
    HERE / "run_mlp2_rank512_refit_v1.py",
    HERE / "mlp0_native_down_program.py",
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
        "mlp2_error_rayleigh_v1_collector_independent_audit"
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
    stem = f"mlp2_error_rayleigh_v1_{role.lower()}"
    return {
        "authority": HERE / f"{stem}_authority.json",
        "ledger": HERE / f"{stem}_ledger.pt",
        "receipt": HERE / f"{stem}_receipt.json",
        "failure": HERE / f"{stem}_failure.json",
        "lock": Path(f"/workspace/runs/.{stem}.lock"),
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
    return value


def validate_predictor_unlock() -> tuple[dict[str, Any], str]:
    value, digest = stable_json(PREDICTOR_RECEIPT)
    required = {
        "schema", "status", "design_ledger_sha256", "design_receipt_sha256",
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
    return value, digest


def parent_snapshot() -> dict[str, Any]:
    return {
        "parents": prior.validate_parents(),
        "program_integrity": prior.expected_program_integrity(),
        "row_receipt_sha256": file_sha256(ROWS_RECEIPT),
        "checkpoint": facade.validate_snapshot().__dict__,
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


def collect(model, rows: torch.Tensor, programs, c512, device):
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
            seed = CONTROL_SEED + 100*pi + 10*bi
            controls = core.control_error_bank(bank["errors"][program_name], seed)
            control_hashes[f"{program_name}|{background}"] = {
                name: row_life.base.tensor_sha256(value) for name, value in controls.items()
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


def validate_ledger(value: Any, authority_sha: str, role: str) -> dict[str, Any]:
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


def publish_failure(paths, claim, exc: BaseException, authority, opened: bool):
    failure = {
        "schema": "mlp2_error_rayleigh_v1_collector_failure",
        "status": "terminal_failure_no_receipt", "role": authority.get("role") if authority else None,
        "error": repr(exc), "authority_exists": paths["authority"].exists(),
        "model_or_response_may_have_opened": opened,
        "artifact_hashes": {name: file_sha256(path) for name, path in paths.items()
                            if name not in ("lock", "failure", "receipt") and path.is_file()},
    }
    if not paths["receipt"].exists() and not paths["failure"].exists():
        base.atomic_json(paths["failure"], failure)


def run(role: str) -> None:
    paths = role_paths(role)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{role} collector namespace already exists")
    if role == "HELDOUT" and not PREDICTOR_RECEIPT.is_file():
        raise RuntimeError("HELDOUT remains locked until DESIGN predictor receipt")
    claim = row_life.base.acquire_claim(paths["lock"])
    authority = None
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
            if any(paths[name].exists() for name in ("authority", "ledger", "receipt", "failure")) \
                    or protected_snapshot(authority) != protected:
                raise RuntimeError("collector authority inputs changed")
            row_life.base.require_claim(claim, paths["lock"])

        base.atomic_json(paths["authority"], authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(paths["authority"])
        opened = True; started = time.time()
        rows, observed_sha = stable_torch(Path(entry["path"]), entry["file_sha256"])
        if observed_sha != entry["file_sha256"] or tuple(rows.shape) != (DOCUMENTS, 257) \
                or row_life.base.tensor_sha256(rows) != entry["tensor_sha256"]:
            raise RuntimeError(f"{role} row tensor changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        verify_protected(protected, authority, claim, paths)
        programs = load_programs(device)
        c512 = c512_tensors(device)
        with torch.inference_mode():
            features, finite, control_hashes, calls = collect(
                model, rows, programs, c512, device,
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
            if any(paths[name].exists() for name in ("ledger", "receipt", "failure")):
                raise RuntimeError("collector terminal raced ledger")

        base.atomic_torch(paths["ledger"], ledger, pre_link_check=ledger_guard)
        replay, ledger_sha = stable_torch(paths["ledger"])
        validate_ledger(replay, authority_sha, role)
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

        base.atomic_json(paths["receipt"], receipt, pre_link_check=receipt_guard)
        print(json.dumps(receipt, sort_keys=True, indent=2))
    except BaseException as exc:
        publish_failure(paths, claim, exc, authority, opened)
        raise
    finally:
        row_life.base.release_claim(claim, paths["lock"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=ROLE_NAMES, required=True)
    run(parser.parse_args().role)


if __name__ == "__main__":
    main()
