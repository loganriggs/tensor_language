#!/usr/bin/env python3
"""V2 transaction wrapper; reuses the audited v1 scientific functions exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for source_root in (ROOT, HERE, BQ):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
from mlp0_native_down_program import load_program
import prepare_mlp2_trajectory_robust_r512_v2_eval_rows as row_life
import run_mlp2_trajectory_robust_r512_v1_physical_eval as science

io = science.base
refit = science.refit
ROWS_RECEIPT = row_life.RECEIPT
AUTHORITY = HERE / "mlp2_trajectory_robust_r512_v2_physical_eval_authority.json"
LEDGER = HERE / "mlp2_trajectory_robust_r512_v2_physical_eval_ledger.pt"
RESULT = HERE / "mlp2_trajectory_robust_r512_v2_physical_eval_result.json"
RECEIPT = HERE / "mlp2_trajectory_robust_r512_v2_physical_eval_receipt.json"
FAILURE = HERE / "mlp2_trajectory_robust_r512_v2_physical_eval_failure.json"
LOCK = Path("/workspace/runs/.mlp2_trajectory_robust_r512_v2_physical_eval.lock")

AUTHORITY_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_authority"
LEDGER_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_ledger"
RESULT_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_result"
RECEIPT_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_receipt"
FAILURE_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_failure"


def committed_sources() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    return commit, row_life.source_hashes(commit)


def verify_sources(commit: str, expected: dict[str, str]) -> None:
    if row_life.source_hashes(commit) != expected:
        raise RuntimeError("v2 evaluation source closure changed")


def validate_row_receipt(value: Any, sources: dict[str, str]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != row_life.RECEIPT_SCHEMA \
            or value.get("recovery_admission") != row_life.recovery_admission():
        raise RuntimeError("v2 row recovery receipt or admission changed")
    core = dict(value)
    core.pop("recovery_admission")
    core["schema"] = "mlp2_trajectory_robust_r512_v1_physical_eval_rows"
    science.validate_row_receipt(core, sources)
    return value


def protected_snapshot(authority: dict[str, Any]) -> dict[str, Any]:
    verify_sources(authority["source_commit"], authority["source_hashes"])
    audit, audit_sha = row_life.validate_independent_audit(authority["source_hashes"])
    rows, rows_sha = io.stable_json(ROWS_RECEIPT, authority["row_receipt_sha256"])
    validate_row_receipt(rows, authority["source_hashes"])
    parents = science.validate_parents()
    programs = science.expected_program_integrity()
    recovery = row_life.recovery_admission()
    if parents != authority["parents"] or programs != authority["program_integrity"] \
            or recovery != authority["recovery_admission"] \
            or audit_sha != authority["audit_sha256"]:
        raise RuntimeError("v2 evaluation protected parent changed")
    snapshot = Path(facade.DEFAULT_SNAPSHOT)
    return {
        "source_commit": authority["source_commit"],
        "source_hashes": authority["source_hashes"],
        "audit_sha256": audit_sha, "audit_commit": audit["audited_source_commit"],
        "row_receipt_sha256": rows_sha,
        "row_file_hashes": {k: io.file_sha256(Path(v["path"]))
                            for k, v in rows["entries"].items()},
        "parents": parents, "program_integrity": programs,
        "recovery_admission": recovery,
        "checkpoint_config_sha256": io.file_sha256(snapshot / "config.json"),
        "checkpoint_weights_sha256": io.file_sha256(snapshot / "pytorch_model.bin"),
    }


def verify_protected(expected: dict[str, Any], authority: dict[str, Any], claim) -> None:
    row_life.base.require_claim(claim, LOCK)
    if protected_snapshot(authority) != expected:
        raise RuntimeError("v2 evaluation protected snapshot changed")
    row_life.base.require_claim(claim, LOCK)


def validate_ledger(value: Any, authority_sha: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or value.get("schema") != LEDGER_SCHEMA \
            or value.get("recovery_admission") != row_life.recovery_admission():
        raise RuntimeError("v2 evaluation ledger recovery schema changed")
    core = dict(value); core.pop("recovery_admission"); core["schema"] = (
        "mlp2_trajectory_robust_r512_v1_physical_eval_ledger"
    )
    return science.validate_ledger(core, authority_sha)


def derive_result(ledgers: dict[str, torch.Tensor], runtime: float,
                  bundle: dict[str, Any]) -> dict[str, Any]:
    value = science.derive_result(ledgers, runtime, bundle)
    if value.pop("schema") != "mlp2_trajectory_robust_r512_v1_physical_eval_result":
        raise RuntimeError("v1 pure result schema changed")
    value["schema"] = RESULT_SCHEMA
    value["recovery_admission"] = row_life.recovery_admission()
    return value


def artifact_snapshot() -> dict[str, str | None]:
    return {path.name: io.file_sha256(path) if path.is_file() else None
            for path in (AUTHORITY, LEDGER, RESULT)}


def failure_terminal_guard(
    claim, expected_artifacts: dict[str, str | None],
    expected_authority: dict[str, Any] | None,
    expected_protected: dict[str, Any] | None,
) -> None:
    row_life.recovery_admission()
    if expected_authority is None:
        if expected_artifacts[AUTHORITY.name] is not None or expected_protected is not None:
            raise RuntimeError("v2 absent-authority failure state changed")
    else:
        value, digest = io.stable_json(AUTHORITY, expected_artifacts[AUTHORITY.name])
        if value != expected_authority or expected_protected is None \
                or protected_snapshot(value) != expected_protected \
                or digest != expected_artifacts[AUTHORITY.name]:
            raise RuntimeError("v2 failure authority/protected state changed")
    if artifact_snapshot() != expected_artifacts or RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("v2 evaluation failure aggregate or terminal changed")
    row_life.base.require_claim(claim, LOCK)


def validate_receipt(value: Any, authority_sha: str, ledger_sha: str,
                     result_sha: str) -> dict[str, Any]:
    expected = {
        "schema": RECEIPT_SCHEMA, "status": "result_complete_receipt_last",
        "authority_sha256": authority_sha, "ledger_sha256": ledger_sha,
        "result_sha256": result_sha, "evaluation_opened": True,
        "recovery_admission": row_life.recovery_admission(),
    }
    if value != expected:
        raise RuntimeError("v2 evaluation receipt semantics changed")
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if json.loads(encoded) != expected:
        raise RuntimeError("v2 evaluation canonical receipt replay changed")
    return expected


def publish_failure(claim, exc: BaseException, authority: dict[str, Any] | None,
                    protected: dict[str, Any] | None,
                    evaluation_opened: bool) -> dict[str, Any]:
    frozen = artifact_snapshot(); published = frozen[AUTHORITY.name] is not None
    value = {
        "schema": FAILURE_SCHEMA, "status": "terminal_failure_no_receipt",
        "error": repr(exc), "authority_exists": published,
        "evaluation_may_have_opened": evaluation_opened,
        "protected_snapshot": protected if published else None,
        "artifact_snapshot": frozen,
        "recovery_admission": row_life.recovery_admission(),
    }
    if not RECEIPT.exists() and not FAILURE.exists():
        def guard() -> None:
            failure_terminal_guard(
                claim, frozen, authority if published else None,
                protected if published else None,
            )
        io.atomic_json(FAILURE, value, pre_link_check=guard)
    return value


def main() -> None:
    if any(path.exists() for path in (AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("v2 evaluation namespace already exists")
    claim = row_life.base.acquire_claim(LOCK)
    authority = None; protected = None; evaluation_opened = False
    try:
        commit, sources = committed_sources()
        audit, audit_sha = row_life.validate_independent_audit(sources)
        recovery = row_life.recovery_admission()
        parents = science.validate_parents()
        program_integrity = science.expected_program_integrity()
        row_receipt, row_receipt_sha = io.stable_json(ROWS_RECEIPT)
        validate_row_receipt(row_receipt, sources)
        entry = row_receipt["entries"]["EVALUATION"]
        authority = {
            "schema": AUTHORITY_SCHEMA, "status": "frozen_before_evaluation_open",
            "source_commit": commit, "source_hashes": sources,
            "audit_sha256": audit_sha, "audit_reviewer": audit["reviewer"],
            "parents": parents, "program_integrity": program_integrity,
            "recovery_admission": recovery, "row_receipt_sha256": row_receipt_sha,
            "evaluation_rows_sha256": entry["file_sha256"],
            "arms": list(science.ARMS), "scored_slice": [64, 256],
            "outcome_access": False,
        }
        protected = protected_snapshot(authority)

        def authority_guard() -> None:
            row_life.base.require_claim(claim, LOCK); row_life.recovery_admission()
            verify_sources(commit, sources); row_life.validate_independent_audit(sources)
            io.stable_json(ROWS_RECEIPT, row_receipt_sha)
            if science.validate_parents() != parents \
                    or science.expected_program_integrity() != program_integrity \
                    or any(path.exists() for path in (AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE)):
                raise RuntimeError("v2 authority inputs or namespace changed")
            row_life.base.require_claim(claim, LOCK)

        io.atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = io.file_sha256(AUTHORITY)
        started = time.time(); evaluation_opened = True
        rows, rows_sha = io.stable_torch(Path(entry["path"]), entry["file_sha256"])
        if rows_sha != authority["evaluation_rows_sha256"] \
                or tuple(rows.shape) != (192, 257) \
                or refit.row_life.tensor_sha256(rows) != entry["tensor_sha256"]:
            raise RuntimeError("v2 evaluation rows changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        verify_protected(protected, authority, claim)
        c = load_program(science.base.C512_PATH)
        c_tensors = {key: c[key].to(device) for key in ("intercept", "left", "right")}
        old_bundle, _ = io.stable_torch(science.base.FULL_BUNDLE, science.base.FULL_BUNDLE_SHA)
        robust_bundle, _ = io.stable_torch(science.ROBUST_BUNDLE, science.ROBUST_BUNDLE_SHA)
        programs = {
            "FULL512": refit.build_from_state(old_bundle["programs"]["FULL512"], device).eval(),
            "CONTINUE512": refit.build_from_state(robust_bundle["programs"]["CONTINUE512"], device).eval(),
            "ROBUST512": refit.build_from_state(robust_bundle["programs"]["ROBUST512"], device).eval(),
        }
        if science.validate_program_integrity(old_bundle, robust_bundle) != program_integrity:
            raise RuntimeError("v2 loaded program integrity changed")
        ledgers = {arm: [] for arm in science.ARMS}
        calls = {arm: {
            "outer_calls": 0, "outer_returns": 0,
            "attention_sites": {str(site): 0 for site in range(18)},
            "native_mlp_sites": {str(site): 0 for site in range(18)},
            "candidate_c512": 0,
            "candidate_mlp2": {name: 0 for name in programs},
        } for arm in science.ARMS}
        with torch.inference_mode():
            for start in range(0, 192, 4):
                batch = rows[start:start + 4]
                tokens, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)
                logits = {}
                for arm in science.ARMS:
                    def attention(event: facade.AttentionEvent, arm=arm):
                        calls[arm]["attention_sites"][str(event.site)] += 1
                        return event.block.attn(event.state, event.first_value)

                    def mlp(event: facade.EarlyMLPEvent, arm=arm):
                        if event.site == 0 and arm in science.C512_ARMS:
                            calls[arm]["candidate_c512"] += 1
                            return science.base.c512_write(event, c_tensors)
                        program = science.PROGRAM_FOR_ARM.get(arm)
                        if event.site == 2 and program is not None:
                            calls[arm]["candidate_mlp2"][program] += 1
                            return programs[program](event.state)
                        calls[arm]["native_mlp_sites"][str(event.site)] += 1
                        return event.block.mlp(event.state)

                    calls[arm]["outer_calls"] += 1
                    logits[arm] = facade.forward_with_dispatch(model, tokens, attention, mlp)
                    calls[arm]["outer_returns"] += 1
                native = logits["NATIVE"]
                for arm in science.ARMS:
                    ledgers[arm].append(refit.reduce_document(native, logits[arm], targets))
        packed = {arm: torch.cat(parts) for arm, parts in ledgers.items()}
        if calls != science.expected_call_census():
            raise RuntimeError("v2 evaluation call census changed")
        ledger = {
            "schema": LEDGER_SCHEMA, "arms": packed, "calls": calls,
            "authority_sha256": authority_sha, "checkpoint": checkpoint.__dict__,
            "program_integrity": program_integrity, "recovery_admission": recovery,
        }

        def ledger_guard() -> None:
            row_life.recovery_admission()
            verify_protected(protected, authority, claim)
            if any(path.exists() for path in (LEDGER, RESULT, RECEIPT, FAILURE)):
                raise RuntimeError("v2 terminal raced ledger")
            row_life.base.require_claim(claim, LOCK)

        io.atomic_torch(LEDGER, ledger, pre_link_check=ledger_guard)
        reloaded, ledger_sha = io.stable_torch(LEDGER)
        replay_arms = validate_ledger(reloaded, authority_sha)
        runtime = time.time() - started
        result = derive_result(replay_arms, runtime, robust_bundle)
        result["parents"] = {"authority": authority_sha, "ledger": ledger_sha}
        result["program_integrity"] = program_integrity

        def result_guard() -> None:
            row_life.recovery_admission()
            verify_protected(protected, authority, claim)
            io.stable_torch(LEDGER, ledger_sha)
            if any(path.exists() for path in (RESULT, RECEIPT, FAILURE)):
                raise RuntimeError("v2 terminal raced result")
            row_life.base.require_claim(claim, LOCK)

        io.atomic_json(RESULT, result, pre_link_check=result_guard)
        reloaded_result, result_sha = io.stable_json(RESULT)
        expected = derive_result(replay_arms, runtime, robust_bundle)
        if reloaded_result != result or expected != {
            key: value for key, value in result.items()
            if key not in ("parents", "program_integrity")
        } or reloaded_result["program_integrity"] != program_integrity:
            raise RuntimeError("v2 result semantic replay changed")
        receipt = validate_receipt({
            "schema": RECEIPT_SCHEMA, "status": "result_complete_receipt_last",
            "authority_sha256": authority_sha, "ledger_sha256": ledger_sha,
            "result_sha256": result_sha, "evaluation_opened": True,
            "recovery_admission": recovery,
        }, authority_sha, ledger_sha, result_sha)
        rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)

        def receipt_guard() -> None:
            row_life.recovery_admission()
            verify_protected(protected, authority, claim)
            io.stable_json(AUTHORITY, authority_sha); io.stable_torch(LEDGER, ledger_sha)
            io.stable_json(RESULT, result_sha)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("v2 terminal raced receipt")
            row_life.base.require_claim(claim, LOCK)

        print(rendered)
        io.atomic_json(RECEIPT, receipt, pre_link_check=receipt_guard)
    except BaseException as exc:
        publish_failure(claim, exc, authority, protected, evaluation_opened)
        raise
    finally:
        row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
