#!/usr/bin/env python3
"""Source-closed one-shot equality-tensor FINAL/OOD v2 transaction."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

import bilin18_observed_model_facade as facade
import circuit_campaign_runtime as runtime
import circuit_campaign_statistics as statistics
import induction_equality_tensor_discovery as discovery
import prepare_block3_native_down_behavioral_port_v1_rows as atomic
import prepare_induction_equality_tensor_final_ood_v2_rows as rows_v2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "INDUCTION_EQUALITY_TENSOR_FINAL_OOD_V2_PREREGISTRATION.md"
AUDIT = HERE / "induction_equality_tensor_final_ood_v2_independent_audit.json"
AUTHORITY = HERE / "induction_equality_tensor_final_ood_v2_authority.json"
LEDGER = HERE / "induction_equality_tensor_final_ood_v2_ledger.json"
RESULT = HERE / "induction_equality_tensor_final_ood_v2_result.json"
MANIFEST = HERE / "induction_equality_tensor_final_ood_v2_manifest.json"
RECEIPT = HERE / "induction_equality_tensor_final_ood_v2_receipt.json"
FAILURE = HERE / "induction_equality_tensor_final_ood_v2_failure.json"
LOCK = Path("/workspace/runs/.induction_equality_tensor_final_ood_v2.lock")
ROLES = ("final_natural", "ood_code")
ARMS = discovery.ARMS
SELECTED = discovery.SELECTED
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = "induction-equality-final-ood-v2:bootstrap:0"
SELECT_TARGET = 0.5122487687425222
SELECT_EXTRACTION = 0.9739717690344445
OUTPUTS = (AUTHORITY, LEDGER, RESULT, MANIFEST, RECEIPT, FAILURE)
DIRECT_SOURCES = (
    Path(__file__).resolve(), HERE / "test_induction_equality_tensor_final_ood_v2.py",
    PREREG, HERE / "INDUCTION_EQUALITY_TENSOR_FINAL_OOD_PREREGISTRATION.md",
    HERE / "induction_equality_tensor_final_ood_independent_audit.json",
    HERE / "induction_equality_tensor_discovery.json", HERE / "induction_equality_tensor_discovery.py",
    HERE / "test_induction_equality_tensor_discovery.py", HERE / "circuit_induction_tensor.py",
    HERE / "test_circuit_induction_tensor.py", HERE / "bilin18_observed_model_facade.py",
    HERE / "test_bilin18_observed_model_facade.py", HERE / "circuit_campaign_runtime.py",
    HERE / "test_circuit_campaign_runtime.py", HERE / "circuit_campaign_statistics.py",
    HERE / "test_circuit_campaign_statistics.py", rows_v2.Path(rows_v2.__file__).resolve(),
    HERE / "test_prepare_induction_equality_tensor_final_ood_v2_rows.py",
    ROOT / "jacclust/__init__.py", ROOT / "jacclust/tt_model.py",
)
SOURCE_PATHS = tuple(dict.fromkeys((*DIRECT_SOURCES, *rows_v2.SOURCE_PATHS)))


def file_sha256(path: Path) -> str:
    return atomic.file_sha256(path)


def source_closure(commit: str) -> dict[str, str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    result = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"v2 execution source drift: {relative}")
        result[relative] = digest
    return result


def stable_json(path: Path) -> dict[str, Any]:
    before = file_sha256(path); raw = path.read_bytes(); after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"unstable JSON read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def _row_receipt() -> dict[str, Any]:
    value = stable_json(rows_v2.RECEIPT)
    if value.get("schema") != "induction_equality_tensor_final_ood_v2_rows_receipt" \
            or value.get("status") != "frozen_before_any_v2_model_forward" \
            or value.get("old_v1_role_tensors_deserialized") is not False \
            or value.get("outcome_access") is not False \
            or set(value.get("entries", {})) != set(rows_v2.ROLES):
        raise RuntimeError("fresh v2 row receipt is not authoritative")
    return value


def protected_snapshot() -> dict[str, str | None]:
    paths = [*SOURCE_PATHS, AUDIT, rows_v2.RECEIPT, discovery.OUTPUT,
             facade.DEFAULT_SNAPSHOT / "config.json", facade.DEFAULT_SNAPSHOT / "pytorch_model.bin"]
    if rows_v2.RECEIPT.is_file():
        receipt = _row_receipt()
        paths.extend(Path(receipt["entries"][role]["path"]) for role in rows_v2.ROLES)
    return {str(path.resolve()): file_sha256(path) if path.is_file() else None for path in paths}


def validate_audit(commit: str, sources: Mapping[str, str]) -> dict[str, Any]:
    value = stable_json(AUDIT)
    if set(value) != {"schema", "status", "outcome_access", "audited_source_commit", "audited_source_hashes", "tests_passed", "reviewer"} \
            or value.get("schema") != "induction_equality_tensor_final_ood_v2_independent_audit" \
            or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_commit") != commit \
            or value.get("audited_source_hashes") != dict(sources) \
            or type(value.get("tests_passed")) is not int or value["tests_passed"] < 1 \
            or not isinstance(value.get("reviewer"), str) or not value["reviewer"]:
        raise RuntimeError("v2 execution audit is not an exact GO")
    return value


def _clean_before_authority() -> None:
    if any(path.exists() for path in OUTPUTS):
        raise RuntimeError("v2 execution namespace is not pristine")


def freeze_authority() -> dict[str, Any]:
    _clean_before_authority()
    claim = atomic.acquire_claim(LOCK)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        sources = source_closure(commit); audit = validate_audit(commit, sources)
        row_receipt = _row_receipt(); snapshot = protected_snapshot()
        authority = {
            "schema": "induction_equality_tensor_final_ood_v2_authority",
            "status": "frozen_before_fresh_rows_or_model_load", "outcome_access": False,
            "source_commit": commit, "source_hashes": sources, "audit": audit,
            "row_receipt_sha256": file_sha256(rows_v2.RECEIPT),
            "role_file_sha256s": {role: row_receipt["entries"][role]["file_sha256"] for role in ROLES},
            "discovery_sha256": rows_v2.DISCOVERY_SHA256,
            "checkpoint_weights_sha256": facade.WEIGHTS_SHA256,
            "protected_snapshot": snapshot,
            "outputs": {path.stem: str(path.resolve()) for path in OUTPUTS},
        }
        def guard():
            source_closure(commit); validate_audit(commit, sources)
            if protected_snapshot() != snapshot or any(path.exists() for path in OUTPUTS):
                raise RuntimeError("authority inputs or namespace changed")
            atomic.require_claim(claim, LOCK)
        atomic.write_json_create_only(authority, AUTHORITY, pre_link_check=guard)
        return authority
    finally:
        atomic.release_claim(claim, LOCK)


def validate_authority() -> dict[str, Any]:
    authority = stable_json(AUTHORITY)
    expected_keys = {"schema", "status", "outcome_access", "source_commit", "source_hashes", "audit", "row_receipt_sha256", "role_file_sha256s", "discovery_sha256", "checkpoint_weights_sha256", "protected_snapshot", "outputs"}
    if set(authority) != expected_keys or authority["schema"] != "induction_equality_tensor_final_ood_v2_authority" \
            or authority["status"] != "frozen_before_fresh_rows_or_model_load" \
            or authority["outcome_access"] is not False \
            or source_closure(authority["source_commit"]) != authority["source_hashes"] \
            or validate_audit(authority["source_commit"], authority["source_hashes"]) != authority["audit"] \
            or protected_snapshot() != authority["protected_snapshot"] \
            or authority["outputs"] != {path.stem: str(path.resolve()) for path in OUTPUTS}:
        raise RuntimeError("v2 authority semantic replay failed")
    return authority


def _load_role(role: str, authority: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _row_receipt(); entry = receipt["entries"][role]; path = Path(entry["path"])
    before = file_sha256(path)
    if before != entry["file_sha256"] or before != authority["role_file_sha256s"][role]:
        raise RuntimeError("v2 role binding changed before load")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != before or value.get("role") != role or value.get("schema") != "induction_equality_tensor_final_ood_v2_role" \
            or value["rows"].dtype != torch.long or tuple(value["rows"].shape) != (192, 257) \
            or len(value["records"]) != 192:
        raise RuntimeError("v2 role semantic load failed")
    return value


def _plans() -> runtime.CircuitPlan:
    native = runtime.ArmPlan.build("native", runtime.ArmKind.NATIVE)
    candidates = tuple(runtime.ArmPlan.build(
        arm, runtime.ArmKind.CANDIDATE,
        attention_replacements={site: f"{arm}:L{site}" for site in SELECTED},
    ) for arm in ARMS[1:])
    return runtime.CircuitPlan("induction-equality-tensor-final-ood-v2", 18, (native, *candidates))


def _merge_ledgers(target, batch):
    overlap = set(target) & set(batch)
    if overlap:
        raise RuntimeError("document ledger repeats IDs")
    target.update(batch)


@torch.no_grad()
def score_role(model: torch.nn.Module, role: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]; records = payload["records"]; cells = payload["copy_cells"]
    document_ids = tuple(str(record.get("document_id") or f"code:{record['path']}") for record in records)
    if len(set(document_ids)) != 192:
        raise RuntimeError("role is not one row per source document")
    masks = {name: cells[name] for name in ("positive", "matched_negative", "off_target")}
    masks["all"] = torch.zeros_like(masks["positive"]); masks["all"][:, 64:256] = True
    support = {name: {"tokens": int(mask.sum()), "documents": int((mask.any(1)).sum())} for name, mask in masks.items()}
    if any(value["tokens"] < 200 or value["documents"] < 30 for value in support.values()):
        raise RuntimeError("role support is below the frozen minimum")
    plan = _plans(); ledger = {}; site_totals = {arm: [[0, 0, 0, 0] for _ in range(18)] for arm in ARMS}
    outer = {arm: {"forwards": 0, "returns": 0, "documents": 0} for arm in ARMS}
    replay_max = 0.0
    device = next(model.parameters()).device
    for start in range(0, 192, 4):
        batch_rows = rows[start:start + 4]; tokens = batch_rows[:, :-1].to(device)
        logits = {}
        for arm in ARMS:
            callbacks = {}
            if arm != "native":
                for site, heads in SELECTED.items():
                    attention = model.transformer.h[site].attn
                    def callback(event, arm=arm, attention=attention, heads=heads, tokens=tokens):
                        writes, _ = discovery.replay_attention_site(event.state, event.first_value, attention, heads, tokens)
                        return writes[arm], event.first_value
                    callbacks[f"{arm}:L{site}"] = callback
            owner = runtime.CircuitForwardOwner(plan=plan, arm=arm, attention_replacements=callbacks)
            logits[arm] = owner.run(model, tokens, require_production=True).cpu()
            closure = owner.closure
            outer[arm]["forwards"] += closure.completed_outer_forwards; outer[arm]["returns"] += closure.outer_returns; outer[arm]["documents"] += closure.document_count
            for site, value in enumerate(closure.sites):
                counts = (value.native_attention_calls, value.replacement_attention_calls, value.native_mlp_calls, value.replacement_mlp_calls)
                site_totals[arm][site] = [a + b for a, b in zip(site_totals[arm][site], counts)]
        replay_max = max(replay_max, float((logits["full_replay"] - logits["native"]).abs().max()))
        batch_ledger = statistics.reduce_document_batch(
            logits, batch_rows, {name: mask[start:start + 4] for name, mask in masks.items()},
            document_ids[start:start + 4], kl_pairs=tuple(("native", arm) for arm in ARMS[1:]),
        )
        _merge_ledgers(ledger, batch_ledger)
        del logits
    expected = 48
    for arm in ARMS:
        if outer[arm] != {"forwards": expected, "returns": expected, "documents": 192}:
            raise RuntimeError("outer call census changed")
        for site, counts in enumerate(site_totals[arm]):
            replaced = arm != "native" and site in SELECTED
            expected_counts = [0, expected, expected, 0] if replaced else [expected, 0, expected, 0]
            if counts != expected_counts:
                raise RuntimeError("physical per-site call census changed")
    return {"documents": document_ids, "support": support, "ledger": ledger, "outer": outer, "sites": site_totals, "replay_max_abs": replay_max}


def _encode_cell(value: statistics.DocumentCellSums) -> dict[str, Any]:
    return asdict(value)


def _decode_cell(value: Mapping[str, Any]) -> statistics.DocumentCellSums:
    return statistics.DocumentCellSums(
        n=value["n"], support_sha256=value["support_sha256"],
        arms=tuple(statistics.ArmCellSums(**item) for item in value["arms"]),
        directed_kls=tuple(statistics.DirectedKLSums(**item) for item in value["directed_kls"]),
    )


def _specs(role: str):
    return (
        statistics.CoordinateSpec(f"{role}:target", statistics.CoordinateKind.TARGET_DAMAGE, role, "positive", candidate_arm="remove_equality"),
        statistics.CoordinateSpec(f"{role}:specificity", statistics.CoordinateKind.SPECIFICITY, role, "positive", candidate_arm="remove_equality", comparison_cell="matched_negative"),
        statistics.CoordinateSpec(f"{role}:off", statistics.CoordinateKind.TARGET_DAMAGE, role, "off_target", candidate_arm="remove_equality"),
        statistics.CoordinateSpec(f"{role}:extraction", statistics.CoordinateKind.EXTRACTION_RECOVERY, role, "positive", candidate_arm="extract_equality", stake_arm="heads_deleted"),
        statistics.CoordinateSpec(f"{role}:deranged", statistics.CoordinateKind.EXTRACTION_RECOVERY, role, "positive", candidate_arm="deranged_equality", stake_arm="heads_deleted"),
    )


def analyze(roles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    reports = {}
    for role in ROLES:
        ledger = roles[role]["ledger"]; specs = _specs(role)
        boot = statistics.simultaneous_document_bootstrap({role: ledger}, specs, repetitions=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        point, low, high = (boot.point_estimates.tolist(), boot.simultaneous_lower_bounds.tolist(), boot.simultaneous_upper_bounds.tolist())
        target, specificity, off, extraction, deranged = point
        if role == "final_natural":
            gates = {"target": low[0] > 0, "specificity": low[1] > 0, "extraction": extraction >= .80 and low[3] >= .60, "collateral": high[2] <= .01 and off <= .10 * target, "deranged": high[4] < .5 * low[3]}
        else:
            gates = {"target": low[0] > 0 and target >= .5 * SELECT_TARGET, "specificity": low[1] > 0, "extraction": extraction >= .60 and low[3] >= .40 and extraction >= .5 * SELECT_EXTRACTION, "collateral": high[2] <= .02 and off <= .20 * target, "deranged": high[4] < .5 * low[3]}
        gates.update({"support": all(x["tokens"] >= 200 and x["documents"] >= 30 for x in roles[role]["support"].values()), "replay": roles[role]["replay_max_abs"] <= 1e-4, "calls": True})
        reports[role] = {"coordinate_names": list(boot.coordinate_names), "point": point, "simultaneous_low": low, "simultaneous_high": high, "gates": gates, "passed": all(gates.values())}
    return {"roles": reports, "passed_both_roles": all(value["passed"] for value in reports.values()), "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}}


def semantic_validate(ledger_payload: Mapping[str, Any], result: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    decoded = {}
    for role in ROLES:
        decoded[role] = {"support": ledger_payload["roles"][role]["support"], "outer": ledger_payload["roles"][role]["outer"], "sites": ledger_payload["roles"][role]["sites"], "replay_max_abs": ledger_payload["roles"][role]["replay_max_abs"], "ledger": {doc: {cell: _decode_cell(value) for cell, value in cells.items()} for doc, cells in ledger_payload["roles"][role]["ledger"].items()}}
    expected = analyze(decoded)
    if result != {"schema": "induction_equality_tensor_final_ood_v2_result", "authority_sha256": file_sha256(AUTHORITY), **expected}:
        raise RuntimeError("v2 result semantic replay failed")


def execute() -> dict[str, Any]:
    authority = validate_authority(); claim = atomic.acquire_claim(LOCK)
    protected = authority["protected_snapshot"]
    try:
        if any(path.exists() for path in (LEDGER, RESULT, MANIFEST, RECEIPT, FAILURE)):
            raise RuntimeError("v2 outcome namespace is spent")
        atomic.require_claim(claim, LOCK)
        loaded_roles = {role: _load_role(role, authority) for role in ROLES}
        if protected_snapshot() != protected:
            raise RuntimeError("protected inputs changed before model load")
        weights = facade.DEFAULT_SNAPSHOT / "pytorch_model.bin"; before = file_sha256(weights)
        model, _ = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=False)
        if file_sha256(weights) != before or before != facade.WEIGHTS_SHA256:
            raise RuntimeError("checkpoint changed across load")
        measured = {role: score_role(model, role, loaded_roles[role]) for role in ROLES}
        ledger_payload = {"schema": "induction_equality_tensor_final_ood_v2_ledger", "authority_sha256": file_sha256(AUTHORITY), "raw_payloads_published": False, "roles": {role: {"documents_sha256": hashlib.sha256("\0".join(measured[role]["documents"]).encode()).hexdigest(), "support": measured[role]["support"], "outer": measured[role]["outer"], "sites": measured[role]["sites"], "replay_max_abs": measured[role]["replay_max_abs"], "ledger": {doc: {cell: _encode_cell(value) for cell, value in cells.items()} for doc, cells in measured[role]["ledger"].items()}} for role in ROLES}}
        atomic.write_json_create_only(ledger_payload, LEDGER)
        ledger_reload = stable_json(LEDGER)
        provisional = {"schema": "induction_equality_tensor_final_ood_v2_result", "authority_sha256": file_sha256(AUTHORITY), **analyze(measured)}
        atomic.write_json_create_only(provisional, RESULT); result_reload = stable_json(RESULT)
        semantic_validate(ledger_reload, result_reload, authority)
        manifest = {"schema": "induction_equality_tensor_final_ood_v2_manifest", "authority_sha256": file_sha256(AUTHORITY), "ledger_sha256": file_sha256(LEDGER), "result_sha256": file_sha256(RESULT), "protected_snapshot": protected}
        atomic.write_json_create_only(manifest, MANIFEST)
        receipt = {"schema": "induction_equality_tensor_final_ood_v2_receipt", "status": "complete_receipt_last", "authority_sha256": file_sha256(AUTHORITY), "ledger_sha256": file_sha256(LEDGER), "result_sha256": file_sha256(RESULT), "manifest_sha256": file_sha256(MANIFEST), "passed_both_roles": result_reload["passed_both_roles"]}
        def guard():
            validate_authority(); semantic_validate(stable_json(LEDGER), stable_json(RESULT), authority)
            if stable_json(MANIFEST) != manifest or protected_snapshot() != protected or FAILURE.exists() or RECEIPT.exists():
                raise RuntimeError("v2 receipt terminal replay failed")
            atomic.require_claim(claim, LOCK)
        atomic.write_json_create_only(receipt, RECEIPT, pre_link_check=guard)
        return receipt
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            partial = {str(path): file_sha256(path) if path.is_file() else None for path in (LEDGER, RESULT, MANIFEST)}
            failure = {"schema": "induction_equality_tensor_final_ood_v2_failure", "status": "terminal_failure_no_receipt", "authority_sha256": file_sha256(AUTHORITY), "error_type": type(error).__name__, "error": str(error), "partial_artifacts": partial, "protected_snapshot": protected_snapshot()}
            def fail_guard():
                if RECEIPT.exists() or FAILURE.exists() or partial != {str(path): file_sha256(path) if path.is_file() else None for path in (LEDGER, RESULT, MANIFEST)}:
                    raise RuntimeError("v2 failure terminal changed")
                atomic.require_claim(claim, LOCK)
            atomic.write_json_create_only(failure, FAILURE, pre_link_check=fail_guard)
        raise
    finally:
        atomic.release_claim(claim, LOCK)


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
