#!/usr/bin/env python3
"""Receipt-last, reporting-only recovery for the spent Family-F v1 program."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch


ROOT = Path("/workspace/tensor_language")
sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
import block3_consequence_family_f_call_ledger as v1_calls
import collect_block3_native_gate_fit_v1 as collector
import fit_block3_consequence_family_f_v1 as v1


HERE = ROOT / "basis_aligned" / "polynomial_causal"
PREREG = HERE / "BLOCK3_CONSEQUENCE_FAMILY_F_V2_RECOVERY.md"
RUNNER = HERE / "recover_block3_consequence_family_f_v2.py"
TEST = HERE / "test_recover_block3_consequence_family_f_v2.py"
AUTHORITY = HERE / "block3_consequence_family_f_v2_recovery_authority.json"
RESULTS = HERE / "block3_consequence_family_f_v2_recovery_results.json"
RECEIPT = HERE / "block3_consequence_family_f_v2_recovery_receipt.json"
FAILURE = HERE / "block3_consequence_family_f_v2_recovery_failure.json"
LOCK = Path("/workspace/runs/.block3_consequence_family_f_v2_recovery.lock")

V1_PINS = {
    str(v1.AUTHORITY.relative_to(ROOT)):
        "70a4f751d6f79438263eb44f235b24b14334527aca4c169afa77dca6fc701e7d",
    str(v1.PROGRAMS.relative_to(ROOT)):
        "d4af5bfbae03f8df9be8127e2e06c6f1a66b189be180ce72e5c74b6c7ac7a038",
    str(v1.FAILURE.relative_to(ROOT)):
        "1bb45f2645576fadef564562ef37f98abfb64afb75af8396b882fe63b783f79b",
}
SOURCE_PATHS = tuple(dict.fromkeys((*v1.SOURCE_PATHS,
    str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT)),
    str(TEST.relative_to(ROOT)),
)))
MAX_WALL_SECONDS = 15 * 60
MAX_ALLOCATED_CUDA_BYTES = 30 * 1024 ** 3
N_BATCHES = v1.life.ROW_COUNT // v1.LOGICAL_BATCH


def file_sha256(path: Path) -> str:
    return collector.file_sha256(path)


def logical_sha256(value: Any) -> str:
    return v1.logical_sha256(value)


def output_namespace() -> tuple[Path, ...]:
    return AUTHORITY, RESULTS, RECEIPT, FAILURE, LOCK


def require_pristine_namespace() -> None:
    spent = [str(path) for path in output_namespace() if path.exists()]
    if spent:
        raise RuntimeError(f"family-F v2 recovery namespace is spent: {spent}")


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"family-F v2 source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"live family-F v2 source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source_closure(value: Mapping[str, Any]) -> None:
    if set(value) != {"commit", "paths", "sha256"} or set(
        value["paths"]
    ) != set(SOURCE_PATHS) or logical_sha256({
        "commit": value["commit"], "paths": value["paths"],
    }) != value["sha256"]:
        raise RuntimeError("family-F v2 source closure changed")
    for relative, digest in value["paths"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"family-F v2 live source drift: {relative}")


def v1_file_binding() -> dict[str, Any]:
    if v1.RESULTS.exists() or v1.RECEIPT.exists():
        raise RuntimeError("spent Family-F v1 unexpectedly has a result or receipt")
    observed = {
        relative: file_sha256(ROOT / relative) for relative in V1_PINS
    }
    if observed != V1_PINS:
        raise RuntimeError("spent Family-F v1 artifact bytes changed")
    body = {
        "file_sha256s": observed,
        "v1_results_absent": True,
        "v1_receipt_absent": True,
        "recovery_kind": "fixed_program_reporting_only_no_refit",
    }
    return {**body, "sha256": logical_sha256(body)}


def authority(
    source: Mapping[str, Any], v1_binding: Mapping[str, Any],
    prior: Mapping[str, Any], rows: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt,
) -> dict[str, Any]:
    protocol = {
        "fit_rows": v1.life.ROW_COUNT,
        "logical_batch": v1.LOGICAL_BATCH,
        "prefixes": N_BATCHES,
        "teacher_suffixes": N_BATCHES,
        "student_suffixes_per_arm": N_BATCHES,
        "student_arms": list(v1.call_contract.REPORT_STUDENT_ARMS),
        "optimizer_steps": 0,
        "program_refits": 0,
        "v1_optimizer_traces": "unavailable_from_spent_v1_nonpromotive",
        "polarization_currency": "independent_device_local_relative_gate",
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
    }
    body = {
        "schema": "block3_consequence_family_f_v2_recovery_authority",
        "status": "frozen_before_v1_outcome_parent_row_or_checkpoint_tensor_load",
        "source_closure": dict(source),
        "v1_binding": dict(v1_binding),
        "prior_artifact_binding": dict(prior),
        "row_binding": dict(rows),
        "checkpoint": asdict(checkpoint),
        "protocol": protocol,
        "output_paths": {
            "results": str(RESULTS), "receipt": str(RECEIPT),
            "failure": str(FAILURE),
        },
    }
    return {**body, "authority_sha256": logical_sha256(body)}


def require_resource_ceiling(started: float) -> tuple[float, int]:
    elapsed = time.time() - started
    allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    if elapsed > MAX_WALL_SECONDS or allocated > MAX_ALLOCATED_CUDA_BYTES:
        raise RuntimeError(
            f"family-F v2 resource ceiling exceeded: seconds={elapsed}, bytes={allocated}"
        )
    return elapsed, allocated


def require_published_authority(frozen: Mapping[str, Any]) -> None:
    """Treat the exact published v2 authority as the tensor-load capability."""
    if not AUTHORITY.is_file():
        raise RuntimeError("family-F v2 authority capability is absent")
    published = json.loads(AUTHORITY.read_text())
    body = {key: value for key, value in published.items() if key != "authority_sha256"}
    if published != dict(frozen) or published.get("authority_sha256") != logical_sha256(body):
        raise RuntimeError("family-F v2 authority capability changed")


@dataclass
class RecoveryCalls:
    started: float | None = None
    prefixes: int = 0
    teacher_suffixes: int = 0
    students: Counter[str] = field(default_factory=Counter)

    def _check_resources(self) -> None:
        if self.started is not None:
            require_resource_ceiling(self.started)

    def record_prefix(self, phase: str, arm: str, donor: bool = False) -> None:
        self._check_resources()
        if phase != "postfit_report" or arm != v1.call_contract.REPORT_SHARED_ARM or donor:
            raise RuntimeError("family-F v2 unexpected prefix call")
        self.prefixes += 1

    def record_teacher_suffix(self, phase: str, arm: str) -> None:
        self._check_resources()
        if phase != "postfit_report" or arm != v1.call_contract.REPORT_SHARED_ARM:
            raise RuntimeError("family-F v2 unexpected teacher suffix")
        self.teacher_suffixes += 1

    def record_student_suffix(self, phase: str, arm: str) -> None:
        self._check_resources()
        if phase != "postfit_report" or arm not in v1.call_contract.REPORT_STUDENT_ARMS:
            raise RuntimeError("family-F v2 unexpected student suffix")
        self.students[arm] += 1

    def validate_exact(self) -> dict[str, Any]:
        expected_students = {
            arm: N_BATCHES for arm in v1.call_contract.REPORT_STUDENT_ARMS
        }
        if self.prefixes != N_BATCHES or self.teacher_suffixes != N_BATCHES or dict(
            self.students
        ) != expected_students:
            raise RuntimeError("family-F v2 reporting call census changed")
        return {
            "schema": "block3_consequence_family_f_v2_recovery_calls",
            "prefixes": self.prefixes,
            "teacher_suffixes": self.teacher_suffixes,
            "student_suffixes": expected_students,
            "optimizer_steps": 0,
            "program_refits": 0,
        }


def load_v1_after_authority(
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require_published_authority(frozen)
    before = {relative: file_sha256(ROOT / relative) for relative in V1_PINS}
    if before != V1_PINS:
        raise RuntimeError("family-F v1 changed before recovery load")
    old_authority = json.loads(v1.AUTHORITY.read_text())
    old_failure = json.loads(v1.FAILURE.read_text())
    old_program = torch.load(v1.PROGRAMS, map_location="cpu", weights_only=True)
    after = {relative: file_sha256(ROOT / relative) for relative in V1_PINS}
    if after != before:
        raise RuntimeError("family-F v1 changed during recovery load")
    if old_authority.get("authority_sha256") != old_program.get(
        "authority_sha256"
    ) or old_failure.get("status") != "terminal_failure_no_receipt" or old_failure.get(
        "results_exists"
    ) is not False or old_failure.get("receipt_exists") is not False:
        raise RuntimeError("family-F v1 authority/program/failure join changed")
    v1_calls.FamilyFCallLedger.replay_complete_receipt(old_failure["partial_call_ledger"])
    return old_authority, old_failure, old_program


def load_rows_after_authority(
    frozen: Mapping[str, Any], rows_binding: Mapping[str, Any],
) -> torch.Tensor:
    require_published_authority(frozen)
    before = file_sha256(v1.life.ROWS)
    raw = torch.load(v1.life.ROWS, map_location="cpu", weights_only=True)
    rows = raw["rows"] if isinstance(raw, dict) and set(raw) == {"rows"} else raw
    if before != v1.life.ROWS_FILE_SHA256 or file_sha256(v1.life.ROWS) != before or (
        rows_binding.get("row_file_sha256") != before
    ) or collector.tensor_sha256(rows) != v1.life.ROWS_RAW_SHA256:
        raise RuntimeError("family-F v2 row tensor changed")
    return rows.contiguous()


def load_parents_after_authority(
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_published_authority(frozen)
    expected = frozen["prior_artifact_binding"]["file_sha256s"]
    paths = (collector.PAYLOAD, v1.life.PRIOR_PATHS[4])
    before = {path: file_sha256(path) for path in paths}
    if any(expected[str(path.relative_to(ROOT))] != digest for path, digest in before.items()):
        raise RuntimeError("family-F v2 parent binding changed")
    parent = torch.load(paths[0], map_location="cpu", weights_only=True)
    family_a = torch.load(paths[1], map_location="cpu", weights_only=True)
    if any(file_sha256(path) != digest for path, digest in before.items()):
        raise RuntimeError("family-F v2 parent changed during load")
    return parent, family_a


def instrument_model_calls(model: torch.nn.Module) -> tuple[dict[str, Counter[int]], list[Any]]:
    counts = {"attention": Counter(), "mlp": Counter()}
    handles = []
    for site, block in enumerate(model.transformer.h):
        handles.append(block.attn.register_forward_hook(
            lambda _m, _a, _o, site=site: counts["attention"].update([site])
        ))
        handles.append(block.mlp.register_forward_hook(
            lambda _m, _a, _o, site=site: counts["mlp"].update([site])
        ))
    return counts, handles


def validate_model_calls(counts: Mapping[str, Counter[int]]) -> dict[str, Any]:
    expected = {site: N_BATCHES if site <= v1.LAYER else 19 * N_BATCHES for site in range(18)}
    if dict(counts["attention"]) != expected or dict(counts["mlp"]) != expected:
        raise RuntimeError("family-F v2 model call census changed")
    return {kind: {str(site): count for site, count in expected.items()} for kind in counts}


def execute_recovery(
    frozen: Mapping[str, Any], checkpoint: facade.CheckpointReceipt, started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, artifact = load_v1_after_authority(frozen)
    rows = load_rows_after_authority(frozen, frozen["row_binding"])
    parent, family_a = load_parents_after_authority(frozen)
    model, loaded = facade.load_bilin18(
        device=v1.DEVICE, dtype=torch.float32, verify_weights_sha256=True,
    )
    if loaded != checkpoint:
        raise RuntimeError("family-F v2 checkpoint load changed")
    torch.cuda.reset_peak_memory_stats()
    model_before = v1.model_state_sha256(model)
    block = model.transformer.h[v1.LAYER]
    left = block.mlp.Left.weight.detach()
    right = block.mlp.Right.weight.detach()
    native_down = block.mlp.Down.weight.detach()
    native_bias = block.mlp.Down_bias.detach()
    balanced_left, balanced_right, _ = collector.balance_product_gauge(left, right)
    reconstructed, reconstructed_supports = v1.reconstruct_programs_from_sealed_parents(
        value=artifact, left=left, right=right, native_down=native_down,
        native_bias=native_bias, parent_payload=parent, raw_a_programs=family_a,
    )
    if set(reconstructed_supports) != set(artifact["supports"]) or any(
        not torch.equal(reconstructed_supports[name].cpu(), artifact["supports"][name])
        for name in reconstructed_supports
    ):
        raise RuntimeError("family-F v2 support reconstruction changed")
    family_a_supports = {
        budget: family_a["programs"][f"activation_selected_k{budget}"]["indices"].long()
        for budget in v1.BUDGETS
    }
    v1.semantic_validate_program_artifact(
        artifact, expected_authority_sha256=artifact["authority_sha256"],
        prefilter=parent["prefilter_indices"].long(),
        reconstructed_programs=reconstructed, family_a_supports=family_a_supports,
    )
    programs = {
        name: v1._materialize_program(payload, device=v1.DEVICE)
        for name, payload in artifact["programs"].items()
    }
    calls = RecoveryCalls(started=started)
    counts, handles = instrument_model_calls(model)
    try:
        row_to_document = torch.tensor(
            frozen["row_binding"]["row_to_document"], dtype=torch.long,
        )
        report = v1.report_fit_arms(
            model=model, rows=rows,
            row_weights=v1.core.source_document_row_weights(row_to_document),
            calls=calls, balanced_left=balanced_left, balanced_right=balanced_right,
            native_down=native_down, native_bias=native_bias,
            prefilter_indices=parent["prefilter_indices"].long().to(v1.DEVICE),
            teacher_scores=artifact["scores"]["teacher"].to(v1.DEVICE),
            programs=programs, started=started,
        )
    finally:
        for handle in handles:
            handle.remove()
    call_receipt = calls.validate_exact()
    model_calls = validate_model_calls(counts)
    model_after = v1.model_state_sha256(model)
    if model_after != model_before or collector.tensor_sha256(rows) != v1.life.ROWS_RAW_SHA256:
        raise RuntimeError("family-F v2 model or rows changed")
    cpu_programs = {
        name: v1._materialize_program(payload, device="cpu")
        for name, payload in artifact["programs"].items()
    }
    polarization = {
        name: {
            "cpu": v1.deployed_polarization_replay(cpu_programs[name]),
            "cuda": v1.deployed_polarization_replay(programs[name]),
        }
        for name in programs
    }
    elapsed, allocated = require_resource_ceiling(started)
    result = {
        "schema": "block3_consequence_family_f_v2_recovery_results",
        "status": "fit_reporting_recovered_no_validation_or_final_opened",
        "authority_sha256": frozen["authority_sha256"],
        "v1_programs_file_sha256": V1_PINS[str(v1.PROGRAMS.relative_to(ROOT))],
        "postfit_report": report,
        "program_prices": {name: v1.core.program_price(program) for name, program in cpu_programs.items()},
        "polarization_replay_by_device": polarization,
        "report_call_ledger": call_receipt,
        "model_call_ledger": model_calls,
        "v1_fit_call_ledger_status": "complete_exact_from_preserved_failure",
        "v1_optimizer_traces": "unavailable_from_spent_v1_nonpromotive",
        "model_state_before_sha256": model_before,
        "model_state_after_sha256": model_after,
        "fit_rows_loaded": v1.life.ROW_COUNT,
        "validation_rows_loaded": 0,
        "final_rows_loaded": 0,
        "ground_truth_target_tokens_used": 0,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
        "elapsed_seconds": elapsed,
        "maximum_allocated_cuda_bytes": allocated,
    }
    return result, artifact


def semantic_validate_result(
    value: Mapping[str, Any], frozen: Mapping[str, Any],
    artifact: Mapping[str, Any] | None = None,
) -> None:
    arms = set(v1.call_contract.REPORT_STUDENT_ARMS)
    programs = arms - {"continuous_teacher_F1"}
    required = {
        "schema", "status", "authority_sha256", "v1_programs_file_sha256",
        "postfit_report", "program_prices", "polarization_replay_by_device",
        "report_call_ledger", "model_call_ledger", "v1_fit_call_ledger_status",
        "v1_optimizer_traces", "model_state_before_sha256",
        "model_state_after_sha256", "fit_rows_loaded", "validation_rows_loaded",
        "final_rows_loaded", "ground_truth_target_tokens_used",
        "authorized_for_validation", "authorized_for_final",
        "authorized_for_global_ledger_credit", "elapsed_seconds",
        "maximum_allocated_cuda_bytes",
    }
    if set(value) != required or value.get(
        "schema"
    ) != "block3_consequence_family_f_v2_recovery_results" or value.get(
        "status"
    ) != "fit_reporting_recovered_no_validation_or_final_opened" or value.get(
        "authority_sha256"
    ) != frozen["authority_sha256"] or set(value.get("postfit_report", {})) != arms or set(
        value.get("program_prices", {})
    ) != programs or set(value.get("polarization_replay_by_device", {})) != programs:
        raise RuntimeError("family-F v2 result schema or arm registry changed")
    if any(
        set(metrics) != {"document_balanced_teacher_kl", "row_mean_teacher_kl", "summed_write_nrmse"}
        or any(not math.isfinite(number) or number < -1e-7 for number in metrics.values())
        for metrics in value["postfit_report"].values()
    ):
        raise RuntimeError("family-F v2 report metrics changed")
    if any(
        set(devices) != {"cpu", "cuda"} or any(
            set(metrics) != {"max_absolute", "max_relative"}
            or any(not math.isfinite(number) or number < 0 for number in metrics.values())
            or metrics["max_relative"] > v1.core.REPLAY_RELATIVE_LIMIT
            for metrics in devices.values()
        )
        for devices in value["polarization_replay_by_device"].values()
    ):
        raise RuntimeError("family-F v2 polarization replay changed")
    expected_report_calls = {
        "schema": "block3_consequence_family_f_v2_recovery_calls",
        "prefixes": N_BATCHES,
        "teacher_suffixes": N_BATCHES,
        "student_suffixes": {
            arm: N_BATCHES for arm in v1.call_contract.REPORT_STUDENT_ARMS
        },
        "optimizer_steps": 0,
        "program_refits": 0,
    }
    if value.get("report_call_ledger") != expected_report_calls:
        raise RuntimeError("family-F v2 reporting call census changed")
    expected_model = {site: N_BATCHES if site <= v1.LAYER else 19 * N_BATCHES for site in range(18)}
    if set(value.get("model_call_ledger", {})) != {"attention", "mlp"}:
        raise RuntimeError("family-F v2 model call family registry changed")
    for kind in ("attention", "mlp"):
        if value["model_call_ledger"].get(kind) != {
            str(site): count for site, count in expected_model.items()
        }:
            raise RuntimeError("family-F v2 model call replay changed")
    if value.get("v1_programs_file_sha256") != V1_PINS[
        str(v1.PROGRAMS.relative_to(ROOT))
    ] or value.get("v1_fit_call_ledger_status") != "complete_exact_from_preserved_failure" or value.get(
        "v1_optimizer_traces"
    ) != "unavailable_from_spent_v1_nonpromotive" or value.get(
        "model_state_before_sha256"
    ) != value.get("model_state_after_sha256") or not isinstance(value.get(
        "model_state_before_sha256"
    ), str) or len(value["model_state_before_sha256"]) != 64 or value.get(
        "fit_rows_loaded"
    ) != 480 or any(
        value.get(key) != expected for key, expected in {
            "validation_rows_loaded": 0, "final_rows_loaded": 0,
            "ground_truth_target_tokens_used": 0,
            "authorized_for_validation": False, "authorized_for_final": False,
            "authorized_for_global_ledger_credit": False,
        }.items()
    ):
        raise RuntimeError("family-F v2 lineage or permissions changed")
    if not isinstance(value.get("elapsed_seconds"), (int, float)) or not (
        0 <= value["elapsed_seconds"] <= MAX_WALL_SECONDS
    ) or not isinstance(value.get("maximum_allocated_cuda_bytes"), int) or not (
        0 <= value["maximum_allocated_cuda_bytes"] <= MAX_ALLOCATED_CUDA_BYTES
    ):
        raise RuntimeError("family-F v2 resource receipt changed")
    if artifact is not None:
        cpu_programs = {
            name: v1._materialize_program(payload, device="cpu")
            for name, payload in artifact["programs"].items()
        }
        expected_prices = {
            name: v1.core.program_price(program) for name, program in cpu_programs.items()
        }
        if value["program_prices"] != expected_prices:
            raise RuntimeError("family-F v2 program prices do not reconstruct")
        expected_cpu = {
            name: v1.deployed_polarization_replay(program)
            for name, program in cpu_programs.items()
        }
        if any(
            value["polarization_replay_by_device"][name]["cpu"] != replay
            for name, replay in expected_cpu.items()
        ):
            raise RuntimeError("family-F v2 canonical CPU polarization does not reconstruct")


def semantic_validate_receipt(
    value: Mapping[str, Any], *, frozen: Mapping[str, Any],
    authority_hash: str, result_hash: str,
) -> None:
    expected = {
        "schema": "block3_consequence_family_f_v2_recovery_receipt",
        "status": "fit_reporting_recovery_complete_receipt_last",
        "authority_sha256": frozen["authority_sha256"],
        "authority_file_sha256": authority_hash,
        "results_file_sha256": result_hash,
        "v1_programs_file_sha256": V1_PINS[str(v1.PROGRAMS.relative_to(ROOT))],
        "source_closure_sha256": frozen["source_closure"]["sha256"],
        "validation_rows_loaded": 0,
        "final_rows_loaded": 0,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
    }
    if dict(value) != expected:
        raise RuntimeError("family-F v2 receipt changed")


def publish_receipt_last(
    receipt: Mapping[str, Any], *, claim: Any, started: float,
) -> None:
    """The final resource/lock gate and the only v2 receipt publication site."""
    require_resource_ceiling(started)
    claim.verify()
    collector.create_json(RECEIPT, dict(receipt))


def run() -> dict[str, Any]:
    require_pristine_namespace()
    claim = collector.acquire_claim(LOCK)
    started = time.time()
    try:
        source = source_closure()
        v1_binding = v1_file_binding()
        prior = v1.life.prior_artifact_binding()
        rows = v1.life.row_binding()
        checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
        frozen = authority(source, v1_binding, prior, rows, checkpoint)
        claim.verify()
        collector.create_json(AUTHORITY, frozen)
        if json.loads(AUTHORITY.read_text()) != frozen:
            raise RuntimeError("family-F v2 authority publication changed")
        result, artifact = execute_recovery(frozen, checkpoint, started)
        semantic_validate_result(result, frozen, artifact)
        claim.verify()
        collector.create_json(RESULTS, result)
        reloaded = json.loads(RESULTS.read_text())
        if reloaded != result:
            raise RuntimeError("family-F v2 result publication changed")
        semantic_validate_result(reloaded, frozen, artifact)
        authority_hash = file_sha256(AUTHORITY)
        result_hash = file_sha256(RESULTS)
        receipt = {
            "schema": "block3_consequence_family_f_v2_recovery_receipt",
            "status": "fit_reporting_recovery_complete_receipt_last",
            "authority_sha256": frozen["authority_sha256"],
            "authority_file_sha256": authority_hash,
            "results_file_sha256": result_hash,
            "v1_programs_file_sha256": V1_PINS[str(v1.PROGRAMS.relative_to(ROOT))],
            "source_closure_sha256": source["sha256"],
            "validation_rows_loaded": 0,
            "final_rows_loaded": 0,
            "authorized_for_validation": False,
            "authorized_for_final": False,
            "authorized_for_global_ledger_credit": False,
        }
        semantic_validate_receipt(
            receipt, frozen=frozen, authority_hash=authority_hash,
            result_hash=result_hash,
        )
        claim.verify()
        verify_source_closure(source)
        if v1_file_binding() != v1_binding:
            raise RuntimeError("family-F v2 terminal v1 binding changed")
        v1.life.verify_prior_artifact_binding(prior)
        v1.life.verify_row_binding(rows)
        if facade.validate_snapshot(verify_weights_sha256=True) != checkpoint:
            raise RuntimeError("family-F v2 terminal checkpoint changed")
        if json.loads(AUTHORITY.read_text()) != frozen or file_sha256(
            AUTHORITY
        ) != authority_hash or file_sha256(RESULTS) != result_hash:
            raise RuntimeError("family-F v2 terminal bytes changed")
        publish_receipt_last(receipt, claim=claim, started=started)
        published_receipt = json.loads(RECEIPT.read_text())
        semantic_validate_receipt(
            published_receipt, frozen=frozen, authority_hash=authority_hash,
            result_hash=result_hash,
        )
        return result
    except BaseException as error:
        if not FAILURE.exists() and not RECEIPT.exists():
            try:
                claim.verify()
                collector.create_json(FAILURE, {
                    "schema": "block3_consequence_family_f_v2_recovery_failure",
                    "status": "terminal_failure_no_receipt",
                    "error_type": type(error).__name__, "error": str(error),
                    "authority_exists": AUTHORITY.exists(),
                    "results_exists": RESULTS.exists(),
                    "receipt_exists": RECEIPT.exists(),
                    "elapsed_seconds": time.time() - started,
                })
            except BaseException:
                pass
        raise
    finally:
        claim.release()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
