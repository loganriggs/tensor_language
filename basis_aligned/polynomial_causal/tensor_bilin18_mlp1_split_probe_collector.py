#!/usr/bin/env python3
"""Create-only MLP1 same-context paired categorical-Fisher collection.

This is the production bridge from the frozen CPU plan to one aggregate-only result.
It rebuilds the admitted rank-640 tensor program, reuses only the frozen physical MLP1
direction bank, evaluates both independent probe halves on the exact same ordered
contexts, and publishes scalar response-geometry diagnostics. Raw logits, targets,
VJPs, response matrices, frames, and projectors do not enter the result artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

import finite_horizon_tangent_bundle as bundle
import freeze_mlp1_split_probe_plan as frozen
import tensor_bilin18_tangent_authority as authority_helpers
import tensor_bilin18_tangent_collector as tangent_collector
import tensor_bilin18_tangent_pilot as parent_pilot


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_mlp1_split_probe_results.json"
AUTHORITY_RECEIPT = HERE / "tensor_bilin18_mlp1_split_probe_authority_receipt.json"
RUN_LOCK = HERE / ".tensor_bilin18_mlp1_split_probe.lock"
PLAN = HERE / "mlp1_split_probe_plan.json"
PREREG = HERE / "MLP1_SPLIT_PROBE_PREREGISTRATION.md"
ROWS = HERE.parent / "bilinear_quotient/.rowcache/fineweb_n96_skip80.pt"
ROW_AUTHORITY = HERE.parent / "bilinear_quotient/.rowcache/fineweb_oracle_v2_receipt.json"
RANK640_PARENT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
PARENT_RESULT = HERE / "tensor_bilin18_tangent_pilot_results.json"
PARENT_PROGRAM_AUTHORITY = HERE / "tensor_bilin18_tangent_authority_receipt.json"
PARENT_GEOMETRY = HERE / "tensor_bilin18_tangent_geometry.pt"
PARENT_GEOMETRY_AUTHORITY = HERE / "tensor_bilin18_tangent_geometry_receipt.json"

EXPECTED_PLAN_SHA256 = "ff802543b9e3a7a7ddabc427679059c6404b83abdec49e1bf565b98ab878d518"
EXPECTED_PLAN_FINGERPRINT = "236d83c6779b064e266a51594edaab2bf4c961006c4ab7905f0e946aa48e16c6"
EXPECTED_RANK640_SHA256 = "639fb8480efee790403113079333100bd63bb61426f6fd6e4dcebd89b21c337d"
EXPECTED_CAUSAL_SHA256 = "73bd18ee81067775680b7d579036e6ec8c04b41116cd3e516b8460a7e7c7ab20"
EXPECTED_PARENT_RESULT_SHA256 = frozen.EXPECTED_PARENT_RESULT_SHA256
EXPECTED_PARENT_PROGRAM_AUTHORITY_SHA256 = frozen.EXPECTED_PROGRAM_AUTHORITY_SHA256
EXPECTED_PARENT_GEOMETRY_SHA256 = frozen.EXPECTED_GEOMETRY_SHA256
EXPECTED_PARENT_GEOMETRY_AUTHORITY_SHA256 = frozen.EXPECTED_GEOMETRY_AUTHORITY_SHA256
EXPECTED_MLP1_DIRECTIONS_SHA256 = frozen.EXPECTED_MLP1_DIRECTIONS_SHA256

SOURCE_SITE = 1
INJECTION_POSITION = 128
SCORE_START = 128
SCORE_STOP = 256
CONTEXTS = 16
PROBES_PER_HALF = 32
PRODUCTION_BATCH = 4
PRODUCTION_WIDTH = 1152
PRODUCTION_TOKEN_VOCAB = 50_257
PRODUCTION_LOGIT_VOCAB = 50_304
RANK640_STORED_VALUES = 516_707_766


def _deduplicate(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


SOURCES = _deduplicate((
    Path(__file__), PREREG, PLAN,
    HERE / "freeze_mlp1_split_probe_plan.py",
    HERE / "test_freeze_mlp1_split_probe_plan.py",
    HERE / "finite_horizon_tangent_bundle.py",
    HERE / "test_finite_horizon_tangent_bundle.py",
    HERE / "test_tensor_bilin18_mlp1_split_probe_collector.py",
    *parent_pilot.SOURCES,
))


def file_sha256(path: Path) -> str:
    return authority_helpers.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return authority_helpers.canonical_sha256(value)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def protected_snapshot() -> dict[str, Any]:
    """Bind the complete transitive scientific source closure and immutable parents."""
    authority_helpers.require_committed_sources(SOURCES)
    sources = {str(path): file_sha256(path) for path in SOURCES}
    immutable = {
        "plan": file_sha256(PLAN),
        "rows": file_sha256(ROWS),
        "row_authority": file_sha256(ROW_AUTHORITY),
        "rank640_predictive": file_sha256(RANK640_PARENT),
        "rank640_causal": file_sha256(CAUSAL_PARENT),
        "parent_tangent_result": file_sha256(PARENT_RESULT),
        "parent_program_authority": file_sha256(PARENT_PROGRAM_AUTHORITY),
        "parent_geometry": file_sha256(PARENT_GEOMETRY),
        "parent_geometry_authority": file_sha256(PARENT_GEOMETRY_AUTHORITY),
    }
    expected = {
        "plan": EXPECTED_PLAN_SHA256,
        "rows": frozen.EXPECTED_ROWS_FILE_SHA256,
        "row_authority": frozen.EXPECTED_ROW_AUTHORITY_SHA256,
        "rank640_predictive": EXPECTED_RANK640_SHA256,
        "rank640_causal": EXPECTED_CAUSAL_SHA256,
        "parent_tangent_result": EXPECTED_PARENT_RESULT_SHA256,
        "parent_program_authority": EXPECTED_PARENT_PROGRAM_AUTHORITY_SHA256,
        "parent_geometry": EXPECTED_PARENT_GEOMETRY_SHA256,
        "parent_geometry_authority": EXPECTED_PARENT_GEOMETRY_AUTHORITY_SHA256,
    }
    if immutable != expected:
        raise RuntimeError("MLP1 split-probe immutable parent identity changed")
    snapshot = {
        "source_closure": sources,
        "immutable_inputs": immutable,
        "git": authority_helpers.git_identity(SOURCES),
    }
    snapshot["fingerprint"] = canonical_sha256(snapshot)
    return snapshot


def load_plan_rows_geometry() -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Validate exact plan semantics and return only selected inputs and MLP1 directions."""
    if file_sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("serialized MLP1 split-probe plan changed")
    plan = json.loads(PLAN.read_text())
    rebuilt = frozen.build_plan()
    if plan != rebuilt or plan.get("plan_fingerprint") != EXPECTED_PLAN_FINGERPRINT:
        raise RuntimeError("MLP1 split-probe plan differs from its frozen builder")
    if plan.get("status") != "frozen_cpu_plan_no_gpu_authority" or plan[
        "decision"
    ].get("consequence_stage_authorized") is not False:
        raise RuntimeError("MLP1 split-probe plan semantics changed")

    predictive = json.loads(RANK640_PARENT.read_text())
    causal = json.loads(CAUSAL_PARENT.read_text())
    if predictive.get("status") != "pass" or predictive.get("rank") != 640 or (
        causal.get("status") != "rank640_robust_pass"
    ):
        raise RuntimeError("rank640 predictive or causal parent is not admitted")
    if file_sha256(RANK640_PARENT) != EXPECTED_RANK640_SHA256 or file_sha256(
        CAUSAL_PARENT
    ) != EXPECTED_CAUSAL_SHA256:
        raise RuntimeError("rank640 parent bytes changed")

    rows = torch.load(ROWS, map_location="cpu", weights_only=True)
    if tuple(rows.shape) != (96, 513) or rows.dtype != torch.int64 or (
        frozen.tensor_raw_sha256(rows) != frozen.EXPECTED_ROWS_RAW_SHA256
    ):
        raise RuntimeError("MLP1 split-probe parent rows changed")
    indices = tuple(plan["selection"]["row_indices"])
    selected = rows[list(indices), :SCORE_STOP].contiguous()
    if tuple(selected.shape) != (CONTEXTS, SCORE_STOP) or frozen.tensor_raw_sha256(
        selected
    ) != plan["selection"]["model_input_256_raw_sha256"]:
        raise RuntimeError("selected MLP1 model inputs changed")

    geometry_bank = parent_pilot.load_frozen_geometry()
    geometry = geometry_bank.geometries[SOURCE_SITE]
    directions = geometry.directions.contiguous().clone()
    if tuple(directions.shape) != (32, PRODUCTION_WIDTH) or tensor_sha256(
        directions
    ) != EXPECTED_MLP1_DIRECTIONS_SHA256:
        raise RuntimeError("frozen MLP1 physical direction bank changed")
    return plan, selected, directions


@dataclass(frozen=True)
class PairedBatchResult:
    first: Mapping[str, torch.Tensor]
    second: Mapping[str, torch.Tensor]
    receipt: Mapping[str, Any]


class MLP1PairedProbeTransaction:
    """One-use graph transaction; only direction-projected response blocks escape."""

    def __init__(
        self, *, program, tokens: torch.Tensor, row_ids: Sequence[str],
        directions: torch.Tensor, first_probe_seeds: Sequence[int],
        second_probe_seeds: Sequence[int], injection_position: int,
        score_start: int, score_stop: int, production: bool = True,
    ) -> None:
        rows = tuple(row_ids)
        first = tuple(first_probe_seeds)
        second = tuple(second_probe_seeds)
        if not rows or len(rows) != len(set(rows)) or len(rows) != tokens.shape[0]:
            raise ValueError("paired-probe row identities must align uniquely")
        if not first or len(first) != len(second) or set(first) & set(second) or (
            len(set(first)) != len(first) or len(set(second)) != len(second)
        ):
            raise ValueError("paired-probe seed halves must be equal, unique, and disjoint")
        if not torch.is_tensor(tokens) or tokens.ndim != 2 or tokens.dtype != torch.long or (
            tokens.device != program.token_embedding.device
        ):
            raise ValueError("paired-probe tokens have the wrong device, shape, or dtype")
        if not 0 <= injection_position <= score_start < score_stop <= tokens.shape[1]:
            raise ValueError("paired-probe causal support is malformed")
        if not torch.is_tensor(directions) or tuple(directions.shape) != (
            len(first), program.width,
        ) or directions.device.type != "cpu" or directions.dtype != torch.float64 or (
            directions.requires_grad or not bool(torch.isfinite(directions).all())
        ):
            raise ValueError("paired-probe directions are malformed")
        if int(torch.linalg.matrix_rank(directions)) != len(first):
            raise ValueError("paired-probe direction bank is rank deficient")
        if production:
            cost = program.cost_receipt()
            if tuple(tokens.shape) != (PRODUCTION_BATCH, SCORE_STOP) or len(first) != (
                PROBES_PER_HALF
            ) or injection_position != INJECTION_POSITION or score_start != SCORE_START or (
                score_stop != SCORE_STOP or program.width != PRODUCTION_WIDTH
                or program.logit_vocab != PRODUCTION_LOGIT_VOCAB
                or program.vocab_size != PRODUCTION_LOGIT_VOCAB
                or int(cost["total_stored_values"]) != RANK640_STORED_VALUES
                or int(cost["native_calls_per_forward"]) != 0
                or not bool(cost["total_input_support"])
                or int(tokens.min()) < 0 or int(tokens.max()) >= PRODUCTION_TOKEN_VOCAB
            ):
                raise ValueError("production paired-probe program contract changed")
        self.__program = program
        self.__tokens = tokens.contiguous().clone()
        self.__tokens_sha256 = tensor_sha256(self.__tokens)
        self.__row_ids = rows
        self.__directions = directions.contiguous().clone()
        self.__directions_sha256 = tensor_sha256(self.__directions)
        self.__first_seeds = first
        self.__second_seeds = second
        self.__injection_position = injection_position
        self.__score_start = score_start
        self.__score_stop = score_stop
        self.__closed = False

    @property
    def closed(self) -> bool:
        return self.__closed

    @property
    def aliases_revoked(self) -> bool:
        return self.__closed and all(getattr(self, name) is None for name in (
            "_MLP1PairedProbeTransaction__program",
            "_MLP1PairedProbeTransaction__tokens",
            "_MLP1PairedProbeTransaction__tokens_sha256",
            "_MLP1PairedProbeTransaction__row_ids",
            "_MLP1PairedProbeTransaction__directions",
            "_MLP1PairedProbeTransaction__directions_sha256",
            "_MLP1PairedProbeTransaction__first_seeds",
            "_MLP1PairedProbeTransaction__second_seeds",
            "_MLP1PairedProbeTransaction__injection_position",
            "_MLP1PairedProbeTransaction__score_start",
            "_MLP1PairedProbeTransaction__score_stop",
        ))

    def _revoke(self) -> None:
        self.__program = None
        self.__tokens = None
        self.__tokens_sha256 = None
        self.__row_ids = None
        self.__directions = None
        self.__directions_sha256 = None
        self.__first_seeds = None
        self.__second_seeds = None
        self.__injection_position = None
        self.__score_start = None
        self.__score_stop = None
        self.__closed = True

    def consume(self) -> PairedBatchResult:
        if self.__closed:
            raise RuntimeError("paired-probe graph transaction is spent")
        program = self.__program
        tokens = self.__tokens
        row_ids = self.__row_ids
        directions = self.__directions
        first_seeds = self.__first_seeds
        second_seeds = self.__second_seeds
        injection = self.__injection_position
        score_start = self.__score_start
        score_stop = self.__score_stop
        token_hash = self.__tokens_sha256
        direction_hash = self.__directions_sha256
        logits = leaves = targets = projected = None
        try:
            assert program is not None and tokens is not None and row_ids is not None
            assert directions is not None and first_seeds is not None and second_seeds is not None
            assert injection is not None and score_start is not None and score_stop is not None
            assert token_hash is not None and direction_hash is not None
            if tensor_sha256(tokens) != token_hash or tensor_sha256(directions) != direction_hash:
                raise RuntimeError("paired-probe owned input changed after construction")
            if tuple(program.parameters()) or any(
                value.requires_grad or value.grad is not None for value in program.buffers()
            ):
                raise RuntimeError("paired-probe program gradient state changed")
            logits, leaves, forward_receipt = (
                tangent_collector._forward_with_additive_write_leaves(
                    program, tokens, source_sites=(SOURCE_SITE,),
                )
            )
            if tuple(logits.shape) != (*tokens.shape, program.logit_vocab) or (
                logits.dtype != torch.float32 or not logits.requires_grad
            ):
                raise RuntimeError("paired-probe logits violate the graph contract")
            seeds = first_seeds + second_seeds
            targets = tangent_collector.stateless_categorical_fisher_targets(
                logits, row_ids, seeds, score_start=score_start, score_stop=score_stop,
            )
            log_probabilities = F.log_softmax(
                logits[:, score_start:score_stop].float(), dim=-1,
            )
            target_device = targets.to(logits.device)
            projected = torch.empty(
                len(seeds), len(row_ids), directions.shape[0], dtype=torch.float64,
            )
            rows = torch.arange(len(row_ids), device=logits.device)
            for probe in range(len(seeds)):
                selected = torch.gather(
                    log_probabilities, -1, target_device[probe].unsqueeze(-1),
                ).squeeze(-1)
                gradient = torch.autograd.grad(
                    selected.sum(), leaves[SOURCE_SITE],
                    retain_graph=probe + 1 < len(seeds), create_graph=False,
                    allow_unused=False,
                )[0]
                chosen = gradient[rows, injection].detach().cpu().double()
                if tuple(chosen.shape) != (len(row_ids), program.width) or not bool(
                    torch.isfinite(chosen).all()
                ):
                    raise RuntimeError("paired-probe MLP1 VJP is malformed")
                projected[probe] = chosen @ directions.T
            first_block = projected[:len(first_seeds)].contiguous()
            second_block = projected[len(first_seeds):].contiguous()
            first_rows = {
                row_id: first_block[:, row].clone() for row, row_id in enumerate(row_ids)
            }
            second_rows = {
                row_id: second_block[:, row].clone() for row, row_id in enumerate(row_ids)
            }
            receipt = {
                "status": "complete",
                "row_ids_first": list(row_ids),
                "row_ids_second": list(row_ids),
                "same_ordered_contexts": True,
                "tokens_sha256": token_hash,
                "directions_sha256": direction_hash,
                "first_probe_seeds": list(first_seeds),
                "second_probe_seeds": list(second_seeds),
                "probe_halves_disjoint": not bool(set(first_seeds) & set(second_seeds)),
                "first_target_ids_sha256": tensor_sha256(targets[:len(first_seeds)]),
                "second_target_ids_sha256": tensor_sha256(targets[len(first_seeds):]),
                "first_response_sha256": canonical_sha256([
                    [row_id, tensor_sha256(first_rows[row_id])] for row_id in row_ids
                ]),
                "second_response_sha256": canonical_sha256([
                    [row_id, tensor_sha256(second_rows[row_id])] for row_id in row_ids
                ]),
                "source_site": SOURCE_SITE,
                "injection_position": injection,
                "score_support": [score_start, score_stop],
                "forward": forward_receipt,
                "raw_logits_returned": False,
                "raw_targets_returned": False,
                "raw_vjps_returned": False,
            }
        finally:
            logits = leaves = targets = projected = None
            self._revoke()
        receipt["graph_aliases_revoked"] = self.aliases_revoked
        return PairedBatchResult(first=first_rows, second=second_rows, receipt=receipt)


def freeze_authority(
    run_lock: authority_helpers.RunLock, runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if AUTHORITY_RECEIPT.exists():
        raise RuntimeError("MLP1 paired-probe authority is create-only and already exists")
    run_lock.assert_owned()
    before = protected_snapshot()
    plan, _, directions = load_plan_rows_geometry()
    program, receipt = parent_pilot.build_rank640_program(torch.device("cuda"))
    authority_helpers.validate_program_receipt(receipt)
    manifest = authority_helpers.program_buffer_manifest(program)
    parent_authority = json.loads(PARENT_PROGRAM_AUTHORITY.read_text())
    if manifest != parent_authority.get("program_buffers"):
        raise RuntimeError("rebuilt rank640 program differs from parent program authority")
    after = protected_snapshot()
    if after != before:
        raise RuntimeError("protected inputs changed while freezing paired-probe authority")
    result = {
        "status": "mlp1_split_probe_authority_frozen_no_outcomes",
        "protected_snapshot": before,
        "plan_fingerprint": plan["plan_fingerprint"],
        "plan_sha256": file_sha256(PLAN),
        "rank640_predictive_sha256": file_sha256(RANK640_PARENT),
        "rank640_causal_sha256": file_sha256(CAUSAL_PARENT),
        "parent_tangent_result_sha256": file_sha256(PARENT_RESULT),
        "parent_program_authority_sha256": file_sha256(PARENT_PROGRAM_AUTHORITY),
        "parent_geometry_sha256": file_sha256(PARENT_GEOMETRY),
        "parent_geometry_authority_sha256": file_sha256(PARENT_GEOMETRY_AUTHORITY),
        "mlp1_directions_sha256": tensor_sha256(directions),
        "program_receipt": receipt,
        "program_buffers": manifest,
        "runtime_environment": dict(runtime_environment),
        "score_targets_sampled": False,
        "score_gradients_computed": False,
        "result_computed": False,
    }
    authority_helpers.publish_json_create_only(
        AUTHORITY_RECEIPT, result, ownership_check=run_lock.assert_owned,
    )
    return result


def validate_authority(
    value: Any, *, snapshot: Mapping[str, Any], runtime_environment: Mapping[str, Any],
) -> None:
    required = {
        "status", "protected_snapshot", "plan_fingerprint", "plan_sha256",
        "rank640_predictive_sha256", "rank640_causal_sha256",
        "parent_tangent_result_sha256", "parent_program_authority_sha256",
        "parent_geometry_sha256", "parent_geometry_authority_sha256",
        "mlp1_directions_sha256", "program_receipt", "program_buffers",
        "runtime_environment", "score_targets_sampled", "score_gradients_computed",
        "result_computed",
    }
    if not isinstance(value, dict) or set(value) != required or value["status"] != (
        "mlp1_split_probe_authority_frozen_no_outcomes"
    ) or value["protected_snapshot"] != dict(snapshot) or value["runtime_environment"] != (
        dict(runtime_environment)
    ) or value["plan_fingerprint"] != EXPECTED_PLAN_FINGERPRINT or value[
        "plan_sha256"
    ] != EXPECTED_PLAN_SHA256 or any(value[key] is not False for key in (
        "score_targets_sampled", "score_gradients_computed", "result_computed",
    )):
        raise RuntimeError("MLP1 paired-probe authority schema changed")
    exact = {
        "rank640_predictive_sha256": EXPECTED_RANK640_SHA256,
        "rank640_causal_sha256": EXPECTED_CAUSAL_SHA256,
        "parent_tangent_result_sha256": EXPECTED_PARENT_RESULT_SHA256,
        "parent_program_authority_sha256": EXPECTED_PARENT_PROGRAM_AUTHORITY_SHA256,
        "parent_geometry_sha256": EXPECTED_PARENT_GEOMETRY_SHA256,
        "parent_geometry_authority_sha256": EXPECTED_PARENT_GEOMETRY_AUTHORITY_SHA256,
        "mlp1_directions_sha256": EXPECTED_MLP1_DIRECTIONS_SHA256,
    }
    if any(value[key] != expected for key, expected in exact.items()):
        raise RuntimeError("MLP1 paired-probe authority parent identity changed")
    authority_helpers.validate_program_receipt(value["program_receipt"])


def run(
    run_lock: authority_helpers.RunLock, runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("MLP1 paired-probe result is create-only and already exists")
    if not AUTHORITY_RECEIPT.exists():
        raise RuntimeError("freeze MLP1 paired-probe authority before measurement")
    started = time.time()
    run_lock.assert_owned()
    before = protected_snapshot()
    authority_hash = file_sha256(AUTHORITY_RECEIPT)
    frozen_authority = json.loads(AUTHORITY_RECEIPT.read_text())
    validate_authority(
        frozen_authority, snapshot=before, runtime_environment=runtime_environment,
    )
    plan, selected_rows, directions = load_plan_rows_geometry()
    build_started = time.time()
    program, program_receipt = parent_pilot.build_rank640_program(torch.device("cuda"))
    authority_helpers.validate_program_receipt(program_receipt)
    manifest = authority_helpers.program_buffer_manifest(program)
    if manifest != frozen_authority["program_buffers"]:
        raise RuntimeError("measurement program differs from frozen paired-probe authority")
    build_seconds = time.time() - build_started

    row_ids = tuple(plan["selection"]["row_ids"])
    documents = tuple(plan["selection"]["document_ids"])
    first_seeds = tuple(plan["probe_halves"]["first"]["probe_seeds"])
    second_seeds = tuple(plan["probe_halves"]["second"]["probe_seeds"])
    if len(row_ids) != CONTEXTS or len(set(documents)) != CONTEXTS or set(
        first_seeds
    ) & set(second_seeds):
        raise RuntimeError("paired-probe context or seed ledger changed")
    first_rows: list[torch.Tensor] = []
    second_rows: list[torch.Tensor] = []
    batch_receipts = []
    target_hashes: set[str] = set()
    response_started = time.time()
    for start in range(0, CONTEXTS, PRODUCTION_BATCH):
        stop = start + PRODUCTION_BATCH
        batch_ids = row_ids[start:stop]
        transaction = MLP1PairedProbeTransaction(
            program=program,
            tokens=selected_rows[start:stop].to("cuda").contiguous(),
            row_ids=batch_ids, directions=directions,
            first_probe_seeds=first_seeds, second_probe_seeds=second_seeds,
            injection_position=INJECTION_POSITION, score_start=SCORE_START,
            score_stop=SCORE_STOP, production=True,
        )
        batch = transaction.consume()
        if not transaction.aliases_revoked or batch.receipt["row_ids_first"] != list(
            batch_ids
        ) or batch.receipt["row_ids_second"] != list(batch_ids):
            raise RuntimeError("paired-probe half ordering or transaction closure changed")
        for key in ("first_target_ids_sha256", "second_target_ids_sha256"):
            target_hash = str(batch.receipt[key])
            if target_hash in target_hashes:
                raise RuntimeError("paired-probe target bank replayed across half or batch")
            target_hashes.add(target_hash)
        first_rows.extend(batch.first[row_id] for row_id in batch_ids)
        second_rows.extend(batch.second[row_id] for row_id in batch_ids)
        batch_receipts.append(dict(batch.receipt))
        del transaction, batch
        torch.cuda.empty_cache()
        print(f"MLP1 paired-probe batch {stop // PRODUCTION_BATCH}/4", flush=True)
    first = torch.cat(first_rows, dim=0)
    second = torch.cat(second_rows, dim=0)
    if tuple(first.shape) != (CONTEXTS * PROBES_PER_HALF, 32) or (
        first.shape != second.shape or len(target_hashes) != 8
    ):
        raise RuntimeError("paired-probe response bank is incomplete")
    analysis = bundle.analyze_repeated_probe_physical_bundle(
        first, second, directions,
        probes_per_half=plan["operator"]["probes_per_half"],
        fixed_ranks=tuple(plan["operator"]["fixed_physical_projector_ranks"]),
        energy_fraction=plan["analysis"]["energy_fraction"],
        gap_ratio=plan["analysis"]["gap_ratio"],
        local_rank_limit=plan["analysis"]["local_rank_limit"],
        maximum_local_rank_difference=plan["analysis"]["maximum_half_rank_difference"],
        maximum_same_context_distance=plan["analysis"][
            "maximum_same_context_physical_projector_distance"
        ],
        minimum_context_fraction=plan["analysis"]["minimum_context_fraction"],
        minimum_bundle_distance_lcb=plan["analysis"][
            "minimum_cross_minus_same_bootstrap_lcb_95"
        ],
        bootstrap_repetitions=plan["analysis"]["bootstrap_repetitions"],
        bootstrap_seed=plan["analysis"]["bootstrap_seed"],
        promotion_contexts=tuple(plan["selection"]["promotion_context_indices"]),
    )
    response_seconds = time.time() - response_started
    del first, second, first_rows, second_rows
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "status": analysis["status"],
        "scope": (
            "conditional historical-row MLP1 final-output categorical-Fisher response "
            "geometry; no encoder-gauge, finite-replacement, CE, or removal claim"
        ),
        "plan_fingerprint": plan["plan_fingerprint"],
        "analysis": analysis,
        "program": program_receipt,
        "program_buffers": manifest,
        "parents": {
            "rank640_predictive_sha256": file_sha256(RANK640_PARENT),
            "rank640_causal_sha256": file_sha256(CAUSAL_PARENT),
            "parent_tangent_result_sha256": file_sha256(PARENT_RESULT),
            "parent_program_authority_sha256": file_sha256(PARENT_PROGRAM_AUTHORITY),
            "parent_geometry_sha256": file_sha256(PARENT_GEOMETRY),
            "parent_geometry_authority_sha256": file_sha256(PARENT_GEOMETRY_AUTHORITY),
        },
        "execution": {
            "batches": len(batch_receipts),
            "unique_target_hashes": len(target_hashes),
            "batch_receipts": batch_receipts,
            "same_ordered_rows_documents_tokens_positions": True,
            "raw_logits_returned": False,
            "raw_targets_returned": False,
            "raw_vjps_returned": False,
            "raw_responses_returned": False,
            "physical_frames_returned": False,
            "projectors_returned": False,
            "checkpoint_collected_before_response_measurement": True,
        },
        "provenance": {
            "protected_snapshot": before,
            "authority_sha256": authority_hash,
            "selected_model_input_raw_sha256": plan[
                "selection"
            ]["model_input_256_raw_sha256"],
            "row_ids": list(row_ids),
            "document_ids": list(documents),
            "mlp1_directions_sha256": tensor_sha256(directions),
        },
        "runtime_environment": dict(runtime_environment),
        "runtime_s": {
            "program_build": build_seconds,
            "paired_fisher_responses_and_analysis": response_seconds,
            "total": time.time() - started,
        },
    }
    after = protected_snapshot()
    if after != before or file_sha256(AUTHORITY_RECEIPT) != authority_hash or (
        authority_helpers.program_buffer_manifest(program) != manifest
    ) or frozen.tensor_raw_sha256(selected_rows) != plan[
        "selection"
    ]["model_input_256_raw_sha256"] or tensor_sha256(directions) != (
        EXPECTED_MLP1_DIRECTIONS_SHA256
    ):
        raise RuntimeError("paired-probe protected state changed before publication")
    authority_helpers.publish_json_create_only(
        OUTPUT, result, ownership_check=run_lock.assert_owned,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-authority", action="store_true")
    arguments = parser.parse_args()
    runtime = authority_helpers.configure_production_runtime()
    with authority_helpers.exclusive_run_lock(RUN_LOCK) as run_lock:
        result = (
            freeze_authority(run_lock, runtime)
            if arguments.freeze_authority else run(run_lock, runtime)
        )
    print(json.dumps({
        "status": result["status"],
        "plan_fingerprint": result.get("plan_fingerprint"),
        "result": str(AUTHORITY_RECEIPT if arguments.freeze_authority else OUTPUT),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
