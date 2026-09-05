#!/usr/bin/env python3
# BQGATE: Phase-0 has frozen opposing predictions, exact price, and no optimizer.
"""Run the Task-14 MLP15/17 full-rank conditional-response panel.

This is the cheap Phase-0 causal screen preregistered in
TASK14_MLP15_17_FULL_RANK_CONDITIONAL_RESPONSE_PANEL_PREREGISTRATION.md.
It patches only the final-token input to ``block.mlp.Down``.  No rank,
reconstruction, activation-energy, or parameter-count objective exists here.

For each relation H is the recipient prompt with head 11.3 replaced by the
natural donor head.  At each MLP, delta_z=z_H-z_B.  Removal uses z_H-delta_z
and sufficiency uses z_B+delta_z, so their signed logit effects must cancel.
The joint arm changes MLP15 first, then captures the live MLP17 product reached
after that intervention before changing MLP17.

Prediction A (instrument): native answer/foil replay <=1e-4, all product and
logit endpoints <=1e-4, exact bilinear closure <=1e-5, and joint provenance is
exactly the registered five events with live MLP17 matching an independently
recomputed MLP15-reset trajectory.

Prediction B (broad): independently on FIT and SELECT, the joint q>=0.10 and
absolute coherence>=0.50 in both recipient states of every registered target
class; at least one single MLP has q>=0.05 in every such cell.

Prediction C (direction-specific): independently on FIT and SELECT, joint
paired q>=0.10 for plural recipients, <0.05 for singular recipients, and at
least two other target classes contain a recipient-state cell with q<0.05.
FIT/SELECT disagreement is mixed.  Any control arm above 0.20 of the frozen
FIT target scale is control failure.  Negative beta is reported as a
compensatory response, not clipped.

Maximum price: 74 forwards, 2,214 sequence examples, zero backwards/updates,
and 35,592 raw numeric evidence bytes.  Phase 0B is deferred.  Publication is
create-only; GPU execution is managed-queue only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Protocol, Sequence

import numpy as np

import circuit_intervention_primitives as primitives
import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as reader
import task14_program_a_torch_backend as program_backend


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "ops/task14_mlp15_17_full_rank_panel_contract_v1.json"
DONORS = ROOT / "ops/circuit_battery_task14_fit_localization_donors_v2.json"
RESULT = ROOT / "circuits/followups/task14_mlp15_17_full_rank_panel_v1_result.json"
CONTRACT_SHA256 = "28843ebc049cfbbd5fe1a3894b0f05c0bdd09ef2957979717790d62f443c7b28"
DONORS_SHA256 = "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a"
SHARD_SHA256 = program_backend.SHARD_SHA256
PARENT_SHA256 = reader.PARENT_SHA256
HEAD_SITE = reader.HEAD_SITE
MODULES = (15, 17)
CONDITIONS = (
    "H", "MLP15_reset", "MLP15_rescue", "MLP17_reset", "MLP17_rescue",
    "joint_reset", "joint_rescue",
)
EXPECTED_JOINT_EVENTS = (
    "mlp15_product_enter",
    "mlp15_reset_applied",
    "mlp17_product_enter_after_mlp15",
    "mlp17_live_product_captured",
    "mlp17_reset_applied",
)
BATCH_SIZE = 32
REPLAY_ATOL = 1e-4
CLOSURE_ATOL = 1e-5
BROAD_Q_MIN = 0.10
BROAD_COHERENCE_MIN = 0.50
SINGLE_Q_MIN = 0.05
DIRECTION_LIVE_Q_MIN = 0.10
DIRECTION_WEAK_Q_MAX = 0.05
CONTROL_MAX = 0.20
RAW_NUMERIC_EVIDENCE_BYTES = 35_592

PRODUCT_PLANS = {
    "H": primitives.ProductPlan("H", (
        primitives.ProductAction(15, "observe", capture_key="z15_H"),
        primitives.ProductAction(17, "observe", capture_key="z17_H"),
    )),
    "MLP15_reset": primitives.ProductPlan("MLP15_reset", (
        primitives.ProductAction(15, "reset", base_key="z15"),
        primitives.ProductAction(17, "observe", capture_key="z17_live"),
    )),
    "MLP15_rescue": primitives.ProductPlan("MLP15_rescue", (
        primitives.ProductAction(15, "rescue", base_key="z15"),
    )),
    "MLP17_reset": primitives.ProductPlan("MLP17_reset", (
        primitives.ProductAction(17, "reset", base_key="z17"),
    )),
    "MLP17_rescue": primitives.ProductPlan("MLP17_rescue", (
        primitives.ProductAction(17, "rescue", base_key="z17"),
    )),
    "joint_reset": primitives.ProductPlan("joint_reset", (
        primitives.ProductAction(
            15, "reset", base_key="z15",
            events_before=("mlp15_product_enter",),
            events_after=("mlp15_reset_applied",),
        ),
        primitives.ProductAction(
            17, "reset", base_key="z17", capture_key="z17_live",
            events_before=("mlp17_product_enter_after_mlp15",),
            capture_event="mlp17_live_product_captured",
            events_after=("mlp17_reset_applied",),
        ),
    )),
    "joint_rescue": primitives.ProductPlan("joint_rescue", (
        primitives.ProductAction(15, "rescue", base_key="z15"),
        primitives.ProductAction(17, "rescue", base_key="z17"),
    )),
}


class Phase0Error(ValueError):
    """Frozen authority, backend behavior, or publication was invalid."""


@dataclass(frozen=True)
class Relation:
    ordinal: int
    record_id: str
    split: str
    target_endpoint_id: str
    donor_endpoint_id: str
    arm: str
    family: str
    matching: str
    expected_relation: str
    role: str
    target_class: str
    recipient_subject_state: int


@dataclass(frozen=True)
class NativeState:
    answer_foil: tuple[float, float]
    full_logits: object
    head11_3: object
    x15: object
    z15: object
    x17: object
    z17: object


@dataclass(frozen=True)
class ConditionOutput:
    answer_foil: tuple[tuple[float, float], ...]
    full_logits: object
    diagnostics: tuple[Mapping[str, object], ...]
    provenance: tuple[tuple[str, ...], ...]


class Phase0Backend(Protocol):
    def native(self, endpoints: Sequence[program_backend.Endpoint]) -> tuple[NativeState, ...]: ...

    def conditional(
        self, relations: Sequence[Relation], *, condition: str,
        endpoints: Mapping[str, program_backend.Endpoint],
        native: Mapping[str, NativeState],
    ) -> ConditionOutput: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()  # type: ignore[union-attr]
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise Phase0Error("backend returned a nonfinite array")
    return result


def _array_sha(value: object) -> str:
    array = np.ascontiguousarray(_array(value))
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + memoryview(array)).hexdigest()


def _pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise Phase0Error("backend returned a malformed answer/foil pair")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair):
        raise Phase0Error("backend returned a nonfinite answer/foil pair")
    return pair


def _target_class(record: Mapping[str, object]) -> str:
    if record["expected_relation"] == "same_subject_zero_projected_effect":
        return "control"
    arm, matching = str(record["arm"]), str(record["matching"])
    if arm == "answer_change" and matching == "paired":
        return "paired"
    if arm in {"answer_change", "P_positive_transfer"} and matching.startswith("cross_noun"):
        return "cross_noun"
    if arm == "cross_syntax":
        return "literal_cross_syntax"
    if arm in {"C_to_ordinary_singular", "ordinary_singular_to_C"}:
        return "complete_subject_transfer"
    raise Phase0Error(f"target relation has no registered class: {record}")


def _load_plan():
    expected = ((CONTRACT, CONTRACT_SHA256), (DONORS, DONORS_SHA256),
                (program_backend.SHARD_PATH, SHARD_SHA256), (reader.PARENT, PARENT_SHA256))
    for path, digest in expected:
        if _sha256(path) != digest:
            raise Phase0Error(f"immutable source changed: {path.name}")
    contract = json.loads(CONTRACT.read_text())
    if contract.get("phase") != "phase0_full_rank_conditional_response":
        raise Phase0Error("Phase0 contract identity changed")
    endpoints = program_backend.load_discovery_endpoints()
    donors = json.loads(DONORS.read_text())
    donor_meta = {row["endpoint_id"]: row for row in donors["endpoints"]}
    split_groups = {
        "FIT": set(contract["data"]["fit_group_numbers"]),
        "SELECT": set(contract["data"]["select_group_numbers"]),
    }
    materialized: dict[str, list[Relation]] = {"FIT": [], "SELECT": []}
    for record in donors["records"]:
        if record.get("partition") != "DISCOVERY":
            continue
        target_id, donor_id = record["target_endpoint_id"], record["donor_endpoint_id"]
        if target_id not in endpoints or donor_id not in endpoints:
            raise Phase0Error("relation endpoint is outside the DISCOVERY shard")
        target_group = int(donor_meta[target_id]["group_number"])
        donor_group = int(donor_meta[donor_id]["group_number"])
        split = next((name for name, groups in split_groups.items()
                      if target_group in groups and donor_group in groups), None)
        if split is None:
            continue
        expected_relation = str(record["expected_relation"])
        if expected_relation == "opposite_subject_toward_donor":
            role = "target"
        elif expected_relation == "same_subject_zero_projected_effect":
            role = "control"
        else:
            raise Phase0Error("unknown expected_relation")
        materialized[split].append(Relation(
            ordinal=int(record["ordinal"]), record_id=str(record["record_id"]), split=split,
            target_endpoint_id=str(target_id), donor_endpoint_id=str(donor_id),
            arm=str(record["arm"]), family=str(record["family"]),
            matching=str(record["matching"]), expected_relation=expected_relation,
            role=role, target_class=_target_class(record),
            recipient_subject_state=int(donor_meta[target_id]["subject_state"]),
        ))
    expected_counts = contract["data"]["target_class_state_counts"]
    for split in ("FIT", "SELECT"):
        materialized[split].sort(key=lambda row: row.ordinal)
        rows = materialized[split]
        if len(rows) != int(contract["data"][f"{split.lower()}_relation_count"]):
            raise Phase0Error(f"{split} relation count changed")
        if _canonical_sha([row.ordinal for row in rows]) != contract["data"][f"{split.lower()}_ordinal_sha256"]:
            raise Phase0Error(f"{split} ordinal hash changed")
        observed: dict[str, dict[str, int]] = {}
        for row in rows:
            if row.role == "target":
                state_key = "+1" if row.recipient_subject_state == 1 else "-1"
                observed.setdefault(row.target_class, {}).setdefault(state_key, 0)
                observed[row.target_class][state_key] += 1
        if observed != expected_counts[split]:
            raise Phase0Error(f"{split} target class/state census changed: {observed}")
    relations = tuple(materialized["FIT"] + materialized["SELECT"])
    if len(relations) != 298 or len({row.record_id for row in relations}) != 298:
        raise Phase0Error("inner relation census changed")
    _rows, parent = reader._load()
    native_parent, _head_parent = reader._parent_maps(parent)
    return contract, endpoints, relations, native_parent


def _chunks(values: Sequence[object]):
    return tuple(values[start:start + BATCH_SIZE] for start in range(0, len(values), BATCH_SIZE))


class Task14MLPPhase0TorchBackend:
    """One narrow extension of the established exact Task14 manual forward."""

    def __init__(self, base: producer.Bilin18TorchBackend) -> None:
        self.torch, self.F, self.model, self.device = base.torch, base.F, base.model, base.device

    @classmethod
    def load(cls, device: str = "cuda") -> "Task14MLPPhase0TorchBackend":
        return cls(producer.Bilin18TorchBackend.load(device))

    def _tensorize(self, endpoints: Sequence[program_backend.Endpoint]):
        lengths = tuple(len(endpoint.token_ids) for endpoint in endpoints)
        maximum = max(lengths)
        tokens = self.torch.zeros(
            len(endpoints), maximum, dtype=self.torch.long, device=self.device,
        )
        for index, endpoint in enumerate(endpoints):
            tokens[index, :lengths[index]] = self.torch.tensor(
                endpoint.token_ids, dtype=self.torch.long, device=self.device,
            )
        return tokens, lengths

    def native(self, endpoints: Sequence[program_backend.Endpoint]) -> tuple[NativeState, ...]:
        output, captures, _diagnostics, _provenance = self._forward(
            endpoints, relations=None, condition="native", native=None,
        )
        return tuple(NativeState(
            answer_foil=output.answer_foil[index], full_logits=output.full_logits[index],
            head11_3=captures[index]["head11_3"], x15=captures[index]["x15"],
            z15=captures[index]["z15"], x17=captures[index]["x17"],
            z17=captures[index]["z17"],
        ) for index in range(len(endpoints)))

    def conditional(
        self, relations: Sequence[Relation], *, condition: str,
        endpoints: Mapping[str, program_backend.Endpoint], native: Mapping[str, NativeState],
    ) -> ConditionOutput:
        if condition not in CONDITIONS:
            raise Phase0Error(f"unknown condition {condition}")
        targets = [endpoints[row.target_endpoint_id] for row in relations]
        output, _captures, diagnostics, provenance = self._forward(
            targets, relations=relations, condition=condition, native=native,
        )
        return ConditionOutput(output.answer_foil, output.full_logits, diagnostics, provenance)

    @dataclass(frozen=True)
    class _Output:
        answer_foil: tuple[tuple[float, float], ...]
        full_logits: np.ndarray

    def _forward(self, endpoints, *, relations, condition, native):
        torch, F, model = self.torch, self.F, self.model
        tokens, lengths = self._tensorize(endpoints)
        positions = tuple(endpoint.final_position for endpoint in endpoints)
        captures = [dict() for _ in endpoints]
        diagnostics = [dict(endpoint_error=0.0) for _ in endpoints]
        runtime = None
        if relations is not None:
            base_vectors = {}
            for row in relations:
                state = native[row.target_endpoint_id]
                base_vectors[(row.record_id, "z15")] = state.z15
                base_vectors[(row.record_id, "z17")] = state.z17
            runtime = primitives.ProductInterventionRuntime(
                PRODUCT_PLANS[condition], row_ids=[row.record_id for row in relations],
                positions=positions, base_vectors=base_vectors,
            )
        with torch.no_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x0, v1 = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0

                def c_proj_pre(_module, arguments):
                    value = arguments[0]
                    if layer != 11:
                        return None
                    changed = value
                    width, start, stop = 128, 3 * 128, 4 * 128
                    if relations is None:
                        for index, position in enumerate(positions):
                            captures[index]["head11_3"] = value[index, position, start:stop].detach().cpu().clone()
                        return None
                    changed = value.clone()
                    for index, (row, position) in enumerate(zip(relations, positions)):
                        replacement = native[row.donor_endpoint_id].head11_3
                        replacement = torch.as_tensor(replacement, device=value.device, dtype=value.dtype)
                        if tuple(replacement.shape) != (width,):
                            raise Phase0Error("donor head has wrong shape")
                        changed[index, position, start:stop] = replacement
                    return (changed,) + tuple(arguments[1:])

                attention_handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
                try:
                    attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                finally:
                    attention_handle.remove()
                x = live + attention
                mlp_input = F.rms_norm(x, (model.config.n_embd,))
                if relations is None and layer in MODULES:
                    key = str(layer)

                    def native_down_pre(_module, arguments):
                        product = arguments[0]
                        for index, position in enumerate(positions):
                            captures[index][f"x{key}"] = mlp_input[index, position].detach().cpu().clone()
                            captures[index][f"z{key}"] = product[index, position].detach().cpu().clone()
                        return None

                    down_hook = native_down_pre
                elif runtime is not None and PRODUCT_PLANS[condition].action_at(layer) is not None:
                    if condition == "H":
                        for index, position in enumerate(positions):
                            diagnostics[index][f"x{layer}_H"] = mlp_input[index, position].detach().cpu().clone()
                    down_hook = runtime.hook(layer)
                else:
                    down_hook = None
                down_handle = block.mlp.Down.register_forward_pre_hook(down_hook) if down_hook else None
                try:
                    mlp = block.mlp(mlp_input)
                finally:
                    if down_handle is not None:
                        down_handle.remove()
                x = x + mlp
            logits = 30.0 * torch.tanh(
                model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0
            )
            final = torch.stack([logits[i, length - 1].float() for i, length in enumerate(lengths)])
            pairs = tuple((float(final[i, endpoint.answer_id]), float(final[i, endpoint.foil_id]))
                          for i, endpoint in enumerate(endpoints))
            full = final.detach().cpu().numpy().astype(np.float32, copy=True)
        if runtime is None:
            provenance = tuple(() for _ in endpoints)
        else:
            provenance = runtime.provenance()
            for index, row in enumerate(relations):
                diagnostics[index]["endpoint_error"] = runtime.endpoint_error[row.record_id]
                for capture_key in ("z15_H", "z17_H", "z17_live"):
                    value = runtime.captures.get((row.record_id, capture_key))
                    if value is not None:
                        diagnostics[index][capture_key] = value
            if condition == "H":
                for index, row in enumerate(relations):
                    base_state = native[row.target_endpoint_id]
                    for layer in MODULES:
                        x_base = torch.as_tensor(getattr(base_state, f"x{layer}"))
                        z_base = torch.as_tensor(getattr(base_state, f"z{layer}"))
                        x_live = torch.as_tensor(diagnostics[index][f"x{layer}_H"])
                        z_live = torch.as_tensor(diagnostics[index][f"z{layer}_H"])
                        block = model.transformer.h[layer]
                        left_weight = block.mlp.Left.weight.detach().cpu()
                        right_weight = block.mlp.Right.weight.detach().cpu()
                        error32 = primitives.bilinear_closure_max_abs(
                            left_weight.float(), right_weight.float(),
                            x_base.float(), x_live.float(), z_base.float(), z_live.float(),
                        )
                        error64 = primitives.bilinear_closure_max_abs(
                            left_weight.double(), right_weight.double(), x_base.double(), x_live.double(),
                            z_base.double(), z_live.double(),
                        )
                        diagnostics[index][f"closure{layer}"] = max(error32, error64)
        return self._Output(pairs, full), tuple(captures), tuple(diagnostics), provenance


def _score(pair: tuple[float, float], role: str) -> float:
    answer, foil = pair
    return foil - answer if role == "target" else answer - foil


def _diff_summary(a: object, b: object) -> dict[str, float]:
    return primitives.full_vocab_difference(_array(a), _array(b))


def _cell_metric(records: Sequence[Mapping[str, object]], effect: str) -> dict[str, float]:
    return primitives.signed_response_metrics(
        [float(row["head_effect"]) for row in records],
        [float(row[effect]) for row in records],
    )


def _classify_split(metrics: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, object]:
    cells = tuple(f"{name}|{state}" for name in (
        "paired", "cross_noun", "literal_cross_syntax", "complete_subject_transfer"
    ) for state in (-1, 1))
    broad = all(
        metrics["joint"][cell]["q"] >= BROAD_Q_MIN
        and metrics["joint"][cell]["absolute_coherence"] >= BROAD_COHERENCE_MIN
        and max(metrics["mlp15"][cell]["q"], metrics["mlp17"][cell]["q"]) >= SINGLE_Q_MIN
        for cell in cells
    )
    direction = (
        metrics["joint"]["paired|1"]["q"] >= DIRECTION_LIVE_Q_MIN
        and metrics["joint"]["paired|-1"]["q"] < DIRECTION_WEAK_Q_MAX
        and sum(
            any(metrics["joint"][f"{name}|{state}"]["q"] < DIRECTION_WEAK_Q_MAX
                for state in (-1, 1))
            for name in ("cross_noun", "literal_cross_syntax", "complete_subject_transfer")
        ) >= 2
    )
    pattern = "broad" if broad else "direction_specific" if direction else "mixed"
    return {"pattern": pattern, "broad": broad, "direction_specific": direction}


def compile_dryrun() -> dict[str, object]:
    contract, endpoints, relations, _native_parent = _load_plan()
    calls = math.ceil(len(endpoints) / BATCH_SIZE) + len(CONDITIONS) * math.ceil(len(relations) / BATCH_SIZE)
    return {
        "schema": "task14_mlp15_17_full_rank_panel_dryrun_v1",
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "contract_sha256": CONTRACT_SHA256, "donors_sha256": DONORS_SHA256,
        "shard_sha256": SHARD_SHA256, "parent_sha256": PARENT_SHA256,
        "endpoint_count": len(endpoints), "relation_count": len(relations),
        "relations_by_split": {
            split: sum(row.split == split for row in relations) for split in ("FIT", "SELECT")
        },
        "conditions": list(CONDITIONS),
        "full_vocab_baselines": {
            "MLP15_removal": "logits(MLP15_reset)-logits(H)",
            "MLP15_sufficiency": "logits(MLP15_rescue)-logits(MLP15_reset)",
            "MLP17_removal": "logits(MLP17_reset)-logits(H)",
            "MLP17_sufficiency": "logits(MLP17_rescue)-logits(MLP17_reset)",
            "joint_removal": "logits(joint_reset)-logits(H)",
            "joint_sufficiency": "logits(joint_rescue)-logits(joint_reset)",
        },
        "maximum_price": {
            "forward_calls": calls, "example_evaluations": 128 + 7 * 298,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": RAW_NUMERIC_EVIDENCE_BYTES,
            "non_numeric_provenance": "five event labels and two SHA256 strings per joint-reset relation",
        },
        "bars": {
            "native_and_endpoint_atol": REPLAY_ATOL, "bilinear_closure_atol": CLOSURE_ATOL,
            "broad_q_min": BROAD_Q_MIN, "broad_coherence_min": BROAD_COHERENCE_MIN,
            "single_q_min": SINGLE_Q_MIN, "direction_live_q_min": DIRECTION_LIVE_Q_MIN,
            "direction_weak_q_max": DIRECTION_WEAK_Q_MAX, "control_max": CONTROL_MAX,
        },
        "phase0B": contract["exact_bilinear_expansion"]["phase0B_status"],
    }


def run_science(*, backend: Phase0Backend | None = None, device: str = "cuda",
                clock=time.perf_counter) -> dict[str, object]:
    _contract, endpoints, relations, native_parent = _load_plan()
    executor = backend if backend is not None else Task14MLPPhase0TorchBackend.load(device)
    native: dict[str, NativeState] = {}
    native_evidence = []
    replay_error = 0.0
    forwards = evaluations = 0
    started = clock()
    endpoint_values = tuple(endpoints.values())
    for chunk in _chunks(endpoint_values):
        states = executor.native(chunk)  # type: ignore[arg-type]
        forwards += 1
        evaluations += len(chunk)
        if len(states) != len(chunk):
            raise Phase0Error("native backend count differs from batch")
        for endpoint, state in zip(chunk, states):
            observed = _pair(state.answer_foil)
            row_id, side = endpoint.endpoint_id.rsplit(":", 1)
            expected = native_parent[(row_id, side)]
            replay_error = max(replay_error, *(abs(a - b) for a, b in zip(observed, expected)))
            native[endpoint.endpoint_id] = state
            native_evidence.append({
                "endpoint_id": endpoint.endpoint_id,
                "answer_logit": observed[0], "foil_logit": observed[1],
            })
    if set(native) != set(endpoints):
        raise Phase0Error("native endpoint coverage changed")

    evidence = []
    closure_error = endpoint_error = cancellation_error = live_recompute_error = 0.0
    provenance_valid = True
    for chunk in _chunks(relations):
        outputs = {
            condition: executor.conditional(
                chunk, condition=condition, endpoints=endpoints, native=native,
            )
            for condition in CONDITIONS
        }
        forwards += len(CONDITIONS)
        evaluations += len(chunk) * len(CONDITIONS)
        for condition, output in outputs.items():
            if len(output.answer_foil) != len(chunk) or _array(output.full_logits).shape[0] != len(chunk):
                raise Phase0Error(f"{condition} backend count differs from batch")
        for index, row in enumerate(chunk):
            base = native[row.target_endpoint_id]
            base_pair = _pair(base.answer_foil)
            pairs = {name: _pair(outputs[name].answer_foil[index]) for name in CONDITIONS}
            logits = {name: _array(outputs[name].full_logits)[index] for name in CONDITIONS}
            base_logits = _array(base.full_logits)
            score_b = _score(base_pair, row.role)
            scores = {name: _score(pair, row.role) for name, pair in pairs.items()}
            effects = {
                "head_effect": scores["H"] - score_b,
                "mlp15_effect": scores["MLP15_reset"] - scores["H"],
                "mlp17_effect": scores["MLP17_reset"] - scores["H"],
                "joint_effect": scores["joint_reset"] - scores["H"],
            }
            summaries = {
                "MLP15_removal": _diff_summary(logits["MLP15_reset"], logits["H"]),
                "MLP15_sufficiency": _diff_summary(logits["MLP15_rescue"], logits["MLP15_reset"]),
                "MLP17_removal": _diff_summary(logits["MLP17_reset"], logits["H"]),
                "MLP17_sufficiency": _diff_summary(logits["MLP17_rescue"], logits["MLP17_reset"]),
                "joint_removal": _diff_summary(logits["joint_reset"], logits["H"]),
                "joint_sufficiency": _diff_summary(logits["joint_rescue"], logits["joint_reset"]),
            }
            cancellation_error = max(
                cancellation_error,
                float(np.max(np.abs((logits["MLP15_reset"] - logits["H"])
                                    + (logits["MLP15_rescue"] - logits["MLP15_reset"])))),
                float(np.max(np.abs((logits["MLP17_reset"] - logits["H"])
                                    + (logits["MLP17_rescue"] - logits["MLP17_reset"])))),
                float(np.max(np.abs((logits["joint_reset"] - logits["H"])
                                    + (logits["joint_rescue"] - logits["joint_reset"])))),
                float(np.max(np.abs(logits["H"] - logits["MLP15_rescue"]))),
                float(np.max(np.abs(logits["H"] - logits["MLP17_rescue"]))),
                float(np.max(np.abs(logits["H"] - logits["joint_rescue"]))),
            )
            hdiag = outputs["H"].diagnostics[index]
            closure15, closure17 = float(hdiag["closure15"]), float(hdiag["closure17"])
            closure_error = max(closure_error, closure15, closure17)
            endpoint_error = max(endpoint_error, *(float(outputs[name].diagnostics[index].get(
                "endpoint_error", 0.0)) for name in CONDITIONS))
            independent_z17 = outputs["MLP15_reset"].diagnostics[index].get("z17_live")
            joint_z17 = outputs["joint_reset"].diagnostics[index].get("z17_live")
            if independent_z17 is None or joint_z17 is None:
                provenance_valid = False
                live_norm = float("nan")
                independent_hash = joint_hash = "missing"
            else:
                live_recompute_error = max(
                    live_recompute_error,
                    float(np.max(np.abs(_array(independent_z17) - _array(joint_z17)))),
                )
                live_norm = float(np.linalg.norm(
                    _array(joint_z17).astype(np.float64) - _array(hdiag["z17_H"]).astype(np.float64)
                ))
                independent_hash, joint_hash = _array_sha(independent_z17), _array_sha(joint_z17)
                provenance_valid &= independent_hash == joint_hash
            events = outputs["joint_reset"].provenance[index]
            provenance_valid &= events == EXPECTED_JOINT_EVENTS
            condition_pairs = {
                name: {"answer_logit": pair[0], "foil_logit": pair[1]}
                for name, pair in pairs.items()
            }
            evidence.append({
                "ordinal": row.ordinal, "record_id": row.record_id, "split": row.split,
                "role": row.role, "target_class": row.target_class,
                "recipient_subject_state": row.recipient_subject_state,
                "arm": row.arm, "family": row.family, "matching": row.matching,
                **effects, "condition_logits": condition_pairs,
                "full_vocab_differences": summaries,
                "closure15_max_abs": closure15, "closure17_max_abs": closure17,
                "live_z17_difference_norm": live_norm,
                "joint_provenance": list(events),
                "z17_live_sha256": joint_hash,
                "z17_independent_recompute_sha256": independent_hash,
            })

    instrument_valid = (
        replay_error <= REPLAY_ATOL and endpoint_error <= REPLAY_ATOL
        and cancellation_error <= REPLAY_ATOL and closure_error <= CLOSURE_ATOL
        and live_recompute_error <= REPLAY_ATOL and provenance_valid
    )
    target_scale = statistics.median(
        abs(float(row["head_effect"])) for row in evidence
        if row["split"] == "FIT" and row["role"] == "target"
    )
    if not math.isfinite(target_scale) or target_scale <= 1e-12:
        instrument_valid = False

    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    decisions = {}
    effects = {"mlp15": "mlp15_effect", "mlp17": "mlp17_effect", "joint": "joint_effect"}
    for split in ("FIT", "SELECT"):
        metrics[split] = {component: {} for component in effects}
        for component, key in effects.items():
            for target_class in (
                "paired", "cross_noun", "literal_cross_syntax", "complete_subject_transfer"
            ):
                for state in (-1, 1):
                    rows = [row for row in evidence if row["split"] == split
                            and row["role"] == "target" and row["target_class"] == target_class
                            and row["recipient_subject_state"] == state]
                    if not rows:
                        raise Phase0Error("registered class/state cell is empty")
                    metrics[split][component][f"{target_class}|{state}"] = _cell_metric(rows, key)
        decisions[split] = _classify_split(metrics[split])

    controls = {}
    control_failure = False
    for split in ("FIT", "SELECT"):
        controls[split] = {}
        arms = sorted({str(row["arm"]) for row in evidence
                       if row["split"] == split and row["role"] == "control"})
        for arm in arms:
            rows = [row for row in evidence if row["split"] == split
                    and row["role"] == "control" and row["arm"] == arm]
            controls[split][arm] = {}
            for component, key in effects.items():
                normalized = math.sqrt(statistics.fmean(float(row[key]) ** 2 for row in rows)) / target_scale
                controls[split][arm][component] = normalized
                control_failure |= normalized > CONTROL_MAX

    patterns = (decisions["FIT"]["pattern"], decisions["SELECT"]["pattern"])
    terminal = (
        "instrument_invalid" if not instrument_valid else
        "control_failure" if control_failure else
        "broad_task14_full_rank_response_supported" if patterns == ("broad", "broad") else
        "direction_specific_compensatory_response_supported"
        if patterns == ("direction_specific", "direction_specific") else
        "mixed_full_rank_response_screen"
    )
    return {
        "schema": "task14_mlp15_17_full_rank_panel_result_v1",
        "screen_tier_only": True, "execution_policy": "managed_queue_only",
        "contract_sha256": CONTRACT_SHA256, "donors_sha256": DONORS_SHA256,
        "shard_sha256": SHARD_SHA256, "parent_sha256": PARENT_SHA256,
        "terminal": terminal,
        "predictions": {
            "pred_a_instrument_valid": instrument_valid,
            "pred_b_broad_by_split": {s: bool(decisions[s]["broad"]) for s in decisions},
            "pred_c_direction_specific_by_split": {
                s: bool(decisions[s]["direction_specific"]) for s in decisions
            },
            "control_failure": control_failure,
        },
        "split_decisions": decisions, "target_scale": target_scale,
        "target_metrics": metrics, "control_metrics": controls,
        "instrument": {
            "native_replay_max_abs": replay_error,
            "product_endpoint_max_abs": endpoint_error,
            "logit_endpoint_cancellation_max_abs": cancellation_error,
            "bilinear_closure_max_abs": closure_error,
            "joint_live_recompute_max_abs": live_recompute_error,
            "joint_provenance_valid": provenance_valid,
        },
        "native_endpoint_evidence": native_evidence,
        "evidence": evidence,
        "active_price": {
            "forward_calls": forwards, "example_evaluations": evaluations,
            "backward_calls": 0, "model_updates": 0,
            "raw_numeric_evidence_bytes": RAW_NUMERIC_EVIDENCE_BYTES,
        },
        "serial_seconds": clock() - started,
    }


def _write_create_only(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = -1
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise Phase0Error(f"{name} must be absent or exactly 1")
    if args.dry_run or any(os.environ.get(name) == "1" for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL")):
        print(json.dumps(compile_dryrun(), sort_keys=True))
        return
    result = run_science()
    _write_create_only(RESULT, result)
    print(json.dumps({key: result[key] for key in (
        "terminal", "predictions", "instrument", "active_price", "serial_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
