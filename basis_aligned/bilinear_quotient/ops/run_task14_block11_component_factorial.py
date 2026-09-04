#!/usr/bin/env python3
# BQGATE: three frozen science predictions are emitted by this managed experiment.
"""Task 14 block-11 exact component factorial (FIT-only follow-up).

Prior art and motivation
------------------------
Sections 1548 and 1550--1552 already show that the copula heads do not choose
singular versus plural agreement and that approximate replacement of individual
modules does not localize the agreement state.  The frozen exact-state screen
instead found a sharp transition at block 11:

    resid:11  0.3304497527 target recovery
    attn:11   0.6126028456
    mlp:11    0.0638338862
    resid:12  0.9369324969

This experiment does not repeat that scan.  It opens the three missing pair
cells in the exact 2^3 factorial over donor components at the final prediction
position:

    R = block-11 live input after its learned residual/embedding mixing
    A = block-11 attention output after its output projection
    M = block-11 MLP output

Each component is cached on both the native base and native donor trajectory.
Every arm replaces the block-11 output at the final prediction position by the
literal sum of its three independently chosen cached components, then runs the
unchanged recipient suffix.  This matters: the earlier resid:11 and attn:11
screens let downstream block components recompute, so they motivate the block
choice but are not fixed-component factorial corners.  For each row, Moebius
inversion gives the unique main, pair, and three-way terms whose sum is the
full donor-component response.

Registered predictions and null
-------------------------------
pred_a_instrument_replays_parent: captured base/donor native logits, the empty
    fixed-component arm, the donor-M-only arm, and R+A+M replay their exact
    parent counterparts to max absolute logit error 1e-4.
pred_b_ra_is_selective_sufficiency: on A1 and A2, R+A reaches at least 0.80
    mean donor recovery while remaining below the frozen P <= 0.20 and
    C <= 0.35 control ceilings.
pred_c_attention_dominates_m_increment: adding A conditional on R contributes
    at least 0.35 mean recovery in both A1 and A2, while adding donor M after
    R+A changes each family recovery by at most 0.15.

The scientific null is any valid run in which pred_b or pred_c fails.  A
pred_a failure makes the instrument invalid, not a scientific null.  Pair and
higher-order interaction signs are reported without post-hoc success bars:
positive target terms mean complementarity, negative terms mean redundancy.

Literal maximum price: 40 forward calls and 1,280 example evaluations (eight
base/donor capture calls plus eight arms times four family batches), zero
backward calls, zero updates, and 10,240 retained raw logit bytes. GPU execution is only
through ops/enqueue.sh; the dry-run path imports no torch and touches no queue.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Callable, Mapping, Protocol, Sequence

import circuit_experiment_spec as framework
import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import circuit_prior_art
import mobius


ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / "circuits/fast_screens/task14_subject_verb_agreement_full_state_v1_result.json"
PRIOR_ART = ROOT / "circuits/task14_block11_component_factorial_prior_art.json"
RESULT = ROOT / "circuits/followups/task14_block11_component_factorial_v1_result.json"

EXPERIMENT_ID = "task14-block11-component-factorial-v1"
PARENT_SHA256 = "bcabc936a78ee4955843c201684c2150e90897160a9201524f49c1bf71dc5744"
PRIOR_ART_SHA256 = "003e0d64a78553c1ecedcb614b76016b95ad43078a39c51669ea2cad3c3170cf"
AUTHORITY_SHA256 = "9b8ede7d17b0358467438b7f8fda7703bba1c93c9c594d55454404c1bb6e21cc"
CHECKPOINT_WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"

FACTORS = ("R", "A", "M")
ARM_MASKS = tuple(range(1 << len(FACTORS)))
TERM_MASKS = tuple(range(1, 1 << len(FACTORS)))
FAMILIES = ("A1", "A2", "P", "C")
KNOWN_PARENT_SITES = {4: "mlp:11", 7: "resid:12"}
BATCH_SIZE = 32
MAX_FORWARD_CALLS = 40
MAX_EXAMPLE_EVALUATIONS = 1280
MAX_EVIDENCE_BYTES = 10240
REPLAY_ATOL = 1.0e-4
RA_TARGET_RECOVERY_MIN = 0.80
CONDITIONAL_ATTENTION_RECOVERY_MIN = 0.35
CONDITIONAL_M_RECOVERY_MAX = 0.15

PREDICTION_KEYS = (
    "pred_a_instrument_replays_parent",
    "pred_b_ra_is_selective_sufficiency",
    "pred_c_attention_dominates_m_increment",
)
PRED_A, PRED_B, PRED_C = PREDICTION_KEYS
REGISTERED_PREDICTIONS = {
    PRED_A: (
        "Native, empty, fixed-M, and full fixed-component logits replay the parent within 1e-4."
    ),
    PRED_B: (
        "R+A reaches >=0.80 recovery while passing the frozen P/C control ceilings."
    ),
    PRED_C: (
        "A adds >=0.35 recovery given R and M adds at most 0.15 after R+A."
    ),
}


class FactorialError(ValueError):
    """The frozen sources, execution plan, or evidence are inconsistent."""


class FactorialBackend(Protocol):
    def capture_native(self, batch: producer.ModelBatch) -> producer.BatchOutput: ...

    def patched(
        self,
        batch: producer.ModelBatch,
        *,
        mask: int,
        component_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput: ...


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_factors(mask: int) -> frozenset[str]:
    if type(mask) is not int or not 0 <= mask < 1 << len(FACTORS):
        raise FactorialError("factor mask is outside the exact three-factor lattice")
    return frozenset(factor for bit, factor in enumerate(FACTORS) if mask & (1 << bit))


def mask_label(mask: int) -> str:
    chosen = mask_factors(mask)
    return "empty" if not chosen else "+".join(factor for factor in FACTORS if factor in chosen)


def component_source(mask: int, factor: str) -> str:
    """Return the fixed natural source selected for one factorial component."""
    if factor not in FACTORS:
        raise FactorialError("unknown block factor")
    return "donor" if factor in mask_factors(mask) else "base"


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise FactorialError(f"cannot read JSON source: {path}") from error
    if type(value) is not dict:
        raise FactorialError(f"JSON source is not an object: {path}")
    return value


def load_sources() -> tuple[list[dict[str, object]], dict[str, object], str]:
    """Hash-check the parent, authority adapter, and prior-art receipt on CPU."""
    if sha256(PARENT) != PARENT_SHA256:
        raise FactorialError("frozen Task 14 full-state parent changed")
    parent = _load_json(PARENT)
    if parent.get("schema") != "circuit_fast_screen_result_v1" \
            or parent.get("candidate_id") != candidate.TASK_ID \
            or parent.get("authority_sha256") != AUTHORITY_SHA256:
        raise FactorialError("parent result identity differs from Task 14")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != AUTHORITY_SHA256:
        raise FactorialError("Task 14 adapted authority changed")
    prior = _load_json(PRIOR_ART)
    prior_digest = circuit_prior_art.validate_source_files(prior, ROOT)
    if prior_digest != PRIOR_ART_SHA256:
        raise FactorialError("block-11 prior-art receipt changed")
    return rows, parent, prior_digest


def _parent_maps(parent: Mapping[str, object]) -> tuple[
    dict[tuple[str, str], tuple[float, float]],
    dict[tuple[str, str], tuple[float, float]],
]:
    run = parent.get("run")
    if type(run) is not dict:
        raise FactorialError("parent lacks its run record")
    native_records = run.get("native_logits")
    intervention_records = run.get("intervention_logits")
    if type(native_records) is not list or type(intervention_records) is not list:
        raise FactorialError("parent lacks native or intervention logits")
    native: dict[tuple[str, str], tuple[float, float]] = {}
    for record in native_records:
        if type(record) is not dict:
            raise FactorialError("parent native record is malformed")
        key = (str(record["row_id"]), str(record["side"]))
        pair = (float(record["answer_logit"]), float(record["foil_logit"]))
        if key in native or any(not math.isfinite(value) for value in pair):
            raise FactorialError("parent native record is duplicate or nonfinite")
        native[key] = pair
    interventions: dict[tuple[str, str], tuple[float, float]] = {}
    for record in intervention_records:
        if type(record) is not dict or type(record.get("site")) is not dict:
            raise FactorialError("parent intervention record is malformed")
        key = (str(record["site"]["site_id"]), str(record["row_id"]))
        pair = (float(record["answer_logit"]), float(record["foil_logit"]))
        if key in interventions or any(not math.isfinite(value) for value in pair):
            raise FactorialError("parent intervention record is duplicate or nonfinite")
        interventions[key] = pair
    return native, interventions


def _batch(rows: Sequence[Mapping[str, object]], side: str) -> producer.ModelBatch:
    if side not in {"base", "donor"}:
        raise FactorialError("batch side must be base or donor")
    prefix = "base" if side == "base" else "donor"
    return producer.ModelBatch(
        row_ids=tuple(str(row["row_id"]) for row in rows),
        side=side,  # type: ignore[arg-type]
        token_rows=tuple(tuple(int(token) for token in row[f"{prefix}_ids"]) for row in rows),
        answer_ids=tuple(int(row[f"{prefix}_answer_id"]) for row in rows),
        foil_ids=tuple(int(row[f"{prefix}_foil_id"]) for row in rows),
        semantic_positions=tuple(int(row[f"{prefix}_semantic_position"]) for row in rows),
    )


def _family_batches(rows: Sequence[Mapping[str, object]]) -> tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]:
    output = []
    for family in FAMILIES:
        selected = tuple(row for row in rows if row["transform_id"] == family)
        if len(selected) != BATCH_SIZE:
            raise FactorialError(f"{family} must contain exactly one 32-row batch")
        output.append((family, selected))
    return tuple(output)


class Block11FactorialTorchBackend(producer.Bilin18TorchBackend):
    """Exact native forward with a fixed-component block-11 output assembly."""

    def _assemble(
        self,
        native_output: object,
        batch: producer.ModelBatch,
        mask: int,
        component_cache: Mapping[tuple[str, str], object],
    ) -> object:
        """Set the prediction-position block output to chosen R + A + M."""
        mask_factors(mask)
        changed = native_output.clone()
        for index, (row_id, position) in enumerate(
            zip(batch.row_ids, batch.semantic_positions)
        ):
            pieces = []
            for factor in FACTORS:
                key = (row_id, f"block11:{component_source(mask, factor)}:{factor}")
                value = component_cache.get(key)
                if value is None:
                    raise FactorialError(f"component cache lacks {key}")
                pieces.append(value.to(
                    device=native_output.device, dtype=native_output.dtype,
                ))
            changed[index, position] = pieces[0] + pieces[1] + pieces[2]
        return changed

    def _forward_factorial(
        self,
        batch: producer.ModelBatch,
        *,
        capture: bool,
        mask: int,
        component_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput:
        mask_factors(mask)
        torch, functional, model = self.torch, self.F, self.model
        tokens, lengths = self._tensor_batch(batch)
        captured: dict[tuple[str, str], object] = {}
        with torch.no_grad():
            state = functional.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            embedding_state, first_value = state, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * state + block.lambdas[1] * embedding_state
                if layer == 11 and capture:
                    self._save(captured, batch, f"block11:{batch.side}:R", live)
                attention, first_value = block.attn(
                    functional.rms_norm(live, (model.config.n_embd,)), first_value,
                )
                if layer == 11 and capture:
                    self._save(captured, batch, f"block11:{batch.side}:A", attention)
                state = live + attention
                mlp_output = block.mlp(
                    functional.rms_norm(state, (model.config.n_embd,))
                )
                if layer == 11 and capture:
                    self._save(captured, batch, f"block11:{batch.side}:M", mlp_output)
                state = state + mlp_output
                if layer == 11 and not capture:
                    state = self._assemble(
                        state, batch, mask, component_cache,
                    )
            logits = 30.0 * torch.tanh(
                model.lm_head(functional.rms_norm(state, (model.config.n_embd,))) / 30.0
            )
            pairs = tuple(
                (
                    float(logits[index, length - 1, batch.answer_ids[index]].float()),
                    float(logits[index, length - 1, batch.foil_ids[index]].float()),
                )
                for index, length in enumerate(lengths)
            )
        return producer.BatchOutput(pairs, captured)

    def capture_native(self, batch: producer.ModelBatch) -> producer.BatchOutput:
        return self._forward_factorial(
            batch, capture=True, mask=0, component_cache={},
        )

    def patched(
        self,
        batch: producer.ModelBatch,
        *,
        mask: int,
        component_cache: Mapping[tuple[str, str], object],
    ) -> producer.BatchOutput:
        return self._forward_factorial(
            batch, capture=False, mask=mask, component_cache=component_cache,
        )


def _capability(parent: Mapping[str, object]) -> kernel.CapabilityEvidence:
    run = parent["run"]
    if type(run) is not dict or type(run.get("capability_cells")) is not list:
        raise FactorialError("parent capability evidence is missing")
    cells = tuple(
        kernel.FamilyCapabilityEvidence(
            family=str(cell["family"]),  # type: ignore[arg-type]
            correct_count=int(cell["correct_count"]),
            observed_count=int(cell["expected_count"]),
            expected_count=int(cell["expected_count"]),
            cell_id=str(cell["cell_id"]),
        )
        for cell in run["capability_cells"]
    )
    return kernel.CapabilityEvidence(cells)


def _target_scale(
    rows: Sequence[Mapping[str, object]],
    native: Mapping[tuple[str, str], tuple[float, float]],
) -> float:
    denominators = []
    for row in rows:
        if row["transform_id"] not in {"A1", "A2"}:
            continue
        row_id = str(row["row_id"])
        base = native[(row_id, "base")]
        donor = native[(row_id, "donor")]
        denominator = (donor[0] - donor[1]) + (base[0] - base[1])
        if denominator <= kernel.MIN_DONOR_DENOMINATOR:
            raise FactorialError("target donor denominator is invalid")
        denominators.append(denominator)
    scale = statistics.median(denominators)
    if not math.isfinite(scale) or scale <= kernel.MIN_DONOR_DENOMINATOR:
        raise FactorialError("target effect scale is invalid")
    return scale


def _normalized_response(
    family: str,
    base_pair: tuple[float, float],
    donor_pair: tuple[float, float],
    patched_pair: tuple[float, float],
    target_scale: float,
) -> float:
    base_margin = base_pair[0] - base_pair[1]
    patched_margin = patched_pair[0] - patched_pair[1]
    if family in {"A1", "A2"}:
        donor_margin = donor_pair[0] - donor_pair[1]
        return kernel.signed_pairwise_donor_recovery(
            -base_margin, donor_margin, -patched_margin,
        )
    return (patched_margin - base_margin) / target_scale


def _score_arm(
    mask: int,
    rows: Sequence[Mapping[str, object]],
    native: Mapping[tuple[str, str], tuple[float, float]],
    logits: Mapping[str, tuple[float, float]],
    capability: kernel.CapabilityEvidence,
    target_scale: float,
) -> kernel.SiteScreenResult:
    label = mask_label(mask)
    site = kernel.SiteRef("module", f"block11:{label}")
    evidence = []
    for row in rows:
        row_id, family = str(row["row_id"]), str(row["transform_id"])
        base_pair, donor_pair, patched_pair = (
            native[(row_id, "base")], native[(row_id, "donor")], logits[row_id]
        )
        base_margin = base_pair[0] - base_pair[1]
        patched_margin = patched_pair[0] - patched_pair[1]
        if family in {"A1", "A2"}:
            base_score = -base_margin
            donor_score = donor_pair[0] - donor_pair[1]
            intervened_score, effect_scale = -patched_margin, None
        else:
            base_score, donor_score = base_margin, None
            intervened_score, effect_scale = patched_margin, target_scale
        evidence.append(kernel.ScalarInterventionEvidence(
            record_id=f"{label}|{row_id}", pair_id=row_id,
            family=family,  # type: ignore[arg-type]
            evidence_kind="module", site_id=site.site_id,
            base_score=base_score, donor_score=donor_score,
            intervened_score=intervened_score, effect_scale=effect_scale,
        ))
    return kernel.score_site(
        site, evidence=tuple(evidence),
        expected_record_ids=tuple(record.record_id for record in evidence),
        capability=capability, c_answer_changes=False,
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise FactorialError("interaction summary received empty or nonfinite values")
    return {
        "mean": math.fsum(values) / len(values),
        "mean_absolute": math.fsum(abs(value) for value in values) / len(values),
        "positive_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def interaction_decomposition(
    rows: Sequence[Mapping[str, object]],
    native: Mapping[tuple[str, str], tuple[float, float]],
    arm_logits: Mapping[int, Mapping[str, tuple[float, float]]],
    target_scale: float,
) -> dict[str, object]:
    terms_by_family: dict[str, dict[str, list[float]]] = {
        family: {mask_label(mask): [] for mask in TERM_MASKS} for family in FAMILIES
    }
    arms_by_family: dict[str, dict[str, list[float]]] = {
        family: {mask_label(mask): [] for mask in ARM_MASKS} for family in FAMILIES
    }
    shapley_by_family: dict[str, dict[str, list[float]]] = {
        family: {factor: [] for factor in FACTORS} for family in FAMILIES
    }
    closure_max_abs = 0.0
    for row in rows:
        row_id, family = str(row["row_id"]), str(row["transform_id"])
        values: dict[frozenset[str], float] = {}
        for mask in ARM_MASKS:
            value = _normalized_response(
                family, native[(row_id, "base")], native[(row_id, "donor")],
                arm_logits[mask][row_id], target_scale,
            )
            values[mask_factors(mask)] = value
            arms_by_family[family][mask_label(mask)].append(value)
        dividends = mobius.dividends(values)
        reconstructed = mobius.reconstruct(dividends, FACTORS)
        closure_max_abs = max(
            closure_max_abs, abs(reconstructed - values[frozenset(FACTORS)])
        )
        shares = mobius.shapley(dividends)
        for mask in TERM_MASKS:
            terms_by_family[family][mask_label(mask)].append(
                dividends[mask_factors(mask)]
            )
        for factor in FACTORS:
            shapley_by_family[family][factor].append(shares[factor])
    families = {}
    for family in FAMILIES:
        families[family] = {
            "arm_responses": {
                label: _summary(values)
                for label, values in arms_by_family[family].items()
            },
            "mobius_terms": {
                label: _summary(values)
                for label, values in terms_by_family[family].items()
            },
            "shapley_component_credit": {
                factor: _summary(values)
                for factor, values in shapley_by_family[family].items()
            },
        }
    target_terms = {}
    for label in (mask_label(mask) for mask in TERM_MASKS):
        target_terms[label] = {
            key: (
                float(families["A1"]["mobius_terms"][label][key])
                + float(families["A2"]["mobius_terms"][label][key])
            ) / 2.0
            for key in ("mean", "mean_absolute", "positive_fraction")
        }
    return {
        "response_axis": (
            "A1/A2 signed donor recovery; P/C signed margin movement divided by "
            "the frozen median A1/A2 donor-effect scale"
        ),
        "target_term_sign": (
            "positive pair/higher-order term means complementarity; negative means redundancy"
        ),
        "closure_max_abs": closure_max_abs,
        "families": families,
        "equal_weight_A1_A2_terms": target_terms,
    }


def _max_pair_difference(
    first: Mapping[str, tuple[float, float]],
    second: Mapping[str, tuple[float, float]],
    row_ids: Sequence[str],
) -> float:
    return max(
        abs(first[row_id][coordinate] - second[row_id][coordinate])
        for row_id in row_ids for coordinate in (0, 1)
    )


def compile_dryrun() -> dict[str, object]:
    rows, _parent, prior_digest = load_sources()
    calls = []
    batches = _family_batches(rows)
    for side in ("base", "donor"):
        for family, family_rows in batches:
            descriptor = {
                "kind": "native_component_capture", "family": family,
                "side": side, "row_ids": [str(row["row_id"]) for row in family_rows],
            }
            descriptor["call_id"] = framework.canonical_sha256(descriptor)
            calls.append(descriptor)
    for mask in ARM_MASKS:
        for family, family_rows in batches:
            descriptor = {
                "kind": "block11_factorial_intervention", "mask": mask,
                "arm": mask_label(mask), "family": family, "side": "base",
                "row_ids": [str(row["row_id"]) for row in family_rows],
            }
            descriptor["call_id"] = framework.canonical_sha256(descriptor)
            calls.append(descriptor)
    if len(calls) != MAX_FORWARD_CALLS:
        raise FactorialError("compiled call count differs from literal price")
    plan = {
        "schema": "task14_block11_component_factorial_dryrun_v1",
        "experiment_id": EXPERIMENT_ID,
        "fit_only": True,
        "factors": list(FACTORS),
        "arms": [mask_label(mask) for mask in ARM_MASKS],
        "new_fixed_component_cells": ["R", "A", "R+A", "R+M", "A+M"],
        "source_hashes": {
            "parent_result": PARENT_SHA256,
            "authority": AUTHORITY_SHA256,
            "prior_art": prior_digest,
            "checkpoint_weights": CHECKPOINT_WEIGHTS_SHA256,
        },
        "registered_predictions": REGISTERED_PREDICTIONS,
        "null": "valid run in which pred_b or pred_c fails; pred_a failure is invalid",
        "price": {
            "forward_calls": MAX_FORWARD_CALLS,
            "example_evaluations": MAX_EXAMPLE_EVALUATIONS,
            "backward_calls": 0,
            "model_updates": 0,
            "evidence_bytes": MAX_EVIDENCE_BYTES,
        },
        "call_manifest_sha256": framework.canonical_sha256(calls),
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
    }
    plan["compiled_sha256"] = framework.canonical_sha256(plan)
    return plan


def run_science(
    *,
    backend: FactorialBackend | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    rows, parent, prior_digest = load_sources()
    batches = _family_batches(rows)
    native, parent_interventions = _parent_maps(parent)
    capability = _capability(parent)
    target_scale = _target_scale(rows, native)
    executor = backend if backend is not None else Block11FactorialTorchBackend.load("cuda")
    started = clock()
    component_cache: dict[tuple[str, str], object] = {}
    native_replay: dict[str, dict[str, tuple[float, float]]] = {
        "base": {}, "donor": {},
    }
    forward_calls = 0
    evaluations = 0
    for side in ("base", "donor"):
        for _family, family_rows in batches:
            batch = _batch(family_rows, side)
            output = executor.capture_native(batch)
            forward_calls += 1
            evaluations += len(family_rows)
            if len(output.answer_foil) != len(family_rows):
                raise FactorialError("native capture returned the wrong number of logits")
            component_cache.update(output.captured)
            native_replay[side].update(zip(batch.row_ids, output.answer_foil))
    expected_cache = {
        (str(row["row_id"]), f"block11:{side}:{factor}")
        for row in rows for side in ("base", "donor") for factor in FACTORS
    }
    if set(component_cache) != expected_cache:
        raise FactorialError("native component cache coverage is incomplete or unexpected")

    arm_logits: dict[int, dict[str, tuple[float, float]]] = {}
    for mask in ARM_MASKS:
        values: dict[str, tuple[float, float]] = {}
        for _family, family_rows in batches:
            batch = _batch(family_rows, "base")
            output = executor.patched(
                batch, mask=mask, component_cache=component_cache,
            )
            forward_calls += 1
            evaluations += len(family_rows)
            if len(output.answer_foil) != len(family_rows):
                raise FactorialError("factorial arm returned the wrong number of logits")
            values.update(zip(batch.row_ids, output.answer_foil))
        if len(values) != len(rows):
            raise FactorialError("factorial arm row coverage is incomplete")
        arm_logits[mask] = values
    elapsed = clock() - started
    if forward_calls != MAX_FORWARD_CALLS or evaluations != MAX_EXAMPLE_EVALUATIONS:
        raise FactorialError("active price differs from the compiled exact price")

    arm_scores = {
        mask: _score_arm(
            mask, rows, native, arm_logits[mask], capability, target_scale,
        )
        for mask in ARM_MASKS
    }
    if any(score.terminal == "invalid" for score in arm_scores.values()):
        raise FactorialError("generic scorer rejected complete factorial evidence")

    all_ids = [str(row["row_id"]) for row in rows]
    base_parent = {row_id: native[(row_id, "base")] for row_id in all_ids}
    donor_parent = {row_id: native[(row_id, "donor")] for row_id in all_ids}
    known_replay = {
        "base_native": _max_pair_difference(
            native_replay["base"], base_parent, all_ids,
        ),
        "donor_native": _max_pair_difference(
            native_replay["donor"], donor_parent, all_ids,
        ),
        "empty": _max_pair_difference(arm_logits[0], base_parent, all_ids),
    }
    parent_m = {
        row_id: parent_interventions[(KNOWN_PARENT_SITES[4], row_id)]
        for row_id in all_ids
    }
    known_replay["M"] = _max_pair_difference(
        arm_logits[4], parent_m, all_ids,
    )
    parent_ram = {
        row_id: parent_interventions[(KNOWN_PARENT_SITES[7], row_id)]
        for row_id in all_ids
    }
    known_replay["R+A+M"] = _max_pair_difference(
        arm_logits[7], parent_ram, all_ids,
    )
    replay_max = max(known_replay.values())

    ra = arm_scores[3]
    r = arm_scores[1]
    ram = arm_scores[7]
    if ra.a1 is None or ra.a2 is None or r.a1 is None or r.a2 is None \
            or ram.a1 is None or ram.a2 is None \
            or ra.p_invariance_effect is None or ra.c_absolute_recovery is None:
        raise FactorialError("required arm score fields are absent")
    conditional_attention = {
        "A1": ra.a1.mean_effect - r.a1.mean_effect,
        "A2": ra.a2.mean_effect - r.a2.mean_effect,
    }
    conditional_m_after_ra = {
        "A1": ram.a1.mean_effect - ra.a1.mean_effect,
        "A2": ram.a2.mean_effect - ra.a2.mean_effect,
    }
    predictions = {
        PRED_A: replay_max <= REPLAY_ATOL,
        PRED_B: (
            ra.a1.mean_effect >= RA_TARGET_RECOVERY_MIN
            and ra.a2.mean_effect >= RA_TARGET_RECOVERY_MIN
            and ra.p_invariance_effect <= kernel.MAX_P_INVARIANCE_EFFECT
            and ra.c_absolute_recovery <= kernel.MAX_C_ABSOLUTE_RECOVERY
        ),
        PRED_C: (
            conditional_attention["A1"] >= CONDITIONAL_ATTENTION_RECOVERY_MIN
            and conditional_attention["A2"] >= CONDITIONAL_ATTENTION_RECOVERY_MIN
            and abs(conditional_m_after_ra["A1"]) <= CONDITIONAL_M_RECOVERY_MAX
            and abs(conditional_m_after_ra["A2"]) <= CONDITIONAL_M_RECOVERY_MAX
        ),
    }
    terminal = (
        "invalid" if not predictions[PRED_A]
        else "screen" if all(predictions.values()) else "null"
    )
    result = {
        "schema": "task14_block11_component_factorial_result_v1",
        "experiment_id": EXPERIMENT_ID,
        "candidate_id": "subject_verb.number_agreement.block11_component_interaction",
        "screen_tier_only": True,
        "fit_only": True,
        "execution_policy": "managed_queue_only",
        "source_hashes": {
            "parent_result": PARENT_SHA256,
            "authority": AUTHORITY_SHA256,
            "prior_art": prior_digest,
            "checkpoint_weights": CHECKPOINT_WEIGHTS_SHA256,
        },
        "dryrun": compile_dryrun(),
        "terminal": terminal,
        "predictions": predictions,
        "registered_predictions": REGISTERED_PREDICTIONS,
        "bars": {
            "replay_max_abs_logit": REPLAY_ATOL,
            "ra_minimum_A1_and_A2_recovery": RA_TARGET_RECOVERY_MIN,
            "conditional_attention_minimum_A1_and_A2_recovery": (
                CONDITIONAL_ATTENTION_RECOVERY_MIN
            ),
            "conditional_M_after_RA_maximum_absolute_A1_and_A2_recovery": (
                CONDITIONAL_M_RECOVERY_MAX
            ),
            "maximum_P_effect": kernel.MAX_P_INVARIANCE_EFFECT,
            "maximum_C_effect": kernel.MAX_C_ABSOLUTE_RECOVERY,
        },
        "instrument_replay": {
            "max_abs_logit_by_arm": known_replay,
            "overall_max_abs_logit": replay_max,
        },
        "conditional_effects": {
            "attention_added_given_R": conditional_attention,
            "M_added_given_R_plus_A": conditional_m_after_ra,
        },
        "arm_scores": {
            mask_label(mask): managed.literal_json(asdict(score))
            for mask, score in arm_scores.items()
        },
        "interactions": interaction_decomposition(
            rows, native, arm_logits, target_scale,
        ),
        "timing": {
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "evidence_bytes": 8 * evaluations,
            "seconds": elapsed,
        },
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Task 14 block-11 component factorial.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compile and print the CPU-only execution plan without loading the model",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dryrun = compile_dryrun()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite prior result: {RESULT}")
    result = run_science()
    payload = managed.atomic_create_json(RESULT, result)
    summary = {
        "terminal": result["terminal"],
        "predictions": result["predictions"],
        "forward_calls": result["timing"]["forward_calls"],
        "example_evaluations": result["timing"]["example_evaluations"],
        "seconds": result["timing"]["seconds"],
        "result_sha256": hashlib.sha256(payload).hexdigest(),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
