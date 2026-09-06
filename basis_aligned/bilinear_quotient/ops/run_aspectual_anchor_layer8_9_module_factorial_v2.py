#!/usr/bin/env python3
# BQGATE: frozen A-E predictions; all CUDA execution is managed-queue only.
"""Exact L8/L9 module factorial at the aspectual-anchor residual onset."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import circuit_fast_screen_candidate_aspectual as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_layer8_9_module_factorial_v2.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/aspectual_anchor_has_vs_had_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_layer8_9_module_factorial_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.layer8_9_module_factorial_v2"
EXPECTED_PRIOR_SHA256 = "7db613656cb55bab762d55d2347b3bd931e86ef8c0671e82a32f3b3ffd6b119b"
EXPECTED_PARENT_SHA256 = "5ca2125e7d18bd6a377efcfa0c3a361b949e5a8fff4c053ae7481b4384c4fb94"
EXPECTED_AUTHORITY_SHA256 = "ca707c7720f0f36b43d7a01751bfc9ce9abeb1c3b7e0939f1616de82f4b468c3"
MODULES = ("attn:08", "mlp:08", "attn:09", "mlp:09")
HEADS = tuple(f"attn:09:head:{head:02d}" for head in range(9))
MODEL_FORWARDS_MAX = 416
EXAMPLE_EVALUATIONS_MAX = 13312


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def module_subsets() -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset
        for width in range(len(MODULES) + 1)
        for subset in itertools.combinations(MODULES, width)
    )


def arm_id(modules: tuple[str, ...]) -> str:
    return "native_empty" if not modules else "+".join(modules)


def validate_static() -> tuple[list[dict[str, object]], screen.CircuitFastScreenSpec, dict[str, object]]:
    if sha256(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior-art hash changed")
    if sha256(PARENT_RESULT) != EXPECTED_PARENT_SHA256:
        raise ExperimentError("parent result hash changed")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(PARENT_RESULT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if parent.get("candidate_id") != candidate.TASK_ID or parent.get("terminal") != "screen":
        raise ExperimentError("parent is not the frozen aspectual screen")
    rows = candidate.build_rows(candidate.TASK_ID)
    if candidate.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("candidate authority changed")
    spec = parent_runner.build_spec(rows)
    compiled = screen.compile_screen(spec, rows)
    if compiled["authority"]["fit_rows"]["records_sha256"] != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("compiled authority changed")
    if len(module_subsets()) != 16 or len(HEADS) != 9:
        raise ExperimentError("registered arm inventory changed")
    return rows, spec, parent


class MultiPatchBackend(producer.Bilin18TorchBackend):
    """Use temporary hooks to replace several exact cached writes in one run."""

    def patched_bank(
        self,
        batch: producer.ModelBatch,
        *,
        modules: tuple[str, ...] = (),
        head_site: str | None = None,
        donor_cache: dict[tuple[str, str], object],
    ) -> producer.BatchOutput:
        if head_site is not None and modules:
            raise ExperimentError("head and module interventions cannot share an arm")
        if len(modules) != len(set(modules)) or any(site not in MODULES for site in modules):
            raise ExperimentError("module arm is outside the frozen factorial")
        if head_site is not None and head_site not in HEADS:
            raise ExperimentError("head arm is outside the frozen decomposition")
        handles = []
        try:
            for site_id in modules:
                kind, layer_text = site_id.split(":")
                block = self.model.transformer.h[int(layer_text)]
                if kind == "attn":
                    def replace_attention(_module, _arguments, output, site=site_id):
                        if not isinstance(output, tuple) or not output:
                            raise ExperimentError("attention output contract changed")
                        values = list(output)
                        values[0] = self._replace(values[0], batch, site, donor_cache)
                        return tuple(values)
                    handles.append(block.attn.register_forward_hook(replace_attention))
                elif kind == "mlp":
                    def replace_mlp(_module, _arguments, output, site=site_id):
                        return self._replace(output, batch, site, donor_cache)
                    handles.append(block.mlp.register_forward_hook(replace_mlp))
                else:  # pragma: no cover - guarded above
                    raise ExperimentError("unknown module kind")
            if head_site is not None:
                layer = int(head_site.split(":")[1])
                head = int(head_site.rsplit(":", 1)[1])
                def replace_head(_module, arguments):
                    changed = self._replace_heads(
                        arguments[0], batch, layer, (head,), donor_cache
                    )
                    return (changed,) + tuple(arguments[1:])
                handles.append(
                    self.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
                        replace_head
                    )
                )
            return self.native(batch, capture=False)
        finally:
            for handle in reversed(handles):
                handle.remove()


def score_arm(
    arm_name: str,
    kind: str,
    rows: tuple[dict[str, object], ...],
    spec: screen.CircuitFastScreenSpec,
    backend: MultiPatchBackend,
    donor_cache: dict[tuple[str, str], object],
    native: dict[tuple[str, str], producer.NativeLogitEvidence],
    capability: kernel.CapabilityEvidence,
    target_scale: float,
    *,
    modules: tuple[str, ...] = (),
    head_site: str | None = None,
) -> tuple[kernel.SiteScreenResult, list[dict[str, object]], int, int]:
    evidence: list[kernel.ScalarInterventionEvidence] = []
    raw: list[dict[str, object]] = []
    forward_calls = 0
    evaluations = 0
    answer_changes = {item.transform_id: item.answer_changes for item in spec.task.transforms}
    for family in screen.TRANSFORMS:
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            batch = producer._batch(spec, chunk, "base")
            output = backend.patched_bank(
                batch, modules=modules, head_site=head_site, donor_cache=donor_cache
            )
            forward_calls += 1
            evaluations += len(chunk)
            if len(output.answer_foil) != len(chunk):
                raise ExperimentError("arm output coverage changed")
            for row, pair in zip(chunk, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                row_id = str(row[spec.task.row_id_field])
                native_base = native[(row_id, "base")].margin
                if answer_changes[family]:
                    base_score = -native_base
                    donor_score = native[(row_id, "donor")].margin
                    intervened_score = -(answer - foil)
                    effect_scale = None
                else:
                    base_score = native_base
                    donor_score = None
                    intervened_score = answer - foil
                    effect_scale = target_scale
                typed_family: kernel.Family = family  # type: ignore[assignment]
                evidence.append(kernel.ScalarInterventionEvidence(
                    record_id=f"{arm_name}|{row_id}",
                    pair_id=row_id,
                    family=typed_family,
                    evidence_kind=kind,  # type: ignore[arg-type]
                    site_id=arm_name,
                    base_score=base_score,
                    donor_score=donor_score,
                    intervened_score=intervened_score,
                    effect_scale=effect_scale,
                ))
                raw.append({
                    "arm_id": arm_name,
                    "family": family,
                    "row_id": row_id,
                    "answer_logit": answer,
                    "foil_logit": foil,
                })
    site = kernel.SiteRef(kind, arm_name)  # type: ignore[arg-type]
    scored = kernel.score_site(
        site,
        evidence=tuple(evidence),
        expected_record_ids=tuple(item.record_id for item in evidence),
        capability=capability,
        c_answer_changes=answer_changes["C"],
    )
    return scored, raw, forward_calls, evaluations


def score_json(result: kernel.SiteScreenResult) -> dict[str, object]:
    value = managed.literal_json(asdict(result))
    if not isinstance(value, dict):
        raise ExperimentError("score serialization failed")
    return value


def main() -> None:
    rows, spec, parent = validate_static()
    subsets = module_subsets()
    dryrun = {
        "schema": "aspectual_anchor_layer8_9_module_factorial_dryrun_v2",
        "candidate_id": CANDIDATE_ID,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "execution_policy": "managed_queue_only",
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "module_arm_count": len(subsets),
        "head_arm_count": len(HEADS),
        "registered_arm_count": len(subsets) + len(HEADS),
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,
        "model_updates": 0,
        "terminal_rule": "screen iff A/B/C/E pass; D reported separately",
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    enriched = tuple(screen.validate_fit_authority(spec, rows).values())
    backend = MultiPatchBackend.load("cuda")
    donor_cache: dict[tuple[str, str], object] = {}
    native: dict[tuple[str, str], producer.NativeLogitEvidence] = {}
    native_records: list[producer.NativeLogitEvidence] = []
    forward_calls = 0
    evaluations = 0
    for side in ("base", "donor"):
        for family in screen.TRANSFORMS:
            family_rows = [row for row in enriched if row["transform_id"] == family]
            for chunk in producer._chunks(family_rows, spec.batch_size):
                batch = producer._batch(spec, chunk, side)
                output = backend.native(batch, capture=side == "donor")
                forward_calls += 1
                evaluations += len(chunk)
                donor_cache.update(output.captured)
                for row_id, pair in zip(batch.row_ids, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    record = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
                    native_records.append(record)
                    native[(row_id, side)] = record
    cells, capability = producer._capability(spec, enriched, native)
    current_cells = managed.literal_json([asdict(cell) for cell in cells])
    parent_cells = parent["run"]["capability_cells"]
    capability_exact = current_cells == parent_cells and all(cell.passed for cell in cells)

    denominators = []
    for row in enriched:
        if row["transform_id"] not in {"A1", "A2"}:
            continue
        row_id = str(row[spec.task.row_id_field])
        base = -native[(row_id, "base")].margin
        donor = native[(row_id, "donor")].margin
        kernel.signed_pairwise_donor_recovery(base, donor, base)
        denominators.append(donor - base)
    target_scale = statistics.median(denominators)
    if not math.isfinite(target_scale) or target_scale <= kernel.MIN_DONOR_DENOMINATOR:
        raise ExperimentError("native target scale is invalid")

    arm_scores: dict[tuple[str, ...], kernel.SiteScreenResult] = {}
    raw_records: list[dict[str, object]] = []
    for subset in subsets:
        name = arm_id(subset)
        result, raw, calls, examples = score_arm(
            name, "module", enriched, spec, backend, donor_cache, native,
            capability, target_scale, modules=subset,
        )
        arm_scores[subset] = result
        raw_records.extend(raw)
        forward_calls += calls
        evaluations += examples

    head_scores: dict[str, kernel.SiteScreenResult] = {}
    for head_site in HEADS:
        result, raw, calls, examples = score_arm(
            head_site, "head", enriched, spec, backend, donor_cache, native,
            capability, target_scale, head_site=head_site,
        )
        head_scores[head_site] = result
        raw_records.extend(raw)
        forward_calls += calls
        evaluations += examples

    if any(item.target_recovery is None for item in arm_scores.values()):
        raise ExperimentError("a module arm did not produce a valid recovery")
    full = arm_scores[MODULES]
    without_attn09 = arm_scores[tuple(module for module in MODULES if module != "attn:09")]
    assert full.a1 is not None and full.a2 is not None
    assert full.p_invariance_effect is not None and full.c_absolute_recovery is not None
    pred_b = bool(
        full.a1.mean_effect >= 0.65
        and full.a2.mean_effect >= 0.65
        and full.a1.direction_fraction is not None
        and full.a1.direction_fraction >= 0.90
        and full.a2.direction_fraction is not None
        and full.a2.direction_fraction >= 0.90
        and full.p_invariance_effect <= 0.20
        and full.c_absolute_recovery <= 0.35
    )

    n = len(MODULES)
    shapley: dict[str, float] = {}
    for module in MODULES:
        contribution = 0.0
        for subset in subsets:
            if module in subset:
                continue
            extended = tuple(item for item in MODULES if item in set(subset) | {module})
            weight = (
                math.factorial(len(subset))
                * math.factorial(n - len(subset) - 1)
                / math.factorial(n)
            )
            base_value = float(arm_scores[subset].target_recovery)
            extended_value = float(arm_scores[extended].target_recovery)
            contribution += weight * (extended_value - base_value)
        shapley[module] = contribution
    attn09_drop = float(full.target_recovery) - float(without_attn09.target_recovery)
    pred_c = bool(
        attn09_drop >= 0.25
        and shapley["attn:09"] == max(shapley.values())
    )

    localized_heads = []
    for name, result in head_scores.items():
        if result.a1 is None or result.a2 is None:
            continue
        if (
            result.a1.mean_effect >= 0.15
            and result.a2.mean_effect >= 0.15
            and result.a1.direction_fraction is not None
            and result.a1.direction_fraction >= 0.80
            and result.a2.direction_fraction is not None
            and result.a2.direction_fraction >= 0.80
            and result.p_invariance_effect is not None
            and result.p_invariance_effect <= 0.20
            and result.c_absolute_recovery is not None
            and result.c_absolute_recovery <= 0.35
        ):
            localized_heads.append(name)

    expected_records = (len(subsets) + len(HEADS)) * len(enriched)
    pred_e = bool(
        len(raw_records) == expected_records
        and len({(item["arm_id"], item["row_id"]) for item in raw_records}) == expected_records
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    pred_a = capability_exact
    terminal = "screen" if pred_a and pred_b and pred_c and pred_e else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "composable_layer8_9_module_bank",
        "null": "module_composition_or_attn09_necessity_failed",
        "invalid": "authority_capability_or_coverage_invalid",
    }[terminal]
    elapsed = time.perf_counter() - started
    result = {
        "schema": "aspectual_anchor_layer8_9_module_factorial_result_v2",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": elapsed,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_and_native_capability": pred_a,
            "pred_b_four_module_composition": pred_b,
            "pred_c_attn09_is_necessary_within_bank": pred_c,
            "pred_d_head_resolution": bool(localized_heads),
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "target_scale": target_scale,
            "full_bank": score_json(full),
            "without_attn09": score_json(without_attn09),
            "attn09_full_bank_drop": attn09_drop,
            "factorial_shapley_target_recovery": shapley,
            "largest_shapley_module": max(shapley, key=shapley.get),
            "localized_attn09_singleton_heads": localized_heads,
            "head_resolution_terminal": "screen" if localized_heads else "null",
            "module_arm_count": len(arm_scores),
            "head_arm_count": len(head_scores),
            "raw_record_count": len(raw_records),
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "model_backwards": 0,
            "model_updates": 0,
        },
        "capability_cells": current_cells,
        "module_arms": [
            {
                "modules": list(subset),
                "result": score_json(arm_scores[subset]),
            }
            for subset in subsets
        ],
        "attn09_singleton_heads": [
            {"head_site": name, "result": score_json(head_scores[name])}
            for name in HEADS
        ],
        "intervention_logits": raw_records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "test the localized head or minimal held module subset on an outcome-sealed fresh aspect corpus"
            if terminal == "screen"
            else "retain resid10 as a distributed carrier and close this four-module composition without adaptive rescue"
        ),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "full_target_recovery": full.target_recovery,
        "attn09_drop": attn09_drop,
        "shapley": shapley,
        "localized_heads": localized_heads,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
