#!/usr/bin/env python3
# BQGATE: frozen A-E prospective predictions; CUDA execution is managed-queue only.
"""Prospective fresh-construction validation of the aspectual module circuit."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_fresh_construction_v1 as fresh
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_layer8_9_module_factorial_v2 as discovery
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_fresh_construction_transfer_v1.json"
DISCOVERY_RESULT = ROOT / "circuits/followups/aspectual_anchor_layer8_9_module_factorial_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_construction_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_fresh_construction_transfer_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.fresh_construction_transfer_v1"
EXPECTED_PRIOR_SHA256 = "2d55712e87efcb17c0c8eb54ed560f63a01b0731e953aa2158c8d0285fdac267"
EXPECTED_DISCOVERY_SHA256 = "06918c951ab52c0b6082f440addba553bb994fec02e1a774772473917fd40050"
EXPECTED_BUILDER_SHA256 = "f808d7f01a49c8d22614ea66cc956fd01adcba3f0c396e3f5d0e826f30779a22"
EXPECTED_AUTHORITY_SHA256 = "d57dee7e1402ee14baf885b23aca7bbd7501a4d1aca82ab94c11012a0b8b0f35"
MODEL_FORWARDS_MAX = 40
EXAMPLE_EVALUATIONS_MAX = 640
HEAD_PAIR_ID = "attn:09:heads:01,04"
ARMS = (
    ("module_bank", ("attn:08", "mlp:08", "attn:09", "mlp:09"), None),
    ("head_reduced_bank", ("attn:08", "mlp:08", "mlp:09"), HEAD_PAIR_ID),
    ("attention_minimal_bank", ("attn:08",), HEAD_PAIR_ID),
)


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    if sha256(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior-art hash changed")
    if sha256(DISCOVERY_RESULT) != EXPECTED_DISCOVERY_SHA256:
        raise ExperimentError("discovery result hash changed")
    if sha256(BUILDER) != EXPECTED_BUILDER_SHA256:
        raise ExperimentError("fresh builder hash changed")
    prior = json.loads(PRIOR.read_text())
    discovery_result = json.loads(DISCOVERY_RESULT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if discovery_result.get("terminal") != "screen":
        raise ExperimentError("discovery circuit is not a screen")
    if discovery_result["score"]["localized_attn09_singleton_heads"] != [
        "attn:09:head:01", "attn:09:head:04"
    ]:
        raise ExperimentError("discovery head set changed")
    rows = fresh.build_rows()
    if fresh.validate_rows(rows) != EXPECTED_AUTHORITY_SHA256:
        raise ExperimentError("fresh row authority changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-fresh-construction-transfer-v1",
        authority_sha256=EXPECTED_AUTHORITY_SHA256,
        expected_fit_rows=len(rows),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT",
            forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0,
            model_updates=0,
            evidence_bytes=16384,
        ),
    )
    enriched = screen.validate_fit_authority(spec, rows)
    if len(enriched) != len(rows):
        raise ExperimentError("fresh authority coverage changed")
    return rows, spec


class FreshBackend(discovery.MultiPatchBackend):
    """Allow the frozen H1/H4 pair to replace the full L9 attention write."""

    def patched_bank(
        self,
        batch: producer.ModelBatch,
        *,
        modules: tuple[str, ...] = (),
        head_site: str | None = None,
        donor_cache: dict[tuple[str, str], object],
    ) -> producer.BatchOutput:
        if len(modules) != len(set(modules)) or any(
            site not in discovery.MODULES for site in modules
        ):
            raise ExperimentError("module arm changed")
        if head_site not in {None, HEAD_PAIR_ID}:
            raise ExperimentError("head pair changed")
        if head_site is not None and "attn:09" in modules:
            raise ExperimentError("whole L9 attention and its head pair cannot coexist")
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
                else:  # pragma: no cover
                    raise ExperimentError("module kind changed")
            if head_site is not None:
                def replace_heads(_module, arguments):
                    changed = self._replace_heads(
                        arguments[0], batch, 9, (1, 4), donor_cache
                    )
                    return (changed,) + tuple(arguments[1:])
                handles.append(
                    self.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(
                        replace_heads
                    )
                )
            return self.native(batch, capture=False)
        finally:
            for handle in reversed(handles):
                handle.remove()


def score_json(value: kernel.SiteScreenResult) -> dict[str, object]:
    result = managed.literal_json(asdict(value))
    if not isinstance(result, dict):
        raise ExperimentError("score serialization failed")
    return result


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_fresh_construction_transfer_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "fresh_builder_sha256": EXPECTED_BUILDER_SHA256,
        "fresh_authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "discovery_result_sha256": EXPECTED_DISCOVERY_SHA256,
        "row_count": len(rows),
        "arm_count": len(ARMS),
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,
        "model_updates": 0,
        "stored_fit_scalars": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    enriched = tuple(screen.validate_fit_authority(spec, rows).values())
    backend = FreshBackend.load("cuda")
    donor_cache: dict[tuple[str, str], object] = {}
    native: dict[tuple[str, str], producer.NativeLogitEvidence] = {}
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
                    native[(row_id, side)] = producer.NativeLogitEvidence(
                        row_id, family, side, answer, foil  # type: ignore[arg-type]
                    )
    cells, capability = producer._capability(spec, enriched, native)
    pred_a = bool(cells and all(cell.passed for cell in cells))
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
        raise ExperimentError("fresh target scale invalid")

    scores: dict[str, kernel.SiteScreenResult] = {}
    raw_records: list[dict[str, object]] = []
    for name, modules, head_site in ARMS:
        score, raw, calls, examples = discovery.score_arm(
            name,
            "module",
            enriched,
            spec,
            backend,
            donor_cache,
            native,
            capability,
            target_scale,
            modules=modules,
            head_site=head_site,
        )
        scores[name] = score
        raw_records.extend(raw)
        forward_calls += calls
        evaluations += examples

    module_bank = scores["module_bank"]
    head_bank = scores["head_reduced_bank"]
    minimal_bank = scores["attention_minimal_bank"]
    assert module_bank.a1 is not None and module_bank.a2 is not None
    assert module_bank.p_invariance_effect is not None
    assert module_bank.c_absolute_recovery is not None
    pred_b = bool(
        module_bank.a1.mean_effect >= 0.50
        and module_bank.a2.mean_effect >= 0.50
        and module_bank.a1.direction_fraction is not None
        and module_bank.a1.direction_fraction >= 0.80
        and module_bank.a2.direction_fraction is not None
        and module_bank.a2.direction_fraction >= 0.80
        and module_bank.p_invariance_effect <= 0.20
        and module_bank.c_absolute_recovery <= 0.35
    )
    assert head_bank.target_recovery is not None and module_bank.target_recovery is not None
    assert head_bank.a1 is not None and head_bank.a2 is not None
    retained_fraction = head_bank.target_recovery / module_bank.target_recovery
    pred_c = bool(
        retained_fraction >= 0.80
        and head_bank.a1.mean_effect > 0.0
        and head_bank.a2.mean_effect > 0.0
        and head_bank.a1.direction_fraction is not None
        and head_bank.a1.direction_fraction >= 0.80
        and head_bank.a2.direction_fraction is not None
        and head_bank.a2.direction_fraction >= 0.80
    )
    pred_d = minimal_bank.terminal == "screen"
    expected_records = len(ARMS) * len(enriched)
    pred_e = bool(
        len(raw_records) == expected_records
        and len({(item["arm_id"], item["row_id"]) for item in raw_records}) == expected_records
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = "screen" if pred_a and pred_b and pred_e else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "fresh_construction_module_circuit_transfer",
        "null": "fresh_module_bank_transfer_failed",
        "invalid": "fresh_capability_authority_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_fresh_construction_transfer_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "fresh_builder_sha256": EXPECTED_BUILDER_SHA256,
        "fresh_authority_sha256": EXPECTED_AUTHORITY_SHA256,
        "discovery_result_sha256": EXPECTED_DISCOVERY_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_native_capability": pred_a,
            "pred_b_module_bank_transfer": pred_b,
            "pred_c_head_reduction": pred_c,
            "pred_d_minimal_bank_report": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "target_scale": target_scale,
            "module_bank": score_json(module_bank),
            "head_reduced_bank": score_json(head_bank),
            "head_reduced_retained_fraction": retained_fraction,
            "head_reduction_terminal": "screen" if pred_c else "null",
            "attention_minimal_bank": score_json(minimal_bank),
            "attention_minimal_terminal": "screen" if pred_d else "null",
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "raw_record_count": len(raw_records),
            "model_backwards": 0,
            "model_updates": 0,
            "stored_fit_scalars": 0,
        },
        "capability_cells": managed.literal_json([asdict(cell) for cell in cells]),
        "intervention_logits": raw_records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "promote the frozen module bank; promote H1/H4 reduction only if its separate gate passes"
            if terminal == "screen"
            else "retain discovery status only and close this construction-transfer route"
        ),
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "module_target_recovery": module_bank.target_recovery,
        "head_retained_fraction": retained_fraction,
        "minimal_target_recovery": minimal_bank.target_recovery,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
