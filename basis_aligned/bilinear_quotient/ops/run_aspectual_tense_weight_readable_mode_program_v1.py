#!/usr/bin/env python3
"""Patch only exact weight-readable modes of validated carrier components."""

# BQGATE: EXPERIMENT pred_a_authority_capability_projection_gauge_finiteness_and_exact_price pred_b_weight_readable_modes_are_sufficient pred_c_weight_complements_are_secondary pred_d_weight_modes_are_dimensionally_compressive pred_e_zero_causal_fit
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_group_eval as source_groups
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_tense_carrier_component_greedy_program_v1 as greedy
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_weight_readable_mode_program_v1.json"
AUDIT = ROOT / "circuits/followups/aspectual_tense_weight_tensor_relative_instrument_audit_v1_result.json"
PROGRAMS = ROOT / "circuits/followups/aspectual_tense_carrier_program_backward_pruning_v1_result.json"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
OUT = ROOT / "circuits/followups/aspectual_tense_weight_readable_mode_program_v1_result.json"
CANDIDATE_ID = "aspectual_tense.weight_readable_mode_program_v1"
EXPECTED = {
    "prior": "432a904fd804b651289cc09f0387b7e13965262df4a2ae20f11fcf2c5ea4b224",
    "audit": "a780f4eca5e8983d081190030b1c0123925d8ecfe2ee6d648a9a31e11e787f00",
    "programs": "bb6344f6446a5426a9b6342c30cbcd56ca821a01b5750e1ef3b940a6b52e15c0",
    "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
    "atlas": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
    "greedy": "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413",
    "source_rank": "c7570a2e25b444df84e40953e38d6bbc4b7b15c6d6f6657fda0696fb4eea3d34",
    "source_groups": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
AUTHORITY_PATHS = {
    "positioned": ROOT / "ops/positioned_component_program_eval.py",
    "atlas": ROOT / "ops/subspace_weight_atlas.py",
    "greedy": ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py",
    "source_rank": ROOT / "ops/run_aspectual_tense_l9h1h4_source_position_weight_validation_v1.py",
    "source_groups": ROOT / "ops/attention_source_group_eval.py",
    "das": ROOT / "ops/circuit_das_subspace.py",
    "producer": ROOT / "ops/circuit_fast_screen_producer.py",
}
PRICE = {"model_forwards": 20, "example_evaluations": 335, "records": 201,
         "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
HEADS = (1, 4)


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stored_basis(torch, record):
    shape = tuple(record["shape"])
    values = torch.tensor(record["values_column_major"], dtype=torch.float32)
    basis = values.reshape(shape[1], shape[0]).T.contiguous()
    return torch.linalg.qr(basis.double()).Q.float()


def row_basis(torch, matrix):
    _u, singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    keep = singular > singular.max().clamp_min(1e-30) * 1e-6
    return torch.linalg.qr(vh[keep].T.double()).Q.float().to(matrix.device)


def validate_static():
    paths = {"prior": PRIOR, "audit": AUDIT, "programs": PROGRAMS, "subspaces": SUBSPACES,
             **AUTHORITY_PATHS}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    audit, programs, subspaces = [json.loads(path.read_text()) for path in (AUDIT, PROGRAMS, SUBSPACES)]
    splits, chosen, pools = greedy.validate_static()
    if (audit.get("terminal") != "screen" or programs.get("terminal") != "screen"
            or {task: subspaces["subspaces"][task]["rank"] for task in ("has", "is")}
            != {"has": 18, "is": 3}):
        raise ExperimentError("audit, program, or subspace authority changed")
    return splits, chosen, pools, programs["pruned_paths"], subspaces


def label(component):
    return f"MLP{component.layer}" if component.kind == "mlp" else f"L{component.layer}H{component.heads[0]}"


def patch_projected(backend, batch, donor_batch, components, donor_cache,
                    recipient_banks, donor_banks, projections, *, complement):
    selected = positioned.validate_components(
        components, layers=len(backend.model.transformer.h), heads=backend.model.config.n_head)
    recipients, donors = positioned.validate_position_banks(
        tuple(map(len, batch.token_rows)), tuple(map(len, donor_batch.token_rows)),
        recipient_banks, donor_banks)
    handles, n_head = [], backend.model.config.n_head

    def project(delta, q):
        readable = (delta.float() @ q) @ q.T
        value = delta.float() - readable if complement else readable
        return value.to(delta.dtype)

    def head_hook(component):
        q = projections[label(component)]
        def hook(_module, arguments):
            value, donor = arguments[0], donor_cache[component.site_id]
            changed, width = value.clone(), value.shape[2] // n_head
            for row, (recipient, source) in enumerate(zip(recipients, donors)):
                for rpos, dpos in zip(recipient, source):
                    for head in component.heads:
                        start, stop = head * width, (head + 1) * width
                        live = value[row, rpos, start:stop]
                        delta = donor[row, dpos, start:stop].to(live) - live
                        changed[row, rpos, start:stop] = live + project(delta, q)
            return (changed,) + tuple(arguments[1:])
        return hook

    def mlp_hook(component):
        q = projections[label(component)]
        def hook(_module, _arguments, output):
            changed, donor = output.clone(), donor_cache[component.site_id]
            for row, (recipient, source) in enumerate(zip(recipients, donors)):
                for rpos, dpos in zip(recipient, source):
                    live = output[row, rpos]
                    delta = donor[row, dpos].to(live) - live
                    changed[row, rpos] = live + project(delta, q)
            return changed
        return hook

    for component in selected:
        block = backend.model.transformer.h[component.layer]
        if component.kind == "mlp":
            handles.append(block.mlp.register_forward_hook(mlp_hook(component)))
        else:
            handles.append(block.attn.c_proj.register_forward_pre_hook(head_hook(component)))
    try:
        return backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    splits, chosen, pools, programs, subspaces = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "program_lengths": {task: len(path) for task, path in programs.items()}, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    projections, projection_checks, ranks = {}, [], {}
    for task in ("has", "is"):
        basis = stored_basis(torch, subspaces["subspaces"][task]["basis"]).cuda()
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, HEADS, basis)
        mlp_q = row_basis(torch, read)
        projections[task], ranks[task] = {}, {}
        components = greedy.program_specs(programs[task], pools[task])
        for component in components:
            name = label(component)
            if component.kind == "mlp":
                q = mlp_q
            else:
                mapped = atlas.attention_writer_to_read_map(
                    model.transformer.h[component.layer].attn, component.heads[0], read)["contraction"]
                q = row_basis(torch, mapped)
            projections[task][name], ranks[task][name] = q, int(q.shape[1])
            identity = torch.eye(q.shape[1], device=q.device)
            projection_checks.append(float((q.T @ q - identity).abs().max()))
    records, metrics, all_capable, bank_widths = [], {"has": {}, "is": {}}, True, {}
    forwards = evaluations = 0
    for task in ("has", "is"):
        components = greedy.program_specs(programs[task], pools[task])
        for panel in ("heldout", "a2"):
            split, rows = f"{task}_{panel}", splits[f"{task}_{panel}"]
            base_batch = das._batch(backend, rows, side="base")
            donor_batch = das._batch(backend, rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            bank_widths[split] = sorted(set(map(len, banks)))
            all_capable = all_capable and capable(base_output) and capable(donor_output)
            outputs = {
                "full": positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, components, cache, banks, banks),
                "readable": patch_projected(backend, base_batch, donor_batch, components,
                    cache, banks, banks, projections[task], complement=False),
                "complement": patch_projected(backend, base_batch, donor_batch, components,
                    cache, banks, banks, projections[task], complement=True)}
            metrics[task][panel] = {}
            for arm, output in outputs.items():
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm), split, task)
                records.extend(arm_records)
                metrics[task][panel][arm] = source_groups.summarize(arm_records)
            forwards += 5
            evaluations += 5 * len(rows)
    price = {"model_forwards": forwards, "example_evaluations": evaluations,
             "records": len(records), "fitted_scalars": 0,
             "transformer_backwards": 0, "model_updates": 0}
    finite = all(math.isfinite(float(record["recovery"])) for record in records)
    pred_a = bool(all_capable and finite and max(projection_checks) <= 1e-5 and price == PRICE
                  and bank_widths == {"has_heldout": [3], "has_a2": [3],
                                      "is_heldout": [2], "is_a2": [2]})
    pred_b = all(metrics[task][panel]["readable"]["mean_recovery"]
                 >= 0.80 * metrics[task][panel]["full"]["mean_recovery"]
                 and metrics[task][panel]["readable"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_c = all(abs(metrics[task][panel]["complement"]["mean_recovery"])
                 <= 0.20 * abs(metrics[task][panel]["full"]["mean_recovery"])
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_d = all(rank <= (18 if task == "has" else 3) if name.startswith("MLP") else rank < 128
                 for task in ranks for name, rank in ranks[task].items())
    pred_e = True
    predictions = {
        "pred_a_authority_capability_projection_gauge_finiteness_and_exact_price": pred_a,
        "pred_b_weight_readable_modes_are_sufficient": pred_b,
        "pred_c_weight_complements_are_secondary": pred_c,
        "pred_d_weight_modes_are_dimensionally_compressive": pred_d,
        "pred_e_zero_causal_fit": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_weight_readable_mode_program_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "programs": programs, "projection_ranks": ranks,
              "maximum_projector_gram_error": max(projection_checks),
              "metrics": metrics, "bank_widths": bank_widths,
              "predictions": predictions, "price": price, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "projection_ranks",
        "maximum_projector_gram_error", "metrics", "predictions", "price", "terminal")},
        sort_keys=True))


if __name__ == "__main__":
    main()
