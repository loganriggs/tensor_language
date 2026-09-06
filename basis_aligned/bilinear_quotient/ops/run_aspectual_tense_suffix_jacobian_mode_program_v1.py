#!/usr/bin/env python3
"""Fit-only suffix-Jacobian subspaces for validated carrier programs."""

# BQGATE: EXPERIMENT pred_a_authority_gradient_replay_capability_projection_finiteness_and_exact_price pred_b_suffix_jacobian_modes_are_sufficient pred_c_suffix_jacobian_complements_are_secondary pred_d_gradient_modes_are_dimensionally_compressive pred_e_no_causal_fit_or_model_update
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
import run_aspectual_tense_weight_readable_mode_program_v1 as static_mode


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_suffix_jacobian_mode_program_v1.json"
STATIC_RESULT = ROOT / "circuits/followups/aspectual_tense_weight_readable_mode_program_v1_result.json"
STATIC_RUNNER = ROOT / "ops/run_aspectual_tense_weight_readable_mode_program_v1.py"
PROGRAMS = ROOT / "circuits/followups/aspectual_tense_carrier_program_backward_pruning_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_tense_suffix_jacobian_mode_program_v1_result.json"
CANDIDATE_ID = "aspectual_tense.suffix_jacobian_mode_program_v1"
EXPECTED = {
    "prior": "adde1db7d3aa648235cd4970764ce3aa3f95a766e49a6a728835040c6498394a",
    "static_result": "a49f9d6b06cb37a1759e449b72fb1446b93addae80fda521c0bd6d9af61bed49",
    "static_runner": "e72ec560692b2924c506f592ce83037f353b0bebc5cfa1e3705f1c2aab1a1c8f",
    "programs": "bb6344f6446a5426a9b6342c30cbcd56ca821a01b5750e1ef3b940a6b52e15c0",
    "greedy": "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413",
    "source_rank": "c7570a2e25b444df84e40953e38d6bbc4b7b15c6d6f6657fda0696fb4eea3d34",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
    "source_groups": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
AUTHORITY_PATHS = {
    "greedy": ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py",
    "source_rank": ROOT / "ops/run_aspectual_tense_l9h1h4_source_position_weight_validation_v1.py",
    "positioned": ROOT / "ops/positioned_component_program_eval.py",
    "source_groups": ROOT / "ops/attention_source_group_eval.py",
    "das": ROOT / "ops/circuit_das_subspace.py",
    "producer": ROOT / "ops/circuit_fast_screen_producer.py",
}
ENERGY = 0.95
PRICE = {"model_forwards": 26, "example_evaluations": 407, "records": 201,
         "transformer_backwards": 2, "model_updates": 0}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "static_result": STATIC_RESULT, "static_runner": STATIC_RUNNER,
             "programs": PROGRAMS, **AUTHORITY_PATHS}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    static, programs = [json.loads(path.read_text()) for path in (STATIC_RESULT, PROGRAMS)]
    splits, chosen, pools = greedy.validate_static()
    if (static.get("terminal") != "null" or programs.get("terminal") != "screen"
            or {task: len(path) for task, path in programs["pruned_paths"].items()}
            != {"has": 7, "is": 10}):
        raise ExperimentError("static null or pruned program authority changed")
    return static, splits, chosen, pools, programs["pruned_paths"]


def gradient_forward(backend, batch, components, banks):
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured, handles = {}, []

    def save(name):
        def hook(_module, _arguments, output):
            output.retain_grad()
            captured[name] = output
        return hook

    def save_input(name):
        def hook(_module, arguments):
            arguments[0].retain_grad()
            captured[name] = arguments[0]
        return hook

    for component in components:
        block, name = model.transformer.h[component.layer], static_mode.label(component)
        if component.kind == "mlp":
            handles.append(block.mlp.register_forward_hook(save(name)))
        else:
            handles.append(block.attn.c_proj.register_forward_pre_hook(save_input(name)))
    try:
        with torch.enable_grad():
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
            x, x0, v1 = x.detach().requires_grad_(True), None, None
            x0 = x
            for block in model.transformer.h:
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
            row = torch.arange(len(lengths), device=backend.device)
            last = torch.tensor([length - 1 for length in lengths], device=backend.device)
            answer = torch.tensor(batch.answer_ids, device=backend.device)
            foil = torch.tensor(batch.foil_ids, device=backend.device)
            answer_values, foil_values = logits[row, last, answer], logits[row, last, foil]
            (answer_values - foil_values).sum().backward()
        values = tuple((float(answer_values[i].detach()), float(foil_values[i].detach()))
                       for i in range(len(lengths)))
        gradients = {}
        for component in components:
            name, tensor = static_mode.label(component), captured[static_mode.label(component)]
            selected = []
            for row_index, bank in enumerate(banks):
                for position in bank:
                    value = tensor.grad[row_index, position].float()
                    if component.kind != "mlp":
                        width = value.shape[0] // model.config.n_head
                        head = component.heads[0]
                        value = value[head * width:(head + 1) * width]
                    selected.append(value)
            gradients[name] = torch.stack(selected)
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.zero_grad(set_to_none=True)
    return producer.BatchOutput(values, {}), gradients


def energy_basis(torch, gradients):
    _u, singular, vh = torch.linalg.svd(gradients.float(), full_matrices=False)
    energy = singular.square()
    if not energy.numel() or float(energy.sum()) <= 0:
        raise ExperimentError("gradient span is empty")
    fraction = energy.cumsum(0) / energy.sum()
    rank = int(torch.searchsorted(fraction, torch.tensor(ENERGY, device=fraction.device)).item()) + 1
    q = torch.linalg.qr(vh[:rank].T.double()).Q.float().to(gradients.device)
    return q, singular, float(fraction[rank - 1])


def capable_tensor(output):
    return bool(((output[:, 0] - output[:, 1]) > 0).all())


def capable_output(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    static, splits, chosen, pools, programs = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "gradient_energy": ENERGY,
              "fit_counts": {"has": 16, "is": 8}, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    projections, basis_reports, replay_errors = {}, {}, []
    all_capable = True
    forwards = evaluations = backwards = 0
    for task in ("has", "is"):
        rows, components = splits[f"{task}_fit"], greedy.program_specs(programs[task], pools[task])
        base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        native = backend.native(base_batch, capture=False)
        gradient_output, gradients = gradient_forward(backend, base_batch, components, banks)
        donor_output, _cache = positioned.capture_full_components(
            backend, donor_batch, source_rank.capture_specs(chosen[task]))
        native_tensor = torch.tensor(native.answer_foil, device=backend.device)
        gradient_tensor = torch.tensor(gradient_output.answer_foil, device=backend.device)
        replay_errors.append(float((native_tensor - gradient_tensor).abs().max()))
        all_capable = all_capable and capable_output(native) and capable_tensor(gradient_tensor) and capable_output(donor_output)
        projections[task], basis_reports[task] = {}, {}
        for name, values in gradients.items():
            q, singular, captured_energy = energy_basis(torch, values)
            projections[task][name] = q
            basis_reports[task][name] = {"rank": int(q.shape[1]), "width": int(q.shape[0]),
                "fit_gradient_rows": int(values.shape[0]), "captured_energy": captured_energy,
                "singular_values": [float(value) for value in singular.detach().cpu()],
                "gram_max_abs_error": float((q.T @ q - torch.eye(q.shape[1], device=q.device)).abs().max())}
        forwards += 3
        evaluations += 3 * len(rows)
        backwards += 1
    records, metrics, bank_widths = [], {"has": {}, "is": {}}, {}
    for task in ("has", "is"):
        components = greedy.program_specs(programs[task], pools[task])
        for panel in ("heldout", "a2"):
            split, rows = f"{task}_{panel}", splits[f"{task}_{panel}"]
            base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            bank_widths[split] = sorted(set(map(len, banks)))
            all_capable = all_capable and capable_output(base_output) and capable_output(donor_output)
            outputs = {
                "full": positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, components, cache, banks, banks),
                "jacobian": static_mode.patch_projected(backend, base_batch, donor_batch,
                    components, cache, banks, banks, projections[task], complement=False),
                "complement": static_mode.patch_projected(backend, base_batch, donor_batch,
                    components, cache, banks, banks, projections[task], complement=True)}
            metrics[task][panel] = {}
            for arm, output in outputs.items():
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm), split, task)
                records.extend(arm_records)
                metrics[task][panel][arm] = source_groups.summarize(arm_records)
            forwards += 5
            evaluations += 5 * len(rows)
    price = {"model_forwards": forwards, "example_evaluations": evaluations,
             "records": len(records), "transformer_backwards": backwards, "model_updates": 0}
    reports = [report for task in basis_reports.values() for report in task.values()]
    finite = all(math.isfinite(float(record["recovery"])) for record in records) and all(
        math.isfinite(value) for report in reports for value in report["singular_values"])
    pred_a = bool(max(replay_errors) <= 1e-4 and all_capable and finite and price == PRICE
                  and max(report["gram_max_abs_error"] for report in reports) <= 1e-5
                  and bank_widths == {"has_heldout": [3], "has_a2": [3],
                                      "is_heldout": [2], "is_a2": [2]})
    pred_b = all(metrics[task][panel]["jacobian"]["mean_recovery"]
                 >= 0.90 * metrics[task][panel]["full"]["mean_recovery"]
                 and metrics[task][panel]["jacobian"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_c = all(abs(metrics[task][panel]["complement"]["mean_recovery"])
                 <= 0.20 * abs(metrics[task][panel]["full"]["mean_recovery"])
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_d = all(report["rank"] < report["width"] and report["rank"] <= (48 if task == "has" else 16)
                 for task, task_reports in basis_reports.items() for report in task_reports.values())
    pred_e = backwards == 2 and price["model_updates"] == 0
    predictions = {
        "pred_a_authority_gradient_replay_capability_projection_finiteness_and_exact_price": pred_a,
        "pred_b_suffix_jacobian_modes_are_sufficient": pred_b,
        "pred_c_suffix_jacobian_complements_are_secondary": pred_c,
        "pred_d_gradient_modes_are_dimensionally_compressive": pred_d,
        "pred_e_no_causal_fit_or_model_update": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_suffix_jacobian_mode_program_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "gradient_energy": ENERGY, "programs": programs, "basis_reports": basis_reports,
              "gradient_forward_replay_max_abs_error": max(replay_errors),
              "metrics": metrics, "bank_widths": bank_widths,
              "predictions": predictions, "price": price, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "basis_reports",
        "gradient_forward_replay_max_abs_error", "metrics", "predictions", "price", "terminal")},
        sort_keys=True))


if __name__ == "__main__":
    main()
