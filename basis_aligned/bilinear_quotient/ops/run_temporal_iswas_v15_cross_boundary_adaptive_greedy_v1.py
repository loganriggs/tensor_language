#!/usr/bin/env python3
"""Adaptive selective composition of the v15 four-head core and live modules."""
# BQGATE: EXPERIMENT pred_a_authority_population_closure_finiteness_and_exact_price pred_b_discovery_selective_prefix_exists pred_c_selected_prefix_confirms_without_reselection pred_d_selected_program_is_smaller_than_candidate_pool pred_e_selected_program_has_multiple_necessary_components pred_f_response_operator_agrees_with_behavior
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_all_layer_complete_module_response_atlas_v1 as module_impl


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_cross_boundary_adaptive_greedy_v1.json"
MODULE_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_all_layer_complete_module_response_atlas_v1_result.json"
HEAD_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1_result.json"
HEAD_RUNNER = ROOT / "ops/run_temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_cross_boundary_adaptive_greedy_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_cross_boundary_adaptive_greedy_v1"
EXPECTED = {
    "prior": "14b437da71ff3ba432da5d193656b7a20501680b477e2ad25df0cc68770dd75c",
    "module_result": "b04b049407b9a92e585550951e07ec92374a07fff70bdce8a61aba4613663468",
    "head_result": "cf2427059f698eac94a9e733ff873e1892a49882f60bc587c3ac83c681507025",
    "head_runner": "eb89c0948b03f9df15ce35cdbf07c8483c87a3404b8dd36efb6ebc1a873b2282",
}
HEADS = ("L8H1", "L9H1", "L9H4", "L11H3")
MODULES = (
    "attn:2", "attn:15", "mlp:1", "mlp:2", "mlp:3", "mlp:4", "mlp:6",
    "mlp:7", "mlp:8", "mlp:9", "mlp:10", "mlp:11", "mlp:17",
)
CANDIDATES = HEADS + MODULES
TARGET_PANELS = ("A1", "A2")
CONTROL_PANELS = ("P", "C")
EXACT_FORWARDS = 174


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_head(site: str) -> tuple[int, int]:
    layer, head = site[1:].split("H")
    return int(layer), int(head)


def site_module(backend, site: str):
    kind, layer_text = site.split(":")
    block = backend.model.transformer.h[int(layer_text)]
    return block.attn.c_proj if kind == "attn" else block.mlp


def capture(backend, batch):
    cache, handles = {}, []
    head_layers = sorted({parse_head(site)[0] for site in HEADS})
    for layer in head_layers:
        def save_head(_module, arguments, layer=layer):
            cache[f"head_layer:{layer}"] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save_head))
    for site in MODULES:
        if site.startswith("attn:"):
            def save_attn(_module, arguments, site=site):
                cache[site] = arguments[0].detach().clone()
            handles.append(site_module(backend, site).register_forward_pre_hook(save_attn))
        else:
            def save_mlp(_module, _arguments, output, site=site):
                cache[site] = output.detach().clone()
            handles.append(site_module(backend, site).register_forward_hook(save_mlp))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    expected = {f"head_layer:{layer}" for layer in head_layers} | set(MODULES)
    if set(cache) != expected:
        raise RuntimeError(f"incomplete candidate capture: {sorted(expected - set(cache))}")
    return output, cache


def prefix_patch(batch, values, slices=None):
    def hook(_module, arguments):
        changed = arguments[0].clone()
        for index, query in enumerate(batch.semantic_positions):
            stop = int(query) + 1
            if slices is None:
                changed[index, :stop] = values[index, :stop].to(changed)
            else:
                for left, right in slices:
                    changed[index, :stop, left:right] = values[index, :stop, left:right].to(changed)
        return (changed,) + tuple(arguments[1:])
    return hook


def output_patch(batch, values):
    def hook(_module, _arguments, output):
        changed = output.clone()
        for index, query in enumerate(batch.semantic_positions):
            stop = int(query) + 1
            changed[index, :stop] = values[index, :stop].to(changed)
        return changed
    return hook


def run_patch(backend, batch, cache, support):
    support = set(support)
    handles = []
    width = int(backend.model.config.n_embd // backend.model.config.n_head)
    head_groups = {}
    for site in HEADS:
        if site in support:
            layer, head = parse_head(site)
            head_groups.setdefault(layer, []).append((head * width, (head + 1) * width))
    for layer, slices in head_groups.items():
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
            prefix_patch(batch, cache[f"head_layer:{layer}"], slices)))
    for site in MODULES:
        if site not in support:
            continue
        if site.startswith("attn:"):
            handles.append(site_module(backend, site).register_forward_pre_hook(
                prefix_patch(batch, cache[site])))
        else:
            handles.append(site_module(backend, site).register_forward_hook(
                output_patch(batch, cache[site])))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main():
    paths = {"prior": PRIOR, "module_result": MODULE_RESULT, "head_result": HEAD_RESULT,
             "head_runner": HEAD_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    module_result = json.loads(MODULE_RESULT.read_text())
    head_result = json.loads(HEAD_RESULT.read_text())
    rows = fresh.build_rows()
    panel_counts = {panel: sum(row["transform_id"] == panel for row in rows)
                    for panel in TARGET_PANELS + CONTROL_PANELS}
    derived_heads = tuple(site for site, report in head_result["singleton_reports"].items()
                          if abs(report["behavior"]["signed_projection"]) >= .10
                          and report["behavior"]["direction_fraction"] >= .75)
    derived_modules = tuple(site for site, report in module_result["singleton_reports"].items()
                            if site not in {"input_residual", "attn:8", "attn:9", "attn:11"}
                            and abs(report["behavior"]["signed_projection"]) >= .10)
    authority_ok = bool(observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                        and set(derived_heads) == set(HEADS) and set(derived_modules) == set(MODULES)
                        and panel_counts == {panel: 16 for panel in TARGET_PANELS + CONTROL_PANELS}
                        and len(rows) == len({row["row_id"] for row in rows}) == 64)
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "candidates": list(CANDIDATES), "greedy_extension_arms": 153,
        "toggle_necessity_arms": 17, "model_forwards_exact": EXACT_FORWARDS,
        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    native = backend.native
    forwards = 0

    def counted(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return native(*args, **kwargs)

    backend.native = counted
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_cache = capture(backend, base_batch)
    donor_output, donor_cache = capture(backend, donor_batch)
    aligned = set(base_cache) == set(donor_cache) and all(
        base_cache[key].shape == donor_cache[key].shape for key in base_cache)
    if not aligned:
        raise RuntimeError("base/donor candidate response shapes changed")
    base_state = module_impl.states(torch, backend, base_output, rows)
    donor_state = module_impl.states(torch, backend, donor_output, rows)
    base_logits = das.head_logits(backend, base_state).float()
    donor_logits = das.head_logits(backend, donor_state).float()
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)

    def margins(logits):
        return logits[all_index, answer] - logits[all_index, foil]

    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state
    panel_indices = {
        panel: torch.as_tensor([i for i, row in enumerate(rows) if row["transform_id"] == panel],
                               device=backend.device)
        for panel in TARGET_PANELS + CONTROL_PANELS
    }
    control_ids = [i for i, row in enumerate(rows) if row["transform_id"] in CONTROL_PANELS]
    control_index = torch.as_tensor(control_ids, device=backend.device)

    def report(output):
        state = module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets = {}
        for panel in TARGET_PANELS:
            ix = panel_indices[panel]
            behavior = module_impl.vector_metrics(torch, delta_margin[ix], target_margin[ix])
            behavior["direction_fraction"] = float(
                ((delta_margin[ix] * target_margin[ix]) > 0).float().mean())
            targets[panel] = {
                "behavior": behavior,
                "final_residual": module_impl.vector_metrics(torch, delta_state[ix], target_state[ix]),
            }
        lb, lp = base_logits[control_index], logits[control_index]
        logb, logp = F.log_softmax(lb, -1), F.log_softmax(lp, -1)
        kl = (logb.exp() * (logb - logp)).sum(-1)
        flips = lb.argmax(-1) != lp.argmax(-1)
        controls = {
            "median_kl": float(kl.median()), "max_kl": float(kl.max()),
            "top1_flip_fraction": float(flips.float().mean()),
            "top1_flip_count": int(flips.sum()),
            "flipped_row_ids": [rows[control_ids[i]]["row_id"] for i in range(len(control_ids)) if bool(flips[i])],
        }
        return {"targets": targets, "controls": controls}

    def eligible(report_value):
        a1, controls = report_value["targets"]["A1"]["behavior"], report_value["controls"]
        return bool(a1["signed_projection"] >= .80 and a1["direction_fraction"] >= .875
                    and controls["top1_flip_count"] == 0 and controls["median_kl"] <= .05)

    self_output = run_patch(backend, base_batch, base_cache, CANDIDATES)
    full_output = run_patch(backend, base_batch, donor_cache, CANDIDATES)
    self_state = module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(float((self_state - base_state).abs().max()),
                     float((self_logits - base_logits).abs().max()))
    full_report = report(full_output)

    current = ()
    path, stages = [], []
    all_reports = []
    for stage_number in range(len(CANDIDATES)):
        candidates = []
        for candidate in CANDIDATES:
            if candidate in current:
                continue
            support = tuple(site for site in CANDIDATES if site in set(current) | {candidate})
            value = report(run_patch(backend, base_batch, donor_cache, support))
            selective = value["controls"]["top1_flip_count"] == 0 and value["controls"]["median_kl"] <= .05
            item = {"added": candidate, "support": list(support), "selective": selective, "report": value}
            candidates.append(item)
            all_reports.append(value)
        selective_candidates = [item for item in candidates if item["selective"]]
        pool = selective_candidates or candidates
        chosen = min(pool, key=lambda item: (
            -item["report"]["targets"]["A1"]["behavior"]["signed_projection"],
            CANDIDATES.index(item["added"]),
        ))
        current = tuple(chosen["support"])
        stages.append({"stage": stage_number + 1, "selectivity_fallback": not bool(selective_candidates),
                       "candidates": candidates, "chosen": chosen["added"]})
        path.append({"stage": stage_number + 1, "added": chosen["added"],
                     "support": list(current), "eligible": eligible(chosen["report"]),
                     "report": chosen["report"]})

    selected_step = next((step for step in path if step["eligible"]), None)
    selected_support = tuple(selected_step["support"]) if selected_step else ()
    selected_report = selected_step["report"] if selected_step else None
    toggles, necessary = {}, []
    for candidate in CANDIDATES:
        support = tuple(site for site in selected_support if site != candidate)
        value = report(run_patch(backend, base_batch, donor_cache, support))
        is_selected = candidate in selected_support
        a1_drop = (selected_report["targets"]["A1"]["behavior"]["signed_projection"]
                   - value["targets"]["A1"]["behavior"]["signed_projection"]) if selected_report else 0.0
        material = bool(is_selected and (a1_drop >= .05 or not eligible(value)))
        toggles[candidate] = {"candidate_was_selected": is_selected, "support": list(support),
                              "a1_projection_drop": a1_drop, "materially_necessary": material,
                              "report": value}
        if material:
            necessary.append(candidate)
        all_reports.append(value)

    no_op_error = 0.0
    if selected_report is not None:
        for candidate, value in toggles.items():
            if value["candidate_was_selected"]:
                continue
            for panel in TARGET_PANELS:
                no_op_error = max(no_op_error, abs(
                    value["report"]["targets"][panel]["behavior"]["signed_projection"]
                    - selected_report["targets"][panel]["behavior"]["signed_projection"]))
            no_op_error = max(no_op_error, abs(value["report"]["controls"]["median_kl"]
                                                - selected_report["controls"]["median_kl"]))
    pred_a = bool(authority_ok and aligned and self_error <= 1e-4 and no_op_error <= 1e-7
                  and finite([full_report, all_reports, path, stages, toggles])
                  and forwards == EXACT_FORWARDS)
    pred_b = selected_report is not None
    pred_c = bool(selected_report and selected_report["targets"]["A2"]["behavior"]["signed_projection"] >= .75
                  and selected_report["targets"]["A2"]["behavior"]["direction_fraction"] >= .875)
    pred_d = bool(selected_report and len(selected_support) <= 10)
    pred_e = len(necessary) >= 2
    pred_f = bool(selected_report and all(
        selected_report["targets"][panel]["final_residual"]["signed_projection"] >= .50
        for panel in TARGET_PANELS))
    predictions = {
        "pred_a_authority_population_closure_finiteness_and_exact_price": pred_a,
        "pred_b_discovery_selective_prefix_exists": pred_b,
        "pred_c_selected_prefix_confirms_without_reselection": pred_c,
        "pred_d_selected_program_is_smaller_than_candidate_pool": pred_d,
        "pred_e_selected_program_has_multiple_necessary_components": pred_e,
        "pred_f_response_operator_agrees_with_behavior": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "selective_cross_boundary_circuit"
    elif all(predictions[key] for key in list(predictions)[:5]) and not pred_f:
        terminal = "behavioral_circuit_response_mismatch"
    elif pred_b and not pred_c:
        terminal = "confirmation_failure"
    elif not pred_b or not pred_d:
        terminal = "no_small_selective_prefix"
    elif pred_b and pred_c and pred_d and not pred_e:
        terminal = "nonnecessary_greedy_set"
    else:
        terminal = "partial"
    result = {
        "schema": "temporal_iswas_v15_cross_boundary_adaptive_greedy_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "population": {"panel_counts": panel_counts, "row_ids": [row["row_id"] for row in rows]},
        "candidates": list(CANDIDATES), "instrument": {"aligned_shapes": aligned,
            "base_self_max_abs_error": self_error, "nonselected_toggle_no_op_max_error": no_op_error},
        "full_candidate_report": full_report, "path": path, "stages": stages,
        "selected_support": list(selected_support), "selected_report": selected_report,
        "toggle_reports": toggles, "materially_necessary": necessary,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_exact": EXACT_FORWARDS, "model_forwards_observed": forwards,
                  "example_evaluations": forwards * len(rows), "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0},
        "dryrun": dryrun,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "selected_support", "selected_report", "materially_necessary", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
