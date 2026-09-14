#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 77forwards770seq; fixed-pointer complete-module response screen;0fits.
"""A exact instrument; B live target; C singleton response; D distributed null."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


RUNNER = Path(__file__).resolve()
OPS = RUNNER.parent
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ROWS = POLY / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_ROWS.json"
PREREG = POLY / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_PREREGISTRATION.md"
PRIOR = ROOT / "basis_aligned/bilinear_quotient/circuits/prior_art/successor_fixed_pointer_module_response_v1.json"
BINDING = POLY / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_BINDING.json"
OUT = POLY / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")
FAMILIES = ("month", "digit")
MODULES = tuple(name for layer in range(18) for name in (f"attn{layer}", f"mlp{layer}"))
BARS = {
    "maximum_self_logit_absolute_error": 1e-5,
    "maximum_full_ceiling_logit_absolute_error": 1e-4,
    "minimum_native_accuracy": 0.75,
    "minimum_family_mean_late_backward_target": 1.0,
    "minimum_family_projection": 0.50,
    "minimum_family_cosine": 0.70,
    "maximum_control_rms_fraction": 0.50,
}
PRICE = {"forwards": 77, "sequences": 770, "fits": 0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {
        "rows": ROWS,
        "preregistration": PREREG,
        "prior_art": PRIOR,
        "correction": POLY / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V2_INSTRUMENT_CORRECTION.md",
        "v1_binding": POLY / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_BINDING.json",
        "v1_runner": OPS / "run_successor_fixed_pointer_module_response_v1.py",
        "v1_invalid_result": POLY / "SUCCESSOR_FIXED_POINTER_MODULE_RESPONSE_V1_RESULT.json",
    }
    for name, expected in binding["files"].items():
        if digest(paths[name]) != expected:
            raise RuntimeError(f"bound {name} changed")
    frozen = json.loads(ROWS.read_text())
    if binding["row_manifest_sha256"] != frozen["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed")
    if binding["bars"] != BARS or binding["price"] != PRICE:
        raise RuntimeError("bars or price changed")
    return binding, frozen


def aligned_rows(frozen):
    by_key = {}
    for row in frozen["rows"]:
        key = (row["family"], int(row["final_index"]))
        by_key.setdefault(key, {})[row["condition"]] = row
    keys = sorted(by_key)
    if len(keys) != 10 or any(set(by_key[key]) != set(CONDITIONS) for key in keys):
        raise RuntimeError("frozen rows are not a complete aligned 10x3 factorial")
    return keys, {condition: [by_key[key][condition] for key in keys] for condition in CONDITIONS}


def plan():
    _, frozen = load_bound()
    keys, _ = aligned_rows(frozen)
    return {
        "schema": "successor_fixed_pointer_module_response_v2_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "aligned_groups": len(keys),
        "candidates": list(MODULES),
        "price": PRICE,
        "bars": BARS,
        "predicates": [
            "pred_a_exact_instrument",
            "pred_b_live_target_and_capability",
            "pred_c_singleton_complete_module_response",
            "pred_d_distributed_or_unresolved_null",
        ],
    }


def margin(logits, rows, direction):
    return np.asarray([
        values[int(row["recipient_answer_id"])]
        - values[int(row["donors"][direction]["answer_id"])]
        for values, row in zip(logits, rows)
    ], dtype=np.float64)


def vector_metrics(rescue, target):
    denominator = float(np.dot(target, target))
    rescue_norm = float(np.linalg.norm(rescue))
    target_norm = float(np.linalg.norm(target))
    projection = float(np.dot(rescue, target) / denominator) if denominator else float("nan")
    cosine = float(np.dot(rescue, target) / (rescue_norm * target_norm)) if rescue_norm and target_norm else 0.0
    return {
        "projection": projection,
        "cosine": cosine,
        "residual_rms": float(np.sqrt(np.mean((rescue - target) ** 2))),
        "rescue_rms": float(np.sqrt(np.mean(rescue ** 2))),
        "target_rms": float(np.sqrt(np.mean(target ** 2))),
    }


def main():
    _, frozen = load_bound()
    keys, rows = aligned_rows(frozen)
    if REQUESTED_DRY:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    sys.path.insert(0, str(ROOT / "basis_aligned/qk_mdl/algo_tasks/successor"))
    import torch
    import successor_lib as sl
    from circuit_fast_screen_managed_runner import atomic_create_json

    torch.set_num_threads(2)
    model, cfg = sl.load_model()
    device = next(model.parameters()).device
    query = int(rows["coherent"][0]["query_position"])
    if any(int(row["query_position"]) != query for condition in CONDITIONS for row in rows[condition]):
        raise RuntimeError("query positions differ")
    tensors = {
        condition: torch.tensor([row["token_ids"] for row in rows[condition]], dtype=torch.long, device=device)
        for condition in CONDITIONS
    }
    counts = [0, 0]

    def counted_run(idx, **kwargs):
        counts[0] += 1
        counts[1] += len(idx)
        return sl.run(model, cfg, idx, **kwargs)
    started = time.perf_counter()
    try:
        native_full, caches = {}, {}
        for condition in CONDITIONS:
            native_full[condition], caches[condition] = counted_run(tensors[condition], collect=True)
        native = {condition: native_full[condition][:, query].float().cpu().numpy() for condition in CONDITIONS}

        def patches(source_condition, destination_condition, selected=None):
            head, mlp = {}, {}
            chosen = set(MODULES if selected is None else (selected,))
            for layer in range(18):
                attention_name = f"attn{layer}"
                if attention_name in chosen:
                    hybrid = caches[destination_condition][("h", layer)].clone()
                    hybrid[:, query] = caches[source_condition][("h", layer)][:, query]
                    for h in range(cfg["n_head"]):
                        head[(layer, h)] = hybrid[:, :, h]
                mlp_name = f"mlp{layer}"
                if mlp_name in chosen:
                    hybrid = caches[destination_condition][("m", layer)].clone()
                    hybrid[:, query] = caches[source_condition][("m", layer)][:, query]
                    mlp[layer] = hybrid
            return head, mlp

        self_head = {
            (layer, h): caches["late_swap_incoherent"][("h", layer)][:, :, h]
            for layer in range(18) for h in range(cfg["n_head"])
        }
        self_mlp = {layer: caches["late_swap_incoherent"][("m", layer)] for layer in range(18)}
        self_full, _ = counted_run(tensors["late_swap_incoherent"], patch_head=self_head, patch_mlp=self_mlp)
        full_head, full_mlp = patches("coherent", "late_swap_incoherent")
        ceiling_full, _ = counted_run(tensors["late_swap_incoherent"], patch_head=full_head, patch_mlp=full_mlp)
        self_logits = self_full[:, query].float().cpu().numpy()
        ceiling_logits = ceiling_full[:, query].float().cpu().numpy()

        candidate_logits = {}
        for module in MODULES:
            candidate_logits[module] = {}
            for destination in ("late_swap_incoherent", "early_swap_control"):
                head, mlp = patches("coherent", destination, module)
                values, _ = counted_run(tensors[destination], patch_head=head or None, patch_mlp=mlp or None)
                candidate_logits[module][destination] = values[:, query].float().cpu().numpy()
    finally:
        pass
    elapsed = time.perf_counter() - started

    self_error = float(np.max(np.abs(self_logits - native["late_swap_incoherent"])))
    ceiling_error = float(np.max(np.abs(ceiling_logits - native["coherent"])))
    capability = {}
    for family in FAMILIES:
        family_indices = [i for i, key in enumerate(keys) if key[0] == family]
        capability[family] = {}
        for condition in CONDITIONS:
            correct = [
                int(np.argmax(native[condition][i])) == int(rows[condition][i]["recipient_answer_id"])
                for i in family_indices
            ]
            capability[family][condition] = float(np.mean(correct))

    targets = {
        direction: margin(native["coherent"], rows["coherent"], direction)
        - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], direction)
        for direction in ("backward", "forward")
    }
    reports = {}
    eligible = []
    for module in MODULES:
        late_backward = (
            margin(candidate_logits[module]["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
            - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
        )
        late_forward = (
            margin(candidate_logits[module]["late_swap_incoherent"], rows["late_swap_incoherent"], "forward")
            - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "forward")
        )
        early_backward = (
            margin(candidate_logits[module]["early_swap_control"], rows["early_swap_control"], "backward")
            - margin(native["early_swap_control"], rows["early_swap_control"], "backward")
        )
        family_reports = {}
        passes = True
        for family in FAMILIES:
            ix = np.asarray([i for i, key in enumerate(keys) if key[0] == family])
            stats = vector_metrics(late_backward[ix], targets["backward"][ix])
            target_rms = stats["target_rms"]
            stats["late_forward_control_rms"] = float(np.sqrt(np.mean(late_forward[ix] ** 2)))
            stats["early_backward_control_rms"] = float(np.sqrt(np.mean(early_backward[ix] ** 2)))
            stats["late_forward_control_fraction"] = stats["late_forward_control_rms"] / target_rms
            stats["early_backward_control_fraction"] = stats["early_backward_control_rms"] / target_rms
            stats["passes"] = bool(
                stats["projection"] >= BARS["minimum_family_projection"]
                and stats["cosine"] >= BARS["minimum_family_cosine"]
                and stats["late_forward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
                and stats["early_backward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
            )
            passes &= stats["passes"]
            family_reports[family] = stats
        reports[module] = {"families": family_reports, "passes": bool(passes)}
        if passes:
            eligible.append(module)

    ranking = sorted(MODULES, key=lambda name: (
        -min(reports[name]["families"][family]["projection"] for family in FAMILIES),
        -min(reports[name]["families"][family]["cosine"] for family in FAMILIES),
        MODULES.index(name),
    ))
    pred_a = bool(
        self_error <= BARS["maximum_self_logit_absolute_error"]
        and ceiling_error <= BARS["maximum_full_ceiling_logit_absolute_error"]
        and counts == [PRICE["forwards"], PRICE["sequences"]]
    )
    family_target_means = {
        family: float(np.mean(targets["backward"][[i for i, key in enumerate(keys) if key[0] == family]]))
        for family in FAMILIES
    }
    pred_b = bool(
        all(value >= BARS["minimum_native_accuracy"] for cells in capability.values() for value in cells.values())
        and all(value >= BARS["minimum_family_mean_late_backward_target"] for value in family_target_means.values())
    )
    pred_c = bool(pred_a and pred_b and eligible)
    pred_d = bool(pred_a and pred_b and not eligible)
    terminal = "invalid" if not (pred_a and pred_b) else ("singleton_response" if pred_c else "distributed_or_unresolved")
    result = {
        "schema": "successor_fixed_pointer_module_response_v2_result",
        "terminal": terminal,
        "predictions": {
            "pred_a_exact_instrument": pred_a,
            "pred_b_live_target_and_capability": pred_b,
            "pred_c_singleton_complete_module_response": pred_c,
            "pred_d_distributed_or_unresolved_null": pred_d,
        },
        "bars": BARS,
        "price": {**PRICE, "observed_forwards": counts[0], "observed_sequences": counts[1], "elapsed_seconds": elapsed},
        "instrument": {"self_max_logit_error": self_error, "full_ceiling_max_logit_error": ceiling_error},
        "capability": capability,
        "family_mean_late_backward_target": family_target_means,
        "eligible_modules": eligible,
        "ranking": list(ranking),
        "candidate_reports": reports,
        "row_manifest_sha256": frozen["row_manifest_sha256"],
        "runner_sha256": digest(RUNNER),
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
