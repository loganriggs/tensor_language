#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 11forwards88seq; successor pointer cross-type interaction;0fits.
"""A instrument/live joint; B additive union; C cross-type interaction; D mixed."""
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
ROWS = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json"
PREREG = POLY / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_PREREGISTRATION.md"
CONFIRMATION = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_RESULT.json"
BINDING = POLY / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_BINDING.json"
OUT = POLY / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")
FAMILIES = ("month", "digit")
MODULES = tuple(name for layer in range(18) for name in (f"attn{layer}", f"mlp{layer}"))
GROUPS = {
    "A": tuple(f"attn{x}" for x in range(8, 13)),
    "M": tuple(f"mlp{x}" for x in range(8, 13)),
    "AM": tuple(name for x in range(8, 13) for name in (f"attn{x}", f"mlp{x}")),
}
BARS = {
    "maximum_self_logit_absolute_error": 1e-5,
    "maximum_full_ceiling_logit_absolute_error": 1e-4,
    "minimum_native_accuracy": 0.75,
    "minimum_family_mean_late_backward_target": 1.0,
    "minimum_family_projection": 0.50,
    "minimum_family_cosine": 0.70,
    "maximum_control_rms_fraction": 0.50,
    "maximum_additive_residual_interaction_fraction": 0.10,
    "minimum_additive_pre_score_projection": 0.90,
    "minimum_additive_pre_score_cosine": 0.95,
    "minimum_cross_type_residual_interaction_fraction": 0.20,
    "minimum_cross_type_pre_score_target_projection": 0.10,
}
PRICE = {"forwards": 11, "sequences": 88, "fits": 0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {
        "rows": ROWS,
        "preregistration": PREREG,
        "confirmation_result": CONFIRMATION,
    }
    for name, expected in binding["files"].items():
        if digest(paths[name]) != expected:
            raise RuntimeError(f"bound {name} changed")
    frozen = json.loads(ROWS.read_text())
    if binding["row_manifest_sha256"] != frozen["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed")
    if binding["bars"] != BARS or binding["price"] != PRICE:
        raise RuntimeError("bars or price changed")
    if binding["groups"] != {name: list(members) for name, members in GROUPS.items()}:
        raise RuntimeError("candidate groups changed")
    return binding, frozen


def aligned_rows(frozen):
    by_key = {}
    for row in frozen["rows"]:
        key = (row["family"], int(row["final_index"]))
        by_key.setdefault(key, {})[row["condition"]] = row
    keys = sorted(by_key)
    if len(keys) != 8 or any(set(by_key[key]) != set(CONDITIONS) for key in keys):
        raise RuntimeError("frozen rows are not a complete aligned 8x3 factorial")
    return keys, {condition: [by_key[key][condition] for key in keys] for condition in CONDITIONS}


def plan():
    _, frozen = load_bound()
    keys, _ = aligned_rows(frozen)
    return {
        "schema": "successor_pointer_cross_type_interaction_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "aligned_groups": len(keys),
        "candidates": {name: list(members) for name, members in GROUPS.items()},
        "price": PRICE,
        "bars": BARS,
        "predicates": [
            "pred_a_instrument_and_live_joint",
            "pred_b_additive_union",
            "pred_c_cross_type_interaction",
            "pred_d_mixed_or_unresolved",
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
        native_residual = {
            condition: caches[condition][("r", 17)][:, query].float().cpu().numpy()
            for condition in CONDITIONS
        }
        native_pre_score = {
            condition: model.lm_head(torch.nn.functional.rms_norm(
                caches[condition][("r", 17)][:, query], (cfg["n_embd"],)
            )).float().cpu().numpy()
            for condition in CONDITIONS
        }

        def patches(source_condition, destination_condition, selected=None):
            head, mlp = {}, {}
            chosen = set(MODULES if selected is None else selected)
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

        candidate_logits, candidate_residual, candidate_pre_score = {}, {}, {}
        for module in GROUPS:
            candidate_logits[module] = {}
            candidate_residual[module] = {}
            candidate_pre_score[module] = {}
            for destination in ("late_swap_incoherent", "early_swap_control"):
                head, mlp = patches("coherent", destination, GROUPS[module])
                values, arm_cache = counted_run(
                    tensors[destination], patch_head=head or None, patch_mlp=mlp or None, collect=True
                )
                candidate_logits[module][destination] = values[:, query].float().cpu().numpy()
                final_residual = arm_cache[("r", 17)][:, query]
                candidate_residual[module][destination] = final_residual.float().cpu().numpy()
                candidate_pre_score[module][destination] = model.lm_head(torch.nn.functional.rms_norm(
                    final_residual, (cfg["n_embd"],)
                )).float().cpu().numpy()
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
    def effects(store, destination, direction):
        base = native_pre_score if store is candidate_pre_score else native
        return {
            name: margin(store[name][destination], rows[destination], direction)
            - margin(base[destination], rows[destination], direction)
            for name in GROUPS
        }

    post_late_backward = effects(candidate_logits, "late_swap_incoherent", "backward")
    post_late_forward = effects(candidate_logits, "late_swap_incoherent", "forward")
    post_early_backward = effects(candidate_logits, "early_swap_control", "backward")
    pre_targets = {
        direction: margin(native_pre_score["coherent"], rows["coherent"], direction)
        - margin(native_pre_score["late_swap_incoherent"], rows["late_swap_incoherent"], direction)
        for direction in ("backward", "forward")
    }
    pre_late_backward = effects(candidate_pre_score, "late_swap_incoherent", "backward")
    reports = {}
    joint_still_live = True
    additive_pass = True
    interaction_pass = True
    for family in FAMILIES:
        ix = np.asarray([i for i, key in enumerate(keys) if key[0] == family])
        target = targets["backward"][ix]
        target_rms = float(np.sqrt(np.mean(target ** 2)))
        component = {
            name: vector_metrics(post_late_backward[name][ix], target)
            for name in GROUPS
        }
        for name in GROUPS:
            component[name]["late_forward_control_fraction"] = float(
                np.sqrt(np.mean(post_late_forward[name][ix] ** 2)) / target_rms
            )
            component[name]["early_backward_control_fraction"] = float(
                np.sqrt(np.mean(post_early_backward[name][ix] ** 2)) / target_rms
            )
        joint_pass = bool(
            component["AM"]["projection"] >= BARS["minimum_family_projection"]
            and component["AM"]["cosine"] >= BARS["minimum_family_cosine"]
            and component["AM"]["late_forward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
            and component["AM"]["early_backward_control_fraction"] <= BARS["maximum_control_rms_fraction"]
        )
        joint_still_live &= joint_pass

        post_add = post_late_backward["A"][ix] + post_late_backward["M"][ix]
        post_interaction = post_late_backward["AM"][ix] - post_add
        pre_add = pre_late_backward["A"][ix] + pre_late_backward["M"][ix]
        pre_interaction = pre_late_backward["AM"][ix] - pre_add
        late_forward_interaction = (
            post_late_forward["AM"][ix] - post_late_forward["A"][ix] - post_late_forward["M"][ix]
        )
        early_backward_interaction = (
            post_early_backward["AM"][ix] - post_early_backward["A"][ix] - post_early_backward["M"][ix]
        )
        residual_effects = {
            name: candidate_residual[name]["late_swap_incoherent"][ix]
            - native_residual["late_swap_incoherent"][ix]
            for name in GROUPS
        }
        residual_interaction = residual_effects["AM"] - residual_effects["A"] - residual_effects["M"]
        residual_joint_rms = float(np.sqrt(np.mean(residual_effects["AM"] ** 2)))
        residual_interaction_rms = float(np.sqrt(np.mean(residual_interaction ** 2)))
        residual_fraction = residual_interaction_rms / residual_joint_rms
        pre_additive = vector_metrics(pre_add, pre_late_backward["AM"][ix])
        pre_interaction_target = vector_metrics(pre_interaction, pre_targets["backward"][ix])
        additive_family = bool(
            residual_fraction <= BARS["maximum_additive_residual_interaction_fraction"]
            and pre_additive["projection"] >= BARS["minimum_additive_pre_score_projection"]
            and pre_additive["cosine"] >= BARS["minimum_additive_pre_score_cosine"]
        )
        interaction_family = bool(
            residual_fraction >= BARS["minimum_cross_type_residual_interaction_fraction"]
            and abs(pre_interaction_target["projection"]) >= BARS["minimum_cross_type_pre_score_target_projection"]
            and float(np.sqrt(np.mean(late_forward_interaction ** 2)) / target_rms) <= BARS["maximum_control_rms_fraction"]
            and float(np.sqrt(np.mean(early_backward_interaction ** 2)) / target_rms) <= BARS["maximum_control_rms_fraction"]
        )
        additive_pass &= additive_family
        interaction_pass &= interaction_family
        reports[family] = {
            "component_post_softcap": component,
            "joint_confirmation_passes": joint_pass,
            "post_softcap_additive_vs_joint": vector_metrics(post_add, post_late_backward["AM"][ix]),
            "post_softcap_interaction_vs_target": vector_metrics(post_interaction, target),
            "pre_softcap_additive_vs_joint": pre_additive,
            "pre_softcap_interaction_vs_target": pre_interaction_target,
            "final_residual_joint_rms": residual_joint_rms,
            "final_residual_interaction_rms": residual_interaction_rms,
            "final_residual_interaction_fraction": residual_fraction,
            "interaction_late_forward_control_fraction": float(np.sqrt(np.mean(late_forward_interaction ** 2)) / target_rms),
            "interaction_early_backward_control_fraction": float(np.sqrt(np.mean(early_backward_interaction ** 2)) / target_rms),
            "additive_family_passes": additive_family,
            "interaction_family_passes": interaction_family,
        }

    family_target_means = {
        family: float(np.mean(targets["backward"][[i for i, key in enumerate(keys) if key[0] == family]]))
        for family in FAMILIES
    }
    pred_a = bool(
        self_error <= BARS["maximum_self_logit_absolute_error"]
        and ceiling_error <= BARS["maximum_full_ceiling_logit_absolute_error"]
        and counts == [PRICE["forwards"], PRICE["sequences"]]
        and all(value >= BARS["minimum_native_accuracy"] for cells in capability.values() for value in cells.values())
        and all(value >= BARS["minimum_family_mean_late_backward_target"] for value in family_target_means.values())
        and joint_still_live
    )
    pred_b = bool(pred_a and additive_pass)
    pred_c = bool(pred_a and interaction_pass and not pred_b)
    pred_d = bool(pred_a and not pred_b and not pred_c)
    terminal = "invalid" if not pred_a else ("additive_union" if pred_b else ("cross_type_interaction" if pred_c else "mixed_or_unresolved"))
    result = {
        "schema": "successor_pointer_cross_type_interaction_v1_result",
        "terminal": terminal,
        "predictions": {
            "pred_a_instrument_and_live_joint": pred_a,
            "pred_b_additive_union": pred_b,
            "pred_c_cross_type_interaction": pred_c,
            "pred_d_mixed_or_unresolved": pred_d,
        },
        "bars": BARS,
        "price": {**PRICE, "observed_forwards": counts[0], "observed_sequences": counts[1], "elapsed_seconds": elapsed},
        "instrument": {"self_max_logit_error": self_error, "full_ceiling_max_logit_error": ceiling_error},
        "capability": capability,
        "family_mean_late_backward_target": family_target_means,
        "group_members": {name: list(members) for name, members in GROUPS.items()},
        "family_reports": reports,
        "readout_boundary": "final residual; RMS-normalized pre-tanh unembedding score; post-tanh margin",
        "row_manifest_sha256": frozen["row_manifest_sha256"],
        "runner_sha256": digest(RUNNER),
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
