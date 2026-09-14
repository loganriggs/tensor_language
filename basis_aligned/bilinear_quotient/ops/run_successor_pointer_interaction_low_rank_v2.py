#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 22forwards154seq90readouts; successor cross-type residual low-rank V2 readout correction;0updates.
"""A FIT instrument; B rank selected; C length-seven transfer; D low-rank null."""
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
FIT_ROWS = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json"
HOLD_ROWS = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH7_LOW_RANK_HOLDOUT_V1_ROWS.json"
PREREG = POLY / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_PREREGISTRATION.md"
INTERACTION = POLY / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_RESULT.json"
CONFIRMATION = POLY / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_RESULT.json"
CORRECTION = POLY / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V2_READOUT_CORRECTION.md"
V1_BINDING = POLY / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_BINDING.json"
V1_RUNNER = OPS / "run_successor_pointer_interaction_low_rank_v1.py"
V1_RESULT = POLY / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_RESULT.json"
BINDING = POLY / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V2_BINDING.json"
OUT = POLY / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V2_RESULT.json"
REQUESTED_DRY = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
CONDITIONS = ("coherent", "early_swap_control", "late_swap_incoherent")
FAMILIES = ("month", "digit")
MODULES = tuple(name for layer in range(18) for name in (f"attn{layer}", f"mlp{layer}"))
GROUPS = {
    "A": tuple(f"attn{x}" for x in range(8, 13)),
    "M": tuple(f"mlp{x}" for x in range(8, 13)),
    "AM": tuple(name for x in range(8, 13) for name in (f"attn{x}", f"mlp{x}")),
}
RANKS = (1, 2, 4, 8)
BARS = {
    "maximum_self_logit_absolute_error": 1e-5,
    "maximum_full_ceiling_logit_absolute_error": 1e-4,
    "maximum_direct_readout_logit_absolute_error": 1e-5,
    "minimum_native_accuracy": 0.75,
    "minimum_family_mean_late_backward_target": 1.0,
    "minimum_joint_projection": 0.50,
    "minimum_joint_cosine": 0.70,
    "minimum_synthetic_projection": 0.80,
    "maximum_synthetic_projection": 1.20,
    "minimum_synthetic_cosine": 0.95,
    "minimum_full_vocab_recovery": 0.80,
    "maximum_control_rms_fraction": 0.50,
}
PRICE = {"maximum_forwards": 22, "maximum_sequences": 154, "maximum_readout_rows": 90, "fits": 1, "updates": 0}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"fit_rows": FIT_ROWS, "holdout_rows": HOLD_ROWS, "preregistration": PREREG,
             "interaction_result": INTERACTION, "confirmation_result": CONFIRMATION,
             "correction": CORRECTION, "v1_binding": V1_BINDING, "v1_runner": V1_RUNNER, "v1_invalid_result": V1_RESULT}
    for name, expected in binding["files"].items():
        if digest(paths[name]) != expected:
            raise RuntimeError(f"bound {name} changed")
    fit, hold = json.loads(FIT_ROWS.read_text()), json.loads(HOLD_ROWS.read_text())
    if binding["fit_row_manifest_sha256"] != fit["row_manifest_sha256"] or binding["holdout_row_manifest_sha256"] != hold["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed")
    if binding["bars"] != BARS or binding["price"] != PRICE or binding["ranks"] != list(RANKS):
        raise RuntimeError("bars, ranks, or price changed")
    if binding["groups"] != {name: list(value) for name, value in GROUPS.items()}:
        raise RuntimeError("groups changed")
    return binding, fit, hold


def aligned_rows(frozen, expected):
    by_key = {}
    for row in frozen["rows"]:
        by_key.setdefault((row["family"], int(row["final_index"])), {})[row["condition"]] = row
    keys = sorted(by_key)
    if len(keys) != expected or any(set(by_key[key]) != set(CONDITIONS) for key in keys):
        raise RuntimeError("incomplete aligned factorial")
    return keys, {condition: [by_key[key][condition] for key in keys] for condition in CONDITIONS}


def margin(logits, rows, direction):
    return np.asarray([values[int(row["recipient_answer_id"])] - values[int(row["donors"][direction]["answer_id"])] for values, row in zip(logits, rows)], dtype=np.float64)


def vector_metrics(rescue, target):
    denominator = float(np.dot(target, target)); rn = float(np.linalg.norm(rescue)); tn = float(np.linalg.norm(target))
    return {
        "projection": float(np.dot(rescue, target) / denominator) if denominator else float("nan"),
        "cosine": float(np.dot(rescue, target) / (rn * tn)) if rn and tn else 0.0,
        "residual_rms": float(np.sqrt(np.mean((rescue - target) ** 2))),
        "rescue_rms": float(np.sqrt(np.mean(rescue ** 2))), "target_rms": float(np.sqrt(np.mean(target ** 2))),
    }


def plan():
    _, fit, hold = load_bound(); fk, _ = aligned_rows(fit, 8); hk, _ = aligned_rows(hold, 6)
    return {"schema": "successor_pointer_interaction_low_rank_v2_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "fit_groups": len(fk), "holdout_groups": len(hk),
            "groups": {name: list(value) for name, value in GROUPS.items()}, "ranks": list(RANKS),
            "bars": BARS, "price": PRICE,
            "predicates": ["pred_a_fit_instrument", "pred_b_fit_rank_selected", "pred_c_holdout_transfer", "pred_d_low_rank_null"]}


def main():
    _, fit_frozen, hold_frozen = load_bound()
    fit_keys, fit_rows = aligned_rows(fit_frozen, 8)
    hold_keys, hold_rows = aligned_rows(hold_frozen, 6)
    if REQUESTED_DRY:
        print(json.dumps(plan(), indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    sys.path.insert(0, str(ROOT / "basis_aligned/qk_mdl/algo_tasks/successor"))
    import torch
    import successor_lib as sl
    from circuit_fast_screen_managed_runner import atomic_create_json

    torch.set_num_threads(2); model, cfg = sl.load_model(); device = next(model.parameters()).device
    counts = {"forwards": 0, "sequences": 0, "readout_rows": 0}

    def counted_run(idx, **kwargs):
        counts["forwards"] += 1; counts["sequences"] += len(idx)
        return sl.run(model, cfg, idx, **kwargs)

    def readout(residual, template, query):
        counts["readout_rows"] += len(residual)
        value = template.clone()
        replacement = torch.as_tensor(residual, device=device).to(value.dtype)
        value[:, query] = replacement
        with torch.no_grad():
            raw = model.lm_head(torch.nn.functional.rms_norm(value, (cfg["n_embd"],)))
            logits = 30 * torch.tanh(raw / 30)
        return logits[:, query].float().cpu().numpy()

    def run_dataset(keys, rows):
        query = int(rows["coherent"][0]["query_position"])
        if any(int(row["query_position"]) != query for condition in CONDITIONS for row in rows[condition]):
            raise RuntimeError("query mismatch")
        tensors = {condition: torch.tensor([row["token_ids"] for row in rows[condition]], dtype=torch.long, device=device) for condition in CONDITIONS}
        full, caches = {}, {}
        for condition in CONDITIONS:
            full[condition], caches[condition] = counted_run(tensors[condition], collect=True)
        native = {condition: full[condition][:, query].float().cpu().numpy() for condition in CONDITIONS}
        residual = {condition: caches[condition][("r", 17)][:, query].float().cpu().numpy() for condition in CONDITIONS}

        def patches(source, destination, selected=None):
            head, mlp = {}, {}; chosen = set(MODULES if selected is None else selected)
            for layer in range(18):
                if f"attn{layer}" in chosen:
                    hybrid = caches[destination][("h", layer)].clone(); hybrid[:, query] = caches[source][("h", layer)][:, query]
                    for h in range(cfg["n_head"]): head[(layer, h)] = hybrid[:, :, h]
                if f"mlp{layer}" in chosen:
                    hybrid = caches[destination][("m", layer)].clone(); hybrid[:, query] = caches[source][("m", layer)][:, query]; mlp[layer] = hybrid
            return head, mlp

        self_head = {(layer, h): caches["late_swap_incoherent"][("h", layer)][:, :, h] for layer in range(18) for h in range(cfg["n_head"])}
        self_mlp = {layer: caches["late_swap_incoherent"][("m", layer)] for layer in range(18)}
        self_full, _ = counted_run(tensors["late_swap_incoherent"], patch_head=self_head, patch_mlp=self_mlp)
        fh, fm = patches("coherent", "late_swap_incoherent")
        ceiling_full, _ = counted_run(tensors["late_swap_incoherent"], patch_head=fh, patch_mlp=fm)
        arms, arm_residual = {}, {}
        for name, members in GROUPS.items():
            arms[name], arm_residual[name] = {}, {}
            for destination in ("late_swap_incoherent", "early_swap_control"):
                h, m = patches("coherent", destination, members)
                values, cache = counted_run(tensors[destination], patch_head=h or None, patch_mlp=m or None, collect=True)
                arms[name][destination] = values[:, query].float().cpu().numpy()
                arm_residual[name][destination] = cache[("r", 17)][:, query].float().cpu().numpy()
        return {"keys": keys, "rows": rows, "native": native, "residual": residual, "arms": arms,
                "arm_residual": arm_residual, "residual_templates": {condition: caches[condition][("r", 17)] for condition in CONDITIONS}, "query": query,
                "self_error": float(np.max(np.abs(self_full[:, query].float().cpu().numpy() - native["late_swap_incoherent"]))),
                "ceiling_error": float(np.max(np.abs(ceiling_full[:, query].float().cpu().numpy() - native["coherent"]))) }

    def basic_report(data):
        keys, rows, native, arms = data["keys"], data["rows"], data["native"], data["arms"]
        target = margin(native["coherent"], rows["coherent"], "backward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
        capability = {family: {condition: float(np.mean([int(np.argmax(native[condition][i])) == int(rows[condition][i]["recipient_answer_id"]) for i, key in enumerate(keys) if key[0] == family])) for condition in CONDITIONS} for family in FAMILIES}
        joint = {}
        for family in FAMILIES:
            ix = np.asarray([i for i, key in enumerate(keys) if key[0] == family]); t = target[ix]; tr = float(np.sqrt(np.mean(t ** 2)))
            backward = margin(arms["AM"]["late_swap_incoherent"], rows["late_swap_incoherent"], "backward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
            lf = margin(arms["AM"]["late_swap_incoherent"], rows["late_swap_incoherent"], "forward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "forward")
            eb = margin(arms["AM"]["early_swap_control"], rows["early_swap_control"], "backward") - margin(native["early_swap_control"], rows["early_swap_control"], "backward")
            x = vector_metrics(backward[ix], t); x["late_forward_control_fraction"] = float(np.sqrt(np.mean(lf[ix] ** 2)) / tr); x["early_backward_control_fraction"] = float(np.sqrt(np.mean(eb[ix] ** 2)) / tr)
            x["passes"] = bool(x["projection"] >= BARS["minimum_joint_projection"] and x["cosine"] >= BARS["minimum_joint_cosine"] and x["late_forward_control_fraction"] <= BARS["maximum_control_rms_fraction"] and x["early_backward_control_fraction"] <= BARS["maximum_control_rms_fraction"])
            joint[family] = x
        means = {family: float(np.mean(target[[i for i, key in enumerate(keys) if key[0] == family]])) for family in FAMILIES}
        return target, capability, joint, means

    def interactions(data, destination):
        r = data["arm_residual"]; base = data["residual"][destination]
        return r["AM"][destination] - r["A"][destination] - r["M"][destination] + base

    def rank_report(data, basis, rank):
        rows, keys, native, r = data["rows"], data["keys"], data["native"], data["arm_residual"]
        synthetic = {}
        for destination in ("late_swap_incoherent", "early_swap_control"):
            interaction = interactions(data, destination); sub = basis[:rank]
            projection = (interaction @ sub.T) @ sub
            additive = r["A"][destination] + r["M"][destination] - data["residual"][destination]
            synthetic[destination] = readout(additive + projection, data["residual_templates"][destination], data["query"])
        target = margin(native["coherent"], rows["coherent"], "backward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
        reports = {}; passes = True
        for family in FAMILIES:
            ix = np.asarray([i for i, key in enumerate(keys) if key[0] == family]); tr = float(np.sqrt(np.mean(target[ix] ** 2)))
            synth = margin(synthetic["late_swap_incoherent"], rows["late_swap_incoherent"], "backward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
            exact = margin(data["arms"]["AM"]["late_swap_incoherent"], rows["late_swap_incoherent"], "backward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "backward")
            stats = vector_metrics(synth[ix], exact[ix])
            numerator = float(np.sqrt(np.mean((synthetic["late_swap_incoherent"][ix] - data["arms"]["AM"]["late_swap_incoherent"][ix]) ** 2)))
            denominator = float(np.sqrt(np.mean((data["arms"]["AM"]["late_swap_incoherent"][ix] - native["late_swap_incoherent"][ix]) ** 2)))
            stats["full_vocab_recovery"] = 1.0 - numerator / denominator
            lf = margin(synthetic["late_swap_incoherent"], rows["late_swap_incoherent"], "forward") - margin(native["late_swap_incoherent"], rows["late_swap_incoherent"], "forward")
            eb = margin(synthetic["early_swap_control"], rows["early_swap_control"], "backward") - margin(native["early_swap_control"], rows["early_swap_control"], "backward")
            stats["late_forward_control_fraction"] = float(np.sqrt(np.mean(lf[ix] ** 2)) / tr); stats["early_backward_control_fraction"] = float(np.sqrt(np.mean(eb[ix] ** 2)) / tr)
            stats["passes"] = bool(BARS["minimum_synthetic_projection"] <= stats["projection"] <= BARS["maximum_synthetic_projection"] and stats["cosine"] >= BARS["minimum_synthetic_cosine"] and stats["full_vocab_recovery"] >= BARS["minimum_full_vocab_recovery"] and stats["late_forward_control_fraction"] <= BARS["maximum_control_rms_fraction"] and stats["early_backward_control_fraction"] <= BARS["maximum_control_rms_fraction"])
            passes &= stats["passes"]; reports[family] = stats
        return {"rank": rank, "families": reports, "passes": bool(passes)}

    started = time.perf_counter()
    fit = run_dataset(fit_keys, fit_rows)
    fit_target, fit_capability, fit_joint, fit_means = basic_report(fit)
    fit_readout_error = float(np.max(np.abs(readout(fit["residual"]["late_swap_incoherent"], fit["residual_templates"]["late_swap_incoherent"], fit["query"]) - fit["native"]["late_swap_incoherent"])))
    interaction_fit = interactions(fit, "late_swap_incoherent")
    _, singular, vh = np.linalg.svd(interaction_fit.astype(np.float64), full_matrices=False)
    energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    fit_rank_reports = {str(rank): rank_report(fit, vh, rank) for rank in RANKS}
    selected = next((rank for rank in RANKS if fit_rank_reports[str(rank)]["passes"]), None)
    fit_instrument = bool(fit["self_error"] <= BARS["maximum_self_logit_absolute_error"] and fit["ceiling_error"] <= BARS["maximum_full_ceiling_logit_absolute_error"] and fit_readout_error <= BARS["maximum_direct_readout_logit_absolute_error"] and all(value >= BARS["minimum_native_accuracy"] for cells in fit_capability.values() for value in cells.values()) and all(value >= BARS["minimum_family_mean_late_backward_target"] for value in fit_means.values()) and all(x["passes"] for x in fit_joint.values()) and counts["forwards"] == 11 and counts["sequences"] == 88 and counts["readout_rows"] == 72)

    hold_report = None; hold_instrument = None
    if fit_instrument and selected is not None:
        hold = run_dataset(hold_keys, hold_rows)
        _, hold_capability, hold_joint, hold_means = basic_report(hold)
        hold_readout_error = float(np.max(np.abs(readout(hold["residual"]["late_swap_incoherent"], hold["residual_templates"]["late_swap_incoherent"], hold["query"]) - hold["native"]["late_swap_incoherent"])))
        hold_rank = rank_report(hold, vh, selected)
        hold_instrument = bool(hold["self_error"] <= BARS["maximum_self_logit_absolute_error"] and hold["ceiling_error"] <= BARS["maximum_full_ceiling_logit_absolute_error"] and hold_readout_error <= BARS["maximum_direct_readout_logit_absolute_error"] and all(value >= BARS["minimum_native_accuracy"] for cells in hold_capability.values() for value in cells.values()) and all(value >= BARS["minimum_family_mean_late_backward_target"] for value in hold_means.values()) and all(x["passes"] for x in hold_joint.values()) and counts == {"forwards": 22, "sequences": 154, "readout_rows": 90})
        hold_report = {"instrument": hold_instrument, "capability": hold_capability, "joint": hold_joint, "family_mean_target": hold_means, "direct_readout_error": hold_readout_error, "selected_rank": hold_rank, "row_manifest_sha256": hold_frozen["row_manifest_sha256"]}

    pred_a = fit_instrument
    pred_b = bool(pred_a and selected is not None)
    pred_c = bool(pred_b and hold_instrument and hold_report["selected_rank"]["passes"])
    pred_d = bool(pred_a and (selected is None or (hold_instrument and not pred_c)))
    terminal = "invalid" if not pred_a or (selected is not None and not hold_instrument) else (f"low_rank_{selected}_transfer" if pred_c else "low_rank_null")
    result = {
        "schema": "successor_pointer_interaction_low_rank_v2_result", "terminal": terminal,
        "predictions": {"pred_a_fit_instrument": pred_a, "pred_b_fit_rank_selected": pred_b, "pred_c_holdout_transfer": pred_c, "pred_d_low_rank_null": pred_d},
        "bars": BARS, "price": {**PRICE, "observed_forwards": counts["forwards"], "observed_sequences": counts["sequences"], "observed_readout_rows": counts["readout_rows"], "elapsed_seconds": time.perf_counter() - started},
        "fit": {"instrument": fit_instrument, "capability": fit_capability, "joint": fit_joint, "family_mean_target": fit_means, "direct_readout_error": fit_readout_error, "singular_values": singular.tolist(), "cumulative_energy": energy.tolist(), "rank_reports": fit_rank_reports, "selected_rank": selected, "row_manifest_sha256": fit_frozen["row_manifest_sha256"]},
        "holdout": hold_report, "claim_boundary": "row-projected final-residual interaction basis; coefficient generator and component responses external",
        "runner_sha256": digest(RUNNER),
    }
    atomic_create_json(OUT, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
