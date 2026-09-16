#!/usr/bin/env python3
# BQGATE: 192 frozen code-OOD documents; 864 forward calls; 16 matched directions; 240 seconds; zero fits.
"""Equal-norm same-site directional null for the reversible L8H4 edge."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_l8h4_exact_order_node_code_ood_v2 as v2
import run_equality_l8h4_reversible_edge_code_ood_v3 as v3
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as reversible_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1"
PREREG = P / f"{STEM}_PREREGISTRATION.md"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
V3_RESULT = P / "EQUALITY_L8H4_REVERSIBLE_EDGE_CODE_OOD_V3_RESULT.json"
ROWS = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
DOCUMENTS = 192
CONTROL_SEEDS = tuple(range(2026091700, 2026091716))
CELLS = v2.CELLS
PRICE = {"forward_calls": 864, "ood_documents": DOCUMENTS, "matched_directions": 16,
         "checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_instrument_and_equal_norm": None,
    "pred_b_target_removal_is_behaviorally_material": None,
    "pred_c_target_beats_equal_norm_directions": None,
    "pred_d_target_is_selective": None,
    "pred_e_fixed_zero_fit_control_bank": None,
}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def matched_random_direction(term, seed):
    """Return a deterministic per-position equal-norm isotropic direction."""
    generator = torch.Generator(device=term.device).manual_seed(int(seed))
    random = torch.randn(term.shape, generator=generator, device=term.device, dtype=torch.float32)
    target_norm = term.float().norm(dim=-1, keepdim=True)
    random = random / random.norm(dim=-1, keepdim=True).clamp_min(1e-30) * target_norm
    deployed = random.to(term.dtype)
    deployed_norm = deployed.float().norm(dim=-1, keepdim=True)
    positive = target_norm > 0
    error = float(((deployed_norm - target_norm).abs() / target_norm.clamp_min(1e-30))[positive].max()) \
        if bool(positive.any()) else 0.0
    return deployed, error


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {
        "preregistration": PREREG,
        "v3_result": V3_RESULT,
        "v3_runner": Path(v3.__file__),
        "v2_runner": Path(v2.__file__),
        "package_node": Path(reversible_node.__file__),
        "rows": ROWS,
        "action_parent": Path(action_parent.__file__),
        "diagnosis": Path(diagnosis.__file__),
        "facade": Path(facade.__file__),
    }
    if binding["files"] != {name: digest(path) for name, path in paths.items()} \
            or binding["runner_sha256"] != digest(RUNNER) \
            or binding["control_seeds"] != list(CONTROL_SEEDS) or binding["price"] != PRICE:
        raise ValueError("equal-norm binding changed")
    parent = json.loads(V3_RESULT.read_text())
    if parent["terminal"] != "equality_l8h4_exact_reversible_ood_edge" \
            or not all(parent["predictions"].values()) or parent["maximum_errors"]["removal_logits"] != 0.0:
        raise ValueError("V3 reversible-edge authority changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("role") != "ood_code" or list(payload["rows"].shape) != [DOCUMENTS, 257]:
        raise ValueError("code-OOD row authority changed")
    if not (os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL")):
        action_parent.validate_inputs()
    return payload["rows"], parent


def plan():
    rows, parent = load_bound()
    return {"schema": "equality_l8h4_equal_norm_removal_null_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "documents": len(rows), "control_seeds": list(CONTROL_SEEDS), "price": PRICE,
            "parent_terminal": parent["terminal"]}


@torch.no_grad()
def random_removal_trajectory(model, tokens, seed):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    first_values = None
    norm_error = None
    pair = action_parent.PAIRS[0]
    target_site = action_parent.factor_parent.TERMS[pair[1]][1]
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        state = F.rms_norm(x, (1152,))
        if site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, first_values = block.attn(state, first_values)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(
                state, first_values, block.attn, site, tokens)
            if site == target_site:
                late = factors[pair[1]]
                head = action_parent.factor_parent.TERMS[pair[1]][2]
                raw_payload, output_weight = v2.raw_head_payload(state, first_values, block.attn, head)
                term = reversible_node.execute(late["p"], raw_payload, support, output_weight).to(attention_write.dtype)
                random, norm_error = matched_random_direction(term, seed)
                attention_write = attention_write - random
        x = x + attention_write
        x = x + block.mlp(F.rms_norm(x, (1152,)))
    if norm_error is None:
        raise RuntimeError("L8H4 target site was not reached")
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, norm_error


def _summed_damage(nll, native, mask):
    return float((nll - native)[mask].sum())


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(240)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, parent = load_bound()
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    native_parts, target_parts = [], []
    control_parts = [[] for _ in CONTROL_SEEDS]
    norm_errors = []
    forward_calls = 0
    started = time.perf_counter()
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = rows[start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = v2.trajectory(model, tokens, "native")
        target_logits, _, _ = v3.canonical_trajectory(model, tokens, "canonical_removal")
        forward_calls += 2
        targets = batch_rows[:, 1:].cuda()
        native_parts.append(F.cross_entropy(native_logits.reshape(-1, native_logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
        target_parts.append(F.cross_entropy(target_logits.reshape(-1, target_logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
        for control_index, base_seed in enumerate(CONTROL_SEEDS):
            logits, error = random_removal_trajectory(model, tokens, base_seed + start)
            forward_calls += 1; norm_errors.append(error)
            control_parts[control_index].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    native = torch.cat(native_parts); target = torch.cat(target_parts)
    controls = [torch.cat(parts) for parts in control_parts]
    target_damage = _summed_damage(target, native, masks["copy_positive"])
    random_damage = [_summed_damage(control, native, masks["copy_positive"]) for control in controls]
    random_absolute = [abs(value) for value in random_damage]
    random_median = float(torch.tensor(random_absolute, dtype=torch.float64).median())
    percentile = sum(target_damage > value for value in random_absolute) / len(random_absolute)
    ratio = target_damage / max(random_median, 1e-30)
    target_collateral = abs(float((target - native)[masks["all_noncopy"]].mean()))
    random_collateral = [abs(float((control - native)[masks["all_noncopy"]].mean())) for control in controls]
    random_collateral_median = float(torch.tensor(random_collateral, dtype=torch.float64).median())
    target_cells = {cell: _summed_damage(target, native, masks[cell]) for cell in CELLS}
    target_halves = []
    for lo, hi in ((0, 96), (96, 192)):
        selected = masks["copy_positive"][lo:hi]
        target_halves.append(float((target - native)[lo:hi][selected].sum()))
    finite_values = [target_damage, *random_damage, target_collateral, *random_collateral,
                     *target_cells.values(), *target_halves, *norm_errors]
    pred_a = bool(all(math.isfinite(value) for value in finite_values)
        and max(norm_errors) <= .01 and forward_calls == PRICE["forward_calls"]
        and checkpoint.weights_sha256 == parent["checkpoint_weights_sha256"])
    pred_b = bool(pred_a and target_damage > 0 and min(target_cells.values()) > 0 and min(target_halves) > 0)
    pred_c = bool(pred_a and percentile >= .875 and ratio >= 2.)
    pred_d = bool(pred_a and target_collateral <= .01 and target_collateral <= random_collateral_median)
    pred_e = bool(len(CONTROL_SEEDS) == 16 and len(set(CONTROL_SEEDS)) == 16
        and PRICE["fits"] == PRICE["gradients"] == PRICE["parameter_updates"] == 0)
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d, pred_e)))
    terminal = "equality_l8h4_selective_removal_equal_norm" if all(predictions.values()) \
        else "valid_equality_l8h4_equal_norm_removal_null" if pred_a else "invalid"
    result = {
        "schema": "equality_l8h4_equal_norm_removal_null_v1_result", "terminal": terminal,
        "predictions": predictions,
        "target": {"copy_positive_damage_sum_nat": target_damage, "copy_cell_damage_sum_nat": target_cells,
                   "half_damage_sum_nat": target_halves, "all_noncopy_mean_absolute_nat": target_collateral},
        "equal_norm_controls": {"seeds": list(CONTROL_SEEDS), "copy_positive_damage_sum_nat": random_damage,
            "copy_positive_absolute_damage_median_nat": random_median, "target_to_random_median_ratio": ratio,
            "target_percentile": percentile, "all_noncopy_mean_absolute_nat": random_collateral,
            "all_noncopy_median_absolute_nat": random_collateral_median,
            "maximum_per_position_norm_relative_error": max(norm_errors)},
        "instrument": {"finite": all(math.isfinite(value) for value in finite_values), "forward_calls": forward_calls,
                       "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING),
                       "checkpoint_weights_sha256": checkpoint.weights_sha256},
        "price": PRICE | {"model_loaded": True, "gpu_accessed": True, "queue_touched": True},
        "parent_result_sha256": digest(V3_RESULT), "seconds": time.perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Post-hoc equal-norm same-site directional null on the frozen code-OOD panel; can certify selective removal but adds no fresh OOD evidence.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "target": result["target"],
                      "equal_norm_controls": result["equal_norm_controls"], "seconds": result["seconds"]}, indent=2), flush=True)
    if not pred_a:
        raise RuntimeError("equal-norm control instrument failed")


if __name__ == "__main__":
    main()
