#!/usr/bin/env python3
# BQGATE:64opened equality documents;128forwards;180seconds;post-MLP9 state factorial.
"""Factor the equality action at the complete post-MLP9 state boundary."""
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
import equality_matcher_mlp9_reader_calibration_rung500 as r500
import rung498_copy_task_portability_diagnosis as diagnosis
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_POST_MLP9_STATE_FACTORIAL_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
CAUSAL_NULL = P / "MLP9_EQUALITY_PROJECTED_WRITE_CAUSAL_DISCOVERY_V1_RESULT.json"
START, STOP = 564, 628
ARMS = ("native_custom", "absent_custom", "pre_fixed", "write_only", "both_fixed", "pre_recompute")
CELLS = ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors", "all_noncopy")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    null = json.loads(CAUSAL_NULL.read_text())
    if null["terminal"] != "valid_mlp9_equality_projected_write_causal_null" or not null["predictions"]["pred_a_instrument"]:
        raise ValueError("causal null authority changed")
    rows, _, _, _, scales, _ = action_parent.validate_inputs()
    return rows, scales


def plan():
    rows, _ = load_bound()
    batches = math.ceil((STOP - START) / action_parent.BATCH)
    return {"schema": "equality_post_mlp9_state_factorial_discovery_v1_plan", "opened_documents": STOP - START, "tokens_per_document": int(rows.shape[1] - 1), "arms": list(ARMS), "forward_calls": batches * 8, "new_text": 0, "fits": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def trajectory(model, tokens, *, absent, override=None):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    cached = {}
    capture = {}
    pair = action_parent.PAIRS[0]
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (1152,))
        if not absent or site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[pair[0]][1]:
                cached.update(factors[pair[0]])
            if site == action_parent.factor_parent.TERMS[pair[1]][1]:
                if not cached:
                    raise RuntimeError("donor factors unavailable")
                late = factors[pair[1]]
                attention_write = attention_write - late["native_term"]
            # The factored attention call replaces only the residual write.  As in
            # the authoritative action runner, the first-layer attention cache is
            # threaded through unchanged at later sites.
        pre = x + attention_write
        write = block.mlp(F.rms_norm(pre, (1152,)))
        if site == 9:
            capture = {"pre": pre.detach().float(), "write": write.detach().float(), "post": (pre + write).detach().float()}
            if override is not None:
                post = override(pre, write, block)
                capture["installed"] = post.detach().float()
                x = post
            else:
                x = pre + write
        else:
            x = pre + write
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, capture


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, scales = load_bound()
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    if any(not bool(masks[name][START:STOP].any()) for name in CELLS):
        raise ValueError("mask support changed")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    nll = {name: [] for name in ("native_authority", "absent_authority", *ARMS)}
    replay = {name: [] for name in ("native_logits", "absent_logits", "native_write", "absent_write", "component_closure", "both_install", "recompute_install")}
    bf16_delta_nonadditivity = []
    component_stats = {name: [] for name in ("pre2", "write2", "post2", "pre_write_cross")}
    started = time.perf_counter()

    for start in range(START, STOP, action_parent.BATCH):
        stop = min(start + action_parent.BATCH, STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].cuda()
        native_authority, native_write, _, _ = r500._captured_forward(model, tokens, direct=True)
        absent_authority, absent_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
        native_logits, native = trajectory(model, tokens, absent=False)
        absent_logits, absent_state = trajectory(model, tokens, absent=True)
        dpre = native["pre"] - absent_state["pre"]
        dwrite = native["write"] - absent_state["write"]
        dpost = native["post"] - absent_state["post"]
        component_stats["pre2"].append(float(dpre.double().square().sum()))
        component_stats["write2"].append(float(dwrite.double().square().sum()))
        component_stats["post2"].append(float(dpost.double().square().sum()))
        component_stats["pre_write_cross"].append(float((dpre.double() * dwrite.double()).sum()))
        replay["native_logits"].append(float((native_logits - native_authority.float()).norm() / native_authority.float().norm().clamp_min(1e-30)))
        replay["absent_logits"].append(float((absent_logits - absent_authority.float()).norm() / absent_authority.float().norm().clamp_min(1e-30)))
        replay["native_write"].append(float((native["write"] - native_write.float()).norm() / native_write.float().norm().clamp_min(1e-30)))
        replay["absent_write"].append(float((absent_state["write"] - absent_write.float()).norm() / absent_write.float().norm().clamp_min(1e-30)))
        # Residual additions execute in BF16.  Test closure in that actual
        # arithmetic, rather than comparing the rounded post-state delta with
        # an unrounded FP32 distributive identity (which is not BF16-valid).
        native_rebuilt = (native["pre"].to(torch.bfloat16) + native["write"].to(torch.bfloat16)).float()
        absent_rebuilt = (absent_state["pre"].to(torch.bfloat16) + absent_state["write"].to(torch.bfloat16)).float()
        replay["component_closure"].append(max(
            float((native_rebuilt - native["post"]).norm() / native["post"].norm().clamp_min(1e-30)),
            float((absent_rebuilt - absent_state["post"]).norm() / absent_state["post"].norm().clamp_min(1e-30)),
        ))
        bf16_delta_nonadditivity.append(float((dpost - dpre - dwrite).norm() / dpost.norm().clamp_min(1e-30)))
        fixed = {
            "pre_fixed": lambda pre, write, block: native["pre"].to(pre.dtype) + absent_state["write"].to(write.dtype),
            "write_only": lambda pre, write, block: absent_state["pre"].to(pre.dtype) + native["write"].to(write.dtype),
            "both_fixed": lambda pre, write, block: native["pre"].to(pre.dtype) + native["write"].to(write.dtype),
            "pre_recompute": lambda pre, write, block: native["pre"].to(pre.dtype) + block.mlp(F.rms_norm(native["pre"].to(pre.dtype), (1152,))),
        }
        arm_logits = {"native_custom": native_logits, "absent_custom": absent_logits}
        for arm, override in fixed.items():
            arm_logits[arm], arm_capture = trajectory(model, tokens, absent=True, override=override)
            if arm == "both_fixed":
                replay["both_install"].append(float((arm_capture["installed"] - native["post"]).norm() / native["post"].norm().clamp_min(1e-30)))
            if arm == "pre_recompute":
                replay["recompute_install"].append(float((arm_capture["installed"] - native["post"]).norm() / native["post"].norm().clamp_min(1e-30)))
        targets = batch_rows[:, 1:].cuda()
        for name, logits in (("native_authority", native_authority), ("absent_authority", absent_authority), *arm_logits.items()):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())

    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    native_effect = nll["absent_authority"] - nll["native_authority"]
    reports = {}
    for arm in ("pre_fixed", "write_only", "both_fixed", "pre_recompute"):
        arm_effect = nll["absent_authority"] - nll[arm]
        reports[arm] = {}
        for cell in CELLS:
            selected = masks[cell][START:STOP]
            native_sum = float(native_effect[selected].sum()); arm_sum = float(arm_effect[selected].sum())
            reports[arm][cell] = {"tokens": int(selected.sum()), "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native_authority"])[selected].mean())}
        reports[arm]["halves"] = []
        for lo, hi in ((0, 32), (32, 64)):
            selected = masks["copy_positive"][START + lo:START + hi]
            ns = float(native_effect[lo:hi][selected].sum()); ars = float(arm_effect[lo:hi][selected].sum())
            reports[arm]["halves"].append({"tokens": int(selected.sum()), "recovery": ars / ns if abs(ns) > 1e-30 else None})
    maxima = {name: max(values) for name, values in replay.items()}
    pre_recovery = reports["pre_fixed"]["copy_positive"]["recovery"]
    write_recovery = reports["write_only"]["copy_positive"]["recovery"]
    both_recovery = reports["both_fixed"]["copy_positive"]["recovery"]
    recompute_recovery = reports["pre_recompute"]["copy_positive"]["recovery"]
    interaction = both_recovery - pre_recovery - write_recovery
    pred_a = bool(max(maxima.values()) <= 2e-6)
    pred_b = bool(pred_a and min(both_recovery, recompute_recovery) >= .95 and abs(both_recovery - recompute_recovery) <= .01 and max(abs(reports[arm]["all_noncopy"]["arm_minus_native_mean_nat"]) for arm in ("both_fixed", "pre_recompute")) <= .01)
    pred_c = bool(pred_a and write_recovery <= .10)
    pred_d = bool(pred_a and pre_recovery >= .70 and pre_recovery >= write_recovery + .50)
    pred_e = bool(pred_a and abs(interaction) <= .20)
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_f = bool(pred_a and all(reports[arm][cell]["recovery"] > 0 for arm in ("pre_fixed", "both_fixed") for cell in stable_cells) and all(value["recovery"] > 0 for arm in ("pre_fixed", "both_fixed") for value in reports[arm]["halves"]))
    predictions = {"pred_a_instrument": pred_a, "pred_b_boundary_sufficiency": pred_b, "pred_c_write_only_null_repeats": pred_c, "pred_d_upstream_residual_dominance": pred_d, "pred_e_near_additive_causal_graph": pred_e, "pred_f_cell_stability": pred_f}
    terminal = "equality_post_mlp9_two_edge_state_factor" if all(predictions.values()) else "valid_equality_post_mlp9_state_factorial_null" if pred_a else "invalid"
    sums = {name: sum(values) for name, values in component_stats.items()}
    component = {"pre_over_post_norm": math.sqrt(sums["pre2"] / max(sums["post2"], 1e-30)), "write_over_post_norm": math.sqrt(sums["write2"] / max(sums["post2"], 1e-30)), "pre_write_cosine": sums["pre_write_cross"] / math.sqrt(max(sums["pre2"] * sums["write2"], 1e-30)), "closure_relative_l2": maxima["component_closure"], "maximum_bf16_delta_nonadditivity_relative_l2": max(bf16_delta_nonadditivity)}
    result = {"schema": "equality_post_mlp9_state_factorial_discovery_v1_result", "terminal": terminal, "predictions": predictions, "reports": reports, "copy_recovery_interaction": interaction, "component_geometry": component, "instrument_maxima": maxima, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Opened-panel oracle-state factorial at the complete post-MLP9 boundary for the equality score action; no upstream state extraction, new rows, fresh OOD, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "copy_recovery": {arm: reports[arm]["copy_positive"]["recovery"] for arm in reports}, "interaction": interaction, "component": component, "instrument": maxima, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
