#!/usr/bin/env python3
# BQGATE:192frozen code-OOD documents;240forwards;180seconds;reusable equality score port.
"""Insert frozen early equality scores into the exact L8H4 graph node."""
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
PACKAGE = P / "extracted_circuits/equality_l8h4_reversible_edge_v3"
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_l8h4_exact_order_node_code_ood_v2 as v2
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_REUSABLE_SCORE_PORT_CODE_OOD_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT = P / "EQUALITY_L8H4_REVERSIBLE_EDGE_CODE_OOD_V3_RESULT.json"
OOD_SCORE_PARENT = BQ / "equality_score_code_ood_rung460_results.json"
ROWS = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
DOCUMENTS = 192
TARGET_PAIR = action_parent.PAIRS[0]
CONTROL_PAIR = action_parent.PAIRS[1]
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
    parent = json.loads(PARENT.read_text())
    old = json.loads(OOD_SCORE_PARENT.read_text())
    if parent["terminal"] != "equality_l8h4_exact_reversible_ood_edge" or not all(parent["predictions"].values()):
        raise ValueError("reversible-edge authority changed")
    if not old["pred_a_instrument"] or not old["pred_c_code_causal_effect"] or old["analysis"]["selected_causal_recovery"]["recovery"] != 0.9191778236004662:
        raise ValueError("frozen code score-donor authority changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("role") != "ood_code" or list(payload["rows"].shape) != [DOCUMENTS, 257]:
        raise ValueError("code-OOD row authority changed")
    _, _, _, _, scales, _ = action_parent.validate_inputs()
    return payload["rows"], scales, old


def plan():
    rows, _, _ = load_bound()
    return {"schema": "equality_reusable_score_port_code_ood_v1_plan", "ood_documents": len(rows), "tokens_per_document": int(rows.shape[1] - 1), "forward_calls": math.ceil(len(rows) / action_parent.BATCH) * 5, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def exact_score_donor_forward(model, tokens, pair, scale):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    cached = {}
    diagnostics = {"term_rms": 0.0, "score_cosine_numerator": 0.0, "donor_score2": 0.0, "late_score2": 0.0}
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (1152,))
        if site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[pair[0]][1]:
                cached.update(factors[pair[0]])
            if site == action_parent.factor_parent.TERMS[pair[1]][1]:
                if not cached:
                    raise RuntimeError("score donor unavailable")
                late = factors[pair[1]]
                head = action_parent.factor_parent.TERMS[pair[1]][2]
                raw_payload, output_weight = v2.raw_head_payload(attention_state, v1, block.attn, head)
                donor_score = cached["p"] * scale["score_ratio"]
                donor_term = edge_node.execute(donor_score, raw_payload, support, output_weight)
                attention_write = attention_write - late["native_term"] + donor_term.to(attention_write.dtype)
                selected = support
                diagnostics["term_rms"] = float(donor_term.float().square().mean().sqrt())
                diagnostics["score_cosine_numerator"] = float((donor_score[selected].double() * late["p"][selected].double()).sum())
                diagnostics["donor_score2"] = float(donor_score[selected].double().square().sum())
                diagnostics["late_score2"] = float(late["p"][selected].double().square().sum())
        pre = x + attention_write
        x = pre + block.mlp(F.rms_norm(pre, (1152,)))
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, diagnostics


def summarize(nll, masks):
    native_effect = nll["absent"] - nll["native"]
    output = {}
    for arm in ("projected_target", "exact_target", "exact_control"):
        effect = nll["absent"] - nll[arm]
        output[arm] = {}
        for cell in CELLS:
            selected = masks[cell]
            ns = float(native_effect[selected].sum()); ars = float(effect[selected].sum())
            output[arm][cell] = {"tokens": int(selected.sum()), "native_effect_sum_nat": ns, "recovery": ars / ns if abs(ns) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native"])[selected].mean())}
        output[arm]["halves"] = []
        for lo, hi in ((0, 96), (96, 192)):
            selected = masks["copy_positive"][lo:hi]
            ns = float(native_effect[lo:hi][selected].sum()); ars = float(effect[lo:hi][selected].sum())
            output[arm]["halves"].append({"native_effect_sum_nat": ns, "recovery": ars / ns if abs(ns) > 1e-30 else None})
    return output


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, scales, old = load_bound()
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    names = ("native", "absent", "projected_target", "exact_target", "exact_control")
    nll = {name: [] for name in names}
    live = {"target_min_term_rms": math.inf, "control_min_term_rms": math.inf, "native_effect_halves": []}
    geometry = {name: 0.0 for name in ("numerator", "donor2", "late2")}
    started = time.perf_counter()
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = rows[start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=TARGET_PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        projected_logits, _, _ = action_parent.run_forward(model, tokens, pair=TARGET_PAIR, background="early_present", state="score_donor", scales=scales["L5H5"])
        target_logits, target_diag = exact_score_donor_forward(model, tokens, TARGET_PAIR, scales["L5H5"])
        control_logits, control_diag = exact_score_donor_forward(model, tokens, CONTROL_PAIR, scales["L7H3"])
        live["target_min_term_rms"] = min(live["target_min_term_rms"], target_diag["term_rms"])
        live["control_min_term_rms"] = min(live["control_min_term_rms"], control_diag["term_rms"])
        geometry["numerator"] += target_diag["score_cosine_numerator"]
        geometry["donor2"] += target_diag["donor_score2"]
        geometry["late2"] += target_diag["late_score2"]
        targets = batch_rows[:, 1:].cuda()
        for name, value in zip(names, (native_logits, absent_logits, projected_logits, target_logits, control_logits)):
            nll[name].append(F.cross_entropy(value.reshape(-1, value.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    reports = summarize(nll, masks)
    target = reports["exact_target"]["copy_positive"]["recovery"]
    projected = reports["projected_target"]["copy_positive"]["recovery"]
    control = reports["exact_control"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_a = bool(live["target_min_term_rms"] > 0 and live["control_min_term_rms"] > 0 and all(row["native_effect_sum_nat"] > 0 for row in reports["exact_target"]["halves"]))
    pred_b = bool(pred_a and .85 <= target <= 1.05 and all(reports["exact_target"][cell]["recovery"] > .70 for cell in stable_cells) and all(row["recovery"] > .70 for row in reports["exact_target"]["halves"]))
    pred_c = bool(pred_a and abs(target - projected) <= .03 and all(abs(reports["exact_target"][cell]["recovery"] - reports["projected_target"][cell]["recovery"]) <= .06 for cell in stable_cells) and all(abs(left["recovery"] - right["recovery"]) <= .06 for left, right in zip(reports["exact_target"]["halves"], reports["projected_target"]["halves"])))
    pred_d = bool(pred_a and target >= control + .50 and control <= .35)
    pred_e = bool(pred_a and abs(reports["exact_target"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01)
    pred_f = bool(planned["learned_parameters"] == 0)
    predictions = {"pred_a_live_instrument": pred_a, "pred_b_exact_node_score_reuse": pred_b, "pred_c_arithmetic_agreement": pred_c, "pred_d_donor_specificity": pred_d, "pred_e_preservation": pred_e, "pred_f_reusable_zero_parameter_port": pred_f}
    terminal = "equality_l5h5_score_reusable_ood_port" if all(predictions.values()) else "valid_equality_reusable_score_port_null" if pred_a else "invalid"
    score_cosine = geometry["numerator"] / math.sqrt(max(geometry["donor2"] * geometry["late2"], 1e-30))
    result = {"schema": "equality_reusable_score_port_code_ood_v1_result", "terminal": terminal, "predictions": predictions, "reports": reports, "live_diagnostics": live, "target_score_cosine_on_equality_edges": score_cosine, "frozen_scales": {"target": scales["L5H5"], "control": scales["L7H3"]}, "prior_projected_target_recovery": old["analysis"]["selected_causal_recovery"]["recovery"], "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Frozen natural-scale L5H5 score port reused inside the exact-order L8H4 edge on code OOD, with L7H3 wrong-score control; residual-to-score producers and cross-corpus behavioral amplitude calibration remain unextracted."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "recoveries": {"projected_target": projected, "exact_target": target, "exact_control": control}, "score_cosine": score_cosine, "noncopy_mean": reports["exact_target"]["all_noncopy"]["arm_minus_native_mean_nat"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
