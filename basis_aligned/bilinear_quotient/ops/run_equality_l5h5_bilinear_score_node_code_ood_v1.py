#!/usr/bin/env python3
# BQGATE:192frozen code-OOD documents;192forwards;180seconds;extracted L5H5 score node.
"""Validate the extracted multiplicative L5H5 Q/K score node on code OOD."""
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
PACKAGE = P / "extracted_circuits/equality_l5h5_bilinear_score_node_v1"
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_l8h4_exact_order_node_code_ood_v2 as edge_v2
import run_equality_reusable_score_port_code_ood_v1 as score_parent
from extracted_circuits.equality_l5h5_bilinear_score_node_v1 import node as score_node
from extracted_circuits.equality_l8h4_reversible_edge_v3 import node as edge_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_BILINEAR_SCORE_NODE_CODE_OOD_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT = P / "EQUALITY_REUSABLE_SCORE_PORT_CODE_OOD_V1_RESULT.json"
ROWS = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
MANIFEST = PACKAGE / "manifest.json"
DOCUMENTS = 192
PAIR = action_parent.PAIRS[0]
CELLS = score_parent.CELLS


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
    if parent["terminal"] != "equality_l5h5_score_reusable_ood_port" or not all(parent["predictions"].values()):
        raise ValueError("reusable-score authority changed")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("learned_parameters") != 0 or len(manifest.get("ports", [])) != 5:
        raise ValueError("score-node manifest changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("role") != "ood_code" or list(payload["rows"].shape) != [DOCUMENTS, 257]:
        raise ValueError("code-OOD row authority changed")
    _, _, _, _, scales, _ = action_parent.validate_inputs()
    return payload["rows"], scales, parent, manifest


def plan():
    rows, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_bilinear_score_node_code_ood_v1_plan", "ood_documents": len(rows), "tokens_per_document": int(rows.shape[1] - 1), "forward_calls": math.ceil(len(rows) / action_parent.BATCH) * 4, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def qk_ports(state, attention, head):
    batch, length, _ = state.shape
    heads = action_parent.factor_parent.stage1.HEADS
    width = action_parent.factor_parent.stage1.HEAD_DIM
    linear = lambda weight: F.linear(state, weight.to(state.dtype)).view(batch, length, heads, width)
    q1, k1 = linear(attention.c_q.weight), linear(attention.c_k.weight)
    q2, k2 = linear(attention.c_q2.weight), linear(attention.c_k2.weight)
    cos, sin = attention.rotary(q1)
    module = sys.modules[type(attention).__module__]
    q1 = module.apply_rotary_emb(F.rms_norm(q1, (width,)), cos, sin)
    k1 = module.apply_rotary_emb(F.rms_norm(k1, (width,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (width,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (width,)), cos, sin)
    return q1[:, :, head], k1[:, :, head], q2[:, :, head], k2[:, :, head]


@torch.no_grad()
def extracted_score_donor_forward(model, tokens, scale):
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    reconstructed = None
    diagnostics = {"score_error": 0.0, "composition_error": 0.0, "term_rms": 0.0}
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (1152,))
        if site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[PAIR[0]][1]:
                early = factors[PAIR[0]]
                head = action_parent.factor_parent.TERMS[PAIR[0]][2]
                ports = qk_ports(attention_state, block.attn, head)
                reconstructed = score_node.execute(*ports)
                diagnostics["score_error"] = float((reconstructed.float() - early["p"]).norm() / early["p"].norm().clamp_min(1e-30))
                q1, k1, q2, k2 = [value.float() for value in ports]
                first = [score_node.dot_score(q1[..., :64], k1[..., :64]) * .5, score_node.dot_score(q1[..., 64:], k1[..., 64:]) * .5]
                second = [score_node.dot_score(q2[..., :64], k2[..., :64]) * .5, score_node.dot_score(q2[..., 64:], k2[..., 64:]) * .5]
                full32 = score_node.execute(q1, k1, q2, k2)
                expanded = score_node.compose_scores(first, second)
                diagnostics["composition_error"] = float((full32 - expanded).norm() / full32.norm().clamp_min(1e-30))
            if site == action_parent.factor_parent.TERMS[PAIR[1]][1]:
                if reconstructed is None:
                    raise RuntimeError("reconstructed score unavailable")
                late = factors[PAIR[1]]
                head = action_parent.factor_parent.TERMS[PAIR[1]][2]
                raw_payload, output_weight = edge_v2.raw_head_payload(attention_state, v1, block.attn, head)
                donor_score = reconstructed.float() * scale["score_ratio"]
                donor_term = edge_node.execute(donor_score, raw_payload, support, output_weight)
                diagnostics["term_rms"] = float(donor_term.float().square().mean().sqrt())
                attention_write = attention_write - late["native_term"] + donor_term.to(attention_write.dtype)
        pre = x + attention_write
        x = pre + block.mlp(F.rms_norm(pre, (1152,)))
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, diagnostics


def summarize(nll, masks):
    native_effect = nll["absent"] - nll["native"]
    output = {}
    for arm in ("factor_donor", "extracted_donor"):
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
    rows, scales, parent, manifest = load_bound()
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    names = ("native", "absent", "factor_donor", "extracted_donor")
    nll = {name: [] for name in names}
    errors = {name: [] for name in ("score", "composition", "logits")}
    minimum_term_rms = math.inf
    started = time.perf_counter()
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = rows[start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, _, _ = action_parent.run_forward(model, tokens, direct=True)
        absent_logits, _, _ = action_parent.run_forward(model, tokens, pair=PAIR, background="early_present", state="late_absent", scales=scales["L5H5"])
        factor_logits, _ = score_parent.exact_score_donor_forward(model, tokens, PAIR, scales["L5H5"])
        extracted_logits, diag = extracted_score_donor_forward(model, tokens, scales["L5H5"])
        errors["score"].append(diag["score_error"])
        errors["composition"].append(diag["composition_error"])
        errors["logits"].append(float((extracted_logits - factor_logits).norm() / factor_logits.norm().clamp_min(1e-30)))
        minimum_term_rms = min(minimum_term_rms, diag["term_rms"])
        targets = batch_rows[:, 1:].cuda()
        for name, value in zip(names, (native_logits, absent_logits, factor_logits, extracted_logits)):
            nll[name].append(F.cross_entropy(value.reshape(-1, value.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    reports = summarize(nll, masks)
    maxima = {name: max(values) for name, values in errors.items()}
    factor_recovery = reports["factor_donor"]["copy_positive"]["recovery"]
    extracted_recovery = reports["extracted_donor"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_a = bool(max(maxima["score"], maxima["logits"]) <= 2e-6 and minimum_term_rms > 0 and all(row["native_effect_sum_nat"] > 0 for row in reports["extracted_donor"]["halves"]))
    pred_b = bool(pred_a and abs(extracted_recovery - factor_recovery) <= .001 and all(abs(reports["extracted_donor"][cell]["recovery"] - reports["factor_donor"][cell]["recovery"]) <= .005 for cell in stable_cells) and all(abs(left["recovery"] - right["recovery"]) <= .005 for left, right in zip(reports["extracted_donor"]["halves"], reports["factor_donor"]["halves"])))
    pred_c = bool(pred_a and .85 <= extracted_recovery <= 1.05 and abs(reports["extracted_donor"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01 and all(reports["extracted_donor"][cell]["recovery"] > .70 for cell in stable_cells) and all(row["recovery"] > .70 for row in reports["extracted_donor"]["halves"]))
    pred_d = bool(maxima["composition"] <= 2e-6)
    pred_e = bool(manifest["learned_parameters"] == 0 and planned["learned_parameters"] == 0 and len(manifest["ports"]) == 5)
    pred_f = bool(parent["predictions"]["pred_d_donor_specificity"] and parent["predictions"]["pred_f_reusable_zero_parameter_port"])
    predictions = {"pred_a_exact_score_execution": pred_a, "pred_b_behavioral_identity": pred_b, "pred_c_ood_causal_use": pred_c, "pred_d_compositionality": pred_d, "pred_e_zero_parameter_extraction": pred_e, "pred_f_inherited_specificity": pred_f}
    terminal = "equality_l5h5_bilinear_score_node_extracted_ood" if all(predictions.values()) else "valid_equality_l5h5_bilinear_score_node_null" if pred_d else "invalid"
    result = {"schema": "equality_l5h5_bilinear_score_node_code_ood_v1_result", "terminal": terminal, "predictions": predictions, "reports": reports, "maximum_errors": maxima, "minimum_extracted_term_rms": minimum_term_rms, "parent_wrong_donor_recovery": parent["reports"]["exact_control"]["copy_positive"]["recovery"], "manifest_sha256": digest(MANIFEST), "package_source_sha256": digest(PACKAGE / "node.py"), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Zero-parameter extraction of L5H5's multiplicative score from four post-projection normalized/rotary QK ports, causally reused through the frozen adapter and exact L8H4 node on code OOD; residual-to-QK production remains native."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "factor_recovery": factor_recovery, "extracted_recovery": extracted_recovery, "maximum_errors": maxima, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
