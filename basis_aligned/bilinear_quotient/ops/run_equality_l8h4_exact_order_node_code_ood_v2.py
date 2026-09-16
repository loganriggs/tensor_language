#!/usr/bin/env python3
# BQGATE:192frozen code-OOD documents;240forwards;180seconds;exact-order equality edge.
"""Validate the exact-order extracted L8H4 equality edge on code OOD."""
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
PACKAGE = P / "extracted_circuits/equality_l8h4_score_rawpayload_node_v2"
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(PACKAGE), str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as diagnosis
import node as exact_node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L8H4_EXACT_ORDER_NODE_CODE_OOD_V2"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT = P / "EQUALITY_L8H4_EXTRACTED_NODE_CODE_OOD_V1_RESULT.json"
OOD_PARENT = P / "EQUALITY_A8_EDGE_CODE_OOD_CONFIRMATION_V1_RESULT.json"
ROWS = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
MANIFEST = PACKAGE / "manifest.json"
DOCUMENTS = 192
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
    v1 = json.loads(PARENT.read_text())
    ood = json.loads(OOD_PARENT.read_text())
    if v1["terminal"] != "valid_equality_l8h4_extracted_node_null" or v1["predictions"]["pred_a_executor_replay"] or not v1["predictions"]["pred_f_bilinear_compositionality"]:
        raise ValueError("projected-payload null authority changed")
    if ood["terminal"] != "valid_equality_a8_edge_code_ood_null" or not ood["predictions"]["pred_a_instrument"]:
        raise ValueError("OOD oracle authority changed")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("learned_parameters") != 0 or not manifest.get("operation", "").startswith("output_projection"):
        raise ValueError("exact-node manifest changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("role") != "ood_code" or list(payload["rows"].shape) != [DOCUMENTS, 257]:
        raise ValueError("code-OOD row authority changed")
    action_parent.validate_inputs()
    return payload["rows"], ood, manifest


def plan():
    rows, _, _ = load_bound()
    return {"schema": "equality_l8h4_exact_order_node_code_ood_v2_plan", "ood_documents": len(rows), "tokens_per_document": int(rows.shape[1] - 1), "forward_calls": math.ceil(len(rows) / action_parent.BATCH) * 5, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def raw_head_payload(attention_state, first_value, attention, head):
    batch, length, _ = attention_state.shape
    raw = F.linear(attention_state, attention.c_v.weight.to(attention_state.dtype)).view(batch, length, action_parent.factor_parent.stage1.HEADS, action_parent.factor_parent.stage1.HEAD_DIM)
    value = (1 - attention.lamb) * raw + attention.lamb * first_value.view_as(raw)
    weight = attention.c_proj.weight[:, head * action_parent.factor_parent.stage1.HEAD_DIM:(head + 1) * action_parent.factor_parent.stage1.HEAD_DIM]
    return value[:, :, head], weight


@torch.no_grad()
def trajectory(model, tokens, mode, sources=None):
    if mode not in ("native", "absent", "package_removal", "oracle100", "package100"):
        raise ValueError("unknown trajectory mode")
    if mode in ("oracle100", "package100") and sources is None:
        raise ValueError("edge installation needs sources")
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    cached = {}
    capture = {}
    diagnostics = {"term_error": 0.0, "composition_error": 0.0}
    pair = action_parent.PAIRS[0]
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (1152,))
        if mode == "native" or site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[pair[0]][1]:
                cached.update(factors[pair[0]])
                early_head = action_parent.factor_parent.TERMS[pair[0]][2]
                cached["raw_payload"], _ = raw_head_payload(attention_state, v1, block.attn, early_head)
            if site == action_parent.factor_parent.TERMS[pair[1]][1]:
                if not cached:
                    raise RuntimeError("donor factors unavailable")
                late = factors[pair[1]]
                head = action_parent.factor_parent.TERMS[pair[1]][2]
                raw_payload, output_weight = raw_head_payload(attention_state, v1, block.attn, head)
                package_term = exact_node.execute(late["p"], raw_payload, support, output_weight)
                diagnostics["term_error"] = float((package_term.float() - late["native_term"].float()).norm() / late["native_term"].float().norm().clamp_min(1e-30))
                p1, p2 = late["p"].float(), cached["p"].float()
                v_late = raw_payload.float()
                v_early = cached["raw_payload"].float()
                w32 = output_weight.float()
                composed = exact_node.execute(p1 + p2, v_late + v_early, support, w32)
                expanded = sum(exact_node.execute(p, value, support, w32) for p in (p1, p2) for value in (v_late, v_early))
                diagnostics["composition_error"] = float((composed - expanded).norm() / expanded.norm().clamp_min(1e-30))
                if mode in ("absent", "oracle100", "package100"):
                    attention_write = attention_write - late["native_term"]
                elif mode == "package_removal":
                    attention_write = attention_write - package_term.to(attention_write.dtype)
                if mode == "oracle100":
                    attention_write = sources["native"]["a8"].to(attention_write.dtype)
                elif mode == "package100":
                    attention_write = attention_write + package_term.to(attention_write.dtype)
        pre = x + attention_write
        mlp_write = block.mlp(F.rms_norm(pre, (1152,)))
        if site == 8 and mode in ("oracle100", "package100"):
            mlp_write = sources["absent"]["m8"].to(mlp_write.dtype)
        if site == 8:
            capture["a8"] = attention_write.detach().float(); capture["m8"] = mlp_write.detach().float()
        if site == 9 and mode in ("oracle100", "package100"):
            attention_write = sources["absent"]["a9"].to(attention_write.dtype)
            pre = x + attention_write
            mlp_write = block.mlp(F.rms_norm(pre, (1152,)))
        if site == 9:
            capture["a9"] = attention_write.detach().float(); capture["mlp9"] = mlp_write.detach().float()
        x = pre + mlp_write
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, capture, diagnostics


def summarize(nll, masks):
    native_effect = nll["absent"] - nll["native"]
    output = {}
    for arm in ("absent", "package_removal", "oracle100", "package100"):
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
    rows, ood_parent, manifest = load_bound()
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    names = ("native", "absent", "package_removal", "oracle100", "package100")
    nll = {name: [] for name in names}
    errors = {name: [] for name in ("term", "composition", "removal_logits", "removal_mlp9", "install_logits", "install_mlp9")}
    started = time.perf_counter()
    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = rows[start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_logits, native, _ = trajectory(model, tokens, "native")
        absent_logits, absent, _ = trajectory(model, tokens, "absent")
        sources = {"native": native, "absent": absent}
        removal_logits, removal, removal_diag = trajectory(model, tokens, "package_removal")
        oracle_logits, oracle, _ = trajectory(model, tokens, "oracle100", sources)
        package_logits, package, package_diag = trajectory(model, tokens, "package100", sources)
        errors["term"].append(max(removal_diag["term_error"], package_diag["term_error"]))
        errors["composition"].append(max(removal_diag["composition_error"], package_diag["composition_error"]))
        errors["removal_logits"].append(float((removal_logits - absent_logits).norm() / absent_logits.norm().clamp_min(1e-30)))
        errors["removal_mlp9"].append(float((removal["mlp9"] - absent["mlp9"]).norm() / absent["mlp9"].norm().clamp_min(1e-30)))
        errors["install_logits"].append(float((package_logits - oracle_logits).norm() / oracle_logits.norm().clamp_min(1e-30)))
        errors["install_mlp9"].append(float((package["mlp9"] - oracle["mlp9"]).norm() / oracle["mlp9"].norm().clamp_min(1e-30)))
        targets = batch_rows[:, 1:].cuda()
        for name, value in zip(names, (native_logits, absent_logits, removal_logits, oracle_logits, package_logits)):
            nll[name].append(F.cross_entropy(value.reshape(-1, value.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    reports = summarize(nll, masks)
    maxima = {name: max(values) for name, values in errors.items()}
    oracle_recovery = reports["oracle100"]["copy_positive"]["recovery"]
    package_recovery = reports["package100"]["copy_positive"]["recovery"]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_a = bool(max(maxima[name] for name in ("term", "removal_logits", "removal_mlp9", "install_logits", "install_mlp9")) <= 2e-6)
    pred_b = bool(pred_a and abs(package_recovery - oracle_recovery) <= .001 and all(abs(reports["package100"][cell]["recovery"] - reports["oracle100"][cell]["recovery"]) <= .005 for cell in stable_cells) and all(abs(left["recovery"] - right["recovery"]) <= .005 for left, right in zip(reports["package100"]["halves"], reports["oracle100"]["halves"])))
    pred_c = bool(pred_a and reports["package_removal"]["copy_positive"]["native_effect_sum_nat"] > 0 and all(row["native_effect_sum_nat"] > 0 for row in reports["package_removal"]["halves"]))
    pred_d = bool(pred_a and package_recovery >= .85 and abs(reports["package100"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01 and all(reports["package100"][cell]["recovery"] > .70 for cell in stable_cells) and all(row["recovery"] > .70 for row in reports["package100"]["halves"]))
    pred_e = bool(manifest["learned_parameters"] == 0 and planned["learned_parameters"] == 0)
    pred_f = bool(maxima["composition"] <= 2e-6)
    predictions = {"pred_a_exact_executor_replay": pred_a, "pred_b_behavioral_identity": pred_b, "pred_c_removal": pred_c, "pred_d_ood_installation": pred_d, "pred_e_zero_parameter_extraction": pred_e, "pred_f_bilinear_compositionality": pred_f}
    terminal = "equality_l8h4_exact_extracted_ood_reusable_edge" if all(predictions.values()) else "valid_equality_l8h4_exact_order_node_null" if pred_f else "invalid"
    result = {"schema": "equality_l8h4_exact_order_node_code_ood_v2_result", "terminal": terminal, "predictions": predictions, "reports": reports, "maximum_errors": maxima, "parent_oracle_recovery": ood_parent["reports"]["100"]["copy_positive"]["recovery"], "manifest_sha256": digest(MANIFEST), "package_source_sha256": digest(PACKAGE / "node.py"), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Exact-order zero-new-parameter equality-edge executor tested for removal, oracle-edge installation, and bilinear reuse on frozen code OOD; upstream native score/payload producers remain external and the scalar transfer-calibration null is unchanged."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "oracle_recovery": oracle_recovery, "package_recovery": package_recovery, "maximum_errors": maxima, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
