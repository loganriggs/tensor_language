#!/usr/bin/env python3
# BQGATE:64opened equality documents;192forwards;180seconds;pre-MLP9 three-edge factorial.
"""Factor the equality action across attention8, MLP8, and attention9 writes."""
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

STEM = "EQUALITY_PRE_MLP9_THREE_EDGE_FACTORIAL_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT = P / "EQUALITY_POST_MLP9_STATE_FACTORIAL_DISCOVERY_V1_RESULT.json"
START, STOP = 564, 628
EDGES = ("a8", "m8", "a9")
ARMS = tuple(f"{mask:03b}" for mask in range(8))
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
    if parent["terminal"] != "equality_post_mlp9_two_edge_state_factor" or not all(parent["predictions"].values()):
        raise ValueError("post-MLP9 boundary authority changed")
    rows, _, _, _, scales, _ = action_parent.validate_inputs()
    return rows, scales


def plan():
    rows, _ = load_bound()
    batches = math.ceil((STOP - START) / action_parent.BATCH)
    return {"schema": "equality_pre_mlp9_three_edge_factorial_discovery_v1_plan", "opened_documents": STOP - START, "tokens_per_document": int(rows.shape[1] - 1), "arms": list(ARMS), "forward_calls": batches * 12, "new_text": 0, "fits": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def trajectory(model, tokens, *, absent, sources=None, arm=None):
    if (sources is None) != (arm is None):
        raise ValueError("sources and arm must be supplied together")
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown factorial arm")
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0 = x
    v1 = None
    cached = {}
    capture = {}
    install_errors = []
    pair = action_parent.PAIRS[0]
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (1152,))
        if not absent or site not in action_parent.factor_parent.stage1.SITE_HEADS:
            attention_write, v1 = block.attn(attention_state, v1)
        else:
            attention_write, factors, _support, _ = action_parent.factor_parent._factor_site(attention_state, v1, block.attn, site, tokens)
            if site == action_parent.factor_parent.TERMS[pair[0]][1]:
                cached.update(factors[pair[0]])
            if site == action_parent.factor_parent.TERMS[pair[1]][1]:
                if not cached:
                    raise RuntimeError("donor factors unavailable")
                attention_write = attention_write - factors[pair[1]]["native_term"]
        if site == 8 and sources is not None:
            expected = sources["native" if arm[0] == "1" else "absent"]["a8"].to(attention_write.dtype)
            attention_write = expected
            install_errors.append(float((attention_write.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)))
        if site == 9 and sources is not None:
            expected = sources["native" if arm[2] == "1" else "absent"]["a9"].to(attention_write.dtype)
            attention_write = expected
            install_errors.append(float((attention_write.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)))
        pre = x + attention_write
        mlp_write = block.mlp(F.rms_norm(pre, (1152,)))
        if site == 8 and sources is not None:
            expected = sources["native" if arm[1] == "1" else "absent"]["m8"].to(mlp_write.dtype)
            mlp_write = expected
            install_errors.append(float((mlp_write.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)))
        if site == 8:
            capture["a8"] = attention_write.detach().float()
            capture["m8"] = mlp_write.detach().float()
        if site == 9:
            capture["a9"] = attention_write.detach().float()
            capture["pre9"] = pre.detach().float()
            capture["mlp9"] = mlp_write.detach().float()
        x = pre + mlp_write
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30)).float()
    return logits, capture, max(install_errors, default=0.0)


def _report(nll, masks):
    native_effect = nll["000"] - nll["111"]
    reports = {}
    for arm in ARMS:
        arm_effect = nll["000"] - nll[arm]
        reports[arm] = {}
        for cell in CELLS:
            selected = masks[cell][START:STOP]
            ns = float(native_effect[selected].sum()); ars = float(arm_effect[selected].sum())
            reports[arm][cell] = {"tokens": int(selected.sum()), "recovery": ars / ns if abs(ns) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["111"])[selected].mean())}
        reports[arm]["halves"] = []
        for lo, hi in ((0, 32), (32, 64)):
            selected = masks["copy_positive"][START + lo:START + hi]
            ns = float(native_effect[lo:hi][selected].sum()); ars = float(arm_effect[lo:hi][selected].sum())
            reports[arm]["halves"].append({"tokens": int(selected.sum()), "recovery": ars / ns if abs(ns) > 1e-30 else None})
    return reports


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
    nll = {name: [] for name in (*ARMS, "native_authority", "absent_authority")}
    replay = {name: [] for name in ("native_logits", "absent_logits", "native_mlp9", "absent_mlp9", "corner_native_logits", "corner_absent_logits", "corner_native_pre9", "corner_absent_pre9", "edge_install")}
    edge_delta2 = {edge: 0.0 for edge in EDGES}
    edge_cross = {(left, right): 0.0 for i, left in enumerate(EDGES) for right in EDGES[i + 1:]}
    started = time.perf_counter()

    for start in range(START, STOP, action_parent.BATCH):
        stop = min(start + action_parent.BATCH, STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].cuda()
        native_authority, native_write, _, _ = r500._captured_forward(model, tokens, direct=True)
        absent_authority, absent_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
        native_logits, native, _ = trajectory(model, tokens, absent=False)
        absent_logits, absent_state, _ = trajectory(model, tokens, absent=True)
        sources = {"native": native, "absent": absent_state}
        replay["native_logits"].append(float((native_logits - native_authority.float()).norm() / native_authority.float().norm().clamp_min(1e-30)))
        replay["absent_logits"].append(float((absent_logits - absent_authority.float()).norm() / absent_authority.float().norm().clamp_min(1e-30)))
        replay["native_mlp9"].append(float((native["mlp9"] - native_write.float()).norm() / native_write.float().norm().clamp_min(1e-30)))
        replay["absent_mlp9"].append(float((absent_state["mlp9"] - absent_write.float()).norm() / absent_write.float().norm().clamp_min(1e-30)))
        arm_logits = {}
        arm_captures = {}
        for arm in ARMS:
            arm_logits[arm], arm_captures[arm], install_error = trajectory(model, tokens, absent=True, sources=sources, arm=arm)
            replay["edge_install"].append(install_error)
        replay["corner_native_logits"].append(float((arm_logits["111"] - native_logits).norm() / native_logits.norm().clamp_min(1e-30)))
        replay["corner_absent_logits"].append(float((arm_logits["000"] - absent_logits).norm() / absent_logits.norm().clamp_min(1e-30)))
        replay["corner_native_pre9"].append(float((arm_captures["111"]["pre9"] - native["pre9"]).norm() / native["pre9"].norm().clamp_min(1e-30)))
        replay["corner_absent_pre9"].append(float((arm_captures["000"]["pre9"] - absent_state["pre9"]).norm() / absent_state["pre9"].norm().clamp_min(1e-30)))
        deltas = {edge: native[edge] - absent_state[edge] for edge in EDGES}
        for edge in EDGES:
            edge_delta2[edge] += float(deltas[edge].double().square().sum())
        for pair in edge_cross:
            edge_cross[pair] += float((deltas[pair[0]].double() * deltas[pair[1]].double()).sum())
        targets = batch_rows[:, 1:].cuda()
        for name, logits in (("native_authority", native_authority), ("absent_authority", absent_authority), *arm_logits.items()):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())

    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    reports = _report(nll, masks)
    maxima = {name: max(values) for name, values in replay.items()}
    stability_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    eligible = []
    for arm in ARMS:
        size = arm.count("1")
        if size not in (1, 2):
            continue
        if (reports[arm]["copy_positive"]["recovery"] >= .90
                and abs(reports[arm]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01
                and all(reports[arm][cell]["recovery"] > .70 for cell in stability_cells)
                and all(half["recovery"] > .70 for half in reports[arm]["halves"])):
            eligible.append(arm)
    eligible.sort(key=lambda arm: (arm.count("1"), abs(reports[arm]["copy_positive"]["recovery"] - 1), arm))
    selected = eligible[0] if eligible else None
    recovery = {arm: reports[arm]["copy_positive"]["recovery"] for arm in ARMS}
    pair_interactions = {
        "a8_m8": recovery["110"] - recovery["100"] - recovery["010"] + recovery["000"],
        "a8_a9": recovery["101"] - recovery["100"] - recovery["001"] + recovery["000"],
        "m8_a9": recovery["011"] - recovery["010"] - recovery["001"] + recovery["000"],
    }
    third = recovery["111"] - recovery["110"] - recovery["101"] - recovery["011"] + recovery["100"] + recovery["010"] + recovery["001"] - recovery["000"]
    pred_a = bool(max(maxima.values()) <= 2e-6)
    pred_b = bool(pred_a and abs(recovery["000"]) <= .01 and .99 <= recovery["111"] <= 1.01 and abs(reports["111"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .001)
    pred_c = bool(pred_a and pred_b and selected is not None)
    pred_d = bool(pred_a and max(recovery[arm] for arm in ("100", "010", "001")) >= .50)
    pred_e = bool(pred_a and max(abs(value) for value in pair_interactions.values()) <= .25 and abs(third) <= .15)
    pred_f = bool(pred_c and all(reports[selected][cell]["recovery"] > .70 for cell in stability_cells) and all(half["recovery"] > .70 for half in reports[selected]["halves"]))
    predictions = {"pred_a_instrument": pred_a, "pred_b_corner_sufficiency": pred_b, "pred_c_sparse_support": pred_c, "pred_d_single_edge_concentration": pred_d, "pred_e_low_order_graph": pred_e, "pred_f_selected_stability": pred_f}
    terminal = "equality_pre_mlp9_sparse_oracle_edge_candidate" if all(predictions[key] for key in ("pred_a_instrument", "pred_b_corner_sufficiency", "pred_c_sparse_support", "pred_f_selected_stability")) else "valid_equality_pre_mlp9_three_edge_factorial_null" if pred_a and pred_b else "invalid"
    cosines = {f"{left}_{right}": edge_cross[(left, right)] / math.sqrt(max(edge_delta2[left] * edge_delta2[right], 1e-30)) for left, right in edge_cross}
    result = {"schema": "equality_pre_mlp9_three_edge_factorial_discovery_v1_result", "terminal": terminal, "predictions": predictions, "selected_support": selected, "eligible_supports": eligible, "reports": reports, "factorial": {"copy_recovery": recovery, "pair_interactions": pair_interactions, "third_order_interaction": third}, "edge_geometry": {"delta_norms": {edge: math.sqrt(value) for edge, value in edge_delta2.items()}, "cosines": cosines}, "instrument_maxima": maxima, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Opened-panel oracle-write factorial across attention8, MLP8, and attention9 under the established equality score removal; no extracted upstream writer, new rows, fresh/OOD confirmation, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected": selected, "eligible": eligible, "recovery": recovery, "pair_interactions": pair_interactions, "third": third, "instrument": maxima, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
