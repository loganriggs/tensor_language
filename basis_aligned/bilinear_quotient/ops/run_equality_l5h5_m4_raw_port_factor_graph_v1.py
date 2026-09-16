#!/usr/bin/env python3
# BQGATE:192 natural and 192 code documents;312 forwards;180seconds;M4 raw-port factor-graph extraction.
"""Validate the standalone no-oracle raw-port precision factor graph."""
import json
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

import bilin18_observed_model_facade as facade
import run_equality_l5h5_m4_two_factor_correction_graph_v1 as v1
import run_equality_l5h5_m4_two_factor_correction_graph_v2 as v2
from extracted_circuits.equality_l5h5_m4_raw_port_factor_graph_v1 import node
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_L5H5_M4_RAW_PORT_FACTOR_GRAPH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT_RESULT = P / "EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V2_RESULT.json"
MANIFEST = P / "extracted_circuits/equality_l5h5_m4_raw_port_factor_graph_v1/manifest.json"
DOCUMENTS = v1.DOCUMENTS
CELLS = v1.CELLS
NODES = v2.NODES
VARIANTS = v2.VARIANTS
v1.NODES = NODES
v1.VARIANTS = VARIANTS


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
    parent = json.loads(PARENT_RESULT.read_text())
    if parent["terminal"] != "equality_l5h5_m4_precision_corrected_factor_graph_ood" or not all(parent["predictions"].values()):
        raise ValueError("precision-factor authority changed")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["oracle_score_inputs"] != 0 or manifest["learned_parameters"] != 0:
        raise ValueError("raw-port package boundary changed")
    roles, scales, _, _, explicit_parent, inherited_manifest = v2.load_bound()
    return roles, scales, parent, explicit_parent, inherited_manifest, manifest


def plan():
    roles, _, _, _, _, _ = load_bound()
    return {"schema": "equality_l5h5_m4_raw_port_factor_graph_v1_plan", "natural_documents": len(roles["final_natural"]), "ood_documents": len(roles["ood_code"]), "macro_nodes": list(NODES), "variants": list(VARIANTS), "execution_count": 312, "oracle_score_inputs": 0, "fits": 0, "new_text": 0, "learned_parameters": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def raw_factor_scores(residuals, attention):
    ports = {name: v1.selection_parent.native_ports(attention, residuals[name]) for name in ("baseline", "remainder", "joint", "child")}
    denominators = {name: v1.kernel_parent.rms_denominator(residuals[name]) for name in residuals}
    derived = []
    for index in range(4):
        value = ports["joint"][index].float() * (denominators["joint"] / denominators["child"])
        value = value - ports["remainder"][index].float() * (denominators["remainder"] / denominators["child"])
        value = value + ports["baseline"][index].float() * (denominators["baseline"] / denominators["child"])
        derived.append(value.to(residuals["child"].dtype))
    cos, sin = v1.mode_parent.module_parent.parent.rotary_ports(attention, residuals["joint"])
    nodes = node.decompose(derived, ports["child"], cos, sin)
    child = {
        "full": node.compose(nodes),
        "base": nodes["base"],
        "drop_first": node.remove_node(nodes, "first"),
        "drop_second": node.remove_node(nodes, "second"),
        "drop_cross": node.remove_node(nodes, "cross"),
        "drop_arithmetic": node.remove_node(nodes, "arithmetic"),
        "rolled_arithmetic": node.replace_with_rolled_arithmetic(nodes),
    }
    fixed = {name: v1.kernel_parent.score_from_raw(ports[name], cos, sin) for name in ("baseline", "remainder", "joint")}
    additive = {name: (fixed["baseline"] + (value - fixed["baseline"])) + (fixed["remainder"] - fixed["baseline"]) for name, value in child.items()}
    direct_child = v1.kernel_parent.score_from_raw(ports["child"], cos, sin)
    return additive, nodes, child["full"] - direct_child, fixed


v1.factor_scores = raw_factor_scores
v2.factor_scores = raw_factor_scores


def maximum_behavior_delta(observed, parent):
    maximum = 0.0
    for label in ("natural", "code"):
        for name in NODES:
            for cell in ("copy_positive", "half_0", "half_1"):
                current = observed[label][name][cell]["relative_to_interaction_removal"]
                expected = parent["behavior"][label][name][cell]["relative_to_interaction_removal"]
                maximum = max(maximum, abs(current - expected))
    return maximum


def maximum_score_delta(census, parent):
    return max(abs(census[label]["node_relative_norm"][name] - parent["score_census"][label]["node_relative_norm"][name]) for label in ("natural", "code") for name in NODES)


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    roles, scales, parent, explicit_parent, inherited_manifest, manifest = load_bound()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    started = time.perf_counter()
    basis, writers, bridge_error = v1.split_parent.compile_basis(model)
    census = {label: v2.score_census(model, roles[role], basis, writers) for label, role in (("natural", "final_natural"), ("code", "ood_code"))}
    behavior = {}
    noncopy = {}
    minimum_rms = float("inf")
    for label, role in (("natural", "final_natural"), ("code", "ood_code")):
        masks = v1.selection_parent.masks_for(roles[role])
        nll, rms = v1.collect(model, roles[role], scales["L5H5"], basis, writers)
        minimum_rms = min(minimum_rms, rms)
        behavior[label] = v2.reports(nll, masks)
        noncopy[label] = {name: float((nll[f"drop_{name}"] - nll["full"])[masks["all_noncopy"]].mean()) for name in NODES}
    behavior_delta = maximum_behavior_delta(behavior, parent)
    score_delta = maximum_score_delta(census, parent)
    noncopy_delta = max(abs(noncopy[label][name] - parent["noncopy_incremental_mean_nat"][label][name]) for label in ("natural", "code") for name in NODES)
    roll_delta = abs(behavior["code"]["rolled_arithmetic"]["copy_positive"]["cosine_to_true_arithmetic_removal"] - parent["behavior"]["code"]["rolled_arithmetic"]["copy_positive"]["cosine_to_true_arithmetic_removal"])
    pred_a = bool(explicit_parent["predictions"]["pred_a_lawful_explicit_interaction_graph"] and bridge_error <= 2e-5 and all(census[label]["closure_relative_l2"] <= 2e-6 for label in census))
    pred_b = bool(all(behavior[label]["control"][cell]["relative_l2"] <= 2e-5 and behavior[label]["control"][cell]["cosine"] >= .99999 for label in behavior for cell in CELLS))
    pred_c = bool(behavior_delta <= .002 and score_delta <= 2e-6)
    pred_d = bool(noncopy_delta <= 2e-5 and roll_delta <= .002)
    pred_e = bool(manifest["oracle_score_inputs"] == 0 and manifest["learned_parameters"] == 0 and len(manifest["nodes"]) == 4 and manifest["native_qk_projections"] == 16 and minimum_rms > 0)
    predictions = {"pred_a_raw_port_score_closure": pred_a, "pred_b_ood_behavioral_equivalence": pred_b, "pred_c_node_removal_equivalence": pred_c, "pred_d_selectivity_and_roll_equivalence": pred_d, "pred_e_no_oracle_extraction_boundary": pred_e}
    terminal = "equality_l5h5_m4_raw_port_factor_graph_extracted_ood" if all(predictions.values()) else "valid_equality_l5h5_m4_raw_port_extraction_null" if pred_a and pred_b else "invalid"
    result = {"schema": "equality_l5h5_m4_raw_port_factor_graph_v1_result", "terminal": terminal, "predictions": predictions, "graph": {"nodes": manifest["nodes"], "input_ports": manifest["input_ports"], "oracle_score_inputs": 0, "native_qk_projections": 16, "learned_parameters": 0}, "score_census": census, "behavior": behavior, "noncopy_incremental_mean_nat": noncopy, "equivalence": {"maximum_node_removal_magnitude_absolute_delta": behavior_delta, "maximum_node_score_relative_norm_absolute_delta": score_delta, "maximum_noncopy_mean_absolute_delta_nat": noncopy_delta, "arithmetic_roll_effect_cosine_absolute_delta": roll_delta}, "instrument": {"contracted_operator_reconstruction_relative_l2": bridge_error, "minimum_donor_term_rms": minimum_rms}, "price": planned | {"checkpoint_loads": 1, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "package_manifest_sha256": digest(MANIFEST), "seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Standalone raw-Q/K-port execution of the precision-corrected M4 factor graph with no oracle score input, validated for OOD behavior, removal, selectivity, and composition."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "equivalence": result["equivalence"], "score_census": census, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
