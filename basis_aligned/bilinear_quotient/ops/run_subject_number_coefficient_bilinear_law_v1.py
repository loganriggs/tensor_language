#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 9forwards2144seq; subject-number four-scalar bilinear coefficient law;0updates.
"""A instrument; B reproduce rank one; C native substitution; D strata; E price."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as authority
import run_subject_number_rank1_fresh_confirmation_native_capability_v1 as capability
import native_capability_license as licensing
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as original

RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_PREREGISTRATION.md"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
BINDING = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_coefficient_bilinear_law_v1_result.json"
METHODS = ("base", "exact", "rank1", "law")
SUBSETS = factor_gate.BACKGROUND_SUBSETS
PATCH_CHUNK_ROWS = 256
MAXIMUM_ERROR = 5e-5
BARS = {
    "minimum_rank1_cosine": .995, "maximum_rank1_relative_l2_error": .15, "minimum_rank1_sign_agreement": .95,
    "minimum_native_cosine": .75, "maximum_native_relative_l2_error": .75, "minimum_native_sign_agreement": .75,
    "minimum_intermediate_cosine": .70, "maximum_intermediate_relative_l2_error": .85, "minimum_intermediate_sign_agreement": .70,
    "minimum_template_cosine": .65, "maximum_template_relative_l2_error": .90, "minimum_template_sign_agreement": .65,
}
PREDICTION_REGISTRY = {
    "pred_a_exact_artifact_and_license": None,
    "pred_b_law_reproduces_rank1_fresh": None,
    "pred_c_law_substitutes_native_fresh": None,
    "pred_d_fresh_intermediate_and_templates": None,
    "pred_e_compression_and_price": None,
}
PRED_KEYS = tuple(PREDICTION_REGISTRY)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_price(row_count=32):
    installs = row_count * len(SUBSETS) * len(METHODS); chunks = math.ceil(installs / PATCH_CHUNK_ROWS)
    return {"physical_model_forwards": 1 + chunks, "example_evaluations": row_count * len(authority.ROLES) + installs,
            "causal_installations": installs, "backwards": 0, "parameter_updates": 0,
            "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS, "patch_chunks": chunks}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "rank1_artifact": RANK1, "law_artifact": LAW,
             "original_prototypes": original.PROTOTYPES, "original_causal_result": original.OUT,
             "authority": Path(authority.__file__), "capability_result": capability.RESULT, "capability_license": capability.LICENSE}
    if any(sha(paths[name]) != expected for name, expected in binding["files"].items()): raise ValueError("bound artifact changed")
    if binding["bars"] != BARS or binding["price"] != derive_price() or binding["methods"] != list(METHODS): raise ValueError("bound design changed")
    rank = json.loads(RANK1.read_text())
    if rank["terminal"] != "rank1_frozen_weights_only" or rank["causal_outcomes_read"] or rank["rank_sweep"]: raise ValueError("rank artifact invalid")
    law = json.loads(LAW.read_text())
    if law["terminal"] != "bilinear_scalar_law_frozen_weights_only" or law["causal_outcomes_read"]: raise ValueError("law artifact invalid")
    return binding, rank, law


def compile_plan():
    binding, rank, law = load_bound()
    licensing.validate_causal_preflight(capability.build_gate(), capability.RESULT, capability.LICENSE, expected_license_sha256=binding["files"]["capability_license"], causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    return {"schema": "subject_number_coefficient_bilinear_law_v1_plan", "model_loaded": False, "gpu_accessed": False,
            "queue_touched": False, "methods": list(METHODS), "background_subsets": list(SUBSETS), "rows": 32,
            "rank": 1, "rank1_energy": rank["cumulative_energy"][0], "bars": BARS, "price": derive_price(),
            "law_beta": law["beta"], "coefficient_relative_l2_error": law["coefficient_relative_l2_error"],
            "storage": {"coefficient_scalars_before": 10, "coefficient_scalars_after": 4, "original_total_interface_scalars": law["original_total_interface_scalars"], "compressed_total_interface_scalars": law["compressed_total_interface_scalars"], "total_interface_storage_fraction": law["total_interface_storage_fraction"]},
            "predicates": list(PRED_KEYS), "binding_sha256": sha(BINDING)}


def compile_patch(tokens, heads, rows, torch):
    indices, replacements, specs = [], [], []
    for row_index, row in enumerate(rows):
        for subset in SUBSETS:
            for method in METHODS:
                indices.append(row_index); replacements.append(heads[(row_index, subset, method)]); specs.append((row_index, subset, method))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[:len(rows)][index], "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements), "native_reinstall_mask": torch.zeros(len(specs), dtype=torch.bool, device=tokens.device), "specs": specs}


def evaluate(model, torch, F, facade, rank, law):
    prototype_artifact, _ = original._load_artifacts(); rows = authority.build_rows(); count = len(rows); parent = tangent.parent
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    _, captured, projection, role_closure, inputs = parent._decomposed_forward(model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, count), "opposite": tangent._role_slice(captured, count, 2 * count)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, count), "opposite": tangent._role_slice(inputs, count, 2 * count)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"], model.transformer.h[parent.LAYER].attn, projection, torch, F)
    axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    rank_vectors = {key: float(value) * axis for key, value in rank["coefficients"].items()}
    law_vectors = {key: float(value) * axis for key, value in law["predicted_coefficients"].items()}
    heads = {}
    with torch.no_grad():
        for subset in SUBSETS:
            base = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset, F)).detach()
            exact = function(factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset + "YZ", F)).detach()
            for index, row in enumerate(rows):
                key = f"{row['direction_id']}.cardinality_{len(subset)}"
                heads[(index, subset, "base")] = base[index]; heads[(index, subset, "exact")] = exact[index]
                heads[(index, subset, "rank1")] = base[index] + rank_vectors[key]
                heads[(index, subset, "law")] = base[index] + law_vectors[key]
        patch = compile_patch(tokens, heads, rows, torch); margins, closures = {}, []
        for start in range(0, len(patch["specs"]), PATCH_CHUNK_ROWS):
            stop = min(start + PATCH_CHUNK_ROWS, len(patch["specs"])); logits, _, _, closure = parent.downstream._decomposed_forward(
                model, patch["tokens"][start:stop], patch["finals"][start:stop], torch, F, facade,
                replacement_heads=patch["replacement_heads"][start:stop], native_reinstall_mask=patch["native_reinstall_mask"][start:stop])
            closures.append(closure)
            for local, spec in enumerate(patch["specs"][start:stop]):
                row_index, subset, method = spec; endpoint = rows[row_index]["endpoints"]["opposite_same_lemma"]
                margins[(row_index, subset, method)] = float(logits[local, parent.SUBJECT_POSITION, endpoint["answer_id"]] - logits[local, parent.SUBJECT_POSITION, endpoint["foil_id"]])
    evidence = []
    for index, row in enumerate(rows):
        for subset in SUBSETS:
            base = margins[(index, subset, "base")]
            evidence.append({"row_id": row["row_id"], "direction": row["direction_id"], "template": row["template_id"], "background": subset, "cardinality": len(subset),
                             "native_exact_q": margins[(index, subset, "exact")] - base,
                             "rank1_q": margins[(index, subset, "rank1")] - base,
                             "law_q": margins[(index, subset, "law")] - base})
    exactness = {"role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
                 "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
                 "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in closures),
                 "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in closures)}
    return evidence, exactness


def stats(items, actual, predicted): return original._stats(items, actual, predicted)
def passes(x, c, e, s): return original._passes(x, c, e, s)


def score(evidence, exactness, plan):
    law_rank1 = stats(evidence, "rank1_q", "law_q"); native = stats(evidence, "native_exact_q", "law_q")
    intermediate = stats([x for x in evidence if x["background"] not in {"", "EAUW"}], "native_exact_q", "law_q")
    templates = {t: stats([x for x in evidence if x["template"] == t], "native_exact_q", "law_q") for t in ("behind_beside", "among_behind")}
    instrument = len(evidence) == 512 and all(v <= MAXIMUM_ERROR for v in exactness.values())
    pred_b = passes(law_rank1, BARS["minimum_rank1_cosine"], BARS["maximum_rank1_relative_l2_error"], BARS["minimum_rank1_sign_agreement"])
    pred_c = passes(native, BARS["minimum_native_cosine"], BARS["maximum_native_relative_l2_error"], BARS["minimum_native_sign_agreement"])
    pred_d = passes(intermediate, BARS["minimum_intermediate_cosine"], BARS["maximum_intermediate_relative_l2_error"], BARS["minimum_intermediate_sign_agreement"]) and all(passes(x, BARS["minimum_template_cosine"], BARS["maximum_template_relative_l2_error"], BARS["minimum_template_sign_agreement"]) for x in templates.values())
    pred_e = plan["storage"]["coefficient_scalars_after"] == 4 and plan["storage"]["total_interface_storage_fraction"] < .251 and derive_price() == plan["price"]
    predictions = dict(zip(PRED_KEYS, map(bool, (instrument, instrument and pred_b, instrument and pred_c, instrument and pred_d, instrument and pred_e))))
    return {**exactness, "law_vs_rank1": law_rank1, "law_vs_native": native, "intermediate_law_vs_native": intermediate,
            "template_law_vs_native": templates, "predictions": predictions, "evidence": evidence}


def main():
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    _, rank, law = load_bound(); torch, F, facade = tangent.parent.factors._dependencies(); model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    evidence, exactness = evaluate(model, torch, F, facade, rank, law); scored = score(evidence, exactness, plan)
    terminal = "coefficient_bilinear_law_held" if all(scored["predictions"].values()) else ("invalid" if not scored["predictions"][PRED_KEYS[0]] else "coefficient_bilinear_law_null")
    payload = managed.atomic_create_json(OUT, {"schema": "subject_number_coefficient_bilinear_law_v1_result", "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "score": scored,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "runner_sha256": sha(RUNNER)})
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
