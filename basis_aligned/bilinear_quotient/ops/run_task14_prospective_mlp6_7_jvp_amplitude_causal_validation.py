#!/usr/bin/env python3
"""Open prospective causal amplitudes only after binding sealed downstream-JVP predictions."""

# BQGATE: EXPERIMENT pred_a_instrument_and_temporal_seal pred_b_prospective_amplitude_generation pred_c_direction_and_template_transfer pred_d_background_transfer pred_e_midpoint_beats_base_gradient pred_f_no_lookup_or_scale_fit
from __future__ import annotations

from collections import defaultdict
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate
import run_task14_prospective_jvp_amplitude_native_capability as capability


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1.json"
AMENDMENT = ROOT / "circuits/prior_art/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_price_amendment.json"
PREDICTION = ROOT / "circuits/fast_screens/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_predictions.json"
OUT = ROOT / "circuits/fast_screens/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_result.json"
PRIOR_ART_SHA256 = "b8e10492f622bb08cb0a2ae4370267e6782f9a4028ed907272d5f2ad04bad030"
AMENDMENT_SHA256 = "9ebe5d1ac427c0b0b6663764d05278a0111f003840e90ba9645a68d3f3744c43"
PREDICTION_SHA256 = "9ccaa378508c1bd5b29e53d815b1d8e986bcd23beb689602e641d54240772d77"
CAPABILITY_RESULT_SHA256 = "9ee68c9297995cc5cf1f6a7c29759c7199b258ec35974fdf5c4000d3e5085749"
CAPABILITY_LICENSE_SHA256 = "27acd0cb5e7459630f89188abd2160622e07967ba0ee9194bf26708801fde33c"
BACKGROUNDS = ("", "EAUW")
METHODS = ("base", "exact")
BARS = {"maximum_numerical_absolute_error": 5e-5,
    "minimum_overall_cosine": .999, "maximum_overall_relative_l2_error": .01,
    "minimum_overall_sign_agreement": 1.0, "minimum_cell_cosine": .995,
    "maximum_cell_relative_l2_error": .03, "minimum_cell_sign_agreement": .9375,
    "minimum_background_cosine": .995, "maximum_background_relative_l2_error": .03,
    "minimum_midpoint_error_reduction_over_base": .50}
PRED_KEYS = ("pred_a_instrument_and_temporal_seal", "pred_b_prospective_amplitude_generation",
    "pred_c_direction_and_template_transfer", "pred_d_background_transfer",
    "pred_e_midpoint_beats_base_gradient", "pred_f_no_lookup_or_scale_fit",)


class ProspectiveValidationError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price(row_count=32):
    installs = row_count * len(BACKGROUNDS) * len(METHODS)
    return {"physical_model_forwards": 2,
        "example_evaluations": row_count*len(authority.ROLES) + installs,
        "causal_interventions": installs, "backwards": 0, "parameter_updates": 0}


def _load_prediction():
    if _sha256(PREDICTION) != PREDICTION_SHA256:
        raise ProspectiveValidationError("sealed prediction changed")
    result = json.loads(PREDICTION.read_text())
    if result.get("terminal") != "sealed_prediction" or result.get("causal_outcomes_opened") is not False \
            or not all(result.get("predictions", {}).values()):
        raise ProspectiveValidationError("prediction receipt is not a valid temporal seal")
    return result


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (AMENDMENT, AMENDMENT_SHA256, "price amendment"),
        (capability.RESULT, CAPABILITY_RESULT_SHA256, "capability result"),
        (capability.LICENSE, CAPABILITY_LICENSE_SHA256, "capability license")):
        if _sha256(path) != expected:
            raise ProspectiveValidationError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(), capability.RESULT,
        capability.LICENSE, expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    _load_prediction()
    if derive_price() != {"physical_model_forwards": 2, "example_evaluations": 224,
            "causal_interventions": 128, "backwards": 0, "parameter_updates": 0}:
        raise ProspectiveValidationError("derived price changed")


def compile_plan():
    prediction = _load_prediction(); validate_preflight()
    return {"schema": "task14_prospective_mlp6_7_jvp_amplitude_causal_validation_plan_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "NEW_PROSPECTIVE_TEXT_CAUSAL_VALIDATION",
        "row_count": 32, "backgrounds": list(BACKGROUNDS), "methods": list(METHODS),
        "prior_art_sha256": PRIOR_ART_SHA256, "price_amendment_sha256": AMENDMENT_SHA256,
        "sealed_prediction_sha256": PREDICTION_SHA256,
        "sealed_prediction_created_utc": prediction["created_utc"],
        "capability_license_sha256": CAPABILITY_LICENSE_SHA256,
        "literal_scorer": "no fitted scale, offset, endpoint, or group mean enters any gate",
        "bars": dict(BARS), "predictions": dict(zip(PRED_KEYS, (
            "hashes, temporal seal, complete lattice, and numerical closures pass",
            "overall midpoint cosine>=.999, relative L2<=.01, signs=1",
            "each direction-template cell cosine>=.995, relative L2<=.03, signs>=.9375",
            "each background cosine>=.995 and relative L2<=.03",
            "midpoint reduces relative error at least 50% versus base gradient",
            "literal gates use no lookup or fitted affine repair"))),
        "price": derive_price()}


def _compile_patch(tokens, heads, rows, torch):
    indices, replacements, specs = [], [], []
    for row_index, row in enumerate(rows):
        for background in BACKGROUNDS:
            for method in METHODS:
                indices.append(row_index); replacements.append(heads[(background, method)][row_index])
                specs.append((row_index, background, method, row["direction_id"], row["template_id"]))
    index = torch.tensor(indices, dtype=torch.long, device=tokens.device)
    return {"tokens": tokens[:len(rows)][index],
        "finals": torch.full_like(index, tangent.parent.SUBJECT_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.zeros(len(specs), dtype=torch.bool, device=tokens.device),
        "specs": specs}


def evaluate(model, torch, F, facade):
    rows = authority.build_rows(); n = len(rows); parent = tangent.parent
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    _, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, n),
             "opposite": tangent._role_slice(captured, n, 2*n)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                   "opposite": tangent._role_slice(inputs, n, 2*n)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"],
        model.transformer.h[parent.LAYER].attn, projection, torch, F)
    heads = {}
    with torch.no_grad():
        for background in BACKGROUNDS:
            heads[(background, "base")] = function(gate._raw_for(
                input_roles["recipient"], input_roles["opposite"], background, F)).detach()
            heads[(background, "exact")] = function(gate._raw_for(
                input_roles["recipient"], input_roles["opposite"], background + "YZ", F)).detach()
        patch = _compile_patch(tokens, heads, rows, torch)
        logits, _, _, downstream_closure = parent.downstream._decomposed_forward(
            model, patch["tokens"], patch["finals"], torch, F, facade,
            replacement_heads=patch["replacement_heads"],
            native_reinstall_mask=patch["native_reinstall_mask"])
    endpoint_margins = {}
    for index, (row_index, background, method, direction, template) in enumerate(patch["specs"]):
        endpoint = rows[row_index]["endpoints"]["opposite_same_lemma"]
        margin = float(logits[index, parent.SUBJECT_POSITION, endpoint["answer_id"]]
                       - logits[index, parent.SUBJECT_POSITION, endpoint["foil_id"]])
        endpoint_margins[(row_index, background, method)] = margin
    evidence = []
    for row_index, row in enumerate(rows):
        for background in BACKGROUNDS:
            evidence.append({"row_id": row["row_id"], "direction": row["direction_id"],
                "template": row["template_id"], "background": background,
                "actual_q": endpoint_margins[(row_index, background, "exact")]
                    - endpoint_margins[(row_index, background, "base")]})
    exactness = {"role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": downstream_closure["state_sum_max_absolute_error"],
        "downstream_normalized_closure_max_absolute_error": downstream_closure["normalized_state_max_absolute_error"]}
    return evidence, exactness


def _vector_stats(items, field):
    actual = [x["actual_q"] for x in items]; predicted = [x[field] for x in items]
    dot = sum(a*p for a,p in zip(actual,predicted))
    an = math.sqrt(sum(a*a for a in actual)); pn = math.sqrt(sum(p*p for p in predicted))
    err = math.sqrt(sum((p-a)**2 for a,p in zip(actual,predicted)))
    return {"count": len(items), "cosine": dot/max(an*pn,1e-30),
        "relative_l2_error": err/max(an,1e-30),
        "sign_agreement": sum((a>0)==(p>0) for a,p in zip(actual,predicted))/len(items)}


def _affine_description(items):
    x = [i["midpoint_jvp_q"] for i in items]; y = [i["actual_q"] for i in items]
    xm=sum(x)/len(x); ym=sum(y)/len(y); den=sum((v-xm)**2 for v in x)
    scale=sum((a-xm)*(b-ym) for a,b in zip(x,y))/max(den,1e-30); offset=ym-scale*xm
    repaired=[{**item,"affine":scale*item["midpoint_jvp_q"]+offset} for item in items]
    return {"scale":scale,"offset":offset,"stats":_vector_stats(repaired,"affine"),
        "gate_effect":"DESCRIPTIVE_ONLY_NONE"}


def score(causal_evidence, exactness, bars=BARS):
    prediction = _load_prediction()
    pred = {(x["row_id"],x["background"]):x for x in prediction["evidence"]}
    items=[]
    for actual in causal_evidence:
        value=pred.get((actual["row_id"],actual["background"]))
        if value is None: raise ProspectiveValidationError("causal target lacks sealed prediction")
        items.append({**actual,"base_jvp_q":value["base_jvp_q"],
            "midpoint_jvp_q":value["midpoint_jvp_q"]})
    unique={(x["row_id"],x["background"]) for x in items}
    overall={point:_vector_stats(items,f"{point}_jvp_q") for point in ("base","midpoint")}
    cells={}
    for direction in ("plural_to_singular","singular_to_plural"):
        for template in ("above_below","below_above"):
            selected=[x for x in items if x["direction"]==direction and x["template"]==template]
            cells[f"{direction}__{template}"]=_vector_stats(selected,"midpoint_jvp_q")
    backgrounds={background or "empty":_vector_stats(
        [x for x in items if x["background"]==background],"midpoint_jvp_q")
        for background in BACKGROUNDS}
    improvement=1-overall["midpoint"]["relative_l2_error"]/max(overall["base"]["relative_l2_error"],1e-30)
    instrument=len(items)==64 and len(unique)==64 and all(
        value<=bars["maximum_numerical_absolute_error"] for value in exactness.values())
    middle=overall["midpoint"]
    overall_ok=middle["cosine"]>=bars["minimum_overall_cosine"] and \
        middle["relative_l2_error"]<=bars["maximum_overall_relative_l2_error"] and \
        middle["sign_agreement"]>=bars["minimum_overall_sign_agreement"]
    cells_ok=all(x["cosine"]>=bars["minimum_cell_cosine"] and
        x["relative_l2_error"]<=bars["maximum_cell_relative_l2_error"] and
        x["sign_agreement"]>=bars["minimum_cell_sign_agreement"] for x in cells.values())
    backgrounds_ok=all(x["cosine"]>=bars["minimum_background_cosine"] and
        x["relative_l2_error"]<=bars["maximum_background_relative_l2_error"]
        for x in backgrounds.values())
    predictions=dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and overall_ok),
        bool(instrument and cells_ok),bool(instrument and backgrounds_ok),
        bool(instrument and improvement>=bars["minimum_midpoint_error_reduction_over_base"]),True)))
    return {**exactness,"overall":overall,"by_direction_template":cells,
        "by_background":backgrounds,"midpoint_error_reduction_over_base":improvement,
        "descriptive_post_gate_affine_repair":_affine_description(items),
        "predictions":predictions,"joined_evidence":items}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv); plan=compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":
        print(json.dumps(plan,sort_keys=True)); return
    if OUT.exists(): raise ProspectiveValidationError(f"refusing to overwrite {OUT}")
    torch,F,facade=tangent.parent.factors._dependencies()
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    causal_evidence,exactness=evaluate(model,torch,F,facade); scored=score(causal_evidence,exactness)
    terminal="valid_causal_screen" if scored["predictions"][PRED_KEYS[0]] else "invalid"
    result={"schema":"task14_prospective_mlp6_7_jvp_amplitude_causal_validation_result_v1",
        "candidate_id":authority.CAUSAL_CANDIDATE_ID,"terminal":terminal,
        "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,
        "score":scored,"causal_evidence":causal_evidence,
        "sealed_prediction_sha256":PREDICTION_SHA256}
    payload=managed.atomic_create_json(OUT,result)
    print(json.dumps({"terminal":terminal,"predictions":scored["predictions"],
        "result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
