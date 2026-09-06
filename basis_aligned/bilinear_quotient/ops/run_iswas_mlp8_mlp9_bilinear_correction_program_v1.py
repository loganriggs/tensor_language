#!/usr/bin/env python3
"""Exact MLP9 bilinear factor program under the fully clamped attention9 route."""

# BQGATE: EXPERIMENT pred_a_authority_factor_closure_finiteness_and_price pred_b_full_factor_union_replays_exact_mlp9_correction pred_c_left_right_terms_are_sufficient pred_d_bilinear_interaction_is_secondary pred_e_zero_fit_literal_mlp9_weights
from datetime import datetime, timezone
import hashlib, itertools, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_mlp9_bilinear_correction_program_v1.json"
DIRECT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_direct_skip_weight_compiler_v2_result.json"
DIRECT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_direct_skip_weight_compiler_v2.py"
ATLAS = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_upstream_module_atlas_v2_result.json"
GREEDY = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_greedy_mlp_program_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_mlp9_bilinear_correction_program_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_mlp9_bilinear_correction_program_v1"
FACTORS = ("left_change", "right_change", "bilinear_interaction")
EXPECTED = {"prior": "57bf50a8c00b76aaa5426b86bd05aed0d420687b5e2a8562831772b5414acf83", "direct": "ee569e6cbf16a84dfcb615fc08c6c633d92a4854a867f054dc117af92bf41090",
    "direct_runner": "800adf2988b2114f0f94b4c455bc5a4bfcd5efabb02590da91e4e14046396cb9",
    "atlas": "da3bd0473cf259537f5d70f1dff0ffe513bd5a7b420f6b6317259b522e4a3e67",
    "greedy": "7d530f1fab59de9a26b13905629bdfff847339b6b171a3a95242689a241c85d9",
    "capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2"}
MAX_FORWARDS, MAX_EVALUATIONS = 10, 300

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def cosine(x, y):
    d = float(x.norm()*y.norm()); return float((x*y).sum())/d if d else 0.0
def arm_name(subset): return "+".join(subset) if subset else "empty"
def subsets():
    for n in range(len(FACTORS)+1): yield from itertools.combinations(FACTORS, n)

def capture_mlp9(backend, batch, call):
    box = {}
    def pre(_module, args): box["input"] = args[0].detach().clone()
    def post(_module, _args, output): box["output"] = output.detach().clone()
    mlp = backend.model.transformer.h[9].mlp
    handles = [mlp.register_forward_pre_hook(pre), mlp.register_forward_hook(post)]
    try: output = call()
    finally:
        for handle in handles: handle.remove()
    if set(box) != {"input", "output"}: raise RuntimeError("MLP9 capture incomplete")
    return output, box

def run_arm(backend, batch, base_hidden, complement, positions, base_raw9, base_mlp9, delta):
    def patch(_module, _args, output):
        changed = output.clone()
        for i, query in enumerate(batch.semantic_positions):
            changed[i, :int(query)+1] = (base_mlp9[i, :int(query)+1].float()
                + delta[i, :int(query)+1].float()).to(changed)
        return changed
    handle = backend.model.transformer.h[9].mlp.register_forward_hook(patch)
    try:
        return auxiliary.run_heads(backend, batch, base_hidden, complement, positions,
            {9: base_raw9}, {9: tuple(range(int(backend.model.config.n_head)))})
    finally: handle.remove()

def state(output, rows, backend):
    return backend.torch.stack([backend.torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
        for row in rows]).to(backend.device).float()

def main():
    paths = {"prior": PRIOR, "direct": DIRECT, "direct_runner": DIRECT_RUNNER, "atlas": ATLAS,
        "greedy": GREEDY, "capability": CAPABILITY, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED: raise RuntimeError("MLP9 factor authority changed")
    prior, direct, atlas, greedy, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, DIRECT, ATLAS, GREEDY, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows() if row["family"] in ("A1","A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or direct.get("terminal") != "null"
            or atlas.get("terminal") != "screen" or greedy.get("terminal") != "screen" or len(rows) != 30):
        raise RuntimeError("parent decision or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "rows": len(rows), "arms": [arm_name(x) for x in subsets()],
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    family, _sv, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    q8 = torch.linalg.qr(modes, mode="reduced").Q
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_hidden = weight.capture_mlp8(backend, base_batch)
    _donor_hidden_output, donor_hidden = weight.capture_mlp8(backend, donor_batch)
    down8 = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _s, vh = torch.linalg.svd(q8.T@down8, full_matrices=False)
    hidden_delta = donor_hidden["hidden"].float()-base_hidden["hidden"].float()
    complement = hidden_delta-weight.project(hidden_delta, vh, vh.shape[0])
    raw9 = {}
    def raw_hook(_module, args): raw9["base"] = args[0].detach().clone()
    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(raw_hook)
    try: base_output, base9 = capture_mlp9(backend, base_batch, lambda: backend.native(base_batch, capture=True))
    finally: handle.remove()
    live_output, live9 = capture_mlp9(backend, base_batch, lambda: auxiliary.run_heads(backend, base_batch,
        base_hidden["hidden"], complement, positions, {9: raw9["base"]},
        {9: tuple(range(int(backend.model.config.n_head))) }))
    mlp9 = backend.model.transformer.h[9].mlp
    with torch.no_grad():
        l0, l1 = mlp9.Left(base9["input"]).float(), mlp9.Left(live9["input"]).float()
        if bool(mlp9.config.gated): l0, l1 = torch.nn.functional.silu(l0), torch.nn.functional.silu(l1)
        r0, r1 = mlp9.Right(base9["input"]).float(), mlp9.Right(live9["input"]).float()
        dl, dr = l1-l0, r1-r0
        hidden_factors = {"left_change": dl*r0, "right_change": l0*dr, "bilinear_interaction": dl*dr}
        output_factors = {name: torch.nn.functional.linear(value, mlp9.Down.weight.detach().float(), None)
            for name, value in hidden_factors.items()}
    factor_error = float((sum(output_factors.values())-(live9["output"].float()-base9["output"].float())).abs().max())
    outputs = {}
    for subset in subsets():
        delta = sum((output_factors[name] for name in subset), torch.zeros_like(base9["output"], dtype=torch.float32))
        outputs[arm_name(subset)] = run_arm(backend, base_batch, base_hidden["hidden"], complement,
            positions, raw9["base"], base9["output"], delta)
    forwards, evaluations = 10, 10*len(rows)
    base18, live18 = state(base_output, rows, backend), state(live_output, rows, backend)
    states = {name: state(output, rows, backend) for name, output in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda x: das.head_logits(backend,x)[index,answers]-das.head_logits(backend,x)[index,foils]
    empty, full = states["empty"], states[arm_name(FACTORS)]
    target_effect, target_q8 = margin(full)-margin(empty), (full-empty)@q8
    exact_effect, exact_q8 = margin(live18)-margin(empty), (live18-empty)@q8
    masks = {p: torch.as_tensor([row["family"]==p for row in rows],device=backend.device) for p in ("A1","A2")}
    metrics = {p:{} for p in masks}
    for panel, mask in masks.items():
        for name, value in states.items():
            effect, coord = (margin(value)-margin(empty))[mask], ((value-empty)@q8)[mask]
            metrics[panel][name] = {"behavior_fraction_of_full": float(effect.abs().mean()/target_effect[mask].abs().mean()),
                "behavior_cosine_to_full": cosine(effect,target_effect[mask]),
                "q8_norm_fraction_of_full": float(coord.norm()/target_q8[mask].norm()),
                "q8_cosine_to_full": cosine(coord.reshape(-1),target_q8[mask].reshape(-1))}
    replay_logit = float((das.head_logits(backend,full)-das.head_logits(backend,live18)).abs().max())
    replay_state = float((full-live18).norm()/(live18-empty).norm())
    finite = all(math.isfinite(v) for p in metrics.values() for a in p.values() for v in a.values())
    pred_a = orientation_error<=1e-6 and factor_error<=.001 and finite and forwards<=MAX_FORWARDS and evaluations<=MAX_EVALUATIONS
    pred_b = replay_logit<=1e-4 and replay_state<=1e-4 and cosine(target_effect,exact_effect)>=.999
    two = arm_name(FACTORS[:2])
    pred_c = all(metrics[p][two]["behavior_fraction_of_full"]>=.85 and metrics[p][two]["behavior_cosine_to_full"]>=.95
        and metrics[p][two]["q8_norm_fraction_of_full"]>=.85 for p in masks)
    pred_d = all(metrics[p]["bilinear_interaction"]["behavior_fraction_of_full"]<=.15
        and metrics[p]["bilinear_interaction"]["q8_norm_fraction_of_full"]<=.15 for p in masks)
    pred_e = set(outputs)=={arm_name(x) for x in subsets()}
    predictions = {"pred_a_authority_factor_closure_finiteness_and_price":bool(pred_a),
        "pred_b_full_factor_union_replays_exact_mlp9_correction":bool(pred_b),
        "pred_c_left_right_terms_are_sufficient":bool(pred_c),
        "pred_d_bilinear_interaction_is_secondary":bool(pred_d),
        "pred_e_zero_fit_literal_mlp9_weights":bool(pred_e)}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "bilinear_gate"
    result = {"schema":"iswas_mlp8_mlp9_bilinear_correction_program_result_v1","candidate_id":CANDIDATE_ID,
        "execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),
        "serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,
        "instrument":{"f_linear_orientation_max_abs":orientation_error,"factor_closure_max_abs":factor_error,
            "full_union_exact_logit_max_abs":replay_logit,"full_union_exact_resid18_relative_norm":replay_state,"rows":len(rows)},
        "metrics":metrics,"predictions":predictions,"terminal":terminal,
        "price":{"model_forwards":forwards,"example_evaluations":evaluations,"fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ("candidate_id","instrument","metrics","predictions","terminal","price")},sort_keys=True))

if __name__ == "__main__": main()
