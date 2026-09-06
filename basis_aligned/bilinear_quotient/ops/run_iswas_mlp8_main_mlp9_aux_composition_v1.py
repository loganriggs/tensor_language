#!/usr/bin/env python3
"""Compose the identified main, MLP9, and auxiliary branches on fresh-v11 rows."""

# BQGATE: EXPERIMENT pred_a_authority_capture_self_clamp_finiteness_and_price pred_b_joint_program_is_behaviorally_sufficient pred_c_joint_program_is_q8_sufficient pred_d_factorial_detects_nonlinear_composition pred_e_zero_fit_three_component_extraction
from datetime import datetime, timezone
import hashlib, itertools, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as converter
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_main_mlp9_aux_composition_v1.json"
MLP9 = ROOT / "circuits/followups/iswas_mlp8_mlp9_two_stream_fresh_v11_confirmation_v1_result.json"
MLP9_RUNNER = ROOT / "ops/run_iswas_mlp8_mlp9_two_stream_fresh_v11_confirmation_v1.py"
MAIN = ROOT / "circuits/followups/iswas_mlp8_complement_five_head_fresh_v10_confirmation_v1_result.json"
AUX = ROOT / "circuits/followups/iswas_mlp8_auxiliary_postcue_value_source_v12_transfer_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_main_mlp9_aux_composition_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_main_mlp9_aux_composition_v1"
COMPONENTS = ("main_l9h1h4", "mlp9", "aux_l11h1h3_l15h5")
ATTN_CLAMPS, MLP_CLAMPS = tuple(range(9,16)), tuple(range(9,15))
SELECTED = {9:(1,4), 11:(1,3), 15:(5,)}
EXPECTED = {"prior":"6c23e6480ee5e38881b93053ab4d32f0adb11c8ffe0618538c16c997bf2b4a4f", "mlp9":"b903ddd9c33060b6b52190ec36e0b8c2a887d3d0318cd8c9b9a228b7b80ecc38",
    "mlp9_runner":"6dd571c255bea083b89f51a9cb21bb940148c14f514f7d53cb58a0a29cc88279",
    "main":"62fe910a3c034d5f3121b12b331dc6d80416b32553997cc55c76574fce0a2567",
    "aux":"acacbd917e52a5cdc6d2b43bf374af3f35583376977b0d551bb0c57d78d2fab5",
    "capability":"6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
    "builder":"fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec"}
MAX_FORWARDS, MAX_EVALUATIONS = 13, 390

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def cosine(x,y):
    d=float(x.norm()*y.norm()); return float((x*y).sum())/d if d else 0.0
def arm_name(values): return "+".join(values) if values else "empty"
def subsets():
    for n in range(len(COMPONENTS)+1): yield from itertools.combinations(COMPONENTS,n)
def state(output, rows, backend):
    return backend.torch.stack([backend.torch.as_tensor(output.captured[(row["row_id"],"resid:18")])
        for row in rows]).to(backend.device).float()

def capture_modules(backend, call):
    cache={"attn":{},"mlp":{}}; handles=[]
    for layer in ATTN_CLAMPS:
        def hook(_module,args,layer=layer): cache["attn"][layer]=args[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
    for layer in MLP_CLAMPS:
        def hook(_module,_args,output,layer=layer): cache["mlp"][layer]=output.detach().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(hook))
    try: output=call()
    finally:
        for handle in handles: handle.remove()
    if set(cache["attn"])!=set(ATTN_CLAMPS) or set(cache["mlp"])!=set(MLP_CLAMPS):
        raise RuntimeError("module capture incomplete")
    return output,cache

def run_arm(backend,batch,base_hidden,complement,positions,base,live,subset,*,actuate=True):
    subset=set(subset); handles=[]; n_head=int(backend.model.config.n_head)
    head_dim=int(backend.model.config.n_embd//n_head)
    for layer in ATTN_CLAMPS:
        def patch(_module,args,layer=layer):
            raw=args[0]; changed=raw.clone().view(raw.shape[0],raw.shape[1],n_head,head_dim)
            b=base["attn"][layer].view_as(changed); l=live["attn"][layer].view_as(changed)
            for i,query in enumerate(batch.semantic_positions):
                changed[i,:int(query)+1]=b[i,:int(query)+1].to(changed)
                component = "main_l9h1h4" if layer==9 else "aux_l11h1h3_l15h5"
                if component in subset and layer in SELECTED:
                    heads=list(SELECTED[layer]); changed[i,:int(query)+1,heads]=(b[i,:int(query)+1,heads].float()
                        +(l[i,:int(query)+1,heads].float()-b[i,:int(query)+1,heads].float())).to(changed)
            return (changed.reshape_as(raw),)+tuple(args[1:])
        handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    for layer in MLP_CLAMPS:
        def patch(_module,_args,output,layer=layer):
            changed=output.clone()
            for i,query in enumerate(batch.semantic_positions):
                value=base["mlp"][layer][i,:int(query)+1].float()
                if layer==9 and "mlp9" in subset:
                    value=value+(live["mlp"][layer][i,:int(query)+1].float()-base["mlp"][layer][i,:int(query)+1].float())
                changed[i,:int(query)+1]=value.to(changed)
            return changed
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
    if actuate:
        handles.append(backend.model.transformer.h[8].mlp.Down.register_forward_pre_hook(
            converter.actuation_hook(base_hidden,complement,positions)))
    try: return backend.native(batch,capture=True)
    finally:
        for handle in handles: handle.remove()

def main():
    paths={"prior":PRIOR,"mlp9":MLP9,"mlp9_runner":MLP9_RUNNER,"main":MAIN,"aux":AUX,
        "capability":CAPABILITY,"builder":BUILDER}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED: raise RuntimeError("composition authority changed")
    prior,mlp9,main,aux,capability,subspace=[json.loads(p.read_text()) for p in (PRIOR,MLP9,MAIN,AUX,CAPABILITY,weight.SUBSPACE)]
    allowed={row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows=[row for row in candidate.build_rows() if row["family"] in ("A1","A2") and row["row_id"] in allowed]
    positions=[weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id")!=CANDIDATE_ID or mlp9.get("terminal")!="bilinear_gate"
            or main.get("terminal")!="screen" or aux.get("terminal")!="screen" or len(rows)!=30):
        raise RuntimeError("parent decision or population changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,
        "queue_touched":False,"rows":len(rows),"components":list(COMPONENTS),"arms":[arm_name(x) for x in subsets()],
        "model_forwards_max":MAX_FORWARDS,"example_evaluations_max":MAX_EVALUATIONS,
        "fit_updates":0,"transformer_backwards":0,"model_updates":0}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":
        print(json.dumps(dryrun,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter(); backend=producer.Bilin18TorchBackend.load("cuda"); torch=backend.torch
    family,_sv,_energy=family_builder.build_family(backend,subspace)
    gain=math.prod(float(backend.model.transformer.h[k].lambdas[0].detach().float()) for k in range(12,18))
    modes,orientation_error,_wrong=overlap.residual_modes(backend,family[8],gain); q8=torch.linalg.qr(modes,mode="reduced").Q
    base_batch,donor_batch=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor")
    base_hidden_output,base_hidden=weight.capture_mlp8(backend,base_batch)
    _donor_hidden_output,donor_hidden=weight.capture_mlp8(backend,donor_batch)
    down=backend.model.transformer.h[8].mlp.Down.weight.detach().float(); _u,_s,vh=torch.linalg.svd(q8.T@down,full_matrices=False)
    delta=donor_hidden["hidden"].float()-base_hidden["hidden"].float(); complement=delta-weight.project(delta,vh,vh.shape[0])
    base_output,base=capture_modules(backend,lambda:backend.native(base_batch,capture=True))
    live_output,live=capture_modules(backend,lambda: weight.run_hidden_patch(backend,base_batch,base_hidden["hidden"],complement,positions))
    outputs={arm_name(s):run_arm(backend,base_batch,base_hidden["hidden"],complement,positions,base,live,s) for s in subsets()}
    self_clamp=run_arm(backend,base_batch,base_hidden["hidden"],complement,positions,base,base,(),actuate=False)
    forwards,evaluations=13,13*len(rows)
    base18,live18=state(base_output,rows,backend),state(live_output,rows,backend); states={k:state(v,rows,backend) for k,v in outputs.items()}
    index=torch.arange(len(rows),device=backend.device); answers=torch.as_tensor([r["donor_answer_id"] for r in rows],device=backend.device)
    foils=torch.as_tensor([r["donor_foil_id"] for r in rows],device=backend.device)
    margin=lambda x:das.head_logits(backend,x)[index,answers]-das.head_logits(backend,x)[index,foils]
    live_effect=margin(live18)-margin(base18); live_q8=(live18-base18)@q8
    metrics={}; masks={p:torch.as_tensor([r["family"]==p for r in rows],device=backend.device) for p in ("A1","A2")}
    for panel,mask in masks.items():
        metrics[panel]={}
        for name,value in states.items():
            effect=(margin(value)-margin(base18))[mask]; coord=((value-base18)@q8)[mask]
            metrics[panel][name]={"behavior_fraction_of_live":float(effect.abs().mean()/live_effect[mask].abs().mean()),
                "behavior_cosine_to_live":cosine(effect,live_effect[mask]),"q8_norm_fraction_of_live":float(coord.norm()/live_q8[mask].norm()),
                "q8_cosine_to_live":cosine(coord.reshape(-1),live_q8[mask].reshape(-1))}
    full=arm_name(COMPONENTS); empty_error=float((state(self_clamp,rows,backend)-base18).abs().max())
    singleton_sum=sum((states[c]-states["empty"] for c in COMPONENTS),torch.zeros_like(base18))
    composition_interaction=float((states[full]-states["empty"]-singleton_sum).norm()/(states[full]-states["empty"]).norm())
    finite=all(math.isfinite(v) for p in metrics.values() for a in p.values() for v in a.values())
    pred_a=orientation_error<=1e-6 and empty_error<=.05 and finite and forwards<=MAX_FORWARDS and evaluations<=MAX_EVALUATIONS
    pred_b=all(metrics[p][full]["behavior_fraction_of_live"]>=.75 and metrics[p][full]["behavior_cosine_to_live"]>=.95 for p in masks)
    pred_c=all(metrics[p][full]["q8_norm_fraction_of_live"]>=.75 and metrics[p][full]["q8_cosine_to_live"]>=.95 for p in masks)
    pred_d=math.isfinite(composition_interaction) and set(outputs)=={arm_name(x) for x in subsets()}
    pred_e=set(COMPONENTS)=={"main_l9h1h4","mlp9","aux_l11h1h3_l15h5"}
    predictions={"pred_a_authority_capture_self_clamp_finiteness_and_price":bool(pred_a),
        "pred_b_joint_program_is_behaviorally_sufficient":bool(pred_b),"pred_c_joint_program_is_q8_sufficient":bool(pred_c),
        "pred_d_factorial_detects_nonlinear_composition":bool(pred_d),"pred_e_zero_fit_three_component_extraction":bool(pred_e)}
    terminal="invalid" if not pred_a else "screen" if all(predictions.values()) else "missing_branch"
    result={"schema":"iswas_mlp8_main_mlp9_aux_composition_result_v1","candidate_id":CANDIDATE_ID,
        "execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,
        "authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"f_linear_orientation_max_abs":orientation_error,
        "base_self_clamp_resid18_max_abs":empty_error,"composition_interaction_relative_norm":composition_interaction,"rows":len(rows)},
        "metrics":metrics,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":forwards,"example_evaluations":evaluations,
        "fit_updates":0,"transformer_backwards":0,"model_updates":0}}
    atomic_create_json(OUT,result); print(json.dumps({k:result[k] for k in ("candidate_id","instrument","metrics","predictions","terminal","price")},sort_keys=True))

if __name__=="__main__": main()
