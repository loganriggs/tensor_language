#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_census pred_b_direct_routing_insufficient pred_c_mlp9_prominent pred_d_top5_adequate pred_e_numerator_direction_survives
"""Decompose the downstream response to the fixed late-touching QK1 removal."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]; P=ROOT/"basis_aligned/polynomial_causal"
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_managed_runner as managed
import run_setting2_regional_head9_8_qk1_late_group_fresh_routing_v3 as helper
from regional_cue_row_check_v1 import validate

RUNNER=Path(__file__).resolve(); HELPER=Path(helper.__file__).resolve()
PREREG=P/"SETTING2_REGIONAL_QK1_EDIT_DOWNSTREAM_RESPONSE_CENSUS_V2_CORRECTION.md"
ROWS=P/"ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING=P/"SETTING2_REGIONAL_QK1_EDIT_DOWNSTREAM_RESPONSE_CENSUS_V2_BINDING.json"
PARENT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v3_result.json"
OUT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_edit_downstream_response_census_v2_result.json"
NAMES=tuple(x for layer in range(9,18) for x in (f"attn{layer}",f"mlp{layer}"))
PRICE={"physical_model_executions":12,"sequences_per_arm":48,"arms":2,"response_terms":18,"fits":0,"backwards":0,"parameter_updates":0}
EPS=torch.finfo(torch.float32).eps


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b): return float((a-b).norm()/b.norm().clamp_min(1e-30))
def cosine(a,b): return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))


def load_bound():
    binding=json.loads(BINDING.read_text()); files={"preregistration":PREREG,"rows":ROWS,"row_check":P/"regional_cue_row_check_v1.py","parent_result":PARENT,"intervention_helper":HELPER,"v1_invalid_result":ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_edit_downstream_response_census_v1_result.json"}
    if binding["files"]!={k:sha(v) for k,v in files.items()} or binding["price"]!=PRICE: raise ValueError("bound input or price changed")
    parent=json.loads(PARENT.read_text())
    if parent["terminal"]!="valid_null" or not parent["predictions"]["pred_d_selected_material_and_exceeds_rr"]: raise ValueError("fixed edit lacks valid parent")
    rows=json.loads(ROWS.read_text())["rows"]; checks=validate(rows); buckets={}
    for i,row in enumerate(rows): buckets.setdefault(len(row["ids"]),[]).append(i)
    if len(rows)!=48 or 2*sum(math.ceil(len(v)/8) for v in buckets.values())!=12: raise ValueError("row/price changed")
    return {**binding,"row_checks":checks,"buckets":buckets},rows


def plan():
    bound,rows=load_bound(); return {"schema":"setting2_regional_qk1_edit_downstream_response_census_v2_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,"rows":len(rows),"terms":NAMES,"price":PRICE,"row_checks":bound["row_checks"],"binding_sha256":sha(BINDING)}


@torch.no_grad()
def run_batch(model,tokens,edit,embed_c,write_c):
    batch,length=tokens.shape; x=F.rms_norm(model.transformer.wte(tokens),(1152,)); x0=x; first=None; early={}; captures={}
    for layer,block in enumerate(model.transformer.h[:8]):
        r=block.lambdas[0]*x+block.lambdas[1]*x0; a,first=block.attn(F.rms_norm(r,(1152,)),first); m=block.mlp(F.rms_norm(r+a,(1152,))); early[f"attn{layer}"]=a; early[f"mlp{layer}"]=m; x=r+a+m
    b8,b9=model.transformer.h[8],model.transformer.h[9]; r8=b8.lambdas[0]*x+b8.lambdas[1]*x0; a8,first=b8.attn(F.rms_norm(r8,(1152,)),first); m8=b8.mlp(F.rms_norm(r8+a8,(1152,))); x=r8+a8+m8
    r9=b9.lambdas[0]*x+b9.lambdas[1]*x0; norm9=F.rms_norm(r9,(1152,)); a9,first_out=b9.attn(norm9,first); a9_native=a9; at=b9.attn
    parts=[embed_c*x0]+[write_c[name]*early[name] for name in helper.SOURCE_NAMES[1:]]; carry=sum(parts); carry_native=b9.lambdas[0]*r8+b9.lambdas[1]*x0; rden=(r9.square().mean(-1,keepdim=True)+EPS).sqrt(); normalized=[p/rden for p in parts]; late_ids={helper.SOURCE_NAMES.index(n) for n in helper.LATE_NAMES}; late=sum(normalized[i] for i in range(17) if i in late_ids); rem=sum(normalized[i] for i in range(17) if i not in late_ids)
    qraw=at.c_q(norm9).view(batch,length,9,128); kraw=at.c_k(norm9).view_as(qraw); q2raw=at.c_q2(norm9).view_as(qraw); k2raw=at.c_k2(norm9).view_as(qraw); v=at.c_v(norm9).view_as(qraw); v=(1-at.lamb)*v+at.lamb*first.view_as(v)
    cos,sin=at.rotary(qraw); q=helper.rotate(F.rms_norm(qraw,(128,)),cos,sin); k=helper.rotate(F.rms_norm(kraw,(128,)),cos,sin); q2=helper.rotate(F.rms_norm(q2raw,(128,)),cos,sin); k2=helper.rotate(F.rms_norm(k2raw,(128,)),cos,sin)
    groups=[late,rem]; qg=[helper.rotate(z,cos,sin) for z in helper.group_project(groups,qraw[:,:,8],at.c_q.weight[8*128:9*128])]; kg=[helper.rotate(z,cos,sin) for z in helper.group_project(groups,kraw[:,:,8],at.c_k.weight[8*128:9*128])]
    s1=torch.einsum("bqhd,bkhd->bhqk",q,k)/128; s2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/128; terms=[torch.einsum("bqd,bkd->bqk",qg[u],kg[vv])/128 for u in range(2) for vv in range(2)]; mask=torch.tril(torch.ones(length,length,device=tokens.device,dtype=torch.bool)); pattern=(s1*s2).masked_fill(~mask,0); zall=torch.einsum("bhqk,bkhd->bhqd",pattern,v); manual=at.c_proj(zall.transpose(1,2).contiguous().view_as(norm9)); selected=(terms[0]+terms[1]+terms[2]).masked_fill(~mask,0); zsel=torch.einsum("bqk,bkd->bqd",selected*s2[:,8],v[:,:,8])
    if edit: a9=a9-zsel@at.c_proj.weight[:,8*128:9*128].T
    captures["attn9"]=a9; x=r9+a9; m9=b9.mlp(F.rms_norm(x,(1152,))); captures["mlp9"]=m9; x=x+m9; first=first_out
    for layer in range(10,18):
        block=model.transformer.h[layer]; r=block.lambdas[0]*x+block.lambdas[1]*x0; a,first=block.attn(F.rms_norm(r,(1152,)),first); m=block.mlp(F.rms_norm(r+a,(1152,))); captures[f"attn{layer}"]=a; captures[f"mlp{layer}"]=m; x=r+a+m
    logits=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
    return {"final":x,"logits":logits,"writes":captures,"carry_sq":float((carry-carry_native).square().sum()),"carry_den":float(carry_native.square().sum()),"attn_sq":float((manual-a9_native).square().sum()),"attn_den":float(a9_native.square().sum())}


@torch.no_grad()
def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"): print(json.dumps(planned,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); bound,rows=load_bound(); state=model.state_dict(); lambdas=torch.stack([b.lambdas.detach().double().cpu() for b in model.transformer.h]); embed_c,write_c=helper.coefficients(lambdas); unembed=state["lm_head.weight"].double().cpu()
    gammas={name:float(torch.prod(lambdas[layer+1:18,0])) for layer in range(9,18) for name in (f"attn{layer}",f"mlp{layer}")}
    term_values=torch.empty(18,48,dtype=torch.float64); final_values=torch.empty(48,dtype=torch.float64); actual_values=torch.empty(48,dtype=torch.float64); carry_num=carry_den=0.; attention_error_max=0.; executions=0
    for _,indices in sorted(bound["buckets"].items()):
        for offset in range(0,len(indices),8):
            selected=indices[offset:offset+8]; tokens=torch.tensor([rows[i]["ids"] for i in selected],device="cuda"); native=run_batch(model,tokens,False,embed_c,write_c); edited=run_batch(model,tokens,True,embed_c,write_c); executions+=2
            carry_num+=native["carry_sq"]; carry_den+=native["carry_den"]; attention_error_max=max(attention_error_max,math.sqrt(native["attn_sq"]/max(native["attn_den"],1e-30)))
            readers=torch.stack([unembed[rows[i]["uk_id"]]-unembed[rows[i]["us_id"]] for i in selected]); final_delta=(edited["final"][:,-1]-native["final"][:,-1]).double().cpu(); final_values[selected]=(final_delta*readers).sum(-1)
            for j,name in enumerate(NAMES): term_values[j,selected]=(gammas[name]*(edited["writes"][name][:,-1]-native["writes"][name][:,-1]).double().cpu()*readers).sum(-1)
            for j,row_id in enumerate(selected):
                row=rows[row_id]; actual_values[row_id]=(edited["logits"][j,row["uk_id"]]-edited["logits"][j,row["us_id"]]-native["logits"][j,row["uk_id"]]+native["logits"][j,row["us_id"]]).double().cpu()
    residual_closure=rel(term_values.sum(0),final_values); dt=term_values[:,::2]-term_values[:,1::2]; delta=final_values[::2]-final_values[1::2]; actual=actual_values[::2]-actual_values[1::2]; denom=delta.square().sum().clamp_min(1e-30); families=sorted(set(r["family"] for r in rows)); reports={}
    for j,name in enumerate(NAMES):
        fr={}
        for family in families:
            ids=[i//2 for i in range(0,48,2) if rows[i]["family"]==family]; fr[str(family)]=float(dt[j,ids].norm()/delta[ids].norm().clamp_min(1e-30))
        reports[name]={"change_norm_ratio":float(dt[j].norm()/delta.norm().clamp_min(1e-30)),"aligned_fraction":float((dt[j]*delta).sum()/denom),"family_change_norm_ratios":fr}
    ranking=sorted(reports,key=lambda n:reports[n]["change_norm_ratio"],reverse=True); idx={n:i for i,n in enumerate(NAMES)}; top5_error=rel(sum(dt[idx[n]] for n in ranking[:5]),delta); direct_error=rel(dt[idx["attn9"]],delta); raw_cos=cosine(delta,actual); raw_sign=float(((delta*actual)>0).double().mean()); carry_error=math.sqrt(carry_num/max(carry_den,1e-30)); instrument=executions==12 and carry_error<=1e-6 and residual_closure<=2e-6 and attention_error_max<=1e-6 and bool(torch.isfinite(term_values).all()) and bool(torch.isfinite(actual_values).all())
    predictions={"pred_a_exact_census":bool(instrument),"pred_b_direct_routing_insufficient":bool(instrument and direct_error>=.50),"pred_c_mlp9_prominent":bool(instrument and ranking.index("mlp9")<3 and reports["mlp9"]["change_norm_ratio"]>=.25),"pred_d_top5_adequate":bool(instrument and top5_error<=.35),"pred_e_numerator_direction_survives":bool(instrument and raw_cos>=.70 and raw_sign>=.70)}
    metrics={"carry_source_reconstruction_relative_error":carry_error,"manual_attention9_reconstruction_relative_error":attention_error_max,"final_residual_response_relative_closure":residual_closure,"direct_attention9_replay_relative_error":direct_error,"top5_replay_relative_error":top5_error,"pre_rms_numerator_to_logit_effect_cosine":raw_cos,"pre_rms_numerator_to_logit_effect_sign_agreement":raw_sign,"physical_model_executions":executions,"rows":48,"pairs":24}
    result={"schema":"setting2_regional_qk1_edit_downstream_response_census_v2_result","terminal":"valid_downstream_response_census" if instrument else "invalid","predictions":predictions,"metrics":metrics,"ranking":ranking,"top5_terms":ranking[:5],"term_reports":reports,"residual_propagation_coefficients":gammas,"price":PRICE,"row_checks":planned["row_checks"],"outcome_access":{"target_logits":True,"fits":False},"checkpoint_weights_sha256":"680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3","binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"scope":"Exact layer9-17 native write-response decomposition for the fixed late-touching head9.8 QK1 recursive removal on the opened 48-row diagnostic panel."}
    managed.atomic_create_json(OUT,result); print(json.dumps({"terminal":result["terminal"],"predictions":predictions,"metrics":metrics,"top5":ranking[:5],"top5_ratios":{n:reports[n]["change_norm_ratio"] for n in ranking[:5]}},indent=2)); assert instrument

if __name__=="__main__": main()
