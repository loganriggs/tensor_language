#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_head_partition pred_b_top_head_concentrated pred_c_top_head_family_stable pred_d_top3_replay_attention17 pred_e_top_head_directional
"""Fold the induced attention17 response into its nine native heads."""
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
from regional_grouped_interaction_tools import apply_rotary
from squared_attention_head_tools import projected_head_writes

RUNNER=Path(__file__).resolve(); HELPER=Path(helper.__file__).resolve()
PREREG=P/"SETTING2_REGIONAL_ATTENTION17_HEAD_RESPONSE_FOLD_V1_PREREGISTRATION.md"
ROWS=P/"ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING=P/"SETTING2_REGIONAL_ATTENTION17_HEAD_RESPONSE_FOLD_V1_BINDING.json"
PARENT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v3_result.json"
DOWNSTREAM=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_edit_downstream_response_census_v2_result.json"
OUT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17_head_response_fold_v1_result.json"
NAMES=tuple(x for layer in range(9,18) for x in (f"attn{layer}",f"mlp{layer}"))
PRICE={"physical_model_executions":12,"sequences_per_arm":48,"arms":2,"response_heads":9,"fits":0,"backwards":0,"parameter_updates":0}
EPS=torch.finfo(torch.float32).eps


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b): return float((a-b).norm()/b.norm().clamp_min(1e-30))
def cosine(a,b): return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))


def load_bound():
    binding=json.loads(BINDING.read_text()); files={"preregistration":PREREG,"rows":ROWS,"row_check":P/"regional_cue_row_check_v1.py","parent_result":PARENT,"downstream_census":DOWNSTREAM,"intervention_helper":HELPER,"rotary_helper":RUNNER.parent/"regional_grouped_interaction_tools.py","head_helper":RUNNER.parent/"squared_attention_head_tools.py"}
    if binding["files"]!={k:sha(v) for k,v in files.items()} or binding["price"]!=PRICE: raise ValueError("bound input or price changed")
    parent=json.loads(PARENT.read_text())
    downstream=json.loads(DOWNSTREAM.read_text())
    if parent["terminal"]!="valid_null" or not parent["predictions"]["pred_d_selected_material_and_exceeds_rr"] or downstream["terminal"]!="valid_downstream_response_census" or not downstream["predictions"]["pred_a_exact_census"]: raise ValueError("fixed edit lacks valid parent")
    rows=json.loads(ROWS.read_text())["rows"]; checks=validate(rows); buckets={}
    for i,row in enumerate(rows): buckets.setdefault(len(row["ids"]),[]).append(i)
    if len(rows)!=48 or 2*sum(math.ceil(len(v)/8) for v in buckets.values())!=12: raise ValueError("row/price changed")
    return {**binding,"row_checks":checks,"buckets":buckets},rows


def plan():
    bound,rows=load_bound(); return {"schema":"setting2_regional_attention17_head_response_fold_v1_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,"rows":len(rows),"heads":9,"price":PRICE,"row_checks":bound["row_checks"],"binding_sha256":sha(BINDING)}


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
    head17_error=0.0
    for layer in range(10,18):
        block=model.transformer.h[layer]; r=block.lambdas[0]*x+block.lambdas[1]*x0; normalized=F.rms_norm(r,(1152,)); first_in=first; a,first=block.attn(normalized,first)
        if layer==17:
            at17=block.attn; qraw17=at17.c_q(normalized).view(batch,length,9,128); kraw17=at17.c_k(normalized).view_as(qraw17); q2raw17=at17.c_q2(normalized).view_as(qraw17); k2raw17=at17.c_k2(normalized).view_as(qraw17); value17=at17.c_v(normalized).view_as(qraw17); value17=(1-at17.lamb)*value17+at17.lamb*first_in.view_as(value17); cos17,sin17=at17.rotary(qraw17)
            q17=apply_rotary(F.rms_norm(qraw17,(128,)),cos17,sin17); k17=apply_rotary(F.rms_norm(kraw17,(128,)),cos17,sin17); q217=apply_rotary(F.rms_norm(q2raw17,(128,)),cos17,sin17); k217=apply_rotary(F.rms_norm(k2raw17,(128,)),cos17,sin17)
            heads17=projected_head_writes(q17,k17,q217,k217,value17,at17.c_proj.weight); head17_error=float((heads17.sum(1)-a).norm()/a.norm().clamp_min(1e-30)); captures["attn17_heads"]=heads17
        m=block.mlp(F.rms_norm(r+a,(1152,))); captures[f"attn{layer}"]=a; captures[f"mlp{layer}"]=m; x=r+a+m
    logits=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
    return {"final":x,"logits":logits,"writes":captures,"head17_error":head17_error,"carry_sq":float((carry-carry_native).square().sum()),"carry_den":float(carry_native.square().sum()),"attn_sq":float((manual-a9_native).square().sum()),"attn_den":float(a9_native.square().sum())}


@torch.no_grad()
def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"): print(json.dumps(planned,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); bound,rows=load_bound(); state=model.state_dict(); lambdas=torch.stack([b.lambdas.detach().double().cpu() for b in model.transformer.h]); embed_c,write_c=helper.coefficients(lambdas); unembed=state["lm_head.weight"].double().cpu()
    gammas={name:float(torch.prod(lambdas[layer+1:18,0])) for layer in range(9,18) for name in (f"attn{layer}",f"mlp{layer}")}
    term_values=torch.empty(18,48,dtype=torch.float64); head_values=torch.empty(9,48,dtype=torch.float64); final_values=torch.empty(48,dtype=torch.float64); carry_num=carry_den=0.; attention_error_max=head17_output_error=0.; executions=0
    for _,indices in sorted(bound["buckets"].items()):
        for offset in range(0,len(indices),8):
            selected=indices[offset:offset+8]; tokens=torch.tensor([rows[i]["ids"] for i in selected],device="cuda"); native=run_batch(model,tokens,False,embed_c,write_c); edited=run_batch(model,tokens,True,embed_c,write_c); executions+=2
            carry_num+=native["carry_sq"]; carry_den+=native["carry_den"]; attention_error_max=max(attention_error_max,math.sqrt(native["attn_sq"]/max(native["attn_den"],1e-30))); head17_output_error=max(head17_output_error,native["head17_error"],edited["head17_error"])
            readers=torch.stack([unembed[rows[i]["uk_id"]]-unembed[rows[i]["us_id"]] for i in selected]); final_delta=(edited["final"][:,-1]-native["final"][:,-1]).double().cpu(); final_values[selected]=(final_delta*readers).sum(-1)
            for j,name in enumerate(NAMES): term_values[j,selected]=(gammas[name]*(edited["writes"][name][:,-1]-native["writes"][name][:,-1]).double().cpu()*readers).sum(-1)
            head_delta=(edited["writes"]["attn17_heads"][:,:,-1]-native["writes"]["attn17_heads"][:,:,-1]).double().cpu(); head_values[:,selected]=torch.einsum("bhd,bd->hb",head_delta,readers)
    dt=head_values[:,::2]-head_values[:,1::2]; total17=term_values[NAMES.index("attn17"),::2]-term_values[NAMES.index("attn17"),1::2]; final_delta=final_values[::2]-final_values[1::2]; response_closure=rel(dt.sum(0),total17); families=sorted(set(r["family"] for r in rows)); reports={}
    for head in range(9):
        family_reports={}
        for family in families:
            ids=[i//2 for i in range(0,48,2) if rows[i]["family"]==family]; family_reports[str(family)]={"to_attention17_norm_ratio":float(dt[head,ids].norm()/total17[ids].norm().clamp_min(1e-30)),"to_final_response_norm_ratio":float(dt[head,ids].norm()/final_delta[ids].norm().clamp_min(1e-30)),"aligned_fraction_of_attention17":float((dt[head,ids]*total17[ids]).sum()/total17[ids].square().sum().clamp_min(1e-30))}
        reports[str(head)]={"to_attention17_norm_ratio":float(dt[head].norm()/total17.norm().clamp_min(1e-30)),"to_final_response_norm_ratio":float(dt[head].norm()/final_delta.norm().clamp_min(1e-30)),"attention17_cosine":cosine(dt[head],total17),"family_reports":family_reports}
    ranking=sorted(range(9),key=lambda h:reports[str(h)]["to_attention17_norm_ratio"],reverse=True); top=ranking[0]; top3_error=rel(sum(dt[h] for h in ranking[:3]),total17); carry_error=math.sqrt(carry_num/max(carry_den,1e-30)); downstream=json.loads(DOWNSTREAM.read_text()); parent_ratio=downstream["term_reports"]["attn17"]["change_norm_ratio"]; observed_ratio=float(total17.norm()/final_delta.norm().clamp_min(1e-30)); parent_error=abs(observed_ratio-parent_ratio); instrument=executions==12 and carry_error<=1e-6 and attention_error_max<=1e-6 and head17_output_error<=1e-6 and response_closure<=2e-6 and parent_error<=1e-5 and bool(torch.isfinite(head_values).all())
    predictions={"pred_a_exact_head_partition":bool(instrument),"pred_b_top_head_concentrated":bool(instrument and reports[str(top)]["to_attention17_norm_ratio"]>=.50),"pred_c_top_head_family_stable":bool(instrument and all(v["to_attention17_norm_ratio"]>=.35 and v["aligned_fraction_of_attention17"]>0 for v in reports[str(top)]["family_reports"].values())),"pred_d_top3_replay_attention17":bool(instrument and top3_error<=.25),"pred_e_top_head_directional":bool(instrument and reports[str(top)]["attention17_cosine"]>=.70)}
    metrics={"carry_source_reconstruction_relative_error":carry_error,"manual_attention9_reconstruction_relative_error":attention_error_max,"attention17_head_output_reconstruction_relative_error":head17_output_error,"attention17_head_response_partition_relative_error":response_closure,"attention17_change_norm_ratio":observed_ratio,"parent_attention17_ratio_absolute_error":parent_error,"top3_attention17_replay_relative_error":top3_error,"physical_model_executions":executions,"rows":48,"pairs":24}
    result={"schema":"setting2_regional_attention17_head_response_fold_v1_result","terminal":"attention17_head_localized" if all(predictions.values()) else "valid_attention17_head_split" if instrument else "invalid","predictions":predictions,"metrics":metrics,"ranking":ranking,"head_reports":reports,"price":PRICE,"row_checks":planned["row_checks"],"checkpoint_weights_sha256":"680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3","binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"scope":"Opened-panel exact nine-head decomposition of the induced attention17 response to the fixed late-touching head9.8 QK1 edit; native head factors and output projection, no fit or causal head intervention."}
    managed.atomic_create_json(OUT,result); print(json.dumps({"terminal":result["terminal"],"predictions":predictions,"metrics":metrics,"ranking":ranking,"top":reports[str(top)]},indent=2)); assert instrument

if __name__=="__main__": main()
