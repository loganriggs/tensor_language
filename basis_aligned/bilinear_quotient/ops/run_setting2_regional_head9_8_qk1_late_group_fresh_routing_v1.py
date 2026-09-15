#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_intervention_instrument pred_b_attribution_transfers pred_c_folded_sign_predicts_edit pred_d_selected_material_and_exceeds_rr pred_e_target_reader_selectivity
"""Fresh recursive routing edits for the regional head9.8 QK1 late group."""
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
from regional_cue_row_check_v1 import validate

RUNNER=Path(__file__).resolve()
PREREG=P/"SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FRESH_ROUTING_V1_PREREGISTRATION.md"
ROWS=P/"ODD_FRAMING_FRESH_V1_ROWS.json"
BINDING=P/"SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FRESH_ROUTING_V1_BINDING.json"
PARENT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fold_v1_result.json"
OUT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v1_result.json"
SOURCE_NAMES=("embedding",)+tuple(x for layer in range(8) for x in (f"attn{layer}",f"mlp{layer}"))
LATE_NAMES=("attn5","mlp5","mlp6","mlp7")
ARMS=("native","remove_qk1_late_touching","remove_qk1_remainder_self","remove_qk2_late_touching","remove_current_value_late","remove_head9_8")
PRICE={"physical_model_executions":36,"sequences_per_arm":48,"arms":6,"carry_sources":17,"fits":0,"backwards":0,"parameter_updates":0}
EPS=torch.finfo(torch.float32).eps


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(actual,expected): return float((actual-expected).norm()/expected.norm().clamp_min(1e-30))
def cosine(a,b): return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))


def coefficients(lambdas):
    embed=1.0
    for layer in range(10): embed=float(lambdas[layer,0])*embed+float(lambdas[layer,1])
    writes={}
    for layer in range(8):
        c=float(torch.prod(lambdas[layer+1:10,0])); writes[f"attn{layer}"]=c; writes[f"mlp{layer}"]=c
    return embed,writes


def load_bound():
    binding=json.loads(BINDING.read_text())
    files={"preregistration":PREREG,"rows":ROWS,"row_check":P/"regional_cue_row_check_v1.py","parent_result":PARENT}
    if binding["files"]!={k:sha(v) for k,v in files.items()} or binding["price"]!=PRICE: raise ValueError("bound input or price changed")
    parent=json.loads(PARENT.read_text())
    if parent["terminal"]!="valid_qk1_late_group_fold" or not all(parent["predictions"].values()): raise ValueError("parent late grouping did not pass")
    rows=json.loads(ROWS.read_text())["rows"]; checks=validate(rows); buckets={}
    for i,row in enumerate(rows): buckets.setdefault(len(row["ids"]),[]).append(i)
    if len(rows)!=48 or len(rows)//2!=24 or len(ARMS)*sum(math.ceil(len(v)/8) for v in buckets.values())!=36: raise ValueError("row/price changed")
    return {**binding,"row_checks":checks,"buckets":buckets},rows,parent


def plan():
    bound,rows,_=load_bound()
    return {"schema":"setting2_regional_head9_8_qk1_late_group_fresh_routing_v1_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,"rows":len(rows),"arms":ARMS,"late_names":LATE_NAMES,"price":PRICE,"row_checks":bound["row_checks"],"binding_sha256":sha(BINDING)}


def rotate(x,cos,sin):
    d=x.shape[-1]//2; c=cos[:,:,0].to(x.dtype); s=sin[:,:,0].to(x.dtype)
    return torch.cat((x[...,:d]*c+x[...,d:]*s,x[...,:d]*(-s)+x[...,d:]*c),-1)


def group_project(groups,raw_head,weight):
    den=(raw_head.square().mean(-1,keepdim=True)+EPS).sqrt()
    return [g@weight.T/den for g in groups]


@torch.no_grad()
def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); bound,rows,parent=load_bound(); device=next(model.parameters()).device
    state=model.state_dict(); lambdas=torch.stack([b.lambdas.detach().double().cpu() for b in model.transformer.h]); embed_c,write_c=coefficients(lambdas)
    coefficient9=float(torch.prod(lambdas[10:18,0])); coefficient16=float(lambdas[17,0])
    projection9=state["transformer.h.9.attn.c_proj.weight"].double().cpu(); unembed=state["lm_head.weight"].double().cpu()
    left,right,down=[state[f"transformer.h.17.mlp.{n}.weight"].double().cpu() for n in ("Left","Right","Down")]
    values=torch.empty(len(ARMS),48,2,dtype=torch.float64); fold_selected=torch.empty(48,dtype=torch.float64); fold_rr=torch.empty(48,dtype=torch.float64); fold_complete=torch.empty(48,dtype=torch.float64)
    carry_num=carry_den=attn_num=attn_den=0.0; execution_count=0; late_indices={SOURCE_NAMES.index(n) for n in LATE_NAMES}
    for _,indices in sorted(bound["buckets"].items()):
        for offset in range(0,len(indices),8):
            selected=indices[offset:offset+8]; tokens=torch.tensor([rows[i]["ids"] for i in selected],device=device); batch=len(selected)
            for arm_index,arm in enumerate(ARMS):
                x=F.rms_norm(model.transformer.wte(tokens),(1152,)); x0=x; first_value=None; writes={}
                for layer,block in enumerate(model.transformer.h[:8]):
                    r=block.lambdas[0]*x+block.lambdas[1]*x0; a,first_value=block.attn(F.rms_norm(r,(1152,)),first_value); m=block.mlp(F.rms_norm(r+a,(1152,)))
                    writes[f"attn{layer}"]=a; writes[f"mlp{layer}"]=m; x=r+a+m
                block8,block9=model.transformer.h[8],model.transformer.h[9]
                r8=block8.lambdas[0]*x+block8.lambdas[1]*x0; a8,first_value=block8.attn(F.rms_norm(r8,(1152,)),first_value); m8=block8.mlp(F.rms_norm(r8+a8,(1152,))); x=r8+a8+m8
                r9=block9.lambdas[0]*x+block9.lambdas[1]*x0; norm9=F.rms_norm(r9,(1152,)); a9,first_out=block9.attn(norm9,first_value); at=block9.attn; length=tokens.shape[1]
                parts=[embed_c*x0]+[write_c[name]*writes[name] for name in SOURCE_NAMES[1:]]; carry=sum(parts); carry_native=block9.lambdas[0]*r8+block9.lambdas[1]*x0
                rden=(r9.square().mean(-1,keepdim=True)+EPS).sqrt(); normalized=[p/rden for p in parts]; late=sum(normalized[i] for i in range(17) if i in late_indices); rem=sum(normalized[i] for i in range(17) if i not in late_indices); groups=[late,rem]
                qraw=at.c_q(norm9).view(batch,length,9,128); kraw=at.c_k(norm9).view_as(qraw); q2raw=at.c_q2(norm9).view_as(qraw); k2raw=at.c_k2(norm9).view_as(qraw)
                v=at.c_v(norm9).view_as(qraw); v=(1-at.lamb)*v+at.lamb*first_value.view_as(v)
                cos,sin=at.rotary(qraw); q=rotate(F.rms_norm(qraw,(128,)),cos,sin); k=rotate(F.rms_norm(kraw,(128,)),cos,sin); q2=rotate(F.rms_norm(q2raw,(128,)),cos,sin); k2=rotate(F.rms_norm(k2raw,(128,)),cos,sin)
                qg=[rotate(z,cos,sin) for z in group_project(groups,qraw[:,:,8],at.c_q.weight[8*128:9*128])]; kg=[rotate(z,cos,sin) for z in group_project(groups,kraw[:,:,8],at.c_k.weight[8*128:9*128])]
                q2g=[rotate(z,cos,sin) for z in group_project(groups,q2raw[:,:,8],at.c_q2.weight[8*128:9*128])]; k2g=[rotate(z,cos,sin) for z in group_project(groups,k2raw[:,:,8],at.c_k2.weight[8*128:9*128])]
                s1=torch.einsum("bqhd,bkhd->bhqk",q,k)/128; s2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/128
                s1_terms=[torch.einsum("bqd,bkd->bqk",qg[u],kg[vv])/128 for u in range(2) for vv in range(2)]
                s2_terms=[torch.einsum("bqd,bkd->bqk",q2g[u],k2g[vv])/128 for u in range(2) for vv in range(2)]
                mask=torch.tril(torch.ones(length,length,device=device,dtype=torch.bool)); pattern=(s1*s2).masked_fill(~mask,0); zall=torch.einsum("bhqk,bkhd->bhqd",pattern,v)
                manual=at.c_proj(zall.transpose(1,2).contiguous().view_as(norm9)); attn_num+=float((manual-a9).square().sum()); attn_den+=float(a9.square().sum())
                selected_s1=(s1_terms[0]+s1_terms[1]+s1_terms[2]).masked_fill(~mask,0); rr_s1=s1_terms[3].masked_fill(~mask,0); selected_s2=(s2_terms[0]+s2_terms[1]+s2_terms[2]).masked_fill(~mask,0)
                z_selected=torch.einsum("bqk,bkd->bqd",selected_s1*s2[:,8],v[:,:,8]); z_rr=torch.einsum("bqk,bkd->bqd",rr_s1*s2[:,8],v[:,:,8]); z_qk2=torch.einsum("bqk,bkd->bqd",s1[:,8]*selected_s2,v[:,:,8])
                vlate=(1-at.lamb)*(late@at.c_v.weight[8*128:9*128].T); z_value=torch.einsum("bqk,bkd->bqd",pattern[:,8],vlate)
                z_edit={"remove_qk1_late_touching":z_selected,"remove_qk1_remainder_self":z_rr,"remove_qk2_late_touching":z_qk2,"remove_current_value_late":z_value,"remove_head9_8":zall[:,8]}.get(arm)
                if z_edit is not None: a9=a9+(-z_edit@at.c_proj.weight[:,8*128:9*128].T)
                x=r9+a9; x=x+block9.mlp(F.rms_norm(x,(1152,))); first_value=first_out; mlp16=None
                for layer in range(10,17):
                    block=model.transformer.h[layer]; r=block.lambdas[0]*x+block.lambdas[1]*x0; a,first_value=block.attn(F.rms_norm(r,(1152,)),first_value); m=block.mlp(F.rms_norm(r+a,(1152,))); x=r+a+m
                    if layer==16: mlp16=m[:,-1].double().cpu()
                last=model.transformer.h[17]; raw17=last.lambdas[0]*x+last.lambdas[1]*x0; a17,first_value=last.attn(F.rms_norm(raw17,(1152,)),first_value); pre17=(raw17+a17)[:,-1].double().cpu(); x=raw17+a17+last.mlp(F.rms_norm(raw17+a17,(1152,)))
                logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)).double().cpu()
                for j,row_id in enumerate(selected):
                    row=rows[row_id]; values[arm_index,row_id,0]=logits[j,row["uk_id"]]-logits[j,row["us_id"]]; values[arm_index,row_id,1]=logits[j,row["control_ids"][0]]-logits[j,row["control_ids"][1]]
                if arm_index==0:
                    carry_num+=float((carry-carry_native).square().sum()); carry_den+=float(carry_native.square().sum())
                    p=coefficient16*mlp16; readers=torch.stack([unembed[rows[i]["uk_id"]]-unembed[rows[i]["us_id"]] for i in selected]); fd=readers@down; lp,rp=p@left.T,p@right.T; d17=pre17.square().mean(-1)+EPS
                    residual_reader=((rp*fd)@left+(lp*fd)@right)/d17[:,None]; head_reader=coefficient9*residual_reader@projection9[:,8*128:9*128]
                    fold_selected[selected]=(z_selected[:,-1].double().cpu()*head_reader).sum(-1); fold_rr[selected]=(z_rr[:,-1].double().cpu()*head_reader).sum(-1); fold_complete[selected]=(zall[:,8,-1].double().cpu()*head_reader).sum(-1)
                execution_count+=1
    effects=values-values[0]; pair=lambda x:x[::2]-x[1::2]
    target=pair(effects[:,:,0]); folded_sel=pair(fold_selected); folded_rr=pair(fold_rr); folded_parent=folded_sel+folded_rr; folded_complete=pair(fold_complete); predicted=-folded_sel; actual=target[1]
    families=sorted(set(r["family"] for r in rows)); family_reports={}; replay_by_family={}
    for family in families:
        pair_ids=[i//2 for i in range(0,48,2) if rows[i]["family"]==family]; replay_by_family[str(family)]=rel(folded_sel[pair_ids],folded_parent[pair_ids])
        selected_effect=target[1,pair_ids]; rr_effect=target[2,pair_ids]; full_effect=target[5,pair_ids]; control_effect=effects[1,[i for i,r in enumerate(rows) if r["family"]==family],1]
        family_reports[str(family)]={"native_positive_pairs":int((pair(values[0,:,0])[pair_ids]>0).sum()),"selected_to_full_head_effect_norm_ratio":float(selected_effect.norm()/full_effect.norm().clamp_min(1e-30)),"selected_to_rr_effect_norm_ratio":float(selected_effect.norm()/rr_effect.norm().clamp_min(1e-30)),"selected_control_to_target_rms_ratio":float(control_effect.square().mean().sqrt()/selected_effect.square().mean().sqrt().clamp_min(1e-30)),"qk2_control_to_selected_effect_norm_ratio":float(target[3,pair_ids].norm()/selected_effect.norm().clamp_min(1e-30)),"value_control_to_selected_effect_norm_ratio":float(target[4,pair_ids].norm()/selected_effect.norm().clamp_min(1e-30)),"folded_selected_replay_relative_error":replay_by_family[str(family)]}
    carry_error=math.sqrt(carry_num/max(carry_den,1e-30)); attn_error=math.sqrt(attn_num/max(attn_den,1e-30)); replay=rel(folded_sel,folded_parent); signed_cos=cosine(predicted,actual); sign_agree=float(((predicted*actual)>0).double().mean()); selected_rr_ratio=float(actual.norm()/target[2].norm().clamp_min(1e-30))
    instrument=execution_count==36 and carry_error<=1e-6 and attn_error<=1e-6 and bool(torch.isfinite(values).all())
    predictions={"pred_a_exact_intervention_instrument":bool(instrument),"pred_b_attribution_transfers":bool(instrument and replay<=.25 and max(replay_by_family.values())<=.35),"pred_c_folded_sign_predicts_edit":bool(instrument and signed_cos>=.50 and sign_agree>=.70),"pred_d_selected_material_and_exceeds_rr":bool(instrument and all(v["selected_to_full_head_effect_norm_ratio"]>=.10 and v["selected_to_rr_effect_norm_ratio"]>=2 for v in family_reports.values()) and selected_rr_ratio>=2),"pred_e_target_reader_selectivity":bool(instrument and all(v["selected_control_to_target_rms_ratio"]<=.50 for v in family_reports.values()))}
    metrics={"carry_source_reconstruction_relative_error":carry_error,"manual_attention9_reconstruction_relative_error":attn_error,"folded_selected_replay_relative_error":replay,"folded_parent_to_complete_change_norm_ratio":float(folded_parent.norm()/folded_complete.norm().clamp_min(1e-30)),"folded_prediction_edit_cosine":signed_cos,"folded_prediction_edit_sign_agreement":sign_agree,"selected_to_rr_effect_norm_ratio":selected_rr_ratio,"physical_model_executions":execution_count,"rows":48,"pairs":24}
    result={"schema":"setting2_regional_head9_8_qk1_late_group_fresh_routing_v1_result","terminal":"fresh_selective_qk1_routing_component" if all(predictions.values()) else "valid_null" if instrument else "invalid","predictions":predictions,"metrics":metrics,"family_reports":family_reports,"arms":ARMS,"late_source_names":LATE_NAMES,"price":PRICE,"row_checks":planned["row_checks"],"outcome_access":{"target_logits":True,"unrelated_reader":True,"fits":False},"checkpoint_weights_sha256":"680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3","binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"scope":"Fresh-relative-to-grouping recursive removal of three late-touching head9.8 QK1 carry blocks; QK2, current-value, remainder-self, full-head, and unrelated-reader controls; fixed 48-row controlled regional panel."}
    managed.atomic_create_json(OUT,result); print(json.dumps({"terminal":result["terminal"],"predictions":predictions,"metrics":metrics,"families":family_reports},indent=2)); assert instrument

if __name__=="__main__": main()
