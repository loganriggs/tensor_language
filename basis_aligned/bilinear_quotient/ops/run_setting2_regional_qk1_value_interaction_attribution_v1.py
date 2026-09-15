#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_both_single_branches_live pred_c_explicit_cross_explains_recursive_interaction pred_d_joint_family_transfer pred_e_unrelated_reader_selectivity
"""Attribute recursive interaction to the explicit head9.8 routing-value cross term."""
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
from odd_attention8h2_o_composed_edge_v1 import head8_write
from odd_framing_role_split_v1 import role_masks
from regional_cue_row_check_v1 import validate
from regional_grouped_interaction_tools import apply_rotary, paired_difference

RUNNER=Path(__file__).resolve()
PREREG=P/"SETTING2_REGIONAL_QK1_VALUE_INTERACTION_ATTRIBUTION_V1_PREREGISTRATION.md"
ROWS=P/"SETTING2_REGIONAL_QK1_VALUE_COMPOSITION_FRESH_V1_ROWS.json"
BINDING=P/"SETTING2_REGIONAL_QK1_VALUE_INTERACTION_ATTRIBUTION_V1_BINDING.json"
EDGE=P/"ODD_ATTENTION8H2_ROUTING_CLOSURE_V1_PROGRAM.pt"
PARENT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fold_v1_result.json"
COMPOSITION=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_value_composition_fresh_v1_result.json"
OUT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_value_interaction_attribution_v1_result.json"
SOURCE_NAMES=("embedding",)+tuple(x for layer in range(8) for x in (f"attn{layer}",f"mlp{layer}"))
LATE_NAMES=("attn5","mlp5","mlp6","mlp7")
ARMS=("native","routing_only","value_only","additive_without_cross","joint")
READOUTS=(("target",None),("work_jobs",(670,3946)),("cat_dog",(3797,3290)),("red_blue",(2266,4171)),("monday_tuesday",(3321,3431)),("apple_orange",(17180,10912)))
PRICE={"physical_model_executions":30,"sequences_per_arm":48,"arms":5,"fits":0,"backwards":0,"parameter_updates":0}
EPS=torch.finfo(torch.float32).eps


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(actual,expected): return float((actual-expected).norm()/expected.norm().clamp_min(1e-30))
def cosine(a,b): return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))


def coefficients(lambdas):
    embed=1.0
    for layer in range(10): embed=float(lambdas[layer,0])*embed+float(lambdas[layer,1])
    writes={}
    for layer in range(8):
        coefficient=float(torch.prod(lambdas[layer+1:10,0])); writes[f"attn{layer}"]=coefficient; writes[f"mlp{layer}"]=coefficient
    return embed,writes


def load_bound():
    binding=json.loads(BINDING.read_text())
    files={"preregistration":PREREG,"rows":ROWS,"row_check":P/"regional_cue_row_check_v1.py",
           "role_masks":P/"odd_framing_role_split_v1.py","head8_edge":P/"odd_attention8h2_o_composed_edge_v1.py",
           "grouped_tools":RUNNER.parent/"regional_grouped_interaction_tools.py",
           "edge_program":EDGE,"parent_result":PARENT,"composition_result":COMPOSITION}
    if binding["files"]!={k:sha(v) for k,v in files.items()} or binding["price"]!=PRICE: raise ValueError("bound input or price changed")
    rows_doc=json.loads(ROWS.read_text()); rows=rows_doc["rows"]; checks=validate(rows); buckets={}
    for i,row in enumerate(rows): buckets.setdefault(len(row["ids"]),[]).append(i)
    parent=json.loads(PARENT.read_text())
    composition=json.loads(COMPOSITION.read_text())
    if rows_doc["prior_context_overlap"]!=0 or parent["terminal"]!="valid_qk1_late_group_fold" or not all(parent["predictions"].values()) or composition["terminal"]!="routing_value_composition" or not all(composition["predictions"].values()): raise ValueError("authority invalid")
    executions=len(ARMS)*sum(math.ceil(len(v)/8) for v in buckets.values())
    if len(rows)!=48 or executions!=PRICE["physical_model_executions"]: raise ValueError("row/price changed")
    return binding,rows,checks,buckets


def plan():
    _,rows,checks,_=load_bound()
    return {"schema":"setting2_regional_qk1_value_interaction_attribution_v1_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,"rows":len(rows),"arms":ARMS,"late_names":LATE_NAMES,"price":PRICE,"row_checks":checks,"binding_sha256":sha(BINDING)}


def grouped_projection(groups,raw,weight):
    denominator=(raw.square().mean(-1,keepdim=True)+EPS).sqrt()
    return [group@weight.T/denominator for group in groups]


@torch.no_grad()
def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); binding,rows,checks,buckets=load_bound(); device=next(model.parameters()).device
    edge={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in torch.load(EDGE,map_location="cpu",weights_only=True).items()}
    roles=role_masks(rows); lambdas=torch.stack([b.lambdas.detach().double().cpu() for b in model.transformer.h]); embed_c,write_c=coefficients(lambdas)
    values=torch.empty(len(ARMS),2,len(rows),len(READOUTS),dtype=torch.float64)
    head_norms=torch.empty(len(rows),3,dtype=torch.float64)
    carry_num=carry_den=attention_num=attention_den=half_num=half_den=joint_num=joint_den=0.0; executions=0
    late_indices={SOURCE_NAMES.index(name) for name in LATE_NAMES}
    for _,indices in sorted(buckets.items()):
        for offset in range(0,len(indices),8):
            selected=indices[offset:offset+8]; tokens=torch.tensor([rows[i]["ids"] for i in selected],device=device); batch=len(selected); length=tokens.shape[1]
            for arm_index,arm in enumerate(ARMS):
                x=F.rms_norm(model.transformer.wte(tokens),(1152,)); x0=x; first=None; writes={}
                for layer,block in enumerate(model.transformer.h[:8]):
                    residual=block.lambdas[0]*x+block.lambdas[1]*x0; attention,first=block.attn(F.rms_norm(residual,(1152,)),first); mlp=block.mlp(F.rms_norm(residual+attention,(1152,))); writes[f"attn{layer}"]=attention; writes[f"mlp{layer}"]=mlp; x=residual+attention+mlp
                block8,block9=model.transformer.h[8],model.transformer.h[9]
                residual8=block8.lambdas[0]*x+block8.lambdas[1]*x0; norm8=F.rms_norm(residual8,(1152,)); attention8,first=block8.attn(norm8,first); mlp8=block8.mlp(F.rms_norm(residual8+attention8,(1152,))); x=residual8+attention8+mlp8
                residual9=block9.lambdas[0]*x+block9.lambdas[1]*x0; norm9=F.rms_norm(residual9,(1152,)); attention9,first_out=block9.attn(norm9,first); at=block9.attn
                parts=[embed_c*x0]+[write_c[name]*writes[name] for name in SOURCE_NAMES[1:]]; carry=sum(parts); carry_native=block9.lambdas[0]*residual8+block9.lambdas[1]*x0
                denominator=(residual9.square().mean(-1,keepdim=True)+EPS).sqrt(); normalized=[part/denominator for part in parts]; late=sum(normalized[i] for i in late_indices); remainder=sum(normalized[i] for i in range(len(normalized)) if i not in late_indices)
                qraw=at.c_q(norm9).view(batch,length,9,128); kraw=at.c_k(norm9).view_as(qraw); q2raw=at.c_q2(norm9).view_as(qraw); k2raw=at.c_k2(norm9).view_as(qraw)
                value=at.c_v(norm9).view_as(qraw); value=(1-at.lamb)*value+at.lamb*first.view_as(value)
                cos,sin=at.rotary(qraw); q=apply_rotary(F.rms_norm(qraw,(128,)),cos,sin); k=apply_rotary(F.rms_norm(kraw,(128,)),cos,sin); q2=apply_rotary(F.rms_norm(q2raw,(128,)),cos,sin); k2=apply_rotary(F.rms_norm(k2raw,(128,)),cos,sin)
                groups=(late,remainder); qgroups=[apply_rotary(z,cos,sin) for z in grouped_projection(groups,qraw[:,:,8],at.c_q.weight[8*128:9*128])]; kgroups=[apply_rotary(z,cos,sin) for z in grouped_projection(groups,kraw[:,:,8],at.c_k.weight[8*128:9*128])]
                s1=torch.einsum("bqhd,bkhd->bhqk",q,k)/128; s2=torch.einsum("bqhd,bkhd->bhqk",q2,k2)/128
                terms=[torch.einsum("bqd,bkd->bqk",qgroups[u],kgroups[v])/128 for u in range(2) for v in range(2)]
                mask=torch.tril(torch.ones(length,length,device=device,dtype=torch.bool)); pattern=(s1*s2).masked_fill(~mask,0); selected_pattern=((terms[0]+terms[1]+terms[2])*s2[:,8]).masked_fill(~mask,0)
                zall=torch.einsum("bhqk,bkhd->bhqd",pattern,value); manual=at.c_proj(zall.transpose(1,2).contiguous().view_as(norm9)); attention_num+=float((manual-attention9).square().sum()); attention_den+=float(attention9.square().sum())
                delta_values=torch.zeros(batch,length,128,device=device,dtype=value.dtype); full_writes=[]; half_writes=[]
                for j,row_id in enumerate(selected):
                    city=int(torch.nonzero(torch.tensor(rows[row_id]["ids"],device=device)!=torch.tensor(rows[row_id^1]["ids"],device=device))[0]); ids0=tokens[j:j+1]; idsd=torch.tensor([rows[row_id^1]["ids"]],device=device)
                    recipient=F.rms_norm(model.transformer.wte(ids0),(1152,))[:,city]; donor=F.rms_norm(model.transformer.wte(idsd),(1152,))[:,city]; midpoint=(recipient+donor)/2
                    full=head8_write(edge,norm8[j:j+1],city,recipient,donor); half=head8_write(edge,norm8[j:j+1],city,recipient,midpoint); full_writes.append(full); half_writes.append(half); half_num+=float((half-.5*full).square().sum()); half_den+=float(full.square().sum())
                    framing=roles[row_id]["framing"].to(device); changed_x=x[j:j+1]+half*framing[None,:,None]; changed9=F.rms_norm(block9.lambdas[0]*changed_x+block9.lambdas[1]*x0[j:j+1],(1152,)); changed_current=at.c_v(changed9).view(1,length,9,128)[:,:,8]
                    delta_values[j,framing]=(1-at.lamb)*(changed_current[:,framing]-at.c_v(norm9[j:j+1]).view(1,length,9,128)[:,:,8][:,framing])
                native_head=value[:,:,8]; route_delta=-torch.einsum("bqk,bkd->bqd",selected_pattern,native_head); value_delta=torch.einsum("bqk,bkd->bqd",pattern[:,8],delta_values); cross_delta=-torch.einsum("bqk,bkd->bqd",selected_pattern,delta_values); joint_delta=route_delta+value_delta+cross_delta
                direct_joint=torch.einsum("bqk,bkd->bqd",pattern[:,8]-selected_pattern,native_head+delta_values)-torch.einsum("bqk,bkd->bqd",pattern[:,8],native_head)
                joint_num+=float((joint_delta-direct_joint).square().sum()); joint_den+=float(direct_joint.square().sum())
                edit={"routing_only":route_delta,"value_only":value_delta,"additive_without_cross":route_delta+value_delta,"joint":joint_delta}.get(arm)
                if edit is not None: attention9=attention9+edit@at.c_proj.weight[:,8*128:9*128].T
                if arm_index==0:
                    carry_num+=float((carry-carry_native).square().sum()); carry_den+=float(carry_native.square().sum())
                    head_norms[selected,0]=route_delta[:,-1].double().cpu().norm(dim=-1); head_norms[selected,1]=value_delta[:,-1].double().cpu().norm(dim=-1); head_norms[selected,2]=cross_delta[:,-1].double().cpu().norm(dim=-1)
                x=residual9+attention9; x=x+block9.mlp(F.rms_norm(x,(1152,))); first=first_out
                for block in model.transformer.h[10:]:
                    residual=block.lambdas[0]*x+block.lambdas[1]*x0; attention,first=block.attn(F.rms_norm(residual,(1152,)),first); x=residual+attention+block.mlp(F.rms_norm(residual+attention,(1152,)))
                numerator=model.lm_head(F.rms_norm(x[:,-1],(1152,))).double().cpu(); logits=30*torch.tanh(numerator/30)
                for j,row_id in enumerate(selected):
                    pairs=[(rows[row_id]["uk_id"],rows[row_id]["us_id"])]+[pair for _,pair in READOUTS[1:]]
                    for reader,(left,right) in enumerate(pairs): values[arm_index,0,row_id,reader]=numerator[j,left]-numerator[j,right]; values[arm_index,1,row_id,reader]=logits[j,left]-logits[j,right]
                executions+=1
    effects=values-values[0]; pair_effect=paired_difference(effects,2); native_pairs=paired_difference(values[0],1)
    family_reports={}; families=sorted(set(row["family"] for row in rows))
    for family in families:
        row_ids=[i for i,row in enumerate(rows) if row["family"]==family]; pair_ids=[i//2 for i in row_ids[::2]]
        report={"native_positive_pairs":int((native_pairs[1,pair_ids,0]>0).sum())}
        for scale,name in enumerate(("numerator","logit")):
            route=pair_effect[1,scale,pair_ids,0]; value_effect=pair_effect[2,scale,pair_ids,0]; additive=pair_effect[3,scale,pair_ids,0]; joint=pair_effect[4,scale,pair_ids,0]; interaction=joint-route-value_effect; explicit_cross=joint-additive
            controls={READOUTS[j][0]:float(effects[4,scale,row_ids,j].square().mean().sqrt()/joint.square().mean().sqrt().clamp_min(1e-30)) for j in range(1,len(READOUTS))}
            report[name]={"routing_target_rms":float(route.square().mean().sqrt()),"value_target_rms":float(value_effect.square().mean().sqrt()),"joint_target_rms":float(joint.square().mean().sqrt()),"interaction_target_rms":float(interaction.square().mean().sqrt()),"joint_additive_cosine":cosine(joint,route+value_effect),"signed_joint_mean":float(joint.mean()),"explicit_cross_target_rms":float(explicit_cross.square().mean().sqrt()),"explicit_cross_to_factorial_cosine":cosine(explicit_cross,interaction),"explicit_cross_to_factorial_relative_error":rel(explicit_cross,interaction),"unrelated_reader_ratios":controls}
        smaller=torch.minimum(head_norms[row_ids,0],head_norms[row_ids,1]); report["head_cross_to_smaller_single_mean_ratio"]=float((head_norms[row_ids,2]/smaller.clamp_min(1e-30)).mean())
        logit=report["logit"]; report["recursive_interaction_to_smaller_single_ratio"]=logit["interaction_target_rms"]/max(min(logit["routing_target_rms"],logit["value_target_rms"]),1e-30); family_reports[str(family)]=report
    carry_error=math.sqrt(carry_num/max(carry_den,1e-30)); attention_error=math.sqrt(attention_num/max(attention_den,1e-30)); half_error=math.sqrt(half_num/max(half_den,1e-30)); joint_error=math.sqrt(joint_num/max(joint_den,1e-30)); instrument=executions==PRICE["physical_model_executions"] and max(carry_error,attention_error,half_error,joint_error)<=2e-6 and bool(torch.isfinite(values).all())
    predictions={
        "pred_a_exact_instrument":bool(instrument),
        "pred_b_both_single_branches_live":bool(instrument and sum(r["native_positive_pairs"] for r in family_reports.values())>=24 and all(r["logit"]["routing_target_rms"]>=.002 and r["logit"]["value_target_rms"]>=.002 for r in family_reports.values())),
        "pred_c_explicit_cross_explains_recursive_interaction":bool(instrument and all(r["logit"]["explicit_cross_to_factorial_cosine"]>=.90 and r["logit"]["explicit_cross_to_factorial_relative_error"]<=.35 for r in family_reports.values())),
        "pred_d_joint_family_transfer":bool(instrument and all(r["logit"]["joint_additive_cosine"]>=.5 for r in family_reports.values())),
        "pred_e_unrelated_reader_selectivity":bool(instrument and all(max(r["logit"]["unrelated_reader_ratios"].values())<=.75 for r in family_reports.values())),
    }
    metrics={"carry_source_reconstruction_relative_error":carry_error,"manual_attention9_reconstruction_relative_error":attention_error,"head8_midpoint_half_scaling_relative_error":half_error,"joint_head_algebra_relative_error":joint_error,"physical_model_executions":executions,"rows":len(rows),"pairs":len(rows)//2}
    result={"schema":"setting2_regional_qk1_value_interaction_attribution_v1_result","terminal":"explicit_head_cross_attribution" if all(predictions.values()) else "valid_null" if instrument else "invalid","predictions":predictions,"metrics":metrics,"family_reports":family_reports,"arms":ARMS,"readouts":[name for name,_ in READOUTS],"late_source_names":LATE_NAMES,"price":PRICE,"row_checks":checks,"binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),"checkpoint_weights_sha256":"680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"scope":"Opened-panel five-arm attribution of recursive routing-value nonadditivity to the explicit head9.8 cross term; full-head current-value routing at framing sources, five unrelated readers, no fit."}
    managed.atomic_create_json(OUT,result); print(json.dumps({"terminal":result["terminal"],"predictions":predictions,"metrics":metrics,"families":family_reports},indent=2)); assert instrument


if __name__=="__main__": main()
