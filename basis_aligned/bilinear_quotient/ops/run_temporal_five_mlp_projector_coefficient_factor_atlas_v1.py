#!/usr/bin/env python3
"""Exact operation/source factor atlas for the eight weight-executable response coordinates."""
# BQGATE: EXPERIMENT pred_a_authority_hash_factor_closure_finiteness_and_price pred_b_attention_value_transport_dominates pred_c_attention_causal_suffix_dominates pred_d_mlp_linear_changes_dominate_interaction pred_e_factor_profiles_are_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import attention_source_destination_eval as attention_eval
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_five_mlp_projector_coefficient_factor_atlas_v1.json"
LITERAL=ROOT/"circuits/followups/temporal_five_mlp_literal_projector_weight_execution_v1_result.json"
ATTNLIB=ROOT/"ops/attention_source_destination_eval.py"
OUT=ROOT/"circuits/followups/temporal_five_mlp_projector_coefficient_factor_atlas_v1_result.json"
EXPECTED={"prior":"2b90df7395990fdad5cc4498d53c67a114ca25207fb088eeec9d2d4623927f4b","literal":"0cc7e19a287b433511127692a0a13c1d86447a7818d9efd8aa0e6325d52fedd1","attention_library":"608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd"}
ATTN_FACTORS=("pattern_on_base_value","base_pattern_on_value_change","pattern_value_interaction")
REGIONS=("pre_cue","cue","post_cue")
MLP_FACTORS=("left_change","right_change","bilinear_interaction")

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def norm(x):return float(x.square().sum().sqrt())
def cosine(a,b):return float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))

def capture_all(backend,batch):
 attn_layers=sorted({atlasrun.site_parts(s)[1] for s in comp.SITES if atlasrun.site_parts(s)[0]=="attn"})
 mlp_layers=sorted({atlasrun.site_parts(s)[1] for s in comp.SITES if atlasrun.site_parts(s)[0]=="mlp"})
 attn={};inputs={};handles=[]
 for layer in attn_layers:
  module=backend.model.transformer.h[layer].attn;item={};attn[layer]=item
  def ain(_m,args,module=module,item=item):
   p,v,h=attention_eval._attention_terms(backend,module,args[0],args[1] if len(args)>1 else None)
   item.update(pattern=p.detach().clone(),value=v.detach().clone(),reconstructed=h.detach().clone())
  def hout(_m,args,item=item):
   x=args[0];item["head_output"]=x.detach().clone().view(len(batch.row_ids),x.shape[1],backend.model.config.n_head,-1)
  handles += [module.register_forward_pre_hook(ain),module.c_proj.register_forward_pre_hook(hout)]
 for layer in mlp_layers:
  module=backend.model.transformer.h[layer].mlp
  def minput(_m,args,layer=layer):inputs[layer]=args[0].detach().clone()
  handles.append(module.register_forward_pre_hook(minput))
 try:output=backend.native(batch,capture=True)
 finally:
  for h in handles:h.remove()
 if set(attn)!=set(attn_layers) or set(inputs)!=set(mlp_layers) or any(set(x)!={"pattern","value","reconstructed","head_output"} for x in attn.values()):raise RuntimeError("factor capture incomplete")
 reconstruction=max(float((x["reconstructed"].float()-x["head_output"].float()).abs().max()) for x in attn.values())
 return output,attn,inputs,reconstruction

def cue_positions(rows):
 out=[]
 for r in rows:
  diff=[i for i,(a,b) in enumerate(zip(r["base_ids"],r["donor_ids"])) if a!=b]
  if len(r["base_ids"])!=len(r["donor_ids"]) or len(diff)!=1:raise RuntimeError("factor atlas requires one aligned cue")
  out.append(diff[0])
 return out

def attention_cell(backend,rows,ids,site,q,base,donor,cues):
 torch=backend.torch;_kind,layer,head=atlasrun.site_parts(site);parts={f"{r}::{f}":[] for r in REGIONS for f in ATTN_FACTORS};exact=[]
 for i in ids:
  stop=int(rows[i]["base_semantic_position"])+1;cue=cues[i]
  for query in range(stop):
   exact.append((donor[layer]["head_output"][i,query,head].float()-base[layer]["head_output"][i,query,head].float()).to(backend.device)@q)
   p0=base[layer]["pattern"][i,head,query,:query+1].float().to(backend.device);p1=donor[layer]["pattern"][i,head,query,:query+1].float().to(backend.device)
   v0=base[layer]["value"][i,:query+1,head].float().to(backend.device);v1=donor[layer]["value"][i,:query+1,head].float().to(backend.device);dp,dv=p1-p0,v1-v0
   indices={"pre_cue":[j for j in range(query+1) if j<cue],"cue":[j for j in range(query+1) if j==cue],"post_cue":[j for j in range(query+1) if j>cue]}
   for region,pos in indices.items():
    if not pos:
     zero=q.new_zeros(q.shape[1]);[parts[f"{region}::{factor}"].append(zero) for factor in ATTN_FACTORS];continue
    ix=torch.as_tensor(pos,device=backend.device);terms={"pattern_on_base_value":(dp[ix,None]*v0[ix]).sum(0),"base_pattern_on_value_change":(p0[ix,None]*dv[ix]).sum(0),"pattern_value_interaction":(dp[ix,None]*dv[ix]).sum(0)}
    for factor,value in terms.items():parts[f"{region}::{factor}"].append(value@q)
 tensors={k:torch.stack(v) for k,v in parts.items()};full=torch.stack(exact);recon=sum(tensors.values(),torch.zeros_like(full));factor={f:sum((tensors[f"{r}::{f}"] for r in REGIONS),torch.zeros_like(full)) for f in ATTN_FACTORS};region={r:sum((tensors[f"{r}::{f}"] for f in ATTN_FACTORS),torch.zeros_like(full)) for r in REGIONS};den=norm(full) or 1e-30
 return {"complete_norm":norm(full),"factor_norm_fraction":{k:norm(v)/den for k,v in factor.items()},"region_norm_fraction":{k:norm(v)/den for k,v in region.items()},"causal_suffix_norm_fraction":norm(region["cue"]+region["post_cue"])/den,"closure_rse":float((recon-full).square().sum()/full.square().sum().clamp_min(1e-30)),"profile":backend.torch.as_tensor([norm(factor[k]) for k in ATTN_FACTORS]+[norm(region[k]) for k in REGIONS],device=backend.device)}

def mlp_cell(backend,rows,ids,site,q,base_inputs,donor_inputs):
 torch=backend.torch;_kind,layer,_head=atlasrun.site_parts(site);m=backend.model.transformer.h[layer].mlp;xb,xd=base_inputs[layer],donor_inputs[layer]
 with torch.no_grad():l0,l1=m.Left(xb).float(),m.Left(xd).float();r0,r1=m.Right(xb).float(),m.Right(xd).float();dl,dr=l1-l0,r1-r0;w=m.Down.weight.detach().float().T@q
 values={k:[] for k in MLP_FACTORS};exact=[]
 for i in ids:
  stop=int(rows[i]["base_semantic_position"])+1;values["left_change"].append((dl[i,:stop]*r0[i,:stop])@w);values["right_change"].append((l0[i,:stop]*dr[i,:stop])@w);values["bilinear_interaction"].append((dl[i,:stop]*dr[i,:stop])@w);exact.append(((l1[i,:stop]*r1[i,:stop]-l0[i,:stop]*r0[i,:stop])@w))
 values={k:torch.cat(v) for k,v in values.items()};full=torch.cat(exact);recon=sum(values.values(),torch.zeros_like(full));linear=values["left_change"]+values["right_change"];den=norm(full) or 1e-30
 return {"complete_norm":norm(full),"factor_norm_fraction":{k:norm(v)/den for k,v in values.items()},"linear_norm_fraction":norm(linear)/den,"closure_rse":float((recon-full).square().sum()/full.square().sum().clamp_min(1e-30)),"profile":torch.as_tensor([norm(values[k]) for k in MLP_FACTORS],device=backend.device)}

def main():
 if {"prior":sha(PRIOR),"literal":sha(LITERAL),"attention_library":sha(ATTNLIB)}!=EXPECTED:raise RuntimeError("factor-atlas authority changed")
 dry={"candidate_id":"temporal_auxiliary.five_mlp_projector_coefficient_factor_atlas_v1","dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"sites":comp.SITES,"projectors":("even_fit","odd_fit"),"model_forwards_max":10,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 if json.loads(LITERAL.read_text())["terminal"]!="literal_weight_executable_response_program":raise RuntimeError("literal program not promoted")
 started,tic=now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
 old_tcap,old_icap=json.loads(comp.TCAP.read_text()),json.loads(comp.ICAP.read_text());tr,ir,*_=greedy.rows_and_controls(old_tcap,old_icap);te,to=tr[::2]+ir[::2],tr[1::2]+ir[1::2];controls=matched.control_rows()
 ceb,_,ce0,_,ce1=v1.cap(backend,controls["discovery"]);cob,_,co0,_,co1=v1.cap(backend,controls["validation"]);teb,_,te0,_,te1=v1.cap(backend,te);tob,_,to0,_,to1=v1.cap(backend,to);cbe,_=comp.fit_bases(backend,ceb,ce0,ce1);cbo,_=comp.fit_bases(backend,cob,co0,co1);qe,_=v1.fit_target(backend,teb,te0,te1,cbe);qo,_=v1.fit_target(backend,tob,to0,to1,cbo);bases={"even_fit":qe,"odd_fit":qo}
 stcap,sicap=json.loads(klfit.STC.read_text()),json.loads(klfit.SIC.read_text());trows=sum((population.capable_rows(klfit.sealed_t,stcap,p,12) for p in ("A1","A2")),[]);irows=sum((population.capable_rows(klfit.sealed_i,sicap,p,12) for p in ("A1","A2")),[]);rows=trows+irows;cues=cue_positions(rows)
 bb=__import__('circuit_das_subspace')._batch(backend,rows,side="base");db=__import__('circuit_das_subspace')._batch(backend,rows,side="donor");_bo,ba,bi,berr=capture_all(backend,bb);_do,da,di,derr=capture_all(backend,db)
 task_ids={task:[i for i,r in enumerate(rows) if klfit.task_name(r)==task] for task in ("temporal","iswas")};records={};closures=[];profiles={}
 for label,qs in bases.items():
  records[label]={};profiles[label]={}
  for site,q in qs.items():
   kind=atlasrun.site_parts(site)[0];records[label][site]={};profiles[label][site]={}
   for task,ids in task_ids.items():
    cell=attention_cell(backend,rows,ids,site,q,ba,da,cues) if kind=="attn" else mlp_cell(backend,rows,ids,site,q,bi,di);profiles[label][site][task]=cell.pop("profile");records[label][site][task]=cell;closures.append(cell["closure_rse"])
 stability={site:{task:cosine(profiles["even_fit"][site][task],profiles["odd_fit"][site][task]) for task in task_ids} for site in comp.SITES}
 vals=[v for label in records.values() for site in label.values() for cell in site.values() for x in cell.values() for v in (x.values() if isinstance(x,dict) else [x])]
 pa=max(berr,derr)<=5e-4 and max(closures)<=1e-10 and all(math.isfinite(float(x)) for x in vals)
 attncells=[cell for label in records.values() for site,cells in label.items() if atlasrun.site_parts(site)[0]=="attn" for cell in cells.values()];mlpcells=[cell for label in records.values() for site,cells in label.items() if atlasrun.site_parts(site)[0]=="mlp" for cell in cells.values()]
 pb=all(max(c["factor_norm_fraction"],key=c["factor_norm_fraction"].get)=="base_pattern_on_value_change" and c["factor_norm_fraction"]["base_pattern_on_value_change"]>=.5 for c in attncells);pc=all(c["causal_suffix_norm_fraction"]>=.8 for c in attncells);pd=all(c["linear_norm_fraction"]>=.8 and c["factor_norm_fraction"]["bilinear_interaction"]<=.25 for c in mlpcells);pe=all(v>=.8 for s in stability.values() for v in s.values())
 predictions={"pred_a_authority_hash_factor_closure_finiteness_and_price":bool(pa),"pred_b_attention_value_transport_dominates":bool(pb),"pred_c_attention_causal_suffix_dominates":bool(pc),"pred_d_mlp_linear_changes_dominate_interaction":bool(pd),"pred_e_factor_profiles_are_crossfit_stable":bool(pe)};terminal="invalid" if not pa else "stable_operation_level_coefficient_program" if all(predictions.values()) else "mixed_coefficient_mechanisms"
 summary={"max_factor_closure_rse":max(closures),"attention_value_dominant_cells":sum(max(c["factor_norm_fraction"],key=c["factor_norm_fraction"].get)=="base_pattern_on_value_change" for c in attncells),"attention_cells":len(attncells),"attention_suffix_cells_ge_08":sum(c["causal_suffix_norm_fraction"]>=.8 for c in attncells),"mlp_linear_cells_pass":sum(c["linear_norm_fraction"]>=.8 and c["factor_norm_fraction"]["bilinear_interaction"]<=.25 for c in mlpcells),"mlp_cells":len(mlpcells),"min_crossfit_profile_cosine":min(v for s in stability.values() for v in s.values())}
 result={"schema":"temporal_five_mlp_projector_coefficient_factor_atlas_result_v1","started_utc":started,"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"instrument":{"attention_reconstruction_max_abs":max(berr,derr)},"records":records,"crossfit_profile_cosine":stability,"summary":summary,"predictions":predictions,"terminal":terminal,"price":{"model_forwards":10,"fit_updates":0,"model_updates":0,"transformer_backwards":0}}
 atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("instrument","summary","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
