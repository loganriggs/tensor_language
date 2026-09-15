#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 2forwards576seq; bracket suffix live-amplitude law selection;11fits0backwards0updates.
"""Select a compact suffix effect law by fifth-to-sixth bidirectional transfer."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,math,signal,sys,time
from pathlib import Path
import numpy as np

RUNNER=Path(__file__).resolve();OPS=RUNNER.parent;ROOT=RUNNER.parents[3];POLY=ROOT/"basis_aligned/polynomial_causal"
FIFTH_ROWS=POLY/"BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json";SIXTH_ROWS=POLY/"BRACKET_EMBEDDED_PENDING_OOD_V1_ROWS.json"
FIFTH_RESULT=POLY/"BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_RESULT.json";SIXTH_RESULT=POLY/"BRACKET_EMBEDDED_PENDING_COMPACT_BEHAVIOR_V1_RESULT.json"
PROGRAM=FIFTH_RESULT;PREREG=POLY/"BRACKET_SUFFIX_LIVE_AMPLITUDE_SELECTION_V1_PREREGISTRATION.md";BINDING=POLY/"BRACKET_SUFFIX_LIVE_AMPLITUDE_SELECTION_V1_BINDING.json";OUT=POLY/"BRACKET_SUFFIX_LIVE_AMPLITUDE_SELECTION_V1_RESULT.json"
TYPES=("parenthesis","square","quote");HEAD=8;D=128;PAIRS=("1->60","1->8","60->1","60->8","8->1","8->60")
CANDIDATES=("global_direct","global_norm","global_direct_norm","pair_direct","pair_norm")
PRICE={"forwards":2,"sequences":576,"fits":11,"backwards":0,"updates":0}
BARS={"replay_max":1e-5,"cosine_min":.90,"relative_l2_max":.40,"sign_min":.90,"norm_ratio_min":.60,"norm_ratio_max":1.40}
PREDICTION_REGISTRY={"pred_a_exact_instrument":None,"pred_b_candidate_exists":None,"pred_c_selected_law_cross_transfers":None,"pred_d_finite_nondegenerate_program":None,"pred_e_selection_boundary":None}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def metrics(actual,predicted):
 actual=np.asarray(actual,np.float64);predicted=np.asarray(predicted,np.float64);an=np.linalg.norm(actual);pn=np.linalg.norm(predicted)
 return {"count":len(actual),"cosine":float(actual@predicted/max(an*pn,1e-30)),"relative_l2_error":float(np.linalg.norm(actual-predicted)/max(an,1e-30)),"sign_agreement":float(np.mean((actual>0)==(predicted>0))),"predicted_to_actual_norm_ratio":float(pn/max(an,1e-30))}
def passes(m):return m["cosine"]>=BARS["cosine_min"] and m["relative_l2_error"]<=BARS["relative_l2_max"] and m["sign_agreement"]>=BARS["sign_min"] and BARS["norm_ratio_min"]<=m["predicted_to_actual_norm_ratio"]<=BARS["norm_ratio_max"]
def design(name,records):
 if name=="global_direct":return np.asarray([[r["direct"]] for r in records],np.float64),["direct"]
 if name=="global_norm":return np.asarray([[r["norm"]] for r in records],np.float64),["norm"]
 if name=="global_direct_norm":return np.asarray([[r["direct"],r["norm"]] for r in records],np.float64),["direct","norm"]
 feature="direct" if name=="pair_direct" else "norm";columns=[f"{feature}:{pair}" for pair in PAIRS];X=np.zeros((len(records),6),np.float64)
 for i,r in enumerate(records):X[i,PAIRS.index(r["ordered_pair"])]=r[feature]
 return X,columns
def fit_predict(name,train,test):
 X,columns=design(name,train);Y=np.asarray([r["effect"] for r in train],np.float64);Xt,_=design(name,test);coef=np.linalg.lstsq(X,Y,rcond=None)[0];pred=Xt@coef
 return {"columns":columns,"coefficients":coef.tolist(),"metrics":metrics([r["effect"] for r in test],pred),"train_column_norms":np.linalg.norm(X,axis=0).tolist(),"test_column_norms":np.linalg.norm(Xt,axis=0).tolist()}
def main():
 sys.path.insert(0,str(OPS));from circuit_exactness_preflight import managed_execution_mode,validate_literal_prediction_registry,validate_result_contract
 binding=json.loads(BINDING.read_text());paths={"fifth_rows":FIFTH_ROWS,"sixth_rows":SIXTH_ROWS,"fifth_result":FIFTH_RESULT,"sixth_result":SIXTH_RESULT,"program":PROGRAM,"preregistration":PREREG};assert all(digest(paths[k])==v for k,v in binding["files"].items()) and binding["bars"]==BARS and binding["price"]==PRICE
 rowsets={"fifth":json.loads(FIFTH_ROWS.read_text()),"sixth":json.loads(SIXTH_ROWS.read_text())};results={"fifth":json.loads(FIFTH_RESULT.read_text()),"sixth":json.loads(SIXTH_RESULT.read_text())};program_result=results["fifth"]
 assert all(canonical(v["rows"])==v["row_manifest_sha256"] for v in rowsets.values());validate_literal_prediction_registry(RUNNER.read_text(),PREDICTION_REGISTRY)
 if managed_execution_mode(__import__("os").environ)=="preflight":print(json.dumps({"dryrun":True,"model_loaded":False,"gpu_accessed":False,"panels":{k:v["endpoint_count"] for k,v in rowsets.items()},"candidates":CANDIDATES,"bars":BARS,"price":PRICE,"predicates":list(PREDICTION_REGISTRY)},sort_keys=True));return
 assert not OUT.exists();signal.alarm(600);import torch;import run_bracket_l13h8_source_region_payload_factorial as exact;from circuit_fast_screen_managed_runner import atomic_create_json
 torch.set_num_threads(2);tm,F,facade=exact._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 tagged=[(panel,row) for panel in ("fifth","sixth") for row in rowsets[panel]["rows"]];endpoints=[(panel,row,side) for panel,row in tagged for side in ("base","donor")];length=max(len(r[f"{s}_ids"]) for _,r,s in endpoints);tokens=torch.full((len(endpoints),length),50256,dtype=torch.long,device="cuda");finals=[];sources=[]
 for i,(_panel,row,side) in enumerate(endpoints):ids=row[f"{side}_ids"];tokens[i,:len(ids)]=torch.tensor(ids,device="cuda");finals.append(len(ids)-1);sources.append(row[f"{side}_open_position"])
 finals_t=torch.tensor(finals,device="cuda");sources_t=torch.tensor(sources,device="cuda");ar=torch.arange(len(endpoints),device="cuda");counts=[0,0]
 def count(_m,args,_o):counts[0]+=1;counts[1]+=len(args[0])
 handle=model.transformer.h[0].attn.register_forward_hook(count);tic=time.perf_counter()
 def extended():
  captured={}
  def attention(event):
   if event.site!=13:return event.block.attn(event.state,event.first_value)
   state,first,attn=event.state,event.first_value,event.block.attn;batch,seq,width=state.shape;heads=9;q=exact._linear(state,attn.c_q.weight,F).view(batch,seq,heads,D);k=exact._linear(state,attn.c_k.weight,F).view(batch,seq,heads,D);q2=exact._linear(state,attn.c_q2.weight,F).view(batch,seq,heads,D);k2=exact._linear(state,attn.c_k2.weight,F).view(batch,seq,heads,D);raw=exact._linear(state,attn.c_v.weight,F).view(batch,seq,heads,D);value=(1-attn.lamb)*raw+attn.lamb*first.view_as(raw);cos,sin=attn.rotary(q);rotate=sys.modules[type(attn).__module__].apply_rotary_emb;qn,kn=F.rms_norm(q,(D,)),F.rms_norm(k,(D,));q2n,k2n=F.rms_norm(q2,(D,)),F.rms_norm(k2,(D,));qr,kr=rotate(qn,cos,sin),rotate(kn,cos,sin);q2r,k2r=rotate(q2n,cos,sin),rotate(k2n,cos,sin);s1=torch.einsum("bqhd,bkhd->bhqk",qr,kr)/D;s2=torch.einsum("bqhd,bkhd->bhqk",q2r,k2r)/D;pattern=(s1*s2).masked_fill(~torch.tril(torch.ones(seq,seq,dtype=torch.bool,device=state.device)),0);all_heads=torch.einsum("bhqk,bkhd->bhqd",pattern,value);write=exact._linear(all_heads.transpose(1,2).contiguous().view(batch,seq,width),attn.c_proj.weight,F);weight=attn.c_proj.weight[:,HEAD*D:(HEAD+1)*D];u=exact._linear(value[:,:,HEAD].float(),weight.float(),F);p=pattern[ar,HEAD,finals_t];head=torch.einsum("bk,bkd->bd",p.float(),u);captured.update({"p":p.float().detach(),"u":u.detach(),"head":head.detach(),"q1":qr[ar,finals_t,HEAD].detach(),"q2":q2r[ar,finals_t,HEAD].detach(),"cos":cos.detach(),"sin":sin.detach(),"rotate":rotate});return write,first
  logits=facade.forward_with_dispatch(model,tokens,attention,lambda e:e.block.mlp(e.state),require_production=False).float();return logits,captured
 try:
  with torch.inference_mode():
   extended_logits,factors=extended();independent,_=exact.factor_forward(model,tokens,finals_t,{},tm,F,facade);replay_error=max(float((extended_logits[i,finals[i]]-independent[i,finals[i]]).abs().max()) for i in range(len(endpoints)));del extended_logits,independent;torch.cuda.empty_cache();p=factors["p"][ar,sources_t];u=factors["u"][ar,sources_t]
   tables={name:{key:torch.tensor(value,dtype=torch.float64,device="cuda") if isinstance(value,list) else value for key,value in tab.items()} for name,tab in program_result["program"].items()};du=torch.zeros_like(u);k1=torch.zeros((len(endpoints),D),device="cuda");k2=torch.zeros_like(k1)
   for i,(_panel,row,side) in enumerate(endpoints):
    if row["program_role"]=="target":other="donor" if side=="base" else "base";rt=row[f"{side}_type"];dt=row[f"{other}_type"]
    else:rt=dt=row["inner_type"]
    ri,di=TYPES.index(rt),TYPES.index(dt);tab=tables["payload"];du[i]=((tab["type_coefficients"][di]-tab["type_coefficients"][ri])@tab["basis"]).to(u.dtype)
    for name,target in (("key1",k1),("key2",k2)):tab=tables[name];target[i]=(tab["mean"]+tab["type_coefficients"][di]@tab["basis"]).to(target.dtype)
   cos=factors["cos"][0,sources_t,0][:,None,None,:];sin=factors["sin"][0,sources_t,0][:,None,None,:];rotate=factors["rotate"];k1r=rotate(k1[:,None,None,:],cos,sin)[:,0,0];k2r=rotate(k2[:,None,None,:],cos,sin)[:,0,0];phat=((factors["q1"]*k1r).sum(-1)/D)*((factors["q2"]*k2r).sum(-1)/D);delta=phat[:,None]*(u+du)-p[:,None]*u
 finally:handle.remove()
 targets_by_panel={};effect_maps={"fifth":{(r["row_id"],r["side"]):r["both_effect"] for r in results["fifth"]["records"] if r["program_role"]=="target"},"sixth":{(r["row_id"],r["side"]):r["program_effect"] for r in results["sixth"]["records"] if r["program_role"]=="target"}}
 readout=model.lm_head.weight.detach().float();records=[]
 for i,(panel,row,side) in enumerate(endpoints):
  if row["program_role"]!="target":continue
  other="donor" if side=="base" else "base";recipient=int(row[f"{side}_answer_id"]);donor_answer=int(row[f"{other}_answer_id"]);pair=f"{recipient}->{donor_answer}";dv=delta[i].float();records.append({"panel":panel,"row_id":row["row_id"],"side":side,"ordered_pair":pair,"direct":float(torch.dot(dv,readout[donor_answer]-readout[recipient]).cpu()),"norm":float(dv.norm().cpu()),"effect":float(effect_maps[panel][(row["row_id"],side)])})
 assert len(records)==144 and all(len(effect_maps[p])==72 for p in effect_maps);panel_records={p:[r for r in records if r["panel"]==p] for p in ("fifth","sixth")};reports={};passing=[]
 for name in CANDIDATES:
  forward=fit_predict(name,panel_records["fifth"],panel_records["sixth"]);reverse=fit_predict(name,panel_records["sixth"],panel_records["fifth"]);ok=passes(forward["metrics"]) and passes(reverse["metrics"]);reports[name]={"parameter_count":len(forward["columns"]),"fifth_to_sixth":forward,"sixth_to_fifth":reverse,"passes":bool(ok),"worst_relative_l2":max(forward["metrics"]["relative_l2_error"],reverse["metrics"]["relative_l2_error"])}
  if ok:passing.append(name)
 selected=min(passing,key=lambda n:(reports[n]["parameter_count"],reports[n]["worst_relative_l2"],CANDIDATES.index(n))) if passing else None;program=None
 if selected:
  X,columns=design(selected,records);y=np.asarray([r["effect"] for r in records]);coef=np.linalg.lstsq(X,y,rcond=None)[0];program={"law":selected,"features":columns,"coefficients":coef.tolist(),"parameter_count":len(coef),"fit_panels":["fifth","sixth"],"through_origin":True,"construction_feature":False,"future_rows_accessed":False,"quantized":False,"column_norms":np.linalg.norm(X,axis=0).tolist()}
 finite=program is not None and all(math.isfinite(v) for v in program["coefficients"]+program["column_norms"]) and min(program["column_norms"])>0;boundary=program is not None and not program["construction_feature"] and not program["future_rows_accessed"] and not program["quantized"] and set(x.split(":")[0] for x in program["features"])<= {"direct","norm"};instrument=counts==[2,576] and replay_error<=BARS["replay_max"] and len(records)==144
 predictions={"pred_a_exact_instrument":bool(instrument),"pred_b_candidate_exists":bool(instrument and passing),"pred_c_selected_law_cross_transfers":bool(instrument and selected and reports[selected]["passes"]),"pred_d_finite_nondegenerate_program":bool(instrument and finite),"pred_e_selection_boundary":bool(instrument and boundary)};terminal="invalid" if not instrument else ("live_amplitude_law_selected" if all(predictions.values()) else "live_amplitude_candidates_null")
 result={"schema":"bracket_suffix_live_amplitude_selection_v1_result","terminal":terminal,"predictions":predictions,"instrument":{"extended_independent_replay_max_logit_error":replay_error,"target_count":len(records)},"candidate_reports":reports,"selected_program":program,"price":{**PRICE,"observed_forwards":counts[0],"observed_sequences":counts[1]},"claim_boundary":"Selection on already opened fifth/sixth donor-free program effects only; no seventh construction, new causal effect, construction feature, intercept, rank change, update, or quantization.","fifth_rows_sha256":digest(FIFTH_ROWS),"sixth_rows_sha256":digest(SIXTH_ROWS),"fifth_result_sha256":digest(FIFTH_RESULT),"sixth_result_sha256":digest(SIXTH_RESULT),"binding_sha256":digest(BINDING),"runner_sha256":digest(RUNNER),"checkpoint_sha256":checkpoint.weights_sha256,"wall_seconds":time.perf_counter()-tic,"feature_records":records};validate_result_contract(result,PREDICTION_REGISTRY);atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","candidate_reports","selected_program","price")},indent=2));assert terminal!="invalid"
if __name__=="__main__":main()
