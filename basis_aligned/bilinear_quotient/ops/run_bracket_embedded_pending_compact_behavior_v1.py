#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 4forwards576seq; donor-free bracket source plus frozen six-scalar suffix law;0fits0backwards0updates.
"""Test a complete compact bracket behavior prediction on a sixth construction."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,math,signal,sys,time
from pathlib import Path

RUNNER=Path(__file__).resolve(); OPS=RUNNER.parent; ROOT=RUNNER.parents[3]; POLY=ROOT/"basis_aligned/polynomial_causal"
ROWS=POLY/"BRACKET_EMBEDDED_PENDING_OOD_V1_ROWS.json"; BUILDER=POLY/"build_bracket_embedded_pending_ood_v1_rows.py"
PROGRAM=POLY/"BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_RESULT.json"
SCALAR=ROOT/"basis_aligned/bilinear_quotient/circuits/followups/bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json"
PREREG=POLY/"BRACKET_EMBEDDED_PENDING_COMPACT_BEHAVIOR_V1_PREREGISTRATION.md"; BINDING=POLY/"BRACKET_EMBEDDED_PENDING_COMPACT_BEHAVIOR_V1_BINDING.json"; OUT=POLY/"BRACKET_EMBEDDED_PENDING_COMPACT_BEHAVIOR_V1_RESULT.json"
TYPES=("parenthesis","square","quote"); HEAD=8; D=128
PRICE={"forwards":4,"sequences":576,"fits":0,"backwards":0,"updates":0}
BARS={"replay_max":1e-5,"capability_accuracy_min":.75,"exact_positive_min":.90,"program_cosine_min":.90,"program_relative_l2_max":.40,"program_sign_min":.90,"program_norm_min":.60,"program_norm_max":1.40,"program_pair_cosine_min":.75,"program_pair_relative_l2_max":.50,"program_pair_sign_min":.85,"scalar_cosine_min":.80,"scalar_relative_l2_max":.60,"scalar_sign_min":.90,"scalar_norm_min":.50,"scalar_norm_max":1.50,"scalar_pair_cosine_min":.90,"scalar_pair_relative_l2_max":.65,"scalar_pair_sign_min":.90,"control_to_target_rms_max":.50}
PREDICTION_REGISTRY={"pred_a_exact_instrument_and_capability":None,"pred_b_exact_joint_parent_live":None,"pred_c_donor_free_source_transfers":None,"pred_d_six_scalar_predicts_exact_effect":None,"pred_e_six_scalar_predicts_program_effect":None,"pred_f_control_selectivity":None}
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def metrics(actual,predicted):
 actual=[float(x) for x in actual]; predicted=[float(x) for x in predicted]; dot=sum(a*b for a,b in zip(actual,predicted)); an=math.sqrt(sum(a*a for a in actual)); pn=math.sqrt(sum(p*p for p in predicted))
 return {"count":len(actual),"cosine":dot/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((a-p)**2 for a,p in zip(actual,predicted)))/max(an,1e-30),"sign_agreement":sum((a>0)==(p>0) for a,p in zip(actual,predicted))/len(actual),"predicted_to_actual_norm_ratio":pn/max(an,1e-30)}
def main():
 sys.path.insert(0,str(OPS)); from circuit_exactness_preflight import managed_execution_mode,validate_literal_prediction_registry,validate_result_contract
 binding=json.loads(BINDING.read_text()); paths={"rows":ROWS,"builder":BUILDER,"program":PROGRAM,"scalar":SCALAR,"preregistration":PREREG}; assert all(digest(paths[k])==v for k,v in binding["files"].items())
 frozen=json.loads(ROWS.read_text()); program_result=json.loads(PROGRAM.read_text()); scalar_result=json.loads(SCALAR.read_text()); assert canonical(frozen["rows"])==frozen["row_manifest_sha256"] and program_result["terminal"]=="key_rank2_payload_rank2_interaction_transfer" and binding["bars"]==BARS and binding["price"]==PRICE
 tabs=list(scalar_result["score"]["cross_family_scalar_tables"].values()); scalars={pair:sum(tab[pair] for tab in tabs)/len(tabs) for pair in tabs[0]}; assert scalars==binding["ordered_pair_scalars"]
 validate_literal_prediction_registry(RUNNER.read_text(),PREDICTION_REGISTRY)
 if managed_execution_mode(__import__("os").environ)=="preflight": print(json.dumps({"dryrun":True,"model_loaded":False,"gpu_accessed":False,"rows":72,"endpoints":144,"program_ranks":{k:v["rank"] for k,v in program_result["program"].items()},"ordered_pair_scalars":scalars,"bars":BARS,"price":PRICE,"predicates":list(PREDICTION_REGISTRY)},sort_keys=True)); return
 assert not OUT.exists(); signal.alarm(600); import torch; import run_bracket_l13h8_source_region_payload_factorial as exact; from circuit_fast_screen_managed_runner import atomic_create_json
 torch.set_num_threads(2); tm,F,facade=exact._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 rows=frozen["rows"]; endpoints=[(row,side) for row in rows for side in ("base","donor")]; length=max(len(r[f"{s}_ids"]) for r,s in endpoints); tokens=torch.full((len(endpoints),length),50256,dtype=torch.long,device="cuda"); finals=[]; sources=[]
 for i,(row,side) in enumerate(endpoints): ids=row[f"{side}_ids"]; tokens[i,:len(ids)]=torch.tensor(ids,device="cuda"); finals.append(len(ids)-1); sources.append(row[f"{side}_open_position"])
 finals_t=torch.tensor(finals,device="cuda"); sources_t=torch.tensor(sources,device="cuda"); ar=torch.arange(len(endpoints),device="cuda"); counts=[0,0]
 def count(_m,args,_o): counts[0]+=1; counts[1]+=len(args[0])
 handle=model.transformer.h[0].attn.register_forward_hook(count); tic=time.perf_counter()
 def extended():
  captured={}
  def attention(event):
   if event.site!=13:return event.block.attn(event.state,event.first_value)
   state,first,attn=event.state,event.first_value,event.block.attn; batch,seq,width=state.shape; heads=9
   q=exact._linear(state,attn.c_q.weight,F).view(batch,seq,heads,D); k=exact._linear(state,attn.c_k.weight,F).view(batch,seq,heads,D); q2=exact._linear(state,attn.c_q2.weight,F).view(batch,seq,heads,D); k2=exact._linear(state,attn.c_k2.weight,F).view(batch,seq,heads,D); raw=exact._linear(state,attn.c_v.weight,F).view(batch,seq,heads,D); value=(1-attn.lamb)*raw+attn.lamb*first.view_as(raw); cos,sin=attn.rotary(q); rotate=sys.modules[type(attn).__module__].apply_rotary_emb
   qn,kn=F.rms_norm(q,(D,)),F.rms_norm(k,(D,)); q2n,k2n=F.rms_norm(q2,(D,)),F.rms_norm(k2,(D,)); qr,kr=rotate(qn,cos,sin),rotate(kn,cos,sin); q2r,k2r=rotate(q2n,cos,sin),rotate(k2n,cos,sin)
   s1=torch.einsum("bqhd,bkhd->bhqk",qr,kr)/D; s2=torch.einsum("bqhd,bkhd->bhqk",q2r,k2r)/D; pattern=(s1*s2).masked_fill(~torch.tril(torch.ones(seq,seq,dtype=torch.bool,device=state.device)),0); all_heads=torch.einsum("bhqk,bkhd->bhqd",pattern,value); write=exact._linear(all_heads.transpose(1,2).contiguous().view(batch,seq,width),attn.c_proj.weight,F); weight=attn.c_proj.weight[:,HEAD*D:(HEAD+1)*D]; u=exact._linear(value[:,:,HEAD].float(),weight.float(),F); p=pattern[ar,HEAD,finals_t]; head=torch.einsum("bk,bkd->bd",p.float(),u)
   captured.update({"p":p.float().detach(),"u":u.detach(),"head":head.detach(),"q1":qr[ar,finals_t,HEAD].detach(),"q2":q2r[ar,finals_t,HEAD].detach(),"cos":cos.detach(),"sin":sin.detach(),"rotate":rotate}); return write,first
  logits=facade.forward_with_dispatch(model,tokens,attention,lambda e:e.block.mlp(e.state),require_production=False).float(); return logits,captured
 try:
  with torch.inference_mode():
   native=exact.native_logits(model,tokens,tm,F).cpu(); replay_gpu,factors=extended(); replay=replay_gpu.cpu(); del replay_gpu; torch.cuda.empty_cache(); p=factors["p"][ar,sources_t]; u=factors["u"][ar,sources_t]; donor=ar^1; pd,ud=p[donor],u[donor]
   tables={name:{key:torch.tensor(value,dtype=torch.float64,device="cuda") if isinstance(value,list) else value for key,value in table.items()} for name,table in program_result["program"].items()}
   du=torch.zeros_like(u); k1=torch.zeros((len(endpoints),D),device="cuda"); k2=torch.zeros_like(k1)
   for i,(row,side) in enumerate(endpoints):
    if row["program_role"]=="target": other="donor" if side=="base" else "base"; rt=row[f"{side}_type"]; dt=row[f"{other}_type"]
    else: rt=dt=row["inner_type"]
    ri,di=TYPES.index(rt),TYPES.index(dt); tab=tables["payload"]; du[i]=((tab["type_coefficients"][di]-tab["type_coefficients"][ri])@tab["basis"]).to(u.dtype)
    for name,target in (("key1",k1),("key2",k2)): tab=tables[name]; target[i]=(tab["mean"]+tab["type_coefficients"][di]@tab["basis"]).to(target.dtype)
   cos=factors["cos"][0,sources_t,0][:,None,None,:]; sin=factors["sin"][0,sources_t,0][:,None,None,:]; rotate=factors["rotate"]; k1r=rotate(k1[:,None,None,:],cos,sin)[:,0,0]; k2r=rotate(k2[:,None,None,:],cos,sin)[:,0,0]; phat=((factors["q1"]*k1r).sum(-1)/D)*((factors["q2"]*k2r).sum(-1)/D)
   replacements={"exact":pd[:,None]*ud,"program":phat[:,None]*(u+du)}; arms={}
   for name,replacement in replacements.items(): logits,unused=exact.factor_forward(model,tokens,finals_t,{},tm,F,facade,replacement_terms=replacement,source_positions=sources_t); arms[name]=logits.cpu(); del logits,unused; torch.cuda.empty_cache()
 finally: handle.remove()
 replay_error=max(float((native[i,finals[i]]-replay[i,finals[i]]).abs().max()) for i in range(len(endpoints))); records=[]; caps=defaultdict(list)
 for i,(row,side) in enumerate(endpoints):
  answer=int(row[f"{side}_answer_id"]); other="donor" if side=="base" else "base"; other_answer=int(row[f"{other}_answer_id"]); pair=f"{answer}->{other_answer}"; key=(row["program_role"],answer,side) if row["program_role"]=="control" else ("target",answer,other_answer); caps[key].append(float(exact.closer_margin(native[i,finals[i]],answer))); rec={"row_id":row["row_id"],"side":side,"program_role":row["program_role"],"ordered_pair":pair}
  if row["program_role"]=="target": direction="base_to_donor" if side=="base" else "donor_to_base"; rec["exact_effect"]=float(exact.endpoint_change(replay[i,finals[i]],arms["exact"][i,finals[i]],row,direction)); rec["program_effect"]=float(exact.endpoint_change(replay[i,finals[i]],arms["program"][i,finals[i]],row,direction)); rec["scalar_prediction"]=float(scalars[pair])
  else: before=float(exact.closer_margin(replay[i,finals[i]],answer)); rec["program_control_change"]=float(exact.closer_margin(arms["program"][i,finals[i]],answer)-before)
  records.append(rec)
 cap={"|".join(map(str,k)):{"n":len(v),"accuracy":sum(x>0 for x in v)/len(v),"mean_closer_margin":sum(v)/len(v)} for k,v in sorted(caps.items(),key=lambda z:str(z[0]))}; targets=[r for r in records if r["program_role"]=="target"]; controls=[r for r in records if r["program_role"]=="control"]
 overall={"program_vs_exact":metrics([r["exact_effect"] for r in targets],[r["program_effect"] for r in targets]),"scalar_vs_exact":metrics([r["exact_effect"] for r in targets],[r["scalar_prediction"] for r in targets]),"scalar_vs_program":metrics([r["program_effect"] for r in targets],[r["scalar_prediction"] for r in targets])}; by_pair={}
 for pair in sorted(scalars):
  items=[r for r in targets if r["ordered_pair"]==pair]; by_pair[pair]={"n":len(items),"exact_positive_fraction":sum(r["exact_effect"]>0 for r in items)/len(items),"program_vs_exact":metrics([r["exact_effect"] for r in items],[r["program_effect"] for r in items]),"scalar_vs_exact":metrics([r["exact_effect"] for r in items],[r["scalar_prediction"] for r in items]),"scalar_vs_program":metrics([r["program_effect"] for r in items],[r["scalar_prediction"] for r in items])}
 ranks={k:v["rank"] for k,v in program_result["program"].items()}; target_rms=math.sqrt(sum(r["exact_effect"]**2 for r in targets)/len(targets)); control_rms=math.sqrt(sum(r["program_control_change"]**2 for r in controls)/len(controls)); capable=all(v["accuracy"]>=BARS["capability_accuracy_min"] and v["mean_closer_margin"]>0 for v in cap.values()); instrument=counts==[4,576] and replay_error<=BARS["replay_max"] and set(ranks.values())=={2} and len(targets)==len(controls)==72 and capable
 def program_pass(v,pair=False): return v["cosine"]>=BARS["program_pair_cosine_min" if pair else "program_cosine_min"] and v["relative_l2_error"]<=BARS["program_pair_relative_l2_max" if pair else "program_relative_l2_max"] and v["sign_agreement"]>=BARS["program_pair_sign_min" if pair else "program_sign_min"] and (pair or BARS["program_norm_min"]<=v["predicted_to_actual_norm_ratio"]<=BARS["program_norm_max"])
 def scalar_pass(v,pair=False): return v["cosine"]>=BARS["scalar_pair_cosine_min" if pair else "scalar_cosine_min"] and v["relative_l2_error"]<=BARS["scalar_pair_relative_l2_max" if pair else "scalar_relative_l2_max"] and v["sign_agreement"]>=BARS["scalar_pair_sign_min" if pair else "scalar_sign_min"] and (pair or BARS["scalar_norm_min"]<=v["predicted_to_actual_norm_ratio"]<=BARS["scalar_norm_max"])
 live=all(v["exact_positive_fraction"]>=BARS["exact_positive_min"] for v in by_pair.values()); prog=program_pass(overall["program_vs_exact"]) and all(program_pass(v["program_vs_exact"],True) for v in by_pair.values()); se=scalar_pass(overall["scalar_vs_exact"]) and all(scalar_pass(v["scalar_vs_exact"],True) for v in by_pair.values()); sp=scalar_pass(overall["scalar_vs_program"]) and all(scalar_pass(v["scalar_vs_program"],True) for v in by_pair.values()); selective=control_rms/max(target_rms,1e-30)<=BARS["control_to_target_rms_max"]
 predictions={"pred_a_exact_instrument_and_capability":bool(instrument),"pred_b_exact_joint_parent_live":bool(instrument and live),"pred_c_donor_free_source_transfers":bool(instrument and prog),"pred_d_six_scalar_predicts_exact_effect":bool(instrument and se),"pred_e_six_scalar_predicts_program_effect":bool(instrument and sp),"pred_f_control_selectivity":bool(instrument and selective)}
 if not (counts==[4,576] and replay_error<=BARS["replay_max"] and set(ranks.values())=={2}): terminal="invalid"
 elif not capable: terminal="sixth_construction_capability_null"
 elif all(predictions.values()): terminal="compact_source_and_suffix_behavior_transfer"
 elif predictions["pred_a_exact_instrument_and_capability"] and predictions["pred_b_exact_joint_parent_live"] and predictions["pred_c_donor_free_source_transfers"] and predictions["pred_f_control_selectivity"] and not (predictions["pred_d_six_scalar_predicts_exact_effect"] and predictions["pred_e_six_scalar_predicts_program_effect"]): terminal="fixed_suffix_scalar_null"
 else: terminal="compact_behavior_null"
 result={"schema":"bracket_embedded_pending_compact_behavior_v1_result","terminal":terminal,"predictions":predictions,"instrument":{"native_factor_replay_max_logit_error":replay_error,"program_ranks":ranks},"capability_cells":cap,"overall":overall,"by_ordered_pair":by_pair,"ordered_pair_scalars":scalars,"target_exact_effect_rms":target_rms,"program_control_rms":control_rms,"control_to_target_rms":control_rms/max(target_rms,1e-30),"price":{**PRICE,"observed_forwards":counts[0],"observed_sequences":counts[1]},"claim_boundary":"Prospective sixth-construction test of the frozen donor-free source generator and fixed six-scalar suffix-free behavioral law; no fit, rank/gain change, donor state in program arm, or quantization.","rows_sha256":digest(ROWS),"program_sha256":digest(PROGRAM),"scalar_sha256":digest(SCALAR),"binding_sha256":digest(BINDING),"runner_sha256":digest(RUNNER),"checkpoint_sha256":checkpoint.weights_sha256,"wall_seconds":time.perf_counter()-tic,"records":records}; validate_result_contract(result,PREDICTION_REGISTRY); atomic_create_json(OUT,result); print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","overall","control_to_target_rms","price")},indent=2)); assert terminal!="invalid"
if __name__=="__main__": main()
