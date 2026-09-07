#!/usr/bin/env python3
"""Cross-fit projector stability and exact weight pullbacks for response circuits."""
# BQGATE: EXPERIMENT pred_a_authority_split_hash_finiteness_and_price pred_b_crossfit_subspaces_overlap pred_c_attention_weight_contractions_close pred_d_mlp_weight_contractions_close pred_e_crossfit_weight_maps_agree
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/"circuits/prior_art/temporal_five_mlp_crossfit_response_weight_interface_v1.json";EVEN=ROOT/"circuits/followups/temporal_five_mlp_target_contrast_response_basis_v1_result.json";ODD=ROOT/"circuits/followups/temporal_five_mlp_target_contrast_split_swap_v2_result.json";OUT=ROOT/"circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
EXPECTED={"prior":"e5f88079416516428afc54c6b410e798fe111811fe26f7a256aaca5612c6838d","even":"db8d94bda707546878c78d62bde3c9bbb5b5510557f6fd9b3f042dbb05359f17","odd":"f3895b9e07557be81966914d34f2ba8cda86d08dfd53434609d9d037201ff37c"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(x):return hashlib.sha256(x.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def capture(backend,rows):
 inputs={};handles=[]
 for site in comp.SITES:
  kind,layer,_=atlasrun.site_parts(site)
  if kind=="mlp":
   def hook(_m,args,site=site):inputs[site]=args[0].detach().float().cpu().clone()
   handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(hook))
 try: batch,out,cache,donor_out,donor_cache=v1.cap(backend,rows)
 finally:
  for h in handles:h.remove()
 # v1.cap performs two forwards; hooks retain only donor input. Repeat base inputs
 base_inputs={};handles=[]
 for site in comp.SITES:
  kind,layer,_=atlasrun.site_parts(site)
  if kind=="mlp":
   def hook(_m,args,site=site):base_inputs[site]=args[0].detach().float().cpu().clone()
   handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(hook))
 # Avoid an extra forward by reconstructing selected MLP input from captured target io only is impossible.
 for h in handles:h.remove()
 return batch,out,cache,donor_out,donor_cache,base_inputs,inputs
def cap_inputs(backend,rows):
 # One base and one donor forward, each with selected MLP pre-input hooks.
 import circuit_das_subspace as das
 def one(side):
  got={};hs=[]
  for site in comp.SITES:
   kind,layer,_=atlasrun.site_parts(site)
   if kind=="mlp":
    def hook(_m,args,site=site):got[site]=args[0].detach().float().cpu().clone()
    hs.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(hook))
  batch=das._batch(backend,rows,side=side)
  try:out,cache=atlasrun.capture_native(backend,batch)
  finally:
   for h in hs:h.remove()
  return batch,out,cache,got
 bb,bo,bc,bi=one("base");_db,do,dc,di=one("donor");return bb,bo,bc,do,dc,bi,di
def valid_delta(batch,a,b,sl=None):
 parts=[]
 for i,pos in enumerate(batch.semantic_positions):
  x=b[i,:int(pos)+1]-a[i,:int(pos)+1];parts.append(x if sl is None else x[:,sl])
 return parts[0].new_empty((0,parts[0].shape[-1])) if not parts else __import__('torch').cat(parts)
def cosine(a,b):return float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp_min(1e-30))
def main():
 if {"prior":sha(PRIOR),"even":sha(EVEN),"odd":sha(ODD)}!=EXPECTED:raise RuntimeError("authority changed")
 dry={"candidate_id":"temporal_auxiliary.five_mlp_crossfit_response_weight_interface_v1","dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"model_forwards_max":8,"sites":comp.SITES,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 tcap,icap=json.loads(comp.TCAP.read_text()),json.loads(comp.ICAP.read_text());tr,ir,*_=greedy.rows_and_controls(tcap,icap);te,to=tr[::2]+ir[::2],tr[1::2]+ir[1::2];controls=matched.control_rows();ce,co=controls["discovery"],controls["validation"]
 tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");t=backend.torch
 ceb,_,ce0,_,ce1,_,_=cap_inputs(backend,ce);cob,_,co0,_,co1,_,_=cap_inputs(backend,co);teb,_,te0,_,te1,tebi,tedi=cap_inputs(backend,te);tob,_,to0,_,to1,tobi,todi=cap_inputs(backend,to)
 cbe,_=comp.fit_bases(backend,ceb,ce0,ce1);cbo,_=comp.fit_bases(backend,cob,co0,co1);qe,_=v1.fit_target(backend,teb,te0,te1,cbe);qo,_=v1.fit_target(backend,tob,to0,to1,cbo)
 records={};attn_rse=[];mlp_rse=[];map_cos=[];overlap=[];width=backend.model.config.n_embd//backend.model.config.n_head
 for site in comp.SITES:
  kind,layer,head=atlasrun.site_parts(site);a,b=qe[site],qo[site];u,s,vh=t.linalg.svd(b.T@a);rot=u@vh;principal=[float(x) for x in s];mean_sq=float(s.square().mean());overlap.append(mean_sq)
  maps=[];closures=[]
  for q,batch,base,donor,base_in,donor_in in ((a,tob,to0,to1,tobi,todi),(b,teb,te0,te1,tebi,tedi)):
   if kind=="attn":
    sl=slice(head*width,(head+1)*width);delta=valid_delta(batch,base["attention"][layer],donor["attention"][layer],sl).to(backend.device);w=backend.model.transformer.h[layer].attn.c_proj.weight.detach().float()[:,sl];wm=w@q;direct=((delta@q)@q.T)@w.T;compiled=(delta@q)@wm.T
   else:
    module=backend.model.transformer.h[layer].mlp;xb,xd=base_in[site].to(backend.device),donor_in[site].to(backend.device);hb=module.Left(xb).float()*module.Right(xb).float();hd=module.Left(xd).float()*module.Right(xd).float();delta_h=valid_delta(batch,hb,hd).to(backend.device);w=module.Down.weight.detach().float();wm=w.T@q;direct=(delta_h@w.T@q)@q.T;compiled=(delta_h@wm)@q.T
   rse=float((compiled-direct).square().sum()/direct.square().sum().clamp_min(1e-30));closures.append(rse);maps.append(wm)
  aligned=maps[1]@rot;mc=cosine(maps[0],aligned);map_cos.append(mc);(attn_rse if kind=="attn" else mlp_rse).extend(closures);records[site]={"kind":kind,"principal_cosines":principal,"mean_squared_principal_cosine":mean_sq,"fit_basis_sha256":[thash(a),thash(b)],"weight_map_sha256":[thash(maps[0]),thash(maps[1])],"opposite_half_closure_rse":closures,"aligned_weight_map_cosine":mc}
 finite=all(math.isfinite(x) for r in records.values() for x in r["principal_cosines"]+r["opposite_half_closure_rse"]+[r["aligned_weight_map_cosine"]]);pa=finite and len(records)==8;pb=sum(x>=.5 for x in overlap)>=6 and min(overlap)>=.2;pc=max(attn_rse)<=1e-10 and len(attn_rse)==8;pd=max(mlp_rse)<=1e-10 and len(mlp_rse)==8;pe=sum(x>=.8 for x in map_cos)>=6 and min(map_cos)>=.6
 preds={"pred_a_authority_split_hash_finiteness_and_price":bool(pa),"pred_b_crossfit_subspaces_overlap":bool(pb),"pred_c_attention_weight_contractions_close":bool(pc),"pred_d_mlp_weight_contractions_close":bool(pd),"pred_e_crossfit_weight_maps_agree":bool(pe)};terminal="invalid" if not(pa and pc and pd) else "stable_weight_readable_response_program" if all(preds.values()) else "operational_equivalence_only";result={"schema":"temporal_five_mlp_crossfit_response_weight_interface_result_v1","started_utc":now(),"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"records":records,"summary":{"sites_overlap_ge_05":sum(x>=.5 for x in overlap),"min_overlap":min(overlap),"sites_map_cos_ge_08":sum(x>=.8 for x in map_cos),"min_map_cos":min(map_cos),"max_attention_closure_rse":max(attn_rse),"max_mlp_closure_rse":max(mlp_rse)},"predictions":preds,"terminal":terminal,"price":{"model_forwards":8,"fit_updates":0,"model_updates":0,"transformer_backwards":0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("summary","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
