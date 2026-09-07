#!/usr/bin/env python3
"""Held-out target-versus-control response bases inside fixed upstream sites."""
# BQGATE: EXPERIMENT pred_a_authority_split_alignment_finiteness_and_price pred_b_full_response_validation_program_is_sufficient pred_c_target_basis_preserves_heldout_behavior pred_d_target_basis_is_selective pred_e_basis_beats_control_complement
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
ROOT=Path(__file__).resolve().parents[1]; PRIOR=ROOT/"circuits/prior_art/temporal_five_mlp_target_contrast_response_basis_v1.json"; OLD=ROOT/"circuits/followups/temporal_five_mlp_generic_subspace_complement_program_v1_result.json"; CONTROL=ROOT/"circuits/followups/temporal_five_mlp_matched_control_upstream_atlas_v2_result.json"; OUT=ROOT/"circuits/followups/temporal_five_mlp_target_contrast_response_basis_v1_result.json"
EXPECTED={"prior":"636625fc433d684ad091badadadfead088d48d7568e2f56f203cc768e5822d37","old":"6b45e6c58b2d8ca208368ebc04247c06574511b77be1b7fdcdca1d2999031f86","control":"b75a4b81b8d2407ea3661f21be8b4a3bf967bc8a242fc08d562aa6579b97bb69"}; RANKS=(1,4,8)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def fit_target(backend,batch,base,donor,control_basis):
 t=backend.torch; width=backend.model.config.n_embd//backend.model.config.n_head; result={}; spectra={}
 for site in comp.SITES:
  kind,layer,head=atlasrun.site_parts(site); family="mlp" if kind=="mlp" else "attention"; parts=[]; qc=control_basis[site]
  for i,pos in enumerate(batch.semantic_positions):
   d=donor[family][layer][i,:int(pos)+1]-base[family][layer][i,:int(pos)+1]
   if kind=="attn": d=d[:,head*width:(head+1)*width]
   d=d.to(backend.device).float(); parts.append(d-(d@qc)@qc.T)
  m=t.cat(parts); _u,s,vh=t.linalg.svd(m,full_matrices=False); result[site]=vh[:8].T.contiguous(); spectra[site]=[float(x) for x in s[:12]]
 return result,spectra
def run_project(backend,batch,base,donor,bases,rank):
 width=backend.model.config.n_embd//backend.model.config.n_head; handles=[]
 for site in comp.SITES:
  kind,layer,head=atlasrun.site_parts(site); q=bases[site][:,:rank]
  if kind=="attn":
   b,d=base["attention"][layer],donor["attention"][layer]
   def hook(_m,args,b=b,d=d,q=q,head=head):
    x=args[0].clone(); a,z=head*width,(head+1)*width
    for i,pos in enumerate(batch.semantic_positions):
     n=int(pos)+1; delta=(d[i,:n,a:z]-b[i,:n,a:z]).to(x); x[i,:n,a:z]=b[i,:n,a:z].to(x)+(delta@q)@q.T
    return (x,)+tuple(args[1:])
   handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
  else:
   b,d=base["mlp"][layer],donor["mlp"][layer]
   def hook(_m,_a,out,b=b,d=d,q=q):
    x=out.clone()
    for i,pos in enumerate(batch.semantic_positions):
     n=int(pos)+1; delta=(d[i,:n]-b[i,:n]).to(x); x[i,:n]=b[i,:n].to(x)+(delta@q)@q.T
    return x
   handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(hook))
 try:return backend.native(batch,capture=True)
 finally:
  for h in handles:h.remove()
def cap(backend,rows):
 bb,db=das._batch(backend,rows,side="base"),das._batch(backend,rows,side="donor"); bo,bc=atlasrun.capture_native(backend,bb); do,dc=atlasrun.capture_native(backend,db); return bb,bo,bc,do,dc
def report(backend,rows,base_state,donor_state,state,reader,task_cut,scales=None):
 t=backend.torch; full_b=comp.margins(backend,donor_state,rows)-comp.margins(backend,base_state,rows); db=comp.margins(backend,state,rows)-comp.margins(backend,base_state,rows); full_m=(donor_state-base_state)@reader; dm=(state-base_state)@reader; ids={"temporal":slice(0,task_cut),"iswas":slice(task_cut,len(rows))}; cells={}; proj={}; ctl={}
 for task,ix in ids.items():
  if scales is not None: ctl[task]=float(db[ix].square().mean().sqrt())/scales[task]; continue
  cells[f"{task}_behavior"]=float((db[ix]-full_b[ix]).square().sum()/full_b[ix].square().sum()); proj[task]=float(db[ix]@full_b[ix]/full_b[ix].square().sum())
  for j in range(2):cells[f"{task}_mode{j+1}"]=float((dm[ix,j]-full_m[ix,j]).square().sum()/full_m[ix,j].square().sum())
 return {"worst_target_residual":max(cells.values()),"cells":cells,"behavior_signed_projection":proj} if scales is None else ctl
def main():
 if {"prior":sha(PRIOR),"old":sha(OLD),"control":sha(CONTROL)}!=EXPECTED:raise RuntimeError("authority changed")
 dry={"candidate_id":"temporal_auxiliary.five_mlp_target_contrast_response_basis_v1","dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"model_forwards_max":16,"ranks":RANKS,"fit_updates":0,"model_updates":0,"transformer_backwards":0}
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 old=json.loads(OLD.read_text()); generic=json.loads(comp.GENERIC.read_text()); tcap,icap=json.loads(comp.TCAP.read_text()),json.loads(comp.ICAP.read_text()); tr,ir,*_=greedy.rows_and_controls(tcap,icap)
 td=tr[::2]+ir[::2]; tv=tr[1::2]+ir[1::2]; controls=matched.control_rows(); cd,cv=controls["discovery"],controls["validation"]
 tic=time.perf_counter(); backend=producer.Bilin18TorchBackend.load("cuda"); t=backend.torch; reader,orientation,reader_ok=greedy.physical_reader(backend,json.loads(comp.WEIGHTS.read_text()))
 cdb,cdo,cd0,cdd,cd1=cap(backend,cd); tdb,tdo,td0,tdd,td1=cap(backend,td); tvb,tvo,tv0,tvd,tv1=cap(backend,tv); cvb,cvo,cv0,cvd,cv1=cap(backend,cv)
 cb,_=comp.fit_bases(backend,cdb,cd0,cd1); bases,spectra=fit_target(backend,tdb,td0,td1,cb)
 tvbs,tvds=[atlasrun.states(t,backend,x,tv) for x in (tvo,tvd)]; cvbs,cvds=[atlasrun.states(t,backend,x,cv) for x in (cvo,cvd)]
 full_t,_=atlasrun.run_patch(backend,tvb,tv1,comp.SITES); full_c,_=atlasrun.run_patch(backend,cvb,cv1,comp.SITES); reports={"0":{"target":report(backend,tv,tvbs,tvds,atlasrun.states(t,backend,full_t,tv),reader,len(tr[1::2])),"control":report(backend,cv,cvbs,cvds,atlasrun.states(t,backend,full_c,cv),reader,sum(r["task_id"]==matched.temporal.TASK_ID for r in cv),generic["target_behavior_rms_scales"])}}
 for rank in RANKS:
  to=run_project(backend,tvb,tv0,tv1,bases,rank); co=run_project(backend,cvb,cv0,cv1,bases,rank); reports[str(rank)]={"target":report(backend,tv,tvbs,tvds,atlasrun.states(t,backend,to,tv),reader,len(tr[1::2])),"control":report(backend,cv,cvbs,cvds,atlasrun.states(t,backend,co,cv),reader,sum(r["task_id"]==matched.temporal.TASK_ID for r in cv),generic["target_behavior_rms_scales"]) }
 candidates=[r for r in (4,8) if reports[str(r)]["target"]["worst_target_residual"]<=.15 and all(reports[str(r)]["target"]["behavior_signed_projection"][x]>=.8 for x in ("temporal","iswas"))]; chosen=min(candidates,key=lambda r:max(reports[str(r)]["control"].values())) if candidates else None
 pa=reader_ok and orientation<=1e-6 and all(math.isfinite(v) for z in reports.values() for side in z.values() for v in ([side["worst_target_residual"]]+list(side["cells"].values()) if "cells" in side else side.values())); pb=reports["0"]["target"]["worst_target_residual"]<=.05; pc=chosen is not None; pd=pc and all(reports[str(chosen)]["control"][x]<=.1 for x in ("temporal","iswas")); baseline=old["reports"]["8"]; pe=pd and reports[str(chosen)]["target"]["worst_target_residual"]<=baseline["worst_target_residual"] and max(reports[str(chosen)]["control"].values())<=.5*max(baseline["validation_control_fraction"].values())
 preds={"pred_a_authority_split_alignment_finiteness_and_price":bool(pa),"pred_b_full_response_validation_program_is_sufficient":bool(pb),"pred_c_target_basis_preserves_heldout_behavior":bool(pc),"pred_d_target_basis_is_selective":bool(pd),"pred_e_basis_beats_control_complement":bool(pe)}; terminal="invalid" if not(pa and pb) else "heldout_target_contrast_program" if all(preds.values()) else "target_control_entangled_null" if pc else "target_basis_transfer_null"; result={"schema":"temporal_five_mlp_target_contrast_response_basis_result_v1","started_utc":now(),"finished_utc":now(),"serial_seconds":time.perf_counter()-tic,"authority_sha256":EXPECTED,"spectra":spectra,"reports":reports,"chosen_rank":chosen,"predictions":preds,"terminal":terminal,"price":{"model_forwards":16,"fit_updates":0,"model_updates":0,"transformer_backwards":0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("reports","chosen_rank","predictions","terminal","price")},sort_keys=True))
if __name__=="__main__":main()
