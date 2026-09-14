#!/usr/bin/env python3
"""FIT-select and lexically confirm a low-rank basis for narrative post-value deltas."""
# BQGATE: EXPERIMENT
from __future__ import annotations
from collections import defaultdict
import argparse,hashlib,json,os,statistics
from pathlib import Path
from typing import Sequence
import circuit_fast_screen_managed_runner as managed
import run_narrative_tense_l11h3_suffix_compression_v3 as suffix
ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/'circuits/prior_art/NARRATIVE_TENSE_L11H3_POST_VALUE_LOW_RANK_V1_PREREGISTRATION.md';OUT=ROOT/'circuits/fast_screens/narrative_tense_l11h3_post_value_low_rank_v1_result.json';PRIOR_SHA256='56491e8ee55b0f1861a9f4959d493daeedd45ea42e8ca90b481ed4605657c3b9';RANKS=(1,2,4,8)
PREDICTION_REGISTRY={'pred_a_exact_instrument':None,'pred_b_fit_rank_selected':None,'pred_c_holdout_low_rank_transfer':None,'pred_d_rank_at_most_two':None,'pred_e_rank_four':None,'pred_f_rank_eight':None,'pred_g_holdout_or_fit_null':None}
BARS={'maximum_exact_error':5e-5,'minimum_target_margin_and_CE_fraction':.80,'minimum_target_donorward_fraction':.75,'maximum_control_fraction_of_smallest_target_full_post':.25}
class LowRankError(ValueError):pass
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compile_plan():
 if sha(PRIOR)!=PRIOR_SHA256:raise LowRankError('receipt changed')
 suffix.carrier.compile_plan();rr=suffix.rows()
 return {'schema':'narrative_tense_l11h3_post_value_low_rank_plan_v1','candidate_id':'narrative_tense.l11h3_post_value_low_rank_v1','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'prior_art_sha256':PRIOR_SHA256,'capability_license_sha256':suffix.carrier.LICENSE_SHA256,'fit_groups':list(range(8)),'holdout_groups':list(range(8,16)),'candidate_ranks':list(RANKS),'row_count':len(rr),'bars':BARS,'registered_predictions':list(PREDICTION_REGISTRY),'basis_fit':'uncentered SVD of 16 target FIT exact six-source effective-value head deltas; no logits, controls, or HOLDOUT','price':{'model_forwards':6,'example_evaluations':544,'backwards':0,'parameter_updates':0}}
def mean(x):return statistics.fmean(x)
def _stage(model,rr,basis,ranks,torch,F,facade,fit_basis=False):
 factor=suffix.carrier.core.factor;device=next(model.parameters()).device;ml=max(len(r[x]) for r in rr for x in ('base_ids','donor_ids'));bt,bf=factor._pad(rr,'base',ml,torch,device);dt,df=factor._pad(rr,'donor',ml,torch,device)
 blog,b=factor._factor_forward(model,bt,bf,torch,F,facade);dlog,d=factor._factor_forward(model,dt,df,torch,F,facade)
 exact={'source_sum_max_absolute_error':0.0,'native_reinstall_max_absolute_error':0.0,'full_post_reconstruction_max_absolute_error':0.0}
 for cap in (b,d):exact['source_sum_max_absolute_error']=max(exact['source_sum_max_absolute_error'],float((torch.einsum('bk,bkd->bd',cap['p'],cap['u'])-cap['head']).abs().max()))
 deltas=[]
 for i,r in enumerate(rr):
  bb={k:v[i:i+1] for k,v in b.items()};dd={k:v[i:i+1] for k,v in d.items()};full=suffix.install(bb,dd,r['compression_positions']['all_post'],torch)[0];delta=full-b['head'][i];deltas.append(delta)
 D=torch.stack(deltas)
 if fit_basis:
  target=torch.stack([D[i] for i,r in enumerate(rr) if r['family'] in {'A1','A2'}]);sv=torch.linalg.svdvals(target);_,_,vh=torch.linalg.svd(target,full_matrices=False);basis=vh.detach().clone();singular=[float(x) for x in sv]
 else:singular=[]
 if basis is None:raise LowRankError('basis unavailable')
 arms=('expanded_native','native_reinstall','complete_head','full_post_value')+tuple(f'rank_{r}' for r in ranks);inds=[];heads=[];spec=[];energies=defaultdict(list)
 for i,r in enumerate(rr):
  delta=D[i];hs={'expanded_native':b['head'][i],'native_reinstall':b['head'][i],'complete_head':d['head'][i],'full_post_value':b['head'][i]+delta}
  for rank in ranks:
   B=basis[:rank];proj=(delta@B.T)@B;hs[f'rank_{rank}']=b['head'][i]+proj;den=float(torch.dot(delta,delta));energies[f"{r['family']}/{r['direction_id']}/rank_{rank}"].append(float(torch.dot(proj,proj))/den if den>1e-20 else 1.0)
  for a in arms:inds.append(i);heads.append(hs[a]);spec.append((i,a))
 ix=torch.tensor(inds,dtype=torch.long,device=device);mask=torch.tensor([a!='expanded_native' for _,a in spec],dtype=torch.bool,device=device);nmask=torch.tensor([a=='native_reinstall' for _,a in spec],dtype=torch.bool,device=device)
 plog,_=factor._factor_forward(model,bt[ix],bf[ix],torch,F,facade,replacement_heads=torch.stack(heads),replacement_head_mask=mask,native_reinstall_mask=nmask)
 expanded={i:suffix.metrics(plog[o],int(bf[i]),rr[i],torch) for o,(i,a) in enumerate(spec) if a=='expanded_native'};expanded_logits={i:plog[o] for o,(i,a) in enumerate(spec) if a=='expanded_native'};evidence=[]
 for o,(i,a) in enumerate(spec):
  r=rr[i];q=int(bf[i]);native=expanded[i];changed=suffix.metrics(plog[o],q,r,torch)
  if a=='native_reinstall':exact['native_reinstall_max_absolute_error']=max(exact['native_reinstall_max_absolute_error'],float((plog[o]-expanded_logits[i]).abs().max()))
  evidence.append({'row_id':r['row_id'],'family':r['family'],'cell_id':f"{r['family']}/{r['direction_id']}",'arm':a,'margin_delta':changed['donor_margin']-native['donor_margin'],'donor_ce_gain':native['donor_ce']-changed['donor_ce'],'answer_margin_delta':changed['answer_margin']-native['answer_margin'],'base_answer_CE_change':changed['answer_ce']-native['answer_ce']})
 return evidence,exact,basis,{'singular_values':singular,'mean_projection_energy_by_cell':{k:mean(v) for k,v in sorted(energies.items())}}
def _summaries(ev,ranks):
 cells={}
 for cell in sorted({x['cell_id'] for x in ev if x['family'] in {'A1','A2'}}):
  cells[cell]={}
  for arm in ('complete_head','full_post_value')+tuple(f'rank_{r}' for r in ranks):
   z=[x for x in ev if x['cell_id']==cell and x['arm']==arm];cells[cell][arm]={'count':len(z),'mean_margin_delta':mean([x['margin_delta'] for x in z]),'mean_CE_gain':mean([x['donor_ce_gain'] for x in z]),'donorward_fraction':mean([x['margin_delta']>0 for x in z])}
 return cells
def score_stage(ev,ranks):
 cells=_summaries(ev,ranks);tm=min(abs(x['full_post_value']['mean_margin_delta']) for x in cells.values());tc=min(abs(x['full_post_value']['mean_CE_gain']) for x in cells.values());limits={'answer_margin':.25*tm,'full_vocab_CE':.25*tc};reports={}
 for rank in ranks:
  arm=f'rank_{rank}';target={}
  for cell,x in cells.items():
   y=x[arm];d=x['full_post_value'];mf=y['mean_margin_delta']/d['mean_margin_delta'];cf=y['mean_CE_gain']/d['mean_CE_gain'];target[cell]={'margin_fraction':mf,'CE_fraction':cf,'donorward_fraction':y['donorward_fraction'],'passed':mf>=.80 and cf>=.80 and y['donorward_fraction']>=.75}
  controls={}
  for fam in ('P','C'):
   z=[x for x in ev if x['family']==fam and x['arm']==arm];am=mean([abs(x['answer_margin_delta']) for x in z]);ce=mean([abs(x['base_answer_CE_change']) for x in z]);controls[fam]={'mean_absolute_answer_margin_change':am,'mean_absolute_full_vocab_CE_change':ce,'passed':am<=limits['answer_margin'] and ce<=limits['full_vocab_CE']}
  reports[str(rank)]={'target_cells':target,'controls':controls,'passed':all(v['passed'] for v in target.values()) and all(v['passed'] for v in controls.values())}
 complete=all(x['complete_head']['mean_margin_delta']>0 and x['complete_head']['mean_CE_gain']>0 and x['complete_head']['donorward_fraction']>=.75 for x in cells.values())
 return {'target_cells':cells,'control_limits':limits,'ranks':reports,'complete_head_live':complete}
def main(argv:Sequence[str]|None=None):
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args(argv);plan=compile_plan()
 if args.dry_run or os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise LowRankError('refusing overwrite')
 torch,F,facade=suffix.carrier.core.factor._dependencies();model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32,verify_weights_sha256=True);allrows=suffix.rows();fitrows=[r for r in allrows if r['phase']=='FIT'];holdrows=[r for r in allrows if r['phase']=='HOLDOUT']
 with torch.no_grad():
  fe,fx,basis,fd=_stage(model,fitrows,None,RANKS,torch,F,facade,True);fs=score_stage(fe,RANKS);exact_fit=max(fx.values())<=BARS['maximum_exact_error'] and fs['complete_head_live'];eligible=[r for r in RANKS if fs['ranks'][str(r)]['passed']];selected=min(eligible) if exact_fit and eligible else None
  if selected is not None:he,hx,_,hd=_stage(model,holdrows,basis,(selected,),torch,F,facade,False);hs=score_stage(he,(selected,));exact_hold=max(hx.values())<=BARS['maximum_exact_error'] and hs['complete_head_live'];holdpass=exact_hold and hs['ranks'][str(selected)]['passed']
  else:he=[];hx={};hd={};hs={};exact_hold=False;holdpass=False
 preds={'pred_a_exact_instrument':exact_fit and (selected is None or exact_hold),'pred_b_fit_rank_selected':exact_fit and selected is not None,'pred_c_holdout_low_rank_transfer':exact_fit and selected is not None and holdpass,'pred_d_rank_at_most_two':holdpass and selected<=2,'pred_e_rank_four':holdpass and selected==4,'pred_f_rank_eight':holdpass and selected==8,'pred_g_holdout_or_fit_null':exact_fit and (selected is None or not holdpass)}
 terminal='invalid' if not preds['pred_a_exact_instrument'] else 'fit_no_eligible_rank' if selected is None else f'low_rank_{selected}_transfer' if holdpass else f'rank_{selected}_holdout_null'
 result={'schema':'narrative_tense_l11h3_post_value_low_rank_result_v1','candidate_id':plan['candidate_id'],'terminal':terminal,'plan':plan,'checkpoint_weights_sha256':checkpoint.weights_sha256,'selected_rank':selected,'fit':{'exactness':fx,'diagnostics':fd,'score':fs,'evidence':fe},'holdout':{'opened':selected is not None,'exactness':hx,'diagnostics':hd,'score':hs,'evidence':he},'predictions':preds,'active_price':{'model_forwards':6 if selected is not None else 3,'example_evaluations':544 if selected is not None else 320,'backwards':0,'parameter_updates':0}};payload=managed.atomic_create_json(OUT,result);print(json.dumps({'terminal':terminal,'selected_rank':selected,'result_sha256':hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=='__main__':main()
