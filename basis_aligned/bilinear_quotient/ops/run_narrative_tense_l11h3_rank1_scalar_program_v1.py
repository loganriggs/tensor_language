#!/usr/bin/env python3
"""Test reusable scalar programs on the licensed narrative rank-one basis."""
# BQGATE: EXPERIMENT
from __future__ import annotations
from collections import defaultdict
import hashlib,json,os,statistics
from pathlib import Path
import circuit_fast_screen_managed_runner as managed
import run_narrative_tense_l11h3_suffix_compression_v3 as suffix
ROOT=Path(__file__).resolve().parent.parent;PRIOR=ROOT/'circuits/prior_art/NARRATIVE_TENSE_L11H3_RANK1_SCALAR_PROGRAM_V1_PREREGISTRATION.md';OUT=ROOT/'circuits/fast_screens/narrative_tense_l11h3_rank1_scalar_program_v1_result.json';PRIOR_SHA256='540aaf34ccffc9d8d43c3007933a3911f31cea47dfd50f412289c03779f6fad2'
ARMS=('expanded_native','native_reinstall','complete_head','full_post_value','row_projection','direction_scalar','construction_direction_scalar','global_signed_scalar','wrong_global_scalar')
PREDICTION_REGISTRY={'pred_a_exact_signed_axis_instrument':None,'pred_b_direction_scalar_program':None,'pred_c_global_signed_scalar_program':None,'pred_d_construction_conditioned_scalar_only':None,'pred_e_wrong_sign_negative_control':None,'pred_f_row_projection_only':None,'pred_g_scalar_program_null':None}
BARS={'maximum_exact_error':5e-5,'minimum_recovery':.80,'minimum_donorward_fraction':.75,'minimum_wrong_sign_anti_donor_fraction':.75}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def target_rows(phase):return [r for r in suffix.rows() if r['phase']==phase and r['family'] in {'A1','A2'}]
def compile_plan():
 if sha(PRIOR)!=PRIOR_SHA256:raise ValueError('receipt changed')
 suffix.carrier.compile_plan();return {'schema':'narrative_tense_l11h3_rank1_scalar_program_plan_v1','candidate_id':'narrative_tense.l11h3_rank1_scalar_program_v1','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'prior_art_sha256':PRIOR_SHA256,'capability_license_sha256':suffix.carrier.LICENSE_SHA256,'fit_target_rows':len(target_rows('FIT')),'holdout_target_rows':len(target_rows('HOLDOUT')),'arms':list(ARMS),'bars':BARS,'predicates':list(PREDICTION_REGISTRY),'price':{'forwards':5,'sequences':208,'backwards':0,'parameter_updates':0}}
def capture(model,rr,torch,F,facade):
 factor=suffix.carrier.core.factor;device=next(model.parameters()).device;ml=max(len(r[x]) for r in rr for x in ('base_ids','donor_ids'));bt,bf=factor._pad(rr,'base',ml,torch,device);dt,df=factor._pad(rr,'donor',ml,torch,device);blog,b=factor._factor_forward(model,bt,bf,torch,F,facade);dlog,d=factor._factor_forward(model,dt,df,torch,F,facade);err=max(float((torch.einsum('bk,bkd->bd',c['p'],c['u'])-c['head']).abs().max()) for c in (b,d));D=[]
 for i,r in enumerate(rr):
  bb={k:v[i:i+1] for k,v in b.items()};dd={k:v[i:i+1] for k,v in d.items()};D.append(suffix.install(bb,dd,r['compression_positions']['all_post'],torch)[0]-b['head'][i])
 return bt,bf,b,d,torch.stack(D),err
def fit(model,torch,F,facade):
 rr=target_rows('FIT');_,_,_,_,D,err=capture(model,rr,torch,F,facade);_,sv,vh=torch.linalg.svd(D,full_matrices=False);axis=vh[0].detach().clone();alpha=D@axis;groups=defaultdict(list)
 for i,r in enumerate(rr):groups[(r['family'],r['direction_id'])].append(float(alpha[i]))
 cd={'|'.join(k):statistics.fmean(v) for k,v in groups.items()};directions={d:statistics.fmean([float(alpha[i]) for i,r in enumerate(rr) if r['direction_id']==d]) for d in ('past_to_present','present_to_past')};mean_abs=statistics.fmean(abs(float(x)) for x in alpha);signs={d:1.0 if directions[d]>0 else -1.0 for d in directions};agreement={d:statistics.fmean((float(alpha[i])*directions[d])>0 for i,r in enumerate(rr) if r['direction_id']==d) for d in directions};opposite=directions['past_to_present']*directions['present_to_past']<0;return axis,cd,directions,mean_abs,signs,{'source_sum_max_error':err,'singular_values':[float(x) for x in sv],'coefficients':[{'row_id':r['row_id'],'family':r['family'],'direction':r['direction_id'],'alpha':float(alpha[i])} for i,r in enumerate(rr)],'construction_direction_means':cd,'direction_means':directions,'global_mean_absolute_alpha':mean_abs,'direction_sign_agreement':agreement,'opposite_direction_signs':opposite}
def evaluate(model,axis,cd,dmeans,mean_abs,signs,torch,F,facade):
 rr=target_rows('HOLDOUT');bt,bf,b,d,D,source_err=capture(model,rr,torch,F,facade);factor=suffix.carrier.core.factor;device=bt.device;inds=[];heads=[];spec=[]
 for i,r in enumerate(rr):
  direction=r['direction_id'];base=b['head'][i];proj=float(D[i]@axis)*axis;hs={'expanded_native':base,'native_reinstall':base,'complete_head':d['head'][i],'full_post_value':base+D[i],'row_projection':base+proj,'direction_scalar':base+dmeans[direction]*axis,'construction_direction_scalar':base+cd[r['family']+'|'+direction]*axis,'global_signed_scalar':base+signs[direction]*mean_abs*axis,'wrong_global_scalar':base-signs[direction]*mean_abs*axis}
  for a in ARMS:inds.append(i);heads.append(hs[a]);spec.append((i,a))
 ix=torch.tensor(inds,dtype=torch.long,device=device);mask=torch.tensor([a!='expanded_native' for _,a in spec],dtype=torch.bool,device=device);nmask=torch.tensor([a=='native_reinstall' for _,a in spec],dtype=torch.bool,device=device);plog,_=factor._factor_forward(model,bt[ix],bf[ix],torch,F,facade,replacement_heads=torch.stack(heads),replacement_head_mask=mask,native_reinstall_mask=nmask);expanded={i:suffix.metrics(plog[o],int(bf[i]),rr[i],torch) for o,(i,a) in enumerate(spec) if a=='expanded_native'};elog={i:plog[o] for o,(i,a) in enumerate(spec) if a=='expanded_native'};ev=[];reinstall=0.0
 for o,(i,a) in enumerate(spec):
  r=rr[i];native=expanded[i];changed=suffix.metrics(plog[o],int(bf[i]),r,torch)
  if a=='native_reinstall':reinstall=max(reinstall,float((plog[o]-elog[i]).abs().max()))
  ev.append({'row_id':r['row_id'],'family':r['family'],'cell_id':r['family']+'/'+r['direction_id'],'arm':a,'margin_delta':changed['donor_margin']-native['donor_margin'],'donor_ce_gain':native['donor_ce']-changed['donor_ce']})
 return ev,{'source_sum_max_error':source_err,'native_reinstall_max_error':reinstall}
def score(ev):
 reports={}
 for cell in sorted({x['cell_id'] for x in ev}):
  reports[cell]={};full=[x for x in ev if x['cell_id']==cell and x['arm']=='full_post_value'];fm=statistics.fmean(x['margin_delta'] for x in full);fc=statistics.fmean(x['donor_ce_gain'] for x in full)
  for arm in ARMS:
   z=[x for x in ev if x['cell_id']==cell and x['arm']==arm];m=statistics.fmean(x['margin_delta'] for x in z);c=statistics.fmean(x['donor_ce_gain'] for x in z);reports[cell][arm]={'mean_margin_delta':m,'mean_CE_gain':c,'margin_recovery':m/fm,'CE_recovery':c/fc,'donorward_fraction':statistics.fmean(x['margin_delta']>0 for x in z),'anti_donorward_fraction':statistics.fmean(x['margin_delta']<0 for x in z)}
 def passed(arm):return all(x[arm]['margin_recovery']>=.80 and x[arm]['CE_recovery']>=.80 and x[arm]['donorward_fraction']>=.75 for x in reports.values())
 wrong=all(x['wrong_global_scalar']['anti_donorward_fraction']>=.75 for x in reports.values());return reports,{a:passed(a) for a in ('row_projection','direction_scalar','construction_direction_scalar','global_signed_scalar')},wrong
def main():
 plan=compile_plan()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan,indent=2,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 torch,F,facade=suffix.carrier.core.factor._dependencies();model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():axis,cd,dm,ma,signs,fitdiag=fit(model,torch,F,facade);ev,exact=evaluate(model,axis,cd,dm,ma,signs,torch,F,facade)
 reports,passes,wrong=score(ev);axis_live=fitdiag['opposite_direction_signs'] and min(fitdiag['direction_sign_agreement'].values())==1.0;instrument=max(exact.values())<=5e-5 and fitdiag['source_sum_max_error']<=5e-5 and axis_live and passes['row_projection'];pred={'pred_a_exact_signed_axis_instrument':instrument,'pred_b_direction_scalar_program':instrument and passes['direction_scalar'],'pred_c_global_signed_scalar_program':instrument and passes['global_signed_scalar'],'pred_d_construction_conditioned_scalar_only':instrument and passes['construction_direction_scalar'] and not passes['direction_scalar'],'pred_e_wrong_sign_negative_control':instrument and wrong,'pred_f_row_projection_only':instrument and passes['row_projection'] and not any(passes[a] for a in ('direction_scalar','construction_direction_scalar','global_signed_scalar')),'pred_g_scalar_program_null':instrument and not any(passes[a] for a in ('direction_scalar','construction_direction_scalar','global_signed_scalar'))};terminal='invalid' if not instrument else 'global_signed_scalar_program' if pred['pred_c_global_signed_scalar_program'] else 'direction_scalar_program' if pred['pred_b_direction_scalar_program'] else 'construction_scalar_only' if pred['pred_d_construction_conditioned_scalar_only'] else 'row_projection_only';result={'schema':'narrative_tense_l11h3_rank1_scalar_program_result_v1','terminal':terminal,'predictions':pred,'plan':plan,'fit_diagnostics':fitdiag,'holdout_exactness':exact,'holdout_program_pass':passes,'wrong_sign_pass':wrong,'cell_reports':reports,'evidence':ev,'checkpoint_weights_sha256':checkpoint.weights_sha256,'active_price':plan['price']};payload=managed.atomic_create_json(OUT,result);print(json.dumps({'terminal':terminal,'predictions':pred,'result_sha256':hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=='__main__':main()
