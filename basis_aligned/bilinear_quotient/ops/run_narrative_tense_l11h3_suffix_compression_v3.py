#!/usr/bin/env python3
"""Compress the licensed narrative L11H3 carrier over self-relative suffix sources."""
# BQGATE: EXPERIMENT
from __future__ import annotations
from collections import defaultdict
import argparse, hashlib, json, os, statistics
from pathlib import Path
from typing import Sequence
import attention_source_factor_primitive as primitive
import circuit_fast_screen_managed_runner as managed
import run_narrative_tense_l11h3_combined_carrier_v1 as carrier

ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/'circuits/prior_art/NARRATIVE_TENSE_L11H3_SUFFIX_COMPRESSION_V3_PREREGISTRATION.md'
OUT=ROOT/'circuits/fast_screens/narrative_tense_l11h3_suffix_compression_v3_result.json'
PRIOR_SHA256='381fb13cc87f81e6aa5efa6e46ef71995eba50cf16e1cf5dff6001efab0f5e32'
ARMS=('expanded_native','native_reinstall','complete_head','all_post_value','early_three_value','suffix_three_value','role_value','of_the_value')
PREDICTION_REGISTRY={'pred_a_instrument_live':None,'pred_b_shared_suffix_compresses_post_carrier':None,'pred_c_role_single_source':None,'pred_d_of_the_source_pair':None,'pred_e_suffix_downstream_interaction':None,'pred_f_suffix_distributed_additive':None,'pred_g_suffix_compression_null':None}
BARS={'maximum_exact_error':5e-5,'minimum_complete_donorward_fraction':.75,'minimum_suffix_fraction_of_all_post':.85,'maximum_early_absolute_fraction_of_all_post':.25,'minimum_component_fraction_of_suffix':.70,'minimum_interaction_absolute_fraction_of_suffix':.20,'minimum_donorward_fraction':.75,'maximum_control_fraction_of_smallest_target_all_post':.25}
class ScreenError(ValueError):pass
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows():
 out=carrier.build_rows()
 enc=carrier.authority.ENCODING
 for r in out:
  n=len(r['base_ids']); self=n-1
  r['compression_positions']={'all_post':tuple(range(self-6,self)),'early_three':tuple(range(self-6,self-3)),'suffix_three':tuple(range(self-3,self)),'role':(self-3,),'of_the':tuple(range(self-2,self))}
  if tuple(enc.decode([r['base_ids'][i]]) for i in r['compression_positions']['suffix_three']) != (' role',' of',' the'):
   raise ScreenError('shared self-relative suffix changed')
  if any(r['base_ids'][i]!=r['donor_ids'][i] for i in r['compression_positions']['suffix_three']):raise ScreenError('suffix is not token-identical')
  if r['family'] in {'A1','A2'} and any(r['base_ids'][i]!=r['donor_ids'][i] for i in r['compression_positions']['all_post']):raise ScreenError('target post carrier contains a changed token')
 return out
def compile_plan():
 if sha(PRIOR)!=PRIOR_SHA256:raise ScreenError('receipt changed')
 carrier.compile_plan()
 return {'schema':'narrative_tense_l11h3_suffix_compression_plan_v3','candidate_id':'narrative_tense.l11h3_suffix_compression_v3','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'prior_art_sha256':PRIOR_SHA256,'capability_license_sha256':carrier.LICENSE_SHA256,'row_count':len(rows()),'layer':carrier.core.LAYER,'head':carrier.core.HEAD,'arms':list(ARMS),'bars':BARS,'registered_predictions':list(PREDICTION_REGISTRY),'price':{'model_forwards':6,'example_evaluations':768,'backwards':0,'parameter_updates':0}}
def install(base,donor,pos,torch):
 mask=torch.zeros_like(base['p'],dtype=torch.bool);mask[:,tuple(pos)]=True
 return primitive.replace_head_source_subset(base,donor,mask,'value',torch)
def metrics(logits,q,row,torch):
 da,ba=int(row['donor_answer_id']),int(row['base_answer_id']);foil=int(row['base_foil_id'])
 return {'donor_margin':float(logits[q,da]-logits[q,ba]),'donor_ce':float(-torch.log_softmax(logits[q],-1)[da]),'answer_margin':float(logits[q,ba]-logits[q,foil]),'answer_ce':float(-torch.log_softmax(logits[q],-1)[ba])}
def evaluate(model,torch,F,facade):
 rr=rows();device=next(model.parameters()).device;factor=carrier.core.factor
 ml=max(len(r[x]) for r in rr for x in ('base_ids','donor_ids'))
 bt,bf=factor._pad(rr,'base',ml,torch,device);dt,df=factor._pad(rr,'donor',ml,torch,device)
 evidence=[];exact={'source_sum_max_absolute_error':0.0,'native_reinstall_max_absolute_error':0.0,'early_plus_suffix_head_max_absolute_error':0.0,'role_plus_of_the_head_max_absolute_error':0.0}
 for start in range(0,len(rr),carrier.core.BATCH):
  chunk=rr[start:start+carrier.core.BATCH];btt,bff=bt[start:start+carrier.core.BATCH],bf[start:start+carrier.core.BATCH];dtt,dff=dt[start:start+carrier.core.BATCH],df[start:start+carrier.core.BATCH]
  blog,b=factor._factor_forward(model,btt,bff,torch,F,facade);dlog,d=factor._factor_forward(model,dtt,dff,torch,F,facade)
  for cap in (b,d): exact['source_sum_max_absolute_error']=max(exact['source_sum_max_absolute_error'],float((torch.einsum('bk,bkd->bd',cap['p'],cap['u'])-cap['head']).abs().max()))
  inds=[];heads=[];spec=[]
  for i,r in enumerate(chunk):
   bb={k:v[i:i+1] for k,v in b.items()};dd={k:v[i:i+1] for k,v in d.items()};p=r['compression_positions']
   repl={name:install(bb,dd,p[name],torch)[0] for name in ('all_post','early_three','suffix_three','role','of_the')}
   exact['early_plus_suffix_head_max_absolute_error']=max(exact['early_plus_suffix_head_max_absolute_error'],float((repl['all_post']-(repl['early_three']+repl['suffix_three']-b['head'][i])).abs().max()))
   exact['role_plus_of_the_head_max_absolute_error']=max(exact['role_plus_of_the_head_max_absolute_error'],float((repl['suffix_three']-(repl['role']+repl['of_the']-b['head'][i])).abs().max()))
   hs={'expanded_native':b['head'][i],'native_reinstall':b['head'][i],'complete_head':d['head'][i],'all_post_value':repl['all_post'],'early_three_value':repl['early_three'],'suffix_three_value':repl['suffix_three'],'role_value':repl['role'],'of_the_value':repl['of_the']}
   for a in ARMS:inds.append(i);heads.append(hs[a]);spec.append((i,a))
  ix=torch.tensor(inds,dtype=torch.long,device=device)
  plog,_=factor._factor_forward(model,btt[ix],bff[ix],torch,F,facade,replacement_heads=torch.stack(heads),replacement_head_mask=torch.tensor([a!='expanded_native' for _,a in spec],dtype=torch.bool,device=device),native_reinstall_mask=torch.tensor([a=='native_reinstall' for _,a in spec],dtype=torch.bool,device=device))
  expanded={i:metrics(plog[oi],int(bff[i]),chunk[i],torch) for oi,(i,a) in enumerate(spec) if a=='expanded_native'}
  expanded_logits={i:plog[oi] for oi,(i,a) in enumerate(spec) if a=='expanded_native'}
  for oi,(i,a) in enumerate(spec):
   r=chunk[i];q=int(bff[i]);native=expanded[i];changed=metrics(plog[oi],q,r,torch)
   if a=='native_reinstall':exact['native_reinstall_max_absolute_error']=max(exact['native_reinstall_max_absolute_error'],float((plog[oi]-expanded_logits[i]).abs().max()))
   evidence.append({'row_id':r['row_id'],'family':r['family'],'cell_id':f"{r['family']}/{r['direction_id']}",'arm':a,'margin_delta':changed['donor_margin']-native['donor_margin'],'donor_ce_gain':native['donor_ce']-changed['donor_ce'],'answer_margin_delta':changed['answer_margin']-native['answer_margin'],'base_answer_CE_change':changed['answer_ce']-native['answer_ce']})
 return evidence,exact
def mean(xs):return statistics.fmean(xs)
def summarize(ev):
 out={}
 for cell in sorted({x['cell_id'] for x in ev if x['family'] in {'A1','A2'}}):
  out[cell]={}
  for arm in ARMS:
   z=[x for x in ev if x['cell_id']==cell and x['arm']==arm];out[cell][arm]={'count':len(z),'mean_margin_delta':mean([x['margin_delta'] for x in z]),'mean_CE_gain':mean([x['donor_ce_gain'] for x in z]),'donorward_fraction':mean([x['margin_delta']>0 for x in z])}
 return out
def score(ev,exact):
 cells=summarize(ev);instrument=max(exact.values())<=BARS['maximum_exact_error'] and all(c['complete_head']['mean_margin_delta']>0 and c['complete_head']['mean_CE_gain']>0 and c['complete_head']['donorward_fraction']>=BARS['minimum_complete_donorward_fraction'] for c in cells.values())
 reports={}
 for cell,c in cells.items():
  post=c['all_post_value'];suffix=c['suffix_three_value'];early=c['early_three_value'];role=c['role_value'];ofthe=c['of_the_value']
  def frac(x,d,k):return x[k]/d[k] if abs(d[k])>1e-12 else float('-inf')
  reports[cell]={'suffix_margin_fraction':frac(suffix,post,'mean_margin_delta'),'suffix_CE_fraction':frac(suffix,post,'mean_CE_gain'),'early_margin_fraction':frac(early,post,'mean_margin_delta'),'early_CE_fraction':frac(early,post,'mean_CE_gain'),'role_margin_fraction':frac(role,suffix,'mean_margin_delta'),'role_CE_fraction':frac(role,suffix,'mean_CE_gain'),'of_the_margin_fraction':frac(ofthe,suffix,'mean_margin_delta'),'of_the_CE_fraction':frac(ofthe,suffix,'mean_CE_gain'),'interaction_margin_fraction':(suffix['mean_margin_delta']-role['mean_margin_delta']-ofthe['mean_margin_delta'])/suffix['mean_margin_delta'],'interaction_CE_fraction':(suffix['mean_CE_gain']-role['mean_CE_gain']-ofthe['mean_CE_gain'])/suffix['mean_CE_gain'],'suffix_donorward_fraction':suffix['donorward_fraction'],'role_donorward_fraction':role['donorward_fraction'],'of_the_donorward_fraction':ofthe['donorward_fraction']}
  x=reports[cell];x['suffix_passed']=x['suffix_margin_fraction']>=.85 and x['suffix_CE_fraction']>=.85 and abs(x['early_margin_fraction'])<=.25 and abs(x['early_CE_fraction'])<=.25 and x['suffix_donorward_fraction']>=.75;x['role_passed']=x['role_margin_fraction']>=.70 and x['role_CE_fraction']>=.70 and x['role_donorward_fraction']>=.75;x['of_the_passed']=x['of_the_margin_fraction']>=.70 and x['of_the_CE_fraction']>=.70 and x['of_the_donorward_fraction']>=.75;x['interaction_passed']=abs(x['interaction_margin_fraction'])>=.20 and abs(x['interaction_CE_fraction'])>=.20
 target_margin=min(abs(c['all_post_value']['mean_margin_delta']) for c in cells.values());target_ce=min(abs(c['all_post_value']['mean_CE_gain']) for c in cells.values());limits={'answer_margin':.25*target_margin,'full_vocab_CE':.25*target_ce};controls={}
 for fam in ('P','C'):
  controls[fam]={}
  for arm in ('suffix_three_value','role_value','of_the_value'):
   z=[x for x in ev if x['family']==fam and x['arm']==arm];am=mean([abs(x['answer_margin_delta']) for x in z]);ce=mean([abs(x['base_answer_CE_change']) for x in z]);controls[fam][arm]={'mean_absolute_answer_margin_change':am,'mean_absolute_full_vocab_CE_change':ce,'passed':am<=limits['answer_margin'] and ce<=limits['full_vocab_CE']}
 selective=lambda arm:all(controls[f][arm]['passed'] for f in controls)
 suffix_ok=all(x['suffix_passed'] for x in reports.values()) and selective('suffix_three_value');role_ok=suffix_ok and all(x['role_passed'] for x in reports.values()) and selective('role_value');ofthe_ok=suffix_ok and all(x['of_the_passed'] for x in reports.values()) and selective('of_the_value');inter=suffix_ok and not role_ok and not ofthe_ok and all(x['interaction_passed'] for x in reports.values())
 preds={'pred_a_instrument_live':instrument,'pred_b_shared_suffix_compresses_post_carrier':instrument and suffix_ok,'pred_c_role_single_source':instrument and role_ok,'pred_d_of_the_source_pair':instrument and ofthe_ok,'pred_e_suffix_downstream_interaction':instrument and inter,'pred_f_suffix_distributed_additive':instrument and suffix_ok and not role_ok and not ofthe_ok and not inter,'pred_g_suffix_compression_null':instrument and not suffix_ok}
 return {'exactness':exact,'target_cells':cells,'compression':reports,'control_limits':limits,'controls':controls,'predictions':preds}
def terminal(p):
 if not p['pred_a_instrument_live']:return 'invalid'
 if p['pred_g_suffix_compression_null']:return 'shared_suffix_compression_null'
 if p['pred_c_role_single_source']:return 'role_single_source'
 if p['pred_d_of_the_source_pair']:return 'of_the_source_pair'
 if p['pred_e_suffix_downstream_interaction']:return 'suffix_downstream_interaction'
 if p['pred_f_suffix_distributed_additive']:return 'suffix_distributed_additive'
 return 'suffix_compressed_unclassified'
def main(argv:Sequence[str]|None=None):
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');args=ap.parse_args(argv);plan=compile_plan()
 if args.dry_run or os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise ScreenError('refusing overwrite')
 torch,F,facade=carrier.core.factor._dependencies();model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():ev,exact=evaluate(model,torch,F,facade)
 scored=score(ev,exact);term=terminal(scored['predictions']);result={'schema':'narrative_tense_l11h3_suffix_compression_result_v3','candidate_id':plan['candidate_id'],'terminal':term,'plan':plan,'checkpoint_weights_sha256':checkpoint.weights_sha256,'score':scored,'evidence':ev,'active_price':plan['price']};payload=managed.atomic_create_json(OUT,result);print(json.dumps({'terminal':term,'result_sha256':hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=='__main__':main()
