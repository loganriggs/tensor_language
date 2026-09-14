#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT
"""Fit list cached deltas; test a frozen low-rank basis on +1 and copy."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,os,statistics,sys
from pathlib import Path
RUNNER=Path(__file__).resolve();OPS=RUNNER.parent;ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
LIST_ROWS=POLY/'NUMERIC_SUCCESSOR_SPACED_OOD_V1_ROWS.json';LIST_CAP=POLY/'NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_RESULT.json';PLUS_ROWS=POLY/'NUMERIC_SEQUENCE_NONADJACENT_OOD_V1_ROWS.json';PLUS_CAP=POLY/'NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_RESULT.json';PREREG=POLY/'NUMERIC_SHARED_PAYLOAD_LOW_RANK_V1_PREREGISTRATION.md';OUT=POLY/'NUMERIC_SHARED_PAYLOAD_LOW_RANK_V1_RESULT.json'
HASHES={'list_rows':'90638d886070685f15513dde5dde2b11f8ed806c435e92976eded918e8ac709b','list_cap':'a8acecf1d7325508f09355e62f1798ffd056c8bafcf5296d86cfa82d54f7bcc8','plus_rows':'3cc955eafaed96e6baad778f6d6b6e661b6c40e0b84d3aafd09e5b2078535fae','plus_cap':'9ef85ada3df8893ac0d26a3eaec71a0e4aa178a999fac4c57c6ab7decbd2b0d9','prereg':'b91630b900887d7e82fd059914502969040f32fd06f4f19ed4bd8bf55e3c9941'};RANKS=(1,2,4,8);PRICE={'forwards':6,'sequences':736,'fits':1,'updates':0}
PREDICTION_REGISTRY={'pred_a_exact_instrument':None,'pred_b_fit_rank_selected':None,'pred_c_cross_authority_shared_payload':None,'pred_d_plus_one_only':None,'pred_e_copy_only':None,'pred_f_cross_authority_null':None,'pred_g_rank_one':None,'pred_h_rank_two':None}
BARS={'minimum_margin_and_CE_recovery':.80,'minimum_donorward_fraction':.75,'maximum_native_replay_rse':1e-10,'maximum_source_sum_rse':1e-10,'maximum_cached_decomposition_rse':1e-10,'maximum_installed_error':1e-5}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
 paths={'list_rows':LIST_ROWS,'list_cap':LIST_CAP,'plus_rows':PLUS_ROWS,'plus_cap':PLUS_CAP,'prereg':PREREG}
 for k,p in paths.items():
  if digest(p)!=HASHES[k]:raise RuntimeError('bound file changed '+k)
 lr=json.loads(LIST_ROWS.read_text());pr=json.loads(PLUS_ROWS.read_text())
 for p in (LIST_CAP,PLUS_CAP):
  x=json.loads(p.read_text())
  if 'pass' not in x['terminal'] or not all(x['predictions'].values()):raise RuntimeError('capability not passing')
 fit=[x for x in lr['rows'] if x['program_role']=='list_step_two_spaced'];copy=[x for x in lr['rows'] if x['program_role'] in ('digit_copy_spaced_control','word_copy_spaced_control')]
 if len(fit)!=16 or len(copy)!=32 or len(pr['rows'])!=32:raise RuntimeError('coverage changed')
 test=[{'test_family':'plus_one','representation':x['representation'],**x} for x in pr['rows']]+[{'test_family':'copy','representation':'digit' if x['program_role'].startswith('digit') else 'number_word',**x} for x in copy]
 return fit,test,lr['row_manifest_sha256'],pr['row_manifest_sha256']
def plan():
 fit,test,lh,ph=load();return {'schema':'numeric_shared_payload_low_rank_v1_plan','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'fit_pairs':len(fit),'test_pairs':len(test),'ranks':list(RANKS),'fixed_layer':8,'fixed_heads':[3,7],'factor':'final_source_cached_value_only','bars':BARS,'price':PRICE,'list_manifest_sha256':lh,'plus_manifest_sha256':ph,'predicates':list(PREDICTION_REGISTRY)}
def _list_positions(text,ids,enc):
 pos=[];prefix=''
 for line in text.splitlines(keepends=True):
  label=line.split('.',1)[0]
  if label.isdigit():pos.append(len(enc.encode(prefix)))
  prefix+=line
 if len(pos)!=3 or enc.encode(prefix)!=ids:return (_ for _ in ()).throw(RuntimeError('list mapping'))
 return pos
def endpoints(rows,kind,semantic):
 out=[];idx={}
 for row in rows:
  for side in ('base','donor'):
   x=dict(row[side]);x['query_position']=len(x['ids'])-1;x['source_positions']=_list_positions(x['text'],x['ids'],semantic.ENC) if kind=='fit' or row.get('program_role','').startswith('list_') else [z['token_position'] for z in semantic.endpoint_mapping(x['ids'])['value_positions']];idx[(row['row_id'],side)]=len(out);out.append(x)
 return out,idx
def directions(rows,idx):
 return [(row,d,idx[(row['row_id'],r)],idx[(row['row_id'],o)]) for row in rows for d,r,o in (('base_to_donor','base','donor'),('donor_to_base','donor','base'))]
def capture(model,eps,exact,torch):
 tok,fin,pos=exact._pad(eps,'cuda');native_full=exact.r573.native_logits(model,tok);native=native_full[torch.arange(len(eps),device='cuda'),fin];replay,cap,diag=exact._capture_forward(model,tok,fin,pos);rse=float((native-replay).square().sum())/max(float(native.square().sum()),1e-30);return replay,cap,diag,rse
def replacement(rec,don,exact):return exact._replace(rec,don,'cached')
def delta(rec,don,exact):return (replacement(rec,don,exact)-rec['complete']).reshape(-1)
def run_stage(model,rows,kind,basis,ranks,fit_basis,exact,semantic,torch):
 eps,idx=endpoints(rows,kind,semantic);replay,cap,diag,rse=capture(model,eps,exact,torch);spec=directions(rows,idx);D=torch.stack([delta({k:v[ri] for k,v in cap.items()},{k:v[di] for k,v in cap.items()},exact) for _,_,ri,di in spec])
 if fit_basis:
  sv=torch.linalg.svdvals(D);_,_,vh=torch.linalg.svd(D,full_matrices=False);basis=vh.detach().clone();singular=[float(x) for x in sv]
 else:singular=[]
 examples=[];repls=[];meta=[];energy=defaultdict(list)
 for si,(row,d,ri,di) in enumerate(spec):
  rec={k:v[ri] for k,v in cap.items()};full=replacement(rec,{k:v[di] for k,v in cap.items()},exact);examples.append(eps[ri]);repls.append(full);meta.append((row,d,'full_cached',ri,di));dv=D[si]
  for rank in ranks:
   B=basis[:rank];proj=(dv@B.T)@B;examples.append(eps[ri]);repls.append(rec['complete']+proj.reshape_as(rec['complete']));meta.append((row,d,f'rank_{rank}',ri,di));energy[f'rank_{rank}'].append(float(torch.dot(proj,proj)/torch.dot(dv,dv)))
 pt,pf,_=exact._pad(examples,'cuda');patched,norms,pdiag,ierr=exact._patched_forward(model,pt,pf,torch.stack(repls));ev=[]
 for i,(row,d,arm,ri,di) in enumerate(meta):
  ra=int(eps[ri]['answer_id']);da=int(eps[di]['answer_id']);before=replay[ri];after=patched[i];mb=exact._margin(before,da,ra);ma=exact._margin(after,da,ra)
  ev.append({'row_id':row['row_id'],'family':'fit_list' if kind=='fit' else row['test_family'],'construction':row['construction'],'representation':row.get('representation','list'),'direction':d,'arm':arm,'donor_margin_effect':ma-mb,'donor_ce_gain':exact._ce(before,da)-exact._ce(after,da),'donorward':ma-mb>0,'intervention_norm':float(norms[i])})
 diagnostics={'native_replay_rse':rse,'source_sum_rse':max(diag['head_source_sum_relative_squared_error'],pdiag['head_source_sum_relative_squared_error']),'cached_decomposition_rse':max(diag['value_split_relative_squared_error'],pdiag['value_split_relative_squared_error']),'installed_error':ierr,'singular_values':singular,'mean_projection_energy':{k:statistics.fmean(v) for k,v in energy.items()}}
 return ev,basis,diagnostics,len(eps)+len(examples)
def score(ev,ranks):
 grouped=defaultdict(list)
 for x in ev:grouped[(x['family'],x['construction'],x['representation'],x['direction'],x['arm'])].append(x)
 reports={};family_pass=defaultdict(list)
 for key,z in sorted(grouped.items()):
  full=grouped[key[:-1]+('full_cached',)];fm=statistics.fmean(x['donor_margin_effect'] for x in full);fc=statistics.fmean(x['donor_ce_gain'] for x in full);m=statistics.fmean(x['donor_margin_effect'] for x in z);c=statistics.fmean(x['donor_ce_gain'] for x in z);rep={'n':len(z),'mean_margin_effect':m,'mean_CE_gain':c,'margin_recovery':m/fm if abs(fm)>1e-12 else None,'CE_recovery':c/fc if abs(fc)>1e-12 else None,'donorward_fraction':statistics.fmean(x['donorward'] for x in z)}
  if key[-1]!='full_cached':rep['passed']=rep['margin_recovery']>=.80 and rep['CE_recovery']>=.80 and rep['donorward_fraction']>=.75;family_pass[(key[0],key[-1])].append(rep['passed'])
  reports['|'.join(key)]=rep
 passed={f'{fam}|{arm}':all(v) for (fam,arm),v in family_pass.items()};return reports,passed
def exact_ok(d):return d['native_replay_rse']<=1e-10 and d['source_sum_rse']<=1e-10 and d['cached_decomposition_rse']<=1e-10 and d['installed_error']<=1e-5
def main():
 p=plan()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(p,indent=2,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 sys.path.insert(0,str(OPS));import torch;import run_attn8_h3_h7_cross_behavior_factor_interchange_v2 as exact;import numeric_sequence_semantic_positions_rung577 as semantic;from circuit_fast_screen_managed_runner import atomic_create_json
 fit,test,lh,ph=load();model,checkpoint=exact.r573.facade.load_bilin18(device='cuda',dtype=torch.float32,verify_weights_sha256=True)
 with torch.inference_mode():fe,basis,fd,fs=run_stage(model,fit,'fit',None,RANKS,True,exact,semantic,torch);fr,fp=score(fe,RANKS);eligible=[r for r in RANKS if fp.get(f'fit_list|rank_{r}',False)];selected=min(eligible) if exact_ok(fd) and eligible else None
 if selected is not None:
  with torch.inference_mode():te,_,td,ts=run_stage(model,test,'test',basis,(selected,),False,exact,semantic,torch);tr,tp=score(te,(selected,))
 else:te=[];td={};ts=0;tr={};tp={}
 instrument=exact_ok(fd) and (selected is None or exact_ok(td));plus=selected is not None and tp.get(f'plus_one|rank_{selected}',False);copy=selected is not None and tp.get(f'copy|rank_{selected}',False);pred={'pred_a_exact_instrument':instrument,'pred_b_fit_rank_selected':instrument and selected is not None,'pred_c_cross_authority_shared_payload':instrument and plus and copy,'pred_d_plus_one_only':instrument and plus and not copy,'pred_e_copy_only':instrument and copy and not plus,'pred_f_cross_authority_null':instrument and selected is not None and not plus and not copy,'pred_g_rank_one':instrument and plus and copy and selected==1,'pred_h_rank_two':instrument and plus and copy and selected==2};terminal='invalid' if not instrument else 'fit_no_rank' if selected is None else f'shared_numeric_payload_rank_{selected}' if plus and copy else 'plus_one_only' if plus else 'copy_only' if copy else 'cross_authority_null';result={'schema':'numeric_shared_payload_low_rank_v1_result','terminal':terminal,'predictions':pred,'selected_rank':selected,'fit':{'diagnostics':fd,'reports':fr,'rank_pass':fp,'evidence':fe},'test':{'opened':selected is not None,'diagnostics':td,'reports':tr,'rank_pass':tp,'evidence':te},'price':{**PRICE,'observed_sequences':fs+ts},'manifests':{'list':lh,'plus':ph},'checkpoint_weights_sha256':checkpoint.weights_sha256,'runner_sha256':digest(RUNNER)};atomic_create_json(OUT,result);print(json.dumps({'terminal':terminal,'selected_rank':selected,'predictions':pred,'observed_sequences':fs+ts},sort_keys=True))
if __name__=='__main__':main()
