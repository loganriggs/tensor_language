#!/usr/bin/env python3
"""Whole-family factorial tournament for noise/KL-regularized complement DAS."""
# BQGATE: EXPERIMENT pred_a_authority_disjointness_instrument_finiteness_and_price pred_b_regularization_wins_both_family_folds pred_c_regularized_axes_are_cross_family_stable pred_d_regularized_refit_beats_pooled_on_sealed_v12 pred_e_improvement_is_not_target_sacrifice
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as v10
import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as v12
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_h3_das_hard_feasible_kl_stability_v1 as hard
import run_temporal_h3_das_noisy_worst_environment_multi_reader_v1 as fitlib
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/'circuits/prior_art/temporal_h3_das_family_crossvalidated_regularization_tournament_v1.json';REDTEAM=ROOT.parent/'polynomial_causal/CONSTRAINED_DAS_REGULARIZATION_REDTEAM_2026-09-07.md';HARD=ROOT/'circuits/followups/temporal_h3_das_hard_feasible_kl_stability_v1_result.json';CAP8=hard.CAP8;CAP10=hard.CAP10;CAP12=ROOT/'circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json';OUT=ROOT/'circuits/followups/temporal_h3_das_family_crossvalidated_regularization_tournament_v1_result.json'
EXPECTED={'prior':'42b641000f87d1a685c986dc5c7381dbbc8661af6a973b5a105adaa7103d1d66','redteam':'ddf623d10c4d350f1d9e5113cccb5781a30f71d9b225539c2bfe9dc7c87c263a','hard_feasible':'f5202f4e9af89f84a29f786a44e24fcccc165ee4c9097cccc635278eca6d80b4','fit_library':'bf806b077da4e2fb43612f8f3b5ca318e0d45edac6b2c26ecc2f6accd218b413','v8_builder':'13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf','v8_capability':'fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff','v10_builder':'e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494','v10_capability':'9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad','v12_builder':'4cf4624361ff2bd3c87cd987a7c4d16a1eeefb1c7f78287b78319862ab8d8de9','v12_capability':'4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324'}
ARMS={'no_reg':(False,False),'noise':(True,False),'kl':(False,True),'noise_kl':(True,True)};STEPS=20;CHECKPOINTS=(0,5,10,20);LR=.025;TAU=.10;BARRIER=100.;MAX_FORWARDS=1800;MAX_UPDATES=180
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def unit(x):return x/x.norm().clamp_min(1e-30)
def score(report):return (report['secondary_worst'],report['secondary_mean'])
def contexts(backend,builder,cap):
 rows=hard.rows_for(builder,cap)
 return rows,[fitlib.attach_targets(backend,[r for r in rows if r['transform_id']==p]) for p in ('A1','A2')]
def objective(backend,ctxs,limits,raw,arm,generator):
 torch=backend.torch;noise,use_kl=ARMS[arm];axis=unit(raw);axes=hard.noisy_axes(torch,axis,generator,noise);values=[];violations=[]
 for ctx,bounds in zip(ctxs,limits):
  av=[];vv=[]
  for candidate in axes:
   pieces=fitlib.environment_loss(backend,ctx,candidate@candidate.T,grad=True)[1];terms=('margin_match','l15_match','margin_inert','l15_inert')+(('vocab_match','vocab_inert') if use_kl else ())
   av.append(sum(pieces[k] for k in terms));vv.append(sum(torch.relu(pieces[k]-bounds[k]).square() for k in hard.TARGET_TERMS))
  values.append(sum(av)/len(av));violations.append(sum(vv)/len(vv))
 values=torch.stack(values);return TAU*torch.logsumexp(values/TAU,dim=0)-TAU*math.log(len(ctxs))+BARRIER*torch.stack(violations).sum()
def fit_arm(backend,train,train_limits,selection,selection_limits,initial,arm,fold):
 torch=backend.torch;raw=initial.detach().clone().requires_grad_(True);opt=torch.optim.Adam([raw],lr=LR);generator=torch.Generator(device='cpu').manual_seed(20260907+sum(map(ord,arm+fold)));candidates=[];trace=[]
 def checkpoint(step):
  axis=unit(raw).detach().clone();report=hard.evaluate_axis(backend,selection,selection_limits,axis);trace.append({'step':step,**{k:report[k] for k in ('feasible','max_violation','secondary_mean','secondary_worst')}});candidates.append((step,axis,report))
 checkpoint(0)
 for step in range(1,STEPS+1):
  opt.zero_grad(set_to_none=True);objective(backend,train,train_limits,raw,arm,generator).backward();opt.step()
  if step in CHECKPOINTS:checkpoint(step)
 eligible=[x for x in candidates if x[2]['feasible']];best=min(eligible,key=lambda x:(*score(x[2]),x[0])) if eligible else min(candidates,key=lambda x:(x[2]['max_violation'],*score(x[2]),x[0]));return {'arm':arm,'fold':fold,'best_step':best[0],'best_axis':best[1],'best_report':best[2],'trace':trace}
def refit(backend,ctxs,limits,initial,arm,steps):
 torch=backend.torch;raw=initial.detach().clone().requires_grad_(True);opt=torch.optim.Adam([raw],lr=LR);generator=torch.Generator(device='cpu').manual_seed(20260907+sum(map(ord,arm+'refit')))
 for _ in range(steps):opt.zero_grad(set_to_none=True);objective(backend,ctxs,limits,raw,arm,generator).backward();opt.step()
 return unit(raw).detach()
def strip(fit):return {'arm':fit['arm'],'fold':fit['fold'],'best_step':fit['best_step'],'best_report':fit['best_report'],'trace':fit['trace']}
def main():
 dry={'candidate_id':'temporal_auxiliary.will_vs_had.h3_das_family_crossvalidated_regularization_tournament_v1','dryrun':True,'gpu_accessed':False,'model_loaded':False,'queue_touched':False,'arms':list(ARMS),'folds':['v8_to_v10','v10_to_v8'],'sealed':'v12','steps':STEPS,'checkpoints':CHECKPOINTS,'model_forwards_max':MAX_FORWARDS,'model_updates_max':MAX_UPDATES,'fit_parameters':128}
 if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 paths={'prior':PRIOR,'redteam':REDTEAM,'hard_feasible':HARD,'fit_library':hard.FITLIB,'v8_builder':hard.V8,'v8_capability':CAP8,'v10_builder':hard.V10,'v10_capability':CAP10,'v12_builder':ROOT/'ops/circuit_candidate_temporal_auxiliary_fresh_cues_v12.py','v12_capability':CAP12};observed={k:sha(v) for k,v in paths.items()};started=now();tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;native=backend.native;forward_count=0
 def counted(*args,**kwargs):
  nonlocal forward_count;forward_count+=1;return native(*args,**kwargs)
 backend.native=counted
 for p in backend.model.parameters():p.requires_grad_(False)
 authority=json.loads(HARD.read_text());sel=authority['selection'];pooled=unit(torch.tensor(sel['selected_axis'],device=backend.device).float().unsqueeze(1));caps=[json.loads(p.read_text()) for p in (CAP8,CAP10,CAP12)];rows8,c8=contexts(backend,v8,caps[0]);rows10,c10=contexts(backend,v10,caps[1]);rows12,c12=contexts(backend,v12,caps[2]);all_contexts=c8+c10+c12;closure=max(c['manual_base_margin_max_abs'] for c in all_contexts);limits8=hard.limits_for(backend,c8,pooled);limits10=hard.limits_for(backend,c10,pooled);limits12=hard.limits_for(backend,c12,pooled)
 folds=[('v8_to_v10',c8,limits8,c10,limits10),('v10_to_v8',c10,limits10,c8,limits8)];fits=[]
 for name,train,train_limits,valid,valid_limits in folds:
  for arm in ARMS:fits.append(fit_arm(backend,train,train_limits,valid,valid_limits,pooled,arm,name))
 by_arm={arm:[f for f in fits if f['arm']==arm] for arm in ARMS};arm_scores={arm:{'worst':max(f['best_report']['secondary_worst'] for f in fs),'mean':sum(f['best_report']['secondary_mean'] for f in fs)/2} for arm,fs in by_arm.items()};selected_arm=min(ARMS,key=lambda a:(arm_scores[a]['worst'],arm_scores[a]['mean'],list(ARMS).index(a)));selected_folds=by_arm[selected_arm];selected_steps=[f['best_step'] for f in selected_folds];final_step=min(CHECKPOINTS,key=lambda s:(abs(s-sum(selected_steps)/2),s));final_axis=refit(backend,c8+c10,limits8+limits10,pooled,selected_arm,final_step);sealed={'selected':hard.evaluate_axis(backend,c12,limits12,final_axis),'pooled':hard.evaluate_axis(backend,c12,limits12,pooled)};fold_cos=float((selected_folds[0]['best_axis'].T@selected_folds[1]['best_axis']).abs())
 pooled_folds=[hard.evaluate_axis(backend,valid,valid_limits,pooled) for _,_,_,valid,valid_limits in folds];pb=selected_arm!='no_reg' and all(f['best_step']>0 and score(f['best_report'])<score(by_arm['no_reg'][i]['best_report']) and score(f['best_report'])<score(pooled_folds[i]) for i,f in enumerate(selected_folds));pc=fold_cos>=.8;pd=sealed['selected']['feasible'] and score(sealed['selected'])<score(sealed['pooled']);pe=sealed['selected']['feasible'] and all(sum(sealed['selected']['environments'][i]['secondary_parts'][k] for k in ('vocab_match','vocab_inert'))<sum(sealed['pooled']['environments'][i]['secondary_parts'][k] for k in ('vocab_match','vocab_inert')) for i in range(2));ids=[{r['row_id'] for r in rows} for rows in (rows8,rows10,rows12)];numeric=[x for f in fits for p in f['trace'] for x in p.values() if isinstance(x,(int,float))];updates=8*STEPS+final_step;pa=observed==EXPECTED and authority['terminal']=='family_memorization' and sel['selected_source']=='pooled' and sel['selected_step']==0 and all(c['terminal']=='manifest' for c in caps) and not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]) and closure<=1e-4 and all(math.isfinite(float(x)) for x in numeric+[fold_cos]) and forward_count<=MAX_FORWARDS and updates<=MAX_UPDATES
 preds={'pred_a_authority_disjointness_instrument_finiteness_and_price':bool(pa),'pred_b_regularization_wins_both_family_folds':bool(pb),'pred_c_regularized_axes_are_cross_family_stable':bool(pc),'pred_d_regularized_refit_beats_pooled_on_sealed_v12':bool(pd),'pred_e_improvement_is_not_target_sacrifice':bool(pe)};moved=any(f['best_step']>0 for f in fits);terminal='invalid' if not pa else 'regularized_cross_family_das_candidate' if all(preds.values()) else 'objective_miss' if moved and selected_arm=='no_reg' else 'rank_one_inadequate' if not moved else 'residual_family_memorization' if pb and not(pd and pe) else 'objective_miss';result={'schema':'temporal_h3_das_family_crossvalidated_regularization_tournament_result_v1','started_utc':started,'finished_utc':now(),'serial_seconds':time.perf_counter()-tic,'authority_sha256':EXPECTED,'row_counts':{'v8':len(rows8),'v10':len(rows10),'v12':len(rows12)},'arm_scores':arm_scores,'selected_arm':selected_arm,'selected_fold_steps':selected_steps,'selected_fold_axis_cosine':fold_cos,'final_step':final_step,'fits':[strip(f) for f in fits],'sealed':sealed,'predictions':preds,'terminal':terminal,'price':{'model_forwards_observed':forward_count,'model_forwards_max':MAX_FORWARDS,'model_updates_observed':updates,'model_updates_max':MAX_UPDATES,'fit_parameters':128}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ('row_counts','arm_scores','selected_arm','selected_fold_steps','selected_fold_axis_cosine','final_step','sealed','predictions','terminal','price')},sort_keys=True))
if __name__=='__main__':main()
