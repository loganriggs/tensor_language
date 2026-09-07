#!/usr/bin/env python3
"""Attribute mask30 control flips against top50 and the full parent."""
# BQGATE: EXPERIMENT pred_a_authority_replay_finiteness_and_price pred_b_mask30_adds_no_collateral_over_full pred_c_full_parent_explains_the_failed_absolute_gate pred_d_top50_exposes_the_function_selectivity_tradeoff pred_e_mask30_and_full_control_effects_are_close
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import hidden_unit_subspace_contribution as hu
import pooled_response_projector as pooled
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlas
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_iswas_five_mlp_position_svd_ladder_v1 as svd
import run_temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v1 as compiler
import run_temporal_iswas_rank16_activation_conditioned_hidden_groups_v1 as parent
import run_temporal_iswas_five_mlp_source_dim_subspace_v1 as source
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/'circuits/prior_art/temporal_iswas_rank16_hidden_mask30_control_attribution_v1.json';HOLDOUT=ROOT/'circuits/followups/temporal_iswas_rank16_hidden_mask30_construction_holdout_v1_result.json';OUT=ROOT/'circuits/followups/temporal_iswas_rank16_hidden_mask30_control_attribution_v1_result.json';EXPECTED={'prior':'d9217313be85313964ab26d89e8694f14df74a1b878ced298c1ed15acc46bf90','holdout':'346402de851dd9d600fd767532a9b0238d90296fa3ed6305344af263bfabe63e'};SUPPORT=tuple(source.SUPPORT);MAX_FORWARDS=20
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def main():
 dry={'candidate_id':'temporal_auxiliary.iswas_rank16_hidden_mask30_control_attribution_v1','dryrun':True,'gpu_accessed':False,'model_loaded':False,'queue_touched':False,'arms':['top50','mask30','full'],'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}
 if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 observed={'prior':sha(PRIOR),'holdout':sha(HOLDOUT)};authority=json.loads(HOLDOUT.read_text());started=now();tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;native=backend.native;forward_count=0
 def counted(*args,**kwargs):
  nonlocal forward_count;forward_count+=1;return native(*args,**kwargs)
 backend.native=counted
 fitted=pooled.fit(backend);bases,_,_,_=svd.fit_position_bases(backend,fitted['target_rows']);fb,*_=interface.cap_inputs(backend,fitted['target_rows']);fh0,_=compiler.capture(backend,fb);fh1,_=compiler.capture(backend,das._batch(backend,fitted['target_rows'],side='donor'));tcap,icap=json.loads(greedy.TEMPORAL_CAPABILITY.read_text()),json.loads(greedy.ISWAS_CAPABILITY.read_text());_tr,_ir,tc,ic=greedy.rows_and_controls(tcap,icap);controls=tc+ic;cbatch=das._batch(backend,controls,side='base');cdbatch=das._batch(backend,controls,side='donor');ch0,_=compiler.capture(backend,cbatch);ch1,_=compiler.capture(backend,cdbatch);base_output,base_full=atlas.capture_native(backend,cbatch);base_state=atlas.states(torch,backend,base_output,controls);base_logits=das.head_logits(backend,base_state).float();maps={};top={};full={}
 for site in SUPPORT:
  q=bases[site]['rank16'];a=backend.model.transformer.h[int(site[3:])].mlp.Down.weight.detach().float().T@q;maps[site]=a;top[site]=hu.top_fraction_mask(hu.contribution_energy(parent.flatten(fb,fh1[site]-fh0[site]),a),.5);full[site]=torch.ones_like(top[site])
 masks={'top50':top,'mask30':{s:(top[s] if s=='MLP0' else full[s]) for s in SUPPORT},'full':full};ctx={'rows':controls,'base_logits':base_logits};reports={};logits={};finite=[]
 for name,mask in masks.items():
  out,_=parent.run_selected(backend,cbatch,base_full,{s:ch1[s]-ch0[s] for s in SUPPORT},bases,maps,mask);value=das.head_logits(backend,atlas.states(torch,backend,out,controls)).float();logits[name]=value;summary=klfit.control_metrics(backend,ctx,value);kl=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(value,dim=-1),torch.nn.functional.log_softmax(base_logits,dim=-1),log_target=True,reduction='none').sum(-1);flips=(value.argmax(-1)!=base_logits.argmax(-1));reports[name]={'summary':summary,'flipped_row_ids':[controls[i]['row_id'] for i in range(len(controls)) if bool(flips[i])],'per_row_kl':{controls[i]['row_id']:float(kl[i]) for i in range(len(controls))}};finite += list(kl.cpu())+[summary['median_kl'],summary['max_kl'],summary['top1_flip_fraction']]+list(summary['margin_rms_fraction'].values())
 center=lambda x:x-x.mean(dim=1,keepdim=True);effect_ratio=float((center(logits['mask30'])-center(logits['full'])).square().mean().sqrt()/(center(logits['full'])-center(base_logits)).square().mean().sqrt().clamp_min(1e-30));mask_flips=set(reports['mask30']['flipped_row_ids']);full_flips=set(reports['full']['flipped_row_ids']);old=authority['control'];replay=max(abs(reports['mask30']['summary'][k]-old[k]) for k in ('median_kl','max_kl','top1_flip_fraction'));pa=observed==EXPECTED and authority['terminal']=='null' and replay<=1e-6 and all(math.isfinite(float(x)) for x in finite+[effect_ratio]) and forward_count<=MAX_FORWARDS;pb=mask_flips<=full_flips and reports['mask30']['summary']['median_kl']<=reports['full']['summary']['median_kl']+.002 and all(reports['mask30']['summary']['margin_rms_fraction'][t]<=reports['full']['summary']['margin_rms_fraction'][t]+.02 for t in ('temporal','iswas'));pc=len(mask_flips)>=2 and mask_flips<=full_flips;pd=(len(reports['top50']['flipped_row_ids'])<len(mask_flips) or reports['top50']['summary']['median_kl']<reports['mask30']['summary']['median_kl']) and authority['reports']['mask30']['behavior_signed_projection']['temporal']-authority['reports']['top50']['behavior_signed_projection']['temporal']>=.03;pe=effect_ratio<=.05
 preds={'pred_a_authority_replay_finiteness_and_price':bool(pa),'pred_b_mask30_adds_no_collateral_over_full':bool(pb),'pred_c_full_parent_explains_the_failed_absolute_gate':bool(pc),'pred_d_top50_exposes_the_function_selectivity_tradeoff':bool(pd),'pred_e_mask30_and_full_control_effects_are_close':bool(pe)};terminal='invalid' if not pa else 'parent_limited_selectivity' if all(preds.values()) else 'mask_specific_collateral' if not pb else 'null';result={'schema':'temporal_iswas_rank16_hidden_mask30_control_attribution_result_v1','started_utc':started,'finished_utc':now(),'serial_seconds':time.perf_counter()-tic,'authority_sha256':EXPECTED,'reports':reports,'mask30_full_centered_logit_rms_ratio':effect_ratio,'mask30_replay_max_abs_error':replay,'predictions':preds,'terminal':terminal,'price':{'model_forwards_observed':forward_count,'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ('reports','mask30_full_centered_logit_rms_ratio','mask30_replay_max_abs_error','predictions','terminal','price')},sort_keys=True))
if __name__=='__main__':main()
