#!/usr/bin/env python3
"""Fresh full-model test of activation-conditioned hidden groups in the five rank16 source MLPs."""
# BQGATE: EXPERIMENT pred_a_authority_exactness_finiteness_and_price pred_b_activation_top25_concentrates_local_signal pred_c_activation_top25_is_a_functional_selective_program pred_d_activation_conditioning_beats_static_weights pred_e_activation_top25_complement_is_insufficient
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import hidden_unit_subspace_contribution as hu
import pooled_response_projector as pooled
import run_temporal_five_mlp_family_feasible_kl_refinement_v1 as klfit
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlas
import run_temporal_iswas_five_mlp_position_svd_ladder_v1 as svd
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladder
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as edge
import run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1 as joint
import run_temporal_iswas_rank46_shared8_pairwise_mobius_v1 as pair
import run_temporal_iswas_rank46_task_typed_mode_causal_factorial_v1 as factorial
import run_temporal_iswas_five_mlp_source_dim_subspace_v1 as source
import run_temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v1 as compiler
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/'circuits/prior_art/temporal_iswas_rank16_activation_conditioned_hidden_groups_v1.json';AUDIT=ROOT/'circuits/followups/temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit_result.json';OUT=ROOT/'circuits/followups/temporal_iswas_rank16_activation_conditioned_hidden_groups_v1_result.json'
EXPECTED={'prior':'bf1e6ea2f6d8edfa8da2ea5731e2b13a0354fc727f9bbbf137ce7152c1560e89','audit':'0ef59ddb33f3d0534e5ecc283844189683eb73b9c2f8ba40f7a7af91f4a14d0b'}
SUPPORT=source.SUPPORT;TASKS=source.TASKS;GAIN=1.15;ARMS=('full','act10','act25','act50','weight25','act25_complement');MAX_FORWARDS=39
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def flatten(batch,x):return interface.valid_delta(batch,x.new_zeros(x.shape),x)
def run_selected(backend,batch,base_out,delta_hidden,bases,maps,masks):
 handles=[];response={'attention':{},'mlp':{},'mlp_input':{}}
 for site in SUPPORT:
  layer=int(site[3:]);base=base_out['mlp'][layer];dh=delta_hidden[site];q=bases[site]['rank16'];a=maps[site];mask=masks[site]
  def patch(_m,_args,out,base=base,dh=dh,q=q,a=a,mask=mask):
   changed=out.clone()
   for i,pos in enumerate(batch.semantic_positions):
    stop=int(pos)+1;projected=(dh[i,:stop,mask].float()@a[mask])@q.T
    changed[i,:stop]=base[i,:stop].to(changed)+(GAIN*projected).to(changed)
   return changed
  handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(patch))
 for site in edge.RESPONSE_SITES:
  kind,layer,_=atlas.site_parts(site)
  if kind=='attn':handles.append(backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(lambda _m,args,l=layer:response['attention'].__setitem__(l,args[0].detach().float().clone())))
  else:
   handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(lambda _m,args,l=layer:response['mlp_input'].__setitem__(l,args[0].detach().float().clone())))
   handles.append(backend.model.transformer.h[layer].mlp.register_forward_hook(lambda _m,_a,y,l=layer:response['mlp'].__setitem__(l,y.detach().float().clone())))
 try:output=backend.native(batch,capture=True)
 finally:
  for h in handles:h.remove()
 return output,response
def main():
 dry={'candidate_id':'temporal_auxiliary.iswas_rank16_activation_conditioned_hidden_groups_v1','dryrun':True,'gpu_accessed':False,'model_loaded':False,'queue_touched':False,'arms':list(ARMS),'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}
 if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 expected=dict(EXPECTED);observed={'prior':sha(PRIOR),'audit':sha(AUDIT)};started=now();tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;native=backend.native;forward_count=0
 def counted_native(*args,**kwargs):
  nonlocal forward_count;forward_count+=1;return native(*args,**kwargs)
 backend.native=counted_native
 fitted=pooled.fit(backend);qs=fitted['projector'];_,full_modes=ladder.fit_full_modes(backend,fitted);modes={s:{t:full_modes[s][t][:,:4] for t in TASKS} for s in edge.RESPONSE_SITES};bases,_,_,_=svd.fit_position_bases(backend,fitted['target_rows'])
 fb,*_=interface.cap_inputs(backend,fitted['target_rows']);fh0,fo0=compiler.capture(backend,fb);fdb=das._batch(backend,fitted['target_rows'],side='donor');fh1,fo1=compiler.capture(backend,fdb)
 fresh=oodctx.capture(backend,factorial.TCAP,factorial.ICAP);bh0,bo0=compiler.capture(backend,fresh['batch']);bh1,bo1=compiler.capture(backend,fresh['donor_batch']);ch0,co0=compiler.capture(backend,fresh['control_batch']);ch1,co1=compiler.capture(backend,fresh['control_donor_batch'])
 maps={};masks={k:{} for k in ARMS};local={};closure=[]
 for site in SUPPORT:
  q=bases[site]['rank16'];down=backend.model.transformer.h[int(site[3:])].mlp.Down.weight.detach().float();a=down.T@q;maps[site]=a;fitdh=flatten(fb,fh1[site]-fh0[site]);act=hu.contribution_energy(fitdh,a);weight=a.square().sum(1)
  masks['full'][site]=torch.ones(len(act),dtype=torch.bool,device=backend.device);masks['act10'][site]=hu.top_fraction_mask(act,.10);masks['act25'][site]=hu.top_fraction_mask(act,.25);masks['act50'][site]=hu.top_fraction_mask(act,.50);masks['weight25'][site]=hu.top_fraction_mask(weight,.25);masks['act25_complement'][site]=~masks['act25'][site]
  freshdh=flatten(fresh['batch'],bh1[site]-bh0[site]);all_delta=hu.compiled_delta(freshdh,a,q);sel=hu.compiled_delta(freshdh,a,q,masks['act25'][site]);comp=hu.compiled_delta(freshdh,a,q,masks['act25_complement'][site]);cl=float((sel+comp-all_delta).square().sum()/all_delta.square().sum().clamp_min(1e-30));closure.append(cl);local[site]={'act25_fresh_energy_fraction':float(sel.square().sum()/all_delta.square().sum().clamp_min(1e-30)),'selected_units':int(masks['act25'][site].sum()),'coefficient_closure_rse':cl}
 base_output,base=edge.capture_with_source_patch(backend,fresh['batch'],fresh['base_full'],());donor_output,donor=edge.capture_with_source_patch(backend,fresh['donor_batch'],fresh['base_full'],())
 base_state=atlas.states(torch,backend,base_output,fresh['rows']);donor_state=atlas.states(torch,backend,donor_output,fresh['rows']);target=pair.response_vectors(backend,fresh,base,donor,qs,modes);control_ctx={'rows':fresh['controls'],'base_logits':das.head_logits(backend,atlas.states(torch,backend,fresh['control_base'][0],fresh['controls'])).float()}
 reports={};finite=[]
 for arm in ARMS:
  out,changed=run_selected(backend,fresh['batch'],fresh['base_full'],{s:bh1[s]-bh0[s] for s in SUPPORT},bases,maps,masks[arm]);cout,_=run_selected(backend,fresh['control_batch'],fresh['control_base_full'],{s:ch1[s]-ch0[s] for s in SUPPORT},bases,maps,masks[arm]);vectors=pair.response_vectors(backend,fresh,base,changed,qs,modes);response={t:pair.vector_metrics(torch,vectors[t],target[t]) for t in TASKS};behavior=joint.behavior_report(backend,fresh['rows'],base_state,donor_state,atlas.states(torch,backend,out,fresh['rows']));control=klfit.control_metrics(backend,control_ctx,das.head_logits(backend,atlas.states(torch,backend,cout,fresh['controls'])).float());reports[arm]={'response':response,'behavior_signed_projection':behavior,'control':control};finite += list(behavior.values())+[x for t in TASKS for x in response[t].values()]+[control['median_kl'],control['max_kl'],control['top1_flip_fraction']]
 full_ref=json.loads((ROOT/'circuits/followups/temporal_iswas_five_mlp_rank16_gain_curve_v1_result.json').read_text())['reports']['1.15'];replay=max(max(abs(reports['full']['response'][t][m]-full_ref['response'][t][m]) for t in TASKS for m in ('signed_projection','relative_squared_error','norm_ratio')),max(abs(reports['full']['behavior_signed_projection'][t]-full_ref['behavior_signed_projection'][t]) for t in TASKS),abs(reports['full']['control']['median_kl']-full_ref['control']['median_kl']),abs(reports['full']['control']['top1_flip_fraction']-full_ref['control']['top1_flip_fraction']))
 pa=observed==expected and json.loads(AUDIT.read_text())['terminal']=='distributed_weight_compiled_source_program' and replay<=1e-5 and max(closure)<=1e-6 and all(math.isfinite(float(x)) for x in finite) and forward_count<=MAX_FORWARDS;pb=sum(x['act25_fresh_energy_fraction']>=.8 for x in local.values())>=4;pc=source.functional(reports['act25']) and reports['act25']['control']['median_kl']<=.02 and reports['act25']['control']['top1_flip_fraction']==0;pd=all(reports['act25']['response'][t]['relative_squared_error']<reports['weight25']['response'][t]['relative_squared_error'] for t in TASKS);pe=all(abs(reports['act25_complement']['response'][t]['signed_projection'])<=.5 and abs(reports['act25_complement']['behavior_signed_projection'][t])<=.5 for t in TASKS)
 preds={'pred_a_authority_exactness_finiteness_and_price':bool(pa),'pred_b_activation_top25_concentrates_local_signal':bool(pb),'pred_c_activation_top25_is_a_functional_selective_program':bool(pc),'pred_d_activation_conditioning_beats_static_weights':bool(pd),'pred_e_activation_top25_complement_is_insufficient':bool(pe)};terminal='invalid' if not pa else 'activation_conditioned_hidden_circuit' if all(preds.values()) else 'distributed_activation_program' if source.functional(reports['act50']) else 'null';result={'schema':'temporal_iswas_rank16_activation_conditioned_hidden_groups_result_v1','started_utc':started,'finished_utc':now(),'serial_seconds':time.perf_counter()-tic,'authority_sha256':expected,'local':local,'full_replay_max_abs_error':replay,'reports':reports,'predictions':preds,'terminal':terminal,'price':{'model_forwards_observed':forward_count,'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ('local','full_replay_max_abs_error','reports','predictions','terminal','price')},sort_keys=True))
if __name__=='__main__':main()
