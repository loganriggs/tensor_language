#!/usr/bin/env python3
"""Complete five-site factorial over top50 versus full hidden support."""
# BQGATE: EXPERIMENT pred_a_authority_replays_exactness_finiteness_and_price pred_b_a_specific_site_complement_is_material pred_c_a_parsimonious_functional_mask_exists pred_d_selected_mask_is_response_and_behavior_functional pred_e_selected_mask_is_selective
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
import run_temporal_iswas_rank16_activation_conditioned_hidden_groups_v1 as parent
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/'circuits/prior_art/temporal_iswas_rank16_hidden_complement_factorial_v1.json';PARENT=ROOT/'circuits/followups/temporal_iswas_rank16_activation_conditioned_hidden_groups_v1_result.json';OUT=ROOT/'circuits/followups/temporal_iswas_rank16_hidden_complement_factorial_v1_result.json'
EXPECTED={'prior':'56a66acd40875106769fca2695562fbfd60225c1828c905a325b7bb2970812ea','parent':'89869da0e01f983f5c6f6c64a5bcc65579d16df8a224df8d84a8490b5d9400fb'};SUPPORT=tuple(source.SUPPORT);TASKS=source.TASKS;MAX_FORWARDS=60
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def mobius(values):
 c=list(values)
 for bit in range(5):
  for mask in range(32):
   if mask&(1<<bit):c[mask]-=c[mask^(1<<bit)]
 by={d:sum(c[m]*c[m] for m in range(1,32) if m.bit_count()==d) for d in range(1,6)};total=sum(by.values())
 return {'degree_energy':{str(k):v for k,v in by.items()},'higher_order_fraction':sum(v for k,v in by.items() if k>=2)/max(total,1e-30),'closure_abs':abs(sum(c)-values[-1])}
def main():
 dry={'candidate_id':'temporal_auxiliary.iswas_rank16_hidden_complement_factorial_v1','dryrun':True,'gpu_accessed':False,'model_loaded':False,'queue_touched':False,'arms':32,'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}
 if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 observed={'prior':sha(PRIOR),'parent':sha(PARENT)};started=now();tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;native=backend.native;forward_count=0
 def counted(*args,**kwargs):
  nonlocal forward_count;forward_count+=1;return native(*args,**kwargs)
 backend.native=counted
 fitted=pooled.fit(backend);qs=fitted['projector'];_,full_modes=ladder.fit_full_modes(backend,fitted);modes={s:{t:full_modes[s][t][:,:4] for t in TASKS} for s in edge.RESPONSE_SITES};bases,_,_,_=svd.fit_position_bases(backend,fitted['target_rows']);fb,*_=interface.cap_inputs(backend,fitted['target_rows']);fh0,_=compiler.capture(backend,fb);fh1,_=compiler.capture(backend,das._batch(backend,fitted['target_rows'],side='donor'))
 fresh=oodctx.capture(backend,factorial.TCAP,factorial.ICAP);bh0,_=compiler.capture(backend,fresh['batch']);bh1,_=compiler.capture(backend,fresh['donor_batch']);maps={};top={}
 for site in SUPPORT:
  q=bases[site]['rank16'];a=backend.model.transformer.h[int(site[3:])].mlp.Down.weight.detach().float().T@q;maps[site]=a;score=hu.contribution_energy(parent.flatten(fb,fh1[site]-fh0[site]),a);top[site]=hu.top_fraction_mask(score,.5)
 arms=hu.complement_factorial(top,SUPPORT);base_output,base=edge.capture_with_source_patch(backend,fresh['batch'],fresh['base_full'],());donor_output,donor=edge.capture_with_source_patch(backend,fresh['donor_batch'],fresh['base_full'],());base_state=atlas.states(torch,backend,base_output,fresh['rows']);donor_state=atlas.states(torch,backend,donor_output,fresh['rows']);target=pair.response_vectors(backend,fresh,base,donor,qs,modes);reports={};finite=[]
 for bits,masks in arms:
  out,changed=parent.run_selected(backend,fresh['batch'],fresh['base_full'],{s:bh1[s]-bh0[s] for s in SUPPORT},bases,maps,masks);vectors=pair.response_vectors(backend,fresh,base,changed,qs,modes);response={t:pair.vector_metrics(torch,vectors[t],target[t]) for t in TASKS};behavior=joint.behavior_report(backend,fresh['rows'],base_state,donor_state,atlas.states(torch,backend,out,fresh['rows']));functional=source.functional({'response':response,'behavior_signed_projection':behavior});reports[str(bits)]={'full_sites':[SUPPORT[i] for i in range(5) if bits&(1<<i)],'response':response,'behavior_signed_projection':behavior,'functional':functional};finite += list(behavior.values())+[v for t in TASKS for v in response[t].values()]
 eligible=[m for m in range(32) if reports[str(m)]['functional']];selected=min(eligible,key=lambda m:(m.bit_count(),max(reports[str(m)]['response'][t]['relative_squared_error'] for t in TASKS),m)) if eligible else None
 ch0,_=compiler.capture(backend,fresh['control_batch']);ch1,_=compiler.capture(backend,fresh['control_donor_batch']);cout,_=parent.run_selected(backend,fresh['control_batch'],fresh['control_base_full'],{s:ch1[s]-ch0[s] for s in SUPPORT},bases,maps,dict(arms[selected][1]) if selected is not None else top);control_ctx={'rows':fresh['controls'],'base_logits':das.head_logits(backend,fresh['control_base_state']).float()};control=klfit.control_metrics(backend,control_ctx,das.head_logits(backend,atlas.states(torch,backend,cout,fresh['controls'])).float());finite += [control['median_kl'],control['max_kl'],control['top1_flip_fraction']]
 old=json.loads(PARENT.read_text())['reports'];replay=max(max(abs(reports['0']['response'][t][k]-old['act50']['response'][t][k]) for t in TASKS for k in ('signed_projection','relative_squared_error','norm_ratio')),max(abs(reports['31']['response'][t][k]-old['full']['response'][t][k]) for t in TASKS for k in ('signed_projection','relative_squared_error','norm_ratio')),max(abs(reports['0']['behavior_signed_projection'][t]-old['act50']['behavior_signed_projection'][t]) for t in TASKS),max(abs(reports['31']['behavior_signed_projection'][t]-old['full']['behavior_signed_projection'][t]) for t in TASKS))
 increments={SUPPORT[i]:reports[str(1<<i)]['behavior_signed_projection']['temporal']-reports['0']['behavior_signed_projection']['temporal'] for i in range(5)};mb={t:mobius([reports[str(m)]['behavior_signed_projection'][t] for m in range(32)]) for t in TASKS};mr={t:mobius([reports[str(m)]['response'][t]['signed_projection'] for m in range(32)]) for t in TASKS};higher=max(x['higher_order_fraction'] for x in list(mb.values())+list(mr.values()))
 pa=observed==EXPECTED and json.loads(PARENT.read_text())['terminal']=='null' and len({tuple(v['full_sites']) for v in reports.values()})==32 and replay<=1e-5 and forward_count<=MAX_FORWARDS and all(math.isfinite(float(x)) for x in finite) and max(x['closure_abs'] for x in list(mb.values())+list(mr.values()))<=1e-10;pb=max(increments.values())>=.03;pc=selected is not None and selected.bit_count()<=3;pd=selected is not None and reports[str(selected)]['functional'];pe=control['median_kl']<=.02 and control['top1_flip_fraction']==0
 preds={'pred_a_authority_replays_exactness_finiteness_and_price':bool(pa),'pred_b_a_specific_site_complement_is_material':bool(pb),'pred_c_a_parsimonious_functional_mask_exists':bool(pc),'pred_d_selected_mask_is_response_and_behavior_functional':bool(pd),'pred_e_selected_mask_is_selective':bool(pe)};terminal='invalid' if not pa else 'localized_hidden_complement_program' if all(preds.values()) else 'distributed_five_site_program' if pa and pd and pe and not pc else 'conditional_hidden_support' if pa and pb and pd and pe and higher>=.25 else 'null';result={'schema':'temporal_iswas_rank16_hidden_complement_factorial_result_v1','started_utc':started,'finished_utc':now(),'serial_seconds':time.perf_counter()-tic,'authority_sha256':EXPECTED,'reports':reports,'selected_mask':selected,'selected_full_sites':None if selected is None else reports[str(selected)]['full_sites'],'singleton_temporal_behavior_increments':increments,'mobius':{'behavior':mb,'response':mr,'max_higher_order_fraction':higher},'full_replay_max_abs_error':replay,'selected_control':control,'predictions':preds,'terminal':terminal,'price':{'model_forwards_observed':forward_count,'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ('selected_mask','selected_full_sites','singleton_temporal_behavior_increments','mobius','full_replay_max_abs_error','selected_control','predictions','terminal','price')},sort_keys=True))
if __name__=='__main__':main()
