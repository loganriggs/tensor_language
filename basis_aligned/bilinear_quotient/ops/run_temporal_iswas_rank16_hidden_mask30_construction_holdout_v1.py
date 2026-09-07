#!/usr/bin/env python3
"""No-reselection construction-family confirmation of hidden mask30."""
# BQGATE: EXPERIMENT pred_a_authority_population_replays_finiteness_and_price pred_b_frozen_mask30_is_functional pred_c_mlp0_bottom_half_remains_dispensable pred_d_mask30_beats_all_top50 pred_e_mask30_is_selective
from datetime import datetime,timezone
import hashlib,json,math,os,time
from pathlib import Path
import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as old_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as old_i
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
import run_temporal_iswas_rank46_shared8_pairwise_mobius_v1 as pair
import run_temporal_iswas_rank46_task_mode_complete_rank_ladder_v1 as ladder
import run_temporal_iswas_rank46_task_rank4_joint_source_factorial_v1 as joint
import run_temporal_iswas_rank46_task_rank4_upstream_weight_edge_atlas_v1 as edge
import run_temporal_iswas_five_mlp_source_dim_subspace_v1 as source
ROOT=Path(__file__).resolve().parents[1];PRIOR=ROOT/'circuits/prior_art/temporal_iswas_rank16_hidden_mask30_construction_holdout_v1.json';FACTOR=ROOT/'circuits/followups/temporal_iswas_rank16_hidden_complement_factorial_v1_result.json';TCAP=greedy.TEMPORAL_CAPABILITY;ICAP=greedy.ISWAS_CAPABILITY;OUT=ROOT/'circuits/followups/temporal_iswas_rank16_hidden_mask30_construction_holdout_v1_result.json'
EXPECTED={'prior':'7c3448f61c63fc9282c68363d489d9ef94492aa06e3601a20357621cc9595a21','factorial':'b146297e891d3dc0f19b1bf8453690132849cb38bf10d4a1c4402e8b1aa2684d','temporal_builder':'3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509','temporal_capability':'e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1','iswas_builder':'2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2','iswas_capability':'67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4'};MAX_FORWARDS=25;SUPPORT=tuple(source.SUPPORT);TASKS=source.TASKS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def main():
 dry={'candidate_id':'temporal_auxiliary.iswas_rank16_hidden_mask30_construction_holdout_v1','dryrun':True,'gpu_accessed':False,'model_loaded':False,'queue_touched':False,'arms':['top50','mask30','full'],'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}
 if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 paths={'prior':PRIOR,'factorial':FACTOR,'temporal_builder':greedy.TEMPORAL_BUILDER,'temporal_capability':TCAP,'iswas_builder':greedy.ISWAS_BUILDER,'iswas_capability':ICAP};observed={k:sha(v) for k,v in paths.items()};started=now();tic=time.perf_counter();backend=producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;native=backend.native;forward_count=0
 def counted(*args,**kwargs):
  nonlocal forward_count;forward_count+=1;return native(*args,**kwargs)
 backend.native=counted
 fitted=pooled.fit(backend);qs=fitted['projector'];_,full_modes=ladder.fit_full_modes(backend,fitted);modes={s:{t:full_modes[s][t][:,:4] for t in TASKS} for s in edge.RESPONSE_SITES};bases,_,_,_=svd.fit_position_bases(backend,fitted['target_rows']);fb,*_=interface.cap_inputs(backend,fitted['target_rows']);fh0,_=compiler.capture(backend,fb);fh1,_=compiler.capture(backend,das._batch(backend,fitted['target_rows'],side='donor'))
 tcap,icap=json.loads(TCAP.read_text()),json.loads(ICAP.read_text());tr,ir,tc,ic=greedy.rows_and_controls(tcap,icap);rows=tr+ir;controls=tc+ic;batch=das._batch(backend,rows,side='base');donor_batch=das._batch(backend,rows,side='donor');cbatch=das._batch(backend,controls,side='base');cdbatch=das._batch(backend,controls,side='donor')
 bh0,_=compiler.capture(backend,batch);bh1,_=compiler.capture(backend,donor_batch);ch0,_=compiler.capture(backend,cbatch);ch1,_=compiler.capture(backend,cdbatch);base_output,base_full=atlas.capture_native(backend,batch);donor_output,donor_full=atlas.capture_native(backend,donor_batch);control_output,control_base_full=atlas.capture_native(backend,cbatch)
 maps={};top={};full={}
 for site in SUPPORT:
  q=bases[site]['rank16'];a=backend.model.transformer.h[int(site[3:])].mlp.Down.weight.detach().float().T@q;maps[site]=a;score=hu.contribution_energy(parent.flatten(fb,fh1[site]-fh0[site]),a);top[site]=hu.top_fraction_mask(score,.5);full[site]=torch.ones_like(top[site])
 mask30={s:(top[s] if s=='MLP0' else full[s]) for s in SUPPORT};fresh={'rows':rows,'batch':batch};base_state=atlas.states(torch,backend,base_output,rows);donor_state=atlas.states(torch,backend,donor_output,rows);target=pair.response_vectors(backend,fresh,base_full,donor_full,qs,modes);reports={};finite=[]
 for name,masks in (('top50',top),('mask30',mask30),('full',full)):
  out,changed=parent.run_selected(backend,batch,base_full,{s:bh1[s]-bh0[s] for s in SUPPORT},bases,maps,masks);vectors=pair.response_vectors(backend,fresh,base_full,changed,qs,modes);response={t:pair.vector_metrics(torch,vectors[t],target[t]) for t in TASKS};behavior=joint.behavior_report(backend,rows,base_state,donor_state,atlas.states(torch,backend,out,rows));reports[name]={'response':response,'behavior_signed_projection':behavior};finite+=list(behavior.values())+[x for t in TASKS for x in response[t].values()]
 cout,_=parent.run_selected(backend,cbatch,control_base_full,{s:ch1[s]-ch0[s] for s in SUPPORT},bases,maps,mask30);control_ctx={'rows':controls,'base_logits':das.head_logits(backend,atlas.states(torch,backend,control_output,controls)).float()};control=klfit.control_metrics(backend,control_ctx,das.head_logits(backend,atlas.states(torch,backend,cout,controls)).float());finite += [control['median_kl'],control['max_kl'],control['top1_flip_fraction']]
 new_ids={r['row_id'] for r in rows};old_ids={r['row_id'] for r in old_t.build_rows()}|{r['row_id'] for r in old_i.build_rows()};pa=observed==EXPECTED and json.loads(FACTOR.read_text())['terminal']=='distributed_five_site_program' and bool(tr) and bool(ir) and not(new_ids&old_ids) and all(math.isfinite(float(x)) for x in finite) and forward_count<=MAX_FORWARDS;pb=source.functional(reports['mask30']);pc=all(abs(reports['mask30']['behavior_signed_projection'][t]-reports['full']['behavior_signed_projection'][t])<=.03 and abs(reports['mask30']['response'][t]['signed_projection']-reports['full']['response'][t]['signed_projection'])<=.03 for t in TASKS);pd=reports['mask30']['behavior_signed_projection']['temporal']-reports['top50']['behavior_signed_projection']['temporal']>=.03;pe=control['median_kl']<=.02 and control['top1_flip_fraction']==0
 preds={'pred_a_authority_population_replays_finiteness_and_price':bool(pa),'pred_b_frozen_mask30_is_functional':bool(pb),'pred_c_mlp0_bottom_half_remains_dispensable':bool(pc),'pred_d_mask30_beats_all_top50':bool(pd),'pred_e_mask30_is_selective':bool(pe)};terminal='invalid' if not pa else 'construction_holdout_screen' if all(preds.values()) else 'population_specific_hidden_mask' if not(pb and pc) else 'null';result={'schema':'temporal_iswas_rank16_hidden_mask30_construction_holdout_result_v1','started_utc':started,'finished_utc':now(),'serial_seconds':time.perf_counter()-tic,'authority_sha256':EXPECTED,'row_counts':{'temporal':len(tr),'iswas':len(ir),'controls':len(controls)},'reports':reports,'control':control,'predictions':preds,'terminal':terminal,'price':{'model_forwards_observed':forward_count,'model_forwards_max':MAX_FORWARDS,'fit_updates':0,'model_updates':0,'transformer_backwards':0}};atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ('row_counts','reports','control','predictions','terminal','price')},sort_keys=True))
if __name__=='__main__':main()
