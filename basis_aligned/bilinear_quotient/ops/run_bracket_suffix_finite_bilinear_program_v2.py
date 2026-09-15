#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_finite_suffix_instrument pred_b_sixth_program_selected pred_c_seventh_program_transfers pred_d_bilinear_interaction_compression pred_e_control_selectivity
"""Select and confirm an exact finite bilinear bracket suffix program."""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,math,os,signal,sys,time
from pathlib import Path

RUNNER=Path(__file__).resolve();OPS=RUNNER.parent;ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
SIX_ROWS=POLY/'BRACKET_EMBEDDED_PENDING_OOD_V1_ROWS.json';SEVEN_ROWS=POLY/'BRACKET_CASCADE_PENDING_OOD_V1_ROWS.json';SIX_RESULT=POLY/'BRACKET_EMBEDDED_PENDING_COMPACT_BEHAVIOR_V1_RESULT.json';SEVEN_RESULT=POLY/'BRACKET_CASCADE_PENDING_LIVE_AMPLITUDE_CONFIRMATION_V1_RESULT.json';V1_RESULT=POLY/'BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V1_RESULT.json';PREREG=POLY/'BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_PREREGISTRATION.md';BINDING=POLY/'BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_BINDING.json';OUT=POLY/'BRACKET_SUFFIX_FINITE_BILINEAR_PROGRAM_V2_RESULT.json'
SITES=(13,14,15,16,17);CANDIDATES=('source_only','quadratic','left','right','cross');PRICE={'maximum_forwards':12,'maximum_sequences':1728,'null_forwards':8,'null_sequences':1152,'fits':0,'backwards':0,'updates':0}
BARS={'replay_max':1e-5,'closure_rse_max':1e-10,'exact_positive_min':.90,'overall_cosine_min':.95,'overall_relative_l2_max':.25,'overall_sign_min':.90,'overall_norm_min':.75,'overall_norm_max':1.25,'pair_cosine_min':.85,'pair_relative_l2_max':.35,'pair_sign_min':.85,'control_to_target_rms_max':.50,'active_fraction_min':.90}
PREDICTION_REGISTRY={'pred_a_exact_finite_suffix_instrument':None,'pred_b_sixth_program_selected':None,'pred_c_seventh_program_transfers':None,'pred_d_bilinear_interaction_compression':None,'pred_e_control_selectivity':None}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_bound():
 b=json.loads(BINDING.read_text());paths={'six_rows':SIX_ROWS,'seven_rows':SEVEN_ROWS,'six_result':SIX_RESULT,'seven_result':SEVEN_RESULT,'v1_result':V1_RESULT,'preregistration':PREREG};assert all(digest(paths[k])==v for k,v in b['files'].items());assert b['sites']==list(SITES) and b['candidates']==list(CANDIDATES) and b['bars']==BARS and b['price']==PRICE
 six=json.loads(SIX_ROWS.read_text());seven=json.loads(SEVEN_ROWS.read_text());sr=json.loads(SIX_RESULT.read_text());tr=json.loads(SEVEN_RESULT.read_text());assert sr['predictions']['pred_b_exact_joint_parent_live'] and tr['predictions']['pred_b_exact_joint_parent_live'];v1=json.loads(V1_RESULT.read_text());assert v1['terminal']=='invalid' and v1['selected']=='source_only';assert all(v==1.0 for k,v in v1['sixth']['active_fraction'].items() if k!='exact') and v1['sixth']['active_fraction']['exact']==0.0;return {'sixth':six,'seventh':seven}
def plan():
 panels=load_bound();return {'schema':'bracket_suffix_finite_bilinear_program_v2_plan','model_loaded':False,'gpu_accessed':False,'queue_touched':False,'panels':{k:v['endpoint_count'] for k,v in panels.items()},'sites':list(SITES),'candidate_order':list(CANDIDATES),'bars':BARS,'price':PRICE,'predicates':list(PREDICTION_REGISTRY)}
def metrics(actual,predicted):
 actual=[float(x) for x in actual];predicted=[float(x) for x in predicted];an=math.sqrt(sum(x*x for x in actual));pn=math.sqrt(sum(x*x for x in predicted));return {'count':len(actual),'cosine':sum(x*y for x,y in zip(actual,predicted))/max(an*pn,1e-30),'relative_l2_error':math.sqrt(sum((x-y)**2 for x,y in zip(actual,predicted)))/max(an,1e-30),'sign_agreement':sum((x>0)==(y>0) for x,y in zip(actual,predicted))/len(actual),'predicted_to_actual_norm_ratio':pn/max(an,1e-30)}
def metric_pass(v,pair=False):return v['cosine']>=BARS['pair_cosine_min' if pair else 'overall_cosine_min'] and v['relative_l2_error']<=BARS['pair_relative_l2_max' if pair else 'overall_relative_l2_max'] and v['sign_agreement']>=BARS['pair_sign_min' if pair else 'overall_sign_min'] and (pair or BARS['overall_norm_min']<=v['predicted_to_actual_norm_ratio']<=BARS['overall_norm_max'])
def main():
 panels=load_bound();sys.path.insert(0,str(OPS));from circuit_exactness_preflight import managed_execution_mode,validate_literal_prediction_registry,validate_result_contract;validate_literal_prediction_registry(RUNNER.read_text(),PREDICTION_REGISTRY)
 if managed_execution_mode(os.environ)=='preflight':print(json.dumps(plan(),sort_keys=True));return
 assert not OUT.exists();signal.alarm(600);import torch;import run_bracket_l13h8_source_region_payload_factorial as exact;from circuit_fast_screen_managed_runner import atomic_create_json
 torch.set_num_threads(2);tm,F,facade=exact._dependencies();model,checkpoint=facade.load_bilin18(device='cuda',dtype=torch.float32,verify_weights_sha256=True);device=next(model.parameters()).device;counts=[0,0]
 def count(_m,args,_o):counts[0]+=1;counts[1]+=len(args[0])
 handle=model.transformer.h[0].attn.register_forward_hook(count);tic=time.perf_counter()
 def execute_panel(name,candidates):
  frozen=panels[name];rows=frozen['rows'];endpoints=[(row,side) for row in rows for side in ('base','donor')];length=max(len(r[f'{s}_ids']) for r,s in endpoints);tokens=torch.full((len(endpoints),length),50256,dtype=torch.long,device=device);finals=[];sources=[]
  for i,(row,side) in enumerate(endpoints):ids=row[f'{side}_ids'];tokens[i,:len(ids)]=torch.tensor(ids,device=device);finals.append(len(ids)-1);sources.append(row[f'{side}_open_position'])
  finals_t=torch.tensor(finals,device=device);sources_t=torch.tensor(sources,device=device);ar=torch.arange(len(endpoints),device=device)
  with torch.inference_mode():independent=exact.native_logits(model,tokens,tm,F).cpu()
  captured={};states=torch.zeros((len(endpoints),len(SITES),1152),dtype=torch.float32,device='cpu')
  def cap_attention(event):
   if event.site!=13:return event.block.attn(event.state,event.first_value)
   write,factors=exact.replay_head(event.state,event.first_value,event.block.attn,finals_t,torch,F);captured.update({k:v.detach() for k,v in factors.items()});return write,event.first_value
  def cap_mlp(event):
   write=event.block.mlp(event.state)
   if event.site in SITES:
    for i,q in enumerate(finals):states[i,SITES.index(event.site)]=event.state[i,q].float().cpu()
   return write
  with torch.inference_mode():replay_gpu=facade.forward_with_dispatch(model,tokens,cap_attention,cap_mlp,require_production=False).float();replay=replay_gpu.cpu();del replay_gpu
  replay_error=max(float((independent[i,finals[i]]-replay[i,finals[i]]).abs().max()) for i in range(len(endpoints)));source_p=captured['p'][ar,sources_t];source_u=captured['u'][ar,sources_t];donor=ar^1;replacement=source_p[donor,None]*source_u[donor];outputs={};closures=[];active={}
  def run_arm(arm):
   removed=torch.zeros(len(endpoints),dtype=torch.float64,device='cpu')
   def attention(event):
    if event.site!=13:return event.block.attn(event.state,event.first_value)
    write,factors=exact.replay_head(event.state,event.first_value,event.block.attn,finals_t,torch,F);native=factors['p'][ar,sources_t,None]*factors['u'][ar,sources_t];write=write.clone();write[ar,finals_t]+=(replacement-native).to(write.dtype);return write,event.first_value
   def mlp_fn(event):
    write=event.block.mlp(event.state)
    if event.site in SITES and arm!='exact':
     changed=write.clone();mlp=event.block.mlp;si=SITES.index(event.site)
     for i,q in enumerate(finals):
      x1=event.state[i,q].float();x0=states[i,si].to(device);d=x1-x0;lw=mlp.Left.weight.float();rw=mlp.Right.weight.float();dw=mlp.Down.weight.float();l0=F.linear(x0,lw);r0=F.linear(x0,rw);ld=F.linear(d,lw);rd=F.linear(d,rw);left=F.linear(ld*r0,dw);right=F.linear(l0*rd,dw);quad=F.linear(ld*rd,dw);joint=left+right+quad;retain={'source_only':torch.zeros_like(joint),'quadratic':quad,'left':left,'right':right,'cross':left+right}[arm];complement=joint-retain;changed[i,q]=(changed[i,q].float()-complement).to(changed.dtype);removed[i]+=float(complement.double().norm().cpu())**2;x1d=x1.double();x0d=x0.double();dd=x1d-x0d;lwd=lw.double();rwd=rw.double();dwd=dw.double();l0d=F.linear(x0d,lwd);r0d=F.linear(x0d,rwd);ldd=F.linear(dd,lwd);rdd=F.linear(dd,rwd);j=F.linear(ldd*r0d+l0d*rdd+ldd*rdd,dwd);direct=F.linear(F.linear(x1d,lwd)*F.linear(x1d,rwd)-l0d*r0d,dwd);closures.append(float((j-direct).square().sum()/direct.square().sum().clamp_min(1e-30)))
     write=changed
    return write
   with torch.inference_mode():out=facade.forward_with_dispatch(model,tokens,attention,mlp_fn,require_production=False).float().cpu()
   active[arm]=float((removed.sqrt()>1e-6).float().mean());return out
  for arm in ('exact',*candidates):outputs[arm]=run_arm(arm);torch.cuda.empty_cache()
  records=[]
  for i,(row,side) in enumerate(endpoints):
   answer=int(row[f'{side}_answer_id']);other='donor' if side=='base' else 'base';other_answer=int(row[f'{other}_answer_id']);rec={'row_id':row['row_id'],'side':side,'program_role':row['program_role'],'ordered_pair':f'{answer}->{other_answer}'};before=replay[i,finals[i]]
   if row['program_role']=='target':
    direction='base_to_donor' if side=='base' else 'donor_to_base'
    for arm in outputs:rec[arm+'_effect']=float(exact.endpoint_change(before,outputs[arm][i,finals[i]],row,direction))
   else:
    base=float(exact.closer_margin(before,answer))
    for arm in outputs:rec[arm+'_control_change']=float(exact.closer_margin(outputs[arm][i,finals[i]],answer)-base)
   records.append(rec)
  targets=[r for r in records if r['program_role']=='target'];controls=[r for r in records if r['program_role']=='control'];target_rms=math.sqrt(sum(r['exact_effect']**2 for r in targets)/len(targets));pairs=sorted({r['ordered_pair'] for r in targets});reports={};exact_positive={pair:sum(r['exact_effect']>0 for r in targets if r['ordered_pair']==pair)/sum(r['ordered_pair']==pair for r in targets) for pair in pairs}
  for arm in candidates:
   overall=metrics([r['exact_effect'] for r in targets],[r[arm+'_effect'] for r in targets]);by_pair={pair:metrics([r['exact_effect'] for r in targets if r['ordered_pair']==pair],[r[arm+'_effect'] for r in targets if r['ordered_pair']==pair]) for pair in pairs};control_rms=math.sqrt(sum(r[arm+'_control_change']**2 for r in controls)/len(controls));reports[arm]={'overall':overall,'by_pair':by_pair,'control_to_exact_target_rms':control_rms/max(target_rms,1e-30),'passes':bool(metric_pass(overall) and all(metric_pass(v,True) for v in by_pair.values()) and control_rms/max(target_rms,1e-30)<=BARS['control_to_target_rms_max'])}
  instrument=replay_error<=BARS['replay_max'] and max(closures,default=0)<=BARS['closure_rse_max'] and all(v>=BARS['active_fraction_min'] for arm,v in active.items() if arm!='exact') and len(targets)==len(controls)==72;parent=all(v>=BARS['exact_positive_min'] for v in exact_positive.values());return {'instrument':instrument,'parent_live':parent,'replay_error':replay_error,'maximum_closure_rse':max(closures,default=0),'active_fraction':active,'exact_positive_fraction':exact_positive,'reports':reports,'target_exact_rms':target_rms,'records':records}
 try:
  sixth=execute_panel('sixth',CANDIDATES);selected=next((a for a in CANDIDATES if sixth['parent_live'] and sixth['reports'][a]['passes']),None);seventh=execute_panel('seventh',(selected,)) if selected else None
 finally:handle.remove()
 expected=[12,1728] if selected else [8,1152];instrument=counts==expected and sixth['instrument'] and (seventh is None or seventh['instrument']);transfer=bool(selected and seventh['parent_live'] and seventh['reports'][selected]['passes']);selective=bool(selected and sixth['reports'][selected]['control_to_exact_target_rms']<=BARS['control_to_target_rms_max'] and seventh['reports'][selected]['control_to_exact_target_rms']<=BARS['control_to_target_rms_max']);pred={'pred_a_exact_finite_suffix_instrument':bool(instrument),'pred_b_sixth_program_selected':bool(instrument and selected),'pred_c_seventh_program_transfers':bool(instrument and transfer),'pred_d_bilinear_interaction_compression':bool(instrument and transfer and selected!='source_only'),'pred_e_control_selectivity':bool(instrument and selective)};terminal='invalid' if not instrument else 'suffix_source_only' if transfer and selected=='source_only' else 'suffix_finite_bilinear_program' if transfer else 'suffix_finite_bilinear_fit_null' if not selected else 'suffix_finite_bilinear_transfer_null';result={'schema':'bracket_suffix_finite_bilinear_program_v2_result','terminal':terminal,'predictions':pred,'selected':selected,'sixth':sixth,'seventh':seventh,'price':{**PRICE,'observed_forwards':counts[0],'observed_sequences':counts[1]},'checkpoint_sha256':checkpoint.weights_sha256,'runner_sha256':digest(RUNNER),'binding_sha256':digest(BINDING),'fits':0,'backwards':0,'updates':0,'quantized':False,'wall_seconds':time.perf_counter()-tic};validate_result_contract(result,PREDICTION_REGISTRY);atomic_create_json(OUT,result);print(json.dumps({'terminal':terminal,'predictions':pred,'selected':selected,'sixth_reports':sixth['reports'],'seventh_reports':None if seventh is None else seventh['reports'],'price':result['price']},indent=2));assert terminal!='invalid'
if __name__=='__main__':main()
