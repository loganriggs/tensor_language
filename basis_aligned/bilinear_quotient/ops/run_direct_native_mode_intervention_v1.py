#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_major pred_c_all
"""Native scalar-feature removal; NATIVE_MODE_INTERVENTION_PLAN_V1.md.
pred_a_replay: relative zero/joint projection replay<1e-5.
pred_b_major: modes0/1 effect cosine>.90 and relative error<.40.
pred_c_all: all4 effect cosine>.80 and relative error<.65.
Null: extracted scalar features fail native removal prediction.
Price: 19632 stored candidate coefficients/10 products plus analysis projections;
original native background retained, no whole-model speedup or semantic claim.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
PLAN=dict(documents=list(range(80,96)),context=128,modes=[0,1,2,3,'joint'],native_forwards=16)
PANEL_PATH=None
OUTPUT_STEM='NATIVE_MODE_INTERVENTION_V1'
PROGRAM_FILE='FUSED_ROOT_PROGRAM_V1.pt'
PROGRAM_KEY=4
EXTRACTED_FILE=None
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 context=PLAN['context']
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from frozen_program_evaluation import quartic
 from native_quartic_branch import pure_branch
 from logit_effect_partition import partition
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/(OUTPUT_STEM+'.json');assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();blocks=model.transformer.h;b16=blocks[16];b17=blocks[17]
 assert model.config.bilinear and not model.config.gated
 _,ru=torch.linalg.qr(model.lm_head.weight.float());ru=ru.double()
 scale=float(torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True)['teacher_scale'])
 view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].cuda().double();mu=view['output_mean'].cuda().double()
 writer=torch.linalg.solve_triangular(ru,scale*U,upper=True)
 extracted=None
 if EXTRACTED_FILE is not None:
  from extract_scalar_modes import evaluate as scalar_evaluate
  extracted=torch.load(EXTRACTED_FILE,weights_only=True);scalar_program={k:v.cuda().double() for k,v in extracted['program'].items()};export_writer=extracted['residual_writer'].cuda().double()
 s={k:v.cuda().double() for k,v in torch.load(P/PROGRAM_FILE,weights_only=True)['programs'][PROGRAM_KEY].items()}
 ids=torch.load(PANEL_PATH,weights_only=True) if PANEL_PATH is not None else torch.load(BQ/'.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[80:96,:context+1];records=[];checks=[]
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 checks.append(rel(ru@writer,scale*U))
 if extracted is not None:
  checks.append(rel(export_writer,writer));writer=export_writer
 assert len(ids)==len(PLAN['documents']) and ids.shape[1]==context+1
 for doc,tokens in zip(PLAN['documents'],ids):
  tokens=tokens.cuda()[None];x=F.rms_norm(model.transformer.wte(tokens[:,:context]),(1152,));x0=x;v1=None;cache={}
  def pre16(module,args):cache['x16']=args[0].clone()
  def post16(module,args,value):cache['m16']=value.clone()
  def postattn(module,args,value):cache['attn17']=value[0].clone()
  handles=[b16.mlp.register_forward_pre_hook(pre16),b16.mlp.register_forward_hook(post16),b17.attn.register_forward_hook(postattn)]
  try:
   for i,block in enumerate(blocks):
    if i==17:incoming=x.clone()
    x,v1=block(x,v1,x0)
  finally:
   for handle in handles:handle.remove()
  h=b17.lambdas[0]*incoming+b17.lambdas[1]*x0+cache['attn17'];den=h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps
  pure=pure_branch(cache['m16'],b16.mlp.Down_bias,b17.lambdas[0],b17.mlp.Left.weight,b17.mlp.Right.weight,b17.mlp.Down.weight).double().flatten(0,1)
  y=pure@ru.T/scale;yhat=quartic(s,cache['x16'].flatten(0,1).double());a=(y-mu)@U;b=(yhat-mu)@U
  if extracted is not None:
   scalar=scalar_evaluate(scalar_program,cache['x16'].flatten(0,1).double());checks.append(rel(scalar,b));b=scalar
  # Independently solve the entire rank-four projection for the joint oracle.
  joint=torch.linalg.solve_triangular(ru,scale*((y-mu)@U@U.T).T,upper=True).T
  checks.append(rel(a@writer.T,joint))
  logits=lambda state:30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
  native=logits(x);checks.append(rel(logits(x-torch.zeros_like(x)),native));logp=F.log_softmax(native,dim=-1);p=logp.exp();target=tokens[:,1:context+1].flatten();basece=F.cross_entropy(native.flatten(0,1),target,reduction='none')
  for mode in PLAN['modes']:
   sl=list(range(4)) if mode=='joint' else [mode];effects=[];ces=[];row=dict(document=doc,mode=mode)
   for label,amplitudes in [('native',a),('predicted',b)]:
    edit=(amplitudes[:,sl]@writer[:,sl].T).reshape_as(x).float()/den
    z=logits(x-edit);effect=(z-native).double();ce=F.cross_entropy(z.flatten(0,1),target,reduction='none')-basece
    row[label+'_ce_added']=float(ce.mean());row[label+'_kl']=float((p*(logp-F.log_softmax(z,dim=-1))).sum(-1).mean());row[label+'_argmax_agreement']=float((z.argmax(-1)==native.argmax(-1)).float().mean());row[label+'_effect_energy']=float(effect.square().sum());row[label+'_ce_effect_energy']=float(ce.double().square().sum());effects.append(effect);ces.append(ce.double())
   row['effect_dot']=float((effects[0]*effects[1]).sum());row['effect_error_energy']=float((effects[0]-effects[1]).square().sum());row['ce_effect_dot']=float((ces[0]*ces[1]).sum());row['ce_effect_error_energy']=float((ces[0]-ces[1]).square().sum());row.update(partition(*effects));records.append(row)
  print('document',doc,'done',flush=True)
 summary={}
 for mode in PLAN['modes']:
  rows=[r for r in records if r['mode']==mode];sums={k:sum(r[k] for r in rows) for k in rows[0] if k not in ['document','mode']};entry={k:v/len(rows) for k,v in sums.items() if k.endswith(('ce_added','kl','argmax_agreement'))}
  for prefix in ['effect','ce_effect','centered_effect']:
   en=sums['native_'+prefix+'_energy'];ep=sums['predicted_'+prefix+'_energy'];entry[prefix+'_cosine']=sums[prefix+'_dot']/(en*ep)**.5 if en*ep>0 else None;entry[prefix+'_relative_error']=(sums[prefix+'_error_energy']/en)**.5 if en>0 else None
  entry['native_common_fraction']=sums['native_common_effect_energy']/sums['native_effect_energy'];entry['native_logit_effect_rms']=(sums['native_effect_energy']/(len(ids)*context*50304))**.5;summary[str(mode)]=entry
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_major=all(summary[str(g)]['effect_cosine']>.9 and summary[str(g)]['effect_relative_error']<.4 for g in [0,1]),pred_c_all=all(summary[str(g)]['effect_cosine']>.8 and summary[str(g)]['effect_relative_error']<.65 for g in range(4)))
 result=dict(plan=PLAN,token_hash=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),records=records,summary=summary,predictions=pred,replay_max=max(checks),seconds=time.perf_counter()-start,scope='Matched native-background scalar-mode removal, fixed calibration centering and output directions. No semantics/OOD/selectivity claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,predictions=pred,replay_max=max(checks)),indent=2),flush=True)
if __name__=='__main__':main()
