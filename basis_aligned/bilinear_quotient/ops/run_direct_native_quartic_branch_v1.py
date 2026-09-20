#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preservation pred_c_effect
"""Native branch intervention; NATIVE_QUARTIC_BRANCH_PLAN_V1.md.
pred_a_instrument: exact replay residual/logit and output solve error <1e-5.
pred_b_preservation: ten-product CE damage<.02 and KL<.02.
pred_c_effect: ten-product logit change norm <0.5 ablation norm.
Null: isolated fit fails native adoption. 16 fresh docs context128.
Price: 19632 coefficients/10 products,47312/26 plus shared frame;
subtraction instrument retains native compute, no runtime savings claimed.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
PLAN=dict(documents=list(range(64,80)),context=128,arms=['exact','ablation','quartic10','quartic26'],native_forwards=16)
PANEL_PATH=None
OUTPUT_STEM='NATIVE_QUARTIC_BRANCH_V1'
PROGRAM_SPECS=None
PRIMARY='quartic10'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from frozen_program_evaluation import quartic
 from native_quartic_branch import bilinear,pure_branch,replace_branch
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/(OUTPUT_STEM+'.json');assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();assert not model.config.gated and model.config.bilinear
 blocks=model.transformer.h;b16=blocks[16];b17=blocks[17];_,ru=torch.linalg.qr(model.lm_head.weight.float());ru=ru.double()
 artifact=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);scale=float(artifact['teacher_scale']);programs={'quartic26':artifact['programs'][8],'quartic10':torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4]};solves=[]
 if PROGRAM_SPECS is not None:programs={name:torch.load(P/file,weights_only=True)['programs'][key] for name,(file,key) in PROGRAM_SPECS.items()}
 for name,s in programs.items():
  s={k:v.cuda().double() for k,v in s.items()};key='output_writer' if 'output_writer' in s else 'W'
  for k in [key,'constant']+(['skip_writer'] if 'skip_writer' in s else []):
   y=s[k]*scale;s[k]=torch.linalg.solve_triangular(ru,y[:,None] if y.ndim==1 else y,upper=True)
   if y.ndim==1:s[k]=s[k][:,0]
   solves.append(float((ru@s[k]-y).norm()/y.norm()))
  programs[name]=s
 ids=torch.load(PANEL_PATH,weights_only=True) if PANEL_PATH is not None else torch.load(BQ/'.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[64:80,:129];digest=hashlib.sha256(ids.numpy().tobytes()).hexdigest();rows=[];checks=[];L,R,D=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']]
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 for doc,tokens in zip(PLAN['documents'],ids):
  tokens=tokens.cuda()[None];x=F.rms_norm(model.transformer.wte(tokens[:,:128]),(1152,));x0=x;v1=None;cache={}
  def pre16(module,args):cache['x16']=args[0].clone()
  def post16(module,args,value):cache['m16']=value.clone()
  def postattn(module,args,value):cache['attn17']=value[0].clone()
  def postmlp(module,args,value):cache['mlp17']=value.clone();cache['norm17']=args[0].clone()
  handles=[b16.mlp.register_forward_pre_hook(pre16),b16.mlp.register_forward_hook(post16),b17.attn.register_forward_hook(postattn),b17.mlp.register_forward_hook(postmlp)]
  try:
   for i,block in enumerate(blocks):
    if i==17:incoming=x.clone()
    x,v1=block(x,v1,x0)
  finally:
   for handle in handles:handle.remove()
  h=b17.lambdas[0]*incoming+b17.lambdas[1]*x0+cache['attn17'];original=pure_branch(cache['m16'],b16.mlp.Down_bias,b17.lambdas[0],L,R,D)
  previous=bilinear(cache['x16'],b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight);exact=bilinear(b17.lambdas[0]*previous,L,R,D)
  checks.extend([rel(F.rms_norm(h,(1152,)),cache['norm17']),rel(h+cache['mlp17'],x)])
  logits=lambda state:30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
  native=logits(x);logp=F.log_softmax(native,dim=-1);p=logp.exp();native_ce=float(F.cross_entropy(native.flatten(0,1),tokens[:,1:129].flatten()))
  for arm in PLAN['arms']:
   replacement=exact if arm=='exact' else torch.zeros_like(original) if arm=='ablation' else quartic(programs[arm],cache['x16'].flatten(0,1).double()).reshape_as(original).float()
   mlp=replace_branch(cache['mlp17'],h,original,replacement);changed=x+(mlp-cache['mlp17']);z=logits(changed);delta=z-native;ce=float(F.cross_entropy(z.flatten(0,1),tokens[:,1:129].flatten()))
   row=dict(document=doc,arm=arm,native_ce=native_ce,ce_added=ce-native_ce,kl=float((p*(logp-F.log_softmax(z,dim=-1))).sum(-1).mean()),logit_mse=float(delta.square().mean()),argmax_agreement=float((z.argmax(-1)==native.argmax(-1)).float().mean()),relative_logit_error=rel(z,native),relative_residual_error=rel(changed,x))
   if arm=='exact':checks.extend([row['relative_logit_error'],row['relative_residual_error']])
   rows.append(row)
  print('document',doc,'done',flush=True)
 summary={arm:{k:sum(r[k] for r in rows if r['arm']==arm)/len(ids) for k in ['native_ce','ce_added','kl','logit_mse','argmax_agreement']} for arm in PLAN['arms']};ratio=(summary[PRIMARY]['logit_mse']/summary['ablation']['logit_mse'])**.5 if summary['ablation']['logit_mse']>0 else None
 pred=dict(pred_a_instrument=max(checks+solves)<1e-5,pred_b_preservation=summary[PRIMARY]['ce_added']<.02 and summary[PRIMARY]['kl']<.02,pred_c_effect=ratio is not None and ratio<.5)
 result=dict(plan=PLAN,token_hash=digest,records=rows,summary=summary,checks_max=max(checks),solve_error=max(solves),frame_condition=float(torch.linalg.cond(ru)),ten_product_to_ablation_logit_norm=ratio,predictions=pred,seconds=time.perf_counter()-start,scope='Native additive pure-quartic branch intervention with fixed denominator; not whole-block simplification, semantics or OOD.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['summary','predictions','checks_max','solve_error','ten_product_to_ablation_logit_norm']},indent=2),flush=True)
if __name__=='__main__':main()
