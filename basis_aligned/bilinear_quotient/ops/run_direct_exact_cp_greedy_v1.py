#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_finite pred_b_gain pred_c_explained
"""Native pure quartic, exact-gradient residual CP growth to8atoms.
pred_a_finite all finite; pred_b_gain sampled coefficient error<.99;
pred_c_explained final exact objective improvement/estimated teacher norm>.02.
Two restarts/atom,300Adamsteps, no coefficient sampling in training.
Price5*1152*r plus outputframe; no causal adoption.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(atoms=8,restarts=2,steps=300,lr=.05,evaluation_queries=8192,function_queries=256,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from quartic_cp import cp_gram,directional,cp_entries
 from implicit_quartic import entries
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_EXACT_CP_GREEDY_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight'),w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher[0]/=scale;d=1152;gen=torch.Generator(device='cuda');gen.manual_seed(1651);idx=torch.randint(d,(8192,4),device='cuda',generator=gen);target=torch.cat([entries(*teacher,z) for z in idx.split(512)]);x=torch.randn(256,d,device='cuda',generator=gen);C,L,R,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;fy=((h@L.T)*(h@R.T))@C.T
  factors=[torch.empty(0,d,device='cuda') for _ in range(4)];writer=torch.empty(1152,0,device='cuda');gain=0.;records=[];attempts=[];states={}
 for stage in range(1,9):
  candidates=[]
  for seed in range(2):
   torch.manual_seed(1000*stage+seed);raw=[torch.nn.Parameter(torch.randn(1,d,device='cuda')) for _ in range(4)];opt=torch.optim.Adam(raw,lr=.05);base=None;history=[]
   for step in range(300):
    opt.zero_grad();F=[v/v.norm(dim=1,keepdim=True) for v in raw];K=cp_gram(F,F);cross=directional(*teacher,F).T
    if writer.shape[1]:cross=cross-writer@cp_gram(factors,F)
    c=cross/K[0,0];objective=(c.square().sum()*K[0,0]-2*(cross*c).sum())
    if base is None:base=max(float(-objective.detach()),1e-30)
    assert bool(torch.isfinite(objective))
    if step%100==0:history.append(dict(step=step,residual_explained=float(-objective.detach())))
    (objective/base).backward();opt.step()
    for group in opt.param_groups:group['lr']=.05*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/300)))
   with torch.no_grad():
    F=[v/v.norm(dim=1,keepdim=True) for v in raw];K=cp_gram(F,F);cross=directional(*teacher,F).T
    if writer.shape[1]:cross=cross-writer@cp_gram(factors,F)
    score=float(cross.square().sum()/K[0,0]);candidates.append((score,[v.detach() for v in F]));attempts.append(dict(stage=stage,seed=seed,residual_explained=score,initial_explained=base,history=history))
  score,F=max(candidates,key=lambda z:z[0])
  with torch.no_grad():
   new=[torch.cat([a,b],0) for a,b in zip(factors,F)];K=cp_gram(new,new).double();cross=directional(*teacher,new).T.double();c=torch.linalg.solve(K+1e-6*K.diag().mean()*torch.eye(stage,device='cuda'),cross.T).T.float();proposed=float(2*(cross*c).sum()-((c.double()@K)*c).sum());accepted=proposed>=gain-1e-10
   if accepted:factors=new;writer=c;gain=proposed
   else:
    attempts.append(dict(stage=stage,rejected_refit=True,previous_gain=gain,proposed_gain=proposed,residual_explained=score));print('Rejected worsening refit; retaining previous model',flush=True);break
   if stage in [1,2,4,8]:
    prediction=torch.cat([cp_entries(writer,factors,z) for z in idx.split(512)]);coefficient=float((prediction-target).norm()/target.norm());prod=torch.ones(len(x),stage,device='cuda')
    for a in factors:prod=prod*(x@a.T)
    function=float((prod@writer.T-fy).norm()/fy.norm());row=dict(atoms=stage,explained_fraction_using_estimated_teacher_norm=gain,estimated_coefficient_error=math.sqrt(max(0,1-gain)),evaluation_coefficient_error=coefficient,gaussian_function_error=function,reduced_parameters=5*d*stage,expanded_parameters=(4*d+50304)*stage);records.append(row);states[stage]=dict(C=writer.cpu(),factors=[v.cpu() for v in factors]);print(json.dumps(row),flush=True)
  out.write_text(json.dumps(dict(plan=PLAN,records=records,attempts=attempts,seconds=time.perf_counter()-start),indent=2)+'\n')
 result=dict(plan=PLAN,records=records,attempts=attempts,seconds=time.perf_counter()-start,predictions=dict(pred_a_finite=all(math.isfinite(r['residual_explained']) for r in attempts),pred_b_gain=any(r['evaluation_coefficient_error']<.99 for r in records),pred_c_explained=gain>.02),scope='Exact algebraic residual gradients, greedyCP growth, teacher norm estimated only for scale/error labels. Finite heldout coefficient/function diagnostics; pure twoMLP numerator, no circuit identification.')
 guard_torch_save(dict(students=states,teacher_scale=scale),str(P/'NATIVE_EXACT_CP_GREEDY_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
