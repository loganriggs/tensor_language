#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;4exact coefficient evaluations,32edges rank16.
"""pred_a native tangent FD<=1e-4 and baseline replay<=1e-8;
pred_b nonzero projected gradient>1e-6; pred_c peak GPU allocation<28GiB.
Price only, no fit or circuit claim. Fixed learned32-edge graph.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quartic_selected_edges_v1 import gram,target_cross
from quartic_manifold_cg_v1 import tangent,retract
from composed_quartic_contraction_v1 import contract
STEM='QUARTIC_SELECTED_EDGES_PRICE_V1'
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
 assert all(digest(k)==v for k,v in binding.items())
 control=json.loads((P/'QUARTIC_SELECTED_EDGES_V1_CONTROL.json').read_text())
 assert control['pred_a'] and control['pred_b'] and control['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,coefficient_evaluations=4,edges=32,maximum_batch=256)));return
 out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(300)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 native=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0])
 p=torch.load(P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_PROGRAMS.pt',weights_only=True)['programs'][0]
 b=p['input_readers'].cuda();n=p['inner_weights'].cuda();edges=p['pairs'][:,p['selected']].cuda()
 native[-1]=(metric@p['output_writers'].cuda()).T@native[-1]
 divisor=torch.load(P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt',weights_only=True)['divisor']
 def evaluate(b,n,gradient=False):
  if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
  with torch.set_grad_enabled(gradient):
   k=gram(b,n,edges);c=target_cross(b,n,edges,lambda slots:contract(slots,*native,scale))
   with torch.no_grad():mix=torch.linalg.solve(k,c)
   value=((mix*(k@mix)).sum()-2*(mix*c).sum())/divisor
   if gradient:
    gb,gn=tangent(b,n,*torch.autograd.grad(value,(b,n)))
    return float(value),gb.detach(),gn.detach()
   return float(value)
 torch.cuda.synchronize();tic=time.perf_counter();y,gb,gn=evaluate(b,n,True);torch.cuda.synchronize();gradient_seconds=time.perf_counter()-tic
 old=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_RESULT.json').read_text())['history'][-1]['objective']
 ratio=json.loads((P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_RESULT.json').read_text())['reports'][0]['sparse_capture_ratio']
 replay=abs(y-old*ratio)
 torch.manual_seed(91922);db,dn=tangent(b,n,torch.randn_like(b),torch.randn_like(n));length=(db.square().sum()+dn.square().sum()).sqrt();db/=length;dn/=length
 eps=1e-5;tic=time.perf_counter();plus=evaluate(*retract(b,n,db,dn,eps));minus=evaluate(*retract(b,n,db,dn,-eps));torch.cuda.synchronize();forward_seconds=(time.perf_counter()-tic)/2
 numerical=(plus-minus)/(2*eps);analytic=float((gb*db).sum()+(gn*dn).sum());fd=abs(numerical-analytic)/max(abs(numerical),abs(analytic),1e-10)
 norm=float((gb.square().sum()+gn.square().sum()).sqrt());after=evaluate(*retract(b,n,-gb/norm,-gn/norm,1e-4))
 result={'pred_a':fd<=1e-4 and replay<=1e-8 and after<y,'pred_b':norm>1e-6,'pred_c':torch.cuda.max_memory_allocated()<28*(1<<30)}
 result.update(dict(initial_objective=y,after_small_step=after,baseline_replay_error=replay,tangent_fd_error=fd,projected_gradient_norm=norm,gradient_seconds=gradient_seconds,forward_seconds=forward_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.perf_counter()-start,scope='Native weight-only fixed mixed-edge gradient price; no optimization or behavioral result.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
