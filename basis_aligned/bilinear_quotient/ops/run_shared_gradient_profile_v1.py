#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_gradient pred_c_resources
"""Native144x4/512shared gradient profile, seed1101; no feature fit/adoption.
Saved baseline replay<1e-4, normal<1e-8, streamed loss<1e-6.
Two profiled finite differences each<1e-3; gradient<180s,peak<20GB.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from shared_mixed_gradient import streamed_gradient,coefficient_rect,gaussian_rect
 from check_shared_mixed_gradient import controls
 from sparse_quartic_bank import support,gram as cg,native_cross as cx
 from shared_gaussian_moments import gram as gg,native_cross as gx
 from noncentral_gaussian_cp import project_shifted
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  rows=controls();torch.manual_seed(12200);u=(torch.randn(144,4,1152,dtype=torch.float64)/34).requires_grad_();v=torch.randn_like(u,requires_grad=True);pairs=support(144,512,1101);bu=torch.zeros(144,4,dtype=u.dtype);bv=torch.zeros_like(bu)
  a=coefficient_rect(u,v,pairs[:,:2],pairs);b=gaussian_rect(u,v,pairs[:,:2],pairs,bu,bv);assert a.shape==b.shape==(2,512)
  (a.square().mean()+b.square().mean()).backward();assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()
  C=torch.randn(16,512,dtype=u.dtype);assert (C.T@C)[:2].shape==a.shape
  print(json.dumps(dict(control_cases=len(rows),actual_factor_shapes=list(u.shape),rectangular_gradient=True)));return
 from audit_root_matched_reader import CK
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'SHARED_GRADIENT_PROFILE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127;seed=1101;ridge=1e-6
 source=torch.load(P/f'GAUSSIAN_SUPPORT_EXCHANGE_SEED{seed}_V1.pt',weights_only=True);archive=torch.load(P/'GAUSSIAN_SUPPORT_EXCHANGE_GRAMS_V1.pt',weights_only=True)[seed]
 pairs=source['pairs'].cuda();u,v=[t.cuda().double() for t in source['factors']];lookup={tuple(p):i for i,p in enumerate(archive['pairs'].T.tolist())};idx=torch.tensor([lookup[tuple(p)] for p in source['pairs'].T.tolist()]);lam=archive['coefficient_weight']
 G=(archive['gaussian_gram'][idx][:,idx]+lam*archive['coefficient_gram'][idx][:,idx]).cuda()/(1+lam);X=(archive['gaussian_cross'][:,idx]+lam*archive['coefficient_cross'][:,idx]).cuda()/(1+lam);eye=torch.eye(512,device='cuda',dtype=u.dtype)
 C=torch.linalg.solve(G+ridge*eye,X.T).T;replay=float((C-source['coefficients'].cuda().double()/scale).norm()/C.norm());normal=float((C@(G+ridge*eye)-X).norm()/X.norm())
 saved=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);cache=saved['projections']['covariance'];S=cache['whitener'].cuda();mu=saved['mean'].cuda();loc=torch.linalg.solve(S,mu);zero=tuple(t.cuda() for t in cache['zero_projection'])
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=source['writer'].cuda().double();vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].cuda().double()
 teacher=[readers.T@w(17,'Down')/scale,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')];tr=teacher[:4]+[teacher[4]@S,teacher[5]@S]
 with torch.no_grad():projection=project_shifted(tr,loc,zero)
 U=u.clone().requires_grad_();V=v.clone().requires_grad_();torch.cuda.synchronize();t=time.monotonic();torch.cuda.reset_peak_memory_stats()
 value=streamed_gradient(U,V,pairs,C,teacher,tr,loc,projection,S,mu,lam,chunk=8);torch.cuda.synchronize();gradient_seconds=time.monotonic()-t;peak=torch.cuda.max_memory_allocated()
 expected=float((G*(C.T@C)).sum()-2*(C*X).sum());loss_replay=abs(value-expected)/(1+abs(expected));gn=(U.grad.square().sum()+V.grad.square().sum()).sqrt();pn=(U.square().sum()+V.square().sum()).sqrt();du=U.grad/gn*pn;dv=V.grad/gn*pn;analytic=float((U.grad*du).sum()+(V.grad*dv).sum());print(json.dumps(dict(gradient_seconds=gradient_seconds,peak=peak,loss_replay=loss_replay,gradient_norm=float(gn))),flush=True)
 @torch.no_grad()
 def profiled(a,b):
  g=(gg(a@S,b@S,pairs,a@mu,b@mu)+lam*cg(a,b,pairs))/(1+lam)
  x=(gx(tr,loc,projection,a@S,b@S,pairs,a@mu,b@mu)+lam*torch.cat([cx(teacher,a,b,part) for part in pairs.split(64,dim=1)],1))/(1+lam)
  c=torch.linalg.solve(g+ridge*eye,x.T).T
  return float(-(c*x).sum())
 finite=[]
 for eps in [1e-4,3e-5]:
  plus=profiled(u+eps*du,v+eps*dv);minus=profiled(u-eps*du,v-eps*dv);fd=(plus-minus)/(2*eps);error=abs(fd-analytic)/max(abs(analytic),1e-12);finite.append(dict(epsilon=eps,finite_difference=fd,analytic=analytic,relative_error=error));print(json.dumps(finite[-1]),flush=True)
 pred=dict(pred_a_integrity=replay<1e-4 and normal<1e-8 and loss_replay<1e-6,pred_b_gradient=all(r['relative_error']<1e-3 for r in finite),pred_c_resources=gradient_seconds<180 and peak<20_000_000_000)
 result=dict(predictions=pred,seed=seed,ridge=ridge,coefficient_weight=lam,readout_replay=replay,normal_residual=normal,loss_replay=loss_replay,gradient_seconds=gradient_seconds,peak_gradient_bytes=peak,finite_differences=finite,seconds=time.monotonic()-start,scope='Exact shared144x4/512gradient resource/correctness profile only. No parameter update, native effect/OOD test or adoption. Detached optimal readout includesridge; teacher constant omitted.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
