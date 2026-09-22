"""CPU analytic Jacobians of native selected quartic and frozen CP programs.
Local input sensitivity, not causal attribution or identification of concepts.
"""
import json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;SCALE=19054614563.464127

def native_value_jac(teacher,x):
 C,l,r,D,A,B=teacher;a=x@A.T;b=x@B.T;h=(a*b)@D.T;u=h@l.T;v=h@r.T;y=(u*v)@C.T;grads=[]
 for c in C:
  t=((v*c)@l+(u*c)@r)@D
  grads.append((t*b)@A+(t*a)@B)
 return y,torch.stack(grads,1)

def cp_value_jac(C,factors,x):
 z=[x@f.T for f in factors];y=torch.stack(z).prod(0)@C.T;J=x.new_zeros(len(x),len(C),x.shape[1])
 for k in range(4):
  prod=torch.stack([z[j] for j in range(4) if j!=k]).prod(0)
  J+=torch.einsum('nk,gk,kd->ngd',prod,C,factors[k])
 return y,J

def tangent(J,x):return J-(J*x[:,None,:]).sum(-1,keepdim=True)*x[:,None,:]/x.square().sum(-1)[:,None,None]

def controls():
 out=[]
 for seed,name in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(14000+seed);T=[torch.randn(*s,dtype=torch.float64) for s in [(2,4),(4,3),(4,3),(3,5),(5,3),(5,3)]]
  if name=='shared_input':T[4][1]=T[4][0]
  if name=='shared_output':T[0][1]=T[0][0]
  if name=='squares':T[5]=T[4].clone();T[2]=T[1].clone()
  if name=='cancellation':T[0][0,1]=-T[0][0,0];T[1][1]=T[1][0];T[2][1]=T[2][0]
  x=torch.randn(7,3,dtype=torch.float64,requires_grad=True);y,J=native_value_jac(T,x);ref=torch.stack([torch.autograd.grad(y[:,g].sum(),x,retain_graph=True)[0] for g in range(2)],1);err=float(((J-ref).norm()/ref.norm()).detach());assert err<1e-12
  C=torch.randn(2,5,dtype=torch.float64);fs=[torch.randn(5,3,dtype=torch.float64) for _ in range(4)];z,K=cp_value_jac(C,fs,x);ref=torch.stack([torch.autograd.grad(z[:,g].sum(),x,retain_graph=True)[0] for g in range(2)],1);ce=float(((K-ref).norm()/ref.norm()).detach());assert ce<1e-12
  assert torch.allclose((J*x[:,None,:]).sum(-1),4*y,rtol=1e-11,atol=1e-11)
  assert float((tangent(J,x)*x[:,None,:]).sum(-1).abs().max().detach())<1e-10
  out.append(dict(family=name,native_jacobian_error=err,cp_jacobian_error=ce))
 return out

def main():
 torch.set_num_threads(2);checks=controls();torch.set_grad_enabled(False);start=time.monotonic()
 from audit_root_matched_reader import CK
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();vocab=state['lm_head.weight'].double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 w=lambda l,n:state[f'transformer.h.{l}.mlp.{n}.weight'].double()
 T=[readers.T@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')]
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);indices=torch.arange(256)*64+31;x=data['rows'][indices].double();labels=data['target'][indices].double()/SCALE
 y,J=native_value_jac(T,x);replay=float((y-labels).norm()/labels.norm());assert replay<1e-5
 euler=float(((J*x[:,None,:]).sum(-1)-4*y).norm()/(4*y).norm());assert euler<1e-10
 direction=torch.randn(3,1152,generator=torch.Generator().manual_seed(14010),dtype=torch.float64);direction=direction/direction.norm(dim=1,keepdim=True);finite=[]
 for eps in [1e-3,1e-4]:
  yp,_=native_value_jac(T,x[:3]+eps*direction);ym,_=native_value_jac(T,x[:3]-eps*direction);fd=(yp-ym)/(2*eps);ref=(J[:3]*direction[:,None,:]).sum(-1);err=float((fd-ref).norm()/ref.norm());assert err<1e-6;finite.append(dict(step=eps,relative_error=err))
 weights=torch.tensor(json.loads((P/'OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json').read_text())['weights'],dtype=torch.float64)
 covariance_sqrt=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True)['projections']['covariance']['whitener'].double()
 rows=[]
 for seed in [1001,1002]:
  a=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);pred,K=cp_value_jac(a['coefficients'].double()/SCALE,[f.double() for f in a['factors']],x)
  error=K-J;er=tangent(error,x);jt=tangent(J,x)
  er_cov=er@covariance_sqrt;jt_cov=jt@covariance_sqrt
  feature=[dict(feature=g,covariance_tangent_gradient_error=float(er_cov[:,g].norm()/jt_cov[:,g].norm()),value_error=float((pred[:,g]-y[:,g]).norm()/y[:,g].norm()),ambient_gradient_error=float(error[:,g].norm()/J[:,g].norm()),tangent_gradient_error=float(er[:,g].norm()/jt[:,g].norm())) for g in range(16)]
  coord=[]
  for name,ww in [('uniform',torch.ones_like(weights)),('balanced',weights)]:
   ee=(er.square()*ww[None,:,None]).sum((0,1));nn=(jt.square()*ww[None,:,None]).sum((0,1));share=ee/ee.sum();order=share.argsort(descending=True)
   coord.append(dict(metric=name,weighted_tangent_error=float((ee.sum()/nn.sum()).sqrt()),residual_top64_coordinate_share=float(share[order[:64]].sum()),native_top64_coordinate_share=float(nn.topk(64).values.sum()/nn.sum()),native_energy_on_residual_top64=float(nn[order[:64]].sum()/nn.sum()),effective_residual_coordinates=float(1/share.square().sum()),top16_coordinates=[dict(index=int(i),residual_share=float(share[i])) for i in order[:16]]))
  spectra={}
  for name,jac in [('residual',er),('native',jt)]:
   matrix=jac.reshape(-1,1152);gram=matrix.T@matrix;ev=torch.linalg.eigvalsh(gram).flip(0).clamp_min(0);ev=ev/ev.sum()
   spectra[name]=dict(top_rank_energy={str(k):float(ev[:k].sum()) for k in [1,4,16,64,128,256,512]},effective_rank=float(1/ev.square().sum()))
  rows.append(dict(seed=seed,covariance_tangent_gradient_error=float(er_cov.norm()/jt_cov.norm()),input_gradient_spectra=spectra,features=feature,coordinate_summaries=coord,pooled_value_error=float((pred-y).norm()/y.norm()),ambient_gradient_error=float(error.norm()/J.norm()),tangent_gradient_error=float(er.norm()/jt.norm())))
 result=dict(controls=checks,native_label_replay=replay,native_euler_replay=euler,native_finite_difference=finite,rows=rows,states=256,selection='One state per each of256previouslyopenedfreshdocuments, position31, fixed before derivatives.',seconds=time.monotonic()-start,scope='Exact local Jacobian of selected16-output purequartic map in1152normalizedinputcoordinates. Tangentprojection removes radialrescaling. Coordinate concentrations are basis-dependent sensitivity summaries, not attribution of observed residual values or semantic/causal identification. No gradientprediction fitting, no newdata claim.')
 (P/'RESIDUAL_INPUT_SENSITIVITY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['controls','rows']},indent=2))
 for r in rows:print(r['seed'],r['ambient_gradient_error'],r['tangent_gradient_error'],r['coordinate_summaries'],flush=True)
if __name__=='__main__':main()
