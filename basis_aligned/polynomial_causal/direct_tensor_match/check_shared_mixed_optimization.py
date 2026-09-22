"""Planted shared-quadratic recovery: exact coefficients plus exact Gaussian quadrature.
All optimizer/rate/restart arms reported; finite-dimensional test, not native ranking.
"""
import itertools,json,time,math
from pathlib import Path
import numpy as np
import torch
from sparse_quartic_bank import entries,features
from shared_quadratic_bank import normalize_bank
P=Path(__file__).resolve().parent

def main(steps=100,width=4,rates=(.03,.1),dimension=3):
 torch.set_num_threads(2);dtype=torch.float64;start=time.monotonic();d=dimension
 indices=torch.tensor(list(itertools.product(range(d),repeat=4)))
 nodes,weights=np.polynomial.hermite.hermgauss(5);ind=torch.tensor(list(itertools.product(range(5),repeat=d)));z=torch.tensor(nodes*2**.5,dtype=dtype)[ind];prob=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ind].prod(1)
 targetpairs=torch.triu_indices(3,3);pairs=torch.triu_indices(width,width);rows=[]
 for family_index,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(12400+family_index);U=torch.randn(3,2,d,dtype=dtype);V=torch.randn_like(U);C=torch.randn(2,6,dtype=dtype)
  if family=='shared_input':U[1]=U[0]
  if family=='shared_output':C[1]=.7*C[0]
  if family=='squares':V=U.clone()
  if family=='cancellation':
   U[1]=U[0];V[1]=V[0];C[:,2]=3.;C[:,4]=-3.
  U,V=normalize_bank(U,V);mean=torch.randn(d,dtype=dtype)*.3;S=torch.diag(torch.tensor([.7,1.,1.4],dtype=dtype) if d==3 else torch.linspace(.7,1.4,d,dtype=dtype));x=z@S.T+mean
  target0=entries(U,V,targetpairs,indices)@C.T;target1=features(x,U,V,targetpairs)@C.T
  energy0=target0.square().sum();energy1=(target1.square()*prob[:,None]).sum()
  # Exact planted replay checks the target representation, including canceling products.
  assert torch.isfinite(energy0+energy1) and energy0>0 and energy1>0
  for seed in [1,2]:
   for optname in ['Adam','Muon']:
    for rate in rates:
     torch.manual_seed(12500+seed);params=[torch.nn.Parameter(torch.randn(width*2,d,dtype=dtype)) for _ in range(2)]
     if optname=='Adam':opt=torch.optim.Adam(params,lr=rate)
     else:opt=torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
     best=None;initial=None;initial_metrics=None
     for step in range(steps+1):
      a,b=normalize_bank(*[p.reshape(width,2,d) for p in params]);phi0=entries(a,b,pairs,indices);phi1=features(x,a,b,pairs)
      A=torch.cat([phi0/(2*energy0).sqrt(),phi1*(prob/(2*energy1)).sqrt()[:,None]])
      Y=torch.cat([target0/(2*energy0).sqrt(),target1*(prob/(2*energy1)).sqrt()[:,None]])
      G=A.T@A;X=Y.T@A;c=torch.linalg.solve(G+1e-8*torch.eye(pairs.shape[1],dtype=dtype),X.T).T
      residual=A@c.T-Y;loss=residual.square().sum()+1e-8*c.square().sum();value=float(loss.detach())
      if initial is None:
       initial=value
       with torch.no_grad():initial_metrics=dict(coefficient_error=float((phi0@c.T-target0).norm()/target0.norm()),gaussian_error=float(((prob[:,None]*(phi1@c.T-target1).square()).sum()/energy1).sqrt()),coefficient_design_rank=int(torch.linalg.matrix_rank(phi0)),ambient_quartic_dimension=math.comb(d+3,4))
      if best is None or value<best['loss']:
       with torch.no_grad():best=dict(loss=value,step=step,coefficient_error=float((phi0@c.T-target0).norm()/target0.norm()),gaussian_error=float(((prob[:,None]*(phi1@c.T-target1).square()).sum()/energy1).sqrt()))
      if step==steps:break
      opt.zero_grad();loss.backward();assert all(torch.isfinite(p.grad).all() for p in params);opt.step()
      for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
     row=dict(family=family,seed=seed,optimizer=optname,rate=rate,initial_loss=initial,initial=initial_metrics,best=best,recovery=best['coefficient_error']<.01 and best['gaussian_error']<.01);rows.append(row);print(json.dumps(row),flush=True)
 result=dict(rows=rows,steps=steps,width=width,dimension=d,rates=list(rates),seconds=time.monotonic()-start,scope='Five planted shared-quadratic families, two random starts each, Adam/Muon at.03/.1,100steps. Teacher3quadratics x2terms, student4x2 complete10rootpairs,2outputs,d3. Exact81coefficient entries and5^3Gauss-Hermite degree8-exact quadrature under shifted diagonalGaussian. Loss averages relative squared coefficient/Gaussian errors plusreadoutridge1e-8. No probes or evaluation-panel selection; finite toy optimizer ranking doesnotprove native convergence/identifiability.')
 output='SHARED_MIXED_OPTIMIZATION_CONTROLS_V1.json' if steps==100 and width==4 and d==3 else f'SHARED_MIXED_OPTIMIZATION_W{width}_S{steps}_V1.json'
 if d!=3:output=output.replace('_V1.json',f'_D{d}_V1.json')
 result['scope']=result['scope'].replace('100steps',f'{steps}steps').replace('student4x2 complete10rootpairs',f'student{width}x2 complete{pairs.shape[1]}rootpairs').replace('at.03/.1',f'at{list(rates)}').replace('d3',f'd{d}').replace('Exact81coefficient entries and5^3',f'Exact{d**4}coefficient entries and5^{d}')
 (P/output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=100);parser.add_argument('--width',type=int,default=4);parser.add_argument('--rate',type=float,nargs='+',default=[.03,.1]);parser.add_argument('--dimension',type=int,default=3);a=parser.parse_args();main(a.steps,a.width,a.rate,a.dimension)
