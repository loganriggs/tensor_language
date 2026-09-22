"""Weight-derived active input span for densely embedded quartic controls."""
import argparse,json,time
import torch
from toy_local_quartic_residual import planted
from toy_paired_residual_learning import fit_readout
from quartic_cp_profile import normalize_factors
from correlated_gaussian_cp import metric
from gaussian_cp_derivative_gram import gram as derivative_gram
from audit_conditional_residual_accounting import P

def main(ambient_control=False):
 torch.set_num_threads(2);start=time.monotonic();rows=[];geometry=[]
 gen=torch.Generator().manual_seed(41001);embedding=torch.linalg.qr(torch.randn(1152,4,dtype=torch.float64,generator=gen),mode='reduced').Q
 mu=embedding@torch.tensor([.1,-.2,.3,.1],dtype=torch.float64)
 for family in [0,1,3,4,5]:
  _,_,_,rf,rc=planted(0 if family==5 else family);rf=[a.clone() for a in rf]
  if family==5:rf[0][:]=rf[0][0].clone();rf[1][:]=rf[1][0].clone()
  wide=[f@embedding.T for f in rf];bias=[f@mu for f in wide]
  M=derivative_gram(wide,bias,input_coefficients=rc);M=(M+M.T)/2;ev,V=torch.linalg.eigh(M);keep=ev>1e-10*ev[-1];V=V[:,keep];rank=V.shape[1];assert rank==4
  span=float((embedding-V@(V.T@embedding)).norm()/embedding.norm());assert span<1e-10
  reduced=[f@V for f in wide];location=V.T@mu;rb=[f@location for f in reduced]
  x=torch.randn(23,1152,dtype=torch.float64,generator=gen);actual=torch.stack([x@f.T for f in wide]).prod(0)@rc.T;compressed=torch.stack([(x@V)@f.T for f in reduced]).prod(0)@rc.T;replay=float((actual-compressed).norm()/actual.norm());assert replay<1e-10
  norms={name:((rc.T@rc)*metric(reduced,rb,reduced,rb,rho)).sum() for name,rho in [('value',None),('response',.5)]}
  geometry.append(dict(family=family,rank=rank,span_error=span,function_replay=replay,eigenvalues_top=ev[-6:].tolist(),shared_projection_floats=1152*rank,student_floats=1152*rank+4*4*rank+8,variable_products=12))
  if ambient_control:
   reduced=wide;location=mu;rb=bias;rank=1152
  for seed in [0,1]:
   torch.manual_seed(24100+seed);params=[torch.randn(4,rank,dtype=torch.float64,requires_grad=True) for _ in range(4)];opt=torch.optim.Adam(params,lr=.1);best=None
   for step in range(251):
    fs=normalize_factors(params);bs=[f@location for f in fs];G=metric(fs,bs,fs,bs);X=rc@metric(reduced,rb,fs,bs);loss,C=fit_readout(G,X);value=float(loss.detach())
    if best is None or value<best[0]:best=(value,step,[f.detach().clone() for f in fs],C.detach().clone())
    if step==250:break
    opt.zero_grad();(loss/norms['value']).backward();opt.step()
   with torch.no_grad():
    _,step,fs,C=best;bs=[f@location for f in fs];errors={}
    for name,rho in [('value',None),('response',.5)]:
     G=metric(fs,bs,fs,bs,rho);X=rc@metric(reduced,rb,fs,bs,rho);energy=norms[name]+((C.T@C)*G).sum()-2*(C*X).sum();assert energy>-1e-9*norms[name];errors[name+'_error']=float((energy.clamp_min(0)/norms[name]).sqrt())
   row=dict(ambient_control=ambient_control,fit_dimension=rank,family=family,seed=seed,selected_step=step,**errors);rows.append(row);print(row,flush=True)
 (P/('WIDE_QUARTIC_SUBSPACE_AMBIENT_V1.json' if ambient_control else 'WIDE_QUARTIC_SUBSPACE_V1.json')).write_text(json.dumps(dict(rows=rows,geometry=geometry,prediction=None if ambient_control else sum(r['value_error']<.05 for r in rows)>=8,seconds=time.monotonic()-start,scope=('Ten direct1152-dimensional Adam controls on same densely rotated targets. Different random-start dimensions; not identical initial functions. Geometry fields describe recovered subspace and are not ambient student costs.' if ambient_control else 'Exact weight-derived subspace discovery on five planted densely embedded quartics, then10Adamfits. No native compact-subspace or semantic guarantee.')),indent=2)+'\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--ambient-control',action='store_true');args=parser.parse_args();main(args.ambient_control)
