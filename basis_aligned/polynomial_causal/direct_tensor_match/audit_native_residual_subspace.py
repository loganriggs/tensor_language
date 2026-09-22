"""Gaussian-probe subspaces for the actual native small-output residual."""
import json,time
import torch
from audit_residual_input_sensitivity import native_value_jac,cp_value_jac
from audit_conditional_residual_accounting import P,SCALE,load
from audit_root_matched_reader import CK

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double();U=state['lm_head.weight'].double();UW=U@writer;readers=U.T@UW/UW.square().sum(0);del U,UW
 w=lambda l,n:state[f'transformer.h.{l}.mlp.{n}.weight'].double()
 T=[readers[:,4:].T@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')]
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);mu=cache['mean'].double();S=cache['projections']['covariance']['whitener'].double();weights=torch.tensor(json.loads((P/'OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json').read_text())['weights'][4:],dtype=torch.float64);weights/=weights.mean()
 panels=[];eulers=[]
 for seed in [42001,42002]:
  z=torch.randn(128,1152,dtype=torch.float64,generator=torch.Generator().manual_seed(seed));x=mu+z@S.T;ys=[];js=[]
  for xx in x.split(32):
   yy,jj=native_value_jac(T,xx);ys.append(yy);js.append(jj)
  y=torch.cat(ys);J=torch.cat(js);euler=float(((J*x[:,None,:]).sum(-1)-4*y).norm()/(4*y).norm());assert euler<1e-10;eulers.append(euler);panels.append((x,y,J))
 x,y,J=panels[0];direction=torch.randn(2,1152,dtype=torch.float64,generator=torch.Generator().manual_seed(42003));direction/=direction.norm(dim=1,keepdim=True);eps=1e-4;yp,_=native_value_jac(T,x[:2]+eps*direction);ym,_=native_value_jac(T,x[:2]-eps*direction);ref=(J[:2]*direction[:,None,:]).sum(-1);finite=float(((yp-ym)/(2*eps)-ref).norm()/ref.norm());assert finite<1e-6
 rows=[]
 for seed in [1001,1002]:
  p,sha=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');residuals=[]
  for x,y,J in panels:
   pred,K=cp_value_jac(p['coefficients'][4:]/SCALE,p['factors'],x);residuals.append((J-K)@S)
  A=(residuals[0]*weights.sqrt()[None,:,None]).reshape(-1,1152);H=A.T@A/128;ev,V=torch.linalg.eigh(H);order=ev.argsort(descending=True);ev=ev[order];V=V[:,order];captures=[]
  for rank in [4,16,32,64,128,256,512]:
   frame=V[:,:rank];rr=[]
   for residual in residuals:
    total=residual.square().sum((0,2));captured=(residual@frame).square().sum((0,2));rr.append(dict(weighted_capture=float((weights*captured).sum()/(weights*total).sum()),per_output_capture=(captured/total).tolist()))
   captures.append(dict(rank=rank,training=rr[0],checking=rr[1]))
  row=dict(seed=seed,parent_sha256=sha,eigenvalues=ev.tolist(),captures=captures,training_samples=128,checking_samples=128);rows.append(row);print(seed,[(r['rank'],r['training']['weighted_capture'],r['checking']['weighted_capture']) for r in captures],flush=True)
 pred=all(next(c for c in r['captures'] if c['rank']==64)['checking']['weighted_capture']>=.9 for r in rows)
 (P/'NATIVE_RESIDUAL_SUBSPACE_V1.json').write_text(json.dumps(dict(rows=rows,prediction=pred,native_euler_errors=eulers,native_finite_difference=finite,seconds=time.monotonic()-start,scope='Monte Carlo Gaussian derivative geometry for native-minus-parent outputs4–15, not text-tangent geometry or function-error guarantee. Independent synthetic check probes; no candidate export or queue changes.'),indent=2)+'\n')
if __name__=='__main__':main()
