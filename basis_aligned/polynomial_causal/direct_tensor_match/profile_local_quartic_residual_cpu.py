"""Actual native96atom/12output residual gradient and finite differences on CPU."""
import json,time,resource
import torch
from local_quartic_residual import objective
from quartic_cp_profile import normalize_factors
from noncentral_gaussian_cp import project_shifted
from audit_root_matched_reader import CK
from audit_conditional_residual_accounting import P,SCALE,load

def main():
 torch.set_num_threads(2);torch.manual_seed(25000);start=time.monotonic();dtype=torch.float64
 with torch.no_grad():
  cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double();loc=torch.linalg.solve(S,mu);state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');vocab=state['lm_head.weight'].double();uw=vocab@cache['writer'].double();readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
  def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].double()
  t=[readers[:,4:].T@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')];tr=t[:4]+[t[4]@S,t[5]@S];pr=project_shifted(tr,loc,tuple(a[4:].double() for a in cache['projections']['covariance']['zero_projection']));parent,h=load('MIXED_CP_FEATURES_SEED1001_V1.pt');pf=parent['factors'];pc=parent['coefficients'][4:]/SCALE
  weights=torch.tensor(json.loads((P/'OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json').read_text())['weights'][4:],dtype=dtype);weights/=weights.mean()
 params=[torch.randn(96,1152,dtype=dtype,requires_grad=True) for _ in range(4)]
 def fn(params):return objective(t,tr,loc,pr,S,mu,pf,pc,normalize_factors(params),8,weights)
 begin=time.monotonic();loss,C,info=fn(params);loss.backward();elapsed=time.monotonic()-begin;assert all(torch.isfinite(a.grad).all() for a in params)
 gn=sum(a.grad.square().sum() for a in params).sqrt();pn=sum(a.square().sum() for a in params).sqrt();directions=[a.grad/gn*pn for a in params];analytic=float(sum((a.grad*d).sum() for a,d in zip(params,directions)));print('Native forward/backward',elapsed,'normal',info,flush=True)
 checks=[]
 with torch.no_grad():
  for eps in [1e-4,3e-5]:
   plus=fn([a+eps*d for a,d in zip(params,directions)])[0];minus=fn([a-eps*d for a,d in zip(params,directions)])[0];fd=float((plus-minus)/(2*eps));error=abs(fd-analytic)/max(abs(analytic),1e-20);checks.append(dict(epsilon=eps,analytic=analytic,finite_difference=fd,relative_error=error));print(checks[-1],flush=True)
 result=dict(normal_residual=info['normal_residual'],finite_differences=checks,forward_backward_seconds=elapsed,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-start,pass_integrity=info['normal_residual']<1e-8 and all(c['relative_error']<1e-3 for c in checks),source_sha256=h,shape=dict(outputs=12,per_output=8,new_atoms=96,input_width=1152,parent_atoms=512),scope='Actual native CPUfloat64 derivative profile. No optimizerstep/fitting or GPUtimingclaim; noexport oradoption.')
 (P/'LOCAL_QUARTIC_RESIDUAL_NATIVE_PROFILE_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
