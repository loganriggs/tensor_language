import itertools,json,math,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def radial_tensor(Q):return (torch.einsum('ij,kl->ijkl',Q,Q)+torch.einsum('ik,jl->ijkl',Q,Q)+torch.einsum('il,jk->ijkl',Q,Q))/3

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1812);checks=[]
 for d,r in [(6,2),(8,3)]:
  U=torch.linalg.qr(torch.randn(d,r))[0];projection=U@U.T;H=radial_tensor(torch.eye(d));S=radial_tensor(projection);got=float((H-S).norm()/H.norm());expected=math.sqrt(1-r*(r+2)/(d*(d+2)));assert abs(got-expected)<1e-12;checks.append(dict(d=d,subspace_dimension=r,error=got,formula=expected))
 out=P/'INPUT_MODE_CALIBRATION_V1.json';assert not out.exists();records=[];start=time.perf_counter();d=128;den=math.sqrt(d*(d+2)/3)
 for n,seed in itertools.product([256,512,1024,4096],range(8)):
  gen=torch.Generator();gen.manual_seed(1812+seed);b,c,e=[torch.randint(2,(n,d),generator=gen).double()*2-1 for _ in range(3)];t=(b*(c*e).sum(1,keepdim=True)+c*(b*e).sum(1,keepdim=True)+e*(b*c).sum(1,keepdim=True))/(3*den);G=t.T@t/n;ev=torch.linalg.eigvalsh(G).clamp_min(0)
  for r in [4,16,32]:records.append(dict(probes=n,seed=seed,rank=r,normalized_trace=float(ev.sum()),finite_probe_tail_error=float((ev[:-r].sum()/ev.sum()).sqrt()),true_input_mode_tail_error=math.sqrt(1-r/d)))
 bounds=[]
 for d in [16,32,128,1152]:
  s=min(d,32);bounds.append(dict(d=d,cp_atoms=8,maximum_input_span=s,radial_projection_lower_bound=math.sqrt(1-s*(s+2)/(d*(d+2))),previous_pair_unfolding_bound=math.sqrt(max(0,4*(d*(d+1)/2-48)/(3*d*(d+2))))))
 result=dict(projection_checks=checks,bounds=bounds,records=records,seconds=time.perf_counter()-start,scope='Radial quartic: exact best coefficient projection onto any r-dimensional inputspace. CP8 inputspan≤32 gives necessary bound, not an attainable CP optimum. Rademacher Gram calibration illustrates finite-sample spectral bias, not a native error certificate.')
 out.write_text(json.dumps(result,indent=2)+'\n');print('bounds',bounds);print('seconds',result['seconds'])
if __name__=='__main__':main()
