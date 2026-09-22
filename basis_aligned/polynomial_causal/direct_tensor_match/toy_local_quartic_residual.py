"""Five planted output-local quartic residuals; matched Adam/Muon restarts."""
import json,time,math
import torch
from local_quartic_residual import objective
from quartic_cp_profile import normalize_factors
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import project_shifted
from audit_conditional_residual_accounting import P

def planted(seed):
 torch.manual_seed(24000+seed);dtype=torch.float64;d=4;n=6
 f=normalize_factors([torch.randn(n,d,dtype=dtype) for _ in range(4)])
 if seed==1:f[1]=f[0];f[3]=f[2]
 if seed==2:f[2][:4]=f[0][:4];f[3][:4]=f[1][:4]
 if seed==3:f[0][1]=f[0][0];f[1][1]=f[1][0];f[2][3]=f[2][2];f[3][3]=f[3][2]
 if seed==4:f=[f[0]]*4
 C=torch.zeros(2,n,dtype=dtype);C[0,:2]=torch.tensor([1.,-.6]);C[1,2:4]=torch.tensor([.8,.5]);C[:,4:]=torch.randn(2,2,dtype=dtype)*.2
 L1=torch.stack([v for i in range(n) for v in [f[0][i],f[2][i]]]);R1=torch.stack([v for i in range(n) for v in [f[1][i],f[3][i]]]);eye=torch.eye(2*n,dtype=dtype)
 teacher=[C,eye[::2],eye[1::2],eye,L1,R1];pf=[a[4:].clone() for a in f];pc=C[:,4:].clone()
 residualC=C[:,:4];rf=[a[:4] for a in f];return teacher,pf,pc,rf,residualC

def main():
 torch.set_num_threads(2);start=time.monotonic();rows=[];names=['generic','paired_squares','squared_quadratics','shared_quadratic_factors','fourth_powers']
 for family in range(5):
  t,pf,pc,rf,rc=planted(family);dtype=torch.float64;S=torch.eye(4,dtype=dtype);mu=torch.tensor([.1,-.2,.3,.1],dtype=dtype);proj=project_shifted(t,mu);norm=((rc.T@rc)*gram_dynamic(rf,[a@mu for a in rf],rf,[a@mu for a in rf])).sum()
  for optim in ['adam','muon']:
   for seed in [0,1]:
    torch.manual_seed(24100+seed);params=[torch.randn(4,4,dtype=dtype,requires_grad=True) for _ in range(4)];opt=(torch.optim.Adam if optim=='adam' else torch.optim.Muon)(params,lr=.03);best=None
    for step in range(101):
     fs=normalize_factors(params);loss,C,info=objective(t,t,mu,proj,S,mu,pf,pc,fs,2,ridge=1e-6);value=float(loss.detach())
     if best is None or value<best[0]:best=(value,step,[f.detach().clone() for f in fs],C.detach().clone())
     if step==100:break
     opt.zero_grad();(loss/norm).backward();opt.step()
    value,step,fs,C=best;G=gram_dynamic(fs,[a@mu for a in fs],fs,[a@mu for a in fs]);cross=rc@gram_dynamic(rf,[a@mu for a in rf],fs,[a@mu for a in fs]);err=(norm+((C.T@C)*G).sum()-2*(C*cross).sum()).clamp_min(0)/norm
    row=dict(family=names[family],optimizer=optim,seed=seed,selected_step=step,relative_error=float(err.sqrt()),objective=value);rows.append(row);print(json.dumps(row),flush=True)
 out=dict(rows=rows,steps=100,lr=.03,restarts=2,seconds=time.monotonic()-start,scope='Five planted native-two-layerquartic targets with exactlytwo residualatoms peroutput; same4inputdimensions/100steps/.03rate/twostarts. Readouts profiled, choosebytrainingobjective. ExactGaussianpopulationerror fromindependentresidualCPmoments, no text. Small matchedpilot not comprehensiveoptimizercomparison.')
 (P/'LOCAL_QUARTIC_RESIDUAL_TOYS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
