"""Independent gradient and moment checks before native hybrid learning."""
import json
import torch
from toy_local_quartic_residual import planted
from hybrid_local_quartic import profile
from local_quartic_residual import objective as old
from quartic_cp_profile import normalize_factors
from mixed_gaussian_cp import gram_dynamic
from noncentral_gaussian_cp import project_shifted
from audit_conditional_residual_accounting import P

def controls():
 torch.set_num_threads(2);rows=[]
 for family in [0,1,3,4,5]:
  t,pf,pc,rf,rc=planted(0 if family==5 else family)
  if family==5:
   rf=[f.clone() for f in rf];rf[0][:]=rf[0][0].clone();rf[1][:]=rf[1][0].clone()
  mu=torch.tensor([.1,-.2,.3,.1],dtype=torch.float64);S=torch.eye(4,dtype=torch.float64)
  torch.manual_seed(45000+family);x=torch.randn(97,4,dtype=torch.float64)+mu
  residual=torch.stack([x@f.T for f in rf]).prod(0)@rc.T
  sensitivity=1/(1+(x@torch.randn(4,2,dtype=torch.float64)).square()).square()
  weights=torch.tensor([.7,1.3],dtype=torch.float64)
  pars=[torch.randn(4,4,dtype=torch.float64,requires_grad=True) for _ in range(4)]
  fs=normalize_factors(pars);b=[f@mu for f in fs];rb=[f@mu for f in rf]
  G=gram_dynamic(fs,b,fs,b);X=rc@gram_dynamic(rf,rb,fs,b);phi=torch.stack([x@f.T for f in fs]).prod(0)
  for eta in [0.,.5,1.]:
   loss,C,info=profile(G,X,phi,residual,sensitivity,2,weights,eta)
   direct,dC,_=profile(G,X,phi,residual,sensitivity,2,weights,eta,detach=False)
   g1=torch.autograd.grad(loss,pars,retain_graph=True);g2=torch.autograd.grad(direct,pars,retain_graph=True)
   gradient=max(float((a-b).norm()/b.norm().clamp_min(1e-20)) for a,b in zip(g1,g2));assert gradient<1e-8
   if eta==1:
    w=sensitivity/sensitivity.mean(0);independent=loss.new_zeros(())
    for out in range(2):
     sl=slice(out*2,(out+1)*2);c=C[out,sl];pred=phi[:,sl]@c
     independent+=weights[out]*((w[:,out]*(pred.square()-2*pred*residual[:,out])).mean()+1e-6*c.square().sum())
    assert torch.allclose(loss,independent,rtol=1e-10,atol=1e-10)
   if eta==0 and family!=5:
    reference,_,_=old(t,t,mu,project_shifted(t,mu),S,mu,pf,pc,fs,2,weights)
    assert torch.allclose(loss,reference,rtol=1e-9,atol=1e-9)
    refgrad=torch.autograd.grad(reference,pars,retain_graph=True)
    assert max(float((a-b).norm()/b.norm().clamp_min(1e-20)) for a,b in zip(g1,refgrad))<1e-8
   rows.append(dict(family=family,eta=eta,envelope_gradient_error=gradient,normal_residual=info['normal_residual']))
 return rows

def main():
 rows=controls()
 (P/'HYBRID_LOCAL_QUARTIC_CONTROLS_V1.json').write_text(json.dumps(dict(rows=rows,pass_all=True,scope='Five distinct planted structures; no optimizer fitting or native result. eta0 prior objective on original four teacher forms, eta1 independent empirical energy, eta.5 envelope gradient.'),indent=2)+'\n');print('15 hybrid moment/gradient cases PASS')
if __name__=='__main__':main()
