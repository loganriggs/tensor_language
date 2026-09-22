"""Check normalization VJP, including the radial gauge-null direction."""
import json
from pathlib import Path
import torch
from shared_training_gradient import backward
from shared_quadratic_bank import normalize_bank
from sparse_quartic_bank import support,gram as cg,native_cross as cx
from shared_gaussian_moments import gram as gg,native_cross as gx
from noncentral_gaussian_cp import project_shifted

def main():
 torch.set_num_threads(2);rows=[];dtype=torch.float64
 for seed in range(3):
  torch.manual_seed(12300+seed);params=[torch.randn(8,3,dtype=dtype,requires_grad=True) for _ in range(2)];U,V=normalize_bank(*[p.reshape(4,2,3) for p in params]);pairs=support(4,7,12);C=torch.randn(2,7,dtype=dtype)
  teacher=[torch.randn(*shape,dtype=dtype) for shape in [(2,3),(3,3),(3,3),(3,4),(4,3),(4,3)]];S=torch.eye(3,dtype=dtype);mu=torch.randn(3,dtype=dtype);projection=project_shifted(teacher,mu);lam=17.;normalizer=31.
  G=(gg(U,V,pairs,U@mu,V@mu)+lam*cg(U,V,pairs))/(1+lam);X=(gx(teacher,mu,projection,U,V,pairs,U@mu,V@mu)+lam*cx(teacher,U,V,pairs))/(1+lam)
  loss=((G*(C.T@C)).sum()-2*(C*X).sum())/normalizer;ref=torch.autograd.grad(loss,params)
  value=backward(params,4,2,pairs,C,teacher,teacher,mu,projection,S,mu,lam,normalizer,chunk=2)
  a=torch.cat([p.grad.flatten() for p in params]);b=torch.cat([g.flatten() for g in ref]);err=float((a-b).norm()/b.norm());assert err<1e-10
  radial=sum((p.grad*p.detach()).reshape(4,-1).sum(1) for p in params);scale=sum((p.grad*p.detach()).abs().reshape(4,-1).sum(1) for p in params);raderr=float((radial.abs()/scale.clamp_min(1e-30)).max());assert raderr<1e-10
  replay=abs(value/normalizer-float(loss.detach()))/(1+abs(float(loss.detach())));assert replay<1e-10
  rows.append(dict(seed=seed,normalized_gradient_error=err,radial_null_error=raderr,value_replay=replay))
 Path(__file__).with_name('SHARED_GRADIENT_PULLBACK_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
