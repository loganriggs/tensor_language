import itertools,json
from pathlib import Path
import numpy as np
import torch
from implicit_quartic import entries
from quartic_gaussian import mean
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1631);d=5
params=[torch.randn(*s) for s in [(3,4),(4,5),(4,5),(5,7),(7,d),(7,d)]];C,L2,R2,D1,L1,R1=params;ids=torch.tensor(list(itertools.product(range(d),repeat=4)));H=entries(*params,ids).T.reshape(3,d,d,d,d);trace=torch.einsum('viikl->vkl',H);doubletrace=trace.diagonal(dim1=-2,dim2=-1).sum(-1)
nodes,weights=np.polynomial.hermite.hermgauss(5);x=torch.tensor(list(itertools.product(nodes,repeat=d)))*2**.5;w=torch.tensor([np.prod(z) for z in itertools.product(weights,repeat=d)])/np.pi**(d/2);h=((x@L1.T)*(x@R1.T))@D1.T;f=((h@L2.T)*(h@R2.T))@C.T
q=torch.einsum('ni,vij,nj->nv',x,trace,x);f0=3*doubletrace.expand_as(f);f2=6*(q-doubletrace);f4=f-6*q+3*doubletrace;components=[f0,f2,f4];theory=[9*doubletrace.square().sum(),72*trace.square().sum(),24*H.square().sum()];actual=[(v.square().sum(-1)*w).sum() for v in components];total=(f.square().sum(-1)*w).sum();energy_error=max(float(abs(a-b)/total) for a,b in zip(theory,actual));cross=max(float(abs(((components[i]*components[j]).sum(-1)*w).sum())/total) for i,j in itertools.combinations(range(3),2));mu=mean(*params);mean_error=float((mu-(f*w[:,None]).sum(0)).norm()/mu.norm());assert max(energy_error,cross,mean_error)<1e-12
radial_coeff=mu/(d*(d+2));radial=x.square().sum(-1).square()[:,None]*radial_coeff;radial_energy=radial_coeff.square().sum()*d*(d+2)*(d+4)*(d+6);radial_orth=float(abs((((f-radial)*radial).sum(-1)*w).sum())/total);assert radial_orth<1e-12
out=dict(gaussian_energy=float(total),chaos_energies={str(k):float(e) for k,e in zip([0,2,4],theory)},chaos_fractions={str(k):float(e/total) for k,e in zip([0,2,4],theory)},energy_identity_relative_error=energy_error,orthogonality_relative_error=cross,implicit_mean_relative_error=mean_error,radial_projection_orthogonality=radial_orth,radial_relative_error=float(((total-radial_energy)/total).sqrt()),scope='Independent exact-degree quadrature for E||H x^4||²=24||H||²+72||Tr H||²+9||Tr²H||²; exact factor-contracted mean; radial baseline. Standard isotropic Gaussian only.')
(P/'QUARTIC_GAUSSIAN_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
