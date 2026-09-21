"""Trace term in Gaussian source error when metric and centering moments differ.
For delta~N(0,Sigma_new), e=delta' DeltaQ delta-tr(Sigma_old DeltaQ),
E[e^2]=2||R_new DeltaQ R_new||F^2+tr((Sigma_new-Sigma_old)DeltaQ)^2.
No fit or native outcome selection; this is a metric diagnostic.
"""
from pathlib import Path
import torch,json,itertools,numpy as np
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
# Independent Gaussian quadrature on small random forms.
x,w=np.polynomial.hermite_e.hermegauss(4);g=torch.tensor(list(itertools.product(x,repeat=3)),dtype=torch.float64);weight=torch.tensor([np.prod(v)/(2*np.pi)**1.5 for v in itertools.product(w,repeat=3)],dtype=torch.float64);checks=[]
for seed in range(5):
 rng=torch.Generator().manual_seed(711+seed);R=torch.randn(3,3,generator=rng,dtype=torch.float64);Snew=R@R.T+.5*torch.eye(3);eig,V=torch.linalg.eigh(Snew);root=(V*eig.sqrt())@V.T;A=torch.randn(3,3,generator=rng,dtype=torch.float64);Sold=A@A.T;Q=torch.randn(3,3,generator=rng,dtype=torch.float64);Q=.5*(Q+Q.T);z=g@root;values=((z@Q)*z).sum(1)-torch.trace(Sold@Q);numeric=(weight*values.square()).sum();analytic=2*(root@Q@root).square().sum()+torch.trace((Snew-Sold)@Q).square();checks.append(float(abs(numeric-analytic)/analytic))
assert max(checks)<1e-12
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);B=torch.eye(len(root))-d['inverse_root']@d['old_covariance']@d['inverse_root'];T=d['teacher']
def pair_matrices(p):
 n=p['shared_reader'].shape[1];cores=[]
 for output in range(2):
  G=torch.zeros(n,n,dtype=torch.float64)
  for pos,(i,j,kind) in enumerate(p['product_indices'].T.tolist()):
   val=p['product_weights'][pos,output]
   if kind==0:G[i,i]+=val
   elif kind==1:G[i,i]+=val;G[j,j]-=val
   else:G[i,j]+=val/2;G[j,i]+=val/2
  cores.append(p['shared_reader']@G@p['shared_reader'].T)
 return cores

def matrices(p):
 L=p.get('left_reader',p.get('shared_reader'));R=p.get('right_reader',L);raw=torch.einsum('ir,ro,jr->oij',L,p['product_weights'],R);return .5*(raw+raw.transpose(-1,-2))
base=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True);candidates={'separate':torch.stack([q for p in base.values() for q in pair_matrices(p)])}
for prefix in ['SHARED','MIXED']:
 result=json.loads((P/f'{prefix}_PRODUCT_NATIVE_FIT_V1.json').read_text());p=torch.load(P/f'{prefix}_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True)[result['winner']];candidates[prefix.lower()]=matrices(p)
p=torch.load(P/'PARTIAL_GRAPH_FROZEN_V1.pt',weights_only=True);candidates['profiled_graph']=torch.cat([matrices(p['shared_mixed']),torch.stack(pair_matrices(p['private_pair']))])
records=[];reference_mean=torch.einsum('ij,oji->o',B,T);reference_energy=2*T.square().sum()+reference_mean.square().sum()
for name,Qs in candidates.items():
 M=torch.stack([root@Q@root for Q in Qs])/d['scales'][:,None,None];delta=M-T;mean=torch.einsum('ij,oji->o',B,delta);variation=2*delta.square().sum((1,2));energy=variation+mean.square()
 records.append(dict(candidate=name,coefficient_relative_error=float(delta.norm()/T.norm()),gaussian_remainder_relative_error=float((energy.sum()/reference_energy).sqrt()),mean_error_energy_fraction=float(mean.square().sum()/energy.sum()),per_pair_mean_error_energy_fraction=[float(mean[2*j:2*j+2].square().sum()/energy[2*j:2*j+2].sum()) for j in range(3)],normalized_source_error_means=mean.tolist()))
out=dict(quadrature_relative_errors=checks,records=records,native_input_mean_square_min=float(d['z'].square().mean(1).min()),native_input_mean_square_max=float(d['z'].square().mean(1).max()),scope='Exact Gaussian surrogate source-remainder metric with old affine/trace centering preserved and expanded covariance. Not the quartic composed component metric, nor a fit, native intervention, or proof that this term causes failures.')
(P/'SOURCE_GAUSSIAN_TRACE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
