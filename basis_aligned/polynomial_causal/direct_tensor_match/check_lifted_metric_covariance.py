"""Five equal-covariance distributions with different quadratic functional metrics."""
from pathlib import Path
import json,itertools,torch
P=Path(__file__).parent;torch.set_num_threads(2);dtype=torch.float64;d=8
ij=torch.triu_indices(d,d,1);m=d+len(ij[0]);theta=torch.zeros(m,dtype=dtype);theta[0]=1;theta[1]=-1
families={'gaussian':(3.,1.),'rademacher':(1.,1.),'coordinate_axes':(float(d),0.),'sparse_independent':(4.,1.),'uniform_sphere':(3*d/(d+2),d/(d+2))}
rows=[]
for index,(name,(kappa,rho)) in enumerate(families.items()):
 M=torch.zeros(m,m,dtype=dtype);M[:d,:d]=(kappa-rho)*torch.eye(d,dtype=dtype)+rho*torch.ones(d,d,dtype=dtype);M[d:,d:]=2*rho*torch.eye(m-d,dtype=dtype)
 g=torch.Generator().manual_seed(9220+index);n=100000
 if name=='gaussian':x=torch.randn(n,d,generator=g,dtype=dtype)
 elif name=='rademacher':x=2*torch.randint(2,(n,d),generator=g).to(dtype)-1
 elif name=='coordinate_axes':
  axes=torch.randint(d,(n,),generator=g);sign=2*torch.randint(2,(n,),generator=g).to(dtype)-1;x=torch.nn.functional.one_hot(axes,d).to(dtype)*sign[:,None]*d**.5
 elif name=='sparse_independent':x=(2*torch.randint(2,(n,d),generator=g).to(dtype)-1)*2*(torch.rand(n,d,generator=g)<.25)
 else:x=torch.randn(n,d,generator=g,dtype=dtype);x=x/x.norm(dim=1)[:,None]*d**.5
 phi=torch.cat([x.square(),2**.5*x[:,ij[0]]*x[:,ij[1]]],1);q=x[:,0].square()-x[:,1].square();replay=float((phi@theta-q).abs().max());assert replay<1e-12
 exact=float(theta@M@theta);sample=float(q.square().mean());se=float(q.square().std()/n**.5);zscore=abs(sample-exact)/se if se else 0.;assert zscore<5
 empirical=phi.T@phi/n;metricerr=float((empirical-M).norm()/M.norm());assert metricerr<.04
 ev=torch.linalg.eigvalsh(M);rank=int((ev>1e-10).sum());regularized=M+.01*torch.eye(m,dtype=dtype);mineig=float(torch.linalg.eigvalsh(regularized).min());assert mineig>.009999
 rows.append(dict(distribution=name,input_covariance='I_8 exactly by construction',population_fourth_moment_diagonal=kappa,population_cross_square_moment=rho,quadratic_feature_dimensions=m,metric_rank=rank,exact_error_for_x1square_minus_x2square=exact,empirical_error=sample,monte_carlo_standard_errors=zscore,lifted_metric_relative_replay=metricerr,feature_replay=replay,identity_floor=.01,regularized_min_eigenvalue=mineig))
assert max(abs(r['exact_error_for_x1square_minus_x2square']-expected) for r,expected in zip(rows,[4.,0.,16.,6.,3.2]))<1e-12
out=dict(records=rows,scope='Five distribution/metric controls, not five new structural recovery experiments. Same raw-input covariance, different fourth-moment operators. Positive coefficient floor removes toy metric nullspaces, not a guarantee of native/OOD accuracy.',native_quadratic_feature_dimension=1152*1153//2,empirical_metric_rank_upper_bound_for_current_training_sites=1536)
(P/'LIFTED_METRIC_COVARIANCE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
