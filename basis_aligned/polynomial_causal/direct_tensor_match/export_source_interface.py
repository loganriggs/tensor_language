"""Compile centered source forms into compact explicit native-z/h programs."""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
programs=torch.load(p/'MIDPOINT_SOURCE_COVARIANCE_PROGRAMS_V1.pt',weights_only=True);data=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);z=data['z'].flatten(0,1).double();scale=data['recipient_scale'].flatten().double();hread=data['h_reader'].flatten().double();exports={};records=[]
for name in ['isotropic_0','isotropic_16','covariance_16','covariance_64']:
 e=programs[name];mu=e['mu'];out=dict(h_reader=e['a'].clone(),residual_writer=torch.linalg.solve(e['R_U'],e['writer']),alpha=e['alpha'].clone(),beta=e['beta'].clone());old=[];new=[]
 for k in ['a','b']:
  P=e[k+'_reader'];lam=e[k+'_eigenvalues'];pm=mu@P;l=e[k+'_linear'];c=e[k+'_constant'];center=e[k+'_quadratic_mean'];linear=l-2*P@(lam*pm);bias=c-mu@l+(lam*(pm.square()-center)).sum()
  out[k+'_reader']=P.clone();out[k+'_eigenvalues']=lam.clone();out[k+'_linear']=linear;out[k+'_bias']=bias
  old.append(c+(z-mu)@l+((((z-mu)@P).square()-center)*lam).sum(1));new.append(bias+z@linear+((z@P).square()*lam).sum(1))
 phi=lambda q:((hread-.5*q[0])/scale-e['alpha'])*(q[1]/scale-e['beta'])
 err=float((phi(new)-phi(old)).norm()/phi(old).norm());source_err=max(float((a-b).norm()/a.norm()) for a,b in zip(old,new));writererr=float((e['R_U']@out['residual_writer']-e['writer']).norm()/e['writer'].norm());assert max(err,source_err,writererr)<1e-10
 records.append(dict(candidate=name,scalar_replay=err,source_replay=source_err,writer_replay=writererr,stored_scalars=sum(v.numel() for v in out.values()),source_squares=sum(out[k+'_eigenvalues'].numel() for k in ['a','b']),final_variable_products=1));exports[name]=out
file=p/'MIDPOINT_SOURCE_INTERFACE_V1.pt';torch.save(exports,file)
result=dict(records=records,artifact_bytes=file.stat().st_size,scope='Exact representation-only compilation of frozen approximations. Inputs native normalized MLP16 z and lastMLP h; s(h)=sqrt(mean(h^2)+float32eps) explicit, output residual write. No full QR/calibration arrays required. Both inputs still native, h includes upstream contributions. No wholemodel savings or new behavioral claim.')
(p/'MIDPOINT_SOURCE_INTERFACE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
