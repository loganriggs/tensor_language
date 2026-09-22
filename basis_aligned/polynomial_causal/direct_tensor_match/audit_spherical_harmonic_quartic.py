"""Native quartic Fischer components from cached exact Gaussian traces.
No tensor expansion; dropping harmonic degree4 is an approximation."""
import json,time
from pathlib import Path
import torch
from native_quartic_gaussian_projection import project
from audit_residual_input_sensitivity import native_value_jac
P=Path(__file__).resolve().parent

def components(y,x,mean,Q):
 d=x.shape[1];r2=x.square().sum(1);q=torch.einsum('ni,vij,nj->nv',x,Q,x)
 h0=mean/(d*(d+2));h2=(q-2*r2[:,None]*mean/d)/(d+4)
 p0=r2[:,None].square()*h0;p2=r2[:,None]*h2
 return p0,p2,y-p0-p2

def controls():
 torch.set_num_threads(2);dtype=torch.float64;rows=[]
 for seed in range(3):
  torch.manual_seed(18000+seed);teacher=[torch.randn(*shape,dtype=dtype) for shape in [(2,4),(4,3),(4,3),(3,5),(5,3),(5,3)]]
  mean,Q=project(teacher);x=torch.randn(7,3,dtype=dtype,requires_grad=True);y,_=native_value_jac(teacher,x);p0,p2,h4=components(y,x,mean,Q);h2=p2/x.square().sum(1)[:,None]
  def lap(z):
   values=[]
   for out in range(2):
    g=torch.autograd.grad(z[:,out].sum(),x,create_graph=True,retain_graph=True)[0]
    values.append(sum(torch.autograd.grad(g[:,j].sum(),x,retain_graph=True)[0][:,j] for j in range(3)))
   return torch.stack(values,1)
  errors=[float(lap(t).abs().max()) for t in [h2,h4]];assert max(errors)<1e-9,errors
  rows.append(dict(seed=seed,h2_laplacian_max=errors[0],h4_laplacian_max=errors[1],trace_replay=float((Q.diagonal(dim1=-2,dim2=-1).sum(-1)-2*mean).abs().max())))
 return rows

def main():
 torch.set_num_threads(2);checked=controls();torch.set_grad_enabled(False);start=time.monotonic()
 a=torch.load(P/'GAUSSIAN_CP_TEACHER_PROJECTION_V1.pt',weights_only=True);mean=a['mean'].double();Q=a['quadratic'].double();scale=a['scale']
 writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'];assert torch.allclose(a['writer'].double(),writer.double())
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);ix=torch.arange(256)*64+31;x=data['rows'][ix].double();y=data['target'][ix].double()/scale
 panels=[('opened256states',x,y)]
 old=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();label=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/scale
 matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(matched['pairs_flat']).T
 # Exact source traces alone determine degree0/2 component; compare opened values.
 rows=[]
 for name,x,y in panels:
  p0,p2,p4=components(y,x,mean,Q);pred=p0+p2
  rows.append(dict(panel=name,value_error=float((pred-y).norm()/y.norm()),constant_only_error=float((p0-y).norm()/y.norm()),feature_errors=((pred-y).square().sum(0)/y.square().sum(0)).sqrt().tolist(),component_norms_over_target=[float(v.norm()/y.norm()) for v in [p0,p2,p4]],radius_squared_relative_error=float((x.square().sum(1)/1152-1).abs().max())))
 # Compute only pair endpoints, preserve directed pair accounting.
 ids=torch.unique(torch.cat([rec,don]));pos={int(j):i for i,j in enumerate(ids)};ri=torch.tensor([pos[int(j)] for j in rec]);di=torch.tensor([pos[int(j)] for j in don]);xx=old[ids];yy=label[ids];p0,p2,_=components(yy,xx,mean,Q);pred=p0+p2;ref=yy[di,1]-yy[ri,1]
 response=float(((pred[di,1]-pred[ri,1])-ref).norm()/ref.norm())
 result=dict(controls=checked,rows=rows,root1_response_error=response,seconds=time.monotonic()-start,scope='Fischer harmonic decomposition of native selectedpurequartic16outputmap. Degree0/2onfixedradius fromexactweighttraces; degree4remainderimplicit. No claimuniformspherelaw matchestext; no newfit, no smallerexport, no selectiveadoption. Gaussiantracesreused; normalizationepsilon acknowledged.')
 (P/'SPHERICAL_HARMONIC_QUARTIC_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
