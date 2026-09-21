"""Export learned metric-coordinate readers to the native shared/private graph."""
import torch
from local_shared_reader_graph import decode,expand

def export_overlap(params,weights,template,inverse,data):
 out=dict(input_basis=(inverse@params[0]).detach().cpu().clone(),pairs={})
 for j in range(3):
  old=template['pairs'][str(j)];p=dict(old)
  p['shared_map']=params[1+2*j].detach().cpu().clone();p['private_reader']=(inverse@params[2+2*j]).detach().cpu().clone();p['product_weights']=weights[j].detach().cpu().clone();out['pairs'][str(j)]=p
 dense=expand(out)
 for j in range(3):
  hat=decode(dense[str(j)]);p=out['pairs'][str(j)]
  for key,true,q in zip(('a','b'),data['pairs'][j]['Qs'],hat):
   delta=true-q;p[key+'_linear']=2*delta@data['mu'];p[key+'_bias']=torch.trace(data['old_covariance']@delta)-data['mu']@delta@data['mu']
 return out

def assess_overlap(program,data):
 from local_shared_reader_graph import price,source_reads,component_scalars
 from fit_local_shared_reader_graph import as_global
 from global_mixed_source_graph import score
 bundle=expand(program);Q=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]);hat=torch.cat([decode(bundle[str(j)]) for j in range(3)]);S=torch.linalg.inv(data['inverse_root']);E=hat-Q
 errors=[]
 for true,error in [(Q,E),(S@Q@S,S@E@S)]:
  energy=true.square().sum((-1,-2)).reshape(3,2).sum(1);errors.append(float((error.square().sum((-1,-2)).reshape(3,2).sum(1)/energy).mean().sqrt()))
 ids=data['indices'];z=data['z'][ids];h=data['h'][ids];linear=torch.stack([bundle[str(j)][k+'_linear'] for j in range(3) for k in ('a','b')],1);bias=torch.stack([bundle[str(j)][k+'_bias'] for j in range(3) for k in ('a','b')]);dense=torch.einsum('ni,oij,nj->no',z,hat,z)+z@linear+bias;actual=source_reads(z,program);replay=float((actual-dense).norm()/dense.norm());assert replay<1e-8
 scores=score(as_global(bundle),data);truth=torch.stack([pair['truth'][ids] for pair in data['pairs']],1);values=((component_scalars(z,h,program)-truth).norm(dim=0)/(truth-truth.mean(0)).norm(dim=0)).tolist();assert max(abs(a-b) for a,b in zip(values,scores['per_mode_errors']))<1e-8
 return dict(native_error=errors[0],covariance_error=errors[1],execution_replay=replay,**scores,**price(program))
